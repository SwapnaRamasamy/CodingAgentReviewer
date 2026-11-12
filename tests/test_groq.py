"""Groq configuration and the real agent hierarchy, exercised without network access."""
import json
from functools import partial

import httpx
import pytest
from langchain_openai import ChatOpenAI

from reviewer.activity import Activity
from reviewer.agents import Reviewer, demo_result
from reviewer.model_config import (
    GROQ_BASE_URL,
    GROQ_MODEL,
    GROQ_MODELS,
    configured_models,
)
from reviewer.sources import demo_snapshot


@pytest.fixture(autouse=True)
def virtual_groq_capacity(monkeypatch):
    from reviewer.provider_limits import GroqRateLimiter

    current, gates = [0.0], {}
    def sleep(seconds):
        current[0] += seconds
    def shared(api_key, model):
        key = (api_key, model)
        if key not in gates:
            gates[key] = GroqRateLimiter(clock=lambda: current[0], sleep=sleep)
        return gates[key]
    monkeypatch.setattr("reviewer.agents.shared_groq_limiter", shared)


def completion(message, finish_reason="stop"):
    return httpx.Response(200, json={
        "id": "offline-groq", "object": "chat.completion", "created": 0, "model": GROQ_MODEL,
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    })


def test_groq_defaults_use_the_requested_model_for_every_role():
    assert configured_models(GROQ_BASE_URL, environ={}) == GROQ_MODELS
    assert set(GROQ_MODELS.values()) == {GROQ_MODEL}
    assert configured_models(GROQ_BASE_URL, environ={"REVIEW_TESTS_MODEL": "other-groq-model"}) == {
        **GROQ_MODELS, "tests": "other-groq-model",
    }


def test_groq_access_uses_catalog_for_slash_model_and_leaves_transport_open(monkeypatch):
    paths = []

    def respond(request):
        paths.append(request.url.path)
        if request.url.path != "/openai/v1/models":
            return httpx.Response(404, json={
                "error": {"message": "Synthetic per-model lookup failure", "code": "model_not_found"},
            })
        return httpx.Response(200, json={
            "object": "list", "data": [
                {"id": GROQ_MODEL, "object": "model", "created": 0, "owned_by": "openai", "active": True},
            ],
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr("reviewer.agents.ChatOpenAI", partial(ChatOpenAI, http_client=client))
        reviewer = Reviewer(Activity(), model=GROQ_MODEL, role_models=GROQ_MODELS,
                            base_url=GROQ_BASE_URL, api_key="unused-groq-key")
        reviewer.check_model_access()
        reviewer.check_model_access()
        assert not client.is_closed
    assert paths == ["/openai/v1/models"] * 2


def test_groq_real_hierarchy_preserves_tools_and_structured_reviews(monkeypatch):
    from reviewer.groq_review import GroqFinding

    roles = ("quality", "security", "tests")
    counts = dict.fromkeys(("orchestrator", *roles), 0)
    requests = []

    def respond(request):
        assert str(request.url) == GROQ_BASE_URL + "/chat/completions"
        payload = json.loads(request.content)
        assert payload["model"] == GROQ_MODEL
        assert payload["reasoning_effort"] == "low"
        if any("name" in message for message in payload["messages"]):
            return httpx.Response(400, json={"error": {"code": "invalid_request_error"}})
        assert "reasoning_format" not in payload
        assert "max_tokens" not in payload
        system = payload["messages"][0]["content"]
        markers = {"quality": "quality code-review subagent", "security": "security code-review subagent",
                   "tests": "test-review subagent"}
        role = next((name for name, marker in markers.items() if marker in system), "orchestrator")
        index = counts[role]
        counts[role] += 1
        requests.append((role, payload))
        tool_names = {item["function"]["name"] for item in payload.get("tools", [])}
        if "response_format" in payload:
            assert role != "orchestrator"
            assert not any(key in payload for key in ("tools", "tool_choice", "parallel_tool_calls"))
            assert payload["response_format"]["type"] == "json_schema"
            assert payload["response_format"]["json_schema"]["strict"] is True
            assert payload["max_completion_tokens"] == 3072
            finding = demo_result(role).findings[0].model_dump(mode="json")
            draft = {key: finding[key] for key in GroqFinding.model_fields}
            return completion({"role": "assistant", "content": json.dumps({
                "findings": [draft], "limitations": [],
            })}, "stop")
        assert payload["parallel_tool_calls"] is False
        assert payload["max_completion_tokens"] == 1536
        assert payload.get("tool_choice", "auto") != "required"
        if role == "orchestrator":
            assert tool_names == {"task", "write_todos"}
            if index < len(roles):
                name, args = "task", {"subagent_type": roles[index], "description": "Review " + roles[index]}
            else:
                return completion({"role": "assistant", "content": "All outcomes collected."}, "stop")
        elif index == 0:
            assert tool_names == {"read_file", "search_code"}
            name, args = "read_file", {"path": "app.py", "start_line": 1, "end_line": 12}
        else:
            return completion({"role": "assistant", "content": "Evidence inspected; finalize review."}, "stop")
        return completion({
            "role": "assistant", "content": None,
            "tool_calls": [{"id": f"{role}-{index}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}],
        }, "tool_calls")

    monkeypatch.setattr("reviewer.agents.scan", lambda *args: ([], []))
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr("reviewer.agents.ChatOpenAI", partial(ChatOpenAI, http_client=client))
        reviewer = Reviewer(Activity(), model=GROQ_MODEL, role_models=GROQ_MODELS,
                            base_url=GROQ_BASE_URL, api_key="unused-groq-key")
        result = reviewer.orchestrate(demo_snapshot())
    assert not result["orchestration_error"], result
    assert all(not item.error for item in result["results"]), result
    assert [item.agent for item in result["results"]] == list(roles)
    assert all(len(item.findings) == 1 for item in result["results"])
    assert counts == {"orchestrator": 4, "quality": 3, "security": 3, "tests": 3}
    assert len(requests) == 13


@pytest.mark.parametrize("catalog_models", [[], [{"id": "gpt-oss-120b"}], [{"id": GROQ_MODEL, "active": False}]])
def test_groq_catalog_rejects_absent_wrongly_prefixed_or_inactive_models(monkeypatch, catalog_models):
    from reviewer.diagnostics import ModelCatalogMismatch, safe_error

    def respond(request):
        assert request.url.path == "/openai/v1/models"
        data = [{"object": "model", "created": 0, "owned_by": "openai", **model} for model in catalog_models]
        return httpx.Response(200, json={"object": "list", "data": data})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr("reviewer.agents.ChatOpenAI", partial(ChatOpenAI, http_client=client))
        reviewer = Reviewer(Activity(), model=GROQ_MODEL, base_url=GROQ_BASE_URL,
                            api_key="unused-groq-key", include_tests=False)
        with pytest.raises(ModelCatalogMismatch) as caught:
            reviewer.check_model_access()
        assert caught.value.roles == ("orchestrator", "quality", "security")
        assert "tests" not in safe_error(caught.value)
        assert "Groq" in safe_error(caught.value)
        assert not client.is_closed


@pytest.mark.parametrize("include_tests", [False, True])
def test_groq_catalog_checks_only_selected_models(monkeypatch, include_tests):
    from reviewer.diagnostics import ModelCatalogMismatch

    def respond(request):
        assert request.url.path == "/openai/v1/models"
        return httpx.Response(200, json={
            "object": "list", "data": [{"id": GROQ_MODEL, "object": "model", "created": 0, "owned_by": "openai"}],
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr("reviewer.agents.ChatOpenAI", partial(ChatOpenAI, http_client=client))
        reviewer = Reviewer(
            Activity(), model=GROQ_MODEL, base_url=GROQ_BASE_URL, api_key="unused-groq-key",
            role_models={"tests": "unavailable-tests-model"}, include_tests=include_tests,
        )
        if include_tests:
            with pytest.raises(ModelCatalogMismatch) as caught:
                reviewer.check_model_access()
            assert caught.value.roles == ("tests",)
        else:
            reviewer.check_model_access()
        assert not client.is_closed


def test_groq_catalog_preserves_authentication_failures(monkeypatch):
    import openai

    from reviewer.diagnostics import safe_error

    def respond(request):
        assert request.url.path == "/openai/v1/models"
        return httpx.Response(401, json={
            "error": {"message": "PRIVATE_PROVIDER_PAYLOAD", "code": "invalid_api_key"},
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr("reviewer.agents.ChatOpenAI", partial(ChatOpenAI, http_client=client))
        reviewer = Reviewer(Activity(), model=GROQ_MODEL, base_url=GROQ_BASE_URL, api_key="unused-groq-key")
        with pytest.raises(openai.AuthenticationError) as caught:
            reviewer.check_model_access()
        assert "authentication failed" in safe_error(caught.value)
        assert "PRIVATE_" not in safe_error(caught.value)
        assert not client.is_closed


@pytest.mark.parametrize("base_url,strip_names", [
    (GROQ_BASE_URL, True),
    ("https://api.openai.com/v1", False),
])
def test_provider_message_names_preserve_history_and_tool_links(monkeypatch, base_url, strip_names):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    messages = [
        HumanMessage(content="Review this snapshot.", id="seed-human", additional_kwargs={"name": "legacy_caller"}),
        AIMessage(content="", name="quality_reviewer", id="seed-assistant",
                  additional_kwargs={"name": "legacy_reviewer"},
                  tool_calls=[{"name": "read_file", "args": {"path": "app.py"}, "id": "prior-read"}]),
        ToolMessage(content="Snapshot evidence.", name="read_file", id="seed-tool", tool_call_id="prior-read"),
    ]
    original = [message.model_dump(mode="json") for message in messages]
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        wire_messages = payload["messages"]
        if strip_names:
            assert all("name" not in message for message in wire_messages)
        else:
            assert wire_messages[1]["name"] == "legacy_caller"
            assert wire_messages[2]["name"] == "quality_reviewer"
        assert wire_messages[2]["tool_calls"][0]["function"]["name"] == "read_file"
        assert wire_messages[2]["tool_calls"][0]["id"] == "prior-read"
        assert wire_messages[3]["tool_call_id"] == "prior-read"
        assert payload["tool_choice"] == "required"
        assert {item["function"]["name"] for item in payload["tools"]} == {
            "read_file", "search_code", "ReviewBatch",
        }
        return httpx.Response(200, json={
            "id": "offline-names", "object": "chat.completion", "created": 0,
            "model": GROQ_MODEL,
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{
                    "id": "final-review", "type": "function", "function": {
                        "name": "ReviewBatch",
                        "arguments": json.dumps({"findings": [], "limitations": ["Partial snapshot."]}),
                    },
                }],
            }}],
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr("reviewer.agents.ChatOpenAI", partial(ChatOpenAI, http_client=client))
        reviewer = Reviewer(Activity(), model="other-tool-model", base_url=base_url, api_key="unused")
        state = reviewer.specialist("quality", demo_snapshot()).invoke({"messages": messages})
    assert len(requests) == 1
    assert state["structured_response"].limitations == ["Partial snapshot."]
    assert [message.model_dump(mode="json") for message in messages] == original


@pytest.mark.parametrize("role", ["quality", "security", "tests"])
def test_rejected_gathering_tool_can_finalize_but_stays_incomplete(monkeypatch, role):
    from reviewer.groq_review import GroqFinding

    requests = []
    finding = demo_result(role).findings[0].model_dump(mode="json")
    draft = {key: finding[key] for key in GroqFinding.model_fields}
    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if "response_format" not in payload:
            return httpx.Response(400, json={"error": {
                "code": "tool_use_failed", "failed_generation": "PRIVATE_MODEL_OUTPUT",
            }})
        assert not any(key in payload for key in ("tools", "tool_choice", "parallel_tool_calls"))
        return completion({"role": "assistant", "content": json.dumps({
            "findings": [draft], "limitations": [],
        })})

    monkeypatch.setattr("reviewer.agents.scan", lambda *args: ([], []))
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr("reviewer.agents.ChatOpenAI", partial(ChatOpenAI, http_client=client))
        reviewer = Reviewer(Activity(), model=GROQ_MODEL, base_url=GROQ_BASE_URL, api_key="unused")
        result = reviewer.review(role, demo_snapshot())
    assert len(requests) == 2
    assert "incomplete" in result.error
    assert len(result.findings) == 1
    assert any("gathering stopped early" in text for text in result.limitations)
    assert "PRIVATE_" not in result.model_dump_json() + str(reviewer.activity.snapshot())


@pytest.mark.parametrize("delay,calls", [("2", 3), ("120", 1), (None, 1)])
def test_groq_rate_retry_is_bounded_and_not_a_structured_output_retry(monkeypatch, delay, calls):
    requests = []
    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return httpx.Response(429, headers={} if delay is None else {"retry-after": delay},
                                  json={"error": {"code": "rate_limit_exceeded", "message": "PRIVATE_BODY"}})
        if "response_format" not in payload:
            return completion({"role": "assistant", "content": "No additional lookup needed."})
        return completion({"role": "assistant", "content": '{"findings": [], "limitations": []}'})

    monkeypatch.setattr("reviewer.agents.scan", lambda *args: ([], []))
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr("reviewer.agents.ChatOpenAI", partial(ChatOpenAI, http_client=client))
        reviewer = Reviewer(Activity(), model=GROQ_MODEL, base_url=GROQ_BASE_URL, api_key="unused")
        assert reviewer.llm().max_retries == 0
        result = reviewer.review("quality", demo_snapshot())
    assert len(requests) == calls
    assert bool(result.error) is (calls == 1)
    assert "PRIVATE_" not in result.model_dump_json() + str(reviewer.activity.snapshot())


def test_groq_strict_schema_and_source_validation():
    from pydantic import ValidationError

    from reviewer.diagnostics import InvalidReviewEvidence
    from reviewer.groq_review import GroqFinding, GroqReviewBatch

    schema = GroqReviewBatch.model_json_schema()
    def check(node):
        if isinstance(node, dict):
            if "properties" in node:
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
                assert "default" not in node
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)
    check(schema)
    original = demo_result("quality").findings[0].model_dump(mode="json")
    draft = {key: original[key] for key in GroqFinding.model_fields}
    batch = GroqReviewBatch(findings=[draft], limitations=[])
    batch.findings[0].evidence = "fabricated source"
    with pytest.raises(InvalidReviewEvidence):
        batch.validated_review(demo_snapshot())
    with pytest.raises(ValidationError):
        GroqReviewBatch(findings=[], limitations=[], extra="unexpected")


def test_groq_finalization_failure_preserves_scanner_findings(monkeypatch):
    calls = []
    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if "response_format" not in payload:
            return completion({"role": "assistant", "content": "Evidence inspected."})
        return completion({"role": "assistant", "content": '{"findings": [], "limitations": [], "extra": true}'})
    candidate = demo_result("quality").findings[0]
    monkeypatch.setattr("reviewer.agents.scan", lambda *args: ([candidate], []))
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr("reviewer.agents.ChatOpenAI", partial(ChatOpenAI, http_client=client))
        result = Reviewer(Activity(), model=GROQ_MODEL, base_url=GROQ_BASE_URL, api_key="unused").review(
            "quality", demo_snapshot(),
        )
    assert len(calls) == 2
    assert result.error
    assert result.findings == [candidate]


@pytest.mark.parametrize("scenario", ["truncated_inspection", "shortened_evidence"])
def test_reduced_evidence_cannot_appear_complete(monkeypatch, scenario):
    requests = []
    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if "response_format" not in payload:
            return completion(
                {"role": "assistant", "content": "Inspection observations need final validation."},
                "length" if scenario == "truncated_inspection" else "stop",
            )
        assert not any(key in payload for key in ("tools", "tool_choice", "parallel_tool_calls"))
        return completion({"role": "assistant", "content": '{"findings": [], "limitations": []}'})

    if scenario == "shortened_evidence":
        def estimate(messages, **kwargs):
            data = json.loads(messages[-1].content)
            return 8001 if data["evidence"] else 100
        monkeypatch.setattr("reviewer.structured_review.request_tokens", estimate)
    monkeypatch.setattr("reviewer.agents.scan", lambda *args: ([], []))
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr("reviewer.agents.ChatOpenAI", partial(ChatOpenAI, http_client=client))
        reviewer = Reviewer(Activity(), model=GROQ_MODEL, base_url=GROQ_BASE_URL, api_key="unused")
        result = reviewer.review("quality", demo_snapshot())
    assert len(requests) == 2
    assert "incomplete" in result.error
    expected = "gathering stopped early" if scenario == "truncated_inspection" else "evidence was shortened"
    assert any(expected in limitation for limitation in result.limitations)
    assert any("omitted source and tests" in limitation for limitation in result.limitations)


def test_groq_revision_preserves_inline_patches_and_finding_identity(monkeypatch):
    finding = demo_result("quality").findings[0].assign_id()
    batch = {"findings": [finding.model_dump(mode="json")], "limitations": []}
    requests = []
    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if "response_format" not in payload:
            return completion({"role": "assistant", "content": "Existing finding inspected."})
        assert not any(key in payload for key in ("tools", "tool_choice", "parallel_tool_calls"))
        assert payload["response_format"]["json_schema"]["name"] == "ReviewBatch"
        return completion({"role": "assistant", "content": json.dumps(batch)})
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr("reviewer.agents.ChatOpenAI", partial(ChatOpenAI, http_client=client))
        reviewer = Reviewer(Activity(), model=GROQ_MODEL, base_url=GROQ_BASE_URL, api_key="unused")
        revised = reviewer.revise(finding, demo_snapshot(), "Keep the fix minimal.")
    assert len(requests) == 2
    assert revised.id == finding.id
    assert revised.edits == finding.edits
    assert revised.edits
