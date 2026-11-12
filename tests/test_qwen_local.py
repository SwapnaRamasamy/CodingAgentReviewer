"""Real local agent graphs and Ollama-compatible JSON requests, without inference."""
import json
from functools import partial

import httpx
import pytest
from langchain_openai import ChatOpenAI

from reviewer.activity import Activity
from reviewer.agents import Reviewer, demo_result
from reviewer.local_review import LocalFinding, LocalReviewBatch
from reviewer.model_config import LOCAL_BASE_URL, LOCAL_MODEL
from reviewer.sources import demo_snapshot


def completion(content=None, tool=None):
    message = {"role": "assistant", "content": content}
    if tool:
        name, arguments, call_id = tool
        message["tool_calls"] = [{
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]
    return httpx.Response(200, json={
        "id": "local-offline", "object": "chat.completion", "created": 0, "model": LOCAL_MODEL,
        "choices": [{"index": 0, "finish_reason": "tool_calls" if tool else "stop", "message": message}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},
    })


def compact_finding(role):
    original = demo_result(role).findings[0].model_dump(mode="json")
    return {key: original[key] for key in LocalFinding.model_fields}


def anchored_payload(payload, role, invalid=False):
    data = json.loads(payload["messages"][-1]["content"])
    original = demo_result(role).findings[0]
    reference = next(item for item in data["source_anchors"]
                     if item["path"] == original.path and item["line"] == original.line)
    finding = {key: value for key, value in compact_finding(role).items()
               if key not in {"path", "line", "evidence"}}
    finding["anchor_id"] = "UNKNOWN_REFERENCE" if invalid else reference["id"]
    return {"findings": [finding], "limitations": []}


@pytest.mark.parametrize("include_tests", [False, True])
def test_qwen_local_hierarchy_separates_tools_and_json(monkeypatch, include_tests):
    roles = ("quality", "security", "tests") if include_tests else ("quality", "security")
    counts = dict.fromkeys(("orchestrator", *roles), 0)
    phases = []
    def respond(request):
        assert str(request.url) == LOCAL_BASE_URL + "/chat/completions"
        payload = json.loads(request.content)
        assert payload["model"] == LOCAL_MODEL
        assert payload["reasoning_effort"] == "none"
        assert payload["max_tokens"] == 2048
        assert "max_completion_tokens" not in payload
        markers = {"quality": "quality code-review subagent", "security": "security code-review subagent",
                   "tests": "test-review subagent"}
        system = payload["messages"][0]["content"]
        role = next((role for role, marker in markers.items() if marker in system), "orchestrator")
        index = counts[role]
        counts[role] += 1
        if "response_format" in payload:
            assert role in roles
            assert not any(key in payload for key in ("tools", "tool_choice", "parallel_tool_calls"))
            schema = payload["response_format"]["json_schema"]
            assert schema["name"] == "AnchoredLocalReviewBatch"
            assert "Return JSON matching this schema" in system
            assert "at most one" in system
            phases.append((role, "json"))
            return completion(json.dumps(anchored_payload(payload, role)))
        names = {tool["function"]["name"] for tool in payload.get("tools", [])}
        assert "ReviewBatch" not in names and "LocalReviewBatch" not in names
        assert payload.get("tool_choice", "auto") != "required"
        if role == "orchestrator":
            assert names == {"task", "write_todos"}
            if index < len(roles):
                return completion(tool=("task", {"subagent_type": roles[index], "description": "Inspect source"},
                                        f"task-{index}"))
            return completion("Selected review results collected.")
        assert names == {"read_file", "search_code"}
        phases.append((role, "evidence"))
        if index == 0:
            return completion(tool=("read_file", {"path": "app.py", "start_line": 1, "end_line": 12},
                                    f"read-{role}"))
        return completion("Evidence inspected; return the final review separately.")
    monkeypatch.setattr("reviewer.agents.scan", lambda *args: ([], []))
    monkeypatch.setattr("reviewer.agents.shared_groq_limiter",
                        lambda *args: pytest.fail("Local inference must not use Groq pacing"))
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr("reviewer.agents.ChatOpenAI", partial(ChatOpenAI, http_client=client))
        reviewer = Reviewer(Activity(), model=LOCAL_MODEL, base_url=LOCAL_BASE_URL, api_key="ollama",
                            include_tests=include_tests)
        result = reviewer.orchestrate(demo_snapshot())
    assert not result["orchestration_error"], result
    assert [item.agent for item in result["results"]] == list(roles)
    assert all(not item.error for item in result["results"]), result
    assert all(len(item.findings) == 1 for item in result["results"])
    assert all(item.findings[0].edits == [] for item in result["results"])
    assert counts == {"orchestrator": len(roles) + 1, **dict.fromkeys(roles, 3)}
    assert all((role, "json") in phases for role in roles)
    assert not any(event["tool"] == "model.rate_limit" for event in reviewer.activity.snapshot())


def test_qwen_local_invalid_evidence_gets_one_bounded_correction(monkeypatch):
    native_calls, requests = [0], []
    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if "response_format" not in payload:
            return completion("Inspect supplied excerpts.")
        native_calls[0] += 1
        return completion(json.dumps(anchored_payload(payload, "quality", invalid=native_calls[0] == 1)))
    monkeypatch.setattr("reviewer.agents.scan", lambda *args: ([], []))
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr("reviewer.agents.ChatOpenAI", partial(ChatOpenAI, http_client=client))
        reviewer = Reviewer(Activity(), model=LOCAL_MODEL, base_url=LOCAL_BASE_URL, api_key="ollama")
        result = reviewer.review("quality", demo_snapshot())
    assert not result.error, result
    assert len(result.findings) == 1
    assert len(requests) == 4
    assert any(event["status"] == "retrying" for event in reviewer.activity.snapshot())


def test_qwen_local_final_output_cap_reaches_the_wire(monkeypatch):
    requests = []
    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return completion('{"findings": [], "limitations": []}')
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr("reviewer.agents.ChatOpenAI", partial(ChatOpenAI, http_client=client))
        model = Reviewer(Activity(), model=LOCAL_MODEL, base_url=LOCAL_BASE_URL, api_key="ollama").llm(
            tool_calls=False, output_tokens=1024,
        )
        result = model.with_structured_output(
            LocalReviewBatch, method="json_schema", strict=True,
        ).invoke("Return an empty findings list and empty limitations list.")
    assert isinstance(result, LocalReviewBatch)
    assert requests[0]["max_tokens"] == 1024
    assert "tools" not in requests[0]


def test_qwen_local_oversized_gathering_only_sends_the_bounded_final_request(monkeypatch):
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        return completion('{"findings": [], "limitations": []}')
    monkeypatch.setattr("reviewer.agent_runtime.request_tokens", lambda *args, **kwargs: 7501)
    monkeypatch.setattr("reviewer.agents.scan", lambda *args: ([], []))
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr("reviewer.agents.ChatOpenAI", partial(ChatOpenAI, http_client=client))
        reviewer = Reviewer(Activity(), model=LOCAL_MODEL, base_url=LOCAL_BASE_URL, api_key="ollama")
        result = reviewer.review("quality", demo_snapshot())
    assert "incomplete" in result.error
    assert any("gathering stopped early" in limitation for limitation in result.limitations)
    assert len(requests) == 1
    assert "response_format" in requests[0]
    assert "tools" not in requests[0]


def test_local_source_references_are_bound_to_real_changed_lines():
    from langchain_core.messages import AIMessage

    from reviewer.local_anchors import anchored_review, anchored_schema, source_anchors

    snapshot = demo_snapshot().model_copy(update={
        "commit_scoped": True, "changed_paths": ["app.py"], "changed_lines": {"app.py": [4]},
    })
    history = [AIMessage(content="", tool_calls=[{
        "name": "read_file", "id": "read", "args": {"path": "app.py", "start_line": 7, "end_line": 10},
    }])]
    anchors = source_anchors(snapshot, history)
    assert len(anchors) == 1
    reference = next(iter(anchors.values()))
    assert reference["line"] == 4
    assert reference["evidence"] == snapshot.files["app.py"].splitlines()[3]
    schema = anchored_schema(anchors)
    assert schema["$defs"]["AnchoredLocalFinding"]["properties"]["anchor_id"]["enum"] == list(anchors)
    draft = {key: value for key, value in compact_finding("security").items()
             if key not in {"path", "line", "evidence"}}
    draft["anchor_id"] = reference["id"]
    batch = anchored_review({"findings": [draft], "limitations": []}, anchors).validated_review(snapshot)
    assert batch.findings[0].line == batch.findings[0].end_line == 4
    assert batch.findings[0].evidence == reference["evidence"]


def test_unknown_local_source_reference_cannot_create_a_citation():
    from reviewer.diagnostics import InvalidReviewEvidence
    from reviewer.local_anchors import anchored_review, source_anchors

    anchors = source_anchors(demo_snapshot(), [])
    draft = {key: value for key, value in compact_finding("quality").items()
             if key not in {"path", "line", "evidence"}}
    draft["anchor_id"] = "S_DOES_NOT_EXIST"
    with pytest.raises(InvalidReviewEvidence):
        anchored_review({"findings": [draft], "limitations": []}, anchors)


def test_local_source_reference_budget_is_bounded_and_ignores_bad_tool_arguments():
    from langchain_core.messages import AIMessage

    from reviewer.local_anchors import source_anchors

    history = [AIMessage(content="", tool_calls=[{
        "name": "read_file", "id": "bad", "args": {"path": [], "start_line": "bad", "end_line": "bad"},
    }])]
    anchors = source_anchors(demo_snapshot(), history, budget=400)
    assert anchors
    assert sum(len(json.dumps(item, ensure_ascii=False).encode("utf-8")) for item in anchors.values()) <= 400
