"""Exercise model routing and OpenAI request settings without network calls."""
import json
from functools import partial

import httpx
import pytest
from langchain_openai import ChatOpenAI

from reviewer.activity import Activity
from reviewer.agents import Reviewer
from reviewer.model_config import OPENAI_BASE_URL, OPENAI_MODELS, configured_models


def test_openai_defaults_and_role_overrides():
    assert configured_models(OPENAI_BASE_URL, environ={}) == OPENAI_MODELS
    overrides = {"REVIEW_QUALITY_MODEL": " gpt-5.5 ", "REVIEW_TESTS_MODEL": " "}
    assert configured_models(OPENAI_BASE_URL, environ=overrides) == {**OPENAI_MODELS, "quality": "gpt-5.5"}


def test_custom_provider_uses_shared_model_unless_overridden():
    names = configured_models(
        "http://localhost:11434/v1",
        {"REVIEW_MODEL": "local-review", "REVIEW_SECURITY_MODEL": "local-security"},
    )
    assert names == {
        "orchestrator": "local-review", "quality": "local-review",
        "security": "local-security", "tests": "local-review",
    }


def test_existing_callers_keep_shared_model_and_role_overrides_are_copied():
    reviewer = Reviewer(Activity(), model="shared", api_key="unused")
    assert set(reviewer.model_names().values()) == {"shared"}
    overrides = dict(OPENAI_MODELS)
    reviewer = Reviewer(Activity(), model="fallback", role_models=overrides, api_key="unused")
    overrides["quality"] = "changed-after-construction"
    assert reviewer.model_names() == OPENAI_MODELS


@pytest.mark.parametrize("role,expected", list(OPENAI_MODELS.items()))
def test_role_model_reaches_openai_request(monkeypatch, role, expected):
    requests = []

    def respond(request):
        assert str(request.url) == OPENAI_BASE_URL + "/chat/completions"
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json={
            "id": "offline-routing", "object": "chat.completion", "created": 0,
            "model": expected,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "Review complete."}}],
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr("reviewer.agents.ChatOpenAI", partial(ChatOpenAI, http_client=client))
        reviewer = Reviewer(
            Activity(), model=OPENAI_MODELS["orchestrator"], role_models=OPENAI_MODELS,
            base_url=OPENAI_BASE_URL, api_key="unused",
        )
        reviewer.model_for(role).invoke("Review this synthetic snapshot.")
    assert len(requests) == 1
    assert requests[0]["model"] == expected
    assert "temperature" not in requests[0]
    assert "max_tokens" not in requests[0]
    assert "reasoning_effort" not in requests[0]


def test_access_check_checks_each_distinct_role_model(monkeypatch):
    requested = []

    def respond(request):
        name = request.url.path.rsplit("/", 1)[-1]
        requested.append(name)
        return httpx.Response(200, json={
            "id": name, "object": "model", "created": 0, "owned_by": "openai",
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr("reviewer.agents.ChatOpenAI", partial(ChatOpenAI, http_client=client))
        reviewer = Reviewer(
            Activity(), model="unused-fallback", role_models=OPENAI_MODELS,
            base_url=OPENAI_BASE_URL, api_key="unused",
        )
        reviewer.check_model_access()
    assert requested == ["gpt-5.4-mini", "gpt-5.4"]


@pytest.mark.parametrize("failed_lookup", [False, True])
def test_access_checks_leave_shared_transport_open_for_reviews(monkeypatch, failed_lookup):
    import openai

    requests = []
    rejected = False

    def respond(request):
        nonlocal rejected
        requests.append(request.url.path)
        if "/models/" in request.url.path:
            name = request.url.path.rsplit("/", 1)[-1]
            if failed_lookup and not rejected:
                rejected = True
                return httpx.Response(403, json={
                    "error": {"message": "Synthetic access denial", "type": "permission_error"},
                })
            return httpx.Response(200, json={
                "id": name, "object": "model", "created": 0, "owned_by": "openai",
            })
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "offline-after-access", "object": "chat.completion", "created": 0,
            "model": payload["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "Review complete."}}],
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as shared:
        # Keep ChatOpenAI's real client construction, including the shared-transport factory.
        monkeypatch.setattr("langchain_openai.chat_models.base._get_default_httpx_client",
                            lambda *args, **kwargs: shared)
        reviewer = Reviewer(
            Activity(), model=OPENAI_MODELS["orchestrator"], role_models=OPENAI_MODELS,
            base_url=OPENAI_BASE_URL, api_key="unused",
        )
        if failed_lookup:
            with pytest.raises(openai.PermissionDeniedError):
                reviewer.check_model_access()
        else:
            reviewer.check_model_access()
        assert not shared.is_closed
        later = Reviewer(
            Activity(), model=OPENAI_MODELS["orchestrator"], role_models=OPENAI_MODELS,
            base_url=OPENAI_BASE_URL, api_key="unused",
        )
        assert later.model_for("orchestrator").invoke("Synthetic review.").content == "Review complete."
        later.check_model_access()
        assert not shared.is_closed
    assert "/v1/chat/completions" in requests
    assert requests[-2:] == ["/v1/models/gpt-5.4-mini", "/v1/models/gpt-5.4"]


def test_local_qwen_defaults_and_shared_overrides():
    from reviewer.model_config import (
        DEFAULT_BASE_URL,
        DEFAULT_MODELS,
        LOCAL_BASE_URL,
        LOCAL_MODEL,
        LOCAL_MODELS,
    )

    assert DEFAULT_BASE_URL == LOCAL_BASE_URL
    assert DEFAULT_MODELS == LOCAL_MODELS
    assert configured_models(environ={}) == LOCAL_MODELS
    assert set(LOCAL_MODELS.values()) == {LOCAL_MODEL}
    assert configured_models(LOCAL_BASE_URL, {"REVIEW_MODEL": "other-local"}) == dict.fromkeys(
        LOCAL_MODELS, "other-local",
    )
