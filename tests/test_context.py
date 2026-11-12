import json
from functools import partial

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from reviewer.activity import Activity
from reviewer.agent_runtime import bounded_messages
from reviewer.agents import Reviewer


def test_pruning_keeps_tool_exchanges_complete():
    messages = [SystemMessage(content="instructions"), HumanMessage(content="repository"),
                AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "old"}]),
                ToolMessage(content="x" * 25000, tool_call_id="old"),
                AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "new"}]),
                ToolMessage(content="evidence", tool_call_id="new"), HumanMessage(content="final review")]
    result = bounded_messages(messages)
    assert result == messages[:2] + messages[4:]
    assert len(messages) == 7


def test_oversized_initial_context_has_clear_error():
    with pytest.raises(ValueError, match="context budget exceeded"):
        bounded_messages([SystemMessage(content="instructions"), HumanMessage(content="x" * 25000)])


def test_local_timeout_and_output_budget():
    reviewer = Reviewer(Activity(), model="code-review-local", base_url="http://localhost:11434/v1", api_key="ollama")
    model = reviewer.llm()
    assert model.request_timeout == 600
    assert model.extra_body == {"max_tokens": 2048}
    assert model.reasoning_effort == "none"
    assert model.max_retries == 0


@pytest.mark.parametrize("base_url,local", [
    ("http://127.0.0.1:11434/v1", True),
    ("https://api.openai.com/v1", False),
])
def test_local_model_controls_reach_tool_call_requests(monkeypatch, base_url, local):
    requests = []

    def respond(request):
        assert request.url.path == "/v1/chat/completions"
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "local-control-check", "object": "chat.completion", "created": 0,
            "model": "test-review-model",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{
                    "id": "review-quality", "type": "function", "function": {
                        "name": "task",
                        "arguments": json.dumps({"subagent_type": "quality", "description": "Review correctness"}),
                    },
                }],
            }}],
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr("reviewer.agents.ChatOpenAI", partial(ChatOpenAI, http_client=client))
        reviewer = Reviewer(Activity(), model="test-review-model", base_url=base_url, api_key="unused")
        task = {"type": "function", "function": {
            "name": "task", "description": "Delegate a review to a specialist.",
            "parameters": {"type": "object", "properties": {
                "subagent_type": {"type": "string"}, "description": {"type": "string"},
            }, "required": ["subagent_type", "description"]},
        }}
        response = reviewer.model_for("orchestrator").bind_tools([task]).invoke("Review this synthetic snapshot.")

    assert response.tool_calls[0]["args"]["subagent_type"] == "quality"
    assert len(requests) == 1
    payload = requests[0]
    assert payload["tools"][0]["function"]["name"] == "task"
    if local:
        assert payload["reasoning_effort"] == "none"
        assert payload["max_tokens"] == 2048
        assert "max_completion_tokens" not in payload
    else:
        assert "reasoning_effort" not in payload
        assert "max_tokens" not in payload
        assert "max_completion_tokens" not in payload
