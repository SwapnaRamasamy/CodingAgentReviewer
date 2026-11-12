"""Budget, capability, and activity boundaries around framework-managed agent loops."""
from __future__ import annotations

import json
from threading import Lock

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from reviewer.diagnostics import AgentBudgetExceeded, ContextBudgetExceeded
from reviewer.provider_limits import rate_limit_delay, request_tokens

LOCAL_MODEL_LOCK = Lock()
LOCAL_MESSAGE_BYTES = 24000


def bounded_messages(messages, budget=LOCAL_MESSAGE_BYTES):
    result = list(messages)
    def size():
        return len(json.dumps([m.model_dump(mode="json") for m in result], ensure_ascii=False).encode("utf-8"))
    while size() > budget:
        if len(result) > 3 and result[2].type == "ai":
            end = 3
            while end < len(result) and result[end].type == "tool":
                end += 1
            del result[2:end]
        else:
            raise ContextBudgetExceeded("Local model context budget exceeded; use a smaller review scope.")
    return result


class ReviewMiddleware(AgentMiddleware):
    """One instance per invocation; the global lock covers inference only, never task()."""
    def __init__(self, activity, agent, system_prompt, local, allowed_tools, max_models=6, max_tools=8,
                 subagent_roles=("quality", "security", "tests"), omit_message_names=False,
                 rate_limiter=None, reserved_models=0):
        self.activity, self.agent, self.system_prompt = activity, agent, system_prompt
        self.local, self.allowed_tools = local, frozenset(allowed_tools)
        self.max_models, self.max_tools = max_models, max_tools
        self.subagent_roles = frozenset(subagent_roles)
        self.omit_message_names = omit_message_names
        self.rate_limiter, self.reserved_models = rate_limiter, reserved_models
        self.last_messages = []
        self.model_calls = self.tool_calls = 0
        self.counter_lock = Lock()

    def wrap_model_call(self, request, handler):
        with self.counter_lock:
            exhausted = self.tool_calls >= self.max_tools
            if self.reserved_models and self.model_calls >= self.max_models - self.reserved_models - 1:
                exhausted = True
        tools = [t for t in request.tools if (t.get("name") if isinstance(t, dict) else t.name) in self.allowed_tools]
        if exhausted:
            tools = []
        system = SystemMessage(content=self.system_prompt)
        messages = list(request.messages)
        if self.omit_message_names:
            # Groq rejects messages[].name. Keep graph history and function names intact.
            messages = [
                message.model_copy(update={
                    "name": None,
                    "additional_kwargs": {
                        key: value for key, value in message.additional_kwargs.items() if key != "name"
                    },
                })
                for message in messages
            ]
        if self.local:
            schema_size = len(json.dumps([convert_to_openai_tool(t) for t in tools]).encode("utf-8"))
            # Reserve extra room for framework structured-output schemas and template tokens.
            budget = LOCAL_MESSAGE_BYTES - schema_size
            bounded = bounded_messages([system, *messages], budget=budget)
            messages = bounded[1:]
        self.last_messages = messages
        updated = request.override(system_message=system, messages=messages, tools=tools)
        def invoke():
            return handler(updated)
        estimate = request_tokens([system, *messages], tools=tools) if self.rate_limiter else 0
        if self.local:
            estimate = request_tokens([system, *messages], tools=tools, output_tokens=2048)
            if estimate > 7500:
                raise ContextBudgetExceeded("Local inference exceeds the bounded 8K context allowance.")
        return self.call_model(invoke, estimated_tokens=estimate)

    def call_model(self, function, *, final=False, estimated_tokens=0):
        for attempt in range(2):
            with self.counter_lock:
                limit = self.max_models if final else self.max_models - self.reserved_models
                if self.model_calls >= limit:
                    raise AgentBudgetExceeded("Model call budget exhausted; review is incomplete.")
                self.model_calls += 1
            def invoke():
                if self.local:
                    with LOCAL_MODEL_LOCK:
                        return function()
                if self.rate_limiter:
                    return self.rate_limiter.run(function, estimated_tokens, self.activity, self.agent)
                return function()
            try:
                return self.activity.run(
                    self.agent, "model.structured_review" if final else "model.deep_agent", invoke,
                )
            except Exception as error:
                delay = rate_limit_delay(error) if self.rate_limiter else None
                if attempt or delay is None or delay > 60:
                    raise
                self.activity.add(
                    self.agent, "model.rate_limit", "waiting",
                    summary="Provider requested a cooldown; allowing one retry within the existing call budget.",
                )
        raise AssertionError("Unreachable model retry state")

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        name = call["name"]
        with self.counter_lock:
            denied = name not in self.allowed_tools or self.tool_calls >= self.max_tools
            if not denied:
                self.tool_calls += 1
        if name == "task" and call.get("args", {}).get("subagent_type") not in self.subagent_roles:
            denied = True
        if denied:
            return ToolMessage(content="Tool unavailable or call budget exhausted.", tool_call_id=call["id"])
        if self.agent == "orchestrator":
            return self.activity.run(self.agent, name, handler, request)
        return handler(request)
