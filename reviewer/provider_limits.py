"""Bounded, shared pacing for Groq inference. API keys are never retained or logged."""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from functools import lru_cache
from threading import Lock

from langchain_core.utils.function_calling import convert_to_openai_tool

from reviewer.diagnostics import ContextBudgetExceeded


class ProviderRequestBudgetExceeded(ContextBudgetExceeded):
    pass


class ProviderCooldown(RuntimeError):
    pass


def rate_limit_delay(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        response = getattr(error, "response", None)
        if getattr(error, "status_code", None) == 429 or type(error).__name__ == "RateLimitError":
            try:
                delay = float(response.headers.get("retry-after", ""))
            except (AttributeError, TypeError, ValueError):
                delay = None
            if delay is not None and math.isfinite(delay) and 0 <= delay <= 86400:
                return delay
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    return None


def request_tokens(messages, tools=(), schema=None, output_tokens=1536):
    """Conservative estimate, including schema and reserved completion; not tokenizer telemetry."""
    payload = {
        "messages": [message.model_dump(mode="json") for message in messages],
        "tools": [convert_to_openai_tool(tool) for tool in tools],
        "schema": schema,
    }
    size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    return (size + 2) // 3 + 128 + output_tokens


def observed_tokens(value):
    if isinstance(value, (list, tuple)):
        counts = [observed_tokens(item) for item in value]
        known = [count for count in counts if count is not None]
        return sum(known) if known else None
    get = value.get if isinstance(value, dict) else lambda key, default=None: getattr(value, key, default)
    metadata = get("response_metadata", {}) or {}
    usage = get("usage_metadata") or metadata.get("token_usage", {})
    if isinstance(usage, dict):
        total = usage.get("total_tokens")
        if isinstance(total, (int, float)) and math.isfinite(total) and total >= 0:
            return int(total)
    for field in ("raw", "result"):
        nested = get(field)
        if nested is not None and nested is not value:
            return observed_tokens(nested)
    return None


class GroqRateLimiter:
    def __init__(self, *, token_limit=8000, request_limit=30,
                 clock=time.monotonic, sleep=time.sleep, max_wait=65):
        self.token_limit, self.request_limit = token_limit, request_limit
        self.clock, self.sleep, self.max_wait = clock, sleep, max_wait
        self._calls = deque()
        self._cooldown = 0.0
        self._inference = Lock()

    def run(self, function, estimated_tokens, activity, agent):
        estimate = max(1, int(estimated_tokens))
        if estimate > self.token_limit:
            raise ProviderRequestBudgetExceeded("Groq request exceeds the conservative per-minute token budget.")
        deadline, announced = self.clock() + self.max_wait, False

        def pause(delay):
            nonlocal announced
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise ProviderCooldown("Shared Groq capacity was not available within the wait limit.")
            if not announced:
                activity.add(agent, "model.rate_limit", "waiting",
                             summary="Waiting for shared Groq request and token capacity.")
                announced = True
            self.sleep(min(1.0, max(0.001, delay), remaining))

        # Lock inference only. Tool execution and nested subagents remain outside this lock.
        while not self._inference.acquire(blocking=False):
            pause(1.0)
        try:
            while True:
                now = self.clock()
                while self._calls and self._calls[0][0] + 60 <= now:
                    self._calls.popleft()
                delay = max(0.0, self._cooldown - now)
                over_requests = len(self._calls) >= self.request_limit
                over_tokens = sum(item[1] for item in self._calls) + estimate > self.token_limit
                if self._calls and (over_requests or over_tokens):
                    delay = max(delay, self._calls[0][0] + 60 - now)
                if delay <= 0:
                    break
                pause(delay)
            reservation = [self.clock(), estimate]
            self._calls.append(reservation)
            try:
                result = function()
            except Exception as error:
                delay = rate_limit_delay(error)
                if delay is not None:
                    self._cooldown = max(self._cooldown, self.clock() + delay)
                raise
            actual = observed_tokens(result)
            if actual is not None:
                reservation[1] = actual
            return result
        finally:
            self._inference.release()


_registry_lock = Lock()


@lru_cache(maxsize=256)
def _shared_limiter(key_digest, model):
    return GroqRateLimiter()


def shared_groq_limiter(api_key, model):
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    with _registry_lock:
        return _shared_limiter(digest, model)
