"""Provider pacing and retries use a virtual clock, never live APIs or real waits."""
from types import SimpleNamespace

import pytest

from reviewer.activity import Activity
from reviewer.agent_runtime import ReviewMiddleware
from reviewer.diagnostics import AgentBudgetExceeded, ContextBudgetExceeded, safe_error
from reviewer.provider_limits import (
    GroqRateLimiter,
    ProviderCooldown,
    observed_tokens,
    rate_limit_delay,
    shared_groq_limiter,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        assert 0 < seconds <= 1
        self.now += seconds


def limiter(**kwargs):
    clock = FakeClock()
    return GroqRateLimiter(clock=clock.clock, sleep=clock.sleep, **kwargs), clock, Activity()


def failure(delay=None, status=429):
    error = RuntimeError("PRIVATE_PROVIDER_BODY")
    error.status_code = status
    error.response = SimpleNamespace(
        status_code=status, headers={} if delay is None else {"retry-after": str(delay)},
    )
    return error


def throw(error):
    raise error


def test_token_window_waits_without_real_sleep():
    gate, clock, activity = limiter()
    gate.run(lambda: None, 8000, activity, "quality")
    gate.run(lambda: None, 1, activity, "security")
    assert clock.now == pytest.approx(60)
    assert len(activity.snapshot()) == 1


def test_request_window_waits():
    gate, clock, activity = limiter()
    for _ in range(30):
        gate.run(lambda: None, 1, activity, "quality")
    gate.run(lambda: None, 1, activity, "security")
    assert clock.now == pytest.approx(60)


def test_observed_usage_replaces_estimate():
    gate, clock, activity = limiter()
    message = SimpleNamespace(usage_metadata={"total_tokens": 10})
    gate.run(lambda: {"raw": message}, 8000, activity, "quality")
    gate.run(lambda: None, 7990, activity, "security")
    assert clock.now == 0
    assert observed_tokens(SimpleNamespace(result=[message])) == 10
    assert observed_tokens(None) is None


def test_failed_attempt_keeps_token_reservation():
    gate, clock, activity = limiter()
    with pytest.raises(RuntimeError):
        gate.run(lambda: throw(failure()), 8000, activity, "quality")
    gate.run(lambda: None, 1, activity, "security")
    assert clock.now == pytest.approx(60)


def test_retry_after_cooldown_is_shared_without_hidden_retry():
    gate, clock, activity = limiter()
    calls = []
    def invoke():
        calls.append(True)
        raise failure("2.5")
    with pytest.raises(RuntimeError):
        gate.run(invoke, 1, activity, "quality")
    assert len(calls) == 1
    gate.run(lambda: None, 1, activity, "security")
    assert clock.now == pytest.approx(2.5)


def test_long_cooldown_has_bounded_wait_and_releases_lock():
    gate, clock, activity = limiter()
    with pytest.raises(RuntimeError):
        gate.run(lambda: throw(failure(120)), 1, activity, "quality")
    with pytest.raises(ProviderCooldown) as caught:
        gate.run(lambda: None, 1, activity, "security")
    assert clock.now == pytest.approx(65)
    assert "bounded wait" in safe_error(caught.value)
    clock.now = 121
    gate.run(lambda: None, 1, activity, "quality")


def test_oversized_request_is_not_sent():
    gate, _, activity = limiter()
    with pytest.raises(ContextBudgetExceeded) as caught:
        gate.run(lambda: pytest.fail("must not invoke"), 8001, activity, "quality")
    assert "conservative Groq token budget" in safe_error(caught.value)


@pytest.mark.parametrize("delay", [None, "invalid", "nan", "inf", "-1", "86401"])
def test_invalid_retry_after_does_not_authorize_retry(delay):
    assert rate_limit_delay(failure(delay)) is None


def test_retry_after_requires_known_rate_limit_and_stays_safe():
    assert rate_limit_delay(failure(1, status=400)) is None
    wrapped = ValueError("PRIVATE_OUTER")
    wrapped.__cause__ = failure(2)
    assert rate_limit_delay(wrapped) == 2
    assert "at least 2 seconds" in safe_error(wrapped)
    assert "PRIVATE_" not in safe_error(wrapped)


def test_registry_shares_by_digest_and_model():
    first = shared_groq_limiter("FAKE_TEST_KEY_A", "model-a")
    assert first is shared_groq_limiter("FAKE_TEST_KEY_A", "model-a")
    assert first is not shared_groq_limiter("FAKE_TEST_KEY_B", "model-a")
    assert first is not shared_groq_limiter("FAKE_TEST_KEY_A", "model-b")


def test_failed_429_retries_consume_the_existing_call_budget():
    gate, _, activity = limiter()
    runtime = ReviewMiddleware(activity, "quality", "", False, set(), rate_limiter=gate)
    calls = []
    def invoke():
        calls.append(True)
        raise failure(0)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            runtime.call_model(invoke, estimated_tokens=1)
    with pytest.raises(AgentBudgetExceeded):
        runtime.call_model(invoke, estimated_tokens=1)
    assert len(calls) == runtime.model_calls == 6


def test_inspection_reserves_the_final_call():
    runtime = ReviewMiddleware(Activity(), "quality", "", False, set(), reserved_models=1)
    for _ in range(5):
        runtime.call_model(lambda: None)
    with pytest.raises(AgentBudgetExceeded):
        runtime.call_model(lambda: pytest.fail("no inspection budget left"))
    runtime.call_model(lambda: None, final=True)
    assert runtime.model_calls == 6
