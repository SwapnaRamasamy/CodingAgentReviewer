"""Failure diagnostics preserve useful types without exposing request data."""
import logging

import pytest

from reviewer.activity import Activity
from reviewer.diagnostics import safe_error


def test_unknown_failure_keeps_exception_type_without_message_or_credentials(caplog):
    activity = Activity()

    def fail():
        private_key = "PRIVATE_API_KEY_DO_NOT_LOG"
        raise TypeError("Provider response contains PRIVATE_SOURCE_DO_NOT_LOG " + private_key)

    with caplog.at_level(logging.WARNING), pytest.raises(TypeError) as caught:
        activity.run("orchestrator", "model.deep_agent", fail)
    message = safe_error(caught.value)
    assert "TypeError" in message
    assert "reviewer.activity:" in message
    assert "TypeError" in activity.snapshot()[-1]["summary"]
    assert "PRIVATE_API_KEY" not in message + str(activity.snapshot()) + caplog.text
    assert "PRIVATE_SOURCE" not in message + str(activity.snapshot()) + caplog.text
    assert str(__file__) not in message + caplog.text


def test_chained_failure_keeps_underlying_type_without_raw_payload():
    try:
        try:
            raise TypeError("PRIVATE_PROVIDER_RESPONSE")
        except TypeError as cause:
            raise RuntimeError("PRIVATE_WRAPPER_MESSAGE") from cause
    except RuntimeError as error:
        message = safe_error(error)
    assert "RuntimeError" in message
    assert "TypeError" in message
    assert "PRIVATE_" not in message


def test_unknown_exception_class_name_from_untrusted_module_is_not_rendered():
    error_type = type("PRIVATE_SECRET_CLASS_NAME", (Exception,), {"__module__": "untrusted"})
    message = safe_error(error_type("PRIVATE_PAYLOAD"))
    assert "unclassified exception" in message
    assert "PRIVATE_" not in message


@pytest.mark.parametrize("code,expected", [
    ("tool_use_failed", "missing or invalid tool call"),
    ("json_validate_failed", "rejected generated JSON"),
    ("PRIVATE_UNKNOWN_CODE", "do not identify the cause"),
])
def test_bad_request_reports_only_recognized_codes(code, expected, caplog):
    import httpx
    from openai import BadRequestError

    error = BadRequestError(
        "PRIVATE_EXCEPTION_MESSAGE",
        response=httpx.Response(400, request=httpx.Request("POST", "https://example.invalid")),
        body={"error": {
            "code": code, "message": "PRIVATE_PROVIDER_PAYLOAD",
            "failed_generation": "PRIVATE_GENERATED_SOURCE",
        }},
    )
    activity = Activity()

    def fail():
        raise error

    with caplog.at_level(logging.WARNING), pytest.raises(BadRequestError):
        activity.run("quality", "model.deep_agent", fail)
    message = safe_error(error)
    assert expected in message
    assert "400" in message
    if code in {"tool_use_failed", "json_validate_failed"}:
        assert code in message
    assert "PRIVATE_" not in message + str(activity.snapshot()) + caplog.text
