"""Local output validation uses the real specialist graph and SDK parser."""
from functools import partial

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError
from test_deep_agents import setup_reviewer

from reviewer.activity import Activity
from reviewer.agents import Reviewer, demo_result
from reviewer.diagnostics import InvalidReviewEvidence, MissingStructuredReview, safe_error
from reviewer.local_review import LocalReviewBatch
from reviewer.sources import demo_snapshot


def compact_payload():
    finding = demo_result("quality").findings[0]
    return {
        "findings": [{
            key: value for key, value in finding.model_dump(mode="json").items()
            if key in {"title", "severity", "confidence", "path", "line", "evidence",\
                       "impact", "suggested_fix", "suggested_tests"}
        }],
        "limitations": [],
    }


def test_compact_review_derives_locations_and_keeps_domain_validation():
    payload = compact_payload()
    batch = LocalReviewBatch.model_validate(payload).validated_review(demo_snapshot())
    finding = batch.findings[0]
    assert finding.line == finding.end_line == 10
    assert finding.edits == []
    assert finding.evidence in demo_snapshot().files[finding.path]
    payload["findings"][0]["line"] = 1
    with pytest.raises(InvalidReviewEvidence):
        LocalReviewBatch.model_validate(payload).validated_review(demo_snapshot())


def test_local_invalid_evidence_is_corrected_once(monkeypatch):
    import json

    reviewer, models = setup_reviewer(monkeypatch)
    reviewer.local = True
    invalid = compact_payload()
    invalid["findings"][0]["line"] = 1
    models["quality"].script = [
        AIMessage(content=json.dumps(invalid)), AIMessage(content=json.dumps(compact_payload())),
    ]
    result = reviewer.review("quality", demo_snapshot())
    assert not result.error
    assert len(result.findings) == 1
    assert len(models["quality"].seen) == 2
    assert any(event["status"] == "retrying" for event in reviewer.activity.snapshot())


def test_repeated_invalid_output_preserves_scanners_and_stays_failed(monkeypatch):
    reviewer, models = setup_reviewer(monkeypatch)
    reviewer.local = True
    scanner = demo_result("quality").findings[0]
    monkeypatch.setattr("reviewer.agents.scan", lambda *args: ([scanner], []))
    models["quality"].script = [AIMessage(content="{}"), AIMessage(content="{}")]
    result = reviewer.review("quality", demo_snapshot())
    assert result.error
    assert result.findings == [scanner]
    assert len(models["quality"].seen) == 2


def test_remote_invalid_output_does_not_add_a_review_retry(monkeypatch):
    reviewer, models = setup_reviewer(monkeypatch)
    models["quality"].script = [MissingStructuredReview("Synthetic invalid response")]
    result = reviewer.review("quality", demo_snapshot())
    assert result.error
    assert len(models["quality"].seen) == 1


def test_sdk_truncation_is_actionable_without_leaking_completion(monkeypatch):
    def respond(request):
        return httpx.Response(200, json={
            "id": "truncated", "object": "chat.completion", "created": 0, "model": "test-local",
            "choices": [{"index": 0, "finish_reason": "length", "message": {
                "role": "assistant", "content": "PRIVATE_COMPLETION_DO_NOT_LOG",
            }}],
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr("reviewer.agents.ChatOpenAI", partial(ChatOpenAI, http_client=client))
        reviewer = Reviewer(Activity(), model="test-local", base_url="http://127.0.0.1:11434/v1", api_key="unused")
        with pytest.raises(Exception) as caught:
            reviewer.specialist("tests", demo_snapshot()).invoke({"messages": [("user", "Synthetic review.")]})
    assert type(caught.value).__name__ == "LengthFinishReasonError"
    message = safe_error(caught.value)
    assert "truncated" in message
    assert "PRIVATE_COMPLETION" not in message
    assert "PRIVATE_COMPLETION" not in str(reviewer.activity.snapshot())


def test_local_revision_keeps_patch_capable_schema(monkeypatch):
    reviewer, models = setup_reviewer(monkeypatch)
    reviewer.local = True
    finding = demo_result("quality").findings[0]
    revised = reviewer.revise(finding, demo_snapshot(), "Keep the fix minimal.")
    assert revised.edits == finding.edits
    assert len(models["quality"].seen) == 2


def test_compact_review_rejects_unchanged_source():
    snapshot = demo_snapshot().model_copy(update={"commit_scoped": True, "changed_lines": {"app.py": [1]}})
    with pytest.raises(InvalidReviewEvidence):
        LocalReviewBatch.model_validate(compact_payload()).validated_review(snapshot)



def test_compact_review_rejects_whitespace_evidence():
    payload = compact_payload()
    payload["findings"][0]["evidence"] = " "
    with pytest.raises(InvalidReviewEvidence):
        LocalReviewBatch.model_validate(payload).validated_review(demo_snapshot())



def test_local_output_budget_rejects_multiple_model_findings():
    payload = compact_payload()
    payload["findings"] = payload["findings"] * 2
    with pytest.raises(ValidationError):
        LocalReviewBatch.model_validate(payload)
