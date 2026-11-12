"""Optional test review changes selected scope without hiding reviewer failures."""
from functools import partial

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from test_deep_agents import call, setup_reviewer

from reviewer.activity import Activity
from reviewer.agents import Reviewer
from reviewer.models import AgentResult
from reviewer.reporting import markdown_report
from reviewer.sources import demo_snapshot
from reviewer.workflow import aggregate, build_workflow


@pytest.mark.parametrize("include_tests", [False, True])
def test_demo_workflow_completes_only_the_selected_reviewers(include_tests):
    reviewer = Reviewer(Activity(), demo=True, include_tests=include_tests)
    graph = build_workflow(reviewer)
    config = {"configurable": {"thread_id": "optional-tests-demo"}}
    snapshot = demo_snapshot()
    graph.invoke({"snapshot": snapshot, "results": []}, config)
    report = graph.get_state(config).values["report"]
    expected = {"quality", "security", "tests"} if include_tests else {"quality", "security"}
    assert report.complete
    assert {result.agent for result in report.agents} == expected
    assert {finding.agent for finding in report.findings} == expected
    assert ("tests" in reviewer.model_names()) is include_tests
    assert any("Tests reviewer disabled" in item for item in report.limitations) is not include_tests
    assert "Reviewers: " + ", ".join(sorted(expected)) in markdown_report(report, snapshot)


def test_disabled_tests_cannot_be_delegated_or_scanned(monkeypatch):
    parent = [
        call("task", {"subagent_type": role, "description": "Review"}, str(index))
        for index, role in enumerate(("tests", "quality", "security"))
    ] + [AIMessage(content="Selected reviews complete.")]
    reviewer, models = setup_reviewer(monkeypatch, parent=parent, include_tests=False)
    scanned = []

    def scan(snapshot, activity, role):
        scanned.append(role)
        return [], []

    monkeypatch.setattr("reviewer.agents.scan", scan)
    graph = build_workflow(reviewer)
    config = {"configurable": {"thread_id": "optional-tests-delegation"}}
    graph.invoke({"snapshot": demo_snapshot(), "results": []}, config)
    report = graph.get_state(config).values["report"]
    assert report.complete, report.limitations
    assert {result.agent for result in report.agents} == {"quality", "security"}
    assert scanned == ["quality", "security"]
    assert not models["tests"].seen
    assert len(models["quality"].seen) == len(models["security"].seen) == 2
    prompt = models["orchestrator"].seen[0][0][0].content
    assert "quality, security." in prompt
    assert "quality, security, tests" not in prompt
    assert "all three" not in prompt
    history = models["orchestrator"].seen[-1][0]
    assert any("Tool unavailable" in str(message.content) for message in history)
    assert any(event["agent"] == "tests" and event["status"] == "skipped"
               for event in reviewer.activity.snapshot())


@pytest.mark.parametrize("roles,failed,scope_complete,complete", [
    (("quality", "security"), "", True, True),
    (("quality",), "", True, False),
    (("quality", "quality"), "", True, False),
    (("quality", "tests"), "", True, False),
    (("quality", "security", "tests"), "", True, False),
    (("quality", "security"), "security", True, False),
    (("quality", "security"), "", False, False),
])
def test_optional_scope_requires_exact_successful_selected_results(roles, failed, scope_complete, complete):
    snapshot = demo_snapshot().model_copy(update={"scope_complete": scope_complete})
    results = [AgentResult(agent=role, error="Synthetic failure" if role == failed else "") for role in roles]
    report = aggregate(snapshot, results, expected_roles=("quality", "security"))
    assert report.complete is complete


def test_access_check_ignores_disabled_tests_model(monkeypatch):
    requested = []

    def respond(request):
        name = request.url.path.rsplit("/", 1)[-1]
        requested.append(name)
        assert name != "unavailable-tests-model"
        return httpx.Response(200, json={"id": name, "object": "model", "created": 0, "owned_by": "openai"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr("reviewer.agents.ChatOpenAI", partial(ChatOpenAI, http_client=client))
        reviewer = Reviewer(
            Activity(), model="orchestrator-model", api_key="unused", include_tests=False,
            base_url="https://api.openai.com/v1",
            role_models={"quality": "quality-model", "security": "security-model", "tests": "unavailable-tests-model"},
        )
        reviewer.check_model_access()
    assert requested == ["orchestrator-model", "quality-model", "security-model"]
