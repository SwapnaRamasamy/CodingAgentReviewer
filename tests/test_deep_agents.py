"""Exercise the real Deep Agents task tool and LangChain specialist graphs offline."""
import json

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import Field

from reviewer.activity import Activity
from reviewer.agents import Reviewer, demo_result
from reviewer.prompts import ORCHESTRATOR_SYSTEM_PROMPT, SUBAGENT_SYSTEM_PROMPTS
from reviewer.sources import demo_snapshot
from reviewer.workflow import build_workflow, resume_review


class ScriptedModel(ChatOpenAI):
    script: list = Field(default_factory=list, exclude=True)
    seen: list = Field(default_factory=list, exclude=True)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append((messages, kwargs))
        if not self.script:
            raise AssertionError("Unexpected extra model call")
        response = self.script.pop(0)
        if isinstance(response, Exception):
            raise response
        if kwargs.get("response_format") and response.tool_calls and response.tool_calls[0]["name"] == "ReviewBatch":
            response = AIMessage(content=json.dumps(response.tool_calls[0]["args"]))
        return ChatResult(generations=[ChatGeneration(message=response)])


def call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def setup_reviewer(monkeypatch, parent=None, security_error=False, include_tests=True):
    monkeypatch.setattr("reviewer.agents.scan", lambda *args: ([], []))
    reviewer = Reviewer(Activity(), model="test-deep-review", api_key="unused", include_tests=include_tests)
    models = {}
    if parent is None:
        parent = [call("task", {"subagent_type": role, "description": f"Review {role}"}, role)
                  for role in SUBAGENT_SYSTEM_PROMPTS] + [AIMessage(content="All specialist outcomes collected.")]
    models["orchestrator"] = ScriptedModel(model="test-deep-review", api_key="unused", script=parent)
    for role in SUBAGENT_SYSTEM_PROMPTS:
        batch = {"findings": [f.model_dump(mode="json") for f in demo_result(role).findings], "limitations": []}
        script = [call("read_file", {"path": "app.py"}, role + "-read"),
                  call("ReviewBatch", batch, role + "-result")]
        if security_error and role == "security":
            script = [RuntimeError("PRIVATE_API_KEY_do_not_leak")]
        models[role] = ScriptedModel(model="test-deep-review", api_key="unused", script=script)
    reviewer.model_for = lambda role: models[role]
    return reviewer, models


def test_real_delegation_isolated_prompts_and_approval(monkeypatch):
    reviewer, models = setup_reviewer(monkeypatch)
    graph = build_workflow(reviewer)
    config = {"configurable": {"thread_id": "deep-test"}}
    graph.invoke({"snapshot": demo_snapshot(), "results": []}, config)
    state = graph.get_state(config)
    report = state.values["report"]
    assert report.complete, report.limitations
    assert state.next == ("human_review",)
    assert {r.agent for r in report.agents} == {"quality", "security", "tests"}
    assert len(report.findings) == 3
    assert models["orchestrator"].seen[0][0][0].content == ORCHESTRATOR_SYSTEM_PROMPT
    for role, prompt in SUBAGENT_SYSTEM_PROMPTS.items():
        messages, kwargs = models[role].seen[0]
        assert messages[0].content == prompt
        assert "initial_snippets" in messages[1].content
        assert not any(m.type == "tool" for m in messages)
        assert {t["function"]["name"] for t in kwargs["tools"]} == {"read_file", "search_code", "ReviewBatch"}
    parent_tools = {t["function"]["name"] for t in models["orchestrator"].seen[0][1]["tools"]}
    assert parent_tools == {"task", "write_todos"}
    # Final parent history has bounded summaries, not the specialists' source reads.
    parent_text = json.dumps([m.model_dump(mode="json") for m in models["orchestrator"].seen[-1][0]])
    assert "1: import subprocess" not in parent_text
    assert any(e["agent"] == "orchestrator" and e["tool"] == "task" for e in reviewer.activity.snapshot())
    before = {role: len(model.seen) for role, model in models.items()}
    result = resume_review(graph, config, [
        {"finding_id": f.id, "action": "approve" if f.agent == "quality" else "reject"}
        for f in report.findings
    ])
    assert "raise ValueError" in result["updated_files"]["app.py"]
    assert before == {role: len(model.seen) for role, model in models.items()}


def test_skipped_delegation_is_incomplete(monkeypatch):
    reviewer, _ = setup_reviewer(monkeypatch, parent=[AIMessage(content="Everything is fine.")])
    result = reviewer.orchestrate(demo_snapshot())
    assert len(result["results"]) == 3
    assert all("did not complete delegation" in item.error for item in result["results"])


def test_failed_specialist_preserves_others_without_leaking_errors(monkeypatch):
    reviewer, _ = setup_reviewer(monkeypatch, security_error=True)
    result = reviewer.orchestrate(demo_snapshot())
    by_role = {r.agent: r for r in result["results"]}
    assert not by_role["quality"].error
    assert by_role["security"].error
    assert not by_role["tests"].error
    assert "PRIVATE_API_KEY" not in str(result)
    assert "PRIVATE_API_KEY" not in str(reviewer.activity.snapshot())


def test_duplicate_delegation_does_not_repeat_specialist(monkeypatch):
    calls = [call("task", {"subagent_type": role, "description": "Review"}, str(i))
             for i, role in enumerate(["quality", "quality", "security", "tests"])]
    reviewer, models = setup_reviewer(monkeypatch, parent=calls + [AIMessage(content="Done")])
    result = reviewer.orchestrate(demo_snapshot())
    assert all(not r.error for r in result["results"]), result
    assert len(models["quality"].seen) == 2


def test_supervisor_cannot_call_filesystem_or_unknown_subagents(monkeypatch):
    parent = [call("write_file", {"file_path": "/bad.py", "content": "bad"}, "bad-write"),
              call("task", {"subagent_type": "general-purpose", "description": "Execute code"}, "bad-task"),
              AIMessage(content="Stopped")]
    reviewer, models = setup_reviewer(monkeypatch, parent=parent)
    result = reviewer.orchestrate(demo_snapshot())
    assert all(r.error for r in result["results"])
    assert not any(models[role].seen for role in SUBAGENT_SYSTEM_PROMPTS)
    assert any("unavailable" in str(m.content).lower() for m in models["orchestrator"].seen[-1][0])


def test_model_loop_is_bounded(monkeypatch):
    parent = [call("write_todos", {"todos": []}, str(i)) for i in range(12)]
    reviewer, models = setup_reviewer(monkeypatch, parent=parent)
    result = reviewer.orchestrate(demo_snapshot())
    assert "budget" in result["orchestration_error"]
    assert len(models["orchestrator"].seen) == 8


def test_orchestrator_failure_marks_report_incomplete(monkeypatch):
    parent = [call("task", {"subagent_type": role, "description": "Review"}, role)
              for role in SUBAGENT_SYSTEM_PROMPTS] + [RuntimeError("secret transport error")]
    reviewer, _ = setup_reviewer(monkeypatch, parent=parent)
    graph = build_workflow(reviewer)
    config = {"configurable": {"thread_id": "parent-failure"}}
    graph.invoke({"snapshot": demo_snapshot(), "results": []}, config)
    report = graph.get_state(config).values["report"]
    assert not report.complete
    assert len(report.findings) == 3
    assert all(not result.error for result in report.agents)
    assert any("Orchestrator:" in item for item in report.limitations)
    assert "secret transport error" not in report.model_dump_json()


def test_parallel_task_batch_with_serial_local_inference(monkeypatch):
    parent = [AIMessage(content="", tool_calls=[
        {"name": "task", "args": {"subagent_type": role, "description": "Review"}, "id": role}
        for role in SUBAGENT_SYSTEM_PROMPTS
    ]), AIMessage(content="All outcomes collected")]
    reviewer, models = setup_reviewer(monkeypatch, parent=parent)
    reviewer.local = True
    result = reviewer.orchestrate(demo_snapshot())
    assert not result["orchestration_error"], result
    assert all(not r.error for r in result["results"]), result
    assert all(len(models[role].seen) == 2 for role in SUBAGENT_SYSTEM_PROMPTS)


def test_revision_uses_specialist_prompt_and_preserves_finding_identity(monkeypatch):
    reviewer, models = setup_reviewer(monkeypatch)
    finding = demo_result("quality").findings[0]
    finding.assign_id()
    revised = reviewer.revise(finding, demo_snapshot(), "Keep the fix minimal")
    assert revised.id == finding.id
    assert revised.path == finding.path
    assert revised.line == finding.line
    assert "quality code-review subagent" in models["quality"].seen[0][0][0].content
    assert "Revise the proposed fix" in models["quality"].seen[0][0][0].content
    assert not models["orchestrator"].seen


def test_local_empty_json_cannot_count_as_completed_review(monkeypatch):
    reviewer, models = setup_reviewer(monkeypatch)
    reviewer.local = True
    models["quality"].script = [AIMessage(content="{}"), AIMessage(content="{}")]
    result = reviewer.review("quality", demo_snapshot())
    assert result.error
    assert not result.findings
    assert models["quality"].seen[0][1]["response_format"]["type"] == "json_schema"
