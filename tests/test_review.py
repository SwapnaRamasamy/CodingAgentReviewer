import asyncio
from types import SimpleNamespace

import pytest

from reviewer.activity import Activity
from reviewer.agents import Reviewer, snapshot_tools
from reviewer.fixes import apply_edits, patch_text, validate_edits
from reviewer.github import GitHub, decode_result, file_text
from reviewer.models import AgentResult, Edit
from reviewer.publishing import publish_draft
from reviewer.sources import demo_snapshot, eligible, local_snapshot, parse_github_url
from reviewer.workflow import aggregate, build_workflow, resume_review


def test_graph_pauses_then_applies_only_approved_fix():
    snapshot = demo_snapshot()
    reviewer = Reviewer(Activity(), demo=True)
    graph = build_workflow(reviewer)
    config = {"configurable": {"thread_id": "test-approval"}}
    graph.invoke({"snapshot": snapshot, "results": []}, config)
    state = graph.get_state(config)
    assert state.next == ("human_review",)
    report = state.values["report"]
    assert len(report.agents) == 3
    assert len(report.findings) == 3
    assert report.findings[0].severity.value == "High"
    finding = next(f for f in report.findings if f.agent == "quality")
    decisions = [
        {"finding_id": f.id, "action": "approve" if f.id == finding.id else "reject"}
        for f in report.findings
    ]
    result = resume_review(graph, config, decisions)
    assert not graph.get_state(config).next
    assert "raise ValueError" in result["updated_files"]["app.py"]
    assert "shell=True" in result["updated_files"]["app.py"]
    assert "db.execute" in result["updated_files"]["app.py"]
    assert "raise ValueError" not in snapshot.files["app.py"]


def test_graph_can_reopen_human_review():
    graph = build_workflow(Reviewer(Activity(), demo=True))
    config = {"configurable": {"thread_id": "reopen"}}
    graph.invoke({"snapshot": demo_snapshot(), "results": []}, config)
    report = graph.get_state(config).values["report"]
    resume_review(graph, config, [{"finding_id": f.id, "action": "reject"} for f in report.findings])
    graph.update_state(config, {"report": report}, as_node="combine")
    graph.invoke(None, config)
    assert graph.get_state(config).next == ("human_review",)
    result = resume_review(graph, config, [{"finding_id": f.id, "action": "reject"} for f in report.findings])
    assert result["patch"] == ""


def test_partial_failure_keeps_other_reviewers():
    class FailingReviewer(Reviewer):
        def review(self, agent, snapshot):
            if agent == "security":
                raise RuntimeError("do not expose credentials")
            return super().review(agent, snapshot)
    graph = build_workflow(FailingReviewer(Activity(), demo=True))
    config = {"configurable": {"thread_id": "failure"}}
    graph.invoke({"snapshot": demo_snapshot(), "results": []}, config)
    report = graph.get_state(config).values["report"]
    assert not report.complete
    assert len(report.findings) == 2
    assert "do not expose" not in report.model_dump_json()


def test_invalid_evidence_and_patch_are_handled():
    snapshot = demo_snapshot()
    finding = Reviewer(Activity(), demo=True).review("quality", snapshot).findings[0]
    bad_evidence = finding.model_copy(update={"evidence": "invented source"})
    report = aggregate(snapshot, [AgentResult(agent="quality", findings=[bad_evidence])])
    assert not report.findings
    finding.edits = [Edit(path="app.py", original="missing", replacement="pass")]
    report = aggregate(snapshot, [AgentResult(agent="quality", findings=[finding])])
    assert len(report.findings) == 1
    assert not report.findings[0].edits
    assert report.limitations


def test_duplicate_finding_preserves_sources_and_fix():
    snapshot = demo_snapshot()
    finding = Reviewer(Activity(), demo=True).review("quality", snapshot).findings[0]
    other = finding.model_copy(deep=True)
    other.agent = "security"
    report = aggregate(snapshot, [
        AgentResult(agent="quality", findings=[finding]),
        AgentResult(agent="security", findings=[other]),
    ])
    assert len(report.findings) == 1
    assert report.findings[0].sources == ["quality", "security"]


def test_ambiguous_conflicting_and_unsafe_edits_rejected():
    with pytest.raises(ValueError):
        apply_edits({"a.py": "x x"}, [Edit(path="a.py", original="x", replacement="y")])
    with pytest.raises(ValueError):
        apply_edits({"a.py": "hello"}, [
            Edit(path="a.py", original="hello", replacement="world"),
            Edit(path="a.py", original="hello", replacement="other"),
        ])
    with pytest.raises(ValueError):
        apply_edits({"a.py": "x"}, [Edit(path="../outside", original="x", replacement="y")])
    with pytest.raises(ValueError):
        validate_edits({"a.py": "x = 1\n"}, [Edit(path="a.py", original="1", replacement="(")])


def test_patch_and_identical_edits():
    edit = Edit(path="a.py", original="x = 1\n", replacement="x = 2\n")
    changed = apply_edits({"a.py": "x = 1\n"}, [edit, edit])
    patch = patch_text({"a.py": "x = 1\n"}, changed)
    assert "--- a/a.py" in patch
    assert "+x = 2" in patch


def test_local_snapshot_exclusions(tmp_path):
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=secret", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("ignored = 1", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / ".gitignore").write_text("skip.py\n", encoding="utf-8")
    (tmp_path / "tests" / "skip.py").write_text("skip = 1", encoding="utf-8")
    snapshot = local_snapshot(tmp_path)
    assert snapshot.files == {"app.py": "x = 1\n"}
    assert len(snapshot.revision) == 64
    assert not eligible("../a.py")
    assert not eligible("node_modules/a.py")


def test_github_urls():
    assert parse_github_url("https://github.com/acme/project/pull/42") == ("acme", "project", 42)
    assert parse_github_url("https://github.com/acme/project.git") == ("acme", "project", None)
    with pytest.raises(ValueError):
        parse_github_url("https://github.com.evil/acme/project")


def test_snapshot_tools_are_bounded_and_no_external_access():
    activity = Activity()
    tools = {t.name: t for t in snapshot_tools(demo_snapshot(), activity, "security")}
    assert tools["read_file"].invoke({"path": "../secret"}) == "File unavailable in snapshot."
    assert "app.py:7" in tools["search_code"].invoke({"query": "shell=True"})
    assert "SECRET" not in str(activity.snapshot())


def test_mcp_content_decoding():
    assert file_text({"content": [{"type": "resource", "resource": {"text": "x = 1"}}]}) == "x = 1"
    assert file_text({"content": [{"type": "text", "text": "status"}, {"type": "text", "text": ""}]}) == ""
    assert file_text({"content": "eCA9IDE=", "encoding": "base64"}) == "x = 1"
    with pytest.raises(ValueError):
        file_text({"content": [{"type": "resource_link", "uri": "https://example.com"}]})
    result = SimpleNamespace(isError=False, structuredContent={"sha": "abc"}, content=[])
    assert decode_result(result) == {"sha": "abc"}
    with pytest.raises(RuntimeError):
        decode_result(SimpleNamespace(isError=True))


def test_mcp_readonly_allowlist_and_schema_validation():
    class Session:
        def __init__(self):
            self.calls = []
        async def call_tool(self, name, arguments):
            self.calls.append((name, arguments))
            return SimpleNamespace(isError=False, structuredContent={"sha": "a" * 40}, content=[])
    session = Session()
    tools = {"get_commit": SimpleNamespace(inputSchema={
        "type": "object", "required": ["owner", "repo", "sha"],
        "properties": {"owner": {"type": "string"}, "repo": {"type": "string"}, "sha": {"type": "string"}},
    })}
    github = GitHub(session, tools, Activity())
    assert asyncio.run(github.commit("acme", "repo", "main")) == "a" * 40
    with pytest.raises(ValueError):
        asyncio.run(github.call("push_files", owner="acme"))
    assert len(session.calls) == 1


def test_publish_stale_revision_never_writes():
    class FakeGitHub:
        writable = True
        tools = dict.fromkeys(["get_commit", "create_branch", "push_files", "create_pull_request"])
        async def commit(self, **kwargs):
            return "b" * 40
        async def call(self, *args, **kwargs):
            raise AssertionError("No writes expected")
    snapshot = demo_snapshot().model_copy(update={
        "owner": "acme", "repo": "repo", "branch": "main", "revision": "a" * 40,
    })
    updated = dict(snapshot.files, **{"app.py": "x = 1\n"})
    with pytest.raises(ValueError, match="changed since review"):
        asyncio.run(publish_draft(FakeGitHub(), snapshot, updated, "fix", "body"))


def test_publish_orders_writes_and_uses_draft():
    class FakeGitHub:
        writable = True
        tools = dict.fromkeys(["get_commit", "create_branch", "push_files", "create_pull_request"])
        def __init__(self):
            self.calls = []
        async def commit(self, **kwargs):
            return "a" * 40
        async def call(self, name, **kwargs):
            self.calls.append((name, kwargs))
            return {"html_url": "https://github.com/acme/repo/pull/1"}
    snapshot = demo_snapshot().model_copy(update={
        "owner": "acme", "repo": "repo", "branch": "main", "revision": "a" * 40,
    })
    github = FakeGitHub()
    updated = dict(snapshot.files, **{"app.py": "x = 1\n"})
    asyncio.run(publish_draft(github, snapshot, updated, "fix", "body", "review/test"))
    assert [name for name, _ in github.calls] == ["create_branch", "push_files", "create_pull_request"]
    assert github.calls[-1][1]["draft"] is True
    assert github.calls[-1][1]["base"] == "main"


def test_real_scanners_run_on_demo():
    from reviewer.scanners import scan
    findings, limitations = scan(demo_snapshot(), Activity(), "security")
    assert any("B602" in finding.title for finding in findings)
    assert not any("Bandit failed" in message for message in limitations)
