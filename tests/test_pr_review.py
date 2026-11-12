import base64
import hashlib
from copy import deepcopy
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from reviewer import pr_review as pr
from reviewer.activity import Activity
from reviewer.agents import Reviewer
from reviewer.commit_scope import added_lines, finding_in_scope, line_ranges
from reviewer.models import AgentResult, Finding, Severity
from reviewer.workflow import aggregate

OLD, HEAD, PARENT = "a" * 40, "b" * 40, "c" * 40
URL = "https://github.com/acme/project/pull/7"
SOURCE = "before = 1\nadded = 2\nafter = 3\n"
PATCH = "@@ -1,2 +1,3 @@\n before = 1\n+added = 2\n after = 3"


@pytest.fixture
def github(monkeypatch):
    state = SimpleNamespace(
        detail={"state": "open", "commits": 2, "head": {
            "sha": HEAD, "ref": "feature", "repo": {"full_name": "acme/project"},
        }},
        members=[OLD, HEAD], parents=[{"sha": PARENT}], calls=[],
        items=[{"filename": "app.py", "status": "modified", "patch": PATCH, "additions": 1, "deletions": 0,
                "sha": hashlib.sha1(b"blob " + str(len(SOURCE.encode())).encode() + b"\0" + SOURCE.encode()).hexdigest()}],
        sources={"app.py": SOURCE, ".gitignore": ""},
    )

    def request(client, path, body=None):
        state.calls.append((path, body))
        if body is not None:
            return {"id": 19, "html_url": URL + "#pullrequestreview-19"}
        if "/contents/" in path:
            from urllib.parse import unquote
            name = unquote(path.split("/contents/")[1].split("?")[0])
            if name not in state.sources:
                raise pr.PRReviewError("missing")
            text = state.sources[name]
            return {"type": "file", "size": len(text), "encoding": "base64",
                    "content": base64.b64encode(text.encode()).decode()}
        if "/compare/" in path:
            return {"files": deepcopy(state.items)}
        if "/pulls/7/commits?" in path:
            page = int(path.rsplit("page=", 1)[-1])
            return [{"sha": sha} for sha in state.members[(page - 1) * 100:page * 100]]
        if "/commits/" in path:
            sha = path.split("/commits/")[1].split("?")[0]
            page = int(path.rsplit("page=", 1)[-1])
            return {"sha": sha, "parents": deepcopy(state.parents),
                    "files": deepcopy(state.items[(page - 1) * 100:page * 100])}
        if path.endswith("/pulls/7"):
            return deepcopy(state.detail)
        raise AssertionError(path)

    monkeypatch.setattr(pr.GitHubPRClient, "request", request)
    return state


def test_collect_historical_commit_pins_source_and_parent(github):
    snapshot = pr.collect_pr_commit(URL, OLD[:8])
    assert snapshot.revision == OLD
    assert snapshot.pr_head_revision == HEAD
    assert snapshot.base_revision == PARENT
    assert snapshot.files == {"app.py": SOURCE}
    assert snapshot.changed_lines == {"app.py": [2]}
    assert snapshot.scope_complete
    assert all(f"ref={OLD}" in path for path, _ in github.calls if "/contents/" in path)
    assert not any(body for _, body in github.calls)


def test_default_uses_latest_commit_and_fork_source(github):
    github.detail["head"]["repo"]["full_name"] = "contributor/fork"
    snapshot = pr.collect_pr_commit(URL)
    assert snapshot.revision == HEAD
    assert snapshot.content_owner == "contributor"
    assert any(path.startswith("/repos/contributor/fork/contents/") for path, _ in github.calls)


@pytest.mark.parametrize("ref", ["d" * 40, "not-a-sha"])
def test_unrelated_or_invalid_commit_rejected(github, ref):
    with pytest.raises(pr.PRReviewError, match="commit|SHA"):
        pr.collect_pr_commit(URL, ref)
    assert not any("/contents/" in path for path, _ in github.calls)


def test_ambiguous_prefix_rejected(github):
    github.members = ["a" * 40, "a" * 39 + "b"]
    github.detail["head"]["sha"] = github.members[-1]
    with pytest.raises(pr.PRReviewError, match="uniquely"):
        pr.collect_pr_commit(URL, "aaaaaaa")


def test_pr_commit_cap_rejected(github):
    github.detail["commits"] = 251
    with pytest.raises(pr.PRReviewError, match="250"):
        pr.collect_pr_commit(URL)


def test_root_and_merge_comparisons(github):
    github.parents = []
    assert pr.collect_pr_commit(URL, OLD).base_revision == ""
    github.parents = [{"sha": PARENT}, {"sha": "d" * 40}]
    snapshot = pr.collect_pr_commit(URL, OLD)
    assert snapshot.base_revision == PARENT
    assert any(f"/compare/{PARENT}...{OLD}" in path for path, _ in github.calls)


def test_rename_changes_use_new_path(github):
    github.items[0].update(filename="new.py", previous_filename="old.py", status="renamed")
    github.sources["new.py"] = SOURCE
    snapshot = pr.collect_pr_commit(URL, OLD)
    assert snapshot.changed_lines == {"new.py": [2]}
    assert "new.py" in snapshot.files


def test_removed_and_missing_patches_mark_partial(github):
    github.items.extend([
        {"filename": "deleted.py", "status": "removed", "patch": "@@ -1 +0,0 @@\n-old"},
        {"filename": "binary.py", "status": "modified"},
    ])
    snapshot = pr.collect_pr_commit(URL, OLD)
    assert not snapshot.scope_complete
    assert any("Deleted file" in text for text in snapshot.limitations)
    assert any("Missing, binary" in text for text in snapshot.limitations)
    assert not aggregate(snapshot, [AgentResult(agent=role) for role in ("quality", "security", "tests")]).complete


def test_deleted_only_never_falls_back_to_repository_review(github):
    github.items[0]["status"] = "removed"
    with pytest.raises(pr.PRReviewError, match="no supported"):
        pr.collect_pr_commit(URL, OLD)


def test_diff_parser_counts_hunks_and_literal_plus_source():
    patch = "@@ -1 +1,2 @@\n old\n+++value\n\\ No newline at end of file\n@@ -8,0 +9 @@\n+last"
    assert added_lines(patch, 2, 0) == [2, 9]
    assert line_ranges([2, 3, 4, 9]) == [[2, 4], [9, 9]]


@pytest.mark.parametrize("patch, additions", [
    ("@@ -1 +1,2 @@\n old", 1),
    ("@@ -1 +1 @@\n old", 1),
    ("+no header", 1),
    ("", 0),
])
def test_incomplete_diff_never_expands_scope(patch, additions):
    with pytest.raises(ValueError):
        added_lines(patch, additions, 0)


def finding(evidence, line=1, end_line=3):
    return Finding(
        agent="quality", title="Candidate", severity=Severity.MEDIUM, confidence=.9,
        path="app.py", line=line, end_line=end_line, evidence=evidence,
        impact="Example impact", suggested_fix="Example fix",
    )


def test_scope_checks_evidence_not_broad_citation_range(github):
    snapshot = pr.collect_pr_commit(URL, OLD)
    unchanged = finding("before = 1")
    changed = finding("added = 2")
    assert not finding_in_scope(snapshot, unchanged)
    assert finding_in_scope(snapshot, changed)
    report = aggregate(snapshot, [AgentResult(agent="quality", findings=[unchanged, changed])])
    assert [f.evidence for f in report.findings] == ["added = 2"]
    assert [f.evidence for f in report.agents[0].findings] == ["added = 2"]


def test_model_context_centers_deep_changes_and_compresses_lines(github):
    snapshot = pr.collect_pr_commit(URL, OLD)
    snapshot.files = {"app.py": "\n".join(f"value_{i} = {i}" for i in range(1, 201))}
    snapshot.changed_lines = {"app.py": [150, 151]}
    context = Reviewer(Activity(), model="unused").context(snapshot, [])
    assert "150: value_150" in context["initial_snippets"][0]
    assert context["changed_line_ranges"] == {"app.py": [[150, 151]]}


def test_post_is_explicit_exact_body_and_historical_commit(github):
    snapshot = pr.collect_pr_commit(URL, OLD)
    assert not any(body is not None for _, body in github.calls)
    body = "Edited human-reviewed text\n"
    result = pr.post_pr_review(snapshot, body, "test-token")
    posts = [(path, value) for path, value in github.calls if value is not None]
    assert result["id"] == 19
    assert posts == [("/repos/acme/project/pulls/7/reviews",
                      {"commit_id": OLD, "event": "COMMENT", "body": body})]


@pytest.mark.parametrize("change", ["head", "closed", "membership"])
def test_stale_or_closed_pr_cannot_post(github, change):
    snapshot = pr.collect_pr_commit(URL, OLD)
    if change == "head":
        github.detail["head"]["sha"] = "d" * 40
    elif change == "closed":
        github.detail["state"] = "closed"
    else:
        github.members = [HEAD]
        github.detail["commits"] = 1
    with pytest.raises(pr.PRReviewError):
        pr.post_pr_review(snapshot, "review", "test-token")
    assert not any(body is not None for _, body in github.calls)


def test_token_required_before_post_network(github):
    snapshot = pr.collect_pr_commit(URL, OLD)
    before = len(github.calls)
    with pytest.raises(pr.PRReviewError, match="token"):
        pr.post_pr_review(snapshot, "review", "")
    assert len(github.calls) == before


def test_uncertain_post_transport_is_not_retried(monkeypatch):
    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            raise URLError("private credential must not leak")

    monkeypatch.setattr(pr, "build_opener", lambda *args: Opener())
    with pytest.raises(pr.PRReviewError, match="uncertain") as error:
        pr.GitHubPRClient("test-token").request("/repos/acme/project/pulls/7/reviews", {"body": "review"})
    assert len(calls) == 1
    assert "private credential" not in str(error.value)


def test_quote_ending_in_newline_does_not_claim_next_changed_line(github):
    snapshot = pr.collect_pr_commit(URL, OLD)
    assert not finding_in_scope(snapshot, finding("before = 1\n"))
    assert finding_in_scope(snapshot, finding("added = 2\n"))


def test_collect_paginates_pr_commits_and_commit_files(github):
    github.members = [f"{i:040x}" for i in range(1, 102)]
    github.detail["commits"] = len(github.members)
    github.detail["head"]["sha"] = github.members[-1]
    github.items = [{"filename": f"data/{i}.bin", "status": "modified"} for i in range(100)] + github.items
    snapshot = pr.collect_pr_commit(URL)
    assert snapshot.revision == github.members[-1]
    assert snapshot.files == {"app.py": SOURCE}
    assert any("/pulls/7/commits?" in path and "page=2" in path for path, _ in github.calls)
    assert any(f"/commits/{snapshot.revision}?" in path and "page=2" in path for path, _ in github.calls)


def test_symlink_dereference_or_wrong_blob_is_not_reviewed(github):
    github.sources["app.py"] = "unrelated = 1\nunrelated = 2\nunrelated = 3\n"
    with pytest.raises(pr.PRReviewError, match="no supported"):
        pr.collect_pr_commit(URL, OLD)


def test_head_change_during_collection_rejected(github, monkeypatch):
    original = pr.GitHubPRClient.request
    reads = []

    def request(client, path, body=None):
        if path.endswith("/pulls/7"):
            reads.append(path)
            if len(reads) > 1:
                github.detail["head"]["sha"] = "d" * 40
        return original(client, path, body)

    monkeypatch.setattr(pr.GitHubPRClient, "request", request)
    with pytest.raises(pr.PRReviewError, match="changed while collecting"):
        pr.collect_pr_commit(URL, OLD)
