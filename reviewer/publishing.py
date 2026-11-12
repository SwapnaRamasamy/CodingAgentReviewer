from __future__ import annotations

import uuid

from reviewer.scanners import validate_python


async def publish_draft(github, snapshot, updated_files, title, body, branch=None):
    """Only called by the explicit UI publish action. Never invoked by review nodes."""
    if not github.writable:
        raise ValueError("A write-enabled MCP connection is required")
    if not snapshot.owner or not snapshot.branch or len(snapshot.revision) != 40:
        raise ValueError("Publishing requires a GitHub branch snapshot")
    if any("Fork PR:" in s for s in snapshot.limitations):
        raise ValueError("Fork PR publishing is unsupported; download the patch")
    required = {"get_commit", "create_branch", "push_files", "create_pull_request"}
    missing = required - github.tools.keys()
    if missing:
        raise ValueError("Missing GitHub MCP capabilities: " + ", ".join(sorted(missing)))
    changes = {p: text for p, text in updated_files.items() if snapshot.files.get(p) != text}
    if not changes or any(p not in snapshot.files for p in changes):
        raise ValueError("No valid approved existing-file changes to publish")
    syntax_errors = validate_python(changes)
    if syntax_errors:
        raise ValueError("Changed Python files have syntax errors")
    params = {"owner": snapshot.owner, "repo": snapshot.repo}
    current = await github.commit(**params, ref=snapshot.branch)
    if current != snapshot.revision:
        raise ValueError("Source branch changed since review. Run a new review before publishing.")
    branch = branch or "agent-review/" + uuid.uuid4().hex[:12]
    # Caller retains branch on failure so partially completed actions are inspectable.
    await github.call("create_branch", **params, branch=branch, from_branch=snapshot.branch)
    branch_sha = await github.commit(**params, ref=branch)
    if branch_sha != snapshot.revision:
        raise ValueError(f"Source moved during branch creation. Inspect {branch}; no files were pushed.")
    await github.call(
        "push_files", **params, branch=branch,
        files=[{"path": p, "content": text} for p, text in sorted(changes.items())],
        message="Apply human-approved code review fixes",
    )
    return await github.call(
        "create_pull_request", **params, title=title, body=body, head=branch, base=snapshot.branch, draft=True,
    )
