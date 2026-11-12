"""Collect a PR commit and explicitly publish a summary review via GitHub REST."""
from __future__ import annotations

import base64
import hashlib
import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

import pathspec

from reviewer.commit_scope import MAX_DIFF_CHARS, added_lines
from reviewer.models import Snapshot, safe_path
from reviewer.public_github import PublicRepositoryError, github_ssl_context
from reviewer.sources import MAX_FILE_BYTES, MAX_FILES, MAX_TOTAL_BYTES, add_file, eligible, parse_github_url

SHA = re.compile(r"[0-9a-fA-F]{40}")
MAX_REVIEW_CHARS = 60_000


class PRReviewError(ValueError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Credentials must never be forwarded to a redirect destination.
        return None


class GitHubPRClient:
    def __init__(self, token=""):
        self.token = token.strip()

    def request(self, path, body=None):
        if not path.startswith("/repos/") or "://" in path or "\r" in path or "\n" in path:
            raise PRReviewError("Invalid GitHub API path.")
        headers = {
            "User-Agent": "CodeReviewStudio",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request("https://api.github.com" + path, data=data, headers=headers)
        try:
            opener = build_opener(NoRedirect(), HTTPSHandler(context=github_ssl_context()))
            # A POST is attempted once. A timeout must not automatically publish twice.
            with opener.open(request, timeout=60) as response:
                raw = response.read(8_000_001)
            if len(raw) > 8_000_000:
                raise PRReviewError("GitHub response exceeded the supported size; review a smaller commit.")
            return json.loads(raw)
        except HTTPError as error:
            messages = {
                301: "The repository moved. Use its current GitHub PR URL.",
                302: "GitHub redirected this request. Use the current repository PR URL.",
                401: "GitHub authentication failed. Check the GitHub token.",
                403: "GitHub denied access or its request limit was reached. Check token permissions and rate limits.",
                404: "PR, commit, or file not found. Private repositories require a GitHub token with read access.",
                422: "GitHub rejected the review or commit selection. Check the PR and selected commit.",
                429: "GitHub request limit reached. Try again later.",
            }
            raise PRReviewError(messages.get(error.code, f"GitHub returned HTTP {error.code}.")) from None
        except PublicRepositoryError as error:
            raise PRReviewError(str(error)) from None
        except (URLError, OSError, ValueError) as error:
            if isinstance(error, PRReviewError):
                raise
            if body is not None:
                raise PRReviewError(
                    "Posting outcome is uncertain. Inspect the PR on GitHub before attempting another post."
                ) from None
            raise PRReviewError(
                "Could not read GitHub. Check your connection and system-trusted HTTPS certificates."
            ) from None

    def file(self, owner, repo, path, sha, expected_sha=None):
        payload = self.request(
            f"/repos/{owner}/{repo}/contents/{quote(path, safe='/')}?ref={sha}"
        )
        if not isinstance(payload, dict) or payload.get("type") != "file":
            raise PRReviewError("GitHub did not return an inline source file.")
        if payload.get("size", 0) > MAX_FILE_BYTES or payload.get("encoding") != "base64":
            raise PRReviewError("Source file is too large or is not available inline.")
        try:
            raw = base64.b64decode(payload["content"], validate=False)
            if len(raw) > MAX_FILE_BYTES:
                raise PRReviewError("Source file exceeds the file budget.")
            if expected_sha is not None:
                actual = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
                if not SHA.fullmatch(expected_sha) or actual != expected_sha.lower():
                    raise PRReviewError("Source content does not match the commit's Git blob.")
            return raw.decode("utf-8")
        except (KeyError, ValueError, UnicodeError):
            raise PRReviewError("Source file is not valid UTF-8 text.") from None


def pr_identity(url):
    value = url.strip()
    formatted = re.fullmatch(r"\[[^\]\r\n]*\]\((https://github\.com/[^\s()]+)\)", value)
    if formatted:
        value = formatted.group(1)
    try:
        owner, repo, number = parse_github_url(value)
    except ValueError:
        raise PRReviewError("Enter a PR URL like https://github.com/owner/repo/pull/123.") from None
    if not number or owner in {".", ".."} or repo in {"", ".", ".."}:
        raise PRReviewError("Enter a PR URL like https://github.com/owner/repo/pull/123.")
    return owner, repo, number


def pr_details(client, owner, repo, number):
    detail = client.request(f"/repos/{owner}/{repo}/pulls/{number}")
    if not SHA.fullmatch(detail.get("head", {}).get("sha", "")):
        raise PRReviewError("GitHub did not return a valid PR head SHA.")
    if detail.get("commits", 0) > 250:
        raise PRReviewError("This PR exceeds GitHub's 250-commit listing limit. Use a smaller PR.")
    return detail


def pr_commits(client, owner, repo, number, detail):
    commits = []
    for page in range(1, 4):
        batch = client.request(f"/repos/{owner}/{repo}/pulls/{number}/commits?per_page=100&page={page}")
        if not isinstance(batch, list):
            raise PRReviewError("GitHub did not return a commit list.")
        commits.extend(item.get("sha", "") for item in batch)
        if len(batch) < 100:
            break
    if (
        not commits or len(commits) > 250
        or any(not SHA.fullmatch(sha) for sha in commits)
        or len(set(commits)) != len(commits)
        or ("commits" in detail and len(commits) != detail["commits"])
        or detail["head"]["sha"] not in commits
    ):
        raise PRReviewError("PR commit listing is incomplete or changed. Reload the PR and try again.")
    return commits


def collect_pr_commit(url, ref="", token=""):
    owner, repo, number = pr_identity(url)
    client = GitHubPRClient(token)
    detail = pr_details(client, owner, repo, number)
    members = pr_commits(client, owner, repo, number, detail)
    requested = ref.strip().lower()
    if requested and not re.fullmatch(r"[0-9a-f]{7,40}", requested):
        raise PRReviewError("Enter a commit SHA of 7 to 40 hexadecimal characters, or leave it blank for the latest.")
    matches = [sha for sha in members if sha.lower().startswith(requested)] if requested else [detail["head"]["sha"]]
    if len(matches) != 1:
        raise PRReviewError("The commit must uniquely identify a commit in this PR.")
    sha = matches[0]
    endpoint = f"/repos/{owner}/{repo}"
    commit = client.request(f"{endpoint}/commits/{sha}?per_page=100&page=1")
    if commit.get("sha") != sha:
        raise PRReviewError("GitHub returned a different commit. Reload the PR.")
    parents = [parent.get("sha", "") for parent in commit.get("parents", [])]
    if any(not SHA.fullmatch(parent) for parent in parents):
        raise PRReviewError("GitHub returned an invalid parent SHA.")
    parent = parents[0] if parents else ""
    limitations = [
        "Only this commit's added/modified text lines can anchor findings; surrounding code is context.",
        "Only changed source files are included. Unchanged tests and callers may be unavailable.",
        "Nested .gitignore rules are not interpreted.",
    ]
    complete = True
    items = list(commit.get("files", []))
    if len(parents) > 1:
        comparison = client.request(f"{endpoint}/compare/{parent}...{sha}?per_page=1&page=1")
        items = comparison.get("files", [])
        limitations.append("Merge commit reviewed against its first parent.")
        if len(items) >= 300:
            complete = False
            limitations.append("Merge comparison reached GitHub's 300-file cap; coverage may be incomplete.")
    elif len(items) == 100:
        for page in range(2, 31):
            batch = client.request(f"{endpoint}/commits/{sha}?per_page=100&page={page}")
            if batch.get("sha") != sha:
                raise PRReviewError("Commit changed during file collection.")
            page_files = batch.get("files", [])
            items.extend(page_files)
            if len(page_files) < 100:
                break
        else:
            complete = False
            limitations.append("Commit listing reached GitHub's 3,000-file cap; coverage may be incomplete.")
    if not parent:
        limitations.append("Root commit reviewed against the empty tree.")

    content_owner, content_repo = owner, repo
    head_repo = detail.get("head", {}).get("repo") or {}
    full_name = head_repo.get("full_name", f"{owner}/{repo}")
    if full_name != f"{owner}/{repo}":
        try:
            content_owner, content_repo, _ = parse_github_url("https://github.com/" + full_name)
        except ValueError:
            raise PRReviewError("GitHub returned an invalid fork repository.") from None
        if content_owner in {".", ".."} or content_repo in {"", ".", ".."}:
            raise PRReviewError("GitHub returned an invalid fork repository.")

    try:
        ignores = client.file(content_owner, content_repo, ".gitignore", sha).splitlines()
    except PRReviewError:
        ignores = []
        limitations.append("Root .gitignore unavailable; built-in exclusions used.")
    spec = pathspec.PathSpec.from_lines("gitwildmatch", ignores)
    files, changed_lines, changed_paths, patches = {}, {}, [], []
    for item in items:
        path = item.get("filename", "")
        if not safe_path(path) or not eligible(path) or spec.match_file(path):
            complete = False
            limitations.append(f"Excluded unsupported or ignored change: {path}")
            continue
        changed_paths.append(path)
        if item.get("status") == "removed":
            complete = False
            limitations.append(f"Deleted file has no head-line evidence and is not reviewed: {path}")
            continue
        patch = item.get("patch") or ""
        try:
            lines = added_lines(patch, item.get("additions"), item.get("deletions"))
        except ValueError:
            complete = False
            limitations.append(f"Missing, binary, or truncated patch; skipped: {path}")
            continue
        if not lines:
            complete = False
            limitations.append(f"Deletion-only or rename-only change has no added-line anchor: {path}")
            continue
        if len(files) >= MAX_FILES or sum(len(text.encode("utf-8")) for text in files.values()) >= MAX_TOTAL_BYTES:
            complete = False
            limitations.append(f"Source budget exceeded; skipped: {path}")
            continue
        try:
            text = client.file(content_owner, content_repo, path, sha, expected_sha=item.get("sha", ""))
        except PRReviewError:
            complete = False
            limitations.append(f"Could not retrieve or verify source at the selected commit: {path}")
            continue
        if max(lines) > len(text.splitlines()):
            complete = False
            limitations.append(f"Patch line numbers do not match the source; skipped: {path}")
            continue
        add_file(files, path, text, limitations)
        if path not in files:
            complete = False
            continue
        changed_lines[path] = lines
        patches.append(f"\nFILE {path}\nSTATUS {item.get('status', 'modified')}\n{patch}\n")
    if not files:
        raise PRReviewError(
            "This commit has no supported added/modified text lines available for review. "
            "Deletion-only, binary, ignored, oversized, or unavailable changes cannot be reviewed."
        )
    latest = pr_details(client, owner, repo, number)
    if latest["head"]["sha"] != detail["head"]["sha"]:
        raise PRReviewError("PR changed while collecting source. Reload and review again.")
    diff = "".join(patches)
    if len(diff) > MAX_DIFF_CHARS:
        complete = False
        limitations.append("Diff context truncated to 40,000 characters.")
    return Snapshot(
        source=f"https://github.com/{owner}/{repo}/pull/{number}",
        owner=owner, repo=repo, pr_number=number, revision=sha,
        content_owner=content_owner, content_repo=content_repo,
        branch=detail["head"].get("ref", ""), base_revision=parent,
        pr_head_revision=detail["head"]["sha"], commit_scoped=True,
        scope_complete=complete, files=files, changed_paths=changed_paths,
        changed_lines=changed_lines, diff=diff[:MAX_DIFF_CHARS], limitations=limitations,
    )


def post_pr_review(snapshot, body, token):
    """Explicit UI action only; never called by a model or review workflow."""
    if not token.strip():
        raise PRReviewError("Enter a GitHub token with Pull requests: write permission to post.")
    if not snapshot.commit_scoped or not snapshot.pr_number or not SHA.fullmatch(snapshot.revision):
        raise PRReviewError("Run a PR commit review before posting.")
    if not body.strip() or len(body) > MAX_REVIEW_CHARS:
        raise PRReviewError(f"Review text must contain 1 to {MAX_REVIEW_CHARS:,} characters.")
    identity = pr_identity(snapshot.source)
    if identity != (snapshot.owner, snapshot.repo, snapshot.pr_number):
        raise PRReviewError("Review target does not match the collected PR.")
    client = GitHubPRClient(token)
    detail = pr_details(client, *identity)
    if detail.get("state") != "open":
        raise PRReviewError("The PR is no longer open. Review its current state before posting.")
    if detail["head"]["sha"] != snapshot.pr_head_revision:
        raise PRReviewError("PR changed since review. Run a new review before posting.")
    if snapshot.revision not in pr_commits(client, *identity, detail):
        raise PRReviewError("The reviewed commit is no longer in the PR. Run a new review.")
    result = client.request(
        f"/repos/{snapshot.owner}/{snapshot.repo}/pulls/{snapshot.pr_number}/reviews",
        body={"commit_id": snapshot.revision, "event": "COMMENT", "body": body},
    )
    if not isinstance(result, dict) or not result.get("id"):
        raise PRReviewError("Posting outcome is uncertain. Inspect the PR on GitHub before posting again.")
    return result
