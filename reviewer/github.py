from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from contextlib import asynccontextmanager
from datetime import timedelta

import jsonschema
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from reviewer.models import Snapshot
from reviewer.sources import MAX_FILES, add_file, eligible, parse_github_url


def decode_result(result):
    if result.isError:
        raise RuntimeError("GitHub MCP returned an error; check access and server configuration")
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    blocks = [b.model_dump(mode="json") for b in result.content]
    texts = [b["text"] for b in blocks if b.get("type") == "text"]
    if len(texts) == 1:
        try:
            return json.loads(texts[0])
        except json.JSONDecodeError:
            pass
    return {"content": blocks}


def file_text(payload):
    """Decode GitHub MCP typed blocks, legacy base64 objects, and empty files."""
    if isinstance(payload, dict) and "content" in payload and isinstance(payload["content"], str):
        if payload.get("encoding") == "base64":
            return base64.b64decode(payload["content"]).decode("utf-8")
        return payload["content"]
    if isinstance(payload, dict) and "content" in payload and isinstance(payload["content"], list):
        resources = [b["resource"] for b in payload["content"] if b.get("type") == "resource"]
        for resource in resources:
            if "text" in resource:
                return resource["text"]
            if "blob" in resource:
                return base64.b64decode(resource["blob"]).decode("utf-8")
        texts = [b["text"] for b in payload["content"] if b.get("type") == "text"]
        # GitHub sends a status text block followed by the actual text block.
        if len(texts) >= 2:
            return texts[-1]
    raise ValueError("MCP did not return inline file content; resource-link-only files are skipped")


class GitHub:
    def __init__(self, session, tools, activity, writable=False):
        self.session, self.tools, self.activity, self.writable = session, tools, activity, writable

    async def call(self, name, **arguments):
        allowed = {"get_commit", "get_file_contents", "pull_request_read"}
        if self.writable:
            allowed |= {"create_branch", "push_files", "create_pull_request"}
        if name not in allowed:
            raise ValueError("Tool is outside this connection's allowlist")
        if name not in self.tools:
            raise ValueError(f"GitHub MCP server does not expose {name}")
        jsonschema.validate(arguments, self.tools[name].inputSchema)
        self.activity.add("orchestrator", f"github.{name}", "running")
        import time
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(self.session.call_tool(name, arguments), timeout=90)
            value = decode_result(result)
        except Exception:
            self.activity.add("orchestrator", f"github.{name}", "failed", time.monotonic() - started)
            raise
        self.activity.add("orchestrator", f"github.{name}", "completed", time.monotonic() - started)
        return value

    async def commit(self, owner, repo, ref):
        value = await self.call("get_commit", owner=owner, repo=repo, sha=ref)
        sha = value.get("sha", "")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
            raise ValueError("MCP did not return a full immutable commit SHA")
        return sha

    async def snapshot(self, url, ref="main"):
        owner, repo, pr = parse_github_url(url)
        params = dict(owner=owner, repo=repo)
        limitations, changed, diff = [], [], ""
        if pr:
            detail = await self.call("pull_request_read", method="get", pullNumber=pr, **params)
            head = detail["head"]
            branch = head["ref"]
            if head.get("repo", {}).get("full_name", f"{owner}/{repo}") != f"{owner}/{repo}":
                limitations.append("Fork PR: patch download supported; draft PR publishing is disabled.")
            sha = head["sha"]
            for page in range(1, 11):
                items = await self.call(
                    "pull_request_read", method="get_files", pullNumber=pr, page=page, perPage=100, **params
                )
                if isinstance(items, dict):
                    items = items.get("files", [])
                for item in items:
                    if item.get("status") != "removed":
                        changed.append(item["filename"])
                    if item.get("patch"):
                        diff += f"\n--- {item['filename']}\n{item['patch']}\n"
                if len(items) < 100:
                    break
            else:
                limitations.append("PR file listing capped at 1,000 entries.")
        else:
            branch = ref
            sha = await self.commit(owner, repo, ref)
        if not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
            raise ValueError("Invalid commit SHA")
        files, candidates = {}, list(changed)
        directories = [""]
        entries, directories_seen = 0, 0
        # Include related tests/config; changed files are fetched before discovered files.
        ignore_patterns = []
        try:
            ignore_content = await self.call("get_file_contents", path=".gitignore", ref=sha, **params)
            ignore_patterns = file_text(ignore_content).splitlines()
        except Exception:
            limitations.append("Root .gitignore unavailable; built-in exclusions used.")
        import pathspec
        spec = pathspec.PathSpec.from_lines("gitwildmatch", ignore_patterns)
        while directories and directories_seen < 100:
            directory = directories.pop(0)
            directories_seen += 1
            listing = await self.call("get_file_contents", path=directory, ref=sha, **params)
            if isinstance(listing, dict):
                listing = listing.get("entries", listing.get("files", []))
            if not isinstance(listing, list):
                raise ValueError("MCP directory response was not an array")
            for item in listing:
                entries += 1
                path = item["path"]
                kind = item.get("type")
                if spec.match_file(path + ("/" if kind == "dir" else "")):
                    continue
                if kind == "dir":
                    from reviewer.sources import IGNORED_DIRS
                    if path.rsplit("/", 1)[-1] not in IGNORED_DIRS:
                        directories.append(path)
                elif kind == "file" and eligible(path):
                    candidates.append(path)
            if entries >= 2000:
                limitations.append("Repository discovery stopped at 2,000 entries.")
                break
        if directories:
            limitations.append("Repository directory listing was truncated.")
        unique = list(dict.fromkeys(candidates))
        supported = [p for p in unique if eligible(p) and not spec.match_file(p)]
        for path in supported[:MAX_FILES]:
            try:
                payload = await self.call("get_file_contents", path=path, ref=sha, **params)
                add_file(files, path, file_text(payload), limitations)
            except Exception:
                limitations.append(f"Could not retrieve inline text: {path}")
        if len(supported) > MAX_FILES:
            limitations.append(f"File budget: omitted {len(supported) - MAX_FILES} supported files.")
        if not files:
            raise ValueError("No supported inline source files retrieved")
        if pr:
            latest = await self.call("pull_request_read", method="get", pullNumber=pr, **params)
            if latest["head"]["sha"] != sha:
                raise ValueError("PR changed while collecting context; retry the review")
        limitations.append("Nested .gitignore files are not interpreted for GitHub inputs.")
        if len(diff) > 40_000:
            limitations.append("PR diff truncated to 40,000 characters.")
        return Snapshot(
            source=url, revision=sha, owner=owner, repo=repo, branch=branch, pr_number=pr,
            files=files, changed_paths=changed, diff=diff[:40_000], limitations=limitations,
        )


@asynccontextmanager
async def connect(activity, writable=False):
    command = os.getenv("GITHUB_MCP_COMMAND", "").strip()
    if command:
        args = json.loads(os.getenv("GITHUB_MCP_ARGS", "[]"))
        params = StdioServerParameters(command=command, args=args, env=dict(os.environ))
        transport = stdio_client(params)
    else:
        token = os.getenv("GITHUB_TOKEN", "")
        if not token:
            raise ValueError("Set GITHUB_TOKEN or configure a local GitHub MCP server")
        transport = streamablehttp_client(
            os.getenv("GITHUB_MCP_URL", "https://api.githubcopilot.com/mcp/"),
            headers={"Authorization": f"Bearer {token}", "X-MCP-Readonly": "false" if writable else "true"},
            timeout=timedelta(seconds=30),
        )
    async with transport as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            tools = {}
            cursor = None
            while True:
                page = await session.list_tools(cursor=cursor)
                tools.update({tool.name: tool for tool in page.tools})
                cursor = page.nextCursor
                if not cursor:
                    break
            yield GitHub(session, tools, activity, writable)
