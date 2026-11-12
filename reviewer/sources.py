from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pathspec

from reviewer.models import Snapshot, safe_path

EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".rb", ".php",
    ".cs", ".c", ".cpp", ".h", ".sql", ".json", ".toml", ".yaml", ".yml",
    ".md", ".txt", ".ini", ".cfg", ".xml", ".sh", ".html", ".css",
}
IGNORED_DIRS = {".git", ".venv", "venv", "node_modules", "dist", "build", "__pycache__", ".pytest_cache"}
MAX_FILES = 60
MAX_FILE_BYTES = 60_000
MAX_TOTAL_BYTES = 240_000


def eligible(path):
    p = Path(path)
    lower = p.name.lower()
    return (
        safe_path(path)
        and not set(p.parts).intersection(IGNORED_DIRS)
        and not lower.startswith((".env", ".secret", "credentials", "id_rsa", "id_ed25519"))
        and lower not in {"secrets.toml", "package-lock.json", "yarn.lock", "uv.lock", "poetry.lock"}
        and p.suffix.lower() in EXTENSIONS
    )


def add_file(files, path, content, limitations):
    size = len(content.encode("utf-8"))
    if "\x00" in content or size > MAX_FILE_BYTES:
        limitations.append(f"Skipped binary or oversized file: {path}")
        return
    if len(files) >= MAX_FILES or sum(len(v.encode("utf-8")) for v in files.values()) + size > MAX_TOTAL_BYTES:
        limitations.append(f"Context budget exceeded; skipped {path}")
        return
    files[path] = content


def local_snapshot(directory):
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Choose a repository directory")
    files, limitations = {}, []
    visited = 0

    def walk(folder, inherited):
        nonlocal visited
        lines = list(inherited)
        ignore = folder / ".gitignore"
        if ignore.is_file() and not ignore.is_symlink():
            prefix = folder.relative_to(root).as_posix()
            for raw in ignore.read_text(encoding="utf-8", errors="replace").splitlines():
                if prefix == ".":
                    lines.append(raw)
                elif raw and not raw.startswith("#"):
                    negation = "!" if raw.startswith("!") else ""
                    pattern = raw.lstrip("!").lstrip("/")
                    lines.append(f"{negation}{prefix}/{pattern}")
        spec = pathspec.PathSpec.from_lines("gitwildmatch", lines)
        for child in sorted(folder.iterdir()):
            visited += 1
            if visited > 5000:
                return
            if child.is_symlink():
                continue
            relative = child.relative_to(root).as_posix()
            if spec.match_file(relative + ("/" if child.is_dir() else "")):
                continue
            if child.is_dir():
                if child.name not in IGNORED_DIRS:
                    walk(child, lines)
            elif eligible(relative):
                if child.stat().st_size > MAX_FILE_BYTES:
                    limitations.append(f"Skipped oversized file: {relative}")
                else:
                    try:
                        add_file(files, relative, child.read_text(encoding="utf-8"), limitations)
                    except (UnicodeError, OSError):
                        limitations.append(f"Could not read text file: {relative}")
    walk(root, [])
    if visited > 5000:
        limitations.append("Directory traversal stopped at 5,000 entries")
    if not files:
        raise ValueError("No supported source files found")
    digest = hashlib.sha256()
    for path, content in sorted(files.items()):
        digest.update(path.encode() + b"\0" + content.encode() + b"\0")
    return Snapshot(
        source=str(root), revision=digest.hexdigest(), files=files,
        limitations=limitations + ["Local snapshot revision hashes only the included files."],
    )


def parse_github_url(url):
    match = re.fullmatch(
        r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:/pull/(\d+))?/?", url.strip()
    )
    if not match:
        raise ValueError("Use https://github.com/owner/repo or https://github.com/owner/repo/pull/123")
    owner, repo, pr = match.groups()
    return owner, repo.removesuffix(".git"), int(pr) if pr else None


def demo_snapshot():
    return Snapshot(source="Built-in demo", revision="demo-v1", files={
        "app.py": 'import subprocess\n\ndef lookup(user_input, db):\n    return db.execute("SELECT * FROM users WHERE name = \'" + user_input + "\'")\n\ndef run(command):\n    return subprocess.check_output(command, shell=True)\n\ndef average(values):\n    return sum(values) / len(values)\n',
        "tests/test_app.py": "from app import average\n\ndef test_average():\n    assert average([2, 4]) == 3\n",
    }, limitations=["Demo uses fixed example findings; no model or GitHub call is made."])
