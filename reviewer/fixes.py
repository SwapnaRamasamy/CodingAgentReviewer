from __future__ import annotations

import difflib

from reviewer.models import Edit, safe_path
from reviewer.scanners import validate_python


def apply_edits(files, edits):
    """Apply approved edits atomically in memory; reject ambiguity and overlap."""
    updated = dict(files)
    seen = set()
    for edit in edits:
        key = (edit.path, edit.original, edit.replacement)
        if key in seen:
            continue
        seen.add(key)
        if not safe_path(edit.path) or edit.path not in files:
            raise ValueError(f"Edit targets an unavailable path: {edit.path}")
        if updated[edit.path].count(edit.original) != 1:
            raise ValueError(f"Edit in {edit.path} is ambiguous, overlapping, or stale")
        updated[edit.path] = updated[edit.path].replace(edit.original, edit.replacement, 1)
    return updated


def validate_edits(files, edits):
    updated = apply_edits(files, edits)
    errors = validate_python({p: value for p, value in updated.items() if value != files[p]})
    if errors:
        raise ValueError("Python syntax validation failed: " + "; ".join(errors))
    return updated


def patch_text(original, updated):
    chunks = []
    for path in sorted(updated):
        if updated[path] != original[path]:
            # keepends preserves code and produces standard unified diffs.
            before, after = original[path], updated[path]
            lines = list(difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile=f"a/{path}", tofile=f"b/{path}",
            ))
            for line in lines:
                chunks.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    return "".join(chunks)


def edits_from_replacement(path, original, replacement):
    if original == replacement:
        return []
    if not original:
        raise ValueError("Editing an empty file is not supported by exact-replacement patches")
    return [Edit(path=path, original=original, replacement=replacement)]
