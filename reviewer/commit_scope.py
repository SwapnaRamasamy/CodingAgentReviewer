"""Deterministic scope checks for single-commit reviews."""
import re

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
MAX_DIFF_CHARS = 40_000


def added_lines(patch, additions=None, deletions=None):
    """Validate complete unified hunks and return their added head line numbers."""
    result, added, removed = [], 0, 0
    remaining_old = remaining_new = 0
    current = None
    for line in patch.splitlines():
        match = HUNK.match(line)
        if match:
            if remaining_old or remaining_new:
                raise ValueError("Incomplete diff hunk")
            _old, old_count, new, new_count = match.groups()
            remaining_old = int(old_count) if old_count is not None else 1
            remaining_new = int(new_count) if new_count is not None else 1
            current = int(new)
        elif line.startswith("\\ No newline at end of file"):
            continue
        elif current is None:
            raise ValueError("Missing diff hunk")
        elif line.startswith("+"):
            if remaining_new <= 0 or current < 1:
                raise ValueError("Invalid added line")
            result.append(current)
            current += 1
            remaining_new -= 1
            added += 1
        elif line.startswith("-"):
            remaining_old -= 1
            removed += 1
        elif line.startswith(" "):
            remaining_old -= 1
            remaining_new -= 1
            current += 1
        else:
            raise ValueError("Invalid diff line")
        if remaining_old < 0 or remaining_new < 0:
            raise ValueError("Invalid diff counts")
    if current is None or remaining_old or remaining_new:
        raise ValueError("Missing or truncated patch")
    if additions is not None and added != additions:
        raise ValueError("Missing added lines")
    if deletions is not None and removed != deletions:
        raise ValueError("Missing removed lines")
    return result


def finding_in_scope(snapshot, finding):
    if not snapshot.commit_scoped:
        return True
    changed = set(snapshot.changed_lines.get(finding.path, []))
    if not changed or not finding.evidence.strip():
        return False
    lines = snapshot.files.get(finding.path, "").splitlines()
    if not 1 <= finding.line <= finding.end_line <= len(lines):
        return False
    excerpt = "\n".join(lines[finding.line - 1:finding.end_line])
    # Check the quote itself, not a potentially overbroad declared line range.
    start = excerpt.find(finding.evidence)
    while start >= 0:
        first = finding.line + excerpt[:start].count("\n")
        last = first + finding.evidence.rstrip("\n").count("\n")
        if changed.intersection(range(first, last + 1)):
            return True
        start = excerpt.find(finding.evidence, start + 1)
    return False


def line_ranges(lines):
    """Compress changed lines for bounded model context."""
    ranges = []
    for line in sorted(set(lines)):
        if ranges and line == ranges[-1][1] + 1:
            ranges[-1][1] = line
        else:
            ranges.append([line, line])
    return ranges
