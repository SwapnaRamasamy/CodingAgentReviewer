from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from reviewer.models import Finding, Severity, safe_path


def isolated_env():
    # Scanners need PATH and platform variables, never model/GitHub credentials.
    keep = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE", "LOCALAPPDATA"}
    return {k: v for k, v in os.environ.items() if k.upper() in keep}


def run_scanner(module, args, cwd):
    result = subprocess.run(
        [sys.executable, "-I", "-m", module, *args],
        cwd=cwd, env=isolated_env(), text=True, capture_output=True, timeout=90,
        encoding="utf-8", errors="replace", shell=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"{module} did not complete successfully")
    return result.stdout


def scan(snapshot, activity, agent):
    findings, limitations = [], []
    with tempfile.TemporaryDirectory(prefix="code-review-") as temp:
        root = Path(temp)
        for path, content in snapshot.files.items():
            if not safe_path(path):
                raise ValueError("Unsafe snapshot path")
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        if agent == "security":
            if any(p.endswith(".py") for p in snapshot.files):
                try:
                    output = activity.run(agent, "bandit", run_scanner, "bandit", ["-r", ".", "-f", "json", "-q"], temp)
                    payload = json.loads(output)
                    for item in payload.get("results", []):
                        p = Path(item["filename"])
                        path = p.resolve().relative_to(root.resolve()).as_posix() if p.is_absolute() else p.as_posix().removeprefix("./")
                        line = item["line_number"]
                        severity = {"HIGH": Severity.HIGH, "MEDIUM": Severity.MEDIUM}.get(item["issue_severity"], Severity.LOW)
                        confidence = {"HIGH": .95, "MEDIUM": .75, "LOW": .5}.get(item["issue_confidence"], .5)
                        if path in snapshot.files and 1 <= line <= len(snapshot.files[path].splitlines()):
                            findings.append(Finding(
                                agent=agent, title=f'{item["test_id"]}: {item["issue_text"]}',
                                severity=severity, confidence=confidence, path=path, line=line, end_line=line,
                                evidence=snapshot.files[path].splitlines()[line-1],
                                impact="Static security candidate; exploitability needs code and deployment context.",
                                suggested_fix="Inspect the flagged operation and use a safe alternative appropriate to its inputs.",
                                sources=["Bandit"],
                            ))
                    if payload.get("errors"):
                        limitations.append("Bandit could not analyze some files.")
                except Exception:
                    limitations.append("Bandit failed or timed out; static security scan incomplete.")
            requirements = snapshot.files.get("requirements.txt")
            if requirements:
                pins = [s.strip() for s in requirements.splitlines() if s.strip() and not s.lstrip().startswith("#")]
                if pins and all(re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+", pin) for pin in pins):
                    try:
                        output = activity.run(
                            agent, "pip-audit", run_scanner, "pip_audit",
                            ["-r", "requirements.txt", "--no-deps", "--disable-pip", "--no-progress-spinner", "-f", "json"], temp,
                        )
                        payload = json.loads(output)
                        for dependency in payload.get("dependencies", []):
                            if dependency.get("skip_reason"):
                                limitations.append(f'Dependency audit skipped {dependency["name"]}.')
                            for vuln in dependency.get("vulns", []):
                                needle = f'{dependency["name"]}=={dependency["version"]}'
                                lines = requirements.splitlines()
                                line = next((i for i, text in enumerate(lines, 1) if text.strip().lower() == needle.lower()), 1)
                                findings.append(Finding(
                                    agent=agent, title=f'{vuln["id"]} in {dependency["name"]}',
                                    severity=Severity.MEDIUM, confidence=.99, path="requirements.txt",
                                    line=line, end_line=line, evidence=lines[line-1],
                                    impact=vuln.get("description", "Known vulnerable dependency version."),
                                    suggested_fix="Review compatibility and upgrade to a fixed version: " + ", ".join(vuln.get("fix_versions", [])),
                                    sources=["pip-audit", vuln["id"]],
                                ))
                    except Exception:
                        limitations.append("Dependency audit failed or timed out; vulnerability data unavailable.")
                else:
                    limitations.append("Dependency audit requires plain name==version pins; includes, URLs and options are refused.")
            else:
                limitations.append("No requirements.txt: dependency vulnerabilities were not audited.")
        elif agent == "quality":
            try:
                output = activity.run(
                    agent, "ruff", run_scanner, "ruff",
                    ["check", ".", "--isolated", "--select", "F", "--output-format", "json", "--no-cache"], temp,
                )
                for item in json.loads(output):
                    path = Path(item["filename"]).resolve().relative_to(root.resolve()).as_posix()
                    line = item["location"]["row"]
                    if path in snapshot.files and 1 <= line <= len(snapshot.files[path].splitlines()):
                        findings.append(Finding(
                            agent=agent, title=f'{item["code"]}: {item["message"]}',
                            severity=Severity.LOW, confidence=.95, path=path, line=line, end_line=line,
                            evidence=snapshot.files[path].splitlines()[line-1],
                            impact="Potential Python correctness or maintainability issue.",
                            suggested_fix="Resolve the diagnostic after checking its surrounding code.",
                            sources=["Ruff"],
                        ))
            except Exception:
                limitations.append("Ruff failed or timed out; Python lint check incomplete.")
    return findings, limitations


def validate_python(files):
    errors = []
    for path, content in files.items():
        if path.endswith(".py"):
            try:
                ast.parse(content, filename=path)
            except SyntaxError as error:
                errors.append(f"{path}:{error.lineno}: {error.msg}")
    return errors
