from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Severity(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    INFORMATIONAL = "Informational"


SEVERITY_ORDER = {severity: index for index, severity in enumerate(Severity)}


def safe_path(path: str) -> bool:
    p = PurePosixPath(path)
    return bool(path) and not p.is_absolute() and "\\" not in path and ":" not in path and all(
        part not in ("", ".", "..") for part in path.split("/")
    )


class Edit(BaseModel):
    """One exact replacement in an existing snapshot file."""
    path: str
    original: str = Field(min_length=1)
    replacement: str


class Finding(BaseModel):
    id: str = ""
    agent: str = ""
    title: str = Field(min_length=1)
    severity: Severity
    confidence: float = Field(ge=0, le=1)
    path: str
    line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    evidence: str = Field(min_length=1)
    impact: str
    suggested_fix: str
    suggested_tests: list[str] = Field(default_factory=list)
    edits: list[Edit] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_location(self):
        if self.end_line < self.line or not safe_path(self.path):
            raise ValueError("Invalid finding location")
        if any(not safe_path(edit.path) for edit in self.edits):
            raise ValueError("Invalid edit path")
        return self

    def assign_id(self):
        identity = f"{self.agent}:{self.path}:{self.line}:{self.title}"
        self.id = hashlib.sha256(identity.encode()).hexdigest()[:12]
        return self


class ReviewBatch(BaseModel):
    findings: list[Finding] = Field(max_length=40)
    limitations: list[str]


class Snapshot(BaseModel):
    source: str
    revision: str
    files: dict[str, str]
    owner: str = ""
    repo: str = ""
    branch: str = ""
    pr_number: int | None = None
    content_owner: str = ""
    content_repo: str = ""
    base_revision: str = ""
    pr_head_revision: str = ""
    commit_scoped: bool = False
    scope_complete: bool = True
    changed_lines: dict[str, list[int]] = Field(default_factory=dict)
    changed_paths: list[str] = Field(default_factory=list)
    diff: str = ""
    limitations: list[str] = Field(default_factory=list)

    def source_url(self, finding: Finding) -> str:
        if not self.owner:
            return ""
        from urllib.parse import quote
        return (
            f"https://github.com/{quote(self.content_owner or self.owner)}/{quote(self.content_repo or self.repo)}/blob/"
            f"{quote(self.revision)}/{quote(finding.path, safe='/')}#L{finding.line}-L{finding.end_line}"
        )


class AgentResult(BaseModel):
    agent: str
    findings: list[Finding] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    error: str = ""


class Report(BaseModel):
    source: str
    revision: str
    findings: list[Finding]
    limitations: list[str]
    agents: list[AgentResult]
    complete: bool


class Decision(BaseModel):
    finding_id: str
    action: Literal["approve", "edit", "reject", "revise"]
    edits: list[Edit] = Field(default_factory=list)
    note: str = ""
