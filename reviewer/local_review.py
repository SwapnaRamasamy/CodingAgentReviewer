"""Small, validated responses for local PR review models."""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from reviewer.commit_scope import finding_in_scope
from reviewer.diagnostics import InvalidReviewEvidence
from reviewer.models import Finding, ReviewBatch, Severity, Snapshot

ShortTest = Annotated[str, Field(min_length=1, max_length=160)]
ShortLimitation = Annotated[str, Field(min_length=1, max_length=200)]


class LocalFinding(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    severity: Severity
    confidence: float = Field(ge=0, le=1)
    path: str = Field(min_length=1)
    line: int = Field(ge=1)
    evidence: str = Field(min_length=1, max_length=360)
    impact: str = Field(min_length=1, max_length=320)
    suggested_fix: str = Field(min_length=1, max_length=320)
    suggested_tests: list[ShortTest] = Field(max_length=2)


class LocalReviewBatch(BaseModel):
    findings: list[LocalFinding] = Field(max_length=1)
    limitations: list[ShortLimitation] = Field(max_length=4)

    def validated_review(self, snapshot: Snapshot) -> ReviewBatch:
        findings = []
        for item in self.findings:
            finding = Finding(
                **item.model_dump(), end_line=item.line + item.evidence.count("\n"),
            )
            lines = snapshot.files.get(finding.path, "").splitlines()
            if (
                not 1 <= finding.line <= finding.end_line <= len(lines)
                or not finding.evidence.strip()
                or finding.evidence not in "\n".join(lines[finding.line - 1:finding.end_line])
                or not finding_in_scope(snapshot, finding)
            ):
                raise InvalidReviewEvidence("Finding evidence must match the selected commit's source.")
            findings.append(finding)
        return ReviewBatch(
            findings=findings,
            limitations=[*self.limitations, (
                "Local model output is limited to one prioritized finding per specialist; "
                "additional issues may be unreported. Inline patch edits are omitted."
            )],
        )
