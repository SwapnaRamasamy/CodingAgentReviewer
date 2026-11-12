from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from reviewer.commit_scope import finding_in_scope
from reviewer.fixes import patch_text, validate_edits
from reviewer.models import SEVERITY_ORDER, AgentResult, Decision, Report, Snapshot


class State(TypedDict, total=False):
    snapshot: Snapshot
    results: Annotated[list[AgentResult], operator.add]
    report: Report
    decisions: list[dict]
    patch: str
    updated_files: dict[str, str]
    validation: list[str]
    orchestration_error: str


def aggregate(snapshot, results, expected_roles=("quality", "security", "tests")):
    findings, limitations = [], list(snapshot.limitations)
    if "tests" not in expected_roles:
        limitations.append("Tests reviewer disabled by user; test coverage was not reviewed by that specialist.")
    by_key = {}
    for result in results:
        limitations.extend(f"{result.agent}: {s}" for s in result.limitations)
        if result.error:
            limitations.append(f"{result.agent}: {result.error}")
        for original in result.findings:
            finding = original.model_copy(deep=True)
            lines = snapshot.files.get(finding.path, "").splitlines()
            if (
                not 1 <= finding.line <= finding.end_line <= len(lines)
                or not finding.evidence.strip()
                or finding.evidence not in "\n".join(lines[finding.line-1:finding.end_line])
            ):
                limitations.append(f"{result.agent}: discarded finding without valid source evidence.")
                continue
            if not finding_in_scope(snapshot, finding):
                limitations.append(f"{result.agent}: discarded finding outside the selected commit's changed lines.")
                continue
            if finding.edits:
                try:
                    validate_edits(snapshot.files, finding.edits)
                except ValueError:
                    finding.edits = []
                    limitations.append(f"{result.agent}: removed invalid patch for {finding.path}:{finding.line}.")
            finding.assign_id()
            # Conservative identity avoids merging distinct issues on the same line.
            key = (finding.path, finding.line, finding.title.casefold().strip())
            if key in by_key:
                existing = by_key[key]
                existing.sources = sorted(set(existing.sources + finding.sources + [result.agent]))
                if SEVERITY_ORDER[finding.severity] < SEVERITY_ORDER[existing.severity]:
                    existing.severity = finding.severity
                if not existing.edits:
                    existing.edits = finding.edits
            else:
                finding.sources = sorted(set(finding.sources + [result.agent]))
                by_key[key] = finding
                findings.append(finding)
    findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], -f.confidence, f.path, f.line, f.title))
    return Report(
        source=snapshot.source, revision=snapshot.revision, findings=findings,
        limitations=list(dict.fromkeys(limitations)),
        agents=sorted(
            [result.model_copy(update={"findings": [
                finding for finding in result.findings if finding_in_scope(snapshot, finding)
            ]}) for result in results], key=lambda result: result.agent,
        ),
        complete=(
            len(results) == len(expected_roles)
            and {result.agent for result in results} == set(expected_roles)
            and all(not result.error for result in results)
            and snapshot.scope_complete
        ),
    )


def build_workflow(reviewer):
    graph = StateGraph(State)

    def orchestrate(state):
        return reviewer.orchestrate(state["snapshot"])

    graph.add_node("orchestrator", orchestrate)
    graph.add_edge(START, "orchestrator")

    def combine(state):
        report = reviewer.activity.run(
            "orchestrator", "validate_and_prioritize", aggregate, state["snapshot"], state["results"],
            reviewer.selected_roles,
        )
        if state.get("orchestration_error"):
            report.complete = False
            report.limitations.append(state["orchestration_error"])
        return {"report": report}

    def human_review(state):
        response = interrupt({
            "message": "Review proposed fixes. Approve, edit, reject or request revision before preparing a patch.",
            "finding_ids": [f.id for f in state["report"].findings],
        })
        return {"decisions": response}

    def prepare(state):
        try:
            decisions = [Decision.model_validate(d) for d in state["decisions"]]
            findings = {f.id: f for f in state["report"].findings}
            if len({d.finding_id for d in decisions}) != len(decisions):
                raise ValueError("Duplicate finding decisions")
            edits = []
            for decision in decisions:
                if decision.finding_id not in findings:
                    raise ValueError("Unknown finding")
                if decision.action == "revise":
                    raise ValueError("Request a revised fix and review it before approval")
                if decision.action == "approve":
                    edits.extend(findings[decision.finding_id].edits)
                elif decision.action == "edit":
                    edits.extend(decision.edits)
            updated = reviewer.activity.run(
                "orchestrator", "validate_approved_patch", validate_edits, state["snapshot"].files, edits
            )
            return {
                "updated_files": updated, "patch": patch_text(state["snapshot"].files, updated),
                "validation": [
                    "Exact source matches and patch conflicts checked.",
                    "Changed Python files passed syntax validation.",
                    "Application tests were not executed; runtime behavior is unverified.",
                ],
            }
        except ValueError as error:
            return {"patch": "", "updated_files": {}, "validation": [str(error)]}

    graph.add_node("combine", combine)
    graph.add_node("human_review", human_review)
    graph.add_node("prepare_patch", prepare)
    graph.add_edge("orchestrator", "combine")
    graph.add_edge("combine", "human_review")
    graph.add_edge("human_review", "prepare_patch")
    graph.add_edge("prepare_patch", END)
    return graph.compile(checkpointer=InMemorySaver())


def resume_review(graph, config, decisions):
    return graph.invoke(Command(resume=decisions), config)
