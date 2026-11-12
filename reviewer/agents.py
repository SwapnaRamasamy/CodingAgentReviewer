from __future__ import annotations

import json
from threading import Lock
from urllib.parse import urlsplit

from deepagents import create_deep_agent
from deepagents.profiles.harness import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    register_harness_profile,
)
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from reviewer.agent_runtime import ReviewMiddleware
from reviewer.commit_scope import finding_in_scope, line_ranges
from reviewer.diagnostics import MissingStructuredReview, ModelCatalogMismatch, is_output_error, safe_error
from reviewer.local_review import LocalReviewBatch
from reviewer.model_config import LOCAL_MODEL, LOCAL_SOURCE_MODEL
from reviewer.models import AgentResult, Edit, Finding, ReviewBatch, Severity
from reviewer.prompts import (
    LOCAL_OUTPUT_RULES,
    REVISION_SYSTEM_PROMPT,
    SUBAGENT_SYSTEM_PROMPTS,
    orchestrator_system_prompt,
)
from reviewer.provider_limits import shared_groq_limiter
from reviewer.scanners import scan

ROLES = {
    "quality": "Find consequential correctness bugs, edge cases, error-handling defects, and anti-patterns. Avoid cosmetic findings.",
    "security": "Trace untrusted inputs to dangerous operations. Examine authorization, injection, secrets, traversal, SSRF and unsafe configuration. Explain attack prerequisites. Scanner candidates are not proof of exploitability.",
    "tests": "Compare source behavior with existing tests. Find important missing boundary, failure, regression and security cases. Do not invent coverage percentages. Quote the implementation line whose behavior lacks testing.",
}


def snapshot_tools(snapshot, activity, agent, max_lines=120, max_chars=16000):
    @tool
    def read_file(path: str, start_line: int = 1, end_line: int = 100) -> str:
        """Read at most 120 numbered lines from a file in the immutable review snapshot."""
        def read():
            if path not in snapshot.files:
                return "File unavailable in snapshot."
            if start_line < 1 or end_line < start_line:
                return "Invalid line range."
            lines = snapshot.files[path].splitlines()
            stop = min(end_line, start_line + max_lines - 1)
            result = "\n".join(f"{i}: {lines[i-1][:1000]}" for i in range(start_line, min(stop, len(lines))+1))
            return result[:max_chars]
        return activity.run(agent, "read_file", read)

    @tool
    def search_code(query: str) -> str:
        """Literal case-insensitive search across included snapshot files; up to 30 matches."""
        def search():
            if not query or len(query) > 200:
                return "Provide a query of 1–200 characters."
            matches = []
            for path, content in snapshot.files.items():
                for line, text in enumerate(content.splitlines(), 1):
                    if query.casefold() in text.casefold():
                        matches.append(f"{path}:{line}: {text[:300]}")
                        if len(matches) == (10 if max_lines < 120 else 30):
                            return "\n".join(matches)
            return "\n".join(matches) or "No matches."
        return activity.run(agent, "search_code", search)
    return [read_file, search_code]


class Reviewer:
    def __init__(self, activity, model="", demo=False, base_url=None, api_key=None, role_models=None,
                 include_tests=True):
        self.activity, self.model, self.demo = activity, model.strip(), demo
        self.selected_roles = tuple(role for role in ROLES if include_tests or role != "tests")
        self.role_models = {
            role: name.strip() for role, name in (role_models or {}).items() if name and name.strip()
        }
        self.base_url = base_url.strip().rstrip("/") if base_url and base_url.strip() else None
        self.api_key = api_key.strip() if api_key else None
        endpoint = urlsplit(self.base_url or "")
        self.local = endpoint.hostname in {"localhost", "127.0.0.1", "::1"} and endpoint.port == 11434
        self.groq = endpoint.hostname == "api.groq.com"

    def model_name_for(self, agent):
        return self.role_models.get(agent, self.model)

    def model_names(self):
        return {role: self.model_name_for(role) for role in ("orchestrator", *self.selected_roles)}

    def llm(self, model=None, *, tool_calls=True, output_tokens=None):
        selected = self.model if model is None else model
        if not selected:
            raise ValueError("Configure a model name")
        kwargs = {"model": selected, "timeout": 600 if self.local else 60, "max_retries": 0 if self.local or self.groq else 1}
        if self.local:
            # ChatOpenAI renames max_tokens to max_completion_tokens, which Ollama ignores.
            kwargs["extra_body"] = {"max_tokens": 2048 if output_tokens is None else output_tokens}
            kwargs["use_responses_api"] = False
            kwargs["reasoning_effort"] = "none"
        if self.groq and selected == "openai/gpt-oss-120b":
            # Groq GPT-OSS uses Chat Completions and supports sequential tool calls.
            kwargs["use_responses_api"] = False
            kwargs["reasoning_effort"] = "low"
            kwargs["max_completion_tokens"] = 1536 if output_tokens is None else output_tokens
            kwargs["include_response_headers"] = True
            if tool_calls:
                kwargs["model_kwargs"] = {"parallel_tool_calls": False}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return ChatOpenAI(**kwargs)

    def check_model_access(self):
        """Check credentials and model metadata without generating tokens or sending source."""
        # ChatOpenAI borrows a process-cached HTTP transport shared with review models.
        # Closing this SDK wrapper would close that transport and break later requests.
        client = self.model_for("orchestrator").root_client
        if self.groq:
            # Groq documents the catalog endpoint for discovering available model IDs.
            # Compare exact IDs here instead of placing slash-containing IDs in a URL.
            catalog = self.activity.run(
                "configuration", "model.catalog", client.with_options(timeout=15, max_retries=0).models.list,
            )
            available = {model.id for model in catalog.data if getattr(model, "active", True) is not False}

            def validate_selected_models():
                missing = [role for role, name in self.model_names().items() if name not in available]
                if missing:
                    raise ModelCatalogMismatch(missing)

            self.activity.run("configuration", "model.access", validate_selected_models)
            return
        for name in dict.fromkeys(self.model_names().values()):
            self.activity.run(
                "configuration", "model.access", client.with_options(timeout=15, max_retries=0).models.retrieve,
                name,
            )

    def model_for(self, agent):
        """Select the role's model while sharing provider settings and credentials."""
        return self.llm(self.model_name_for(agent))

    def groq_limiter(self, agent):
        return shared_groq_limiter(self.api_key or "", self.model_name_for(agent)) if self.groq else None

    def context(self, snapshot, findings):
        snippets, used = [], 0
        budget = 6000 if self.local or self.groq else 18000
        ordered = list(dict.fromkeys(snapshot.changed_paths + list(snapshot.files)))
        for path in ordered:
            if path in snapshot.files and used < budget:
                lines = snapshot.files[path].splitlines()
                selected = set(range(1, min(30, len(lines)) + 1))
                if snapshot.commit_scoped:
                    selected = set()
                    for changed in snapshot.changed_lines.get(path, [])[:30]:
                        selected.update(range(max(1, changed - 3), min(len(lines), changed + 3) + 1))
                excerpt = "\n".join(f"{i}: {lines[i - 1][:300]}" for i in sorted(selected)[:60])
                entry = f"FILE {path}\n{excerpt}"[:budget-used]
                snippets.append(entry)
                used += len(entry)
        return {
            "revision": snapshot.revision, "files": list(snapshot.files), "initial_snippets": snippets,
            "commit_scoped": snapshot.commit_scoped, "base_revision": snapshot.base_revision,
            "scope": "Only findings evidenced on added/modified lines of this commit." if snapshot.commit_scoped else "Snapshot",
            "changed_line_ranges": {path: line_ranges(lines)[:100] for path, lines in snapshot.changed_lines.items()},
            "changed_paths": snapshot.changed_paths,
            "diff": snapshot.diff[:2000 if self.local else 6000 if self.groq else 20000],
            "scanner_candidates": [{"title": f.title[:250], "path": f.path, "line": f.line,
                                    "evidence": f.evidence[:500]} for f in findings[:2 if self.local else 25]],
        }

    def specialist(self, agent, snapshot, system_prompt=None):
        prompt = system_prompt or SUBAGENT_SYSTEM_PROMPTS[agent]
        if (
            self.groq and self.model_name_for(agent) == "openai/gpt-oss-120b"
            or self.local and self.model_name_for(agent) in {LOCAL_MODEL, LOCAL_SOURCE_MODEL}
        ):
            from reviewer.structured_review import specialist
            return specialist(
                self, agent, snapshot, prompt,
                snapshot_tools(snapshot, self.activity, agent,
                               max_lines=40 if self.local else 60, max_chars=1500 if self.local else 3000),
                revision=system_prompt is not None,
            )
        compact = self.local and system_prompt is None
        schema = LocalReviewBatch if compact else ReviewBatch
        if compact:
            prompt += LOCAL_OUTPUT_RULES
        return create_agent(
            model=self.model_for(agent), system_prompt=prompt,
            tools=snapshot_tools(snapshot, self.activity, agent),
            response_format=ProviderStrategy(schema) if self.local else ToolStrategy(schema),
            middleware=[ReviewMiddleware(
                self.activity, agent, prompt, self.local, {"read_file", "search_code"},
                omit_message_names=self.groq, rate_limiter=self.groq_limiter(agent),
            )],
            name=f"{agent}_reviewer",
        )

    def review(self, agent, snapshot, assignment=""):
        if self.demo:
            return demo_result(agent)
        findings, limitations = [], []
        try:
            findings, limitations = scan(snapshot, self.activity, agent)
            findings = [finding for finding in findings if finding_in_scope(snapshot, finding)]
            limitations.append("Subagent uses bounded excerpts and tools; repository coverage is partial. Tests were not executed.")
            specialist = self.specialist(agent, snapshot)
            context = self.context(snapshot, findings)
            context["delegated_objective"] = assignment[:1500]
            for attempt in range(2 if self.local else 1):
                try:
                    state = specialist.invoke({"messages": [HumanMessage(content=json.dumps(context))]},
                                              {"recursion_limit": 24})
                    batch = state.get("structured_response")
                    if self.local:
                        if not isinstance(batch, LocalReviewBatch):
                            raise MissingStructuredReview("Subagent did not return a compact structured review.")
                        batch = batch.validated_review(snapshot)
                    elif not isinstance(batch, ReviewBatch):
                        raise MissingStructuredReview("Subagent did not return a structured review.")
                    break
                except Exception as error:
                    if not (self.local and attempt == 0 and is_output_error(error)):
                        raise
                    self.activity.add(
                        agent, "model.output_correction", "retrying",
                        summary="Retrying invalid local output once. " + safe_error(error),
                    )
                    limitations.append("Local model output required a correction; output and coverage remain bounded.")
                    context["output_correction"] = (
                        "Your previous response was invalid or cut off. Return concise fields using the provided schema, "
                        "with at most one prioritized supported finding and exact source evidence at valid paths and lines. "
                        "Do not copy complete files or generate patches. An empty findings list is appropriate "
                        "only when no supported issue is found. " + safe_error(error)
                    )
            for finding in batch.findings:
                finding.agent, finding.sources = agent, [agent]
            return AgentResult(agent=agent, findings=findings + batch.findings,
                               limitations=limitations + batch.limitations,
                               error=("Evidence coverage was reduced; the review is incomplete."
                                      if state.get("gathering_incomplete") else ""))
        except Exception as error:  # noqa: BLE001 - preserve scanner findings at the agent failure boundary
            self.activity.add(agent, "subagent.result", "failed", summary=safe_error(error))
            return AgentResult(agent=agent, findings=findings, limitations=limitations, error=safe_error(error))

    def orchestrate(self, snapshot):
        if "tests" not in self.selected_roles:
            self.activity.add("tests", "reviewer.selection", "skipped", summary="Tests reviewer disabled by user.")
        if self.demo:
            results = []
            for role in self.selected_roles:
                try:
                    results.append(self.review(role, snapshot))
                except Exception:  # noqa: BLE001 - isolate demo reviewer failures
                    results.append(AgentResult(agent=role, error="Reviewer failed unexpectedly; this review is incomplete."))
            return {"results": results, "orchestration_error": ""}

        prompt = orchestrator_system_prompt(self.selected_roles)
        results, locks = {}, {role: Lock() for role in self.selected_roles}
        def subagent(role):
            def run(state):
                with locks[role]:
                    if role not in results:
                        assignment = state["messages"][-1].content
                        results[role] = self.review(role, snapshot, str(assignment))
                    result = results[role]
                # Full findings go directly to validation; the supervisor sees only a bounded summary.
                summary = {"agent": role, "status": "failed" if result.error else "completed",
                           "finding_count": len(result.findings), "error": result.error,
                           "titles": [f.title[:160] for f in result.findings[:3]]}
                return {"messages": [AIMessage(content=json.dumps(summary))], "structured_response": summary}
            return {"name": role, "description": ROLES[role], "runnable": RunnableLambda(run)}

        error = ""
        try:
            # Disable unrelated built-ins: only planning and the three named review tasks are exposed.
            register_harness_profile(f"openai:{self.model_name_for('orchestrator')}", HarnessProfile(
                excluded_tools=frozenset({"ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute"}),
                excluded_middleware=frozenset({"SummarizationMiddleware"}),
                general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
            ))
            supervisor = create_deep_agent(
                model=self.model_for("orchestrator"), system_prompt=prompt,
                subagents=[subagent(role) for role in self.selected_roles],
                middleware=[TodoListMiddleware(), ReviewMiddleware(
                    self.activity, "orchestrator", prompt, self.local,
                    {"task", "write_todos"}, max_models=8, max_tools=8, subagent_roles=self.selected_roles,
                    omit_message_names=self.groq, rate_limiter=self.groq_limiter("orchestrator"),
                )],
                name="code_review_orchestrator",
            )
            supervisor.invoke({"messages": [HumanMessage(content=json.dumps({
                "request": "Review this snapshot. Delegate once each to "
                           + ", ".join(self.selected_roles) + ", then summarize outcomes.",
                "revision": snapshot.revision, "file_count": len(snapshot.files),
                "source": snapshot.source,
            }))]}, {"recursion_limit": 32})
        except Exception as failure:  # noqa: BLE001 - retain completed subagents if supervision fails
            self.activity.add("orchestrator", "orchestrator.result", "failed", summary=safe_error(failure))
            error = "Orchestrator: " + safe_error(failure)
        for role in self.selected_roles:
            if role not in results:
                results[role] = AgentResult(agent=role, error="Orchestrator did not complete delegation to this specialist.")
        return {"results": [results[role] for role in self.selected_roles], "orchestration_error": error}

    def revise(self, finding, snapshot, note):
        if self.demo:
            return finding.model_copy(deep=True)
        role = finding.agent
        prompt = SUBAGENT_SYSTEM_PROMPTS[role] + "\n" + REVISION_SYSTEM_PROMPT
        paths = {finding.path, *(edit.path for edit in finding.edits)}
        relevant = snapshot.model_copy(update={
            "files": {path: text for path, text in snapshot.files.items() if path in paths},
            "changed_paths": [path for path in snapshot.changed_paths if path in paths],
            "diff": "",
        })
        context = self.context(relevant, [])
        context.update({"finding": finding.model_dump(mode="json"), "human_feedback": note[:4000]})
        specialist = self.specialist(role, snapshot, system_prompt=prompt)
        state = self.activity.run(
            role, "model.revise_fix", specialist.invoke,
            {"messages": [HumanMessage(content=json.dumps(context))]}, {"recursion_limit": 24},
        )
        batch = state.get("structured_response")
        if not isinstance(batch, ReviewBatch):
            raise MissingStructuredReview("Revision did not return a structured review.")
        if len(batch.findings) != 1:
            raise ValueError("Expected exactly one revised finding")
        revised = finding.model_copy(deep=True)
        revised.edits = batch.findings[0].edits
        revised.suggested_fix = batch.findings[0].suggested_fix
        revised.suggested_tests = batch.findings[0].suggested_tests
        return revised



def demo_result(agent):
    if agent == "quality":
        finding = Finding(
            agent=agent, title="Empty input causes division by zero", severity=Severity.MEDIUM, confidence=.99,
            path="app.py", line=10, end_line=10, evidence="return sum(values) / len(values)",
            impact="Calling average([]) raises ZeroDivisionError.",
            suggested_fix="Reject empty input with a clear ValueError.",
            suggested_tests=["Assert average([]) raises ValueError.", "Test negative and single-item inputs."],
            edits=[Edit(path="app.py", original="    return sum(values) / len(values)",
                        replacement='    if not values:\n        raise ValueError("values must not be empty")\n    return sum(values) / len(values)')],
        )
    elif agent == "security":
        finding = Finding(
            agent=agent, title="Untrusted input is concatenated into SQL", severity=Severity.HIGH, confidence=.98,
            path="app.py", line=4, end_line=4, evidence="db.execute",
            impact="If user_input is attacker-controlled, SQL syntax can be injected into this query.",
            suggested_fix="Use SQLite parameter binding.",
            suggested_tests=["Pass a quote-containing name.", "Verify SQL injection input is treated as a literal."],
            edits=[Edit(
                path="app.py",
                original='    return db.execute("SELECT * FROM users WHERE name = \'" + user_input + "\'")',
                replacement='    return db.execute("SELECT * FROM users WHERE name = ?", (user_input,))',
            )],
        )
    else:
        finding = Finding(
            agent=agent, title="Command execution lacks security regression tests", severity=Severity.MEDIUM,
            confidence=.9, path="app.py", line=7, end_line=7, evidence="shell=True",
            impact="Existing tests only cover average; command handling has no tests.",
            suggested_fix="Define an allowed command interface and add tests rejecting shell metacharacters.",
            suggested_tests=["Verify unapproved commands are rejected.", "Mock subprocess and check shell=False."],
        )
    return AgentResult(agent=agent, findings=[finding], limitations=["Fixed demo example; not an exhaustive analysis."])
