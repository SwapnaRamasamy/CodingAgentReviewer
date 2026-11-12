from __future__ import annotations

import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import streamlit as st
from dotenv import load_dotenv

from reviewer.activity import Activity
from reviewer.agents import Reviewer
from reviewer.diagnostics import safe_error
from reviewer.fixes import patch_text, validate_edits
from reviewer.model_config import (
    DEFAULT_BASE_URL,
    GROQ_BASE_URL,
    GROQ_MODEL,
    LOCAL_BASE_URL,
    LOCAL_MODEL,
    configured_models,
)
from reviewer.pr_review import MAX_REVIEW_CHARS, PRReviewError, collect_pr_commit, post_pr_review, pr_identity
from reviewer.reporting import markdown_report
from reviewer.workflow import build_workflow

load_dotenv()
st.set_page_config(page_title="PR Code Review Studio", page_icon="🔎", layout="wide")
st.title("PR Code Review Studio")
st.caption("Review one commit with quality and security reviewers, plus an optional tests reviewer.")


def background(function, activity, *args):
    progress = st.empty()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(function, *args)
        while not future.done():
            events = activity.snapshot()
            if events:
                event = events[-1]
                progress.info(f'{event["agent"]} · {event["tool"]} · {event["status"]}')
            time.sleep(.15)
        progress.empty()
        return future.result()


def use_local_defaults():
    st.session_state["model_base_url"] = LOCAL_BASE_URL
    for role in ("orchestrator", "quality", "security", "tests"):
        st.session_state[f"{role}_model"] = LOCAL_MODEL


def use_groq_defaults():
    st.session_state["model_base_url"] = GROQ_BASE_URL
    for role in ("orchestrator", "quality", "security", "tests"):
        st.session_state[f"{role}_model"] = GROQ_MODEL


with st.sidebar:
    st.header("Review setup")
    st.button("Use local defaults", on_click=use_local_defaults)
    st.button("Use Groq defaults", on_click=use_groq_defaults)
    base_url = st.text_input(
        "OpenAI-compatible base URL (optional)", value=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        key="model_base_url",
    )
    defaults = configured_models(base_url)
    model = st.text_input("Orchestrator model", value=defaults["orchestrator"], key="orchestrator_model")
    include_tests = st.checkbox(
        "Include tests reviewer", value=False, key="include_tests",
        help="Reviews test coverage and suggests cases; tests are not executed. Uses extra model tokens. "
             "Applies to the next review.",
    )
    with st.expander("Reviewer models"):
        role_models = {
            role: st.text_input(
                f"{role.capitalize()} model", value=defaults[role], key=f"{role}_model",
                disabled=role == "tests" and not include_tests,
            )
            for role in ("quality", "security", "tests")
        }
    active_role_models = {
        role: name for role, name in role_models.items() if include_tests or role != "tests"
    }
    endpoint = urlsplit(base_url.strip())
    openai_endpoint = endpoint.hostname in {None, "api.openai.com"}
    groq_endpoint = endpoint.hostname == "api.groq.com"
    local_ollama = (
        endpoint.scheme in {"http", "https"}
        and endpoint.hostname in {"localhost", "127.0.0.1", "::1"}
        and endpoint.netloc.rsplit(":", 1)[-1] == "11434"
    )
    if local_ollama:
        api_key = "ollama"
        st.caption("Local Ollama: model analysis runs on this computer; no model API key is needed.")
        if model == LOCAL_MODEL:
            st.caption("Qwen3.5:4B | CPU inference | 8K context. Local reviews may take several minutes.")
    else:
        api_key = st.text_input(
            "Groq API key" if groq_endpoint else "OpenAI API key" if openai_endpoint else "Model API key",
            type="password", key="groq_api_key" if groq_endpoint else "model_api_key",
            help="Used for this browser session. The app does not save this key to .env.",
        )
        if groq_endpoint:
            st.caption(
                "Use a Groq API key. Reviews may pause to share the provider's request and token allowance."
            )
        elif openai_endpoint:
            st.caption("OpenAI API usage requires available API credits or active billing.")
    if endpoint.hostname == "openrouter.ai":
        st.caption("Use an OpenRouter API key. Free models have request limits.")
    if not local_ollama and (openai_endpoint or groq_endpoint):  # noqa: SIM102 - supported metadata APIs
        if st.button("Check model access"):
            if not model.strip() or not api_key.strip() or not all(name.strip() for name in active_role_models.values()):
                st.error("Enter the selected model names and an API key first.")
            else:
                try:
                    with st.spinner("Checking model access..."):
                        Reviewer(
                            Activity(), model=model, role_models=role_models,
                            base_url=base_url or None, api_key=api_key, include_tests=include_tests,
                        ).check_model_access()
                    st.success("Credentials and selected model metadata checks succeeded. Review completion is not tested.")
                except Exception as error:  # noqa: BLE001 - safe UI error boundary
                    st.error(safe_error(error))
    github_token = st.text_input(
        "GitHub token (optional for public PRs)", value=os.getenv("GITHUB_TOKEN", ""),
        type="password", key="github_token",
    )
    st.caption(
        "Posting requires Pull requests: write permission. "
        "Private PRs also require read access to pull requests and repository contents."
    )
    consent = st.checkbox(
        "Allow review using the configured model", key="model_consent",
        help=("Ollama processes included code on this computer." if local_ollama else
              "Included source code is sent to the configured model provider. Provider pricing and request limits apply."),
    )

target = st.text_input("Pull request URL", placeholder="https://github.com/owner/repo/pull/123", key="pr_url")
commit_ref = st.text_input(
    "Commit SHA (optional)", placeholder="Leave blank for the latest PR commit", key="commit_sha",
)
st.caption(
    "Reviews this commit against its first parent. Findings must cite its added or modified lines. "
    "Deleted files and unavailable patches are reported as scope limitations."
)
selection = (target.strip(), commit_ref.strip().lower())
review = st.session_state.get("review")
if review and review["selection"] != selection:
    del st.session_state["review"]
    review = None

if st.button("Start review", type="primary"):
    st.session_state.pop("review", None)
    try:
        pr_identity(target)
        if not model.strip() or not all(name.strip() for name in active_role_models.values()):
            raise PRReviewError("Enter a model name for the orchestrator and each selected reviewer in the sidebar.")
        if not api_key.strip():
            raise PRReviewError("Enter an API key for the configured model provider.")
        if not consent:
            raise PRReviewError("Select 'Allow review using the configured model' in the sidebar.")
        activity = Activity()
        with st.status("Collecting commit changes and reviewing code", expanded=True) as status:
            snapshot = background(
                lambda: activity.run("collector", "github.pr_commit", collect_pr_commit, target, commit_ref, github_token),
                activity,
            )
            reviewer = Reviewer(
                activity, model=model, role_models=role_models,
                base_url=base_url or None, api_key=api_key, include_tests=include_tests,
            )
            graph = build_workflow(reviewer)
            config = {"configurable": {"thread_id": uuid.uuid4().hex}}
            background(graph.invoke, activity, {"snapshot": snapshot, "results": []}, config)
            report = graph.get_state(config).values["report"].model_copy(deep=True)
            ui_id = uuid.uuid4().hex
            body = (
                f"Single-commit review for PR #{snapshot.pr_number}.\n\n"
                f"Compared {snapshot.base_revision or 'empty tree'} → {snapshot.revision}.\n\n"
                + markdown_report(report, snapshot)
            )
            st.session_state["review"] = {
                "selection": selection, "snapshot": snapshot, "report": report,
                "activity": activity, "models": reviewer.model_names(), "reviewer_roles": reviewer.selected_roles,
                "ui_id": ui_id,
                "body": body, "post_attempted": False, "post_result": None,
            }
            status.update(
                label="Review ready" if report.complete else "Review incomplete — see scope and limitations",
                state="complete" if report.complete else "error",
            )
    except PRReviewError as error:
        st.error(str(error))
    except Exception as error:  # noqa: BLE001 - preserve safe UI failure reporting
        st.error("Review could not finish. " + safe_error(error))

review = st.session_state.get("review")
if not review:
    st.info("Enter a PR URL and optional commit SHA, configure your model, and click Start review.")
    st.stop()

snapshot, report = review["snapshot"], review["report"]
st.link_button("Open pull request", snapshot.source)
st.caption(f"Reviewed commit: {snapshot.revision} · Parent: {snapshot.base_revision or 'empty tree'}")
st.caption("Models used: " + " | ".join(f"{role.capitalize()}: {name}" for role, name in review["models"].items()))
st.caption("Reviewers used: " + ", ".join(
    role.capitalize() for role in review.get("reviewer_roles", [result.agent for result in report.agents])
))
if not report.complete:
    st.warning("Review incomplete. Inspect the failures and scope limitations before posting.")
    for result in report.agents:
        if result.error:
            st.error(f"{result.agent.capitalize()}: {result.error}")
    for limitation in report.limitations:
        if limitation.startswith("Orchestrator:"):
            st.error(limitation)

for column, severity in zip(st.columns(4), ["Critical", "High", "Medium", "Low"]):
    column.metric(severity, sum(f.severity.value == severity for f in report.findings))
findings_tab, post_tab, activity_tab = st.tabs(["Review findings", "Post review", "Tool activity"])

with findings_tab:
    if not report.findings:
        st.info("No supported findings on the selected commit's changed lines. Check scope limitations.")
    for finding in report.findings:
        with st.expander(f"{finding.severity.value} · {finding.title}", expanded=True):
            st.write(f"{finding.path}:{finding.line}–{finding.end_line} · {finding.agent}")
            st.link_button("View evidence at reviewed commit", snapshot.source_url(finding))
            st.code(finding.evidence)
            st.write("**Impact:** " + finding.impact)
            st.write("**Suggested fix:** " + finding.suggested_fix)
            if finding.suggested_tests:
                st.write("**Suggested tests**")
                for suggestion in finding.suggested_tests:
                    st.write("• " + suggestion)
            if finding.edits:
                st.code(patch_text(snapshot.files, validate_edits(snapshot.files, finding.edits)), language="diff")
    with st.expander("Scope and limitations", expanded=not report.complete):
        st.write("Completion does not imply full coverage or absence of defects. Application tests were not executed.")
        for limitation in report.limitations:
            st.write("• " + limitation)
        st.code("\n".join(snapshot.files))
    st.download_button("Download Markdown", markdown_report(report, snapshot), "review.md", "text/markdown")
    st.download_button("Download JSON", report.model_dump_json(indent=2), "review.json", "application/json")

with post_tab:
    st.write(f"Post a comment review on PR #{snapshot.pr_number}, tied to commit {snapshot.revision}.")
    body = st.text_area(
        "Review text", value=review["body"], height=360, max_chars=MAX_REVIEW_CHARS,
        key="review_body_" + review["ui_id"], disabled=review["post_attempted"],
    )
    with st.expander("Preview review", expanded=True):
        st.markdown(body)
    if not github_token.strip():
        st.info("Enter a GitHub token in the sidebar to enable posting.")
    if st.button(
        "Post review to PR", type="primary",
        disabled=not github_token.strip() or not body.strip() or review["post_attempted"],
        help="Submits the text above as a comment review on the existing PR and notifies its participants.",
    ):
        review["post_attempted"] = True
        try:
            review["post_result"] = background(
                lambda: review["activity"].run("publisher", "github.post_review", post_pr_review,
                                              snapshot, body, github_token),
                review["activity"],
            )
        except PRReviewError as error:
            review["post_error"] = str(error)
        except Exception:  # noqa: BLE001 - a failed POST must never be retried automatically
            review["post_error"] = "Posting outcome is uncertain. Inspect the PR on GitHub."
        st.rerun()
    if review["post_result"]:
        st.success("Review posted to the pull request.")
        url = review["post_result"].get("html_url", "")
        if url.startswith("https://github.com/"):
            st.link_button("View posted review", url)
    elif review.get("post_error"):
        st.error(review["post_error"])
        st.caption("No automatic retry was made. Inspect the PR before starting a new review or posting again.")

with activity_tab:
    st.caption("Tool activity excludes credentials and raw provider responses.")
    events = review["activity"].snapshot()
    if events:
        st.dataframe(events, use_container_width=True, hide_index=True)
        st.download_button("Download tool activity", json.dumps(events, indent=2), "tool-activity.json", "application/json")

st.caption("Review and posting state live in this browser session. Refreshing or restarting may clear that state.")
