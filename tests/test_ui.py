from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from reviewer import pr_review
from reviewer.agents import Reviewer, demo_result
from reviewer.sources import demo_snapshot

URL = "https://github.com/acme/project/pull/7"
SHA = "a" * 40


def button(app, label):
    return next(item for item in app.button if item.label == label)


def app_test():
    return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()


@pytest.fixture
def offline_review(monkeypatch):
    monkeypatch.setenv("REVIEW_MODEL", "offline-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "unused-environment-key")
    for role in ("quality", "security", "tests"):
        monkeypatch.setenv(f"REVIEW_{role.upper()}_MODEL", "")
    monkeypatch.setenv("GITHUB_TOKEN", "")
    snapshot = demo_snapshot().model_copy(update={
        "source": URL, "owner": "acme", "repo": "project", "pr_number": 7,
        "revision": SHA, "base_revision": "c" * 40, "pr_head_revision": "b" * 40,
        "commit_scoped": True, "changed_lines": {"app.py": [4]}, "changed_paths": ["app.py"],
    })
    monkeypatch.setattr(pr_review, "collect_pr_commit", lambda *args: snapshot.model_copy(deep=True))
    monkeypatch.setattr(Reviewer, "orchestrate", lambda self, snap: {
        "results": [demo_result(role) for role in self.selected_roles], "orchestration_error": "",
    })
    posts = []

    def post(snapshot, body, token):
        posts.append((snapshot.revision, body, token))
        return {"id": 19, "html_url": URL + "#pullrequestreview-19"}

    monkeypatch.setattr(pr_review, "post_pr_review", post)
    return posts


def start(app):
    key = "groq_api_key" if any(field.key == "groq_api_key" for field in app.text_input) else "model_api_key"
    app.text_input(key=key).set_value("test-model-key")
    app.text_input(key="pr_url").set_value(URL)
    app.text_input(key="commit_sha").set_value(SHA)
    app.checkbox(key="model_consent").check()
    button(app, "Start review").click().run()
    assert not app.exception
    return app


def test_pr_only_ui_and_url_validation(offline_review):
    app = app_test()
    assert not app.exception
    assert not app.selectbox
    assert any(field.label == "Pull request URL" for field in app.text_input)
    assert not any(field.label == "Public repository URL" for field in app.text_input)
    button(app, "Start review").click().run()
    assert "Enter a PR URL" in app.error[0].value
    assert not offline_review


def test_consent_required_before_review(offline_review):
    app = app_test()
    app.text_input(key="model_api_key").set_value("test-model-key")
    app.text_input(key="pr_url").set_value(URL)
    button(app, "Start review").click().run()
    assert "Allow review" in app.error[0].value


def test_review_filters_unchanged_findings_and_does_not_post(offline_review):
    app = start(app_test())
    report = app.session_state["review"]["report"]
    assert len(report.findings) == 1
    assert report.findings[0].line == 4
    assert len(app.metric) == 4
    assert button(app, "Post review to PR").disabled
    assert not offline_review


def test_edited_preview_only_posts_on_click_and_once(offline_review):
    app = start(app_test())
    app.text_input(key="github_token").set_value("test-github-token").run()
    app.text_area[0].set_value("Edited review text").run()
    assert not offline_review
    button(app, "Post review to PR").click().run()
    assert not app.exception
    assert offline_review == [(SHA, "Edited review text", "test-github-token")]
    assert button(app, "Post review to PR").disabled
    app.run()
    assert len(offline_review) == 1
    assert any("Review posted" in item.value for item in app.success)


@pytest.mark.parametrize("key,value", [("pr_url", URL + "8"), ("commit_sha", "b" * 40)])
def test_changing_target_clears_previous_posting_controls(offline_review, key, value):
    app = start(app_test())
    app.text_input(key=key).set_value(value).run()
    assert not app.exception
    assert not any(item.label == "Post review to PR" for item in app.button)
    assert not offline_review


def test_failed_post_is_not_retried(offline_review, monkeypatch):
    calls = []

    def fail(*args):
        calls.append(args)
        raise pr_review.PRReviewError("Posting outcome is uncertain.")

    monkeypatch.setattr(pr_review, "post_pr_review", fail)
    app = start(app_test())
    app.text_input(key="github_token").set_value("test-token").run()
    button(app, "Post review to PR").click().run()
    assert not app.exception
    assert any("uncertain" in item.value for item in app.error)
    assert button(app, "Post review to PR").disabled
    app.run()
    assert len(calls) == 1


def test_openai_models_and_ui_key_are_used(offline_review, monkeypatch):
    from reviewer.model_config import OPENAI_BASE_URL
    from reviewer.model_config import OPENAI_MODELS as DEFAULT_MODELS

    monkeypatch.setenv("REVIEW_MODEL", DEFAULT_MODELS["orchestrator"])
    monkeypatch.setenv("OPENAI_BASE_URL", OPENAI_BASE_URL)
    captured = []

    def orchestrate(self, snapshot):
        captured.append((self.api_key, self.model_names(), self.local))
        return {
            "results": [demo_result(role) for role in ("quality", "security", "tests")],
            "orchestration_error": "",
        }

    monkeypatch.setattr(Reviewer, "orchestrate", orchestrate)
    app = app_test()
    assert not app.exception
    field = app.text_input(key="model_api_key")
    assert field.label == "OpenAI API key"
    assert field.value == ""
    assert field.proto.type == field.proto.PASSWORD
    for role, name in DEFAULT_MODELS.items():
        assert app.text_input(key=f"{role}_model").value == name
    app.checkbox(key="include_tests").check()
    app.text_input(key="quality_model").set_value("gpt-5.4-mini")
    app = start(app)
    expected = {**DEFAULT_MODELS, "quality": "gpt-5.4-mini"}
    assert captured == [("test-model-key", expected, False)]
    assert app.session_state["review"]["models"] == expected
    assert "test-model-key" not in str(app.session_state["review"])
    assert not offline_review


def test_api_key_must_be_entered_in_ui(offline_review):
    app = app_test()
    app.text_input(key="pr_url").set_value(URL)
    app.checkbox(key="model_consent").check()
    button(app, "Start review").click().run()
    assert not app.exception
    assert any("Enter an API key" in error.value for error in app.error)
    assert not offline_review


def test_groq_models_and_separate_ui_key_are_used(offline_review, monkeypatch):
    from reviewer.model_config import GROQ_BASE_URL, GROQ_MODEL
    from reviewer.model_config import GROQ_MODELS as DEFAULT_MODELS

    monkeypatch.setenv("REVIEW_MODEL", GROQ_MODEL)
    monkeypatch.setenv("OPENAI_BASE_URL", GROQ_BASE_URL)
    monkeypatch.setenv("GROQ_API_KEY", "unused-environment-groq-key")
    captured = []

    def orchestrate(self, snapshot):
        captured.append((self.api_key, self.model_names(), self.base_url, self.local))
        return {
            "results": [demo_result(role) for role in ("quality", "security", "tests")],
            "orchestration_error": "",
        }

    monkeypatch.setattr(Reviewer, "orchestrate", orchestrate)
    app = app_test()
    app.checkbox(key="include_tests").check()
    field = app.text_input(key="groq_api_key")
    assert field.label == "Groq API key"
    assert field.value == ""
    assert field.proto.type == field.proto.PASSWORD
    assert any(item.label == "Check model access" for item in app.button)
    for role, name in DEFAULT_MODELS.items():
        assert app.text_input(key=f"{role}_model").value == name
    app = start(app)
    assert captured == [("test-model-key", DEFAULT_MODELS, GROQ_BASE_URL, False)]
    assert app.session_state["review"]["models"] == DEFAULT_MODELS
    assert "test-model-key" not in str(app.session_state["review"])
    assert not offline_review


def test_switching_to_groq_does_not_reuse_another_provider_key(offline_review):
    from reviewer.model_config import GROQ_BASE_URL

    app = app_test()
    app.text_input(key="model_api_key").set_value("other-provider-key").run()
    next(field for field in app.text_input if field.label == "OpenAI-compatible base URL (optional)").set_value(
        GROQ_BASE_URL,
    ).run()
    assert not app.exception
    assert app.text_input(key="groq_api_key").value == ""


def test_tests_reviewer_is_opt_in_and_saved_scope_does_not_change(offline_review):
    app = app_test()
    assert not app.checkbox(key="include_tests").value
    assert app.text_input(key="tests_model").disabled
    app = start(app)
    saved = app.session_state["review"]
    assert saved["reviewer_roles"] == ("quality", "security")
    assert saved["report"].complete
    assert {result.agent for result in saved["report"].agents} == {"quality", "security"}
    assert "tests" not in saved["models"]
    assert "Tests reviewer disabled" in saved["body"]
    app.checkbox(key="include_tests").check().run()
    assert app.session_state["review"]["reviewer_roles"] == ("quality", "security")
    app = start(app)
    assert app.session_state["review"]["reviewer_roles"] == ("quality", "security", "tests")
    assert app.session_state["review"]["report"].complete
    assert "Tests reviewer disabled" not in app.session_state["review"]["body"]


def test_empty_disabled_tests_model_does_not_block_review(offline_review):
    app = app_test()
    app.checkbox(key="include_tests").check().run()
    app.text_input(key="tests_model").set_value("").run()
    app.checkbox(key="include_tests").uncheck().run()
    app = start(app)
    assert app.session_state["review"]["report"].complete
    assert "tests" not in app.session_state["review"]["models"]


def test_groq_preset_restores_provider_and_models_without_reusing_key(offline_review):
    from reviewer.model_config import GROQ_BASE_URL, GROQ_MODEL

    app = app_test()
    app.text_input(key="model_api_key").set_value("other-provider-key")
    app.text_input(key="orchestrator_model").set_value("wrong-model")
    button(app, "Use Groq defaults").click().run()
    assert not app.exception
    assert app.text_input(key="model_base_url").value == GROQ_BASE_URL
    for role in ("orchestrator", "quality", "security", "tests"):
        assert app.text_input(key=f"{role}_model").value == GROQ_MODEL
    assert app.text_input(key="groq_api_key").value == ""
    assert not app.checkbox(key="include_tests").value


@pytest.mark.parametrize("include_tests", [False, True])
def test_local_preset_and_review_need_no_hosted_api_key(offline_review, monkeypatch, include_tests):
    from reviewer.model_config import LOCAL_BASE_URL, LOCAL_MODEL

    captured = []
    def orchestrate(self, snapshot):
        captured.append((self.api_key, self.base_url, self.model_names(), self.local))
        return {"results": [demo_result(role) for role in self.selected_roles], "orchestration_error": ""}
    monkeypatch.setattr(Reviewer, "orchestrate", orchestrate)
    app = app_test()
    app.text_input(key="model_api_key").set_value("other-provider-key")
    if include_tests:
        app.checkbox(key="include_tests").check()
    button(app, "Use local defaults").click().run()
    assert not app.exception
    assert app.text_input(key="model_base_url").value == LOCAL_BASE_URL
    for role in ("orchestrator", "quality", "security", "tests"):
        assert app.text_input(key=f"{role}_model").value == LOCAL_MODEL
    assert app.checkbox(key="include_tests").value is include_tests
    assert not any(field.key in {"model_api_key", "groq_api_key"} for field in app.text_input)
    app.text_input(key="pr_url").set_value(URL)
    app.text_input(key="commit_sha").set_value(SHA)
    app.checkbox(key="model_consent").check()
    button(app, "Start review").click().run()
    assert not app.exception
    roles = ("orchestrator", "quality", "security", "tests") if include_tests else (
        "orchestrator", "quality", "security",
    )
    assert captured == [("ollama", LOCAL_BASE_URL, dict.fromkeys(roles, LOCAL_MODEL), True)]
    assert app.session_state["review"]["report"].complete
    assert "other-provider-key" not in str(app.session_state["review"])
