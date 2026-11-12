"""Provider and model defaults for the PR review setup."""
from __future__ import annotations

import os
from urllib.parse import urlsplit

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODELS = {
    "orchestrator": "gpt-5.4-mini",
    "quality": "gpt-5.4",
    "security": "gpt-5.4",
    "tests": "gpt-5.4-mini",
}
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_MODELS = dict.fromkeys(OPENAI_MODELS, GROQ_MODEL)
LOCAL_BASE_URL = "http://127.0.0.1:11434/v1"
LOCAL_SOURCE_MODEL = "qwen3.5:4b"
LOCAL_MODEL = "qwen3.5-review:4b"
LOCAL_MODELS = dict.fromkeys(OPENAI_MODELS, LOCAL_MODEL)
DEFAULT_BASE_URL = LOCAL_BASE_URL
DEFAULT_MODELS = dict(LOCAL_MODELS)


def configured_models(base_url=DEFAULT_BASE_URL, environ=None):
    """Read role overrides with defaults for local Ollama, Groq, and OpenAI."""
    settings = os.environ if environ is None else environ
    endpoint = urlsplit(base_url.strip())
    local = (
        endpoint.hostname in {"localhost", "127.0.0.1", "::1"}
        and endpoint.netloc.rsplit(":", 1)[-1] == "11434"
    )
    defaults = (
        LOCAL_MODELS if local else
        OPENAI_MODELS if endpoint.hostname in {None, "api.openai.com"} else
        GROQ_MODELS if endpoint.hostname == "api.groq.com" else {}
    )
    shared = settings.get("REVIEW_MODEL", "").strip() or defaults.get("orchestrator", LOCAL_MODEL)
    if local:
        defaults = dict.fromkeys(LOCAL_MODELS, shared)
    return {
        "orchestrator": shared,
        **{
            role: settings.get(f"REVIEW_{role.upper()}_MODEL", "").strip() or defaults.get(role, shared)
            for role in ("quality", "security", "tests")
        },
    }
