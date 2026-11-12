"""Actionable failure messages that never include provider payloads or credentials."""
from __future__ import annotations

import math
import ssl


class AgentBudgetExceeded(ValueError):
    pass


class ContextBudgetExceeded(ValueError):
    pass


class MissingStructuredReview(TypeError):
    pass


class ModelCatalogMismatch(ValueError):
    """Selected role models are absent from the provider catalog."""
    def __init__(self, roles):
        self.roles = tuple(role for role in ("orchestrator", "quality", "security", "tests") if role in roles)
        super().__init__("A selected model is absent from the provider catalog.")


class InvalidReviewEvidence(ValueError):
    pass


def _chain(error):
    seen = set()
    while error is not None and id(error) not in seen and len(seen) < 12:
        seen.add(id(error))
        yield error
        source = getattr(error, "source", None)
        error = (
            error.__cause__
            or (source if isinstance(source, Exception) else None)
            or (None if error.__suppress_context__ else error.__context__)
        )



def is_output_error(error):
    return any(
        isinstance(item, (MissingStructuredReview, InvalidReviewEvidence))
        or type(item).__name__ in {
            "LengthFinishReasonError", "ValidationError", "OutputParserException", "StructuredOutputError",
            "StructuredOutputValidationError", "MultipleStructuredOutputsError", "JSONDecodeError",
        }
        for item in _chain(error)
    )


def safe_error(error):
    errors = list(_chain(error))
    for item in errors:
        if isinstance(item, ModelCatalogMismatch):
            roles = ", ".join(item.roles) or "selected reviewers"
            return (
                "Groq's model catalog does not list the configured model for: " + roles
                + ". Check the sidebar model IDs and Groq model access."
            )
    kinds = {type(item).__name__ for item in errors}
    # These fields are inspected only for exact, known codes. Never render their values.
    codes = set()
    statuses = set()
    for item in errors:
        code = getattr(item, "code", None)
        if isinstance(code, str):
            codes.add(code)
        body = getattr(item, "body", None)
        if isinstance(body, dict):
            details = body.get("error", body)
            if isinstance(details, dict) and isinstance(details.get("code"), str):
                codes.add(details["code"])
        status = getattr(item, "status_code", None)
        if type(status) is int:
            statuses.add(status)
    if any(isinstance(item, ssl.SSLCertVerificationError) for item in errors):
        return (
            "HTTPS certificate verification failed. Check the system's trusted certificates or "
            "SSL_CERT_FILE with your IT team, then restart the app."
        )
    if "LengthFinishReasonError" in kinds:
        return "The model response was truncated at the output token limit. Use a more concise review or increase the output allowance."
    if any(isinstance(item, InvalidReviewEvidence) for item in errors):
        return "The model returned findings with invalid source evidence or locations."
    if kinds & {"APITimeoutError", "TimeoutError", "ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout"}:
        return "The request timed out. Check provider availability or use a smaller review scope."
    if "AuthenticationError" in kinds or 401 in statuses:
        return "Model authentication failed (401). Enter an active API key in the sidebar and retry."
    if "PermissionDeniedError" in kinds or 403 in statuses:
        return "Model access was denied (403). Check the API key's project and model permissions."
    if "credit_balance_exhausted" in codes:
        return "The provider's prepaid credits are exhausted. Add API credits before retrying."
    if codes & {"project_spend_limit_exceeded", "organization_spend_limit_exceeded", "billing_hard_limit_reached"}:
        return "The provider's spending limit was reached. Check project/organization billing limits before retrying."
    if "organization_usage_limit_exceeded" in codes:
        return "The provider's organization usage limit was reached. Check API usage limits before retrying."
    if "insufficient_quota" in codes:
        return "API quota or billing is exhausted. Check API credits and project/organization limits before retrying."
    if "ProviderCooldown" in kinds:
        return "Shared Groq capacity remained unavailable after a bounded wait. Wait before starting another review."
    if "RateLimitError" in kinds or 429 in statuses:
        from reviewer.provider_limits import rate_limit_delay
        delay = rate_limit_delay(error)
        if delay is not None:
            return (
                "The provider rate limit was reached (429). Retry after at least "
                + str(math.ceil(delay)) + " seconds. Other agents share this provider allowance."
            )
        return "The provider rate limit was reached (429). Wait before retrying and reduce request frequency."
    if "model_not_found" in codes:
        return "The configured model is unavailable to this API key. Check the sidebar Model and project access."
    if "NotFoundError" in kinds or 404 in statuses:
        return "The provider endpoint or model was not found (404). Check the base URL, model ID, and access."
    if "ProviderRequestBudgetExceeded" in kinds:
        return "This request exceeds the app's conservative Groq token budget. Use a smaller review scope."
    if "context_length_exceeded" in codes or any(isinstance(item, ContextBudgetExceeded) for item in errors):
        return "The model context limit was exceeded. Use a smaller review scope or a model with more context."
    if any(isinstance(item, AgentBudgetExceeded) for item in errors) or "GraphRecursionError" in kinds:
        return "Agent call budget exhausted before a final review was returned. Use a smaller review scope."
    if any(isinstance(item, MissingStructuredReview) for item in errors) or kinds & {
        "ValidationError", "OutputParserException", "StructuredOutputError",
        "StructuredOutputValidationError", "MultipleStructuredOutputsError", "JSONDecodeError",
    }:
        return "The model did not return a valid structured review. Use a model with reliable structured-output support."
    if "APIConnectionError" in kinds or kinds & {"ConnectError", "ConnectionError", "ProxyError"}:
        return "Could not connect to the model provider. Check the base URL, proxy/VPN, firewall, and certificates."
    if "BadRequestError" in kinds or 400 in statuses:
        if "tool_use_failed" in codes:
            return (
                "The provider rejected a missing or invalid tool call (400; tool_use_failed). "
                "The reviewer could not complete its required tool or structured-review call."
            )
        if "json_validate_failed" in codes:
            return (
                "The provider rejected generated JSON (400; json_validate_failed). "
                "The model output did not satisfy the requested JSON format or schema."
            )
        return (
            "The model provider rejected the request (400). The available error details do not identify "
            "the cause. Check request parameters, tool calls, and the output schema."
        )
    if "InternalServerError" in kinds or any(500 <= status < 600 for status in statuses):
        return "The model provider reported a server error. Retry after a short wait."
    return unexpected_error_summary(error)


def unexpected_error_summary(error):
    """Expose exception types and library locations, never messages or frame values."""
    trusted = (
        "builtins", "reviewer", "openai", "langchain", "langchain_core",
        "langchain_openai", "langgraph", "deepagents", "httpx", "httpcore",
        "anyio", "pydantic",
    )

    def known_module(module):
        return isinstance(module, str) and any(
            module == prefix or module.startswith(prefix + ".") for prefix in trusted
        )

    kinds, locations = [], []
    for item in _chain(error):
        kind = type(item)
        if known_module(kind.__module__):
            kinds.append(kind.__name__)
        trace = item.__traceback__
        while trace is not None:
            module = trace.tb_frame.f_globals.get("__name__", "")
            if known_module(module):
                locations.append(f"{module}:{trace.tb_lineno}")
            trace = trace.tb_next
    detail = ", ".join(dict.fromkeys(kinds))[:80] or "unclassified exception"
    if locations:
        detail += "; at " + locations[-1][:90]
    return f"Unexpected failure ({detail}). Raw error data is not recorded."

