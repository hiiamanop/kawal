"""Shared OpenRouter configuration for every remote model call in KAWAL.

KAWAL keeps its classification, NER, and embedding models local. Every remote
LLM/VLM invocation (gated escalation, causal disambiguation, and vision
analysis) goes through OpenRouter so there is a single provider, a single
credential, and a single audit surface.
"""

from __future__ import annotations

import os

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"
DEFAULT_OPENROUTER_APP_NAME = "KAWAL"
DEFAULT_OPENROUTER_REFERER = "https://github.com/hiiamanop/kawal"


def openrouter_base_url(explicit: str | None = None) -> str:
    return (
        explicit
        or os.getenv("KAWAL_OPENROUTER_BASE_URL")
        or os.getenv("OPENROUTER_BASE_URL")
        or DEFAULT_OPENROUTER_BASE_URL
    ).rstrip("/")


def openrouter_api_key(explicit: str | None = None) -> str:
    return (
        explicit
        or os.getenv("KAWAL_OPENROUTER_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or ""
    )


def openrouter_model(explicit: str | None = None, *, env_var: str = "KAWAL_OPENROUTER_MODEL") -> str:
    return (
        explicit
        or os.getenv(env_var)
        or os.getenv("OPENROUTER_MODEL")
        or DEFAULT_OPENROUTER_MODEL
    )


def openrouter_model_for(kind: str) -> str:
    """Resolve a task-specific model with a sensible fallback to the shared one."""

    return openrouter_model(env_var=f"KAWAL_OPENROUTER_{kind.upper()}_MODEL")


def openrouter_headers(explicit_key: str | None = None) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "HTTP-Referer": os.getenv("KAWAL_OPENROUTER_REFERER", DEFAULT_OPENROUTER_REFERER),
        "X-Title": os.getenv("KAWAL_OPENROUTER_APP_NAME", DEFAULT_OPENROUTER_APP_NAME),
    }
    key = openrouter_api_key(explicit_key)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers
