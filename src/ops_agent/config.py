"""Runtime settings for the agent loop, read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

import anthropic

DEFAULT_MODEL = "anthropic/claude-haiku-5.5"
DEFAULT_BASE_URL = "https://openrouter.ai/api"


def _int_env(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    model: str = DEFAULT_MODEL
    max_steps: int = 8
    max_tokens: int = 1024
    base_url: str = DEFAULT_BASE_URL

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
        if not 1 <= self.max_steps <= 12:
            raise ValueError(f"max_steps must be between 1 and 12, got {self.max_steps}")
        if not 64 <= self.max_tokens <= 4096:
            raise ValueError(f"max_tokens must be between 64 and 4096, got {self.max_tokens}")
        if not self.base_url.startswith("https://"):
            raise ValueError("base_url must start with https://")

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model=os.environ.get("OPS_AGENT_MODEL") or DEFAULT_MODEL,
            max_steps=_int_env("OPS_AGENT_MAX_STEPS", 8, 1, 12),
            max_tokens=_int_env("OPS_AGENT_MAX_TOKENS", 1024, 64, 4096),
            base_url=(os.environ.get("OPS_AGENT_BASE_URL") or "").strip() or DEFAULT_BASE_URL,
        )


def make_client(settings: Settings, api_key: str) -> anthropic.Anthropic:
    """Build the SDK client for the Anthropic-compatible endpoint. The key is passed in by the
    caller (never read from the environment here) and is never logged."""
    if not api_key:
        raise ValueError("api_key must not be empty")
    return anthropic.Anthropic(
        base_url=settings.base_url, auth_token=api_key, timeout=60.0, max_retries=1
    )
