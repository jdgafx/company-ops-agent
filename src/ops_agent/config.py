"""Runtime settings for the agent loop, read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

import anthropic

DEFAULT_MODEL = "claude-haiku-5-5"


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

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
        if not 1 <= self.max_steps <= 12:
            raise ValueError(f"max_steps must be between 1 and 12, got {self.max_steps}")
        if not 64 <= self.max_tokens <= 4096:
            raise ValueError(f"max_tokens must be between 64 and 4096, got {self.max_tokens}")

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model=os.environ.get("OPS_AGENT_MODEL") or DEFAULT_MODEL,
            max_steps=_int_env("OPS_AGENT_MAX_STEPS", 8, 1, 12),
            max_tokens=_int_env("OPS_AGENT_MAX_TOKENS", 1024, 64, 4096),
        )


def make_client(settings: Settings) -> anthropic.Anthropic:
    """Build the SDK client. The key comes from ANTHROPIC_API_KEY and is never logged."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set; export it before running the agent")
    return anthropic.Anthropic(api_key=key)
