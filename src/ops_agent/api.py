"""FastAPI app for the ops agent. Run with: uvicorn --factory ops_agent.api:create_app"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from ops_agent.limits import DailyCap, RateLimiter

if TYPE_CHECKING:
    from ops_agent.config import Settings

log = logging.getLogger("ops_agent.api")

KEY_HEADER = "X-Provider-Key"
KEY_MIN_LEN, KEY_MAX_LEN = 16, 256


class ModelCallError(Exception):
    """A per-request client could not be built or the model call failed. Carries no detail."""


class _GuardedMessages:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def create(self, **kwargs: Any) -> Any:
        try:
            return self._inner.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            # SDK errors can echo request details; keep only the class name.
            log.error("model call failed: %s", type(exc).__name__)
            raise ModelCallError from None


class _GuardedClient:
    """Wraps a per-request client so no exception text from the provider path escapes."""

    def __init__(self, inner: Any) -> None:
        self.messages = _GuardedMessages(inner.messages)


def _clean_key(raw: str) -> str | None:
    """Return the stripped key, or None when its shape is invalid. Never echoes the key."""
    key = raw.strip()
    if not KEY_MIN_LEN <= len(key) <= KEY_MAX_LEN:
        return None
    if any(c.isspace() or not c.isprintable() for c in key):
        return None
    return key


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)

    @field_validator("question", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v


def _limits_from_env() -> tuple[int, int, int, str]:
    return (
        int(os.environ.get("OPS_RATE_LIMIT_PER_MIN", "10")),
        int(os.environ.get("OPS_DAILY_REQUEST_CAP", "200")),
        int(os.environ.get("OPS_DAILY_TOKEN_CAP", "500000")),
        os.environ.get("OPS_LIMITS_PATH", "./data/limits.runtime.json"),
    )


def _default_runner(settings: Settings, registry=None, client=None):
    # Built once at app creation, not per request: the Store holds the ticket list,
    # so a per-request Store would lose tickets and restart ids at T-9001.
    reg = registry
    if reg is None:
        from ops_agent.catalog import build_registry
        from ops_agent.store import Store

        tickets = os.environ.get("OPS_TICKETS_PATH") or None
        reg = build_registry(
            Store(os.environ.get("OPS_DATA_DIR", "./data"), tickets_path=tickets)
        )

    def run(question: str, per_request_client=None) -> dict:
        from ops_agent.agent import run_agent

        chosen = client or per_request_client
        if chosen is None:
            raise RuntimeError("no model client: send a key per request")
        return run_agent(question, tools=reg, client=chosen, settings=settings)

    return run


def _make_client_factory(settings: Settings) -> Callable[[str], Any]:
    def factory(api_key: str) -> Any:
        from ops_agent.config import make_client

        return make_client(settings, api_key)

    return factory


def create_app(
    settings: Settings | None = None,
    *,
    runner=None,
    registry=None,
    limits_path=None,
    client_factory: Callable[[str], Any] | None = None,
) -> FastAPI:
    """Build the app. `runner(question) -> result dict`, `registry` and
    `client_factory(api_key) -> client` are test injection points.

    BYOK: every /v1/ask call carries the caller's own key in X-Provider-Key. The key
    builds one client for that request and is dropped; it is never stored or logged.
    """
    if settings is None:
        from ops_agent.config import Settings

        settings = Settings.from_env()
    per_min, req_cap, tok_cap, env_path = _limits_from_env()
    limiter = RateLimiter(per_min)
    daily = DailyCap(limits_path or env_path, req_cap, tok_cap)
    run = runner or _default_runner(settings, registry)
    factory = client_factory or _make_client_factory(settings)
    # Worst case per request: every step pays max_tokens output plus a fixed input allowance
    # (tool definitions, history and tool results). Reserved up front, then settled.
    step_allowance = int(os.environ.get("OPS_STEP_INPUT_TOKEN_ALLOWANCE", "8000"))
    token_estimate = int(getattr(settings, "max_steps", 1)) * (
        int(getattr(settings, "max_tokens", 0)) + step_allowance
    )

    app = FastAPI(title="company-ops-agent")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/v1/ask")
    def ask(body: AskRequest, request: Request):
        # Key checks come first: a keyless or malformed request costs and counts nothing.
        raw_key = request.headers.get(KEY_HEADER, "")
        if not raw_key.strip():
            return JSONResponse(
                {
                    "error": "byok_key_required",
                    "detail": f"send your own provider key in the {KEY_HEADER} header",
                },
                status_code=401,
            )
        key = _clean_key(raw_key)
        if key is None:
            return JSONResponse({"error": "invalid_key_format"}, status_code=400)
        ip = request.client.host if request.client else "unknown"
        allowed, retry_after = limiter.allow(ip)
        if not allowed:
            return JSONResponse(
                {"error": "rate_limited"}, status_code=429, headers={"Retry-After": str(retry_after)}
            )
        if not daily.reserve(token_estimate):
            return JSONResponse({"error": "daily_cap_reached"}, status_code=429)
        try:
            if runner is not None:
                result = run(body.question)  # injected runner owns its own model access
            else:
                try:
                    model_client = _GuardedClient(factory(key))
                except Exception as exc:  # noqa: BLE001
                    log.error("client build failed: %s", type(exc).__name__)
                    raise ModelCallError from None
                result = run(body.question, model_client)
                del model_client
        except ModelCallError:
            daily.settle(token_estimate, None)  # usage unknown: keep the reservation charged
            return JSONResponse({"error": "model_call_failed"}, status_code=502)
        except Exception as exc:  # noqa: BLE001
            log.error("agent run failed: %s", type(exc).__name__)
            daily.settle(token_estimate, None)  # usage unknown: keep the reservation charged
            return JSONResponse({"error": "internal_error"}, status_code=500)
        usage = result.get("usage") or {}
        actual = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
        daily.settle(token_estimate, actual)
        return result

    return app
