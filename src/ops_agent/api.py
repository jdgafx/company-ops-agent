"""FastAPI app for the ops agent. Run with: uvicorn --factory ops_agent.api:create_app"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from ops_agent.limits import DailyCap, RateLimiter

if TYPE_CHECKING:
    from ops_agent.config import Settings

log = logging.getLogger("ops_agent.api")


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

    def run(question: str) -> dict:
        from ops_agent.agent import run_agent
        from ops_agent.config import make_client

        return run_agent(
            question, tools=reg, client=client or make_client(settings), settings=settings
        )

    return run


def create_app(
    settings: Settings | None = None, *, runner=None, registry=None, limits_path=None
) -> FastAPI:
    """Build the app. `runner(question) -> result dict` and `registry` are test injection points."""
    if settings is None:
        from ops_agent.config import Settings

        settings = Settings.from_env()
    per_min, req_cap, tok_cap, env_path = _limits_from_env()
    limiter = RateLimiter(per_min)
    daily = DailyCap(limits_path or env_path, req_cap, tok_cap)
    run = runner or _default_runner(settings, registry)
    token_estimate = int(getattr(settings, "max_tokens", 0)) * int(getattr(settings, "max_steps", 1))

    app = FastAPI(title="company-ops-agent")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/v1/ask")
    def ask(body: AskRequest, request: Request):
        ip = request.client.host if request.client else "unknown"
        allowed, retry_after = limiter.allow(ip)
        if not allowed:
            return JSONResponse(
                {"error": "rate_limited"}, status_code=429, headers={"Retry-After": str(retry_after)}
            )
        if not daily.check(token_estimate):
            return JSONResponse({"error": "daily_cap_reached"}, status_code=429)
        try:
            result = run(body.question)
        except Exception:
            log.exception("agent run failed")
            return JSONResponse({"error": "internal_error"}, status_code=500)
        usage = result.get("usage") or {}
        daily.record(int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0)))
        return result

    return app
