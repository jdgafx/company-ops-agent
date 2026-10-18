"""FastAPI app for the ops agent. Run with: uvicorn --factory ops_agent.api:create_app"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from ops_agent.limits import DailyCap, DemoDailyCap, RateLimiter

if TYPE_CHECKING:
    from ops_agent.config import Settings

log = logging.getLogger("ops_agent.api")


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


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)

    @field_validator("question", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v


def client_identity(request: Request) -> str:
    """Rate-limit identity. Behind a proxy the RIGHTMOST X-Forwarded-For entry is the address
    the nearest proxy saw; earlier entries are client-supplied and can be spoofed."""
    entries = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",")]
    entries = [p for p in entries if p]
    if entries:
        return entries[-1][:64]
    return request.client.host if request.client else "unknown"


DEMO_HOURLY_DETAIL = "You have used this demo's free questions for this hour. Try again later."
DEMO_DAILY_DETAIL = "The free demo has reached its daily limit. Try again tomorrow."


class DemoConfig:
    """Free-demo limits and prices, read once from the environment."""

    def __init__(self) -> None:
        env = os.environ.get
        self.per_hour = int(env("OPS_DEMO_PER_HOUR_PER_CLIENT", "10"))
        self.daily_requests = int(env("OPS_DEMO_DAILY_REQUESTS", "200"))
        self.daily_micro_usd = round(float(env("OPS_DEMO_DAILY_USD", "5.00")) * 1_000_000)
        self.price_in = float(env("OPS_DEMO_PRICE_IN_PER_MTOK", "0.10"))  # USD per 1M tokens
        self.price_out = float(env("OPS_DEMO_PRICE_OUT_PER_MTOK", "0.50"))
        self.path = env("OPS_DEMO_LIMITS_PATH", "./data/demo-limits.runtime.json")

    def micro_usd(self, input_tokens: int, output_tokens: int) -> int:
        """Cost in micro-USD, rounded up. USD per 1M tokens equals micro-USD per token."""
        return math.ceil(input_tokens * self.price_in + output_tokens * self.price_out)


def _limits_from_env() -> tuple[int, int, int, str]:
    return (
        int(os.environ.get("OPS_RATE_LIMIT_PER_MIN", "10")),
        int(os.environ.get("OPS_DAILY_REQUEST_CAP", "200")),
        int(os.environ.get("OPS_DAILY_TOKEN_CAP", "500000")),
        os.environ.get("OPS_LIMITS_PATH", "./data/limits.runtime.json"),
    )


def _default_store():
    from ops_agent.store import Store

    tickets = os.environ.get("OPS_TICKETS_PATH") or None
    return Store(os.environ.get("OPS_DATA_DIR", "./data"), tickets_path=tickets)


def _default_runner(settings: Settings, registry=None, client=None, store=None):
    # Built once at app creation, not per request: the Store holds the ticket list,
    # so a per-request Store would lose tickets and restart ids at T-9001.
    reg = registry
    if reg is None:
        from ops_agent.catalog import build_registry

        reg = build_registry(store or _default_store())

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
    store=None,
) -> FastAPI:
    """Build the app. `runner(question) -> result dict`, `registry`, `store` and
    `client_factory(api_key) -> client` are test injection points.

    The server holds one model key, OPS_DEMO_PROVIDER_KEY, and every /v1/ask call uses it.
    Callers send no key. Without that key the API answers 503 `demo_not_configured`.
    """
    if settings is None:
        from ops_agent.config import Settings

        settings = Settings.from_env()
    per_min, req_cap, tok_cap, env_path = _limits_from_env()
    limiter = RateLimiter(per_min)
    daily = DailyCap(limits_path or env_path, req_cap, tok_cap)
    # One Store serves the default tool registry and /v1/about; it is built once, here.
    if store is None and registry is None and runner is None:
        store = _default_store()
    run = runner or _default_runner(settings, registry, store=store)
    about_store: list = [store]  # filled on first /v1/about when only a registry was injected
    factory = client_factory or _make_client_factory(settings)
    # Worst case per request: every step pays max_tokens output plus a fixed input allowance
    # (tool definitions, history and tool results). Reserved up front, then settled.
    step_allowance = int(os.environ.get("OPS_STEP_INPUT_TOKEN_ALLOWANCE", "8000"))
    token_estimate = int(getattr(settings, "max_steps", 1)) * (
        int(getattr(settings, "max_tokens", 0)) + step_allowance
    )

    # Free demo: the host-held key serves every request. Read once; never returned or logged.
    demo_key = (os.environ.get("OPS_DEMO_PROVIDER_KEY") or "").strip()
    demo = DemoConfig() if demo_key else None
    if demo is not None:
        demo_limiter = RateLimiter(demo.per_hour, window=3600.0)
        demo_daily = DemoDailyCap(
            demo.path, demo.daily_requests, demo.daily_micro_usd
        )
        demo_reserve = demo.micro_usd(
            int(getattr(settings, "max_steps", 1)) * step_allowance,
            int(getattr(settings, "max_steps", 1)) * int(getattr(settings, "max_tokens", 0)),
        )

    app = FastAPI(title="company-ops-agent")

    # CORS for the static wake page: exact origins from OPS_CORS_ORIGINS, none by default.
    cors_origins = [
        o.strip().rstrip("/")
        for o in os.environ.get("OPS_CORS_ORIGINS", "").split(",")
        if o.strip() and o.strip() != "*"
    ]
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type"],
        )

    def ask_demo(question: str, ip: str):
        allowed, retry_after = demo_limiter.allow(ip)
        if not allowed:
            return JSONResponse(
                {"error": "demo_hourly_limit", "detail": DEMO_HOURLY_DETAIL},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
        if not demo_daily.reserve(demo_reserve):
            return JSONResponse(
                {"error": "demo_daily_limit_reached", "detail": DEMO_DAILY_DETAIL},
                status_code=429,
            )
        try:
            if runner is not None:
                result = run(question)
            else:
                try:
                    model_client = _GuardedClient(factory(demo_key))
                except Exception as exc:  # noqa: BLE001
                    log.error("client build failed: %s", type(exc).__name__)
                    raise ModelCallError from None
                result = run(question, model_client)
                del model_client
        except ModelCallError:
            demo_daily.settle(demo_reserve, None)  # usage unknown: keep the reservation charged
            return JSONResponse({"error": "model_call_failed"}, status_code=502)
        except Exception as exc:  # noqa: BLE001
            log.error("agent run failed: %s", type(exc).__name__)
            demo_daily.settle(demo_reserve, None)
            return JSONResponse({"error": "internal_error"}, status_code=500)
        usage = result.get("usage") or {}
        demo_daily.settle(
            demo_reserve,
            demo.micro_usd(int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))),
        )
        return result

    @app.get("/v1/about")
    def about() -> dict:
        if about_store[0] is None:
            from ops_agent.store import Store

            about_store[0] = Store(os.environ.get("OPS_DATA_DIR", "./data"))
        info = about_store[0]
        limits = DemoConfig()
        return {
            "name": "company-ops-agent",
            "description": (
                "A tool-using AI agent that answers questions about a real UK online-retail "
                "dataset by calling lookup tools and showing the calls it made."
            ),
            "dataset": info.dataset_info(),
            "examples": info.examples(),
            "limits": {
                "per_client_per_hour": limits.per_hour,
                "per_day": limits.daily_requests,
            },
            "model": settings.model,
        }

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/v1/ask")
    def ask(body: AskRequest, request: Request):
        if demo is not None:
            return ask_demo(body.question, client_identity(request))
        if runner is None:
            return JSONResponse({"error": "demo_not_configured"}, status_code=503)
        # Injected runner (tests only) owns its own model access; the limits still apply.
        allowed, retry_after = limiter.allow(client_identity(request))
        if not allowed:
            return JSONResponse(
                {"error": "rate_limited"}, status_code=429, headers={"Retry-After": str(retry_after)}
            )
        if not daily.reserve(token_estimate):
            return JSONResponse({"error": "daily_cap_reached"}, status_code=429)
        try:
            result = run(body.question)
        except Exception as exc:  # noqa: BLE001
            log.error("agent run failed: %s", type(exc).__name__)
            daily.settle(token_estimate, None)  # usage unknown: keep the reservation charged
            return JSONResponse({"error": "internal_error"}, status_code=500)
        usage = result.get("usage") or {}
        actual = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
        daily.settle(token_estimate, actual)
        return result

    return app
