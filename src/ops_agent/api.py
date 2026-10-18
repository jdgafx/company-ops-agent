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


def client_identity(request: Request) -> str:
    """Rate-limit identity. Behind a proxy the RIGHTMOST X-Forwarded-For entry is the address
    the nearest proxy saw; earlier entries are client-supplied and can be spoofed."""
    entries = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",")]
    entries = [p for p in entries if p]
    if entries:
        return entries[-1][:64]
    return request.client.host if request.client else "unknown"


DEMO_HOURLY_DETAIL = (
    "You have used this demo's free questions for this hour. Try again later, "
    f"or send your own OpenRouter key in the {KEY_HEADER} header."
)
DEMO_DAILY_DETAIL = (
    "The free demo has reached its daily limit. Try again tomorrow, "
    f"or send your own OpenRouter key in the {KEY_HEADER} header."
)


class DemoConfig:
    """Free-demo limits and prices, read once from the environment."""

    def __init__(self) -> None:
        env = os.environ.get
        self.per_hour = int(env("OPS_DEMO_PER_HOUR_PER_CLIENT", "3"))
        self.daily_requests = int(env("OPS_DEMO_DAILY_REQUESTS", "20"))
        self.daily_micro_usd = round(float(env("OPS_DEMO_DAILY_USD", "0.10")) * 1_000_000)
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

    # Free demo: a host-held key serves keyless requests. Read once; never returned or logged.
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
            allow_headers=["Content-Type", KEY_HEADER],
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

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/v1/ask")
    def ask(body: AskRequest, request: Request):
        # Key checks come first: a keyless or malformed request costs and counts nothing.
        raw_key = request.headers.get(KEY_HEADER, "")
        ip = client_identity(request)
        if not raw_key.strip() and demo is not None:
            return ask_demo(body.question, ip)
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
