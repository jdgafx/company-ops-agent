"""SU.1: in-app total-spend stop for the demo key, measured from the key's usage."""

import logging
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from ops_agent.api import create_app

SETTINGS = NS(model="m", max_steps=2, max_tokens=100)
DEMO_KEY = "sk-demo-SECRETSECRETSECRET"
BYOK_KEY = "sk-test-AAAAAAAAAAAAAAAA"
Q = {"question": "How many open tickets are there?"}
ENV = (
    "OPS_DEMO_PROVIDER_KEY", "OPS_DEMO_TOTAL_USD", "OPS_DEMO_PER_HOUR_PER_CLIENT",
    "OPS_DEMO_DAILY_REQUESTS", "OPS_DEMO_DAILY_USD", "OPS_RATE_LIMIT_PER_MIN",
)


class FakeClient:
    def __init__(self):
        self.messages = self

    def create(self, **kwargs):
        return NS(
            content=[NS(type="text", text="done")],
            stop_reason="end_turn",
            usage=NS(input_tokens=3, output_tokens=2),
        )


class Fetcher:
    """Scripted usage values; an Exception item is raised. The last item repeats."""

    def __init__(self, *values):
        self.values = list(values)
        self.calls = 0

    def __call__(self):
        v = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        if isinstance(v, Exception):
            raise v
        return v


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def env(monkeypatch, tmp_path):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPS_DEMO_PROVIDER_KEY", DEMO_KEY)
    monkeypatch.setenv("OPS_DEMO_LIMITS_PATH", str(tmp_path / "demo.json"))
    monkeypatch.setenv("OPS_LIMITS_PATH", str(tmp_path / "limits.json"))
    monkeypatch.setenv("OPS_DEMO_PER_HOUR_PER_CLIENT", "100")
    monkeypatch.setenv("OPS_DEMO_DAILY_REQUESTS", "100")
    return monkeypatch


def build(fetcher, clock=None):
    from ops_agent.catalog import build_registry
    from ops_agent.store import Store

    app = create_app(
        SETTINGS,
        registry=build_registry(Store("data")),
        client_factory=lambda key: FakeClient(),
        demo_usage_fetcher=fetcher,
        demo_clock=clock or Clock(),
    )
    return TestClient(app)


def used_up(r):
    return (
        r.status_code == 429
        and r.json()["error"] == "demo_budget_used_up"
        and "Retry-After" not in r.headers
        and "X-Provider-Key" in r.json()["detail"]
    )


def test_delta_below_stop_allows(env):
    clock = Clock()
    c = build(Fetcher(186.50, 187.20), clock)
    clock.t += 61
    assert c.post("/v1/ask", json=Q).status_code == 200  # delta 0.70 < 1.00


@pytest.mark.parametrize("usage", [187.50, 190.0])
def test_delta_at_or_above_stop_denies(env, usage):
    clock = Clock()
    c = build(Fetcher(186.50, usage), clock)
    clock.t += 61
    assert used_up(c.post("/v1/ask", json=Q))


def test_env_stop_amount_is_used(env):
    env.setenv("OPS_DEMO_TOTAL_USD", "0.25")
    clock = Clock()
    c = build(Fetcher(10.0, 10.25), clock)
    clock.t += 61
    assert used_up(c.post("/v1/ask", json=Q))


def test_baseline_failure_denies_but_healthz_ok(env):
    c = build(Fetcher(RuntimeError("boom")))
    assert c.get("/healthz").status_code == 200
    assert used_up(c.post("/v1/ask", json=Q))


def test_baseline_failure_stays_denied_for_process_life(env):
    clock = Clock()
    f = Fetcher(RuntimeError("boom"), 0.0)
    c = build(f, clock)
    clock.t += 600
    assert used_up(c.post("/v1/ask", json=Q))
    assert f.calls == 1


def test_fetch_failure_after_baseline_denies(env):
    clock = Clock()
    c = build(Fetcher(5.0, RuntimeError("down")), clock)
    clock.t += 61
    assert used_up(c.post("/v1/ask", json=Q))


def test_fetch_is_cached_within_window(env):
    clock = Clock()
    f = Fetcher(5.0, 5.1)
    c = build(f, clock)
    clock.t += 61
    assert c.post("/v1/ask", json=Q).status_code == 200
    assert c.post("/v1/ask", json=Q).status_code == 200
    assert f.calls == 2  # baseline + one refresh for both requests
    clock.t += 61
    c.post("/v1/ask", json=Q)
    assert f.calls == 3


def test_key_never_in_responses_or_logs(env, caplog):
    caplog.set_level(logging.DEBUG)
    clock = Clock()
    err = RuntimeError(f"Bearer {DEMO_KEY} rejected")
    for fetcher in (Fetcher(err), Fetcher(1.0, err), Fetcher(1.0, 9.0)):
        c = build(fetcher, clock)
        clock.t += 61
        r = c.post("/v1/ask", json=Q)
        assert DEMO_KEY not in r.text and DEMO_KEY not in str(r.headers)
        clock.t += 61
    assert DEMO_KEY not in caplog.text
    assert "boom" not in caplog.text and "rejected" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_byok_works_when_demo_budget_used_up(env):
    clock = Clock()
    c = build(Fetcher(1.0, 50.0), clock)
    clock.t += 61
    assert used_up(c.post("/v1/ask", json=Q))
    r = c.post("/v1/ask", json=Q, headers={"X-Provider-Key": BYOK_KEY})
    assert r.status_code == 200
