"""Free demo mode of /v1/ask: demo key, per-client hourly cap, daily request and USD caps."""

import json
import logging
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from ops_agent.api import create_app
from ops_agent.limits import DemoDailyCap

SETTINGS = NS(model="m", max_steps=2, max_tokens=100)
DEMO_KEY = "sk-demo-SECRETSECRETSECRET"
BYOK_KEY = "sk-test-AAAAAAAAAAAAAAAA"
Q = {"question": "How many open tickets are there?"}

ENV = (
    "OPS_DEMO_PROVIDER_KEY", "OPS_DEMO_PER_HOUR_PER_CLIENT", "OPS_DEMO_DAILY_REQUESTS",
    "OPS_DEMO_DAILY_USD", "OPS_DEMO_PRICE_IN_PER_MTOK", "OPS_DEMO_PRICE_OUT_PER_MTOK",
    "OPS_DEMO_LIMITS_PATH", "OPS_RATE_LIMIT_PER_MIN", "OPS_DAILY_REQUEST_CAP",
    "OPS_DAILY_TOKEN_CAP",
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


class Factory:
    def __init__(self):
        self.keys = []

    def __call__(self, api_key):
        self.keys.append(api_key)
        return FakeClient()


@pytest.fixture
def env(monkeypatch, tmp_path):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPS_DEMO_LIMITS_PATH", str(tmp_path / "demo.json"))
    monkeypatch.setenv("OPS_LIMITS_PATH", str(tmp_path / "limits.json"))
    return monkeypatch


def build(factory=None):
    from ops_agent.catalog import build_registry
    from ops_agent.store import Store

    factory = factory or Factory()
    app = create_app(SETTINGS, registry=build_registry(Store("data")), client_factory=factory)
    return TestClient(app), factory


def xff(addr):
    return {"X-Forwarded-For": addr}


def test_demo_serves_keyless_with_demo_key(env):
    env.setenv("OPS_DEMO_PROVIDER_KEY", DEMO_KEY)
    c, factory = build()
    r = c.post("/v1/ask", json=Q)
    assert r.status_code == 200
    assert factory.keys == [DEMO_KEY]


def test_keyless_without_demo_key_is_401(env):
    c, factory = build()
    r = c.post("/v1/ask", json=Q)
    assert r.status_code == 401 and r.json()["error"] == "byok_key_required"
    assert factory.keys == []


def test_hourly_limit_per_client(env):
    env.setenv("OPS_DEMO_PROVIDER_KEY", DEMO_KEY)
    c, _ = build()
    for _ in range(3):
        assert c.post("/v1/ask", json=Q, headers=xff("1.1.1.1")).status_code == 200
    r = c.post("/v1/ask", json=Q, headers=xff("1.1.1.1"))
    assert r.status_code == 429 and r.json()["error"] == "demo_hourly_limit"
    assert int(r.headers["Retry-After"]) >= 1
    assert "X-Provider-Key" in r.json()["detail"]
    assert c.post("/v1/ask", json=Q, headers=xff("2.2.2.2")).status_code == 200


def test_spoofed_leftmost_forwarded_for_shares_bucket(env):
    env.setenv("OPS_DEMO_PROVIDER_KEY", DEMO_KEY)
    env.setenv("OPS_DEMO_PER_HOUR_PER_CLIENT", "1")
    c, _ = build()
    assert c.post("/v1/ask", json=Q, headers=xff("9.9.9.1, 5.5.5.5")).status_code == 200
    r = c.post("/v1/ask", json=Q, headers=xff("9.9.9.2, 5.5.5.5"))
    assert r.status_code == 429 and r.json()["error"] == "demo_hourly_limit"
    assert c.post("/v1/ask", json=Q, headers=xff("9.9.9.1, 6.6.6.6")).status_code == 200


def test_per_minute_limiter_uses_same_identity(env):
    env.setenv("OPS_RATE_LIMIT_PER_MIN", "1")
    c, _ = build()
    h = {"X-Provider-Key": BYOK_KEY}
    assert c.post("/v1/ask", json=Q, headers={**h, **xff("7.7.7.1, 5.5.5.5")}).status_code == 200
    r = c.post("/v1/ask", json=Q, headers={**h, **xff("7.7.7.2, 5.5.5.5")})
    assert r.status_code == 429 and r.json()["error"] == "rate_limited"
    assert c.post("/v1/ask", json=Q, headers={**h, **xff("7.7.7.1, 6.6.6.6")}).status_code == 200


def test_daily_request_cap(env):
    env.setenv("OPS_DEMO_PROVIDER_KEY", DEMO_KEY)
    env.setenv("OPS_DEMO_DAILY_REQUESTS", "1")
    c, _ = build()
    assert c.post("/v1/ask", json=Q, headers=xff("1.1.1.1")).status_code == 200
    r = c.post("/v1/ask", json=Q, headers=xff("2.2.2.2"))
    assert r.status_code == 429 and r.json()["error"] == "demo_daily_limit_reached"
    assert "daily limit" in r.json()["detail"]


def test_daily_usd_cap(env):
    env.setenv("OPS_DEMO_PROVIDER_KEY", DEMO_KEY)
    env.setenv("OPS_DEMO_DAILY_USD", "0.000001")
    c, factory = build()
    r = c.post("/v1/ask", json=Q)
    assert r.status_code == 429 and r.json()["error"] == "demo_daily_limit_reached"
    assert factory.keys == []


def test_usd_settles_to_actual_usage(env, tmp_path):
    env.setenv("OPS_DEMO_PROVIDER_KEY", DEMO_KEY)
    c, _ = build()
    assert c.post("/v1/ask", json=Q).status_code == 200
    data = json.loads((tmp_path / "demo.json").read_text())
    # 3 input tokens * 0.10 + 2 output tokens * 0.50 = 1.3 micro-USD, rounded up
    assert data["requests"] == 1 and data["micro_usd"] == 2


@pytest.mark.parametrize("content", ["{not json", '{"date": "x"}'])
def test_corrupt_counter_fails_closed(env, tmp_path, content):
    env.setenv("OPS_DEMO_PROVIDER_KEY", DEMO_KEY)
    (tmp_path / "demo.json").write_text(content)
    c, factory = build()
    r = c.post("/v1/ask", json=Q)
    assert r.status_code == 429 and r.json()["error"] == "demo_daily_limit_reached"
    assert factory.keys == []


def test_unreadable_counter_fails_closed(tmp_path):
    (tmp_path / "demo.json").mkdir()  # reading a directory raises OSError
    cap = DemoDailyCap(tmp_path / "demo.json", 5, 1_000_000, today=lambda: "d")
    assert not cap.reserve(1)


def test_demo_cap_failed_run_stays_charged(tmp_path):
    cap = DemoDailyCap(tmp_path / "d.json", 5, 1_000_000, today=lambda: "d")
    assert cap.reserve(300)
    cap.settle(300, None)
    assert json.loads((tmp_path / "d.json").read_text())["micro_usd"] == 300


def test_demo_key_never_leaks(env, caplog):
    env.setenv("OPS_DEMO_PROVIDER_KEY", DEMO_KEY)
    env.setenv("OPS_DEMO_PER_HOUR_PER_CLIENT", "1")
    c, _ = build()
    with caplog.at_level(logging.DEBUG):
        bodies = [c.post("/v1/ask", json=Q).text, c.post("/v1/ask", json=Q).text]
    assert DEMO_KEY not in "".join(bodies)
    assert DEMO_KEY not in caplog.text


def test_demo_key_not_leaked_on_client_failure(env, caplog):
    env.setenv("OPS_DEMO_PROVIDER_KEY", DEMO_KEY)

    def bad_factory(key):
        raise ValueError(f"bad client for {key}")

    c, _ = build(bad_factory)
    with caplog.at_level(logging.DEBUG):
        r = c.post("/v1/ask", json=Q)
    assert r.status_code == 502 and DEMO_KEY not in r.text
    assert DEMO_KEY not in caplog.text


def test_byok_does_not_touch_demo_counters(env, tmp_path):
    env.setenv("OPS_DEMO_PROVIDER_KEY", DEMO_KEY)
    c, factory = build()
    r = c.post("/v1/ask", json=Q, headers={"X-Provider-Key": BYOK_KEY})
    assert r.status_code == 200
    assert factory.keys == [BYOK_KEY]
    assert not (tmp_path / "demo.json").exists()


def test_demo_502_keeps_reservation_charged(env, tmp_path):
    env.setenv("OPS_DEMO_PROVIDER_KEY", DEMO_KEY)

    def bad_factory(key):
        raise ValueError("boom")

    c, _ = build(bad_factory)
    assert c.post("/v1/ask", json=Q).status_code == 502
    assert json.loads((tmp_path / "demo.json").read_text())["micro_usd"] > 0
