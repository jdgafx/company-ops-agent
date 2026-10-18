"""Bring-your-own-key behaviour of /v1/ask. Offline: a fake client factory replaces make_client."""

import json
import logging
import os
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from ops_agent.api import create_app

SETTINGS = NS(model="m", max_steps=2, max_tokens=100)
KEY_A = "sk-test-AAAAAAAAAAAAAAAA"
KEY_B = "sk-test-BBBBBBBBBBBBBBBB"


def _reply():
    return NS(
        content=[NS(type="text", text="done")],
        stop_reason="end_turn",
        usage=NS(input_tokens=3, output_tokens=2),
    )


class FakeClient:
    def __init__(self, error=None):
        self.error = error
        self.messages = self

    def create(self, **kwargs):
        if self.error:
            raise self.error
        return _reply()


class Factory:
    def __init__(self, error_with_key=False, build_error=False):
        self.keys = []
        self.error_with_key = error_with_key
        self.build_error = build_error

    def __call__(self, api_key):
        self.keys.append(api_key)
        if self.build_error:
            raise ValueError(f"bad client for {api_key}")
        err = RuntimeError(f"upstream rejected {api_key}") if self.error_with_key else None
        return FakeClient(err)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("OPS_RATE_LIMIT_PER_MIN", "OPS_DAILY_REQUEST_CAP", "OPS_DAILY_TOKEN_CAP"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(os, "environ", os.environ.copy())


def build(tmp_path, factory, **env):
    os.environ.update({k: str(v) for k, v in env.items()})
    reg = _registry()
    app = create_app(SETTINGS, registry=reg, limits_path=tmp_path / "limits.json",
                     client_factory=factory)
    return TestClient(app)


def _registry():
    from ops_agent.catalog import build_registry
    from ops_agent.store import Store

    return build_registry(Store("data"))


def post(c, key=None, q="How many open tickets?"):
    headers = {} if key is None else {"X-Provider-Key": key}
    return c.post("/v1/ask", json={"question": q}, headers=headers)


def limits(tmp_path):
    p = tmp_path / "limits.json"
    return json.loads(p.read_text()) if p.exists() else {}


def test_missing_header_401_costs_nothing(tmp_path):
    f = Factory()
    c = build(tmp_path, f, OPS_RATE_LIMIT_PER_MIN=1)
    r = post(c)
    assert r.status_code == 401
    assert r.json() == {
        "error": "byok_key_required",
        "detail": "send your own provider key in the X-Provider-Key header",
    }
    assert f.keys == []
    assert limits(tmp_path).get("requests", 0) == 0 and limits(tmp_path).get("tokens", 0) == 0
    # rate limiter untouched: the single allowed request still passes
    assert post(c, KEY_A).status_code == 200


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_or_blank_header_401(tmp_path, value):
    assert post(build(tmp_path, Factory()), value).status_code == 401


@pytest.mark.parametrize(
    "value", ["a" * 15, "a" * 257, "sk-test-with space-0123456789", "sk-test-tab\t-0123456789"]
)
def test_bad_shape_400_not_echoed(tmp_path, value):
    f = Factory()
    r = post(build(tmp_path, f), value)
    assert r.status_code == 400 and r.json() == {"error": "invalid_key_format"}
    assert value.strip() not in r.text and f.keys == []


def test_valid_key_200_and_stripped(tmp_path):
    f = Factory()
    r = post(build(tmp_path, f), f"  {KEY_A}  ")
    assert r.status_code == 200 and r.json()["answer"] == "done" and "trace" in r.json()
    assert f.keys == [KEY_A]


def test_key_does_not_carry_over(tmp_path):
    f = Factory()
    c = build(tmp_path, f, OPS_RATE_LIMIT_PER_MIN=100)
    assert post(c, KEY_A).status_code == 200
    assert post(c).status_code == 401
    assert post(c, KEY_B).status_code == 200
    assert f.keys == [KEY_A, KEY_B]
    assert KEY_A not in (tmp_path / "limits.json").read_text()


def test_failing_model_call_502_without_key(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    f = Factory(error_with_key=True)
    r = post(build(tmp_path, f), KEY_A)
    assert r.status_code == 502 and r.json() == {"error": "model_call_failed"}
    assert KEY_A not in r.text and KEY_A not in caplog.text
    assert "RuntimeError" in caplog.text
    assert limits(tmp_path)["requests"] == 1  # a failed call is still charged


def test_failing_client_build_502_without_key(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    r = post(build(tmp_path, Factory(build_error=True)), KEY_A)
    assert r.status_code == 502 and r.json() == {"error": "model_call_failed"}
    assert KEY_A not in r.text and KEY_A not in caplog.text


def test_key_never_logged(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    c = build(tmp_path, Factory(), OPS_RATE_LIMIT_PER_MIN=100)
    assert post(c, KEY_A).status_code == 200
    c2 = build(tmp_path, Factory(error_with_key=True), OPS_RATE_LIMIT_PER_MIN=100)
    assert post(c2, KEY_B).status_code == 502
    assert KEY_A not in caplog.text and KEY_B not in caplog.text


def test_daily_cap_still_429_with_valid_key(tmp_path):
    f = Factory()
    c = build(tmp_path, f, OPS_DAILY_REQUEST_CAP=1, OPS_RATE_LIMIT_PER_MIN=100)
    assert post(c, KEY_A).status_code == 200
    r = post(c, KEY_A)
    assert r.status_code == 429 and r.json() == {"error": "daily_cap_reached"}
    assert f.keys == [KEY_A]  # no client was built for the capped request


def test_rate_limit_still_applies_with_key(tmp_path):
    c = build(tmp_path, Factory(), OPS_RATE_LIMIT_PER_MIN=1)
    assert post(c, KEY_A).status_code == 200
    assert post(c, KEY_A).status_code == 429


def test_healthz_needs_no_key(tmp_path):
    assert build(tmp_path, Factory()).get("/healthz").json() == {"status": "ok"}


def test_non_model_error_500_logs_class_name_only(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)

    def boom(_q):
        raise KeyError(f"lookup failed for {KEY_A}")

    os.environ["OPS_RATE_LIMIT_PER_MIN"] = "100"
    app = create_app(SETTINGS, runner=boom, limits_path=tmp_path / "limits.json")
    r = post(TestClient(app), KEY_A)
    assert r.status_code == 500 and r.json() == {"error": "internal_error"}
    assert KEY_A not in r.text and KEY_A not in caplog.text
    assert "agent run failed: KeyError" in caplog.text
    assert not any(rec.exc_info for rec in caplog.records)
