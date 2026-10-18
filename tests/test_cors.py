import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ops_agent.api import create_app

SETTINGS = SimpleNamespace(model="m", max_steps=8, max_tokens=100)
ALLOWED = "https://company-ops-static.onrender.com"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("OPS_CORS_ORIGINS", raising=False)
    monkeypatch.setattr(os, "environ", os.environ.copy())


def make(tmp_path):
    app = create_app(SETTINGS, runner=lambda q: {}, limits_path=tmp_path / "limits.json")
    return TestClient(app)


def test_allowed_origin_is_echoed(tmp_path):
    os.environ["OPS_CORS_ORIGINS"] = f"{ALLOWED}, https://other.example"
    r = make(tmp_path).get("/healthz", headers={"Origin": ALLOWED})
    assert r.headers["access-control-allow-origin"] == ALLOWED


def test_foreign_origin_gets_no_header(tmp_path):
    os.environ["OPS_CORS_ORIGINS"] = ALLOWED
    r = make(tmp_path).get("/healthz", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_unset_env_means_no_cors_headers(tmp_path):
    r = make(tmp_path).get("/healthz", headers={"Origin": ALLOWED})
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers


def test_preflight_allows_key_header_for_allowed_origin(tmp_path):
    os.environ["OPS_CORS_ORIGINS"] = ALLOWED
    r = make(tmp_path).options(
        "/v1/ask",
        headers={
            "Origin": ALLOWED,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-provider-key",
        },
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == ALLOWED


def test_wildcard_is_ignored(tmp_path):
    os.environ["OPS_CORS_ORIGINS"] = "*"
    r = make(tmp_path).get("/healthz", headers={"Origin": ALLOWED})
    assert "access-control-allow-origin" not in r.headers
