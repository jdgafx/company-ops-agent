import json
import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ops_agent.api import create_app

SETTINGS = SimpleNamespace(model="m", max_steps=8, max_tokens=100)
RESULT = {
    "answer": "42",
    "stop_reason": "end_turn",
    "steps": 1,
    "trace": [],
    "usage": {"input_tokens": 30, "output_tokens": 20},
    "latency_ms": 5,
}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("OPS_RATE_LIMIT_PER_MIN", "OPS_DAILY_REQUEST_CAP", "OPS_DAILY_TOKEN_CAP"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(os, "environ", os.environ.copy())  # per-test isolation


def make(tmp_path, runner=None, **env):
    os.environ.update({k: str(v) for k, v in env.items()})
    app = create_app(SETTINGS, runner=runner or (lambda q: dict(RESULT)),
                     limits_path=tmp_path / "limits.json")
    return TestClient(app)


def ask(c, q="How many open tickets?"):
    return c.post("/v1/ask", json={"question": q})


def test_healthz(tmp_path):
    r = make(tmp_path).get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


@pytest.mark.parametrize("q", ["", "   ", "ab", "x" * 501])
def test_invalid_question_422(tmp_path, q):
    assert ask(make(tmp_path), q).status_code == 422


def test_missing_body_422(tmp_path):
    assert make(tmp_path).post("/v1/ask", json={}).status_code == 422


def test_happy_path_strips_question(tmp_path):
    seen = []
    c = make(tmp_path, runner=lambda q: seen.append(q) or dict(RESULT))
    r = ask(c, "  hello there  ")
    assert r.status_code == 200 and r.json() == RESULT and seen == ["hello there"]


def test_rate_limit_429_retry_after(tmp_path):
    c = make(tmp_path, OPS_RATE_LIMIT_PER_MIN=2)
    assert ask(c).status_code == 200 and ask(c).status_code == 200
    r = ask(c)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


def test_daily_request_cap(tmp_path):
    c = make(tmp_path, OPS_DAILY_REQUEST_CAP=2, OPS_RATE_LIMIT_PER_MIN=100)
    assert ask(c).status_code == 200 and ask(c).status_code == 200
    assert ask(c).status_code == 429


def test_daily_token_cap(tmp_path):
    # allowance 0 makes the estimate max_tokens*max_steps = 800; each run settles to 50
    c = make(tmp_path, OPS_DAILY_TOKEN_CAP=840, OPS_RATE_LIMIT_PER_MIN=100,
             OPS_STEP_INPUT_TOKEN_ALLOWANCE=0)
    assert ask(c).status_code == 200  # 0+800 <= 840
    assert ask(c).status_code == 429  # 50+800 > 840
    assert json.loads((tmp_path / "limits.json").read_text())["tokens"] == 50


def test_runner_exception_500_generic(tmp_path, caplog):
    def boom(q):
        raise RuntimeError("secret-detail")

    r = ask(make(tmp_path, runner=boom))
    assert r.status_code == 500 and r.json() == {"error": "internal_error"}
    assert "secret-detail" not in r.text
    assert "secret-detail" not in caplog.text  # class name only, never exception text
    assert "RuntimeError" in caplog.text


def test_default_runner_persists_tickets_across_requests(tmp_path, monkeypatch):
    """Regression: tickets created via /v1/ask must reach OPS_TICKETS_PATH and ids must increase."""
    import json
    import shutil
    from types import SimpleNamespace as NS

    from ops_agent import api
    from ops_agent.config import Settings

    data = tmp_path / "data"
    shutil.copytree("data", data)
    tickets = tmp_path / "tickets.json"
    monkeypatch.setenv("OPS_DATA_DIR", str(data))
    monkeypatch.setenv("OPS_TICKETS_PATH", str(tickets))

    calls = {"n": 0}

    class Client:
        class messages:  # mimics the SDK attribute path
            @staticmethod
            def create(**kwargs):
                calls["n"] += 1
                if calls["n"] % 2 == 1:  # ask for the ticket, then finish
                    block = NS(type="tool_use", id=f"t{calls['n']}", name="create_ticket",
                               input={"customer_id": "C-1002", "subject": "s", "body": "b",
                                      "confirm": True})
                    return NS(content=[block], stop_reason="tool_use",
                              usage=NS(input_tokens=1, output_tokens=1))
                text = NS(type="text", text="done")
                return NS(content=[text], stop_reason="end_turn",
                          usage=NS(input_tokens=1, output_tokens=1))

    settings = Settings(model="claude-haiku-5-5", max_steps=4, max_tokens=64)
    run = api._default_runner(settings, client=Client())
    run("open a ticket for C-1002, I confirm")
    run("open a ticket for C-1002, I confirm")
    saved = json.loads(tickets.read_text())
    assert [t["id"] for t in saved] == ["T-9001", "T-9002"]


def test_failed_run_is_charged_and_counted(tmp_path):
    def boom(_q):
        raise RuntimeError("upstream down")

    c = make(tmp_path, runner=boom, OPS_RATE_LIMIT_PER_MIN=100, OPS_DAILY_REQUEST_CAP=5)
    assert ask(c).status_code == 500
    saved = json.loads((tmp_path / "limits.json").read_text())
    assert saved["requests"] == 1 and saved["tokens"] > 0
