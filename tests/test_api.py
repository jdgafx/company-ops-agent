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
                               input={"customer_id": "12748", "subject": "s", "body": "b",
                                      "confirm": True})
                    return NS(content=[block], stop_reason="tool_use",
                              usage=NS(input_tokens=1, output_tokens=1))
                text = NS(type="text", text="done")
                return NS(content=[text], stop_reason="end_turn",
                          usage=NS(input_tokens=1, output_tokens=1))

    settings = Settings(model="claude-haiku-5-5", max_steps=4, max_tokens=64)
    run = api._default_runner(settings, client=Client())
    run("open a ticket for 12748, I confirm")
    run("open a ticket for 12748, I confirm")
    saved = json.loads(tickets.read_text())
    assert [t["id"] for t in saved] == ["T-9001", "T-9002"]


def test_failed_run_is_charged_and_counted(tmp_path):
    def boom(_q):
        raise RuntimeError("upstream down")

    c = make(tmp_path, runner=boom, OPS_RATE_LIMIT_PER_MIN=100, OPS_DAILY_REQUEST_CAP=5)
    assert ask(c).status_code == 500
    saved = json.loads((tmp_path / "limits.json").read_text())
    assert saved["requests"] == 1 and saved["tokens"] > 0


def test_about_describes_the_dataset_examples_limits_and_model(tmp_path):
    c = make(tmp_path)
    r = c.get("/v1/about")
    assert r.status_code == 200
    a = r.json()
    assert set(a) == {"name", "description", "dataset", "examples", "limits", "model"}
    assert a["name"] == "company-ops-agent" and a["description"].endswith(".")
    d = a["dataset"]
    assert d["name"] and d["currency"] == "GBP" and d["doi"] == "10.24432/C5CG6D"
    assert d["source_url"].startswith("https://archive.ics.uci.edu/")
    assert d["license"].startswith("CC BY 4.0") and d["date_pulled"] == "2026-10-18"
    assert d["window"] == {"from": "2011-09-01", "to": "2011-11-30"}
    assert d["counts"] == {"customers": 400, "orders": 3089, "products": 2774, "order_lines": 61962}
    assert a["model"] == "m"
    assert a["limits"] == {"per_client_per_hour": 10, "per_day": 200}
    assert len(a["examples"]) == 5 and all(isinstance(q, str) and q for q in a["examples"])


def test_about_limits_follow_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_DEMO_PER_HOUR_PER_CLIENT", "4")
    monkeypatch.setenv("OPS_DEMO_DAILY_REQUESTS", "50")
    assert make(tmp_path).get("/v1/about").json()["limits"] == {
        "per_client_per_hour": 4, "per_day": 50}


def test_about_examples_are_eval_tasks():
    """Every example question is also a recorded eval task, so it is checked against the data."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    tasks = [json.loads(line) for line in (root / "evals/tasks.jsonl").read_text().splitlines()]
    questions = {t["question"]: t for t in tasks}
    meta = json.loads((root / "data/meta.json").read_text())
    for e in meta["examples"]:
        assert questions[e["question"]]["id"] == e["task_id"]
        assert questions[e["question"]]["expect"]["answer_contains"]  # has a checkable answer


def test_ask_returns_cost_and_ui_trace_fields(tmp_path, monkeypatch):
    """Real agent loop over the real data with a scripted model: cost_usd and trace shape."""
    from types import SimpleNamespace as NS

    monkeypatch.setenv("OPS_DEMO_PROVIDER_KEY", "k-test")
    monkeypatch.setenv("OPS_DEMO_LIMITS_PATH", str(tmp_path / "demo.json"))
    monkeypatch.setenv("OPS_DATA_DIR", "data")
    calls = []

    class Client:
        messages = None

        def __init__(self):
            self.messages = self

        def create(self, **kwargs):
            calls.append(1)
            usage = NS(input_tokens=1000, output_tokens=100)
            if len(calls) == 1:
                block = NS(type="tool_use", id="t1", name="get_customer", input={"customer_id": "12748"})
                return NS(content=[block], stop_reason="tool_use", usage=usage)
            return NS(content=[NS(type="text", text="UK")], stop_reason="end_turn", usage=usage)

    from ops_agent.config import Settings

    app = create_app(Settings(), client_factory=lambda key: Client(),
                     limits_path=tmp_path / "limits.json")
    r = TestClient(app).post("/v1/ask", json={"question": "Which country is 12748 in?"})
    assert r.status_code == 200
    body = r.json()
    assert {"answer", "stop_reason", "steps", "trace", "usage", "latency_ms", "cost_usd"} <= set(body)
    assert body["cost_usd"] == pytest.approx(2 * (1000 * 1e-7 + 100 * 5e-7))
    entry = body["trace"][0]
    assert entry["name"] == "get_customer" and entry["input"] == {"customer_id": "12748"}
    assert entry["ok"] is True and entry["summary"] == "customer 12748 (United Kingdom), 80 orders"
