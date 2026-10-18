import json
import pathlib
from types import SimpleNamespace as NS

import anthropic
import pytest

from ops_agent import agent, catalog
from ops_agent.config import DEFAULT_BASE_URL, DEFAULT_MODEL, Settings, make_client
from ops_agent.evals import run
from ops_agent.evals.replay import ReplayClient, ReplayMismatch
from ops_agent.store import Store

ROOT = pathlib.Path(__file__).resolve().parents[1]
FAKE_KEY = "test-key-not-a-secret-123"


@pytest.fixture(autouse=True)
def data_dir(monkeypatch):
    monkeypatch.setenv("OPS_DATA_DIR", str(ROOT / "data"))


@pytest.fixture
def tasks():
    rows = run.load_tasks(ROOT / "evals/tasks.jsonl")
    return [t for t in rows if t["id"] in ("lookup-01", "write-01")]


@pytest.fixture
def recorded(tasks, tmp_path):
    """Record two tasks with a stub live client; return the recordings dir."""
    rec = tmp_path / "rec"
    report = run.run_all(tasks, mode="record", client=run.ScriptedClient(tasks), recordings_dir=rec)
    assert report["passed"] == 2
    return rec


def replay(tasks, rec):
    return {t["id"]: t for t in run.run_all(tasks, mode="replay", recordings_dir=rec)["tasks"]}


def test_record_then_replay_passes(tasks, recorded):
    report = run.run_all(tasks, mode="replay", recordings_dir=recorded)
    assert report["mode"] == "replay" and report["passed"] == 2
    assert report["est_cost_usd_estimate_only"] == 0.0
    assert report["model"] == Settings().model
    assert sorted(p.name for p in recorded.iterdir()) == ["lookup-01.json", "write-01.json"]
    data = json.loads((recorded / "write-01.json").read_text())
    assert data["task"] == "write-01" and len(data["steps"]) == 2
    assert set(data["steps"][0]["request"]) == {"model", "max_tokens", "system", "tools", "messages"}
    assert set(data["steps"][0]["response"]) == {"content", "stop_reason", "usage"}


def test_recording_has_no_key_and_fixed_ticket_clock(tasks, tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_AGENT_API_KEY", FAKE_KEY)
    rec = tmp_path / "rec"
    run.run_all(tasks, mode="record", client=run.ScriptedClient(tasks), recordings_dir=rec)
    text = "".join(p.read_text() for p in rec.iterdir())
    assert FAKE_KEY not in text
    assert "2026-01-01T00:00:00+00:00" in text  # created_at from EVAL_CLOCK


def test_sdk_blocks_normalise_like_replayed_blocks(tmp_path):
    """A live client returns SDK model objects; the recording must still replay cleanly."""
    task = {"id": "sdk-01", "question": "Which country is customer 12748 in?", "category": "lookup",
            "expect": {"tools_called": ["get_customer"], "answer_contains": ["United Kingdom"]}}
    usage = NS(input_tokens=5, output_tokens=2)

    class Live:
        def __init__(self):
            self.messages = self

        def create(self, **kwargs):
            if len(kwargs["messages"]) == 1:
                block = anthropic.types.ToolUseBlock(
                    id="tu_1", name="get_customer", input={"customer_id": "12748"},
                    type="tool_use")
                return NS(content=[block], stop_reason="tool_use", usage=usage)
            block = anthropic.types.TextBlock(text="United Kingdom", type="text", citations=None)
            return NS(content=[block], stop_reason="end_turn", usage=usage)

    rec = tmp_path / "rec"
    assert run.run_all([task], mode="record", client=Live(), recordings_dir=rec)["passed"] == 1
    assert run.run_all([task], mode="replay", recordings_dir=rec)["passed"] == 1


def test_changed_tool_output_fails_only_that_task(tasks, recorded, monkeypatch):
    monkeypatch.setattr(
        Store, "get_customer",
        lambda self, customer_id: {"found": True, "customer": {"id": customer_id, "country": "Norway"}},
    )
    res = replay(tasks, recorded)
    assert res["write-01"]["passed"]
    assert not res["lookup-01"]["passed"]
    assert "ReplayMismatch" in res["lookup-01"]["failed_checks"][0]
    assert "step 2" in res["lookup-01"]["failed_checks"][0]


def test_changed_system_prompt_fails(tasks, recorded, monkeypatch):
    monkeypatch.setattr(agent, "SYSTEM_PROMPT", agent.SYSTEM_PROMPT + " Be chatty.")
    res = replay(tasks, recorded)
    assert not any(r["passed"] for r in res.values())
    assert all("ReplayMismatch" in r["failed_checks"][0] and "step 1" in r["failed_checks"][0]
               for r in res.values())


def test_changed_tool_schema_fails(tasks, recorded, monkeypatch):
    orig = catalog._obj
    monkeypatch.setattr(catalog, "_obj", lambda props, req: {**orig(props, req), "title": "changed"})
    res = replay(tasks, recorded)
    assert not any(r["passed"] for r in res.values())
    assert "tools" in res["lookup-01"]["failed_checks"][0]


def test_missing_recording_fails_with_clear_message(tasks, recorded):
    (recorded / "lookup-01.json").unlink()
    res = replay(tasks, recorded)
    assert res["write-01"]["passed"]
    assert not res["lookup-01"]["passed"]
    assert "no recording" in res["lookup-01"]["failed_checks"][0]
    assert "--record" in res["lookup-01"]["failed_checks"][0]


def test_extra_request_kwarg_is_a_mismatch(recorded):
    client = ReplayClient(recorded)
    client.begin("lookup-01")
    request = json.loads((recorded / "lookup-01.json").read_text())["steps"][0]["request"]
    client.messages.create(**request)  # unchanged request still matches
    client.begin("lookup-01")
    with pytest.raises(ReplayMismatch, match="tool_choice"):
        client.messages.create(**request, tool_choice={"type": "any"})


def test_generic_error_text_is_not_written_to_results(tmp_path):
    task = {"id": "err-01", "question": "q", "category": "lookup", "expect": {}}

    class Boom:
        def __init__(self):
            self.messages = self

        def create(self, **kwargs):
            raise RuntimeError("secret-detail-from-provider")

    report = run.run_all([task], mode="live", client=Boom())
    out = tmp_path / "r.json"
    out.write_text(json.dumps(report))
    assert "secret-detail-from-provider" not in out.read_text()
    assert report["tasks"][0]["failed_checks"] == ["error: RuntimeError"]


def test_calling_past_recorded_steps_raises(recorded):
    client = ReplayClient(recorded)
    client.begin("lookup-01")
    steps = json.loads((recorded / "lookup-01.json").read_text())["steps"]
    for step in steps:
        client.messages.create(**step["request"])
    with pytest.raises(ReplayMismatch, match="more model calls"):
        client.messages.create(**steps[0]["request"])


def test_cli_replay_gate_needs_no_key(tasks, recorded, monkeypatch, tmp_path):
    monkeypatch.delenv("OPS_AGENT_API_KEY", raising=False)
    tasks_file = tmp_path / "t.jsonl"
    tasks_file.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"min_pass_rate": 1.0}))
    args = ["--replay", "--gate", "--tasks", str(tasks_file), "--gate-file", str(gate),
            "--recordings-dir", str(recorded), "--out", str(tmp_path / "r.json")]
    assert run.main(args) == 0
    (recorded / "write-01.json").unlink()
    assert run.main(args) == 1


@pytest.mark.parametrize("flag", [[], ["--record"]])
def test_live_without_key_exits_2_naming_only_the_variable(flag, monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("OPS_AGENT_API_KEY", raising=False)
    assert run.main([*flag, "--out", str(tmp_path / "r.json")]) == 2
    err = capsys.readouterr().err
    assert "OPS_AGENT_API_KEY" in err and "ANTHROPIC" not in err


def test_make_client_rejects_empty_key_and_ignores_env(monkeypatch):
    for name in ("OPS_AGENT_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.setenv(name, "from-env")
    with pytest.raises(ValueError):
        make_client(Settings(), "")
    client = make_client(Settings(), FAKE_KEY)
    assert client.auth_token == FAKE_KEY
    assert client.api_key != "from-env"


def test_make_client_uses_base_url_and_defaults(monkeypatch):
    monkeypatch.delenv("OPS_AGENT_BASE_URL", raising=False)
    assert Settings.from_env().base_url == DEFAULT_BASE_URL
    assert Settings.from_env().model == DEFAULT_MODEL
    assert str(make_client(Settings.from_env(), FAKE_KEY).base_url).rstrip("/") == DEFAULT_BASE_URL
    monkeypatch.setenv("OPS_AGENT_BASE_URL", "https://proxy.example.com/api")
    client = make_client(Settings.from_env(), FAKE_KEY)
    assert str(client.base_url).rstrip("/") == "https://proxy.example.com/api"
    monkeypatch.setenv("OPS_AGENT_BASE_URL", "  ")
    assert Settings.from_env().base_url == DEFAULT_BASE_URL


def test_non_https_base_url_rejected(monkeypatch):
    with pytest.raises(ValueError, match="https://"):
        Settings(base_url="http://insecure.example.com")
    monkeypatch.setenv("OPS_AGENT_BASE_URL", "http://insecure.example.com")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_store_clock_is_injectable(tmp_path):
    from datetime import UTC, datetime
    fixed = datetime(2030, 5, 6, 7, 8, 9, tzinfo=UTC)
    store = Store(ROOT / "data", tickets_path=tmp_path / "t.json", clock=lambda: fixed)
    ticket = store.create_ticket("12748", "s", "b")["ticket"]
    assert ticket["created_at"] == "2030-05-06T07:08:09+00:00" and ticket["id"] == "T-9001"
