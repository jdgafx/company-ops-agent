import json
import pathlib

import pytest

from ops_agent.evals import run

ROOT = pathlib.Path(__file__).resolve().parents[1]
WRITE = {"create_ticket"}


@pytest.fixture(autouse=True)
def data_dir(monkeypatch):
    monkeypatch.setenv("OPS_DATA_DIR", str(ROOT / "data"))


def write_gate(tmp_path, rate=0.8):
    p = tmp_path / "gate.json"
    p.write_text(json.dumps({"min_pass_rate": rate}))
    return p


def test_scripted_full_suite_passes_and_writes_report(tmp_path):
    out = tmp_path / "r.json"
    code = run.main(["--scripted", "--gate", "--tasks", str(ROOT / "evals/tasks.jsonl"),
                     "--gate-file", str(write_gate(tmp_path)), "--out", str(out)])
    assert code == 0
    report = json.loads(out.read_text())
    assert report["mode"] == "scripted" and report["pass_rate"] == 1.0
    assert report["total"] >= 25 and "est_cost_usd_estimate_only" in report
    assert (ROOT / "data/tickets.json").read_text().strip() == "[]"


def test_broken_expectation_fails_gate(tmp_path):
    task = {"id": "x", "question": "Status of O-5002?", "category": "lookup", "scripted": [],
            "expect": {"tools_called": ["get_order"], "answer_contains": ["delivered"]}}
    tasks = tmp_path / "t.jsonl"
    tasks.write_text(json.dumps(task) + "\n")
    out = tmp_path / "r.json"
    args = ["--scripted", "--tasks", str(tasks), "--out", str(out)]
    assert run.main(args) == 0
    assert json.loads(out.read_text())["pass_rate"] < 1
    assert run.main([*args, "--gate", "--gate-file", str(write_gate(tmp_path))]) == 1


def test_live_without_key_exits_2(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("OPS_AGENT_API_KEY", raising=False)
    assert run.main(["--out", str(tmp_path / "r.json")]) == 2
    assert "OPS_AGENT_API_KEY" in capsys.readouterr().err


def entry(tool, result):
    return {"tool": tool, "result": result}


def test_wrote_detection():
    ok = entry("create_ticket", {"ok": True, "data": {"created": True}})
    assert run.wrote([ok], WRITE)
    assert not run.wrote([entry("create_ticket", {"ok": False, "error": {}})], WRITE)
    assert not run.wrote([entry("create_ticket", {"ok": True, "data": {"created": False}})], WRITE)
    assert not run.wrote([entry("get_order", {"ok": True, "data": {}})], WRITE)


def test_says_missing_phrases():
    assert run.says_missing("Order O-1 was Not Found.")
    assert run.says_missing("There is no customer with that id")
    assert not run.says_missing("The order is delivered.")


def test_check_task_reports_each_failure():
    out = {"answer": "Denver", "trace": [entry("create_ticket", {"ok": True, "data": {}})]}
    expect = {"tools_called": ["get_customer"], "answer_contains": ["Austin"],
              "answer_not_contains": ["denver"], "no_write": True}
    failed = run.check_task(expect, out, WRITE)
    assert len(failed) == 4
