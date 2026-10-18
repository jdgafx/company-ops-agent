"""Eval runner: replays tasks.jsonl through the agent loop and checks each expectation.

Live mode (default) calls the real model and needs ANTHROPIC_API_KEY. --scripted swaps in a
deterministic fake client that issues each task's "scripted" tool calls and answers from the
task's expected fields. It is the regression check for the runner, the checks and the gate
wiring; it does NOT measure model quality.

Each task gets its own Store with a throwaway tickets file, so data/ is never written.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
from datetime import UTC, datetime
from types import SimpleNamespace as NS
from typing import Any

from ops_agent.agent import run_agent
from ops_agent.catalog import build_registry
from ops_agent.config import Settings, make_client
from ops_agent.store import Store

# ESTIMATE only: flat per-million-token prices, not read from any billing source.
EST_INPUT_USD_PER_MTOK = 1.00
EST_OUTPUT_USD_PER_MTOK = 5.00

MISSING_PHRASES = (
    "not found", "no record", "does not", "no order", "no customer", "not in",
    "couldn't find", "could not find", "can't find", "cannot find", "no match",
    "isn't in", "is not in", "no data", "unable to find",
)


class ScriptedClient:
    """Fake Anthropic client: step 1 issues the task's scripted calls, step 2 answers."""

    def __init__(self, tasks: list[dict[str, Any]]):
        self._by_question = {t["question"]: t for t in tasks}
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        messages = kwargs["messages"]
        task = self._by_question[messages[0]["content"]]
        usage = NS(input_tokens=100, output_tokens=20)
        if len(messages) == 1 and task.get("scripted"):
            calls = [
                NS(type="tool_use", id=f"s{i}", name=c["name"], input=c.get("input", {}))
                for i, c in enumerate(task["scripted"])
            ]
            return NS(content=calls, stop_reason="tool_use", usage=usage)
        expect = task["expect"]
        parts = list(expect.get("answer_contains", []))
        if expect.get("must_refuse_or_say_missing"):
            parts.append("not found")
        text = NS(type="text", text="; ".join(parts) or "Done.")
        return NS(content=[text], stop_reason="end_turn", usage=usage)


def wrote(trace: list[dict[str, Any]], write_tools: set[str]) -> bool:
    """True if a write tool succeeded (ok, and not an explicit created=False)."""
    for entry in trace:
        result = entry.get("result") or {}
        created = (result.get("data") or {}).get("created", True)
        if entry["tool"] in write_tools and result.get("ok") and created:
            return True
    return False


def says_missing(answer: str) -> bool:
    low = answer.lower()
    return any(p in low for p in MISSING_PHRASES)


def check_task(expect: dict[str, Any], out: dict[str, Any], write_tools: set[str]) -> list[str]:
    answer = out["answer"]
    low = answer.lower()
    called = {e["tool"] for e in out["trace"]}
    failed = []
    for tool in expect.get("tools_called", []):
        if tool not in called:
            failed.append(f"tool not called: {tool}")
    for s in expect.get("answer_contains", []):
        if s.lower() not in low:
            failed.append(f"answer missing: {s}")
    for s in expect.get("answer_not_contains", []):
        if s.lower() in low:
            failed.append(f"answer contains forbidden: {s}")
    if expect.get("no_write") and wrote(out["trace"], write_tools):
        failed.append("unexpected write")
    if expect.get("must_refuse_or_say_missing") and not says_missing(answer):
        failed.append("answer does not say the record is missing")
    return failed


def run_task(task: dict[str, Any], *, client: Any, settings: Settings, data_dir: pathlib.Path,
             tmp: pathlib.Path) -> dict[str, Any]:
    store = Store(data_dir=data_dir, tickets_path=tmp / f"{task['id']}-tickets.json")
    tools = build_registry(store)
    write_tools = {s.name for s in tools.specs() if s.write}
    try:
        out = run_agent(task["question"], tools=tools, client=client, settings=settings)
    except Exception as exc:  # noqa: BLE001 - one API failure must not abort the whole run
        return {"id": task["id"], "category": task["category"], "passed": False,
                "failed_checks": [f"error: {type(exc).__name__}: {exc}"], "tools": [],
                "latency_ms": 0.0, "input_tokens": 0, "output_tokens": 0, "steps": 0,
                "stop_reason": "error"}
    failed = check_task(task["expect"], out, write_tools)
    return {
        "id": task["id"], "category": task["category"], "passed": not failed,
        "failed_checks": failed, "tools": [e["tool"] for e in out["trace"]],
        "latency_ms": round(out["latency_ms"], 1),
        "input_tokens": out["usage"]["input_tokens"],
        "output_tokens": out["usage"]["output_tokens"],
        "steps": out["steps"], "stop_reason": out["stop_reason"],
    }


def load_tasks(path: pathlib.Path) -> list[dict[str, Any]]:
    lines = path.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def run_all(tasks: list[dict[str, Any]], *, scripted: bool) -> dict[str, Any]:
    data_dir = pathlib.Path(os.environ.get("OPS_DATA_DIR", "data"))
    if scripted:
        settings, client = Settings(), ScriptedClient(tasks)
    else:
        settings = Settings.from_env()
        client = make_client(settings)
    with tempfile.TemporaryDirectory(prefix="ops-eval-") as d:
        results = [run_task(t, client=client, settings=settings, data_dir=data_dir,
                            tmp=pathlib.Path(d)) for t in tasks]
    total = len(results)
    passed = sum(r["passed"] for r in results)
    tin = sum(r["input_tokens"] for r in results)
    tout = sum(r["output_tokens"] for r in results)
    return {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": "scripted" if scripted else settings.model,
        "mode": "scripted" if scripted else "live",
        "pass_rate": passed / total if total else 0.0,
        "passed": passed, "total": total,
        "mean_latency_ms": round(sum(r["latency_ms"] for r in results) / total, 1) if total else 0.0,
        "total_input_tokens": tin, "total_output_tokens": tout,
        "est_cost_usd_estimate_only": round(
            tin * EST_INPUT_USD_PER_MTOK / 1e6 + tout * EST_OUTPUT_USD_PER_MTOK / 1e6, 4),
        "tasks": results,
    }


def print_summary(report: dict[str, Any]) -> None:
    print(f"{'task':<14}{'cat':<12}{'result':<7}{'steps':>5}{'ms':>9}{'tok in/out':>14}")
    for r in report["tasks"]:
        tok = f"{r['input_tokens']}/{r['output_tokens']}"
        print(f"{r['id']:<14}{r['category']:<12}{'PASS' if r['passed'] else 'FAIL':<7}"
              f"{r['steps']:>5}{r['latency_ms']:>9.0f}{tok:>14}")
        for f in r["failed_checks"]:
            print(f"    - {f}")
    print(f"\n{report['mode']} run, model {report['model']}: {report['passed']}/{report['total']} "
          f"passed ({report['pass_rate']:.0%}), mean latency {report['mean_latency_ms']:.0f} ms, "
          f"est. cost ${report['est_cost_usd_estimate_only']:.4f} (estimate)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the ops-agent eval suite.")
    ap.add_argument("--tasks", default="evals/tasks.jsonl")
    ap.add_argument("--gate-file", default="evals/gate.json")
    ap.add_argument("--out", default=None, help="results path (default evals/results-<UTC>.json)")
    ap.add_argument("--gate", action="store_true", help="exit 1 if pass rate < gate threshold")
    ap.add_argument("--scripted", action="store_true", help="deterministic fake client, no network")
    args = ap.parse_args(argv)

    if not args.scripted and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set; refusing to run live. Use --scripted for offline.",
              file=sys.stderr)
        return 2
    report = run_all(load_tasks(pathlib.Path(args.tasks)), scripted=args.scripted)
    out = pathlib.Path(args.out or f"evals/results-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print_summary(report)
    if args.gate:
        threshold = json.loads(pathlib.Path(args.gate_file).read_text())["min_pass_rate"]
        if report["pass_rate"] < threshold:
            print(f"GATE FAIL: pass rate {report['pass_rate']:.2f} < {threshold}")
            return 1
        print(f"GATE PASS: pass rate {report['pass_rate']:.2f} >= {threshold}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
