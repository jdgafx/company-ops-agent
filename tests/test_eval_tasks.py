"""The eval tasks must be answerable from the data: expected values come from the tools."""

import json
import pathlib
from collections import Counter

import pytest

from ops_agent.catalog import build_registry
from ops_agent.store import Store

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASKS = [json.loads(line) for line in (ROOT / "evals/tasks.jsonl").read_text().splitlines()]


def test_task_set_shape():
    assert len(TASKS) == 25 and len({t["id"] for t in TASKS}) == 25
    assert Counter(t["category"] for t in TASKS) == {
        "lookup": 8, "multi_step": 6, "absence": 4, "write": 3, "injection": 4}
    assert not any("fictional" in json.dumps(t).lower() for t in TASKS)


@pytest.mark.parametrize("task", [t for t in TASKS if t["category"] in ("lookup", "multi_step")],
                         ids=lambda t: t["id"])
def test_expected_values_appear_in_tool_results(task, tmp_path):
    reg = build_registry(Store(ROOT / "data", tickets_path=tmp_path / "t.json"))
    results = [reg.call(c["name"], c["input"]) for c in task["scripted"]]
    assert all(r["ok"] for r in results)
    seen = json.dumps(results)
    for want in task["expect"]["answer_contains"]:
        assert want in seen, (task["id"], want)
    assert {c["name"] for c in task["scripted"]} >= set(task["expect"]["tools_called"])


@pytest.mark.parametrize("task", [t for t in TASKS if t["category"] == "absence"
                                  and t["scripted"]], ids=lambda t: t["id"])
def test_absent_tasks_really_find_nothing(task, tmp_path):
    reg = build_registry(Store(ROOT / "data", tickets_path=tmp_path / "t.json"))
    for c in task["scripted"]:
        assert reg.call(c["name"], c["input"])["data"]["found"] is False
