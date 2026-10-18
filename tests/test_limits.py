import json
import os
from pathlib import Path

from ops_agent.limits import DailyCap, RateLimiter


def test_rate_limiter_window():
    t = [0.0]
    rl = RateLimiter(2, 60, clock=lambda: t[0])
    assert rl.allow("a")[0] and rl.allow("a")[0]
    ok, retry = rl.allow("a")
    assert not ok and retry >= 1
    assert rl.allow("b")[0]
    t[0] = 61
    assert rl.allow("a")[0]


def test_daily_cap_requests_and_tokens(tmp_path):
    cap = DailyCap(tmp_path / "l.json", 2, 100, today=lambda: "2026-01-01")
    assert cap.check(10)
    cap.record(40)
    assert cap.check(60) and not cap.check(61)
    cap.record(10)
    assert not cap.check(0)  # request cap hit


def test_resets_on_new_utc_date(tmp_path):
    day = ["2026-01-01"]
    cap = DailyCap(tmp_path / "l.json", 1, 100, today=lambda: day[0])
    cap.record(5)
    assert not cap.check(0)
    day[0] = "2026-01-02"
    assert cap.check(0)


def test_persists_across_instances(tmp_path):
    p = tmp_path / "sub" / "l.json"
    DailyCap(p, 1, 100, today=lambda: "d").record(7)
    again = DailyCap(p, 1, 100, today=lambda: "d")
    assert not again.check(0)
    assert json.loads(p.read_text())["tokens"] == 7


def test_corrupt_file_treated_as_empty(tmp_path):
    p = tmp_path / "l.json"
    p.write_text("{not json")
    assert DailyCap(p, 1, 100, today=lambda: "d").check(0)


def test_atomic_write_uses_replace(tmp_path, monkeypatch):
    p = tmp_path / "l.json"
    calls = []
    real = os.replace

    def spy(src, dst):
        json.loads(Path(src).read_text())  # temp file is complete before it is swapped in
        calls.append((src, dst))
        real(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    cap = DailyCap(p, 5, 100, today=lambda: "d")
    for _ in range(3):
        cap.record(1)
        assert json.loads(p.read_text())["requests"] == len(calls)
    assert len(calls) == 3 and all(d == p for _, d in calls)
    assert [f.name for f in tmp_path.iterdir()] == ["l.json"]
