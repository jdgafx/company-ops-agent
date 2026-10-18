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
    assert cap.reserve(10)
    cap.settle(10, 40)  # real usage replaces the estimate
    assert cap.reserve(60)  # 40 + 60 fits exactly
    cap.settle(60, 60)
    assert not cap.reserve(1)  # token cap hit at 100
    assert json.loads((tmp_path / "l.json").read_text())["tokens"] == 100


def test_request_cap_counts_reservations(tmp_path):
    cap = DailyCap(tmp_path / "l.json", 2, 10_000, today=lambda: "d")
    assert cap.reserve(1) and cap.reserve(1)
    assert not cap.reserve(1)  # request cap hit even before any settle


def test_failed_run_keeps_estimate_charged(tmp_path):
    cap = DailyCap(tmp_path / "l.json", 5, 1000, today=lambda: "d")
    assert cap.reserve(300)
    cap.settle(300, None)  # usage unknown after a failure
    assert json.loads((tmp_path / "l.json").read_text())["tokens"] == 300


def test_resets_on_new_utc_date(tmp_path):
    day = ["2026-01-01"]
    cap = DailyCap(tmp_path / "l.json", 1, 100, today=lambda: day[0])
    assert cap.reserve(5)
    assert not cap.reserve(0)
    day[0] = "2026-01-02"
    assert cap.reserve(0)


def test_persists_across_instances(tmp_path):
    p = tmp_path / "sub" / "l.json"
    assert DailyCap(p, 1, 100, today=lambda: "d").reserve(7)
    again = DailyCap(p, 1, 100, today=lambda: "d")
    assert not again.reserve(0)
    assert json.loads(p.read_text())["tokens"] == 7


def test_corrupt_file_fails_closed(tmp_path):
    p = tmp_path / "l.json"
    p.write_text("{not json")
    cap = DailyCap(p, 100, 100, today=lambda: "d")
    assert not cap.reserve(0)  # a corrupt counter must not reset to zero
    assert p.read_text() == "{not json"  # and is not overwritten


def test_missing_file_is_fresh_start(tmp_path):
    assert DailyCap(tmp_path / "none.json", 1, 100, today=lambda: "d").reserve(0)


def test_unreadable_file_fails_closed(tmp_path):
    p = tmp_path / "dir.json"
    p.mkdir()  # reading a directory raises OSError
    assert not DailyCap(p, 1, 100, today=lambda: "d").reserve(0)


def test_save_failure_denies(tmp_path, monkeypatch):
    cap = DailyCap(tmp_path / "l.json", 5, 100, today=lambda: "d")

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    assert not cap.reserve(1)


def test_concurrent_reservations_never_overshoot(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    cap = DailyCap(tmp_path / "l.json", 1000, 1000, today=lambda: "d")
    with ThreadPoolExecutor(max_workers=16) as pool:
        granted = sum(pool.map(lambda _: cap.reserve(100), range(40)))
    assert granted == 10  # exactly 10 x 100 tokens fit under 1000
    assert json.loads((tmp_path / "l.json").read_text())["tokens"] == 1000


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
        assert cap.reserve(1)
        assert json.loads(p.read_text())["requests"] == len(calls)
    assert len(calls) == 3 and all(d == p for _, d in calls)
    assert [f.name for f in tmp_path.iterdir()] == ["l.json"]
