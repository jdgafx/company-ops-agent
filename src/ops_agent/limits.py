"""Abuse and spend limits: per-IP sliding-window rate limiter and a file-backed daily cap."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger("ops_agent.limits")


class CapStateError(Exception):
    """The counter file exists but cannot be read or parsed. Callers must fail closed."""


class RateLimiter:
    """In-memory sliding window: at most `limit` requests per `window` seconds per key."""

    def __init__(self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.window = window
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        """Record a hit if allowed. Returns (allowed, retry_after_seconds)."""
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, max(1, int(self.window - (now - hits[0])) + 1)
            hits.append(now)
            if len(self._hits) > 10_000:  # bound memory under many distinct IPs
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window]:
                    del self._hits[k]
            return True, 0


def _utc_today() -> str:
    return datetime.now(UTC).date().isoformat()


class DailyCap:
    """File-backed request and token counters keyed by UTC date.

    Fail closed: a counter file that exists but is unreadable or corrupt denies
    every reservation. Only a missing file means "nothing spent yet".
    Single process only: the lock is in-process, so run one worker.
    """

    def __init__(
        self,
        path: str | Path,
        request_cap: int,
        token_cap: int,
        today: Callable[[], str] = _utc_today,
    ):
        self.path = Path(path)
        self.request_cap = request_cap
        self.token_cap = token_cap
        self._today = today
        self._lock = threading.Lock()

    def _load(self) -> dict:
        day = self._today()
        try:
            raw = self.path.read_text()
        except FileNotFoundError:
            return {"date": day, "requests": 0, "tokens": 0}
        except OSError as exc:
            raise CapStateError(f"cannot read {self.path}: {exc}") from exc
        try:
            data = json.loads(raw)
            requests = int(data["requests"])
            tokens = int(data["tokens"])
            stored_day = data["date"]
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise CapStateError(f"corrupt counter file {self.path}: {exc}") from exc
        if stored_day != day:
            return {"date": day, "requests": 0, "tokens": 0}
        return {"date": day, "requests": requests, "tokens": tokens}

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def reserve(self, estimate: int) -> bool:
        """Count one request and `estimate` tokens BEFORE the paid call.

        Returns False (deny) when a cap would be exceeded, the counter file is
        corrupt, or the counter cannot be saved. Reservation and check are one
        step under the lock, so concurrent requests cannot overshoot the caps.
        """
        estimate = max(0, int(estimate))
        with self._lock:
            try:
                d = self._load()
                if d["requests"] >= self.request_cap or d["tokens"] + estimate > self.token_cap:
                    return False
                d["requests"] += 1
                d["tokens"] += estimate
                self._save(d)
            except (CapStateError, OSError) as exc:
                log.error("daily cap denied request: %s", exc)
                return False
            return True

    def settle(self, estimate: int, actual_tokens: int | None) -> None:
        """Replace a reservation's estimate with the real token count.

        `actual_tokens=None` (the run failed, so usage is unknown) keeps the
        estimate charged. Call once per successful reserve().
        """
        with self._lock:
            try:
                d = self._load()
                actual = max(0, int(actual_tokens)) if actual_tokens is not None else int(estimate)
                d["tokens"] = max(0, d["tokens"] + actual - int(estimate))
                self._save(d)
            except (CapStateError, OSError) as exc:
                # Counter is now unreadable; reserve() will deny until it is repaired.
                log.error("daily cap settle failed: %s", exc)
