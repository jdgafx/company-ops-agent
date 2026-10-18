"""Abuse and spend limits: per-IP sliding-window rate limiter and a file-backed daily cap."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path


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
    """File-backed request and token counters keyed by UTC date."""

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
            data = json.loads(self.path.read_text())
            if data.get("date") == day:
                return {
                    "date": day,
                    "requests": int(data.get("requests", 0)),
                    "tokens": int(data.get("tokens", 0)),
                }
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        return {"date": day, "requests": 0, "tokens": 0}

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

    def check(self, tokens_needed_estimate: int = 0) -> bool:
        """True if one more request, and the estimated tokens, fit under today's caps."""
        with self._lock:
            d = self._load()
            return (
                d["requests"] < self.request_cap
                and d["tokens"] + tokens_needed_estimate <= self.token_cap
            )

    def record(self, tokens_used: int) -> None:
        """Count one request and its tokens against today."""
        with self._lock:
            d = self._load()
            d["requests"] += 1
            d["tokens"] += max(0, int(tokens_used))
            self._save(d)
