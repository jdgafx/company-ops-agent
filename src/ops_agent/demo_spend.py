"""Total-spend stop for the free demo key, measured from the key's own usage at OpenRouter."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import urllib.request
from collections.abc import Callable

log = logging.getLogger("ops_agent.demo_spend")

USAGE_URL = "https://openrouter.ai/api/v1/key"


def openrouter_usage_fetcher(key: str, timeout: float = 10.0) -> Callable[[], float]:
    """Build a fetcher returning the key's total USD usage. Raises on any failure."""

    def fetch() -> float:
        req = urllib.request.Request(
            USAGE_URL, headers={"Authorization": f"Bearer {key}"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            usage = float(json.load(resp)["data"]["usage"])
        if not math.isfinite(usage) or usage < 0:
            raise ValueError("bad usage value")
        return usage

    return fetch


class DemoSpendStop:
    """Deny the demo once key usage has grown by `stop_usd` since startup. Fails closed.

    The baseline is read at construction. If that read fails the stop is 'unavailable'
    and denies for the life of the process. Later fetches are cached for `cache_seconds`;
    a failed fetch denies. Only exception class names are logged, never text or the key.
    """

    def __init__(
        self,
        usage_fetcher: Callable[[], float],
        stop_usd: float = 1.00,
        cache_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._fetch = usage_fetcher
        self.stop_usd = stop_usd
        self._cache_seconds = cache_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._baseline: float | None = None
        self._usage: float | None = None
        self._fetched_at = 0.0
        try:
            self._baseline = self._usage = float(usage_fetcher())
            self._fetched_at = clock()
        except Exception as exc:  # noqa: BLE001
            log.error("demo spend baseline unavailable: %s", type(exc).__name__)

    @property
    def available(self) -> bool:
        return self._baseline is not None

    def _refresh(self) -> None:
        now = self._clock()
        if now - self._fetched_at < self._cache_seconds:
            return
        self._fetched_at = now  # failures are cached too, so a down API is not hammered
        try:
            self._usage = float(self._fetch())
        except Exception as exc:  # noqa: BLE001
            self._usage = None
            log.error("demo spend fetch failed: %s", type(exc).__name__)

    def spent(self) -> float | None:
        """Current spend since startup in USD, or None when unknown."""
        if self._baseline is None:
            return None
        with self._lock:
            self._refresh()
            return None if self._usage is None else self._usage - self._baseline

    def allow(self) -> bool:
        spent = self.spent()
        if spent is None:
            return False
        if spent >= self.stop_usd:
            log.warning("demo spend stop reached: %.4f USD", spent)
            return False
        return True
