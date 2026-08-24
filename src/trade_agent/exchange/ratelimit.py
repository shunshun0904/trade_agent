"""Token-bucket rate limiter.

bitbank documents ~10 QUERY/s and ~6 UPDATE/s per user and answers 429 above
that (spec 8: "respect the documented rate limits, back off exponentially").
The limiter is process-local, which is sufficient here: the distributed lock
(spec 8) already guarantees at most one order-placing invocation at a time.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    rate_per_second: float
    capacity: float | None = None
    _tokens: float = field(default=0.0, init=False)
    _updated: float = field(default_factory=time.monotonic, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        if self.capacity is None:
            self.capacity = max(1.0, self.rate_per_second)
        self._tokens = float(self.capacity)

    def acquire(self, tokens: float = 1.0, *, sleep=time.sleep) -> float:
        """Block until `tokens` are available. Returns the seconds waited."""
        with self._lock:
            wait = self._reserve(tokens)
        if wait > 0:
            sleep(wait)
        return wait

    def _reserve(self, tokens: float) -> float:
        now = time.monotonic()
        elapsed = now - self._updated
        self._updated = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_second)
        if self._tokens >= tokens:
            self._tokens -= tokens
            return 0.0
        deficit = tokens - self._tokens
        self._tokens = 0.0
        wait = deficit / self.rate_per_second
        # Charge the wait forward so concurrent callers queue instead of racing.
        self._updated = now + wait
        return wait


class RateLimiter:
    """Two buckets: QUERY (reads) and UPDATE (order / cancel)."""

    def __init__(self, query_per_second: float, update_per_second: float):
        self.query = TokenBucket(query_per_second)
        self.update = TokenBucket(update_per_second)

    def acquire(self, category: str, *, sleep=time.sleep) -> float:
        bucket = self.update if category == "update" else self.query
        return bucket.acquire(sleep=sleep)
