"""The in-progress candle must not reach the indicators.

This was a live defect, and the most consequential one found so far: every
strategist declined every cycle, and all three named the same reason — volume.
The volume they were reading was the volume of an hour that had just started.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from trade_agent.data.indicators import volume_ratio
from trade_agent.data.snapshot import SnapshotBuilder
from trade_agent.models.market import Candle


class _Clock:
    def __init__(self, now: datetime):
        self._now = now

    def now(self) -> datetime:
        return self._now


class _HourlyExchange:
    """24 full hours of steady volume, plus the hour now in progress."""

    FULL_VOLUME = Decimal("4.4")
    PARTIAL_VOLUME = Decimal("0.08")

    def __init__(self, now: datetime):
        self.now = now

    def get_ticker(self) -> dict:
        return {"last": "12435000"}

    def get_depth(self) -> dict:
        return {"asks": [["12436000", "0.5"]], "bids": [["12434000", "0.5"]],
                "asks_over": "0", "bids_under": "0"}

    def get_candles(self, candle_type: str, day) -> list[Candle]:
        step = timedelta(minutes=5) if candle_type == "5min" else timedelta(hours=1)
        current = self.now.replace(minute=0, second=0, microsecond=0)
        if candle_type == "5min":
            current = self.now.replace(
                minute=self.now.minute - self.now.minute % 5,
                second=0, microsecond=0)
        out = []
        for index in range(-40, 1):
            opened = current + index * step
            # The bar at `current` is the one still forming.
            volume = (self.PARTIAL_VOLUME if index == 0 else self.FULL_VOLUME)
            price = Decimal("12435000")
            out.append(Candle(open=price, high=price + 1000, low=price - 1000,
                              close=price, volume=volume, opened_at=opened))
        return out


@pytest.fixture
def now() -> datetime:
    """22 seconds past the hour — when the 30-minute screener actually fires."""
    return datetime(2026, 8, 29, 9, 0, 22, tzinfo=timezone.utc)


def test_the_forming_candle_is_left_out(now):
    from trade_agent.config import get_config

    builder = SnapshotBuilder(_HourlyExchange(now), get_config(), _Clock(now))
    candles = builder._fetch_series("1hour", 24, now, [])

    assert candles, "every candle was dropped"
    newest = max(c.opened_at for c in candles)
    assert newest + timedelta(hours=1) <= now, (
        f"candle opened {newest} is still forming at {now}")
    assert all(c.volume == _HourlyExchange.FULL_VOLUME for c in candles)


def test_volume_ratio_is_not_a_clock(now):
    """With the partial bar included the ratio reports how much of the hour has
    elapsed. The screener runs on the hour and the half hour, so it could never
    reach the 1.5x spike threshold — the trigger was unreachable by
    construction, and the agents read the number as a collapse in liquidity."""
    from trade_agent.config import get_config

    exchange = _HourlyExchange(now)
    builder = SnapshotBuilder(exchange, get_config(), _Clock(now))

    kept = builder._fetch_series("1hour", 24, now, [])
    assert volume_ratio(kept, 20) == Decimal(1), (
        "steady volume must read as 1.0x once the forming bar is gone")

    with_partial = sorted(exchange.get_candles("1hour", now.date()),
                          key=lambda c: c.opened_at)
    assert volume_ratio(with_partial, 20) < Decimal("0.03"), (
        "this is the defect being guarded against; if it no longer "
        "reproduces, the fixture has drifted and the test above is hollow")


def test_the_spike_threshold_is_reachable_again(now):
    """The check that matters to the owner: a real volume spike must now be
    able to cross screening.volume_spike_multiple."""
    from trade_agent.config import get_config

    config = get_config()
    exchange = _HourlyExchange(now)
    builder = SnapshotBuilder(exchange, config, _Clock(now))
    candles = builder._fetch_series("1hour", 24, now, [])

    spiked = candles[:-1] + [candles[-1].model_copy(
        update={"volume": _HourlyExchange.FULL_VOLUME * 2})]
    ratio = volume_ratio(spiked, 20)
    assert ratio is not None and ratio >= config.screening.volume_spike_multiple
