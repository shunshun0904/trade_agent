"""A deterministic offline market.

`trade-agent --local` uses this so a full decision cycle can be run with no
network, no AWS account and no API key. It is a development and demonstration
aid — it is never wired up in the deployed stack, and nothing about its price
series is a claim about how BTC behaves.

The series is a seeded random walk, so two runs with the same seed produce the
same market and the same decisions. That is the property that makes it useful:
a reproducible cycle to read the logs of.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from decimal import Decimal

from ..models.market import Candle
from ..money import dec
from ..timeutil import UTC
from .base import Balance, PairSettings

STEP_MINUTES = {"1min": 1, "5min": 5, "15min": 15, "30min": 30, "1hour": 60,
                "4hour": 240, "8hour": 480, "12hour": 720, "1day": 1440}


class SyntheticMarket:
    """Public-surface-only exchange. Pair it with `PaperExchange` for orders."""

    def __init__(self, *, pair: str = "btc_jpy", start_price=Decimal(15_000_000),
                 seed: int = 20260302, volatility_pct: Decimal = Decimal("0.35"),
                 now=None):
        self.pair = pair
        self.start_price = dec(start_price)
        self.seed = seed
        self.volatility_pct = dec(volatility_pct)
        self._now = now or (lambda: datetime.now(UTC))

    # -- price path --------------------------------------------------------

    def _walk(self, anchor: datetime, count: int, step_minutes: int) -> list[Candle]:
        """Deterministic for a given anchor: the same bucket always replays."""
        rng = random.Random(f"{self.seed}:{anchor.isoformat()}:{step_minutes}")
        price = self.start_price
        # Warm the walk so different buckets do not all start at the same level.
        drift_seed = random.Random(f"{self.seed}:{anchor.date().isoformat()}")
        price += Decimal(drift_seed.randint(-300_000, 300_000))

        band = self.start_price * self.volatility_pct / Decimal(100)
        candles: list[Candle] = []
        for index in range(count):
            move = Decimal(rng.randint(-100, 100)) * band / Decimal(100)
            close = price + move
            high = max(price, close) + Decimal(rng.randint(0, 60)) * band / Decimal(100)
            low = min(price, close) - Decimal(rng.randint(0, 60)) * band / Decimal(100)
            candles.append(Candle(
                open=_round(price), high=_round(high), low=_round(low),
                close=_round(close),
                volume=dec(rng.randint(50, 400)) / Decimal(100),
                opened_at=anchor + timedelta(minutes=step_minutes * index)))
            price = close
        return candles

    def get_candles(self, candle_type: str, day: date | int) -> list[Candle]:
        step = STEP_MINUTES.get(candle_type, 60)
        now = self._now()
        if isinstance(day, int):
            anchor = datetime(day, 1, 1, tzinfo=UTC)
            count = min(int(365 * 24 * 60 / step), 400)
        else:
            anchor = datetime(day.year, day.month, day.day, tzinfo=UTC)
            count = int(24 * 60 / step)
        candles = self._walk(anchor, count, step)
        return [c for c in candles if c.opened_at <= now]

    @property
    def _last(self) -> Decimal:
        candles = self.get_candles("5min", self._now().date())
        if candles:
            return candles[-1].close
        return self.start_price

    def get_ticker(self) -> dict:
        last = self._last
        day = self.get_candles("1hour", self._now().date()) or []
        return {
            "last": str(last),
            "buy": str(_round(last - Decimal(1000))),
            "sell": str(_round(last + Decimal(1000))),
            "high": str(max((c.high for c in day), default=last)),
            "low": str(min((c.low for c in day), default=last)),
            "open": str(day[0].open if day else last),
            "vol": str(sum((c.volume for c in day), Decimal(0))),
            "timestamp": int(self._now().timestamp() * 1000),
        }

    def get_depth(self) -> dict:
        last = self._last
        bid, ask = _round(last - Decimal(1000)), _round(last + Decimal(1000))
        return {
            "asks": [[str(ask + Decimal(i) * 1000), "0.08"] for i in range(20)],
            "bids": [[str(bid - Decimal(i) * 1000), "0.08"] for i in range(20)],
            "asks_over": "0", "bids_under": "0",
            "timestamp": int(self._now().timestamp() * 1000),
        }

    def get_transactions(self, day: date | None = None) -> list[dict]:
        return []

    def get_circuit_break_info(self) -> dict:
        return {"mode": "NONE", "fee_type": "NORMAL"}

    def get_pair_settings(self) -> PairSettings:
        """The spec's documented constants. `verify-pair` against the real
        exchange is the check that matters; this only keeps offline runs going.
        """
        return PairSettings(
            name=self.pair, min_order_btc=Decimal("0.0001"), price_digits=0,
            amount_digits=8, maker_fee_rate=Decimal("-0.0002"),
            taker_fee_rate=Decimal("0.0012"))

    def get_balances(self) -> dict[str, Balance]:  # pragma: no cover - unused
        raise NotImplementedError(
            "SyntheticMarket has no account; wrap it in PaperExchange")


def _round(value: Decimal) -> Decimal:
    return dec(value).quantize(Decimal(1))
