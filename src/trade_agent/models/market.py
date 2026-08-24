"""MarketSnapshot — the only thing an LLM ever sees about the market.

Spec 3: every quantitative value is computed in Python and handed to the model
as a settled fact. Models interpret; they never calculate. The snapshot is also
the reference the deterministic guard checks quoted numbers against (spec 5).
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from ..money import dec
from ..timeutil import iso


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Candle(_Model):
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    opened_at: datetime

    @classmethod
    def from_bitbank(cls, row: list) -> "Candle":
        """bitbank OHLCV row: [open, high, low, close, volume, timestamp_ms]."""
        from ..timeutil import from_ms

        return cls(
            open=dec(row[0]),
            high=dec(row[1]),
            low=dec(row[2]),
            close=dec(row[3]),
            volume=dec(row[4]),
            opened_at=from_ms(row[5]),
        )


class OrderBookSummary(_Model):
    """Aggregate shape of the book. Raw levels never reach the LLM."""

    best_bid: Decimal
    best_ask: Decimal
    spread_jpy: Decimal
    spread_pct: Decimal
    bid_depth_btc: Decimal = Field(description="cumulative size over the sampled bid levels")
    ask_depth_btc: Decimal
    imbalance: Decimal = Field(description="(bid-ask)/(bid+ask) over sampled depth, -1..1")


class Indicators(_Model):
    """Deterministic technical indicators. All fields optional: with too few
    candles we say "unknown" rather than invent a value."""

    sma_short: Decimal | None = None
    sma_long: Decimal | None = None
    ema_short: Decimal | None = None
    ema_long: Decimal | None = None
    rsi: Decimal | None = None
    atr: Decimal | None = None
    atr_pct: Decimal | None = None
    bb_upper: Decimal | None = None
    bb_lower: Decimal | None = None
    bb_width_pct: Decimal | None = None
    vwap_24h: Decimal | None = None
    vwap_deviation_pct: Decimal | None = None
    high_24h: Decimal | None = None
    low_24h: Decimal | None = None
    change_24h_pct: Decimal | None = None
    change_15m_pct: Decimal | None = None
    volume_24h_btc: Decimal | None = None
    volume_ratio: Decimal | None = Field(
        default=None, description="latest candle volume / mean of prior candles")
    realized_vol_pct: Decimal | None = None


class AccountState(_Model):
    """Balances and the open position, as settled facts."""

    equity_jpy: Decimal
    jpy_free: Decimal
    btc_free: Decimal
    btc_locked: Decimal = Decimal(0)
    position_qty_btc: Decimal = Decimal(0)
    position_entry_price: Decimal | None = None
    unrealized_pnl_jpy: Decimal = Decimal(0)


class TradingConstraints(_Model):
    """What the executor is actually allowed to do — restated for the model so
    it cannot propose something structurally impossible."""

    min_order_btc: Decimal
    price_tick: Decimal
    maker_fee_rate: Decimal
    taker_fee_rate: Decimal
    long_only: bool = True
    max_concurrent_positions: int = 1
    per_trade_risk_jpy: Decimal
    entry_max_deviation_pct: Decimal
    round_trip_fee_pct: Decimal = Field(
        description="maker entry + maker exit cost as a percentage of notional; "
                    "a plan whose TP does not clear this is a losing plan")


class MarketSnapshot(_Model):
    """One immutable observation of the world.

    Built once per cycle and shared byte-for-byte by every agent so the
    Anthropic prompt cache can serve it (spec 11).
    """

    snapshot_id: str
    taken_at: datetime
    pair: str
    last_price: Decimal
    mid_price: Decimal
    book: OrderBookSummary
    indicators: Indicators
    account: AccountState
    constraints: TradingConstraints
    recent_candles: list[Candle] = Field(default_factory=list)
    candle_type: str = "1hour"
    data_quality: list[str] = Field(
        default_factory=list,
        description="non-fatal gaps, e.g. 'short candle history'. Agents must "
                    "treat a non-empty list as a reason to be cautious.")

    def indicator_values(self) -> dict[str, Decimal]:
        """Flat name -> value map the guard uses to catch fabricated numbers."""
        values: dict[str, Decimal] = {}
        for name, value in self.indicators.model_dump().items():
            if value is not None:
                values[name] = dec(value)
        values.update(
            last_price=self.last_price,
            mid_price=self.mid_price,
            best_bid=self.book.best_bid,
            best_ask=self.book.best_ask,
            equity_jpy=self.account.equity_jpy,
            jpy_free=self.account.jpy_free,
        )
        return values

    def to_prompt_dict(self) -> dict:
        """Compact, stable-ordered view for the prompt.

        Candles are summarised rather than dumped: 200 OHLCV rows would dominate
        the context window and buy nothing the indicators do not already say.
        """
        data = {
            "snapshot_id": self.snapshot_id,
            "taken_at": iso(self.taken_at),
            "pair": self.pair,
            "last_price": _n(self.last_price),
            "mid_price": _n(self.mid_price),
            "order_book": {k: _n(v) for k, v in self.book.model_dump().items()},
            "indicators": {k: _n(v) for k, v in self.indicators.model_dump().items()
                           if v is not None},
            "account": {k: _n(v) for k, v in self.account.model_dump().items()
                        if v is not None},
            "constraints": {k: _n(v) for k, v in self.constraints.model_dump().items()},
            "recent_candles": [
                {
                    "t": iso(c.opened_at),
                    "o": _n(c.open), "h": _n(c.high),
                    "l": _n(c.low), "c": _n(c.close), "v": _n(c.volume),
                }
                for c in self.recent_candles[-24:]
            ],
            "candle_type": self.candle_type,
            "data_quality": self.data_quality,
        }
        return data

    def to_prompt_json(self) -> str:
        return json.dumps(self.to_prompt_dict(), ensure_ascii=False,
                          sort_keys=True, indent=1)


def _n(value):
    """Render Decimals as JSON numbers without scientific notation."""
    if isinstance(value, Decimal):
        text = format(value, "f")
        return float(text) if "." in text else int(text)
    if isinstance(value, datetime):
        return iso(value)
    return value
