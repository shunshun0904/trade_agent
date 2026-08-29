"""MarketSnapshot assembly (spec 3, layer 1).

One network fan-out per cycle: ticker, depth and two candle series. Everything
the agents will ever be told about the market is decided here.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, timedelta
from decimal import Decimal

from ..config import Config
from ..errors import ExchangeError
from ..models.market import (
    AccountState,
    Candle,
    Indicators,
    MarketSnapshot,
    OrderBookSummary,
    TradingConstraints,
)
from ..models.trading import Position
from ..money import ZERO, dec, deviation_pct
from ..timeutil import Clock, iso
from . import indicators as ind

log = logging.getLogger(__name__)

# Candles per hour, by bitbank candle type — used to convert "N hours" windows
# into a candle count without assuming a particular timeframe.
CANDLES_PER_HOUR = {
    "1min": 60, "5min": 12, "15min": 4, "30min": 2,
    "1hour": 1, "4hour": 0.25, "8hour": 0.125, "12hour": 1 / 12, "1day": 1 / 24,
}


class SnapshotBuilder:
    def __init__(self, exchange, config: Config, clock: Clock):
        self.exchange = exchange
        self.config = config
        self.clock = clock

    def build(self, *, position: Position | None = None,
              equity_override: Decimal | None = None) -> MarketSnapshot:
        now = self.clock.now()
        cfg = self.config
        quality: list[str] = []

        ticker = self.exchange.get_ticker()
        depth = self.exchange.get_depth()
        last = dec(ticker["last"])

        long_candles = self._fetch_series(cfg.snapshot.candle_type,
                                          cfg.snapshot.candle_limit, now, quality)
        short_candles = self._fetch_series(cfg.snapshot.short_candle_type,
                                           cfg.snapshot.short_candle_limit, now, quality)

        book = self._summarise_book(depth, cfg.snapshot.depth_levels)
        indicators = self._indicators(long_candles, short_candles, last, quality)
        account = self._account(last, position, equity_override)
        constraints = self._constraints(account.equity_jpy)

        snapshot = MarketSnapshot(
            snapshot_id=_snapshot_id(now, last),
            taken_at=now,
            pair=cfg.exchange.pair,
            last_price=last,
            mid_price=(book.best_bid + book.best_ask) / Decimal(2),
            book=book,
            indicators=indicators,
            account=account,
            constraints=constraints,
            recent_candles=long_candles[-cfg.snapshot.depth_levels * 2:],
            candle_type=cfg.snapshot.candle_type,
            data_quality=quality,
        )
        return snapshot

    # -- pieces ------------------------------------------------------------

    def _fetch_series(self, candle_type: str, limit: int, now: datetime,
                      quality: list[str]) -> list[Candle]:
        """Walk backwards day by day (or year by year) until `limit` candles."""
        rows: list[Candle] = []
        per_hour = CANDLES_PER_HOUR.get(candle_type, 1)
        keys = self._history_keys(candle_type, limit, per_hour, now)
        for key in keys:
            try:
                rows = self.exchange.get_candles(candle_type, key) + rows
            except ExchangeError as exc:
                log.warning("candle fetch failed for %s/%s: %s", candle_type, key, exc)
                quality.append(f"candle fetch failed for {candle_type}/{key}")
            if len(rows) >= limit:
                break
        # Drop the candle that is still forming. Keeping it made every
        # indicator read a partial bar as if it were a whole one, and it broke
        # volume outright: `volume_ratio` compares the last candle against the
        # mean of the 20 before it, so a 1-hour candle sampled seconds after
        # the hour turned reported ~0.02 — not "volume has collapsed" but
        # "this hour has barely started". The screener runs on the hour and the
        # half hour, so the ratio could never exceed ~0.5 and the 1.5x volume
        # trigger was unreachable. The agents, reading the same number, cited
        # "極度に低い出来高" as a reason to decline in every cycle.
        #
        # The current price is unaffected: it comes from the ticker, not here.
        duration = timedelta(hours=1 / per_hour)
        rows = [c for c in rows if c.opened_at + duration <= now]
        rows.sort(key=lambda c: c.opened_at)
        # Guard against the exchange returning the same day twice.
        deduped: list[Candle] = []
        seen: set[datetime] = set()
        for candle in rows:
            if candle.opened_at not in seen:
                seen.add(candle.opened_at)
                deduped.append(candle)
        if len(deduped) < limit:
            quality.append(
                f"only {len(deduped)}/{limit} {candle_type} candles available")
        return deduped[-limit:]

    @staticmethod
    def _history_keys(candle_type: str, limit: int, per_hour: float,
                      now: datetime) -> list[date | int]:
        """Most recent bucket first; the fetcher stops as soon as it has enough."""
        if candle_type in {"1min", "5min", "15min", "30min", "1hour"}:
            span_hours = limit / per_hour if per_hour else limit
            days = int(span_hours // 24) + 2
            return [(now - timedelta(days=offset)).date() for offset in range(days)]
        return [now.year, now.year - 1]

    @staticmethod
    def _summarise_book(depth: dict, levels: int) -> OrderBookSummary:
        asks = [(dec(p), dec(a)) for p, a in depth.get("asks", [])[:levels]]
        bids = [(dec(p), dec(a)) for p, a in depth.get("bids", [])[:levels]]
        if not asks or not bids:
            raise ExchangeError("order book empty; refusing to build a snapshot")
        best_ask, best_bid = asks[0][0], bids[0][0]
        ask_depth = sum((a for _, a in asks), ZERO)
        bid_depth = sum((a for _, a in bids), ZERO)
        total = ask_depth + bid_depth
        spread = best_ask - best_bid
        return OrderBookSummary(
            best_bid=best_bid,
            best_ask=best_ask,
            spread_jpy=spread,
            spread_pct=spread / best_bid * Decimal(100) if best_bid else ZERO,
            bid_depth_btc=bid_depth,
            ask_depth_btc=ask_depth,
            imbalance=(bid_depth - ask_depth) / total if total else ZERO,
        )

    def _indicators(self, long_candles: list[Candle], short_candles: list[Candle],
                    last: Decimal, quality: list[str]) -> Indicators:
        cfg = self.config
        long_closes = ind.closes(long_candles)
        short_closes = ind.closes(short_candles)

        per_hour_short = CANDLES_PER_HOUR.get(cfg.snapshot.short_candle_type, 12)
        vwap_window = int(24 * per_hour_short)
        vwap_candles = short_candles[-vwap_window:]
        vwap_24h = ind.vwap(vwap_candles) if vwap_candles else None

        per_hour_long = CANDLES_PER_HOUR.get(cfg.snapshot.candle_type, 1)
        day_window = max(1, int(24 * per_hour_long))
        minutes_per_short_candle = 60 / per_hour_short if per_hour_short else 5
        flash_periods = max(
            1, int(cfg.risk.flash_move_window_minutes / minutes_per_short_candle))

        if len(long_closes) < 20:
            quality.append("indicator history is short; treat trend reads as weak")

        upper, lower, band_pct = ind.bollinger(long_closes, 20)
        atr_value = ind.atr(long_candles, 14)
        return Indicators(
            sma_short=ind.sma(long_closes, 20),
            sma_long=ind.sma(long_closes, 50),
            ema_short=ind.ema(long_closes, 12),
            ema_long=ind.ema(long_closes, 26),
            rsi=ind.rsi(long_closes, cfg.screening.rsi_period),
            atr=atr_value,
            atr_pct=(atr_value / last * Decimal(100)) if atr_value and last else None,
            bb_upper=upper,
            bb_lower=lower,
            bb_width_pct=band_pct,
            vwap_24h=vwap_24h,
            vwap_deviation_pct=(deviation_pct(last, vwap_24h)
                                if vwap_24h else None),
            high_24h=ind.highest(long_candles, day_window),
            low_24h=ind.lowest(long_candles, day_window),
            change_24h_pct=ind.change_pct(long_closes, day_window),
            change_15m_pct=ind.change_pct(short_closes, flash_periods),
            volume_24h_btc=sum((c.volume for c in long_candles[-day_window:]), ZERO)
            if long_candles else None,
            volume_ratio=ind.volume_ratio(long_candles, 20),
            realized_vol_pct=ind.realized_vol_pct(long_closes, 20),
        )

    def _account(self, last: Decimal, position: Position | None,
                 equity_override: Decimal | None) -> AccountState:
        balances = self.exchange.get_balances()
        jpy = balances.get(self.config.exchange.quote_asset)
        btc = balances.get(self.config.exchange.base_asset)
        jpy_free = jpy.free if jpy else ZERO
        jpy_total = (jpy.free + jpy.locked) if jpy else ZERO
        btc_free = btc.free if btc else ZERO
        btc_locked = btc.locked if btc else ZERO
        btc_total = btc_free + btc_locked
        equity = equity_override if equity_override is not None else (
            jpy_total + btc_total * last)
        return AccountState(
            equity_jpy=equity,
            jpy_free=jpy_free,
            btc_free=btc_free,
            btc_locked=btc_locked,
            position_qty_btc=position.qty_btc if position else ZERO,
            position_entry_price=position.entry_price if position else None,
            unrealized_pnl_jpy=position.unrealized_pnl_jpy(last) if position else ZERO,
        )

    def _constraints(self, equity: Decimal) -> TradingConstraints:
        cfg = self.config
        maker = cfg.exchange.maker_fee_rate
        return TradingConstraints(
            min_order_btc=cfg.exchange.min_order_btc,
            price_tick=Decimal(1).scaleb(-cfg.exchange.price_digits),
            maker_fee_rate=maker,
            taker_fee_rate=cfg.exchange.taker_fee_rate,
            long_only=True,
            max_concurrent_positions=cfg.risk.max_concurrent_positions,
            per_trade_risk_jpy=equity * cfg.risk.per_trade_risk_pct / Decimal(100),
            entry_max_deviation_pct=cfg.guard.entry_max_deviation_pct,
            round_trip_fee_pct=maker * Decimal(2) * Decimal(100),
        )


def _snapshot_id(now: datetime, last: Decimal) -> str:
    digest = hashlib.sha256(f"{iso(now)}|{last}".encode()).hexdigest()[:12]
    return f"snap-{digest}"
