"""Candle replay with real fees (spec 13, Phase 0).

What this does and does not measure is worth being blunt about.

It measures: the screener's trigger rate, the risk engine's sizing, the fee
model, and the stop/target mechanics. Those are deterministic, so replaying
them over history is meaningful.

It does not measure the agents. Replaying an LLM debate over a year of candles
would cost more than the annual budget and would be contaminated by hindsight
in the model's training data. The default planner is a deterministic baseline,
and its results are a floor to compare against — not a forecast of what the
multi-agent system will do.

Fill model matches the paper exchange: a resting buy needs the market to trade
*through* the limit, stops fill at the stop price as a taker, targets fill at
the target as a maker. Stops win ties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, Protocol

from ..config import Config
from ..data import indicators as ind
from ..models.market import Candle
from ..money import ZERO, dec, floor_to_lot, jpy


@dataclass
class BacktestTrade:
    entry_at: str
    entry: Decimal
    exit_at: str | None
    exit: Decimal | None
    qty: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    fees: Decimal
    net_pnl: Decimal
    reason: str


@dataclass
class BacktestResult:
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_start: Decimal = ZERO
    equity_end: Decimal = ZERO
    candles: int = 0
    triggers: int = 0
    max_drawdown_pct: Decimal = ZERO

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.net_pnl > 0)

    @property
    def net_pnl(self) -> Decimal:
        return sum((t.net_pnl for t in self.trades), ZERO)

    @property
    def total_fees(self) -> Decimal:
        return sum((t.fees for t in self.trades), ZERO)

    def summary(self) -> dict:
        count = len(self.trades)
        return {
            "candles": self.candles,
            "triggers": self.triggers,
            "trades": count,
            "wins": self.wins,
            "win_rate": round(self.wins / count, 3) if count else None,
            "net_pnl_jpy": float(jpy(self.net_pnl)),
            "total_fees_jpy": float(jpy(self.total_fees)),
            "expectancy_jpy": float(jpy(self.net_pnl / Decimal(count)))
            if count else None,
            "equity_start_jpy": float(jpy(self.equity_start)),
            "equity_end_jpy": float(jpy(self.equity_end)),
            "max_drawdown_pct": float(round(self.max_drawdown_pct, 2)),
        }


class Planner(Protocol):
    def __call__(self, candles: list[Candle], config: Config) -> dict | None: ...


class BaselinePlanner:
    """A deterministic pullback-to-VWAP long. Not a recommendation.

    It exists so the harness has something to replay, and so the fee model can
    be checked against a strategy whose behaviour is obvious: with a target
    narrower than the round-trip cost it must lose money, and if the harness
    ever says otherwise the harness is wrong.
    """

    def __init__(self, *, target_atr_multiple: Decimal = Decimal("1.5"),
                 stop_atr_multiple: Decimal = Decimal("1.0")):
        self.target_atr_multiple = target_atr_multiple
        self.stop_atr_multiple = stop_atr_multiple

    def __call__(self, candles: list[Candle], config: Config) -> dict | None:
        if len(candles) < 60:
            return None
        closes = ind.closes(candles)
        rsi = ind.rsi(closes, config.screening.rsi_period)
        atr = ind.atr(candles, 14)
        vwap = ind.vwap(candles[-24:])
        if rsi is None or atr is None or vwap is None or atr <= 0:
            return None
        last = closes[-1]
        if rsi > config.screening.rsi_low or last >= vwap:
            return None
        entry = last
        return {
            "entry": entry,
            "stop_loss": entry - atr * self.stop_atr_multiple,
            "take_profit": entry + atr * self.target_atr_multiple,
        }


def run_backtest(candles: list[Candle], config: Config, *,
                 planner: Planner | None = None,
                 equity: Decimal | None = None,
                 warmup: int = 60) -> BacktestResult:
    planner = planner or BaselinePlanner()
    equity = dec(equity if equity is not None else config.capital.initial_equity_jpy)

    from ..risk.rules import RiskEngine

    risk = RiskEngine(config)
    maker = config.exchange.maker_fee_rate
    taker = config.exchange.taker_fee_rate
    min_lot = config.exchange.min_order_btc

    result = BacktestResult(equity_start=equity, candles=len(candles))
    peak = equity
    open_trade: dict | None = None
    pending: dict | None = None
    # A PostOnly entry that never fills is cancelled, exactly as the executor
    # does after `post_only_timeout_minutes` (spec 8).
    entry_timeout_bars = max(1, config.execution.post_only_timeout_minutes // 60)

    for index in range(warmup, len(candles)):
        window = candles[:index + 1]
        candle = candles[index]

        if pending is not None:
            # Same fill rule as the paper exchange: the market has to trade
            # through the limit, not merely touch it.
            if candle.low < pending["entry"]:
                open_trade = pending
                open_trade["entry_at"] = candle.opened_at.isoformat()
                pending = None
            elif index - pending["placed_index"] >= entry_timeout_bars:
                pending = None
            continue

        if open_trade is not None:
            closed = _resolve(open_trade, candle, taker=taker, maker=maker)
            if closed is not None:
                equity += closed.net_pnl
                peak = max(peak, equity)
                drawdown = (peak - equity) / peak * Decimal(100) if peak > 0 else ZERO
                result.max_drawdown_pct = max(result.max_drawdown_pct, drawdown)
                result.trades.append(closed)
                open_trade = None
            continue

        plan = planner(window, config)
        if plan is None:
            continue
        result.triggers += 1

        sizing = risk.position_size(
            equity=equity, entry=plan["entry"], stop_loss=plan["stop_loss"],
            jpy_available=equity)
        if not sizing.ok:
            continue
        qty = floor_to_lot(sizing.qty_btc, min_lot)
        if qty < min_lot:
            continue

        pending = {
            "placed_index": index,
            "entry_at": candle.opened_at.isoformat(),
            "entry": dec(plan["entry"]),
            "stop_loss": dec(plan["stop_loss"]),
            "take_profit": dec(plan["take_profit"]),
            "qty": qty,
        }

    result.equity_end = equity
    return result


def _resolve(trade: dict, candle: Candle, *, taker: Decimal,
             maker: Decimal) -> BacktestTrade | None:
    """Stops win ties: a bar that spans both levels is booked as a stop.

    Within one bar there is no way to know which level came first, and assuming
    the profitable one would inflate every result this harness produces.
    """
    entry_fee = trade["entry"] * trade["qty"] * maker
    if candle.low <= trade["stop_loss"]:
        exit_price = trade["stop_loss"]
        reason, fee_rate = "stop_loss", taker
    elif candle.high >= trade["take_profit"]:
        exit_price = trade["take_profit"]
        reason, fee_rate = "take_profit", maker
    else:
        return None

    exit_fee = exit_price * trade["qty"] * fee_rate
    fees = entry_fee + exit_fee
    gross = (exit_price - trade["entry"]) * trade["qty"]
    return BacktestTrade(
        entry_at=trade["entry_at"], entry=trade["entry"],
        exit_at=candle.opened_at.isoformat(), exit=exit_price, qty=trade["qty"],
        stop_loss=trade["stop_loss"], take_profit=trade["take_profit"],
        fees=fees, net_pnl=gross - fees, reason=reason)
