"""The 3-day rule (spec 7) — priority 3, below every safety rule.

The owner does not want the system to sit silent for three days. That is an
entertainment requirement, not an edge, and the code says so out loud: probe
trades are flagged, sized at the minimum lot, stopped tightly, excluded from
strategy statistics, and cut off entirely once their cumulative monthly loss
reaches 2% of equity.

What this rule may do:
  * lower the judge's consensus threshold from 2-of-3 to 1-of-3 for one cycle
  * place one mechanical minimum-lot probe when no strategist proposes a buy

What it may never do:
  * fire while the kill switch, the losing-streak brake, the daily loss limit
    or a flash-move pause is active
  * raise position size or risk above the probe limits
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from ..config import Config
from ..models.market import MarketSnapshot
from ..models.state import Halt, HaltReason, SystemState
from ..money import ZERO, dec, quantize_price


@dataclass
class BoredomDecision:
    triggered: bool
    hours_idle: float | None
    reason: str
    consensus_min: int
    blocked_by: HaltReason | None = None


BLOCKING_HALTS = {
    HaltReason.KILL_SWITCH,
    HaltReason.DAILY_LOSS,
    HaltReason.LOSING_STREAK,
    HaltReason.FLASH_MOVE,
    HaltReason.OWNER_PAUSE,
    HaltReason.PROBE_BUDGET,
    HaltReason.DATA_QUALITY,
}


def evaluate_boredom(config: Config, state: SystemState, now: datetime,
                     halts: list[Halt]) -> BoredomDecision:
    """Decide whether the 3-day rule fires this cycle."""
    cfg = config.boredom
    normal_consensus = config.screening.consensus_min

    if not cfg.enabled:
        return BoredomDecision(False, None, "boredom rule disabled",
                               normal_consensus)

    blocking = next((h.reason for h in halts if h.reason in BLOCKING_HALTS), None)
    if blocking is not None:
        # A safety rule is open. Spec 0: this rule loses, always.
        return BoredomDecision(False, state.hours_since_last_entry(now),
                               f"suppressed by {blocking}", normal_consensus,
                               blocked_by=blocking)

    if state.monthly.probe_rule_suspended:
        return BoredomDecision(
            False, state.hours_since_last_entry(now),
            "probe losses reached the monthly cap; rule suspended for this month",
            normal_consensus, blocked_by=HaltReason.PROBE_BUDGET)

    if state.has_position():
        return BoredomDecision(False, state.hours_since_last_entry(now),
                               "a position is open; the clock does not apply",
                               normal_consensus,
                               blocked_by=HaltReason.POSITION_OPEN)

    idle = state.hours_since_last_entry(now)
    if idle is None:
        # No entry has ever happened. Measure from the moment the system
        # started keeping state so a fresh deployment does not fire instantly.
        started = state.updated_at
        idle = ((now - started).total_seconds() / 3600.0) if started else 0.0

    if idle < cfg.no_trade_hours:
        return BoredomDecision(False, idle,
                               f"{idle:.1f}h idle, below the "
                               f"{cfg.no_trade_hours}h threshold",
                               normal_consensus)

    return BoredomDecision(
        True, idle,
        f"{idle:.1f}h without a new entry; relaxing consensus to "
        f"{cfg.relaxed_consensus_min}/3 for one probe",
        cfg.relaxed_consensus_min)


def probe_stop_loss(config: Config, entry: Decimal) -> Decimal:
    """Spec 7: tight stop, within -0.7% of entry."""
    pct = config.boredom.probe_sl_pct
    stop = dec(entry) * (Decimal(1) - pct / Decimal(100))
    return quantize_price(stop, config.exchange.price_digits)


def mechanical_probe_plan(config: Config, snapshot: MarketSnapshot,
                          regime: str | None) -> dict | None:
    """Spec 7: when no strategist proposes a buy, place one mechanical probe.

    A limit slightly under the 24-hour VWAP: it rests as a maker order, it is
    where a pullback would fill, and it needs no view on direction. Returns
    None when the snapshot lacks a VWAP, because guessing an entry price is
    exactly the kind of invention this system refuses to do.
    """
    vwap = snapshot.indicators.vwap_24h
    if vwap is None or vwap <= 0:
        return None
    discount = config.boredom.mechanical_probe_discount_pct
    entry = quantize_price(dec(vwap) * (Decimal(1) - discount / Decimal(100)),
                           config.exchange.price_digits)
    # Never chase: a probe entry above the current market would be a taker fill.
    ceiling = quantize_price(snapshot.book.best_bid, config.exchange.price_digits)
    if entry > ceiling:
        entry = ceiling
    if entry <= 0:
        return None
    stop = probe_stop_loss(config, entry)
    # Target the same distance as the stop, doubled, so the probe is at least
    # break-even-shaped rather than a coin flip that pays fees either way.
    take = quantize_price(entry + (entry - stop) * Decimal(2),
                          config.exchange.price_digits)
    return {
        "entry": entry,
        "stop_loss": stop,
        "take_profit": take,
        "source": "mechanical_probe",
        "regime": regime,
        "rationale": (f"72時間ルール。直近{config.boredom.mechanical_probe_vwap_hours}"
                      "時間VWAPへの押し目に最小ロットで指値。"),
    }


def probe_loss_room_jpy(config: Config, state: SystemState) -> Decimal:
    """How much more the probes may lose this month before the rule stops."""
    limit = (config.capital.initial_equity_jpy
             * config.boredom.monthly_probe_loss_limit_pct / Decimal(100))
    used = -min(ZERO, state.monthly.probe_pnl_jpy)
    return max(ZERO, limit - used)
