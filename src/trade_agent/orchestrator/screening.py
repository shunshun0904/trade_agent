"""The 30-minute screener (spec 9).

Zero LLM calls, zero yen. It answers one question — is anything happening that
is worth paying a full debate for — using thresholds that live in config.

It runs only when flat: with a position open the tick owns the outcome, and a
second opinion cannot act on anything (spec 9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from ..config import Config
from ..models.market import MarketSnapshot
from ..models.state import CycleTrigger, SystemState
from ..money import dec
from ..timeutil import crossed_jst_time


@dataclass
class ScreenResult:
    should_debate: bool
    trigger: CycleTrigger | None = None
    reasons: list[str] = field(default_factory=list)
    suppressed_by: str | None = None

    def summary(self) -> str:
        if self.should_debate:
            return f"{self.trigger}: " + "; ".join(self.reasons)
        return self.suppressed_by or "no trigger condition met"


def evaluate_triggers(config: Config, state: SystemState,
                      snapshot: MarketSnapshot | None, now: datetime, *,
                      halts: list | None = None,
                      debates_today: int = 0,
                      daily_limit: int | None = None) -> ScreenResult:
    """Decide whether to spend an LLM cycle right now.

    Order matters: the cheap disqualifiers run before the market conditions, so
    a blocked system never even reads the indicators.
    """
    cfg = config.schedule
    limit = daily_limit if daily_limit is not None else cfg.daily_full_debate_limit

    if state.has_position():
        return ScreenResult(False, suppressed_by="a position is open")
    if halts:
        return ScreenResult(False, suppressed_by=f"halted: {halts[0].reason}")
    if debates_today >= limit:
        return ScreenResult(
            False, suppressed_by=f"daily debate limit reached ({debates_today}/{limit})")

    cooldown_ok = _cooldown_elapsed(config, state, now)

    # (b) Scheduled floor: 09:00 / 21:00 JST run regardless of conditions, so
    # the system is never completely silent (spec 9).
    for hhmm in cfg.floor_times_jst:
        if crossed_jst_time(state.last_floor_run_at, now, hhmm, cfg.screen_minutes):
            return ScreenResult(True, CycleTrigger.FLOOR,
                                [f"scheduled floor at {hhmm} JST"])

    # (c) One debate after a flash-move pause lifts.
    if state.flash_recovery_pending and not state.in_flash_pause(now):
        return ScreenResult(True, CycleTrigger.FLASH_RECOVERY,
                            ["flash-move pause has lifted"])

    if snapshot is None:
        return ScreenResult(False, suppressed_by="no snapshot available")
    if not cooldown_ok:
        return ScreenResult(
            False,
            suppressed_by=(f"within the {cfg.full_debate_cooldown_minutes}-minute "
                           "cooldown since the last debate"))

    reasons = _market_triggers(config, snapshot)
    if reasons:
        return ScreenResult(True, CycleTrigger.SCREEN, reasons)
    return ScreenResult(False, suppressed_by="no trigger condition met")


def _cooldown_elapsed(config: Config, state: SystemState, now: datetime) -> bool:
    last = state.last_full_debate_at
    if last is None:
        return True
    minutes = (now - last).total_seconds() / 60
    return minutes >= config.schedule.full_debate_cooldown_minutes


def _market_triggers(config: Config, snapshot: MarketSnapshot) -> list[str]:
    """Deterministic conditions from spec 9: breakout, RSI extreme, volume
    spike, VWAP deviation. Thresholds are all configuration."""
    cfg = config.screening
    ind = snapshot.indicators
    price = snapshot.last_price
    reasons: list[str] = []

    if ind.high_24h is not None and price >= ind.high_24h:
        reasons.append(f"price {price} broke the {cfg.breakout_lookback_hours}h "
                       f"high {ind.high_24h}")
    if ind.low_24h is not None and price <= ind.low_24h:
        reasons.append(f"price {price} broke the {cfg.breakout_lookback_hours}h "
                       f"low {ind.low_24h}")
    if ind.rsi is not None:
        if ind.rsi <= cfg.rsi_low:
            reasons.append(f"RSI {ind.rsi:.1f} at or below {cfg.rsi_low}")
        elif ind.rsi >= cfg.rsi_high:
            reasons.append(f"RSI {ind.rsi:.1f} at or above {cfg.rsi_high}")
    if ind.volume_ratio is not None and ind.volume_ratio >= cfg.volume_spike_multiple:
        reasons.append(f"volume {ind.volume_ratio:.2f}x its recent average")
    if (ind.vwap_deviation_pct is not None
            and dec(ind.vwap_deviation_pct) >= cfg.vwap_deviation_pct):
        reasons.append(f"price {ind.vwap_deviation_pct:.2f}% away from the 24h VWAP")
    return reasons


def daily_debate_limit(config: Config, budget_state) -> int:
    """Spec 11: the 80% budget rung cuts the daily cap to one."""
    override = budget_state.daily_debate_limit_override if budget_state else None
    if override is None:
        return config.schedule.daily_full_debate_limit
    return min(override, config.cost.degraded_daily_debate_limit)


def flash_move_pct(snapshot: MarketSnapshot | None) -> Decimal | None:
    return snapshot.indicators.change_15m_pct if snapshot else None
