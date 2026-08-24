"""Risk and capital management — priority 2, and the only layer allowed to
decide how much money is at stake (spec 6).

Two properties matter more than anything else here:

* Nothing in this module can be relaxed by a lower-priority rule. The boredom
  rule (spec 7) can lower the size and loosen a *consensus* threshold; it
  cannot raise risk, and it cannot run while a circuit breaker is open.
* Every limit is computed from equity at the time of the decision, never from
  a cached figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from ..config import Config
from ..models.state import Halt, HaltReason, SystemState
from ..models.trading import TradeRecord
from ..money import ZERO, dec, floor_to_lot, jpy


@dataclass
class SizingResult:
    qty_btc: Decimal
    risk_jpy: Decimal
    risk_limit_jpy: Decimal
    ok: bool
    reason: str = ""


class RiskEngine:
    def __init__(self, config: Config):
        self.config = config

    # -- sizing ------------------------------------------------------------

    def risk_limit_jpy(self, equity: Decimal, *, probe: bool) -> Decimal:
        pct = (self.config.risk.probe_risk_pct if probe
               else self.config.risk.per_trade_risk_pct)
        return jpy(dec(equity) * pct / Decimal(100))

    def position_size(self, *, equity: Decimal, entry: Decimal, stop_loss: Decimal,
                      jpy_available: Decimal, probe: bool = False) -> SizingResult:
        """Size from risk, then clamp to what the account can actually buy.

        Phase 2 (spec 13) fixes every order at the minimum lot regardless of
        what the risk budget would allow; the risk check still runs, so a plan
        whose stop is too wide for even one minimum lot is rejected rather than
        quietly resized.
        """
        cfg = self.config
        min_lot = cfg.exchange.min_order_btc
        limit = self.risk_limit_jpy(equity, probe=probe)
        entry, stop_loss = dec(entry), dec(stop_loss)

        if entry <= 0 or stop_loss <= 0:
            return SizingResult(ZERO, ZERO, limit, False, "entry/stop must be positive")
        if stop_loss >= entry:
            return SizingResult(ZERO, ZERO, limit, False,
                                "long position requires stop_loss < entry")

        per_unit_risk = entry - stop_loss
        affordable = floor_to_lot(dec(jpy_available) / entry, min_lot)

        if probe or cfg.system.phase == 2:
            qty = min_lot
        else:
            qty = floor_to_lot(limit / per_unit_risk, min_lot)

        # Two different failures look alike at the minimum lot, and the owner
        # needs to be able to tell them apart: a stop too wide for the risk
        # budget is a bad plan, while an account too small to buy one lot is a
        # funding problem no plan can fix.
        if qty < min_lot:
            return SizingResult(
                ZERO, ZERO, limit, False,
                f"risk-based size is below the minimum lot {min_lot}: a "
                f"{per_unit_risk} JPY stop distance needs more than the "
                f"{limit} JPY per-trade risk budget allows")
        if affordable < min_lot:
            return SizingResult(
                ZERO, ZERO, limit, False,
                f"affordable size {affordable} is below the minimum lot {min_lot}")
        if qty > affordable:
            qty = affordable

        risk = jpy(per_unit_risk * qty)
        if risk > limit:
            return SizingResult(
                qty, risk, limit, False,
                f"risk {risk} JPY exceeds the {limit} JPY limit at the minimum lot; "
                "the stop is too wide for this account")
        return SizingResult(qty, risk, limit, True)

    # -- circuit breakers --------------------------------------------------

    def evaluate_halts(self, state: SystemState, now: datetime, *,
                       change_15m_pct: Decimal | None = None,
                       budget_stopped: bool = False,
                       for_new_entry: bool = True) -> list[Halt]:
        """Every reason a new position may not be opened right now.

        Returned in priority order so the first entry is the one to report.
        """
        cfg = self.config
        halts: list[Halt] = []

        if state.kill_switch:
            halts.append(Halt(reason=HaltReason.KILL_SWITCH,
                              detail=state.kill_switch_reason or
                              "kill switch engaged; resume_trading is required"))
        if state.owner_paused:
            halts.append(Halt(reason=HaltReason.OWNER_PAUSE,
                              detail=state.owner_pause_reason or "paused by owner"))
        if state.daily_loss_pct() >= cfg.risk.daily_loss_limit_pct:
            halts.append(Halt(
                reason=HaltReason.DAILY_LOSS,
                detail=(f"daily loss {state.daily_loss_pct():.2f}% has reached the "
                        f"{cfg.risk.daily_loss_limit_pct}% limit")))
        if state.in_losing_streak_pause(now):
            halts.append(Halt(reason=HaltReason.LOSING_STREAK,
                              detail=f"{state.losing_streak} consecutive losses",
                              until=state.losing_streak_until))
        if state.in_flash_pause(now):
            halts.append(Halt(reason=HaltReason.FLASH_MOVE,
                              detail="price moved sharply; standing down",
                              until=state.flash_pause_until))
        elif change_15m_pct is not None and abs(dec(change_15m_pct)) > cfg.risk.flash_move_pct:
            halts.append(Halt(
                reason=HaltReason.FLASH_MOVE,
                detail=(f"{change_15m_pct}% move over the last "
                        f"{cfg.risk.flash_move_window_minutes} minutes"),
                until=now + timedelta(hours=cfg.risk.flash_move_pause_hours)))
        if for_new_entry and state.has_position():
            halts.append(Halt(reason=HaltReason.POSITION_OPEN,
                              detail="a position is already open"))
        if budget_stopped:
            halts.append(Halt(reason=HaltReason.BUDGET,
                              detail="monthly LLM budget exhausted"))
        if state.consecutive_private_api_failures >= cfg.exchange.private_failure_threshold:
            halts.append(Halt(
                reason=HaltReason.DATA_QUALITY,
                detail=(f"{state.consecutive_private_api_failures} consecutive "
                        "private API failures; cannot confirm account state")))
        return halts

    def should_kill(self, state: SystemState) -> bool:
        """Spec 6: -20% from *initial capital*, not from the running peak."""
        initial = self.config.capital.initial_equity_jpy
        return state.drawdown_pct(initial) >= self.config.risk.killswitch_drawdown_pct

    def engage_kill_switch(self, state: SystemState, now: datetime,
                           reason: str) -> SystemState:
        state.kill_switch = True
        state.kill_switch_reason = reason
        state.kill_switch_at = now
        return state

    # -- flash-move detection ---------------------------------------------

    def check_flash_move(self, state: SystemState, now: datetime,
                         change_15m_pct: Decimal | None) -> bool:
        """Arm the flash pause. Returns True when it was newly armed."""
        cfg = self.config
        if change_15m_pct is None:
            return False
        if abs(dec(change_15m_pct)) <= cfg.risk.flash_move_pct:
            return False
        until = now + timedelta(hours=cfg.risk.flash_move_pause_hours)
        if state.flash_pause_until and state.flash_pause_until >= until:
            return False
        state.flash_pause_until = until
        state.flash_recovery_pending = True
        return True

    # -- trade outcomes ----------------------------------------------------

    def apply_trade_result(self, state: SystemState, trade: TradeRecord,
                           now: datetime) -> SystemState:
        """Fold a closed trade into equity, streak and the daily/monthly rolls.

        Probe trades count towards equity and the daily loss limit — they are
        real money — but they are kept out of the losing-streak brake and are
        tracked separately so they cannot contaminate strategy statistics
        (spec 7).
        """
        cfg = self.config
        pnl = trade.net_pnl_jpy or ZERO
        state.equity_jpy += pnl
        state.peak_equity_jpy = max(state.peak_equity_jpy, state.equity_jpy)
        state.daily.realized_pnl_jpy += pnl

        if trade.probe:
            state.monthly.probe_pnl_jpy += pnl
            limit = (cfg.capital.initial_equity_jpy
                     * cfg.boredom.monthly_probe_loss_limit_pct / Decimal(100))
            if -state.monthly.probe_pnl_jpy >= limit:
                state.monthly.probe_rule_suspended = True
        else:
            if pnl < 0:
                state.losing_streak += 1
                if state.losing_streak >= cfg.risk.losing_streak_limit:
                    state.losing_streak_until = now + timedelta(
                        hours=cfg.risk.losing_streak_pause_hours)
            else:
                state.losing_streak = 0
                state.losing_streak_until = None

        if self.should_kill(state):
            self.engage_kill_switch(
                state, now,
                f"equity {state.equity_jpy} JPY is "
                f"{state.drawdown_pct(cfg.capital.initial_equity_jpy):.1f}% below "
                "initial capital")
        return state
