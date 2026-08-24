"""System state — the row every safety decision reads before acting.

Spec 10 `system_state`. Stored as a single item so `tick`, `decide`, `screen`
and `mcp` all see one consistent view.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ..money import ZERO


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CycleTrigger(StrEnum):
    """Why a full debate is running (spec 9)."""

    SCREEN = "screen"            # (a) 30-minute screener condition fired
    FLOOR = "floor"              # (b) 09:00 / 21:00 JST scheduled floor
    FLASH_RECOVERY = "flash"     # (c) once, after a flash-move pause lifts
    BOREDOM = "boredom"          # spec 7, exempt from the daily debate cap
    MANUAL = "manual"            # operator-initiated (CLI / test)


class HaltReason(StrEnum):
    KILL_SWITCH = "kill_switch"
    DAILY_LOSS = "daily_loss_limit"
    LOSING_STREAK = "losing_streak"
    FLASH_MOVE = "flash_move"
    OWNER_PAUSE = "owner_pause"
    BUDGET = "llm_budget_exhausted"
    POSITION_OPEN = "position_open"
    DEBATE_LIMIT = "daily_debate_limit"
    COOLDOWN = "cooldown"
    DATA_QUALITY = "data_unavailable"
    PROBE_BUDGET = "probe_loss_limit"


class DailyCounters(_Model):
    """Reset at JST midnight (spec 17.3)."""

    jst_date: str
    realized_pnl_jpy: Decimal = ZERO
    full_debates: int = 0
    llm_cost_jpy: Decimal = ZERO
    start_equity_jpy: Decimal | None = None


class MonthlyCounters(_Model):
    """Reset on the 1st, JST. Drives the budget ladder (spec 11) and the
    probe-loss cap (spec 7)."""

    jst_month: str
    llm_cost_jpy: Decimal = ZERO
    probe_pnl_jpy: Decimal = ZERO
    probe_rule_suspended: bool = False


class SystemState(_Model):
    equity_jpy: Decimal
    peak_equity_jpy: Decimal

    kill_switch: bool = False
    kill_switch_reason: str | None = None
    kill_switch_at: datetime | None = None

    owner_paused: bool = False
    owner_pause_reason: str | None = None

    losing_streak: int = 0
    losing_streak_until: datetime | None = None
    flash_pause_until: datetime | None = None
    flash_recovery_pending: bool = False

    last_entry_at: datetime | None = Field(
        default=None, description="last *new* position entry; drives the 3-day rule")
    last_full_debate_at: datetime | None = None
    last_floor_run_at: datetime | None = None
    last_daily_report_at: datetime | None = None
    last_tick_at: datetime | None = None

    open_position: object | None = Field(
        default=None, description="Position | None; typed loosely to avoid a cycle")

    daily: DailyCounters
    monthly: MonthlyCounters

    consecutive_private_api_failures: int = 0

    # Spec 11: A7 may run through the Batch API, which answers within 24 hours
    # rather than within one invocation. The tick polls this while it is set.
    pending_reflect_batch_id: str | None = None
    pending_reflect_batch_at: datetime | None = None

    updated_at: datetime | None = None
    version: int = 0

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # -- derived predicates ------------------------------------------------

    def drawdown_pct(self, initial_equity: Decimal) -> Decimal:
        """Drawdown from *initial capital*, which is what spec 6 keys the kill
        switch to — not from the running peak."""
        if initial_equity <= 0:
            return ZERO
        return (initial_equity - self.equity_jpy) / initial_equity * Decimal(100)

    def daily_loss_pct(self) -> Decimal:
        base = self.daily.start_equity_jpy or self.equity_jpy
        if base <= 0:
            return ZERO
        return -self.daily.realized_pnl_jpy / base * Decimal(100)

    def hours_since_last_entry(self, now: datetime) -> float | None:
        if self.last_entry_at is None:
            return None
        return (now - self.last_entry_at).total_seconds() / 3600.0

    def in_losing_streak_pause(self, now: datetime) -> bool:
        return self.losing_streak_until is not None and now < self.losing_streak_until

    def in_flash_pause(self, now: datetime) -> bool:
        return self.flash_pause_until is not None and now < self.flash_pause_until

    def has_position(self) -> bool:
        return self.open_position is not None

    @classmethod
    def initial(cls, equity: Decimal, now: datetime, jst_date: str,
                jst_month: str) -> "SystemState":
        return cls(
            equity_jpy=equity,
            peak_equity_jpy=equity,
            daily=DailyCounters(jst_date=jst_date, start_equity_jpy=equity),
            monthly=MonthlyCounters(jst_month=jst_month),
            updated_at=now,
        )


class Halt(_Model):
    """A refusal to open a new position, with the reason attached so it can be
    logged and reported rather than silently dropped."""

    reason: HaltReason
    detail: str = ""
    until: datetime | None = None

    def remaining(self, now: datetime) -> timedelta | None:
        return None if self.until is None else self.until - now
