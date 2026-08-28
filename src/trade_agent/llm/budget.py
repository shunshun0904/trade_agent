"""Token accounting and the monthly budget ladder (spec 11).

    spend < 80%   full schedule
    spend >= 80%  degraded: one full debate per day
    spend >= 100% no LLM calls at all for the rest of the month

The third rung does not stop the system: the 5-minute tick, SL/TP evaluation
and the kill switch are all deterministic and keep running. Losing the ability
to *open* a position is an acceptable failure; losing the ability to *close*
one is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from ..config import CostConfig, LLMConfig
from ..money import ZERO, dec
from ..timeutil import jst_days_remaining_in_month
from .base import TokenUsage

MILLION = Decimal(1_000_000)


class BudgetLadder(StrEnum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    STOPPED = "stopped"


@dataclass(frozen=True)
class BudgetState:
    ladder: BudgetLadder
    spent_jpy: Decimal
    budget_jpy: Decimal
    used_pct: Decimal

    @property
    def llm_allowed(self) -> bool:
        return self.ladder is not BudgetLadder.STOPPED

    @property
    def remaining_jpy(self) -> Decimal:
        return max(ZERO, self.budget_jpy - self.spent_jpy)


class CostMeter:
    """Converts token counts to yen using the configured price list.

    Prices are configuration, not constants in code: they change, and the whole
    budget mechanism is worthless if the numbers drift from reality.
    """

    def __init__(self, llm: LLMConfig, cost: CostConfig):
        self.llm = llm
        self.cost = cost

    def cost_jpy(self, usage: TokenUsage, *, batch: bool = False) -> Decimal:
        p = self.llm.pricing
        usd = (
            dec(usage.input_tokens) * p.input_per_mtok_usd
            + dec(usage.output_tokens) * p.output_per_mtok_usd
            + dec(usage.cache_write_tokens) * p.input_per_mtok_usd
            * p.cache_write_multiplier
            + dec(usage.cache_read_tokens) * p.input_per_mtok_usd
            * p.cache_read_multiplier
        ) / MILLION
        if batch:
            usd *= p.batch_multiplier
        return usd * self.llm.usd_jpy_rate

    def evaluate(self, spent_this_month_jpy: Decimal) -> BudgetState:
        budget = self.cost.llm_budget_jpy
        spent = dec(spent_this_month_jpy)
        used_pct = (spent / budget * Decimal(100)) if budget > 0 else Decimal(100)
        # Two rungs, not three. The old middle rung cut the day to one debate
        # at 80% spent — a count standing in for money. Daily pacing does that
        # job continuously and in the right unit, so what is left here is the
        # floor: at 100% nothing may call a model at all.
        ladder = (BudgetLadder.STOPPED
                  if used_pct >= self.cost.stop_threshold_pct
                  else BudgetLadder.NORMAL)
        return BudgetState(ladder=ladder, spent_jpy=spent, budget_jpy=budget,
                           used_pct=used_pct)

    def remaining_jpy(self, spent_this_month_jpy: Decimal) -> Decimal:
        return max(ZERO, self.cost.llm_budget_jpy - dec(spent_this_month_jpy))

    def daily_allowance_jpy(self, spent_this_month_jpy: Decimal,
                            now: datetime) -> Decimal:
        """What today may still spend, pacing the month across its days.

            (budget - spent so far) / days left this month * multiplier

        This replaces a fixed cap on debates per day. A count was only ever a
        proxy for money, and a poor one: a cycle costs seven or nine calls
        depending on whether consensus is reached, and cache hits move it
        again, so "eight debates" guaranteed no particular sum.

        The division is what makes busy and quiet days both fine. Spending
        heavily today leaves less to divide tomorrow, so the month self-corrects
        without ever refusing a trade for having used up a quota. A quiet run of
        days leaves the whole remaining budget available for a volatile one.
        """
        remaining = self.remaining_jpy(spent_this_month_jpy)
        if remaining <= ZERO:
            return ZERO
        days_left = Decimal(jst_days_remaining_in_month(now))
        return remaining / days_left * self.cost.daily_allowance_multiplier
