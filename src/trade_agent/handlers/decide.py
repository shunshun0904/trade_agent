"""`decide` Lambda — the full debate (spec 9, 17.1).

Invoked asynchronously by `screen`, by the scheduled floor, or by hand. The
caller supplies the cycle id; this function's contract is that running it twice
with the same id places at most one order (spec 14).

It also owns the 21:00 JST daily report, which is written after the cycle so
the report reflects what the cycle just decided.
"""

from __future__ import annotations

import logging

from ..errors import LockNotAcquired
from ..models.state import CycleTrigger
from ..orchestrator.context import AppContext
from ..orchestrator.cycle import CycleOutcome, DecisionCycle
from ..storage.base import DailyReport
from ..timeutil import crossed_jst_time, jst_date_str
from .common import run_handler

log = logging.getLogger(__name__)


def handler(event=None, context=None, *, ctx: AppContext | None = None) -> dict:
    return run_handler("decide", lambda c: run(c, event or {}), ctx=ctx)


def run(ctx: AppContext, event: dict) -> dict:
    trigger = _trigger_from(event)
    cycle_id = event.get("cycle_id")
    cycle = DecisionCycle(ctx, trigger=trigger, cycle_id=cycle_id)

    try:
        outcome = cycle.run()
    except LockNotAcquired as exc:
        log.info("decide skipped: %s", exc)
        return {"skipped": str(exc), "cycle_id": cycle.cycle_id}

    _maybe_write_daily_report(ctx, outcome)
    return {
        "cycle_id": outcome.cycle_id,
        "trigger": str(outcome.trigger),
        "traded": outcome.traded,
        "probe": outcome.probe,
        "reason": outcome.no_trade_reason,
        "buy_proposals": outcome.buy_count,
        "consensus_min": outcome.consensus_min,
        "consensus": float(outcome.consensus) if outcome.consensus else None,
        "regime": outcome.regime,
        "llm_calls": outcome.llm_calls,
        "llm_cost_jpy": float(outcome.llm_cost_jpy),
        "cache_hits": outcome.cache_hits,
        "notes": outcome.notes,
    }


def _trigger_from(event: dict) -> CycleTrigger:
    raw = event.get("trigger")
    if raw:
        try:
            return CycleTrigger(raw)
        except ValueError:
            log.warning("unknown trigger %r; treating as manual", raw)
    return CycleTrigger.MANUAL


def _maybe_write_daily_report(ctx: AppContext, outcome: CycleOutcome) -> None:
    """Spec 9: the daily report is stored after the 21:00 JST cycle.

    No LLM call happens here. The cycle already composed its report from the
    agents' structured output (orchestrator/report.py); this adds the counters
    the owner asks about — equity, P&L, spend, consensus rate, idle time
    (spec 16.2).
    """
    now = ctx.clock.now()
    state = ctx.load_state()
    window = ctx.config.schedule.screen_minutes
    if not crossed_jst_time(state.last_daily_report_at, now,
                            ctx.config.schedule.daily_report_time_jst, window):
        return

    idle = state.hours_since_last_entry(now)
    body = outcome.report_text or (
        f"本日の最終サイクルは見送りでした。理由: {outcome.no_trade_reason}")
    consensus_rate = _consensus_rate(ctx)
    ctx.store.reports.put(DailyReport(
        jst_date=jst_date_str(now),
        created_at=now,
        headline=outcome.headline or "見送り",
        report_text=(
            f"{body}\n\n"
            f"[集計] equity {state.equity_jpy} 円 / 当日実現損益 "
            f"{state.daily.realized_pnl_jpy} 円 / 当月LLM費 "
            f"{state.monthly.llm_cost_jpy:.1f} 円 (予算 {ctx.config.cost.llm_budget_jpy} 円)"
            f" / フル議論 {state.daily.full_debates} 回"
            + (f" / 最終約定から {idle:.1f} 時間" if idle is not None else
               " / 約定履歴なし")),
        equity_jpy=state.equity_jpy,
        realized_pnl_jpy=state.daily.realized_pnl_jpy,
        llm_cost_month_jpy=state.monthly.llm_cost_jpy,
        consensus_rate=consensus_rate,
        hours_since_last_entry=idle))
    state.last_daily_report_at = now
    ctx.save_state(state)


def _consensus_rate(ctx: AppContext) -> float | None:
    """Share of recent cycles that reached consensus, from the judge call log."""
    calls = [c for c in _recent_judge_calls(ctx)]
    if not calls:
        return None
    return round(sum(1 for c in calls if c.ok) / len(calls), 3)


def _recent_judge_calls(ctx: AppContext):
    lister = getattr(ctx.store.agent_calls, "list_all", None)
    if lister is None:
        return []
    return [c for c in lister() if c.agent == "judge"][-30:]
