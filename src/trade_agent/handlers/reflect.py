"""`reflect` Lambda — post-settlement analysis (spec 9, 11, 17.1).

Two rules shape this function.

Aggregates only. Spec 9 forbids drawing a lesson from a single trade, and the
statistics handed to A7 are built from at least `min_trades_for_lessons`
closed, non-probe trades. Below that threshold the function stores nothing and
says why: a lesson learned from four trades is noise that every later cycle
will then treat as knowledge.

Probe trades are excluded. They are entertainment, not strategy (spec 7), and
mixing them into the statistics would teach the system from its own noise.

Latency does not matter here, so the Batch API's 50% discount applies
(spec 11). The batch is submitted and the id parked on system state; the
5-minute tick collects the result when it lands.
"""

from __future__ import annotations

import logging
import uuid
from decimal import Decimal

from ..agents.base import AgentRunner
from ..agents.roster import run_reflect
from ..models.agent_io import ReflectOutput
from ..models.trading import TradeRecord
from ..money import ZERO, jpy
from ..orchestrator.context import AppContext
from ..storage.base import StoredLesson
from .common import run_handler

log = logging.getLogger(__name__)


def handler(event=None, context=None, *, ctx: AppContext | None = None) -> dict:
    return run_handler("reflect", lambda c: run(c, event or {}), ctx=ctx)


def run(ctx: AppContext, event: dict | None = None) -> dict:
    state = ctx.load_state()
    budget = ctx.budget_state(state)
    if not budget.llm_allowed:
        return {"skipped": "monthly LLM budget exhausted"}
    if state.pending_reflect_batch_id:
        return {"skipped": "a reflection batch is already in flight",
                "batch_id": state.pending_reflect_batch_id}

    trades = [t for t in ctx.store.trades.list_recent(
        ctx.config.reflect.lookback_trades, include_probe=False) if t.closed]
    minimum = ctx.config.reflect.min_trades_for_lessons
    if len(trades) < minimum:
        return {"skipped": f"only {len(trades)} closed non-probe trades; "
                           f"{minimum} required before drawing a lesson",
                "trades": len(trades)}

    statistics = build_statistics(trades, ctx.config.reflect.lookback_trades)
    cycle_id = f"reflect-{uuid.uuid4().hex[:8]}"
    runner = _runner(ctx, cycle_id, statistics)

    if ctx.config.llm.batch_api_for_reflect and hasattr(ctx.llm, "submit_batch"):
        batch_id = _submit_batch(ctx, runner, statistics)
        if batch_id:
            state.pending_reflect_batch_id = batch_id
            state.pending_reflect_batch_at = ctx.clock.now()
            ctx.save_state(state)
            return {"submitted_batch": batch_id, "trades": len(trades)}

    output = run_reflect(runner, statistics=statistics)
    stored = _store_lessons(ctx, output, cycle_id, len(trades))
    state.daily.llm_cost_jpy += runner.usage.cost_jpy
    state.monthly.llm_cost_jpy += runner.usage.cost_jpy
    ctx.save_state(state)
    return {"lessons": stored, "trades": len(trades),
            "llm_cost_jpy": float(runner.usage.cost_jpy)}


def collect_batch(ctx: AppContext, state) -> int:
    """Poll an outstanding batch. Returns the number of lessons stored.

    Called from the tick, so it must never raise into the caller's flow — the
    tick's job is the stop loss, not reflection.
    """
    batch_id = state.pending_reflect_batch_id
    if not batch_id or not hasattr(ctx.llm, "poll_batch"):
        return 0
    results = ctx.llm.poll_batch(batch_id, {"reflect": ReflectOutput})
    if results is None:
        return 0

    stored = 0
    for custom_id, response in results.items():
        cycle_id = f"batch-{batch_id[:8]}"
        stored += _store_lessons(ctx, response.parsed, cycle_id, 0)
        state.daily.llm_cost_jpy += response.cost_jpy
        state.monthly.llm_cost_jpy += response.cost_jpy
        log.info("collected reflection %s (%s lessons)", custom_id, stored)
    state.pending_reflect_batch_id = None
    state.pending_reflect_batch_at = None
    return stored


def build_statistics(trades: list[TradeRecord], lookback: int) -> dict:
    """Aggregate view of recent performance — the only thing A7 ever sees."""
    closed = [t for t in trades if t.closed and t.net_pnl_jpy is not None]
    wins = [t for t in closed if (t.net_pnl_jpy or ZERO) > 0]
    losses = [t for t in closed if (t.net_pnl_jpy or ZERO) <= 0]
    rr = [t.r_multiple() for t in closed if t.r_multiple() is not None]

    by_regime: dict[str, dict] = {}
    for trade in closed:
        bucket = by_regime.setdefault(trade.regime or "unknown",
                                      {"trades": 0, "wins": 0, "net_jpy": ZERO})
        bucket["trades"] += 1
        bucket["wins"] += 1 if (trade.net_pnl_jpy or ZERO) > 0 else 0
        bucket["net_jpy"] += trade.net_pnl_jpy or ZERO

    by_exit: dict[str, int] = {}
    for trade in closed:
        by_exit[trade.exit_reason or "unknown"] = by_exit.get(
            trade.exit_reason or "unknown", 0) + 1

    return {
        "window": f"直近{min(len(closed), lookback)}件の決済済みトレード(probe除外)",
        "trade_count": len(closed),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": round(len(wins) / len(closed), 3) if closed else None,
        "net_pnl_jpy": float(jpy(sum((t.net_pnl_jpy or ZERO for t in closed), ZERO))),
        "total_fees_jpy": float(jpy(sum((t.fee_jpy for t in closed), ZERO))),
        "average_r_multiple": float(round(sum(rr, ZERO) / Decimal(len(rr)), 3))
        if rr else None,
        "average_win_jpy": float(jpy(sum((t.net_pnl_jpy or ZERO for t in wins), ZERO)
                                     / Decimal(len(wins)))) if wins else None,
        "average_loss_jpy": float(jpy(sum((t.net_pnl_jpy or ZERO for t in losses), ZERO)
                                      / Decimal(len(losses)))) if losses else None,
        "by_regime": {k: {"trades": v["trades"], "wins": v["wins"],
                          "net_jpy": float(jpy(v["net_jpy"]))}
                      for k, v in by_regime.items()},
        "by_exit_reason": by_exit,
    }


def _runner(ctx: AppContext, cycle_id: str, statistics: dict) -> AgentRunner:
    runner = AgentRunner(llm=ctx.llm, config=ctx.config, store=ctx.store,
                         clock=ctx.clock, cycle_id=cycle_id, router=ctx.router)
    lessons = [f"[{row.regime_tag}] {row.text}"
               for row in ctx.store.lessons.list(limit=ctx.config.snapshot.lessons_in_prompt)]
    runner.set_prefix(
        "{}",  # reflection reasons about history, not about the live market
        lessons=lessons,
        trade_digest=str(statistics),
        state_digest="決済後の振り返りサイクル。市場スナップショットは使用しない。")
    return runner


def _submit_batch(ctx: AppContext, runner: AgentRunner, statistics: dict) -> str | None:
    from ..llm.base import LLMRequest
    from ..agents.prompts import ROLE_PROMPTS

    request = LLMRequest(
        agent="reflect", shared_prefix=runner.prefix,
        role_instruction=ROLE_PROMPTS["reflect"],
        task=str(statistics), output_model=ReflectOutput,
        model=ctx.config.llm.model, max_tokens=ctx.config.llm.max_tokens,
        use_batch=True)
    try:
        return ctx.llm.submit_batch([request])
    except Exception as exc:  # noqa: BLE001 - fall back to a synchronous call
        log.warning("batch submission failed, running synchronously: %s", exc)
        return None


def _store_lessons(ctx: AppContext, output: ReflectOutput, cycle_id: str,
                   trades_analysed: int) -> int:
    now = ctx.clock.now()
    limit = ctx.config.reflect.max_lessons_per_run
    for lesson in output.lessons[:limit]:
        ctx.store.lessons.put(StoredLesson(
            lesson_id=uuid.uuid4().hex[:12],
            created_at=now,
            text=lesson.text,
            regime_tag=str(lesson.regime_tag),
            evidence=lesson.evidence,
            confidence=lesson.confidence,
            trades_analysed=trades_analysed,
            source_cycle_id=cycle_id))
    return min(len(output.lessons), limit)
