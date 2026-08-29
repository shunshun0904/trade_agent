"""`screen` Lambda — every 30 minutes, no LLM (spec 9, 17.1).

Runs only when flat. Evaluates deterministic trigger conditions and, when one
fires, hands off to `decide` with an explicit cycle id. The cycle id is minted
here rather than inside `decide` so the invoke is idempotent: a retried invoke
carries the same id, and the per-cycle lock turns the duplicate into a no-op.

Cost of a screening run that finds nothing: zero. That is the point — it is
what lets the system look at the market 48 times a day on a 2,900 yen budget.
"""

from __future__ import annotations

import json
import logging
import os

from ..errors import ExchangeError
from ..models.state import CycleTrigger
from ..orchestrator.context import AppContext
from ..orchestrator.cycle import new_cycle_id
from ..orchestrator.screening import evaluate_triggers
from ..storage.base import LOCK_EXECUTION
from ..timeutil import iso
from .common import run_handler

log = logging.getLogger(__name__)

DECIDE_FUNCTION_ENV = "TA_DECIDE_FUNCTION_NAME"


def handler(event=None, context=None, *, ctx: AppContext | None = None) -> dict:
    return run_handler("screen", run, ctx=ctx)


def run(ctx: AppContext) -> dict:
    now = ctx.clock.now()
    state = ctx.load_state()
    budget = ctx.budget_state(state)

    if not budget.llm_allowed:
        return {"debate": False, "reason": "monthly LLM budget exhausted"}

    halts = ctx.risk.evaluate_halts(state, now, budget_stopped=False)

    if state.has_position():
        # The other half of the trade. Nothing here can open anything — the
        # review may only tighten an existing position, and with the feature
        # off it costs one config read and returns.
        return _review_open_position(ctx, state, now)

    snapshot = None
    if not halts:
        try:
            snapshot = ctx.snapshot_builder().build(position=None)
        except ExchangeError as exc:
            return {"debate": False, "reason": f"market data unavailable: {exc}"}

    result = evaluate_triggers(ctx.config, state, snapshot, now, halts=halts,
                               cost_meter=ctx.cost_meter)

    if ctx.config.screening.scout_mode and result.should_debate:
        # Optional cheap second opinion before committing to a full debate.
        verdict = _scout(ctx, snapshot, state)
        if verdict is not None and not verdict:
            return {"debate": False, "reason": "scout judged it not worth a debate",
                    "triggers": result.reasons}

    if not result.should_debate:
        return {"debate": False, "reason": result.summary(),
                "debates_today": state.daily.full_debates,
                "llm_spent_today_jpy": float(state.daily.llm_cost_jpy),
                "llm_allowance_today_jpy": float(
                    ctx.cost_meter.daily_allowance_jpy(
                        state.monthly.llm_cost_jpy, now))}

    cycle_id = new_cycle_id(now, result.trigger)
    invoked = _invoke_decide(ctx, cycle_id, result.trigger)
    return {"debate": True, "cycle_id": cycle_id, "trigger": str(result.trigger),
            "triggers": result.reasons, "invoked": invoked}


def _scout(ctx: AppContext, snapshot, state) -> bool | None:
    """Spec 9 scout mode. Off by default; its cost counts against the budget."""
    if snapshot is None:
        return None
    from ..agents.base import AgentRunner
    from ..agents.roster import run_scout

    runner = AgentRunner(llm=ctx.llm, config=ctx.config, store=ctx.store,
                         clock=ctx.clock, cycle_id=f"scout-{snapshot.snapshot_id}",
                         router=ctx.router)
    runner.set_prefix(snapshot.to_prompt_json(), lessons=[], trade_digest="",
                      state_digest="")
    try:
        verdict = run_scout(runner)
    except Exception as exc:  # noqa: BLE001 - a failed scout must not block the cycle
        log.warning("scout call failed: %s", exc)
        return None
    state.daily.llm_cost_jpy += runner.usage.cost_jpy
    state.monthly.llm_cost_jpy += runner.usage.cost_jpy
    ctx.save_state(state)
    return verdict.worth_full_debate


def _invoke_decide(ctx: AppContext, cycle_id: str, trigger: CycleTrigger) -> str:
    """Async invoke with retries disabled (spec 17.3).

    Without a Lambda function name configured — locally, or in tests — the
    cycle is run inline instead, so the same code path is exercised either way.
    """
    payload = {"cycle_id": cycle_id, "trigger": str(trigger)}
    function_name = os.environ.get(DECIDE_FUNCTION_ENV)
    if not function_name:
        from .decide import run as decide_run

        decide_run(ctx, payload)
        return "inline"

    import boto3

    client = boto3.client("lambda")
    client.invoke(FunctionName=function_name, InvocationType="Event",
                  Payload=json.dumps(payload).encode())
    return function_name


def _review_open_position(ctx: AppContext, state, now) -> dict:
    """Spec D-1: review the open position, or explain why not.

    Every failure here is a no-op rather than an error. The deterministic exits
    own this position and keep working — the exchange-side stop is live, the
    5-minute tick evaluates the target — so a review that cannot run costs the
    system nothing but the chance to tighten.
    """
    from ..orchestrator.exit_review import should_review

    position = state.open_position
    if not ctx.config.exit_review.enabled:
        return {"debate": False, "reason": "a position is open; exit review is off"}

    try:
        snapshot = ctx.snapshot_builder().build(position=position)
    except ExchangeError as exc:
        return {"debate": False, "reason": f"market data unavailable: {exc}"}

    decision = should_review(ctx.config, state, snapshot, now,
                             cost_meter=ctx.cost_meter)
    if not decision:
        return {"debate": False, "exit_review": False, "reason": decision.reason}

    outcome = _run_exit_review(ctx, state, snapshot, position, now)
    return {"debate": False, "exit_review": True, "trigger": decision.reason,
            **outcome}


def _run_exit_review(ctx: AppContext, state, snapshot, position, now) -> dict:
    """One LLM call, then apply whatever survives the guard.

    Wrapped whole: an exception anywhere leaves the position exactly as it was.
    That is the fallback the whole design rests on — an unavailable model must
    be indistinguishable from the system as it behaved before this existed.
    """
    from ..agents.base import AgentRunner
    from ..agents.roster import run_exit
    from ..guards.deterministic import DeterministicGuard

    guard = DeterministicGuard(ctx.config, snapshot)
    runner = AgentRunner(llm=ctx.llm, config=ctx.config, store=ctx.store,
                         clock=ctx.clock,
                         cycle_id=f"exit-{position.trade_id}-{position.review_count}",
                         router=ctx.router)
    runner.set_prefix(snapshot.to_prompt_json(), lessons=[], trade_digest="",
                      state_digest="")

    last_price = snapshot.last_price
    payload = {
        "entry_price": float(position.entry_price),
        "qty_btc": float(position.qty_btc),
        "stop_loss": float(position.stop_loss),
        "take_profit": float(position.take_profit),
        "opened_at": iso(position.opened_at),
        "hours_held": round((now - position.opened_at).total_seconds() / 3600, 1),
        "unrealized_pnl_jpy": float(position.unrealized_pnl_jpy(last_price)),
        "protection": position.protection,
        "original_thesis": position.thesis,
        "original_invalidation": position.invalidation,
    }
    allowed = {
        "hold": "何も変えない",
        "raise_stop": f"{float(position.stop_loss)} より高い値のみ",
        "lower_target": f"{float(position.take_profit)} より低い値のみ",
        "note": "損切りを広げる・利確を遠ざけることはできない",
    }

    try:
        decision = run_exit(
            runner, position=payload, allowed=allowed,
            validator=lambda o: guard.validate_exit(o, position=position))
    except Exception as exc:  # noqa: BLE001 - never let a review touch the exit path
        log.warning("exit review failed for %s: %s", position.trade_id, exc)
        _charge(ctx, state, runner)
        return {"applied": False, "error": str(exc)}

    executor = ctx.executor(owner=f"exit-review-{now.timestamp():.0f}")
    if not executor.acquire_lock(LOCK_EXECUTION):
        _charge(ctx, state, runner)
        return {"applied": False, "error": "execution lock held by the tick"}
    try:
        manager = ctx.position_manager(executor)
        update = manager.apply_exit_decision(
            state, position, decision, last_price=last_price, now=now)
    finally:
        executor.release_lock(LOCK_EXECUTION)

    _charge(ctx, state, runner)
    return {"applied": True, "action": decision.action,
            "invalidation_hit": decision.invalidation_hit,
            "rationale": decision.rationale,
            "closed_trade": update.closed.trade_id if update.closed else None,
            "notes": update.notes}


def _charge(ctx: AppContext, state, runner) -> None:
    """Bill the call and persist, whichever way the review ended."""
    state.daily.llm_cost_jpy += runner.usage.cost_jpy
    state.monthly.llm_cost_jpy += runner.usage.cost_jpy
    ctx.save_state(state)
