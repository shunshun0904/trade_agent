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
from ..orchestrator.screening import daily_debate_limit, evaluate_triggers
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
    snapshot = None
    if not state.has_position() and not halts:
        try:
            snapshot = ctx.snapshot_builder().build(position=None)
        except ExchangeError as exc:
            return {"debate": False, "reason": f"market data unavailable: {exc}"}

    limit = daily_debate_limit(ctx.config, budget)
    result = evaluate_triggers(ctx.config, state, snapshot, now, halts=halts,
                               debates_today=state.daily.full_debates,
                               daily_limit=limit)

    if ctx.config.screening.scout_mode and result.should_debate:
        # Optional cheap second opinion before committing to a full debate.
        verdict = _scout(ctx, snapshot, state)
        if verdict is not None and not verdict:
            return {"debate": False, "reason": "scout judged it not worth a debate",
                    "triggers": result.reasons}

    if not result.should_debate:
        return {"debate": False, "reason": result.summary(),
                "debates_today": state.daily.full_debates, "limit": limit}

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
