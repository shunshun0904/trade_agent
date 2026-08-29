"""Spec 14 — the acceptance checklist, one test per box.

Each test names the checklist item it discharges. If one of these fails, the
system does not meet the specification, regardless of what the rest of the
suite says.
"""

from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from trade_agent.errors import GuardRejection, LockNotAcquired
from trade_agent.guards.deterministic import DeterministicGuard
from trade_agent.llm.budget import BudgetLadder, CostMeter
from trade_agent.models.agent_io import AnalystOutput, StrategyOutput
from trade_agent.models.state import CycleTrigger, Halt, HaltReason
from trade_agent.orchestrator.cycle import DecisionCycle
from trade_agent.orchestrator.screening import evaluate_triggers
from trade_agent.roles import STRATEGISTS
from trade_agent.risk.boredom import evaluate_boredom

E = Decimal
TEMPLATE = Path(__file__).resolve().parents[1] / "template.yaml"


# 1. Running `decide` twice with the same cycle id places exactly one order.
def test_1_same_cycle_id_places_one_order(ctx, llm):
    first = DecisionCycle(ctx, trigger=CycleTrigger.MANUAL, cycle_id="acc-1").run()
    assert first.traded

    with pytest.raises(LockNotAcquired):
        DecisionCycle(ctx, trigger=CycleTrigger.MANUAL, cycle_id="acc-1").run()

    assert len(ctx.exchange.orders_sent) == 1
    entries = [o for o in ctx.store.orders.list_recent(50)
               if o.purpose.value == "entry"]
    assert len(entries) == 1


# 2. A tick during a decide cycle causes no double exit and no state tearing.
def test_2_a_tick_during_decide_does_not_double_settle(ctx, llm):
    from trade_agent.handlers.tick import run as tick_run
    from trade_agent.storage.base import LOCK_DECIDE, LOCK_EXECUTION

    cycle = DecisionCycle(ctx, trigger=CycleTrigger.MANUAL, cycle_id="acc-2")
    # Hold both locks the way a mid-flight decide cycle would.
    assert ctx.store.locks.acquire(LOCK_DECIDE, cycle.invocation_id, 600,
                                   ctx.clock.now())
    assert ctx.store.locks.acquire(LOCK_EXECUTION, cycle.invocation_id, 600,
                                   ctx.clock.now())

    result = tick_run(ctx)
    assert result.get("skipped"), "the tick must stand aside, not read torn state"
    assert result["heartbeat"] is True
    assert ctx.exchange.orders_sent == []


# 3. Every function has asynchronous retries disabled.
def test_3_all_functions_disable_async_retries():
    template = _load_template()
    functions = {name: res for name, res in template["Resources"].items()
                 if res["Type"] == "AWS::Serverless::Function"}
    assert functions, "no Lambda functions found in the template"
    for name, resource in functions.items():
        config = resource["Properties"].get("EventInvokeConfig")
        assert config is not None, f"{name} has no EventInvokeConfig"
        assert config["MaximumRetryAttempts"] == 0, f"{name} may be retried by AWS"

    # The schedules must not retry either.
    for name, resource in functions.items():
        for event in (resource["Properties"].get("Events") or {}).values():
            policy = event["Properties"].get("RetryPolicy")
            assert policy and policy["MaximumRetryAttempts"] == 0, \
                f"{name} has a retrying schedule"


# 4. A crash right after ordering does not double-order or lose the position.
def test_4_crash_after_ordering_is_recoverable(ctx, clock):
    """Simulate: the order reached bitbank, the process died before the id was
    written, and the function restarts."""
    from trade_agent.models.trading import (
        OrderIntent, OrderPurpose, OrderRecord, OrderStatus, OrderType, Side)

    intent = OrderIntent(
        client_order_id="crashed", cycle_id="acc-4", pair="btc_jpy",
        side=Side.BUY, order_type=OrderType.LIMIT, qty_btc=E("0.0003"),
        price=ctx.exchange.market.price * E("0.9975"), post_only=True,
        purpose=OrderPurpose.ENTRY, trade_id="trd-acc-4")
    ctx.exchange.market.hold_still()
    ctx.store.orders.put_pending(OrderRecord.from_intent(intent, clock.now()))
    ctx.exchange.create_order(intent)          # reached the exchange
    sent_before = len(ctx.exchange.orders_sent)

    executor = ctx.executor()                  # restart
    notes = executor.reconcile_pending()

    record = ctx.store.orders.get("crashed")
    assert record.exchange_order_id is not None, "the position was lost"
    assert record.status is not OrderStatus.PENDING
    assert len(ctx.exchange.orders_sent) == sent_before, "a duplicate was placed"
    assert any("adopted" in note for note in notes)


# 5. The guard rejects an inverted stop, an unaffordable size and a fabricated
#    indicator value.
def test_5a_guard_rejects_a_stop_above_entry(config, snapshot):
    guard = DeterministicGuard(config, snapshot)
    entry = snapshot.last_price
    with pytest.raises(GuardRejection):
        guard.validate_strategy(StrategyOutput(
            action="buy", entry=float(entry),
            stop_loss=float(entry * E("1.01")),
            take_profit=float(entry * E("1.02")), confidence=0.5,
            thesis="x", invalidation="y"))


def test_5b_guard_rejects_a_size_the_balance_cannot_cover(config, snapshot):
    guard = DeterministicGuard(config, snapshot)
    entry = snapshot.last_price
    violations = guard.check_executable(
        entry=entry, stop_loss=entry * E("0.999"),
        take_profit=entry * E("1.02"), qty_btc=E("1.0"),
        jpy_available=snapshot.account.jpy_free)
    assert any("利用可能残高" in v for v in violations)


def test_5c_guard_rejects_a_fabricated_indicator_value(config, snapshot):
    guard = DeterministicGuard(config, snapshot)
    wrong = snapshot.indicators.rsi + E(25)
    with pytest.raises(GuardRejection) as excinfo:
        guard.validate_analyst(AnalystOutput(
            regime="range", confidence=0.5, key_indicators=["rsi"],
            summary=f"RSIは{wrong}である。", risks=[]))
    assert any("引用値が実値と一致しない" in v for v in excinfo.value.violations)


# 6. The phase-1 proposals never see each other, provably from the logs.
def test_6_phase_one_proposals_are_independent(ctx, llm):
    DecisionCycle(ctx, trigger=CycleTrigger.MANUAL, cycle_id="acc-6").run()
    calls = ctx.store.agent_calls.list_for_cycle("acc-6")
    proposals = [c for c in calls if c.agent in STRATEGISTS]

    assert len(proposals) == len(STRATEGISTS)
    for call in proposals:
        assert set(call.saw_agents) == {"analyst"}
        body = ctx.store.blobs.get_json(call.io_s3_key)
        for other in STRATEGISTS:
            assert other not in body["task"]

    from trade_agent.mcp.tools import call_tool
    log = call_tool(ctx, "get_agent_log", {"cycle_id": "acc-6"})
    assert log["independent_proposals_verified"] is True


# 7. The boredom rule cannot fire while the kill switch is engaged.
def test_7_boredom_rule_is_silent_under_the_kill_switch(config, ctx, clock):
    # Shipped default is now off (screening.consensus_min is 1, which leaves
    # this rule no lever). Spec 7's guarantee is still worth checking.
    config.boredom.enabled = True
    state = ctx.load_state()
    state.last_entry_at = clock.now()
    ctx.save_state(state)
    clock.advance(hours=200)

    state = ctx.load_state()
    state.kill_switch = True
    decision = evaluate_boredom(
        config, state, clock.now(),
        [Halt(reason=HaltReason.KILL_SWITCH, detail="engaged")])
    assert not decision.triggered
    # not relaxed: still whatever the normal threshold is configured to be
    assert decision.consensus_min == config.screening.consensus_min


# 8. At 72 hours and one minute the rule fires and records probe=true.
def test_8_boredom_rule_fires_at_72h_and_records_a_probe(ctx, llm, clock, config):
    # Shipped default is now off (screening.consensus_min is 1, which leaves
    # this rule no lever). Spec 7's guarantee is still worth checking.
    config.boredom.enabled = True
    state = ctx.load_state()
    state.last_entry_at = clock.now()
    ctx.save_state(state)

    clock.advance(hours=71, minutes=59)
    assert not evaluate_boredom(config, ctx.load_state(), clock.now(), []).triggered

    clock.advance(minutes=2)  # 72h 01m
    decision = evaluate_boredom(config, ctx.load_state(), clock.now(), [])
    assert decision.triggered
    assert decision.consensus_min == config.boredom.relaxed_consensus_min

    llm.bias = "wait"  # not one strategist wants to buy
    outcome = DecisionCycle(ctx, trigger=CycleTrigger.BOREDOM,
                            cycle_id="acc-8").run()
    assert outcome.probe is True
    assert outcome.traded, outcome.no_trade_reason
    assert outcome.plan.probe is True
    assert outcome.plan.qty_btc == config.exchange.min_order_btc

    trade = ctx.store.trades.get(outcome.plan.trade_id)
    assert trade.probe is True

    # The probe's stop must sit inside the tight configured distance.
    distance = (outcome.plan.entry - outcome.plan.stop_loss) / outcome.plan.entry
    assert distance * E(100) <= config.boredom.probe_sl_pct + E("0.01")


# 9. Spending is paced across the month; 100% stops LLM calls entirely.
def test_9_budget_paces_daily_then_stops(config, ctx, llm):
    """The 80% rung is gone: it cut the day to one debate, which is a count
    pretending to be a budget. Pacing replaces it and the hard stop stays."""
    from datetime import datetime, timezone

    meter = CostMeter(config.llm, config.cost)
    budget = config.cost.llm_budget_jpy
    now = datetime(2026, 8, 15, 3, 0, tzinfo=timezone.utc)

    assert meter.evaluate(budget * E("0.79")).ladder is BudgetLadder.NORMAL
    # Still allowed to trade at 80%, but with proportionally less to spend.
    at_80 = meter.evaluate(budget * E("0.80"))
    assert at_80.ladder is BudgetLadder.NORMAL
    assert (meter.daily_allowance_jpy(budget * E("0.80"), now)
            < meter.daily_allowance_jpy(E(0), now))

    stopped = meter.evaluate(budget)
    assert stopped.ladder is BudgetLadder.STOPPED
    assert not stopped.llm_allowed
    assert meter.daily_allowance_jpy(budget, now) == 0

    state = ctx.load_state()
    state.monthly.llm_cost_jpy = budget
    ctx.save_state(state)
    outcome = DecisionCycle(ctx, trigger=CycleTrigger.MANUAL,
                            cycle_id="acc-9").run()
    assert not outcome.traded
    assert llm.calls == []


def test_9b_deterministic_monitoring_survives_an_exhausted_budget(ctx, llm):
    """Spec 11: the stop loss is not funded by the LLM budget."""
    from trade_agent.handlers.tick import run as tick_run

    state = ctx.load_state()
    state.monthly.llm_cost_jpy = ctx.config.cost.llm_budget_jpy * E(2)
    ctx.save_state(state)

    result = tick_run(ctx)
    assert "error" not in result
    assert result["equity_jpy"] > 0
    assert llm.calls == []


# 10. Screening runs only when flat and calls no model when nothing fires.
def test_10_screening_is_free_when_nothing_triggers(ctx, llm, clock):
    from trade_agent.handlers.screen import run as screen_run

    state = ctx.load_state()
    state.last_floor_run_at = clock.now()
    state.last_full_debate_at = clock.now() - timedelta(hours=2)
    ctx.save_state(state)

    ctx.exchange.market.quiet()  # range-bound: no trigger condition fires

    result = screen_run(ctx)
    assert result["debate"] is False
    assert llm.calls == [], "a quiet screening pass must cost nothing"
    assert ctx.exchange.orders_sent == []


def test_10b_screening_is_suppressed_while_a_position_is_open(ctx, snapshot, clock):
    state = ctx.load_state()
    state.open_position = object()
    snapshot.indicators.rsi = E(15)  # a condition that would otherwise fire
    result = evaluate_triggers(ctx.config, state, snapshot, clock.now())
    assert not result.should_debate
    assert "position is open" in result.suppressed_by


# 11. Past the daily debate cap, a firing trigger starts nothing — except the
#     boredom rule.
def test_11_spending_the_days_allowance_blocks_further_cycles(ctx, snapshot,
                                                              clock, config):
    meter = CostMeter(config.llm, config.cost)
    state = ctx.load_state()
    state.last_floor_run_at = clock.now()
    state.last_full_debate_at = clock.now() - timedelta(hours=2)
    snapshot.indicators.rsi = E(15)
    state.daily.llm_cost_jpy = meter.daily_allowance_jpy(E(0), clock.now())

    result = evaluate_triggers(config, state, snapshot, clock.now(),
                               cost_meter=meter)
    assert not result.should_debate
    assert "allowance" in result.suppressed_by


def test_11b_the_boredom_rule_is_exempt_from_the_daily_cap(ctx, llm, clock, config):
    # Shipped default is now off (screening.consensus_min is 1, which leaves
    # this rule no lever). Spec 7's guarantee is still worth checking.
    config.boredom.enabled = True
    state = ctx.load_state()
    state.last_entry_at = clock.now()
    state.daily.full_debates = 50   # a count no longer gates anything
    ctx.save_state(state)
    clock.advance(hours=73)

    outcome = DecisionCycle(ctx, trigger=CycleTrigger.BOREDOM,
                            cycle_id="acc-11b").run()
    assert outcome.probe is True
    assert outcome.llm_calls > 0, "the 3-day rule must still be able to run"


# 12. The tables hold enough to audit both the money and the reasoning.
def test_12_every_decision_is_externally_auditable(ctx, llm, clock):
    outcome = DecisionCycle(ctx, trigger=CycleTrigger.MANUAL,
                            cycle_id="acc-12").run()
    assert outcome.traded

    trade = ctx.store.trades.get(outcome.plan.trade_id)
    assert trade.cycle_id == "acc-12"
    assert trade.entry_order_id
    assert trade.stop_loss and trade.take_profit
    assert trade.judge_output_id

    calls = ctx.store.agent_calls.list_for_cycle("acc-12")
    assert {c.agent for c in calls} == {"analyst", *STRATEGISTS}
    for call in calls:
        assert call.model and call.called_at and call.io_s3_key
        body = ctx.store.blobs.get_json(call.io_s3_key)
        assert body["task"] and body["role_instruction"]
        assert body["output"] is not None

    order = ctx.store.orders.get(trade.entry_order_id)
    assert order is not None and order.cycle_id == "acc-12"

    # And the whole chain is reachable from the owner's read-only tools.
    from trade_agent.mcp.tools import call_tool
    log = call_tool(ctx, "get_agent_log", {"trade_id": trade.trade_id,
                                           "include_bodies": True})
    assert log["found"] is True
    assert log["calls"][0]["body"] is not None


def _load_template() -> dict:
    class Loader(yaml.SafeLoader):
        pass

    Loader.add_multi_constructor(
        "!", lambda loader, suffix, node: {"__cfn__": suffix})
    return yaml.load(TEMPLATE.read_text(), Loader=Loader)
