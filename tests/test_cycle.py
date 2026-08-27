"""The decision cycle end to end (spec 4.1, 5, 8)."""

from decimal import Decimal

import pytest

from trade_agent.errors import LockNotAcquired
from trade_agent.models.state import CycleTrigger
from trade_agent.orchestrator.cycle import DecisionCycle, entry_order_id

E = Decimal


def _cycle(ctx, **kwargs) -> DecisionCycle:
    return DecisionCycle(ctx, trigger=CycleTrigger.MANUAL, **kwargs)


def test_a_consensus_cycle_places_one_entry(ctx, llm):
    llm.bias = "mixed"  # two buy proposals, one wait -> 2 of 3
    outcome = _cycle(ctx, cycle_id="cyc-a").run()
    assert outcome.traded, outcome.no_trade_reason
    assert outcome.buy_count == 2
    assert len(ctx.exchange.orders_sent) == 1
    assert ctx.store.orders.get(entry_order_id("cyc-a")) is not None


def test_one_buy_of_three_is_a_no_trade(ctx, llm):
    llm.bias = "wait"
    outcome = _cycle(ctx, cycle_id="cyc-b").run()
    assert not outcome.traded
    assert "consensus not reached" in outcome.no_trade_reason
    assert ctx.exchange.orders_sent == []


def test_the_three_proposals_are_made_blind(ctx, llm):
    _cycle(ctx, cycle_id="cyc-c").run()
    calls = ctx.store.agent_calls.list_for_cycle("cyc-c")
    proposals = [c for c in calls if c.agent.startswith("strategy:")]
    assert len(proposals) == 3
    for call in proposals:
        # Spec 4.1: phase 1 sees the analyst and nothing else.
        assert set(call.saw_agents) == {"analyst"}


def test_critiques_never_reveal_the_author(ctx, llm):
    _cycle(ctx, cycle_id="cyc-d").run()
    for call in ctx.store.agent_calls.list_for_cycle("cyc-d"):
        if not call.agent.startswith("critique:"):
            continue
        body = ctx.store.blobs.get_json(call.io_s3_key)
        assert "strategy:trend" not in body["task"]
        assert "strategy:meanrev" not in body["task"]
        assert '"agent"' not in body["task"]


def test_every_call_is_logged_with_cost(ctx, llm):
    _cycle(ctx, cycle_id="cyc-e").run()
    calls = ctx.store.agent_calls.list_for_cycle("cyc-e")
    assert len(calls) >= 8  # analyst + 3 proposals + 3 critiques + judge + ...
    assert all(c.io_s3_key for c in calls)
    assert all(c.cost_jpy >= 0 for c in calls)
    assert sum(c.input_tokens for c in calls) > 0


def test_the_same_cycle_id_cannot_run_twice(ctx, llm):
    first = _cycle(ctx, cycle_id="cyc-f").run()
    assert first.traded
    with pytest.raises(LockNotAcquired):
        _cycle(ctx, cycle_id="cyc-f").run()
    assert len(ctx.exchange.orders_sent) == 1


def test_a_concurrent_cycle_is_refused(ctx, llm):
    from trade_agent.storage.base import LOCK_DECIDE

    ctx.store.locks.acquire(LOCK_DECIDE, "someone-else", 600, ctx.clock.now())
    with pytest.raises(LockNotAcquired):
        _cycle(ctx, cycle_id="cyc-g").run()
    assert ctx.exchange.orders_sent == []


def test_the_kill_switch_stops_the_cycle_before_any_llm_call(ctx, llm):
    state = ctx.load_state()
    state.kill_switch = True
    state.kill_switch_reason = "test"
    ctx.save_state(state)

    outcome = _cycle(ctx, cycle_id="cyc-h").run()
    from trade_agent.models.state import HaltReason

    assert not outcome.traded
    assert HaltReason.KILL_SWITCH in {h.reason for h in outcome.halts}
    assert llm.calls == []  # not a single token spent while halted


def test_an_exhausted_budget_stops_the_cycle_before_any_llm_call(ctx, llm):
    state = ctx.load_state()
    state.monthly.llm_cost_jpy = ctx.config.cost.llm_budget_jpy
    ctx.save_state(state)

    outcome = _cycle(ctx, cycle_id="cyc-i").run()
    assert not outcome.traded
    assert "budget" in outcome.no_trade_reason
    assert llm.calls == []


def test_a_risk_rejection_stops_the_trade(ctx, llm):
    llm.approve_risk = False
    outcome = _cycle(ctx, cycle_id="cyc-j").run()
    assert not outcome.traded
    assert "risk management rejected" in outcome.no_trade_reason
    assert ctx.exchange.orders_sent == []


def test_the_risk_agent_is_the_last_agent_in_the_chain(ctx, llm):
    """Nothing runs after A4. The remaining gates are all deterministic."""
    _cycle(ctx, cycle_id="cyc-k").run()
    agents = [c.agent for c in ctx.store.agent_calls.list_for_cycle("cyc-k")]
    assert agents[-1] == "risk"
    assert "auditor" not in agents
    assert "commander" not in agents


def test_a_cycle_costs_nine_calls(ctx, llm):
    """1 analyst + 3 proposals + 3 critiques + 1 judge + 1 risk."""
    outcome = _cycle(ctx, cycle_id="cyc-l").run()
    assert outcome.traded
    assert outcome.llm_calls == 9


def test_the_cycle_records_its_own_cost(ctx, llm):
    outcome = _cycle(ctx, cycle_id="cyc-m").run()
    assert outcome.llm_calls > 0
    assert outcome.llm_cost_jpy > 0
    state = ctx.load_state()
    assert state.monthly.llm_cost_jpy == outcome.llm_cost_jpy
    assert state.daily.llm_cost_jpy == outcome.llm_cost_jpy


def test_the_trade_row_exists_before_the_order_is_sent(ctx, llm):
    outcome = _cycle(ctx, cycle_id="cyc-n").run()
    assert outcome.traded
    trade = ctx.store.trades.get(outcome.plan.trade_id)
    assert trade is not None
    assert trade.stop_loss == outcome.plan.stop_loss
    assert trade.take_profit == outcome.plan.take_profit


def test_the_shared_prefix_is_identical_across_agents(ctx, llm):
    _cycle(ctx, cycle_id="cyc-o").run()
    prefixes = {call.shared_prefix for call in llm.calls}
    assert len(prefixes) == 1, "the cacheable prefix must be byte-identical"


def test_sizing_never_exceeds_the_per_trade_risk_limit(ctx, llm):
    outcome = _cycle(ctx, cycle_id="cyc-p").run()
    assert outcome.traded
    limit = ctx.risk.risk_limit_jpy(Decimal(10000), probe=False)
    assert outcome.plan.risk_jpy <= limit
