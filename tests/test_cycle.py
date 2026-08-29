"""The decision cycle end to end (spec 4.1, 5, 8)."""

from decimal import Decimal

import pytest

from trade_agent.errors import LockNotAcquired
from trade_agent.models.state import CycleTrigger
from trade_agent.orchestrator.cycle import DecisionCycle, entry_order_id
from trade_agent.roles import STRATEGISTS

E = Decimal


def _cycle(ctx, **kwargs) -> DecisionCycle:
    return DecisionCycle(ctx, trigger=CycleTrigger.MANUAL, **kwargs)


def test_a_consensus_cycle_places_one_entry(ctx, llm):
    llm.bias = "mixed"  # the first strategist proposes a buy
    outcome = _cycle(ctx, cycle_id="cyc-a").run()
    assert outcome.traded, outcome.no_trade_reason
    assert outcome.buy_count == 1
    assert outcome.buy_count >= ctx.config.screening.consensus_min
    assert len(ctx.exchange.orders_sent) == 1
    assert ctx.store.orders.get(entry_order_id("cyc-a")) is not None


def test_no_buy_proposal_at_all_is_a_no_trade(ctx, llm):
    llm.bias = "wait"
    outcome = _cycle(ctx, cycle_id="cyc-b").run()
    assert not outcome.traded
    assert outcome.buy_count == 0
    assert "consensus not reached" in outcome.no_trade_reason
    assert ctx.exchange.orders_sent == []


def test_every_proposal_is_made_blind(ctx, llm):
    _cycle(ctx, cycle_id="cyc-c").run()
    calls = ctx.store.agent_calls.list_for_cycle("cyc-c")
    proposals = [c for c in calls if c.agent in STRATEGISTS]
    assert len(proposals) == len(STRATEGISTS)
    for call in proposals:
        # Spec 4.1: phase 1 sees the analyst and nothing else.
        assert set(call.saw_agents) == {"analyst"}


def test_critiques_never_reveal_the_author(ctx, llm):
    _cycle(ctx, cycle_id="cyc-d").run()
    for call in ctx.store.agent_calls.list_for_cycle("cyc-d"):
        if not call.agent.startswith("critique:"):
            continue
        body = ctx.store.blobs.get_json(call.io_s3_key)
        for agent in STRATEGISTS:
            assert agent not in body["task"]
        assert '"agent"' not in body["task"]


def test_every_call_is_logged_with_cost(ctx, llm):
    _cycle(ctx, cycle_id="cyc-e").run()
    calls = ctx.store.agent_calls.list_for_cycle("cyc-e")
    # analyst + N proposals
    assert len(calls) == 1 + len(STRATEGISTS)
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


def test_the_final_structural_check_stops_the_trade(ctx, llm, monkeypatch):
    """The last gate is arithmetic now, not an agent.

    A4 used to sit here and could veto. It was removed because the guard
    rejected its answer whenever its numbers disagreed with Python's, which
    left it vetoing arithmetic it was not allowed to change. What must still
    hold is that a failing check stops the order.
    """
    from trade_agent.guards import deterministic

    monkeypatch.setattr(deterministic.DeterministicGuard, "check_executable",
                        lambda *a, **k: ["数量が最小注文数量の整数倍でない"])
    outcome = _cycle(ctx, cycle_id="cyc-j").run()

    assert not outcome.traded
    assert "final structural check failed" in outcome.no_trade_reason
    assert "整数倍" in outcome.no_trade_reason
    assert ctx.exchange.orders_sent == []


def test_the_strategist_is_the_last_agent_in_the_chain(ctx, llm):
    """Nothing runs after A2. Every gate past it is deterministic."""
    _cycle(ctx, cycle_id="cyc-k").run()
    agents = [c.agent for c in ctx.store.agent_calls.list_for_cycle("cyc-k")]
    assert agents == ["analyst", *STRATEGISTS]
    for gone in ("critique", "judge", "risk", "auditor", "commander"):
        assert not any(gone in a for a in agents)


def test_a_cycle_costs_one_call_per_role(ctx, llm):
    """1 analyst + N proposals, and nothing conditional after them.

    Derived rather than written down. This asserted a literal 9 when there
    were three strategists, a critique round, a judge and a risk reviewer; a
    literal would now be asserting a cycle that cannot happen.
    """
    outcome = _cycle(ctx, cycle_id="cyc-l").run()
    assert outcome.traded
    assert outcome.llm_calls == 1 + len(STRATEGISTS)


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


def test_every_cycle_records_why_it_did_or_did_not_trade(ctx, llm, clock):
    """"It is not trading" is the owner's first question, and the answer
    differs across the eight ways a cycle can end without an order. The reason
    used to live only in the Lambda's return value and a CloudWatch line, with
    the daily report keeping whichever cycle happened to cross 21:00 JST."""
    from trade_agent.models.state import CycleTrigger
    from trade_agent.orchestrator.cycle import DecisionCycle

    outcome = DecisionCycle(ctx, trigger=CycleTrigger.MANUAL,
                            cycle_id="cyc-audit-1").run()

    events = [e for e in ctx.store.audit.list_recent(50)
              if e.event_id == "cycle:cyc-audit-1"]
    assert len(events) == 1, "exactly one audit row per cycle"

    event = events[0]
    assert event.action in {"traded", "no_trade"}
    assert event.action == ("traded" if outcome.traded else "no_trade")
    # The reason itself, not just the verdict — that is the point.
    if not outcome.traded:
        assert outcome.no_trade_reason in event.detail
    assert f"buys {outcome.buy_count}/{len(STRATEGISTS)}" in event.detail


def test_a_failing_audit_write_does_not_sink_the_cycle(ctx, llm, monkeypatch):
    """Bookkeeping runs after the decision is already made."""
    from trade_agent.models.state import CycleTrigger
    from trade_agent.orchestrator.cycle import DecisionCycle

    def explode(_event):
        raise RuntimeError("dynamo is having a day")

    monkeypatch.setattr(ctx.store.audit, "put", explode)
    outcome = DecisionCycle(ctx, trigger=CycleTrigger.MANUAL,
                            cycle_id="cyc-audit-2").run()
    assert outcome is not None


def test_the_invalidation_reaches_the_position(ctx, llm):
    """A2 is asked what would kill the idea, and until now the answer was
    dropped: cycle.py collected it into the proposals dict, the guard checked
    any indicator values quoted in it, and nothing downstream ever saw it. It
    is the exit review's central input, so it has to survive to the Position."""
    llm.bias = "buy"
    outcome = _cycle(ctx, cycle_id="cyc-inv").run()
    assert outcome.traded, outcome.no_trade_reason

    assert outcome.plan.invalidation, "the plan lost it before the executor"

    # The entry rests as a PostOnly limit, so the Position only exists once a
    # tick promotes the fill. Build it the way the executor does instead of
    # depending on fill timing.
    from trade_agent.execution.executor import build_position

    record = ctx.store.orders.get(outcome.plan.client_order_id)
    position = build_position(outcome.plan, record, ctx.clock.now())
    assert position.invalidation == outcome.plan.invalidation
    assert position.thesis == outcome.plan.thesis
