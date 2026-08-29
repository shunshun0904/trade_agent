"""The exit review: when it runs, what it may do, and what happens when it
cannot run at all.

The last of those matters most. Exits belong to the 5-minute tick and to a stop
order sitting on the exchange, and neither may become dependent on a model
being reachable — spec 11's rule is that the ability to open a position may be
lost while the ability to close one may not. So every failure path here has to
leave the position byte-for-byte as it was.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from trade_agent.execution.executor import build_position
from trade_agent.handlers.screen import run as screen_run
from trade_agent.handlers.tick import run as tick_run
from trade_agent.models.trading import ExecutionPlan, TradeRecord
from trade_agent.orchestrator.exit_review import should_review

E = Decimal
QTY = E("0.0006")


@pytest.fixture
def held(ctx, config):
    """A context with an open position and the review switched on."""
    config.exit_review.enabled = True
    entry = ctx.exchange.market.price
    plan = ExecutionPlan(
        cycle_id="cyc-h", trade_id="trd-h", entry=entry,
        stop_loss=entry * E("0.99"), take_profit=entry * E("1.02"),
        qty_btc=QTY, risk_jpy=E(45), client_order_id="entry-h",
        thesis="押し目からの反発を取る。",
        invalidation="直近安値を明確に割ったら論拠は消える。")
    ctx.store.trades.put(TradeRecord(
        trade_id="trd-h", cycle_id="cyc-h", pair="btc_jpy", qty_btc=QTY,
        entry_price=plan.entry, entry_order_id="entry-h",
        entry_at=ctx.clock.now(), stop_loss=plan.stop_loss,
        take_profit=plan.take_profit))
    executor = ctx.executor()
    assert executor.place_entry(plan).placed
    record = ctx.store.orders.get("entry-h")
    record.executed_qty_btc = QTY
    record.average_price = plan.entry

    state = ctx.load_state()
    state.open_position = build_position(plan, record, ctx.clock.now())
    ctx.save_state(state)
    return ctx


def _position(ctx):
    return ctx.load_state().open_position


def _levels(ctx) -> tuple[Decimal, Decimal]:
    p = _position(ctx)
    return p.stop_loss, p.take_profit


def _make_due(ctx):
    """Age the position past `max_idle_minutes` so the trigger fires.

    A freshly opened position is deliberately not reviewable — the floor runs
    from `opened_at` when there has been no review yet — so every test that
    wants to exercise the review itself has to say so.
    """
    state = ctx.load_state()
    state.open_position.last_review_at = ctx.clock.now() - timedelta(days=1)
    ctx.save_state(state)


# -- the trigger ----------------------------------------------------------

def test_it_is_off_by_default(ctx, config, snapshot, clock):
    """Shipped off so the deterministic exits produce a baseline first."""
    assert config.exit_review.enabled is False
    state = ctx.load_state()
    assert not should_review(config, state, snapshot, clock.now())


def test_it_waits_out_the_floor(held, config, snapshot, clock):
    state = held.load_state()
    state.open_position.last_review_at = clock.now()
    decision = should_review(config, state, snapshot, clock.now())
    assert not decision
    assert "floor" in decision.reason


def test_a_quiet_market_still_gets_reviewed_eventually(held, config, snapshot, clock):
    """Without this the review would never fire in a flat market — which is
    exactly where a thesis quietly stops being true."""
    state = held.load_state()
    state.open_position.last_review_at = clock.now() - timedelta(
        minutes=config.exit_review.max_idle_minutes + 1)
    state.open_position.last_review_price = snapshot.last_price
    decision = should_review(config, state, snapshot, clock.now())
    assert decision
    assert "without a review" in decision.reason


def test_a_move_of_half_an_atr_triggers_a_review(held, config, snapshot, clock):
    state = held.load_state()
    position = state.open_position
    position.last_review_at = clock.now() - timedelta(
        minutes=config.exit_review.min_minutes + 1)
    atr = Decimal(str(snapshot.indicators.atr))

    position.last_review_price = snapshot.last_price - atr * E("0.1")
    assert not should_review(config, state, snapshot, clock.now())

    position.last_review_price = snapshot.last_price - atr * E("0.6")
    assert should_review(config, state, snapshot, clock.now())


def test_one_position_cannot_spend_the_month(held, config, snapshot, clock):
    state = held.load_state()
    position = state.open_position
    position.last_review_at = clock.now() - timedelta(days=1)
    position.review_count = config.exit_review.max_reviews_per_position

    decision = should_review(config, state, snapshot, clock.now())
    assert not decision
    assert "review limit" in decision.reason


def test_a_working_exit_is_not_second_guessed(held, config, snapshot, clock):
    state = held.load_state()
    state.open_position.last_review_at = clock.now() - timedelta(days=1)
    state.open_position.exit_order_id = "exit-1"
    assert not should_review(config, state, snapshot, clock.now())


# -- applying a decision --------------------------------------------------

def test_holding_changes_nothing(held, llm, clock):
    llm.exit_bias = "hold"
    _make_due(held)
    before = _levels(held)

    result = screen_run(held)
    assert result["exit_review"] is True
    assert result["applied"] is True
    assert result["action"] == "hold"
    assert _levels(held) == before


def test_a_raised_stop_reaches_the_position(held, llm):
    llm.exit_bias = "stop"
    _make_due(held)
    old_stop, old_target = _levels(held)

    assert screen_run(held)["action"] == "raise_stop"
    new_stop, new_target = _levels(held)
    assert new_stop > old_stop
    assert new_target == old_target


def test_a_lowered_target_reaches_the_position(held, llm):
    llm.exit_bias = "target"
    _make_due(held)
    old_stop, old_target = _levels(held)

    assert screen_run(held)["action"] == "lower_target"
    new_stop, new_target = _levels(held)
    assert new_target < old_target
    assert new_stop == old_stop


# -- the fallback ---------------------------------------------------------

def test_a_failing_model_leaves_the_position_alone(held, llm, monkeypatch):
    """The property the whole design rests on: an unreachable model must be
    indistinguishable from the system as it behaved before this existed."""
    _make_due(held)
    before = _levels(held)

    def boom(request):
        raise RuntimeError("the API is down")

    monkeypatch.setattr(llm, "complete", boom)
    result = screen_run(held)

    assert result["applied"] is False
    assert _levels(held) == before
    assert _position(held) is not None


def test_a_rejected_decision_leaves_the_position_alone(held, llm, monkeypatch):
    """A widening suggestion is refused by the guard, retried, and then
    dropped — it must not reach the position by any path."""
    from trade_agent.models.agent_io import ExitOutput

    _make_due(held)
    before = _levels(held)
    old_stop = before[0]

    monkeypatch.setattr(
        llm, "_exit",
        lambda request, facts: ExitOutput(
            action="raise_stop", new_stop_loss=float(old_stop - 100000),
            new_take_profit=None, invalidation_hit=True,
            rationale="損切りに余裕を持たせたい。"))
    result = screen_run(held)

    assert result["applied"] is False
    assert _levels(held) == before


def test_the_budget_gate_stops_the_review(held, config, llm):
    state = held.load_state()
    state.monthly.llm_cost_jpy = config.cost.llm_budget_jpy
    state.daily.llm_cost_jpy = config.cost.llm_budget_jpy
    held.save_state(state)
    _make_due(held)
    before = _levels(held)

    result = screen_run(held)
    assert result.get("exit_review") is not True
    assert _levels(held) == before


# -- the tick stays deterministic -----------------------------------------

def test_the_tick_never_calls_a_model(held, llm):
    """The exit path may gain an optional review; it may not gain a
    dependency. The tick owns the stop and must keep working with the budget
    spent and the API down."""
    llm.calls.clear()
    tick_run(held)
    assert llm.calls == []


def test_a_reviewed_exit_is_marked_as_such(held, llm):
    """Without the mark, an LLM-influenced exit books as a plain stop_loss and
    the before/after comparison becomes impossible after the fact."""
    llm.exit_bias = "stop"
    _make_due(held)
    assert screen_run(held)["action"] == "raise_stop"
    assert _position(held).levels_revised is True


def test_holding_does_not_mark_the_trade(held, llm):
    """A review that changed nothing did not influence the exit, and counting
    it as reviewed would poison the comparison in the other direction."""
    llm.exit_bias = "hold"
    _make_due(held)
    screen_run(held)
    position = _position(held)
    assert position.levels_revised is False
    assert position.review_count == 1


def test_the_review_is_billed(held, llm):
    """It spends the same budget as any other call, and the daily allowance
    gate above only works if the spending is actually recorded."""
    _make_due(held)
    before = held.load_state().monthly.llm_cost_jpy
    screen_run(held)
    assert held.load_state().monthly.llm_cost_jpy > before
