"""Position lifecycle under *local* stop evaluation (spec 8).

These pin the fallback path — the one that runs when the exchange holds no
protective legs. The hand-rolled OCO path has its own suite in test_oco.py.
"""

from decimal import Decimal

import pytest

from trade_agent.models.state import SystemState
from trade_agent.models.trading import ExecutionPlan, OrderPurpose, Side, TradeRecord
from trade_agent.timeutil import jst_date_str, jst_month_str

E = Decimal


@pytest.fixture
def state(clock, config):
    now = clock.now()
    return SystemState.initial(config.capital.initial_equity_jpy, now,
                               jst_date_str(now), jst_month_str(now))


def _open_position(ctx, state, *, probe=False):
    """Place an entry, let it fill, and promote it to a position."""
    ctx.config.execution.oco_mode = "local"
    entry = ctx.exchange.market.price
    plan = ExecutionPlan(
        cycle_id="cyc-1", trade_id="trd-1", entry=entry,
        stop_loss=entry * E("0.99"), take_profit=entry * E("1.02"),
        qty_btc=E("0.0003"), risk_jpy=E(45), probe=probe,
        client_order_id="entry-1")
    ctx.store.trades.put(TradeRecord(
        trade_id="trd-1", cycle_id="cyc-1", pair="btc_jpy", probe=probe,
        qty_btc=plan.qty_btc, entry_price=plan.entry, entry_order_id="entry-1",
        entry_at=ctx.clock.now(), stop_loss=plan.stop_loss,
        take_profit=plan.take_profit))
    executor = ctx.executor()
    assert executor.place_entry(plan).placed
    manager = ctx.position_manager(executor)
    update = manager.step(state, last_price=entry,
                          best_bid=ctx.exchange.market.best_bid,
                          best_ask=ctx.exchange.market.best_ask, plan=plan)
    return manager, update, plan


def test_a_filled_entry_becomes_a_position(ctx, state):
    manager, update, plan = _open_position(ctx, state)
    assert update.opened is not None
    assert state.open_position is not None
    assert state.open_position.stop_loss == plan.stop_loss
    assert state.last_entry_at is not None


def test_the_fill_notifies_the_owner(ctx, state):
    _open_position(ctx, state)
    assert any("約定(新規建て)" in subject for subject, _ in ctx.notifier.sent)


def test_a_stop_breach_sells_at_market(ctx, state):
    manager, _, plan = _open_position(ctx, state)
    breach = plan.stop_loss - E(1000)
    update = manager.step(state, last_price=breach, best_bid=breach,
                          best_ask=breach)
    order = update.exit_submitted
    assert order is not None
    assert order.purpose is OrderPurpose.STOP_LOSS
    assert order.post_only is False  # taker execution, allowed for stops only
    assert order.side is Side.SELL


def test_a_target_breach_rests_as_a_maker_order(ctx, state):
    manager, _, plan = _open_position(ctx, state)
    reached = plan.take_profit + E(1000)
    update = manager.step(state, last_price=reached,
                          best_bid=reached - E(1000), best_ask=reached)
    order = update.exit_submitted
    assert order is not None
    assert order.purpose is OrderPurpose.TAKE_PROFIT
    assert order.post_only is True
    assert order.price >= plan.take_profit


def test_no_exit_while_the_price_sits_between_the_levels(ctx, state):
    manager, _, plan = _open_position(ctx, state)
    update = manager.step(state, last_price=plan.entry,
                          best_bid=plan.entry, best_ask=plan.entry)
    assert update.exit_submitted is None
    assert state.open_position is not None


def test_a_filled_exit_books_the_trade_and_updates_equity(ctx, state, clock):
    manager, _, plan = _open_position(ctx, state)
    start_equity = state.equity_jpy

    # Move the market up so both the exit order and the fill simulation agree.
    ctx.exchange.market.price = plan.take_profit + E(20000)
    manager.step(state, last_price=ctx.exchange.market.price,
                 best_bid=ctx.exchange.market.best_bid,
                 best_ask=ctx.exchange.market.best_ask)
    clock.advance(minutes=5)
    update = manager.step(state, last_price=ctx.exchange.market.price,
                          best_bid=ctx.exchange.market.best_bid,
                          best_ask=ctx.exchange.market.best_ask)

    assert update.closed is not None
    assert update.closed.closed is True
    assert update.closed.net_pnl_jpy is not None
    assert state.open_position is None
    assert state.equity_jpy != start_equity
    assert any("約定(決済)" in subject for subject, _ in ctx.notifier.sent)


def test_an_unfilled_target_exit_expires_and_keeps_the_position(ctx, state, clock,
                                                                config):
    manager, _, plan = _open_position(ctx, state)
    # Flatten the market only now, so the *exit* order cannot fill.
    ctx.exchange.market.hold_still()
    reached = plan.take_profit + E(1000)
    manager.step(state, last_price=reached, best_bid=reached - E(1000),
                 best_ask=reached)
    assert state.open_position.exit_order_id is not None

    clock.advance(minutes=config.execution.tp_exit_timeout_minutes + 1)
    update = manager.step(state, last_price=plan.entry, best_bid=plan.entry,
                          best_ask=plan.entry)
    assert "expired" in " ".join(update.notes)
    assert state.open_position is not None
    assert state.open_position.exit_order_id is None


def test_a_partial_fill_sizes_the_position_to_what_executed(ctx, state, clock):
    from trade_agent.execution.executor import build_position
    from trade_agent.models.trading import OrderRecord, OrderStatus, OrderType

    plan = ExecutionPlan(cycle_id="c", trade_id="t", entry=E(15000000),
                         stop_loss=E(14850000), take_profit=E(15300000),
                         qty_btc=E("0.0006"), risk_jpy=E(90))
    record = OrderRecord(
        client_order_id="o", cycle_id="c", pair="btc_jpy", side=Side.BUY,
        order_type=OrderType.LIMIT, qty_btc=E("0.0006"), price=E(15000000),
        status=OrderStatus.PARTIALLY_FILLED, executed_qty_btc=E("0.0002"),
        average_price=E(15000000), created_at=clock.now(), updated_at=clock.now())

    position = build_position(plan, record, clock.now())
    assert position.qty_btc == E("0.0002")
    assert position.risk_jpy() == E(30)  # scaled from 90 JPY with the fill


def test_the_kill_switch_liquidates_at_market(ctx, state):
    manager, _, plan = _open_position(ctx, state)
    update = manager.force_close(state, state.open_position, "kill_switch")
    assert update.exit_submitted is not None
    assert update.exit_submitted.order_type.value == "market"
    assert state.open_position.exit_reason == "kill_switch"


def test_a_probe_trade_stays_flagged_through_settlement(ctx, state, clock):
    manager, _, plan = _open_position(ctx, state, probe=True)
    assert state.open_position.probe is True
    ctx.exchange.market.price = plan.take_profit + E(20000)
    manager.step(state, last_price=ctx.exchange.market.price,
                 best_bid=ctx.exchange.market.best_bid,
                 best_ask=ctx.exchange.market.best_ask)
    clock.advance(minutes=5)
    update = manager.step(state, last_price=ctx.exchange.market.price,
                          best_bid=ctx.exchange.market.best_bid,
                          best_ask=ctx.exchange.market.best_ask)
    assert update.closed.probe is True
    assert state.losing_streak == 0
