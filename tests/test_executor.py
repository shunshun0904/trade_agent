"""Order execution: idempotency, requote, reconciliation (spec 8)."""

from datetime import timedelta
from decimal import Decimal

import pytest

from trade_agent.errors import DuplicateOrder
from trade_agent.models.trading import (
    ExecutionPlan,
    OrderIntent,
    OrderPurpose,
    OrderStatus,
    OrderType,
    Side,
)
from trade_agent.storage.base import LOCK_EXECUTION

E = Decimal


def _plan(ctx, *, entry=None, client_order_id="fixed-id") -> ExecutionPlan:
    entry = entry if entry is not None else ctx.exchange.market.price
    return ExecutionPlan(
        cycle_id="cyc-1", trade_id="trd-1", entry=entry,
        stop_loss=entry * E("0.99"), take_profit=entry * E("1.02"),
        qty_btc=E("0.0003"), risk_jpy=E(45), client_order_id=client_order_id)


def test_places_a_post_only_limit_buy(ctx):
    executor = ctx.executor()
    result = executor.place_entry(_plan(ctx))
    assert result.placed
    assert result.order.side is Side.BUY
    assert result.order.order_type is OrderType.LIMIT
    assert result.order.post_only is True
    assert result.order.exchange_order_id is not None


def test_the_same_client_order_id_cannot_be_submitted_twice(ctx):
    """The idempotency primitive: a conditional create on the client id.

    This is the innermost of the three defences against a double entry, and it
    holds even with every higher-level guard removed.
    """
    executor = ctx.executor()
    intent = OrderIntent(
        client_order_id="same-key", cycle_id="cyc-1", pair="btc_jpy",
        side=Side.BUY, order_type=OrderType.LIMIT, qty_btc=E("0.0003"),
        price=ctx.exchange.market.price * E("0.9975"), post_only=True,
        purpose=OrderPurpose.ENTRY)

    executor.submit(intent)
    with pytest.raises(DuplicateOrder):
        executor.submit(intent.model_copy(deep=True))
    assert len(ctx.exchange.orders_sent) == 1


def test_an_open_order_blocks_a_second_entry(ctx):
    ctx.exchange.market.hold_still()  # keep the first entry resting, unfilled
    executor = ctx.executor()
    entry = ctx.exchange.market.price * E("0.9975")
    assert executor.place_entry(_plan(ctx, entry=entry)).placed
    second = executor.place_entry(_plan(ctx, entry=entry,
                                        client_order_id="other-id"))
    assert not second.placed
    assert "already open" in second.reason
    assert len(ctx.exchange.orders_sent) == 1


def test_a_filled_entry_blocks_a_second_entry(ctx):
    """Even once the first order is no longer *open*, its trade is."""
    executor = ctx.executor()
    plan = _plan(ctx)
    assert executor.place_entry(plan).placed
    from trade_agent.models.trading import TradeRecord

    ctx.store.trades.put(TradeRecord(
        trade_id="trd-1", cycle_id="cyc-1", pair="btc_jpy", qty_btc=E("0.0003"),
        entry_price=plan.entry, entry_order_id="fixed-id",
        entry_at=ctx.clock.now(), stop_loss=plan.stop_loss,
        take_profit=plan.take_profit))
    record = ctx.store.orders.get("fixed-id")
    record.executed_qty_btc = E("0.0003")
    record.trade_id = "trd-1"
    record.status = OrderStatus.FILLED
    ctx.store.orders.update(record)

    second = executor.place_entry(_plan(ctx, client_order_id="other-id"))
    assert not second.placed
    assert "still open" in second.reason


def test_requote_abandons_the_cycle_when_the_price_moved(ctx):
    executor = ctx.executor()
    planned = ctx.exchange.market.price
    ctx.exchange.market.price = planned * E("1.01")  # 1% away, limit is 0.3%
    result = executor.place_entry(_plan(ctx, entry=planned))
    assert not result.placed
    assert "price moved" in result.reason
    assert ctx.exchange.orders_sent == []


def test_a_small_move_still_places_the_order(ctx):
    executor = ctx.executor()
    planned = ctx.exchange.market.price
    ctx.exchange.market.price = planned * E("1.001")  # 0.1%
    assert executor.place_entry(_plan(ctx, entry=planned)).placed


def test_the_order_row_is_written_before_the_exchange_call(ctx, monkeypatch):
    """A crash between persistence and the API call must leave a pending row."""
    executor = ctx.executor()

    def explode(intent):
        raise RuntimeError("network died mid-call")

    monkeypatch.setattr(ctx.exchange, "create_order", explode)
    with pytest.raises(RuntimeError):
        executor.place_entry(_plan(ctx))

    record = ctx.store.orders.get("fixed-id")
    assert record is not None
    assert record.status is OrderStatus.REJECTED
    assert "network died" in record.error


def test_reconcile_adopts_a_matching_exchange_order(ctx, clock):
    """The crash case: the order reached bitbank but the id was never stored."""
    executor = ctx.executor()
    intent = OrderIntent(
        client_order_id="orphan", cycle_id="cyc-1", pair="btc_jpy", side=Side.BUY,
        order_type=OrderType.LIMIT, qty_btc=E("0.0003"),
        price=ctx.exchange.market.price * E("0.99"), post_only=True,
        purpose=OrderPurpose.ENTRY)
    # Simulate: the exchange has the order, our row does not know about it.
    ctx.exchange.create_order(intent)
    from trade_agent.models.trading import OrderRecord

    ctx.store.orders.put_pending(OrderRecord.from_intent(intent, clock.now()))

    notes = executor.reconcile_pending()
    record = ctx.store.orders.get("orphan")
    assert record.exchange_order_id is not None
    assert any("adopted exchange order" in n for n in notes)


def test_reconcile_never_replaces_an_unmatched_order(ctx, clock, config):
    from trade_agent.models.trading import OrderRecord

    executor = ctx.executor()
    intent = OrderIntent(
        client_order_id="lost", cycle_id="cyc-1", pair="btc_jpy", side=Side.BUY,
        order_type=OrderType.LIMIT, qty_btc=E("0.0003"),
        price=E(14000000), post_only=True, purpose=OrderPurpose.ENTRY)
    ctx.store.orders.put_pending(OrderRecord.from_intent(intent, clock.now()))

    clock.advance(minutes=config.execution.pending_order_stale_minutes + 1)
    notes = executor.reconcile_pending()

    assert ctx.store.orders.get("lost").status is OrderStatus.UNKNOWN
    assert ctx.exchange.orders_sent == []
    assert any("unresolved" in n for n in notes)
    assert any("could not be reconciled" in subject
               for subject, _ in ctx.notifier.sent)


def test_stale_post_only_entries_are_cancelled(ctx, clock, config):
    # A flat market keeps the resting order unfilled; the entry stays inside
    # the 0.3% requote window so it is placed at all.
    ctx.exchange.market.hold_still()
    executor = ctx.executor()
    executor.place_entry(_plan(ctx, entry=ctx.exchange.market.price * E("0.9975")))
    clock.advance(minutes=config.execution.post_only_timeout_minutes + 1)

    notes = executor.expire_stale_entries()
    assert any("canceled" in n for n in notes)
    assert ctx.store.orders.get("fixed-id").status is OrderStatus.CANCELED


def test_the_execution_lock_excludes_a_second_holder(ctx):
    first = ctx.executor(owner="a")
    second = ctx.executor(owner="b")
    assert first.acquire_lock(LOCK_EXECUTION)
    assert not second.acquire_lock(LOCK_EXECUTION)
    first.release_lock(LOCK_EXECUTION)
    assert second.acquire_lock(LOCK_EXECUTION)


def test_an_expired_lease_can_be_taken_over(ctx, clock, config):
    first = ctx.executor(owner="a")
    second = ctx.executor(owner="b")
    assert first.acquire_lock(LOCK_EXECUTION)
    clock.advance(seconds=config.execution.lock_lease_seconds + 1)
    assert second.acquire_lock(LOCK_EXECUTION)
