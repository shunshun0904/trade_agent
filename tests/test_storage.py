"""Persistence: encoding, conditional writes, leases (spec 10)."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from trade_agent.errors import DuplicateOrder
from trade_agent.models.trading import (
    OrderIntent, OrderPurpose, OrderRecord, OrderType, Side)
from trade_agent.storage.base import AgentCallRecord, EquityPoint, StoredLesson
from trade_agent.storage.dynamo import _encode, _from_item, _to_item
from trade_agent.storage.memory import MemoryStore

E = Decimal
NOW = datetime(2026, 3, 2, tzinfo=timezone.utc)


def _order(client_order_id="o1") -> OrderRecord:
    return OrderRecord.from_intent(
        OrderIntent(client_order_id=client_order_id, cycle_id="c",
                    pair="btc_jpy", side=Side.BUY, order_type=OrderType.LIMIT,
                    qty_btc=E("0.0003"), price=E(15000000), post_only=True,
                    purpose=OrderPurpose.ENTRY),
        NOW)


# -- conditional writes ---------------------------------------------------

def test_a_pending_order_cannot_be_created_twice(store):
    store.orders.put_pending(_order())
    with pytest.raises(DuplicateOrder):
        store.orders.put_pending(_order())


def test_updating_an_existing_order_is_allowed(store):
    record = _order()
    store.orders.put_pending(record)
    record.exchange_order_id = "42"
    store.orders.update(record)
    assert store.orders.get("o1").exchange_order_id == "42"


def test_open_and_recent_listings_differ(store):
    from trade_agent.models.trading import OrderStatus

    open_order = _order("open")
    filled = _order("filled")
    filled.status = OrderStatus.FILLED
    store.orders.put_pending(open_order)
    store.orders.put_pending(filled)

    assert {o.client_order_id for o in store.orders.list_open()} == {"open"}
    assert {o.client_order_id for o in store.orders.list_recent()} == \
        {"open", "filled"}


# -- leases ---------------------------------------------------------------

def test_a_lease_excludes_another_owner(store):
    assert store.locks.acquire("lock", "a", 60, NOW)
    assert not store.locks.acquire("lock", "b", 60, NOW)


def test_the_same_owner_can_renew(store):
    assert store.locks.acquire("lock", "a", 60, NOW)
    assert store.locks.acquire("lock", "a", 60, NOW + timedelta(seconds=30))


def test_an_expired_lease_is_free(store):
    store.locks.acquire("lock", "a", 60, NOW)
    assert store.locks.acquire("lock", "b", 60, NOW + timedelta(seconds=61))


def test_release_only_works_for_the_holder(store):
    store.locks.acquire("lock", "a", 60, NOW)
    store.locks.release("lock", "b")
    assert not store.locks.acquire("lock", "c", 60, NOW)
    store.locks.release("lock", "a")
    assert store.locks.acquire("lock", "c", 60, NOW)


# -- state ----------------------------------------------------------------

def test_saving_state_advances_the_version(store, config):
    from trade_agent.models.state import SystemState

    state = SystemState.initial(E(10000), NOW, "2026-03-02", "2026-03")
    saved = store.state.save(state)
    assert saved.version == 1
    assert store.state.save(saved).version == 2


def test_reads_are_copies_not_aliases(store):
    from trade_agent.models.state import SystemState

    store.state.save(SystemState.initial(E(10000), NOW, "2026-03-02", "2026-03"))
    first = store.state.load()
    first.equity_jpy = E(1)
    assert store.state.load().equity_jpy == E(10000)


# -- lessons and equity ---------------------------------------------------

def test_lessons_filter_by_regime_and_include_universal_ones(store):
    for index, tag in enumerate(["range", "trend_up", "all"]):
        store.lessons.put(StoredLesson(
            lesson_id=f"l{index}", created_at=NOW + timedelta(minutes=index),
            text=f"lesson {tag}", regime_tag=tag, evidence="stat",
            confidence=0.5))

    tags = {row.regime_tag for row in store.lessons.list(regime="range")}
    assert tags == {"range", "all"}


def test_equity_points_are_keyed_by_jst_date(store):
    store.equity.put(EquityPoint(jst_date="2026-03-02", equity_jpy=E(10000)))
    store.equity.put(EquityPoint(jst_date="2026-03-03", equity_jpy=E(10100)))
    assert store.equity.get("2026-03-03").equity_jpy == E(10100)
    assert [p.jst_date for p in store.equity.list_recent()] == \
        ["2026-03-03", "2026-03-02"]


def test_agent_calls_are_ordered_within_a_cycle(store):
    for sequence in (3, 1, 2):
        store.agent_calls.put(AgentCallRecord(
            cycle_id="c", agent=f"a{sequence}", sequence=sequence,
            called_at=NOW, model="m"))
    assert [c.sequence for c in store.agent_calls.list_for_cycle("c")] == [1, 2, 3]


# -- DynamoDB encoding ----------------------------------------------------

def test_decimals_survive_encoding_unchanged():
    """Money must never pass through a float on its way to storage."""
    value = E("0.00012345")
    assert _encode(value) is value
    assert isinstance(_encode(E("15000000.5")), Decimal)


def test_datetimes_encode_as_utc_iso_strings():
    assert _encode(NOW) == "2026-03-02T00:00:00Z"


def test_an_order_round_trips_through_an_item():
    record = _order()
    item = _to_item(record, pk=record.client_order_id)
    assert item["pk"] == "o1"
    assert isinstance(item["qty_btc"], Decimal)

    restored = _from_item(OrderRecord, item)
    assert restored.client_order_id == record.client_order_id
    assert restored.qty_btc == record.qty_btc
    assert restored.created_at == record.created_at
    assert restored.side is Side.BUY


def test_none_values_are_dropped_rather_than_stored():
    item = _to_item(_order())
    assert "exchange_order_id" not in item
    assert "error" not in item


def test_memory_and_dynamo_share_the_repository_surface():
    """Both backends must answer the same calls, or a Lambda would behave
    differently from the test suite."""
    from trade_agent.storage.dynamo import (
        _EquityRepo, _LessonRepo, _LockRepo, _OrderRepo, _StateRepo, _TradeRepo)

    memory = MemoryStore()
    pairs = [(memory.orders, _OrderRepo), (memory.locks, _LockRepo),
             (memory.trades, _TradeRepo), (memory.lessons, _LessonRepo),
             (memory.equity, _EquityRepo), (memory.state, _StateRepo)]
    for instance, dynamo_cls in pairs:
        public = {name for name in dir(instance)
                  if not name.startswith("_") and callable(getattr(instance, name))}
        implemented = {name for name in dir(dynamo_cls)
                       if not name.startswith("_")}
        missing = public - implemented - {"list_all"}
        assert not missing, f"{dynamo_cls.__name__} is missing {missing}"
