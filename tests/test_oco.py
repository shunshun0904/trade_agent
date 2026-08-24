"""The hand-rolled OCO (spec 8).

bitbank has no OCO endpoint, so two legs are placed separately and the survivor
is cancelled when one fills. What a *spot* balance can actually back could not
be verified against the live exchange, so the behavioural tests run under both
plausible answers:

    reserve_on_placement=True   the stop claims the BTC immediately, so the
                                take-profit leg is refused outright
    reserve_on_placement=False  the stop claims nothing until it triggers, so
                                both legs are accepted — and the resting target
                                then starves the stop at the moment it fires

Under both, the settled outcome on spot is the same: **the stop owns the
balance and the target is evaluated locally**. That is the honest ceiling for a
single spot balance, and it is still a real gain — the stop now survives this
process dying.

The full two-leg path is exercised separately against an account that can back
both, because the code has to stay correct if bitbank turns out to allow it.
"""

from decimal import Decimal

import pytest

from trade_agent.errors import ExchangeError, OrderNotCancelable, StopOrderRefused
from trade_agent.exchange.paper import PaperExchange
from trade_agent.execution.protection import (
    EXCHANGE_OCO,
    LOCAL,
    STOP_ONLY,
    leg_order_id,
    weighted_exit,
)
from trade_agent.models.state import SystemState
from trade_agent.models.trading import (
    ExecutionPlan,
    OrderPurpose,
    OrderRecord,
    OrderStatus,
    OrderType,
    Side,
    TradeRecord,
)
from trade_agent.timeutil import jst_date_str, jst_month_str

E = Decimal
QTY = E("0.0003")


@pytest.fixture
def state(clock, config):
    now = clock.now()
    return SystemState.initial(config.capital.initial_equity_jpy, now,
                               jst_date_str(now), jst_month_str(now))


@pytest.fixture(params=[True, False],
                ids=["reserves-at-placement", "reserves-at-trigger"])
def reservation(request):
    """Both plausible bitbank behaviours (docs/OPEN-QUESTIONS.md A-1)."""
    return request.param


def _paper(ctx, market, config, clock, *, reserve_on_placement: bool):
    return PaperExchange(
        market, pair=config.exchange.pair,
        maker_fee_rate=config.exchange.maker_fee_rate,
        taker_fee_rate=config.exchange.taker_fee_rate,
        min_order_btc=config.exchange.min_order_btc,
        initial_jpy=config.capital.initial_equity_jpy,
        now=clock.now, reserve_on_placement=reserve_on_placement)


@pytest.fixture
def oco_ctx(ctx, market, config, clock, reservation):
    """A spot account: one balance, so it can back only one sell."""
    config.execution.oco_mode = "auto"
    ctx.exchange.inner = _paper(ctx, market, config, clock,
                                reserve_on_placement=reservation)
    return ctx


@pytest.fixture
def double_backed_ctx(ctx, market, config, clock):
    """An account with enough balance behind both legs.

    Not reachable with one spot position, but the two-leg path has to stay
    correct in case bitbank backs both.
    """
    config.execution.oco_mode = "auto"
    ctx.exchange.inner = _paper(ctx, market, config, clock,
                                reserve_on_placement=True)
    return ctx


def _open(ctx, state, *, stop_pct="0.99", tp_pct="1.02", extra_btc=E(0)):
    """Open a position and let the protection layer arm itself."""
    entry = ctx.exchange.market.price
    plan = ExecutionPlan(
        cycle_id="cyc-1", trade_id="trd-1", entry=entry,
        stop_loss=entry * E(stop_pct), take_profit=entry * E(tp_pct),
        qty_btc=QTY, risk_jpy=E(45), client_order_id="entry-1")
    ctx.store.trades.put(TradeRecord(
        trade_id="trd-1", cycle_id="cyc-1", pair="btc_jpy",
        qty_btc=plan.qty_btc, entry_price=plan.entry, entry_order_id="entry-1",
        entry_at=ctx.clock.now(), stop_loss=plan.stop_loss,
        take_profit=plan.take_profit))
    executor = ctx.executor()
    assert executor.place_entry(plan).placed

    if extra_btc:
        # Fund the second leg so the two-leg path can be exercised.
        ctx.exchange.inner.account.btc_free += extra_btc

    manager = ctx.position_manager(executor)
    update = manager.step(state, last_price=entry,
                          best_bid=ctx.exchange.market.best_bid,
                          best_ask=ctx.exchange.market.best_ask, plan=plan)
    return manager, update, plan


# -- arming ---------------------------------------------------------------

def test_the_stop_leg_is_always_placed(oco_ctx, state):
    """Whichever way the balance reserves, the protective leg exists."""
    _open(oco_ctx, state)
    position = state.open_position

    assert position.stop_order_id is not None
    stop = oco_ctx.store.orders.get(position.stop_order_id)
    assert stop.side is Side.SELL
    assert stop.order_type is OrderType.STOP
    assert stop.trigger_price == position.stop_loss
    assert stop.status.is_protective


def test_a_spot_balance_ends_up_backing_the_stop_only(oco_ctx, state):
    """The ceiling for one spot balance, reached from either direction."""
    _, update, _ = _open(oco_ctx, state)
    position = state.open_position

    assert position.protection == STOP_ONLY
    assert position.take_profit_order_id is None
    assert update.protection == STOP_ONLY


def test_a_deferred_reservation_makes_us_retract_the_target(oco_ctx, state,
                                                            reservation):
    """A resting target would starve the stop, so it is taken back."""
    _, update, _ = _open(oco_ctx, state)
    notes = " ".join(update.notes)

    if reservation:
        assert "refused for want of balance" in notes
    else:
        assert "starve the stop" in notes
        assert "retracted" in notes
        # And it really is gone from the exchange, not merely forgotten.
        tp_id = leg_order_id("trd-1", "tp")
        assert oco_ctx.store.orders.get(tp_id).status is OrderStatus.CANCELED


def test_the_stop_goes_on_before_the_target(oco_ctx, state):
    """If only one leg can exist it must be the protective one."""
    _open(oco_ctx, state)
    sent = [o for o in oco_ctx.exchange.orders_sent if o.side is Side.SELL]
    assert sent, "no protective order was sent"
    assert sent[0].purpose is OrderPurpose.STOP_LOSS


def test_arming_is_idempotent(oco_ctx, state):
    """Re-arming must not place a second stop."""
    from trade_agent.errors import DuplicateOrder

    manager, _, _ = _open(oco_ctx, state)
    before = len(oco_ctx.exchange.orders_sent)

    with pytest.raises(DuplicateOrder):
        manager.protection._place_stop(state.open_position, _Result())
    assert len(oco_ctx.exchange.orders_sent) == before


def test_leg_ids_are_derived_from_the_trade():
    assert leg_order_id("trd-1", "stop") == leg_order_id("trd-1", "stop")
    assert leg_order_id("trd-1", "stop") != leg_order_id("trd-1", "tp")
    assert leg_order_id("trd-1", "stop") != leg_order_id("trd-2", "stop")


def test_a_stop_already_through_the_market_closes_instead(oco_ctx, state, market):
    """bitbank refuses a trigger that would fire immediately (60018)."""
    _, update, _ = _open(oco_ctx, state, stop_pct="1.01")  # stop above market

    assert any("already through the stop" in note for note in update.notes)
    assert state.open_position is None or \
        state.open_position.protection == LOCAL


def test_a_refused_stop_falls_back_to_local_and_warns(oco_ctx, state,
                                                      monkeypatch):
    original = oco_ctx.exchange.inner.create_order

    def refuse_stops(intent):
        if intent.order_type.is_trigger:
            raise StopOrderRefused("stop orders suspended", code=70023)
        return original(intent)

    monkeypatch.setattr(oco_ctx.exchange.inner, "create_order", refuse_stops)
    _, update, _ = _open(oco_ctx, state)

    assert state.open_position.protection == LOCAL
    assert state.open_position.stop_order_id is None
    assert any("falling back to local" in note for note in update.notes)
    assert any("損切り注文を出せませんでした" in subject
               for subject, _ in oco_ctx.notifier.sent)


# -- the stop firing ------------------------------------------------------

def test_a_triggered_stop_closes_the_position(oco_ctx, state, market, clock):
    manager, _, plan = _open(oco_ctx, state)

    market.price = plan.stop_loss - E(50000)     # gap down through the trigger
    clock.advance(minutes=5)
    update = manager.step(state, last_price=market.price,
                          best_bid=market.best_bid, best_ask=market.best_ask)

    assert update.closed is not None
    assert update.closed.exit_reason == "stop_loss"
    assert update.closed.net_pnl_jpy < 0
    assert state.open_position is None


def test_the_stop_survives_this_process_doing_nothing(oco_ctx, state, market,
                                                      clock):
    """The point of the whole exercise: the exchange holds the stop.

    No local evaluation runs here — the position closes because the exchange
    executed the order, which is what makes a missed tick survivable.
    """
    manager, _, plan = _open(oco_ctx, state)
    stop_id = state.open_position.stop_order_id

    market.price = plan.stop_loss - E(50000)
    clock.advance(minutes=5)
    # Ask the exchange directly, with no help from the position manager.
    stop = oco_ctx.store.orders.get(stop_id)
    exchange_view = oco_ctx.exchange.get_order(stop.exchange_order_id)

    assert exchange_view["status"] == "FULLY_FILLED"


# -- the two-leg path -----------------------------------------------------

def test_both_legs_are_kept_when_the_balance_backs_them(double_backed_ctx, state):
    _, update, _ = _open(double_backed_ctx, state, extra_btc=QTY)
    position = state.open_position

    assert position.protection == EXCHANGE_OCO
    assert position.stop_order_id and position.take_profit_order_id
    assert not any("retracted" in note for note in update.notes)


def test_a_filled_stop_cancels_the_target(double_backed_ctx, state, market,
                                          clock):
    manager, _, plan = _open(double_backed_ctx, state, extra_btc=QTY)
    tp_id = state.open_position.take_profit_order_id

    market.price = plan.stop_loss - E(50000)
    clock.advance(minutes=5)
    update = manager.step(state, last_price=market.price,
                          best_bid=market.best_bid, best_ask=market.best_ask)

    assert update.closed is not None
    assert update.closed.exit_reason == "stop_loss"
    assert double_backed_ctx.store.orders.get(tp_id).status is OrderStatus.CANCELED
    assert any("cancelled the" in note for note in update.notes)


def test_a_filled_target_cancels_the_stop(double_backed_ctx, state, market,
                                          clock):
    manager, _, plan = _open(double_backed_ctx, state, extra_btc=QTY)
    stop_id = state.open_position.stop_order_id

    market.price = plan.take_profit + E(50000)
    clock.advance(minutes=5)
    update = manager.step(state, last_price=market.price,
                          best_bid=market.best_bid, best_ask=market.best_ask)

    assert update.closed is not None
    assert update.closed.exit_reason == "take_profit"
    assert update.closed.net_pnl_jpy > 0
    assert double_backed_ctx.store.orders.get(stop_id).status is OrderStatus.CANCELED


def test_a_cancel_that_loses_the_race_is_not_an_error(double_backed_ctx, state,
                                                      market, clock, monkeypatch):
    """bitbank 50010 means the sibling filled between our read and our cancel."""
    manager, _, plan = _open(double_backed_ctx, state, extra_btc=QTY)

    def refuse_cancel(record, reason):
        raise OrderNotCancelable("already filled", code=50010)

    monkeypatch.setattr(manager.executor, "cancel", refuse_cancel)
    market.price = plan.stop_loss - E(50000)
    clock.advance(minutes=5)
    update = manager.step(state, last_price=market.price,
                          best_bid=market.best_bid, best_ask=market.best_ask)

    assert update.closed is not None
    assert any("could not be cancelled" in note for note in update.notes)


def test_a_cancel_failure_is_escalated(double_backed_ctx, state, market, clock,
                                       monkeypatch):
    """A live leg we could not cancel can sell BTC we no longer hold."""
    manager, _, plan = _open(double_backed_ctx, state, extra_btc=QTY)

    def fail_cancel(record, reason):
        raise ExchangeError("bitbank is unreachable")

    monkeypatch.setattr(manager.executor, "cancel", fail_cancel)
    market.price = plan.stop_loss - E(50000)
    clock.advance(minutes=5)
    update = manager.step(state, last_price=market.price,
                          best_bid=market.best_bid, best_ask=market.best_ask)

    assert any("FAILED to cancel" in note for note in update.notes)
    assert any("残注文を取り消せ" in subject
               for subject, _ in double_backed_ctx.notifier.sent)


def test_an_oversell_is_escalated(double_backed_ctx, state, market):
    """A spot balance should make this impossible; if it happens, shout."""
    manager, _, plan = _open(double_backed_ctx, state, extra_btc=QTY)
    position = state.open_position

    for order_id in (position.stop_order_id, position.take_profit_order_id):
        record = double_backed_ctx.store.orders.get(order_id)
        record.status = OrderStatus.FILLED
        record.executed_qty_btc = position.qty_btc
        record.average_price = plan.entry
        double_backed_ctx.store.orders.update(record)

    update = manager.step(state, last_price=plan.entry,
                          best_bid=market.best_bid, best_ask=market.best_ask)
    assert any("OVERSOLD" in note for note in update.notes)
    assert any("建玉より多く売却" in subject
               for subject, _ in double_backed_ctx.notifier.sent)


def test_both_legs_filling_books_the_combined_exit(clock):
    """A gap through both levels is one exit at the weighted average."""
    stop = OrderRecord(
        client_order_id="s", cycle_id="c", pair="btc_jpy", side=Side.SELL,
        order_type=OrderType.STOP, qty_btc=E("0.0002"),
        status=OrderStatus.FILLED, executed_qty_btc=E("0.0002"),
        average_price=E(14000000), fee_jpy=E("3.4"),
        created_at=clock.now(), updated_at=clock.now())
    target = OrderRecord(
        client_order_id="t", cycle_id="c", pair="btc_jpy", side=Side.SELL,
        order_type=OrderType.LIMIT, qty_btc=E("0.0001"),
        status=OrderStatus.FILLED, executed_qty_btc=E("0.0001"),
        average_price=E(15200000), fee_jpy=E("-0.3"),
        created_at=clock.now(), updated_at=clock.now())

    qty, price, fees = weighted_exit([stop, target])
    assert qty == E("0.0003")
    assert price == (E(14000000) * 2 + E(15200000)) / 3
    assert fees == E("3.1")


# -- the backstop ---------------------------------------------------------

def test_local_evaluation_stays_out_of_the_exchange_leg_s_way(oco_ctx, state,
                                                              market):
    """The tick must not sell a position the exchange stop is already holding."""
    manager, _, plan = _open(oco_ctx, state)
    before = len(oco_ctx.exchange.orders_sent)

    update = manager.step(state, last_price=plan.stop_loss - E(1),
                          best_bid=market.best_bid, best_ask=market.best_ask)

    assert update.exit_submitted is None
    assert len(oco_ctx.exchange.orders_sent) == before


def test_the_backstop_covers_the_target_the_exchange_does_not_hold(oco_ctx,
                                                                   state, market):
    manager, _, plan = _open(oco_ctx, state)
    assert state.open_position.protection == STOP_ONLY

    update = manager.step(state, last_price=plan.take_profit + E(1000),
                          best_bid=plan.take_profit, best_ask=plan.take_profit)
    assert update.exit_submitted is not None
    assert update.exit_submitted.purpose is OrderPurpose.TAKE_PROFIT


def test_a_dead_stop_leg_reverts_to_local_and_notifies(oco_ctx, state, market):
    """A rejected or cancelled stop must not leave the position naked."""
    manager, _, plan = _open(oco_ctx, state)
    stop = oco_ctx.store.orders.get(state.open_position.stop_order_id)
    stop.status = OrderStatus.REJECTED
    oco_ctx.store.orders.update(stop)

    update = manager.step(state, last_price=plan.entry, best_bid=market.best_bid,
                          best_ask=market.best_ask)

    assert state.open_position.protection == LOCAL
    assert any("reverting to local" in note for note in update.notes)
    assert any("損切り注文が消えました" in subject
               for subject, _ in oco_ctx.notifier.sent)


def test_the_backstop_sells_once_the_stop_leg_is_gone(oco_ctx, state, market):
    manager, _, plan = _open(oco_ctx, state)
    stop = oco_ctx.store.orders.get(state.open_position.stop_order_id)
    stop.status = OrderStatus.REJECTED
    oco_ctx.store.orders.update(stop)

    update = manager.step(state, last_price=plan.stop_loss - E(1000),
                          best_bid=plan.stop_loss, best_ask=plan.stop_loss)
    assert update.exit_submitted is not None
    assert update.exit_submitted.purpose is OrderPurpose.STOP_LOSS


# -- kill switch ----------------------------------------------------------

def test_the_kill_switch_cancels_the_legs_before_liquidating(oco_ctx, state):
    """A surviving stop would fight the liquidation for the same BTC."""
    manager, _, _ = _open(oco_ctx, state)
    position = state.open_position
    stop_id = position.stop_order_id

    update = manager.force_close(state, position, "kill_switch")

    assert oco_ctx.store.orders.get(stop_id).status is OrderStatus.CANCELED
    assert state.open_position.protection == LOCAL
    assert update.exit_submitted is not None
    assert update.exit_submitted.order_type is OrderType.MARKET


# -- configuration --------------------------------------------------------

def test_local_mode_places_no_legs(oco_ctx, state):
    oco_ctx.config.execution.oco_mode = LOCAL
    _open(oco_ctx, state)
    assert state.open_position.protection == LOCAL
    assert state.open_position.stop_order_id is None
    assert not [o for o in oco_ctx.exchange.orders_sent if o.side is Side.SELL]


def test_stop_limit_mode_sets_a_limit_below_the_trigger(oco_ctx, state, config):
    config.execution.stop_order_type = "stop_limit"
    _open(oco_ctx, state)
    stop = oco_ctx.store.orders.get(state.open_position.stop_order_id)
    assert stop.order_type is OrderType.STOP_LIMIT
    assert stop.price is not None
    assert stop.price < stop.trigger_price


class _Result:
    """Minimal stand-in for ProtectionResult in the idempotency test."""

    def __init__(self):
        self.notes: list[str] = []


def test_an_exit_leg_without_a_price_is_refused_not_booked_at_zero(clock):
    """Valuing a fill at zero would report the position as a total loss."""
    leg = OrderRecord(
        client_order_id="broken", cycle_id="c", pair="btc_jpy", side=Side.SELL,
        order_type=OrderType.STOP, qty_btc=QTY, status=OrderStatus.FILLED,
        executed_qty_btc=QTY, average_price=None, price=None,
        trigger_price=None, created_at=clock.now(), updated_at=clock.now())

    with pytest.raises(ValueError, match="no recorded price"):
        weighted_exit([leg])


def test_a_stop_without_an_average_falls_back_to_its_trigger(clock):
    leg = OrderRecord(
        client_order_id="s", cycle_id="c", pair="btc_jpy", side=Side.SELL,
        order_type=OrderType.STOP, qty_btc=QTY, status=OrderStatus.FILLED,
        executed_qty_btc=QTY, average_price=None, price=None,
        trigger_price=E(14850000), created_at=clock.now(),
        updated_at=clock.now())

    qty, price, _ = weighted_exit([leg])
    assert qty == QTY
    assert price == E(14850000)


def test_a_triggered_stop_does_not_fill_above_its_trigger(oco_ctx, state,
                                                          market, clock):
    """A stop is a worse-case exit, never a lucky one."""
    manager, _, plan = _open(oco_ctx, state)
    market.price = plan.stop_loss - E(50000)
    clock.advance(minutes=5)
    update = manager.step(state, last_price=market.price,
                          best_bid=market.best_bid, best_ask=market.best_ask)

    assert update.closed is not None
    assert update.closed.exit_price <= plan.stop_loss
