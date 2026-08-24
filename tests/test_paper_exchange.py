"""The paper exchange (spec 13, Phase 1)."""

from decimal import Decimal

import pytest

from trade_agent.errors import InsufficientFunds
from trade_agent.models.trading import OrderIntent, OrderType, Side

E = Decimal


def _buy(exchange, price, qty=E("0.0003")) -> OrderIntent:
    return OrderIntent(cycle_id="c", pair="btc_jpy", side=Side.BUY,
                       order_type=OrderType.LIMIT, qty_btc=qty, price=price,
                       post_only=True)


def test_a_buy_reserves_the_cash(exchange, market):
    market.hold_still()
    price = market.price * E("0.99")
    exchange.create_order(_buy(exchange, price))
    balances = exchange.get_balances()
    assert balances["jpy"].locked == price * E("0.0003")
    assert balances["jpy"].free < E(10000)


def test_a_buy_beyond_the_balance_is_refused(exchange, market):
    with pytest.raises(InsufficientFunds):
        exchange.create_order(_buy(exchange, market.price, qty=E("1.0")))


def test_a_resting_order_needs_the_market_to_trade_through_it(exchange, market):
    """Touching the limit is not a fill — the simulator is pessimistic on
    purpose, because Phase 1's exit criterion is a positive expectancy."""
    market.hold_still()
    order = exchange.create_order(_buy(exchange, market.price))
    assert order["status"] == "UNFILLED"

    market.candle_range = E(5000)  # the market now dips through the limit
    refreshed = exchange.get_order(order["order_id"])
    assert refreshed["status"] == "FULLY_FILLED"


def test_a_fill_credits_btc_and_charges_the_maker_rate(exchange, market, config):
    order = exchange.create_order(_buy(exchange, market.price))
    filled = exchange.get_order(order["order_id"])
    assert filled["status"] == "FULLY_FILLED"

    balances = exchange.get_balances()
    assert balances["btc"].free == E("0.0003")
    trades = exchange.get_trades_for_order(order["order_id"])
    assert trades[0]["maker_taker"] == "maker"
    # A maker rebate is a negative fee.
    assert Decimal(trades[0]["fee_amount_quote"]) < 0


def test_cancelling_releases_the_reservation(exchange, market):
    market.hold_still()
    order = exchange.create_order(_buy(exchange, market.price))
    exchange.cancel_order(order["order_id"])
    balances = exchange.get_balances()
    assert balances["jpy"].locked == 0
    assert balances["jpy"].free == E(10000)


def test_an_order_below_the_minimum_lot_is_refused(exchange, market):
    from trade_agent.errors import ExchangeError

    with pytest.raises(ExchangeError):
        exchange.create_order(_buy(exchange, market.price, qty=E("0.00001")))


def test_the_paper_exchange_cannot_reach_the_live_private_api(exchange):
    """It has no credentials and no code path that would use them."""
    assert not hasattr(exchange.inner, "_credentials")
    assert hasattr(exchange.inner, "account")  # the simulated book
