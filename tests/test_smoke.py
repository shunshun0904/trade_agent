"""Fixture sanity: the scripted market must produce a usable snapshot."""

from decimal import Decimal


def test_snapshot_has_indicators(snapshot):
    assert snapshot.last_price > 0
    assert snapshot.book.best_ask > snapshot.book.best_bid
    assert snapshot.indicators.rsi is not None
    assert snapshot.account.equity_jpy == Decimal(10000)
    assert snapshot.constraints.min_order_btc == Decimal("0.0001")


def test_prompt_json_is_stable(snapshot):
    assert snapshot.to_prompt_json() == snapshot.to_prompt_json()
    assert "last_price" in snapshot.to_prompt_json()
