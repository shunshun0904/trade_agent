"""The exit review may only tighten.

This is the entire safety argument for letting a model touch a live position,
so it is checked here against the guard rather than against the prompt. A stop
that can move down, or a target that can move up, is a model with the power to
add risk to money already at stake — and a model holding a loss is precisely
where that power is most dangerous. The shape is refused, not discouraged.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from trade_agent.errors import GuardRejection
from trade_agent.guards.deterministic import DeterministicGuard
from trade_agent.models.agent_io import ExitOutput
from trade_agent.models.trading import Position

ENTRY = Decimal("15000000")
STOP = Decimal("14850000")
TARGET = Decimal("15300000")


@pytest.fixture
def position() -> Position:
    return Position(
        trade_id="trd-1", cycle_id="cyc-1", pair="btc_jpy",
        qty_btc=Decimal("0.0006"), entry_price=ENTRY,
        stop_loss=STOP, take_profit=TARGET,
        opened_at=datetime(2026, 3, 2, tzinfo=timezone.utc))


@pytest.fixture
def guard(config, snapshot) -> DeterministicGuard:
    return DeterministicGuard(config, snapshot)


def _out(**kwargs) -> ExitOutput:
    base = dict(action="hold", new_stop_loss=None, new_take_profit=None,
                invalidation_hit=False, rationale="前提は生きている。")
    base.update(kwargs)
    return ExitOutput(**base)


# -- what must be allowed -------------------------------------------------

def test_holding_is_allowed(guard, position):
    guard.validate_exit(_out(), position=position)


def test_raising_the_stop_is_allowed(guard, position):
    guard.validate_exit(
        _out(action="raise_stop", new_stop_loss=float(STOP + 10000),
             invalidation_hit=True, rationale="前提が崩れた。"),
        position=position)


def test_lowering_the_target_is_allowed(guard, position):
    guard.validate_exit(
        _out(action="lower_target", new_take_profit=float(TARGET - 10000),
             invalidation_hit=True, rationale="上値の前提が崩れた。"),
        position=position)


def test_a_stop_raised_through_the_market_is_allowed(guard, position):
    """This is how "close now" is expressed: there is no separate action, and
    `protection.arm` turns a stop above the market into an immediate exit."""
    guard.validate_exit(
        _out(action="raise_stop", new_stop_loss=float(ENTRY * Decimal("1.01")),
             invalidation_hit=True, rationale="即時決済する。"),
        position=position)


# -- what must be refused -------------------------------------------------

def test_a_widened_stop_is_refused(guard, position):
    with pytest.raises(GuardRejection) as excinfo:
        guard.validate_exit(
            _out(action="raise_stop", new_stop_loss=float(STOP - 1),
                 invalidation_hit=True, rationale="もう少し余裕を持たせる。"),
            position=position)
    assert any("損切り" in v for v in excinfo.value.violations)


def test_a_stop_left_where_it_is_counts_as_widening(guard, position):
    """Strictly greater, not >=. A no-op dressed as an action is a review that
    spent a call and changed nothing while reporting that it did."""
    with pytest.raises(GuardRejection):
        guard.validate_exit(
            _out(action="raise_stop", new_stop_loss=float(STOP),
                 invalidation_hit=True, rationale="同じ位置に置き直す。"),
            position=position)


def test_a_raised_target_is_refused(guard, position):
    with pytest.raises(GuardRejection) as excinfo:
        guard.validate_exit(
            _out(action="lower_target", new_take_profit=float(TARGET + 1),
                 invalidation_hit=False, rationale="もっと伸びる。"),
            position=position)
    assert any("利確" in v for v in excinfo.value.violations)


def test_hold_may_not_carry_prices(guard, position):
    with pytest.raises(GuardRejection):
        guard.validate_exit(
            _out(action="hold", new_stop_loss=float(STOP + 1)),
            position=position)


def test_each_action_may_only_move_its_own_level(guard, position):
    """Moving both at once hides a widening inside a tightening."""
    with pytest.raises(GuardRejection):
        guard.validate_exit(
            _out(action="raise_stop", new_stop_loss=float(STOP + 1),
                 new_take_profit=float(TARGET + 1), invalidation_hit=True,
                 rationale="両方動かす。"),
            position=position)


def test_a_missing_price_is_refused(guard, position):
    with pytest.raises(GuardRejection):
        guard.validate_exit(
            _out(action="raise_stop", invalidation_hit=True,
                 rationale="引き上げる。"),
            position=position)
