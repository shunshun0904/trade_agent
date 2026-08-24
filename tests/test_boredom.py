"""The 3-day rule (spec 7) and its subordination to the safety rules (spec 0)."""

from datetime import timedelta
from decimal import Decimal

import pytest

from trade_agent.models.state import Halt, HaltReason, SystemState
from trade_agent.risk.boredom import (
    evaluate_boredom,
    mechanical_probe_plan,
    probe_loss_room_jpy,
    probe_stop_loss,
)
from trade_agent.risk.rules import RiskEngine
from trade_agent.timeutil import jst_date_str, jst_month_str

E = Decimal


@pytest.fixture
def state(clock, config):
    now = clock.now()
    st = SystemState.initial(config.capital.initial_equity_jpy, now,
                             jst_date_str(now), jst_month_str(now))
    st.last_entry_at = now
    return st


def test_does_not_fire_before_72_hours(config, state, clock):
    clock.advance(hours=71, minutes=59)
    decision = evaluate_boredom(config, state, clock.now(), [])
    assert not decision.triggered
    assert decision.consensus_min == 2


def test_fires_at_72_hours_and_one_minute(config, state, clock):
    clock.advance(hours=72, minutes=1)
    decision = evaluate_boredom(config, state, clock.now(), [])
    assert decision.triggered
    assert decision.consensus_min == config.boredom.relaxed_consensus_min == 1


@pytest.mark.parametrize("reason", [
    HaltReason.KILL_SWITCH,
    HaltReason.DAILY_LOSS,
    HaltReason.LOSING_STREAK,
    HaltReason.FLASH_MOVE,
    HaltReason.OWNER_PAUSE,
])
def test_every_safety_halt_suppresses_the_rule(config, state, clock, reason):
    clock.advance(hours=100)
    decision = evaluate_boredom(config, state, clock.now(),
                                [Halt(reason=reason, detail="x")])
    assert not decision.triggered
    assert decision.blocked_by is reason
    assert decision.consensus_min == 2


def test_an_open_position_stops_the_clock(config, state, clock):
    clock.advance(hours=100)
    state.open_position = object()
    decision = evaluate_boredom(config, state, clock.now(), [])
    assert not decision.triggered


def test_monthly_probe_loss_cap_suspends_the_rule(config, state, clock):
    clock.advance(hours=100)
    state.monthly.probe_rule_suspended = True
    decision = evaluate_boredom(config, state, clock.now(), [])
    assert not decision.triggered
    assert decision.blocked_by is HaltReason.PROBE_BUDGET


def test_disabled_rule_never_fires(config, state, clock):
    config.boredom.enabled = False
    clock.advance(hours=200)
    assert not evaluate_boredom(config, state, clock.now(), []).triggered


def test_a_fresh_deployment_measures_from_first_state(config, clock):
    now = clock.now()
    state = SystemState.initial(config.capital.initial_equity_jpy, now,
                                jst_date_str(now), jst_month_str(now))
    assert state.last_entry_at is None
    clock.advance(hours=1)
    assert not evaluate_boredom(config, state, clock.now(), []).triggered
    clock.advance(hours=72)
    assert evaluate_boredom(config, state, clock.now(), []).triggered


def test_probe_stop_is_within_the_configured_distance(config):
    entry = E(15000000)
    stop = probe_stop_loss(config, entry)
    distance_pct = (entry - stop) / entry * E(100)
    assert abs(distance_pct - config.boredom.probe_sl_pct) < E("0.01")


def test_mechanical_probe_rests_below_the_market(config, snapshot):
    plan = mechanical_probe_plan(config, snapshot, "range")
    assert plan is not None
    assert plan["entry"] <= snapshot.book.best_bid
    assert plan["stop_loss"] < plan["entry"] < plan["take_profit"]


def test_mechanical_probe_declines_without_a_vwap(config, snapshot):
    snapshot.indicators.vwap_24h = None
    assert mechanical_probe_plan(config, snapshot, "range") is None


def test_probe_risk_is_never_above_normal_risk(config):
    risk = RiskEngine(config)
    normal = risk.risk_limit_jpy(E(10000), probe=False)
    probe = risk.risk_limit_jpy(E(10000), probe=True)
    assert probe < normal


def test_probe_loss_room_shrinks_with_losses(config, state):
    full = probe_loss_room_jpy(config, state)
    state.monthly.probe_pnl_jpy = E(-120)
    assert probe_loss_room_jpy(config, state) == full - E(120)
