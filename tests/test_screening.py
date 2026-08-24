"""The 30-minute screener and cycle scheduling (spec 9)."""

from datetime import timedelta
from decimal import Decimal

import pytest

from trade_agent.models.state import CycleTrigger, Halt, HaltReason, SystemState
from trade_agent.orchestrator.screening import daily_debate_limit, evaluate_triggers
from trade_agent.timeutil import jst_date_str, jst_month_str

E = Decimal


@pytest.fixture
def state(clock, config):
    now = clock.now()
    st = SystemState.initial(config.capital.initial_equity_jpy, now,
                             jst_date_str(now), jst_month_str(now))
    # The fixture clock starts at 09:00 JST, which is a scheduled floor. Mark
    # it as already run so tests can exercise the market triggers in isolation.
    st.last_floor_run_at = now
    st.last_full_debate_at = now - timedelta(hours=2)
    return st


def _quiet(snapshot):
    ind = snapshot.indicators
    ind.rsi = E(50)
    ind.high_24h = snapshot.last_price * E("1.05")
    ind.low_24h = snapshot.last_price * E("0.95")
    ind.volume_ratio = E("1.0")
    ind.vwap_deviation_pct = E("0.1")
    return snapshot


def test_quiet_market_costs_nothing(config, state, snapshot, clock):
    result = evaluate_triggers(config, state, _quiet(snapshot), clock.now())
    assert not result.should_debate
    assert "no trigger" in result.summary()


def test_breakout_triggers_a_debate(config, state, snapshot, clock):
    snapshot = _quiet(snapshot)
    snapshot.indicators.high_24h = snapshot.last_price - E(1)
    result = evaluate_triggers(config, state, snapshot, clock.now())
    assert result.should_debate
    assert result.trigger is CycleTrigger.SCREEN


def test_rsi_extreme_triggers_a_debate(config, state, snapshot, clock):
    snapshot = _quiet(snapshot)
    snapshot.indicators.rsi = E(25)
    assert evaluate_triggers(config, state, snapshot, clock.now()).should_debate


def test_volume_spike_triggers_a_debate(config, state, snapshot, clock):
    snapshot = _quiet(snapshot)
    snapshot.indicators.volume_ratio = E("3.0")
    assert evaluate_triggers(config, state, snapshot, clock.now()).should_debate


def test_an_open_position_suppresses_screening(config, state, snapshot, clock):
    snapshot = _quiet(snapshot)
    snapshot.indicators.rsi = E(20)
    state.open_position = object()
    result = evaluate_triggers(config, state, snapshot, clock.now())
    assert not result.should_debate
    assert "position is open" in result.suppressed_by


def test_a_halt_suppresses_screening(config, state, snapshot, clock):
    snapshot = _quiet(snapshot)
    snapshot.indicators.rsi = E(20)
    result = evaluate_triggers(
        config, state, snapshot, clock.now(),
        halts=[Halt(reason=HaltReason.KILL_SWITCH, detail="engaged")])
    assert not result.should_debate


def test_cooldown_blocks_a_second_debate(config, state, snapshot, clock):
    snapshot = _quiet(snapshot)
    snapshot.indicators.rsi = E(20)
    state.last_full_debate_at = clock.now() - timedelta(minutes=5)
    result = evaluate_triggers(config, state, snapshot, clock.now())
    assert not result.should_debate
    assert "cooldown" in result.suppressed_by


def test_daily_limit_blocks_further_debates(config, state, snapshot, clock):
    snapshot = _quiet(snapshot)
    snapshot.indicators.rsi = E(20)
    result = evaluate_triggers(config, state, snapshot, clock.now(),
                               debates_today=8)
    assert not result.should_debate
    assert "daily debate limit" in result.suppressed_by


def test_scheduled_floor_fires_without_a_market_trigger(config, state, snapshot,
                                                        clock):
    state.last_floor_run_at = None
    result = evaluate_triggers(config, state, _quiet(snapshot), clock.now())
    assert result.should_debate
    assert result.trigger is CycleTrigger.FLOOR


def test_the_floor_fires_once_per_slot(config, state, snapshot, clock):
    state.last_floor_run_at = None
    assert evaluate_triggers(config, state, _quiet(snapshot),
                             clock.now()).trigger is CycleTrigger.FLOOR
    state.last_floor_run_at = clock.now()
    clock.advance(minutes=10)
    assert not evaluate_triggers(config, state, _quiet(snapshot),
                                 clock.now()).should_debate


def test_flash_recovery_gets_exactly_one_debate(config, state, snapshot, clock):
    state.flash_recovery_pending = True
    state.flash_pause_until = clock.now() - timedelta(minutes=1)
    result = evaluate_triggers(config, state, _quiet(snapshot), clock.now())
    assert result.trigger is CycleTrigger.FLASH_RECOVERY


def test_flash_recovery_waits_for_the_pause_to_lift(config, state, snapshot, clock):
    state.flash_recovery_pending = True
    state.flash_pause_until = clock.now() + timedelta(minutes=30)
    result = evaluate_triggers(
        config, state, _quiet(snapshot), clock.now(),
        halts=[Halt(reason=HaltReason.FLASH_MOVE, detail="paused")])
    assert not result.should_debate


def test_the_degraded_budget_rung_cuts_the_daily_limit(config):
    from trade_agent.llm.budget import BudgetLadder, BudgetState

    normal = BudgetState(BudgetLadder.NORMAL, E(0), E(2900), E(0))
    degraded = BudgetState(BudgetLadder.DEGRADED, E(2400), E(2900), E(83))
    assert daily_debate_limit(config, normal) == 8
    assert daily_debate_limit(config, degraded) == 1
