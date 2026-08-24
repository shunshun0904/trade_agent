"""Risk and capital management (spec 6)."""

from datetime import timedelta
from decimal import Decimal

import pytest

from trade_agent.models.state import HaltReason, SystemState
from trade_agent.models.trading import TradeRecord
from trade_agent.risk.rules import RiskEngine
from trade_agent.timeutil import jst_date_str, jst_month_str

E = Decimal


@pytest.fixture
def risk(config):
    return RiskEngine(config)


@pytest.fixture
def state(clock, config):
    now = clock.now()
    return SystemState.initial(config.capital.initial_equity_jpy, now,
                               jst_date_str(now), jst_month_str(now))


def _closed_trade(trade_id: str, pnl: Decimal, *, probe: bool = False,
                  now=None) -> TradeRecord:
    return TradeRecord(
        trade_id=trade_id, cycle_id="c", pair="btc_jpy", probe=probe,
        qty_btc=E("0.0001"), entry_price=E(15000000), entry_order_id="o",
        entry_at=now, stop_loss=E(14900000), take_profit=E(15200000),
        exit_price=E(15000000) + pnl * 10000, exit_at=now, net_pnl_jpy=pnl,
        gross_pnl_jpy=pnl, closed=True)


def test_risk_limit_is_one_percent_of_equity(risk):
    assert risk.risk_limit_jpy(E(10000), probe=False) == E(100)
    assert risk.risk_limit_jpy(E(10000), probe=True) == E(50)


def test_size_is_capped_by_the_risk_budget(risk):
    # Cash is deliberately far larger than equity so the affordability clamp
    # cannot be what limits the size.
    result = risk.position_size(equity=E(1000000), entry=E(15000000),
                                stop_loss=E(14900000), jpy_available=E(9000000))
    # 1% of 1,000,000 = 10,000 JPY risk / 100,000 JPY per BTC = 0.1 BTC
    assert result.ok
    assert result.qty_btc == E("0.1")
    assert result.risk_jpy == E(10000)


def test_size_is_capped_by_available_cash(risk):
    result = risk.position_size(equity=E(10000), entry=E(15000000),
                                stop_loss=E(14990000), jpy_available=E(10000))
    # Cash allows only 0.0006 BTC even though the risk budget would allow more.
    assert result.qty_btc == E("0.0006")


def test_rejects_a_stop_too_wide_for_the_minimum_lot(risk):
    result = risk.position_size(equity=E(10000), entry=E(15000000),
                                stop_loss=E(12000000), jpy_available=E(10000))
    assert not result.ok
    assert "risk-based size is below the minimum lot" in result.reason


def test_distinguishes_an_underfunded_account_from_a_wide_stop(risk):
    # Enough risk budget for a lot, but not enough yen to buy one.
    result = risk.position_size(equity=E(1000000), entry=E(15000000),
                                stop_loss=E(14999000), jpy_available=E(500))
    assert not result.ok
    assert "affordable size" in result.reason


def test_rejects_an_inverted_stop(risk):
    result = risk.position_size(equity=E(10000), entry=E(15000000),
                                stop_loss=E(15100000), jpy_available=E(10000))
    assert not result.ok
    assert "stop_loss < entry" in result.reason


def test_phase_two_fixes_the_minimum_lot(config, risk):
    config.system.phase = 2
    result = risk.position_size(equity=E(1000000), entry=E(15000000),
                                stop_loss=E(14900000), jpy_available=E(1000000))
    assert result.qty_btc == config.exchange.min_order_btc


def test_three_losses_open_the_streak_brake(risk, state, clock, config):
    for index in range(3):
        risk.apply_trade_result(state, _closed_trade(f"t{index}", E(-30),
                                                     now=clock.now()), clock.now())
    assert state.losing_streak == 3
    assert state.in_losing_streak_pause(clock.now())
    halts = risk.evaluate_halts(state, clock.now())
    assert HaltReason.LOSING_STREAK in {h.reason for h in halts}

    clock.advance(hours=config.risk.losing_streak_pause_hours + 1)
    assert not state.in_losing_streak_pause(clock.now())


def test_a_win_clears_the_streak(risk, state, clock):
    risk.apply_trade_result(state, _closed_trade("a", E(-30), now=clock.now()),
                            clock.now())
    risk.apply_trade_result(state, _closed_trade("b", E(50), now=clock.now()),
                            clock.now())
    assert state.losing_streak == 0
    assert state.losing_streak_until is None


def test_probe_losses_do_not_feed_the_streak_brake(risk, state, clock):
    for index in range(5):
        risk.apply_trade_result(
            state, _closed_trade(f"p{index}", E(-10), probe=True, now=clock.now()),
            clock.now())
    assert state.losing_streak == 0
    assert state.monthly.probe_pnl_jpy == E(-50)


def test_probe_losses_suspend_the_rule_at_the_monthly_cap(risk, state, clock):
    # 2% of 10,000 = 200 JPY
    risk.apply_trade_result(
        state, _closed_trade("p", E(-200), probe=True, now=clock.now()), clock.now())
    assert state.monthly.probe_rule_suspended


def test_daily_loss_limit_halts_new_entries(risk, state, clock):
    risk.apply_trade_result(state, _closed_trade("a", E(-300), now=clock.now()),
                            clock.now())
    halts = risk.evaluate_halts(state, clock.now())
    assert HaltReason.DAILY_LOSS in {h.reason for h in halts}


def test_kill_switch_fires_at_twenty_percent_of_initial_capital(risk, state, clock):
    state.equity_jpy = E(8100)
    assert not risk.should_kill(state)
    state.equity_jpy = E(8000)
    assert risk.should_kill(state)


def test_kill_switch_measures_from_initial_capital_not_the_peak(risk, state, clock):
    # Doubling then halving is -50% from peak but +0% from initial: no kill.
    state.equity_jpy = E(20000)
    state.peak_equity_jpy = E(20000)
    state.equity_jpy = E(10000)
    assert not risk.should_kill(state)


def test_flash_move_arms_a_pause(risk, state, clock, config):
    assert risk.check_flash_move(state, clock.now(), E("-4.2"))
    assert state.in_flash_pause(clock.now())
    assert state.flash_recovery_pending
    clock.advance(hours=config.risk.flash_move_pause_hours, minutes=1)
    assert not state.in_flash_pause(clock.now())


def test_small_moves_do_not_arm_a_pause(risk, state, clock):
    assert not risk.check_flash_move(state, clock.now(), E("1.5"))
    assert state.flash_pause_until is None


def test_an_open_position_blocks_a_new_entry(risk, state, clock):
    state.open_position = object()
    halts = risk.evaluate_halts(state, clock.now())
    assert HaltReason.POSITION_OPEN in {h.reason for h in halts}


def test_repeated_api_failures_stand_the_system_down(risk, state, clock, config):
    state.consecutive_private_api_failures = config.exchange.private_failure_threshold
    halts = risk.evaluate_halts(state, clock.now())
    assert HaltReason.DATA_QUALITY in {h.reason for h in halts}
