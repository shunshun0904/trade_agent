"""Deterministic guard (spec 5)."""

from decimal import Decimal

import pytest

from trade_agent.errors import GuardRejection
from trade_agent.guards.deterministic import DeterministicGuard, check_quoted_indicators
from trade_agent.models.agent_io import (
    AnalystOutput,
    JudgeOutput,
    RiskOutput,
    StrategyOutput,
)


@pytest.fixture
def guard(config, snapshot):
    return DeterministicGuard(config, snapshot)


def _buy(snapshot, *, entry=None, sl_mult="0.99", tp_mult="1.02"):
    entry = entry if entry is not None else snapshot.last_price
    return StrategyOutput(
        action="buy", entry=float(entry),
        stop_loss=float(entry * Decimal(sl_mult)),
        take_profit=float(entry * Decimal(tp_mult)),
        confidence=0.6, thesis="押し目を拾う", invalidation="安値割れ")


def test_accepts_a_well_formed_long(guard, snapshot):
    guard.validate_strategy(_buy(snapshot))


def test_rejects_stop_above_entry(guard, snapshot):
    bad = _buy(snapshot, sl_mult="1.01")
    with pytest.raises(GuardRejection) as excinfo:
        guard.validate_strategy(bad)
    assert any("価格関係が不正" in v for v in excinfo.value.violations)


def test_rejects_entry_far_from_market(guard, snapshot):
    far = snapshot.last_price * Decimal("1.05")
    with pytest.raises(GuardRejection) as excinfo:
        guard.validate_strategy(_buy(snapshot, entry=far))
    assert any("乖離" in v for v in excinfo.value.violations)


def test_rejects_target_inside_the_round_trip_fee(guard, snapshot):
    # Round-trip maker cost is negative here (a rebate), so use a target that
    # is inside it by construction: one tick above entry.
    entry = snapshot.last_price
    output = StrategyOutput(
        action="buy", entry=float(entry), stop_loss=float(entry * Decimal("0.99")),
        take_profit=float(entry), confidence=0.5, thesis="x", invalidation="y")
    with pytest.raises(GuardRejection):
        guard.validate_strategy(output)


def test_wait_must_not_carry_prices(guard):
    output = StrategyOutput(action="wait", entry=15000000.0, take_profit=None,
                            stop_loss=None, confidence=0.5, thesis="見送り",
                            invalidation="なし")
    with pytest.raises(GuardRejection) as excinfo:
        guard.validate_strategy(output)
    assert any("null" in v for v in excinfo.value.violations)


def test_rejects_fabricated_indicator_values(guard, snapshot):
    true_rsi = snapshot.indicators.rsi
    lie = true_rsi + Decimal(20)
    output = AnalystOutput(regime="range", confidence=0.5,
                           key_indicators=["rsi"],
                           summary=f"RSIは{lie}で中立圏にある。",
                           risks=["ボラ拡大"])
    with pytest.raises(GuardRejection) as excinfo:
        guard.validate_analyst(output)
    assert any("引用値が実値と一致しない" in v for v in excinfo.value.violations)


def test_accepts_correctly_quoted_indicator_values(guard, snapshot):
    rsi = snapshot.indicators.rsi
    guard.validate_analyst(AnalystOutput(
        regime="range", confidence=0.5, key_indicators=["rsi"],
        summary=f"RSIは{rsi:.1f}である。", risks=[]))


def test_threshold_phrasing_is_not_treated_as_a_quote(snapshot, config):
    # "RSIが70を超えたら" is a plan, not a claim about the current value.
    violations = check_quoted_indicators(
        "RSIが70を超えたら利確する。", snapshot, config.guard.indicator_tolerance_pct)
    assert violations == []


def test_rejects_unknown_indicator_names(guard):
    with pytest.raises(GuardRejection) as excinfo:
        guard.validate_analyst(AnalystOutput(
            regime="trend_up", confidence=0.8,
            key_indicators=["ichimoku_cloud"], summary="上昇", risks=[]))
    assert any("存在しない指標名" in v for v in excinfo.value.violations)


def test_judge_cannot_adopt_without_consensus(guard, snapshot):
    entry = snapshot.last_price
    output = JudgeOutput(decision="adopt", consensus=0.9, adopted_proposal_id="P1",
                         entry=float(entry), take_profit=float(entry * Decimal("1.02")),
                         stop_loss=float(entry * Decimal("0.99")), rationale="採用")
    with pytest.raises(GuardRejection) as excinfo:
        guard.validate_judge(output, proposal_ids=["P1", "P2", "P3"],
                             buy_count=1, consensus_min=2)
    assert any("満たない" in v for v in excinfo.value.violations)


def test_judge_cannot_adopt_an_unknown_proposal(guard, snapshot):
    entry = snapshot.last_price
    output = JudgeOutput(decision="adopt", consensus=0.9, adopted_proposal_id="P9",
                         entry=float(entry), take_profit=float(entry * Decimal("1.02")),
                         stop_loss=float(entry * Decimal("0.99")), rationale="採用")
    with pytest.raises(GuardRejection):
        guard.validate_judge(output, proposal_ids=["P1", "P2", "P3"],
                             buy_count=3, consensus_min=2)


def test_risk_output_must_match_the_python_numbers(guard, snapshot):
    entry = snapshot.last_price
    stop = entry * Decimal("0.99")
    output = RiskOutput(approved=True, qty_btc=0.0009, stop_loss=float(stop),
                        take_profit=float(entry * Decimal("1.02")), risk_jpy=90.0,
                        rationale="範囲内", adjustments=[])
    with pytest.raises(GuardRejection) as excinfo:
        guard.validate_risk(output, expected_qty=Decimal("0.0006"),
                            expected_risk_jpy=Decimal("90"), entry=entry,
                            stop_loss=stop)
    assert any("qty_btc" in v for v in excinfo.value.violations)


def test_executable_check_rejects_a_non_lot_multiple(guard, snapshot):
    entry = snapshot.last_price
    violations = guard.check_executable(
        entry=entry, stop_loss=entry * Decimal("0.99"),
        take_profit=entry * Decimal("1.02"), qty_btc=Decimal("0.00015"),
        jpy_available=Decimal(10000))
    assert any("整数倍" in v for v in violations)


def test_executable_check_rejects_an_unaffordable_order(guard, snapshot):
    entry = snapshot.last_price
    violations = guard.check_executable(
        entry=entry, stop_loss=entry * Decimal("0.999"),
        take_profit=entry * Decimal("1.02"), qty_btc=Decimal("0.01"),
        jpy_available=Decimal(10000))
    assert any("利用可能残高" in v for v in violations)


def test_executable_check_rejects_risk_over_the_limit(guard, snapshot):
    entry = snapshot.last_price
    violations = guard.check_executable(
        entry=entry, stop_loss=entry * Decimal("0.90"),
        take_profit=entry * Decimal("1.02"), qty_btc=Decimal("0.0006"),
        jpy_available=Decimal(10000))
    assert any("1トレード上限" in v for v in violations)
