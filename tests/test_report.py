"""The owner-facing report, assembled in Python (formerly the A6 commander).

The point of composing it rather than generating it is that it cannot disagree
with what happened. These tests pin that: every number in the report comes from
the plan, and the agents' own sentences are quoted rather than paraphrased.
"""

from decimal import Decimal

from trade_agent.models.agent_io import AnalystOutput, RiskOutput
from trade_agent.models.state import SystemState
from trade_agent.models.trading import ExecutionPlan
from trade_agent.orchestrator.report import compose_no_trade, compose_traded
from trade_agent.timeutil import jst_date_str, jst_month_str

E = Decimal


def _analyst(**kwargs) -> AnalystOutput:
    return AnalystOutput(**{
        "regime": "range", "confidence": 0.55,
        "key_indicators": ["rsi", "vwap_24h"],
        "summary": "RSIは47.2で中立圏にある。",
        "risks": ["出来高の細り"], **kwargs})


def _plan(**kwargs) -> ExecutionPlan:
    return ExecutionPlan(**{
        "cycle_id": "c", "trade_id": "t", "entry": E(15000000),
        "stop_loss": E(14850000), "take_profit": E(15300000),
        "qty_btc": E("0.0006"), "risk_jpy": E(90),
        "thesis": "押し目を拾い、レンジ上限を目標とする。", **kwargs})


def _state(clock, config) -> SystemState:
    now = clock.now()
    return SystemState.initial(config.capital.initial_equity_jpy, now,
                               jst_date_str(now), jst_month_str(now))


def _risk() -> RiskOutput:
    return RiskOutput(approved=True, qty_btc=0.0006, stop_loss=14850000.0,
                      take_profit=15300000.0, risk_jpy=90.0,
                      rationale="1トレードリスク上限の範囲内。", adjustments=[])


# -- a cycle that traded --------------------------------------------------

def test_the_report_carries_every_number_from_the_plan(clock, config):
    _, body = compose_traded(
        analyst=_analyst(), plan=_plan(), risk=_risk(), state=_state(clock, config),
        buy_count=2, proposal_count=3, consensus=E("0.72"))

    assert "15,000,000" in body      # entry
    assert "14,850,000" in body      # stop
    assert "15,300,000" in body      # target
    assert "0.0006" in body          # size
    assert "90 円" in body           # risk


def test_the_percentages_are_computed_not_asserted(clock, config):
    _, body = compose_traded(
        analyst=_analyst(), plan=_plan(), risk=_risk(), state=_state(clock, config),
        buy_count=2, proposal_count=3, consensus=E("0.72"))

    assert "-1.00%" in body          # stop distance
    assert "+2.00%" in body          # target distance
    assert "2.00" in body            # reward:risk
    assert "equity の 0.90%" in body  # 90 JPY against 10,000


def test_the_agents_sentences_are_quoted_verbatim(clock, config):
    analyst, plan, risk = _analyst(), _plan(), _risk()
    _, body = compose_traded(
        analyst=analyst, plan=plan, risk=risk, state=_state(clock, config),
        buy_count=2, proposal_count=3, consensus=E("0.72"))

    assert analyst.summary in body
    assert plan.thesis in body
    assert risk.rationale in body


def test_the_consensus_is_reported_as_counted(clock, config):
    _, body = compose_traded(
        analyst=_analyst(), plan=_plan(), risk=_risk(), state=_state(clock, config),
        buy_count=2, proposal_count=3, consensus=E("0.72"))
    assert "3案中 2案が買い" in body
    assert "0.72" in body


def test_a_probe_says_so_in_the_headline_and_the_body(clock, config):
    headline, body = compose_traded(
        analyst=_analyst(), plan=_plan(probe=True), risk=_risk(),
        state=_state(clock, config), buy_count=1, proposal_count=3,
        consensus=E("0.4"))

    assert "偵察" in headline
    assert "退屈防止ルール" in body
    assert "統計的な優位性は" in body


def test_the_protection_note_is_included_when_given(clock, config):
    _, body = compose_traded(
        analyst=_analyst(), plan=_plan(), risk=_risk(), state=_state(clock, config),
        buy_count=2, proposal_count=3, consensus=E("0.72"),
        protection_note="損切りは取引所側の stop 注文")
    assert "損切りは取引所側の stop 注文" in body


# -- a cycle that did not trade -------------------------------------------

def test_a_no_trade_explains_itself(clock, config):
    headline, body = compose_no_trade(
        reason="consensus not reached: 1/3 buy proposals, 2 required",
        analyst=_analyst(), state=_state(clock, config), buy_count=1,
        proposal_count=3, consensus=None)

    assert "見送り" in headline
    assert "consensus not reached" in body
    assert "3案中 1案が買い" in body
    assert _analyst().summary in body


def test_a_no_trade_before_the_market_read_still_reports(clock, config):
    """An abort at the safety gate has no analyst output to quote."""
    headline, body = compose_no_trade(
        reason="halted: kill switch engaged", analyst=None,
        state=_state(clock, config), buy_count=0, proposal_count=0,
        consensus=None)

    assert "見送り" in headline
    assert "kill switch" in body


def test_a_long_reason_is_shortened_in_the_headline_only(clock, config):
    reason = "risk management rejected the plan: " + "詳細な理由。" * 20
    headline, body = compose_no_trade(
        reason=reason, analyst=None, state=_state(clock, config),
        buy_count=0, proposal_count=0, consensus=None)

    assert len(headline) < 60
    assert reason in body            # the full text survives in the body


# -- safety state ---------------------------------------------------------

def test_engaged_safety_rules_are_surfaced(clock, config):
    state = _state(clock, config)
    state.losing_streak = 2
    state.monthly.probe_rule_suspended = True

    _, body = compose_traded(
        analyst=_analyst(), plan=_plan(), risk=_risk(), state=state,
        buy_count=2, proposal_count=3, consensus=E("0.72"))

    assert "連敗 2" in body
    assert "probeルール当月停止" in body


def test_a_quiet_system_says_nothing_about_safety(clock, config):
    _, body = compose_traded(
        analyst=_analyst(), plan=_plan(), risk=_risk(), state=_state(clock, config),
        buy_count=2, proposal_count=3, consensus=E("0.72"))
    assert "安全装置" not in body
