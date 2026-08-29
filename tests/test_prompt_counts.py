"""The prompts must not assert numbers the system disagrees with.

Three separate defects of this shape have now been found in production
prompts, each one changing what the agents decided:

* the constitution listed the boredom rule as an absolute principle while
  `boredom.enabled` was false, and a strategist cited it as grounds to decline;
* the judge brief said "buy案が3案中2案未満なら no_trade" while
  `screening.consensus_min` was 1 — twice the real threshold, asserted in the
  same sentence that tells the judge the machine enforces it;
* the critique brief asked for weaknesses in "他の2案" when there was one.

A model cannot check any of these against the code. These tests can.
"""

from __future__ import annotations

import re

import pytest

from trade_agent.agents import prompts
from trade_agent.agents import roster as prompts_roster
from trade_agent.config import get_config
from trade_agent.roles import STRATEGISTS


@pytest.fixture
def config():
    return get_config()


def test_no_prompt_hardcodes_a_strategist_count(config):
    """The counts must arrive through the template, not be written into it."""
    rendered = {"constitution": prompts.constitution(config),
                **prompts.ROLE_PROMPTS}
    allowed = {str(len(STRATEGISTS)), str(config.screening.consensus_min)}

    for name, text in rendered.items():
        for count in re.findall(r"(\d+)案", text):
            assert count in allowed, (
                f"{name} prompt says {count}案, but there are "
                f"{len(STRATEGISTS)} strategists and consensus_min is "
                f"{config.screening.consensus_min}")


def test_every_agent_the_cycle_runs_has_a_role_prompt():
    """A missing entry is a KeyError at the worst moment — inside a live
    decision cycle, after the analyst has already been paid for."""
    needed = {"analyst", *STRATEGISTS, "reflect", "scout"}
    assert needed <= set(prompts.ROLE_PROMPTS)


def test_no_role_prompt_survives_for_an_agent_that_was_removed():
    """A brief left in the map after its phase was deleted is dead text that
    reads like the protocol still has that step."""
    gone = ("contrarian", "trend", "meanrev", "critique", "judge", "risk")
    assert not [a for a in prompts.ROLE_PROMPTS if any(g in a for g in gone)]


def test_the_brief_squeezes_the_stop_from_both_sides():
    """A stop is not a number bolted on after the direction is chosen; it is
    what makes the trade sizeable at all. Both failure modes have to be in the
    brief, because only one of them is intuitive: too tight gets taken out by
    noise, and too wide drops the size below the minimum lot and gets the whole
    plan rejected — which nothing else in the system explains to the agent.
    """
    text = prompts.ROLE_PROMPTS[STRATEGISTS[0]]

    assert "atr_pct" in text                        # too tight
    assert "per_trade_risk_jpy" in text             # too wide
    assert "min_order_btc" in text                  # ...and why


def test_the_brief_says_the_decision_is_the_agents_own():
    """Nothing downstream re-judges whether the trade is worth taking. An
    agent that expects to be second-guessed proposes differently."""
    text = prompts.ROLE_PROMPTS[STRATEGISTS[0]]
    assert "あなただけ" in text


def test_every_snapshot_field_the_strategist_is_told_to_read_exists():
    """The brief tells the agent to check `constraints.per_trade_risk_jpy` and
    `indicators.atr_pct` by name. A name that is not in the snapshot is an
    instruction the model cannot follow and has no way to detect — it will
    either invent the value or quietly skip the check."""
    from trade_agent.models.market import (
        AccountState,
        Indicators,
        TradingConstraints,
    )

    text = prompts.ROLE_PROMPTS[STRATEGISTS[0]]
    known = {
        "constraints": set(TradingConstraints.model_fields),
        "indicators": set(Indicators.model_fields),
        "account": set(AccountState.model_fields),
    }
    referenced = re.findall(r"\b(constraints|indicators|account)\.(\w+)", text)
    assert referenced, "the brief should name the fields it relies on"

    for block, field in referenced:
        assert field in known[block], (
            f"the strategist brief says {block}.{field}, which the snapshot "
            f"does not carry")


def test_the_strategist_is_told_what_the_analyst_gave_it():
    """A1 costs a call per cycle. Its output is handed to A2 in the task
    payload under `market_read`, but for a while nothing in A2's brief
    mentioned that it existed or what to do with it — which makes the analyst
    call something the system pays for and does not use."""
    text = prompts.ROLE_PROMPTS[STRATEGISTS[0]]
    assert "market_read" in text
    for field in ("regime", "confidence", "summary", "risks"):
        assert field in text, f"the brief never names market_read.{field}"


def test_the_strategist_is_told_to_weigh_capital_and_recent_losses():
    """Removing A3 and A4 only works if what they weighed is weighed here.
    A brief that talks about entry and stop geometry but never about how much
    money there is, or how the last few trades went, has moved the decision
    without moving the inputs to it."""
    text = prompts.ROLE_PROMPTS[STRATEGISTS[0]]
    for topic in ("account.equity_jpy", "account.jpy_free",
                  "constraints.per_trade_risk_jpy", "連敗", "当日実現損益",
                  "教訓"):
        assert topic in text, f"the brief never mentions {topic}"


def test_the_analyst_read_actually_reaches_the_strategist():
    """Checked against the assembled request, not the brief: the payload key
    the brief tells the agent to read has to be the key the roster sends."""
    from trade_agent.models.agent_io import AnalystOutput

    sent = {}

    class _Runner:
        def run(self, agent, payload, model, **kwargs):
            sent.update(payload=payload, instructions=kwargs.get("instructions", ""))
            raise RuntimeError("stop here; only the payload matters")

    analyst = AnalystOutput(regime="range", confidence=0.6, key_indicators=["rsi"],
                            summary="往来", risks=["出来高の細り"])
    with pytest.raises(RuntimeError):
        prompts_roster.run_strategy(_Runner(), STRATEGISTS[0], analyst)

    assert "market_read" in sent["payload"]
    assert sent["payload"]["market_read"]["regime"] == "range"
    assert "market_read" in sent["instructions"]
