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
from trade_agent.config import get_config
from trade_agent.roles import STRATEGISTS


@pytest.fixture
def config():
    return get_config()


def test_the_critique_brief_matches_the_number_of_other_proposals():
    others = len(STRATEGISTS) - 1
    text = prompts.critique_role(others=others)
    assert f"他の{others}案" in text

    for wrong in {1, 2, 3} - {others}:
        assert f"他の{wrong}案" not in text


def test_the_judge_brief_states_the_rule_actually_in_force(config):
    minimum = config.screening.consensus_min
    total = len(STRATEGISTS)
    text = prompts.judge_role(total=total, minimum=minimum)

    assert f"{total}案中{minimum}案未満" in text
    assert f"{total}つの独立提案" in text


def test_no_prompt_hardcodes_a_strategist_count(config):
    """The counts must arrive through the template, not be written into it."""
    rendered = {
        "constitution": prompts.constitution(config),
        "critique": prompts.critique_role(others=len(STRATEGISTS) - 1),
        "judge": prompts.judge_role(total=len(STRATEGISTS),
                                    minimum=config.screening.consensus_min),
        **prompts.ROLE_PROMPTS,
    }
    allowed = {str(len(STRATEGISTS)), str(len(STRATEGISTS) - 1),
               str(config.screening.consensus_min)}

    for name, text in rendered.items():
        for count in re.findall(r"(\d+)案", text):
            assert count in allowed, (
                f"{name} prompt says {count}案, but there are "
                f"{len(STRATEGISTS)} strategists and consensus_min is "
                f"{config.screening.consensus_min}")


def test_every_agent_the_cycle_runs_has_a_role_prompt():
    """A missing entry is a KeyError at the worst moment — inside a live
    decision cycle, after the analyst has already been paid for."""
    needed = {"analyst", *STRATEGISTS, "risk", "reflect", "scout"}
    assert needed <= set(prompts.ROLE_PROMPTS)


def test_no_role_prompt_survives_for_an_agent_that_was_removed():
    """The pessimist's brief stayed in the map after the agent was dropped
    would be dead text that reads as if the roster still had three."""
    assert not [a for a in prompts.ROLE_PROMPTS if "contrarian" in a]
