"""`screening.consensus_min` — the largest single control on trade frequency.

Every cycle reaches phase 2; whether anything happens after that is decided
here. The first days of paper trading stopped at exactly this gate — analyst,
three proposals, three critiques, then nothing, because fewer than the required
number of strategists said "buy". The judge and the risk agent never ran and no
order was ever considered.

The threshold lived as a hardcoded 2 inside risk/boredom.py, which is both the
wrong home for it and invisible to anyone tuning the system.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from trade_agent.config import load_config
from trade_agent.errors import ConfigError
from trade_agent.risk.boredom import evaluate_boredom

E = Decimal


def test_the_threshold_comes_from_config(config, clock):
    from trade_agent.models.state import SystemState
    from trade_agent.timeutil import jst_date_str, jst_month_str

    now = clock.now()
    state = SystemState.initial(config.capital.initial_equity_jpy, now,
                                jst_date_str(now), jst_month_str(now))
    state.last_entry_at = now

    for wanted in (1, 2, 3):
        config.screening.consensus_min = wanted
        decision = evaluate_boredom(config, state, clock.now(), [])
        assert decision.consensus_min == wanted


def test_no_hardcoded_threshold_remains():
    """It was `normal_consensus = 2` in risk/boredom.py."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1]
              / "src" / "trade_agent" / "risk" / "boredom.py").read_text()
    assert "normal_consensus = 2" not in source
    assert "config.screening.consensus_min" in source


def test_the_shipped_default_lets_one_strategist_through():
    """The change this was made for: at 1, one strategist finding a setup is
    enough, and the other two are no longer able to veto it."""
    shipped = load_config(use_env=False)
    assert shipped.screening.consensus_min == 1
    assert shipped.boredom.enabled is False, (
        "with consensus_min at 1 the boredom rule has no lever left — its "
        "relaxed threshold equals the normal one")


@pytest.mark.parametrize("bad", [0, 4, -1])
def test_a_threshold_the_three_strategists_cannot_satisfy_is_refused(bad):
    """0 would trade on no proposal at all; 4 could never be reached."""
    config = load_config(use_env=False).model_copy(deep=True)
    config.screening.consensus_min = bad
    with pytest.raises(ConfigError, match="consensus_min"):
        config.__class__.model_validate(config.model_dump())


def test_the_loosened_screening_thresholds_are_what_shipped():
    """These decide how often a debate happens at all. Pinned because the
    first days produced no market trigger whatsoever — only the 09:00/21:00
    floors — and that is the symptom this change is aimed at."""
    shipped = load_config(use_env=False).screening
    assert shipped.rsi_low == E(40) and shipped.rsi_high == E(60)
    assert shipped.volume_spike_multiple == E("1.5")
    assert shipped.vwap_deviation_pct == E("0.5")
