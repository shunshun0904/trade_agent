"""Configuration loading and the rule-priority invariants (spec 0)."""

from decimal import Decimal

import pytest

from trade_agent.config import load_config
from trade_agent.errors import ConfigError

E = Decimal


def test_defaults_match_the_specification(config):
    assert config.exchange.pair == "btc_jpy"
    assert config.exchange.min_order_btc == E("0.0001")
    assert config.capital.initial_equity_jpy == E(10000)
    assert config.risk.per_trade_risk_pct == E(1)
    assert config.risk.daily_loss_limit_pct == E(3)
    assert config.risk.killswitch_drawdown_pct == E(20)
    assert config.risk.max_concurrent_positions == 1
    assert config.boredom.no_trade_hours == 72
    assert config.cost.daily_allowance_multiplier == E("2.0")
    assert config.llm.model == "claude-haiku-4-5"


def test_paper_trading_is_the_default(config):
    assert config.system.paper_trading is True
    assert config.system.phase == 1


def test_environment_variables_override_the_file(monkeypatch):
    monkeypatch.setenv("TA_RISK__PER_TRADE_RISK_PCT", "0.5")
    monkeypatch.setenv("TA_SYSTEM__PAPER_TRADING", "false")
    cfg = load_config()
    assert cfg.risk.per_trade_risk_pct == E("0.5")
    assert cfg.system.paper_trading is False


def test_unknown_keys_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("system:\n  not_a_real_setting: 1\n")
    with pytest.raises(ConfigError):
        load_config(path, use_env=False)


def test_probe_risk_may_not_exceed_normal_risk(tmp_path, config):
    """Spec 0: the entertainment rule cannot loosen a risk rule."""
    data = config.model_dump(mode="json")
    data["risk"]["probe_risk_pct"] = 5.0  # above per_trade_risk_pct of 1.0
    import yaml

    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError) as excinfo:
        load_config(path, use_env=False)
    assert "3-day rule may not relax a risk rule" in str(excinfo.value)


def test_the_daily_limit_must_sit_below_the_kill_switch(tmp_path, config):
    data = config.model_dump(mode="json")
    data["risk"]["daily_loss_limit_pct"] = 25.0
    import yaml

    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError):
        load_config(path, use_env=False)


def test_infrastructure_cost_cannot_consume_the_whole_budget(tmp_path, config):
    data = config.model_dump(mode="json")
    data["cost"]["infra_cost_jpy"] = 3000
    import yaml

    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError):
        load_config(path, use_env=False)


def test_money_values_are_decimals_not_floats(config):
    assert isinstance(config.exchange.min_order_btc, Decimal)
    assert isinstance(config.capital.initial_equity_jpy, Decimal)
    assert isinstance(config.exchange.maker_fee_rate, Decimal)
