"""Typed configuration.

Loaded from ``config/default.yaml``, overlaid with ``config/<env>.yaml``, then
with ``TA_<SECTION>__<KEY>`` environment variables (Lambda's override channel).

Nothing here reads a secret. Secrets live in SSM Parameter Store and are
fetched by :mod:`trade_agent.storage.secrets` (spec 12).
"""

from __future__ import annotations

import functools
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from .errors import ConfigError

ENV_PREFIX = "TA_"
CONFIG_DIR_ENV = "TA_CONFIG_DIR"


class _Section(BaseModel):
    model_config = {"extra": "forbid"}


class SystemConfig(_Section):
    environment: str = "dev"
    paper_trading: bool = True
    phase: int = 1
    display_timezone: str = "Asia/Tokyo"
    log_level: str = "INFO"


class RateLimitConfig(_Section):
    query_per_second: float = 10
    update_per_second: float = 6


class ExchangeConfig(_Section):
    name: str = "bitbank"
    pair: str = "btc_jpy"
    public_base_url: str
    private_base_url: str
    min_order_btc: Decimal
    amount_digits: int = 8
    price_digits: int = 0
    maker_fee_rate: Decimal
    taker_fee_rate: Decimal
    verify_pair_settings: bool = True
    http_timeout_seconds: float = 10
    max_retries: int = 4
    retry_base_delay_seconds: float = 1.0
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    private_failure_threshold: int = 3

    @property
    def base_asset(self) -> str:
        return self.pair.split("_")[0]

    @property
    def quote_asset(self) -> str:
        return self.pair.split("_")[1]


class CapitalConfig(_Section):
    initial_equity_jpy: Decimal
    monthly_target_pct: Decimal = Decimal(10)


class RiskConfig(_Section):
    per_trade_risk_pct: Decimal
    probe_risk_pct: Decimal
    daily_loss_limit_pct: Decimal
    losing_streak_limit: int
    losing_streak_pause_hours: int
    flash_move_pct: Decimal
    flash_move_window_minutes: int
    flash_move_pause_hours: int
    killswitch_drawdown_pct: Decimal
    max_concurrent_positions: int = 1


class BoredomConfig(_Section):
    enabled: bool = True
    no_trade_hours: int = 72
    relaxed_consensus_min: int = 1
    probe_sl_pct: Decimal = Decimal("0.7")
    monthly_probe_loss_limit_pct: Decimal = Decimal("2.0")
    mechanical_probe_vwap_hours: int = 24
    mechanical_probe_discount_pct: Decimal = Decimal("0.2")


class GuardConfig(_Section):
    max_retries: int = 3
    entry_max_deviation_pct: Decimal = Decimal("2.0")
    indicator_tolerance_pct: Decimal = Decimal("0.5")


class ExecutionConfig(_Section):
    requote_max_deviation_pct: Decimal = Decimal("0.3")
    post_only_timeout_minutes: int = 60
    tp_exit_timeout_minutes: int = 30
    lock_lease_seconds: int = 600
    pending_order_stale_minutes: int = 90
    oco_mode: Literal["local", "exchange_oco", "auto"] = "auto"
    stop_order_type: Literal["stop", "stop_limit"] = "stop"
    stop_limit_offset_pct: Decimal = Decimal("0.3")
    local_backstop: bool = True


class ScheduleConfig(_Section):
    tick_minutes: int = 5
    screen_minutes: int = 30
    full_debate_cooldown_minutes: int = 30
    daily_full_debate_limit: int = 8
    floor_times_jst: list[str] = Field(default_factory=lambda: ["09:00", "21:00"])
    daily_report_time_jst: str = "21:00"


class ScreeningConfig(_Section):
    breakout_lookback_hours: int = 24
    rsi_period: int = 14
    rsi_low: Decimal = Decimal(40)
    rsi_high: Decimal = Decimal(60)
    volume_spike_multiple: Decimal = Decimal("1.5")
    vwap_deviation_pct: Decimal = Decimal("0.5")
    scout_mode: bool = False
    # How many of the three independent proposals must say "buy" before the
    # judge is called at all. This is the single largest control on how often
    # the system trades: with the three strategists reaching phase 2 and
    # stopping here, nothing downstream ever runs. Lived as a hardcoded 2 in
    # risk/boredom.py until it became the thing being tuned.
    consensus_min: int = 1


class SnapshotConfig(_Section):
    candle_type: str = "1hour"
    candle_limit: int = 200
    short_candle_type: str = "5min"
    short_candle_limit: int = 120
    depth_levels: int = 10
    lessons_in_prompt: int = 12


class PricingConfig(_Section):
    input_per_mtok_usd: Decimal
    output_per_mtok_usd: Decimal
    cache_write_multiplier: Decimal = Decimal("1.25")
    cache_read_multiplier: Decimal = Decimal("0.10")
    batch_multiplier: Decimal = Decimal("0.50")


class LLMConfig(_Section):
    provider: str = "anthropic"
    model: str = "claude-haiku-4-5"
    contrarian_model: str | None = None
    max_tokens: int = 2000
    timeout_seconds: float = 60
    max_api_retries: int = 2
    use_prompt_cache: bool = True
    cache_min_tokens: int = 4096
    structured_output: bool = True
    batch_api_for_reflect: bool = True
    pricing: PricingConfig
    usd_jpy_rate: Decimal


class CostConfig(_Section):
    total_budget_jpy: Decimal
    infra_cost_jpy: Decimal
    degrade_threshold_pct: Decimal = Decimal(80)
    stop_threshold_pct: Decimal = Decimal(100)
    degraded_daily_debate_limit: int = 1

    @property
    def llm_budget_jpy(self) -> Decimal:
        """Spec 1/11: total budget minus measured infrastructure spend."""
        return self.total_budget_jpy - self.infra_cost_jpy


class ReflectConfig(_Section):
    min_trades_for_lessons: int = 20
    lookback_trades: int = 60
    max_lessons_per_run: int = 5


class StorageConfig(_Section):
    backend: str = "dynamodb"
    table_prefix: str = "trade-agent"
    s3_bucket: str = ""
    agent_log_prefix: str = "agent-logs/"
    backup_prefix: str = "backups/"

    def table(self, name: str) -> str:
        return f"{self.table_prefix}-{name}"


class NotifyConfig(_Section):
    enabled: bool = True
    ses_region: str = "ap-northeast-1"
    from_address: str = ""
    to_address: str = ""


class MCPConfig(_Section):
    server_name: str = "trade-agent"
    protocol_version: str = "2025-06-18"
    ssm_bearer_token_param: str = "/trade-agent/mcp/bearer-token"


class SecretsConfig(_Section):
    ssm_bitbank_api_key: str
    ssm_bitbank_api_secret: str
    ssm_anthropic_api_key: str


class Config(_Section):
    system: SystemConfig
    exchange: ExchangeConfig
    capital: CapitalConfig
    risk: RiskConfig
    boredom: BoredomConfig
    guard: GuardConfig
    execution: ExecutionConfig
    schedule: ScheduleConfig
    screening: ScreeningConfig
    snapshot: SnapshotConfig
    llm: LLMConfig
    cost: CostConfig
    reflect: ReflectConfig
    storage: StorageConfig
    notify: NotifyConfig
    mcp: MCPConfig
    secrets: SecretsConfig

    @model_validator(mode="after")
    def _check_priority_order(self) -> "Config":
        """Spec 0: the safety rules outrank the entertainment rule.

        A probe must never be able to risk more than an ordinary trade, and the
        kill switch must trip before the daily loss limit becomes irrelevant.
        """
        if not 1 <= self.screening.consensus_min <= 3:
            raise ConfigError(
                "screening.consensus_min must be between 1 and 3; there are "
                "three strategists, so anything else can never be satisfied "
                "or can never be missed")
        if self.risk.probe_risk_pct > self.risk.per_trade_risk_pct:
            raise ConfigError(
                "boredom probe risk exceeds the per-trade risk limit; "
                "the 3-day rule may not relax a risk rule (spec 0)"
            )
        if self.risk.daily_loss_limit_pct >= self.risk.killswitch_drawdown_pct:
            raise ConfigError(
                "daily loss limit must be smaller than the kill-switch drawdown"
            )
        if self.cost.llm_budget_jpy <= 0:
            raise ConfigError("infra_cost_jpy leaves no LLM budget")
        return self


def _config_dir() -> Path:
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "config"


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", ""}:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def _env_overlay(known_sections: set[str]) -> dict:
    """`TA_RISK__PER_TRADE_RISK_PCT=0.5` -> {"risk": {"per_trade_risk_pct": 0.5}}."""
    overlay: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX) or "__" not in key:
            continue
        section, _, field = key[len(ENV_PREFIX):].partition("__")
        section = section.lower()
        if section not in known_sections:
            continue
        cursor = overlay.setdefault(section, {})
        parts = [p.lower() for p in field.split("__")]
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _coerce(value)
    return overlay


def load_config(path: str | Path | None = None, *, use_env: bool = True) -> Config:
    """Build the effective configuration.

    `path` overrides the default file for tests; the environment overlay still
    applies unless `use_env` is False.
    """
    config_dir = _config_dir()
    default_path = Path(path) if path else config_dir / "default.yaml"
    if not default_path.exists():
        raise ConfigError(f"config file not found: {default_path}")
    data = yaml.safe_load(default_path.read_text()) or {}

    env_name = os.environ.get("TA_ENV") or data.get("system", {}).get("environment", "dev")
    env_path = config_dir / f"{env_name}.yaml"
    if path is None and env_path.exists() and env_path != default_path:
        data = _deep_merge(data, yaml.safe_load(env_path.read_text()) or {})
        data.setdefault("system", {})["environment"] = env_name

    if use_env:
        data = _deep_merge(data, _env_overlay(set(Config.model_fields)))

    try:
        return Config.model_validate(data)
    except ConfigError:
        raise
    except Exception as exc:  # pydantic ValidationError
        raise ConfigError(f"invalid configuration: {exc}") from exc


@functools.lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide config, cached for Lambda warm starts."""
    return load_config()


def reset_config_cache() -> None:
    get_config.cache_clear()
