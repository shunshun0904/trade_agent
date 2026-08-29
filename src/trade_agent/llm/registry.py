"""Which model answers for which agent (spec 4.2).

Strategists built on one base model share that model's blind spots: they can be
confidently wrong together, and the consensus rule then rubber stamps the
shared error. `llm.agent_models` routes named agents to a different model so at
least one voice fails differently.

This replaced a `contrarian_model` setting that could only ever route
`strategy:contrarian`, an agent that no longer exists. Keys are agent ids
(`strategy:meanrev`, `judge`, ...); anything not listed uses `llm.model`. Every
model named here must appear in `llm.pricing.models`, or the ledger would bill
it at the default model's rate — the config refuses to load otherwise.
"""

from __future__ import annotations

import logging

from ..config import Config
from ..errors import ConfigError
from .base import LLMClient
from .budget import CostMeter

log = logging.getLogger(__name__)

class ModelRouter:
    def __init__(self, config: Config):
        self.config = config

    def model_for(self, agent: str) -> str:
        return self.config.llm.agent_models.get(agent, self.config.llm.model)

    def uses_alternate_model(self) -> bool:
        return any(model != self.config.llm.model
                   for model in self.config.llm.agent_models.values())


def build_llm_client(config: Config, *, secrets=None,
                     cost_meter: CostMeter | None = None) -> LLMClient:
    """Real client when an API key is reachable, stub otherwise.

    Falling back to the stub is only ever right for paper trading; with live
    trading enabled a missing key is a hard error, because a stubbed decision
    reaching the exchange would be the worst possible failure.
    """
    from .anthropic_client import AnthropicLLMClient
    from .stub import StubLLMClient

    meter = cost_meter or CostMeter(config.llm, config.cost)
    if config.llm.provider == "stub":
        if not config.system.paper_trading:
            # The stub is a rule-based fixture, not a strategy. Letting its
            # output reach a live exchange would be the worst failure this
            # system could have.
            raise ConfigError(
                "llm.provider is 'stub' while live trading is enabled; "
                "the offline stub must never drive real orders")
        return StubLLMClient(cost_meter=meter)

    api_key = None
    if secrets is not None:
        api_key = secrets.get_optional(config.secrets.ssm_anthropic_api_key)
    if not api_key:
        if not config.system.paper_trading:
            raise ConfigError(
                "no Anthropic API key available and live trading is enabled")
        log.warning("no Anthropic API key; using the offline stub (paper mode)")
        return StubLLMClient(cost_meter=meter)
    return AnthropicLLMClient(api_key, config.llm, cost_meter=meter)
