"""Anthropic transport: prompt assembly, parsing, usage accounting (spec 11)."""

import json
from decimal import Decimal

import pytest

from trade_agent.errors import LLMError
from trade_agent.llm.anthropic_client import AnthropicLLMClient, _parse_json
from trade_agent.llm.base import LLMRequest
from trade_agent.llm.budget import CostMeter
from trade_agent.llm.registry import CONTRARIAN_AGENT, ModelRouter
from trade_agent.models.agent_io import AnalystOutput

E = Decimal


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Usage:
    def __init__(self, **kwargs):
        self.input_tokens = kwargs.get("input_tokens", 0)
        self.output_tokens = kwargs.get("output_tokens", 0)
        self.cache_read_input_tokens = kwargs.get("cache_read", 0)
        self.cache_creation_input_tokens = kwargs.get("cache_write", 0)


class _Message:
    def __init__(self, text, parsed=None, **usage):
        self.content = [_Block(text)]
        self.usage = _Usage(**usage)
        self.parsed_output = parsed


class _FakeMessages:
    def __init__(self, message):
        self.message = message
        self.parse_calls = []
        self.create_calls = []

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        return self.message

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return self.message


class _FakeClient:
    def __init__(self, message):
        self.messages = _FakeMessages(message)


def _request(config, agent="analyst", prefix="SHARED-PREFIX") -> LLMRequest:
    return LLMRequest(agent=agent, shared_prefix=prefix,
                      role_instruction="ROLE", task="TASK",
                      output_model=AnalystOutput, model=config.llm.model,
                      max_tokens=500)


def _analyst() -> AnalystOutput:
    return AnalystOutput(regime="range", confidence=0.5,
                         key_indicators=["rsi"], summary="s", risks=[])


def test_the_shared_prefix_is_the_cached_block(config):
    parsed = _analyst()
    fake = _FakeClient(_Message(parsed.model_dump_json(), parsed=parsed))
    client = AnthropicLLMClient("k", config.llm, client=fake)
    client.complete(_request(config))

    params = fake.messages.parse_calls[0]
    system = params["system"]
    assert system[0]["text"] == "SHARED-PREFIX"
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    # The role block is per-agent and must NOT be inside the cached prefix.
    assert system[1]["text"] == "ROLE"
    assert "cache_control" not in system[1]


def test_caching_can_be_switched_off(config):
    config.llm.use_prompt_cache = False
    parsed = _analyst()
    fake = _FakeClient(_Message(parsed.model_dump_json(), parsed=parsed))
    AnthropicLLMClient("k", config.llm, client=fake).complete(_request(config))
    assert "cache_control" not in fake.messages.parse_calls[0]["system"][0]


def test_usage_and_cost_are_recorded(config):
    parsed = _analyst()
    fake = _FakeClient(_Message(parsed.model_dump_json(), parsed=parsed,
                                input_tokens=1000, output_tokens=200,
                                cache_read=4000))
    meter = CostMeter(config.llm, config.cost)
    client = AnthropicLLMClient("k", config.llm, client=fake, cost_meter=meter)
    response = client.complete(_request(config))

    assert response.usage.input_tokens == 1000
    assert response.usage.cache_read_tokens == 4000
    assert response.cost_jpy > 0
    assert response.cost_jpy == meter.cost_jpy(response.usage)


def test_it_falls_back_when_structured_output_is_unsupported(config):
    parsed = _analyst()
    message = _Message(parsed.model_dump_json(), parsed=parsed)
    fake = _FakeClient(message)

    def refuse(**kwargs):
        raise RuntimeError("output_format is not supported for this model")

    fake.messages.parse = refuse
    client = AnthropicLLMClient("k", config.llm, client=fake)
    response = client.complete(_request(config))

    assert isinstance(response.parsed, AnalystOutput)
    assert fake.messages.create_calls, "it should have retried without the schema"
    # The fallback prompt carries the schema so the model still knows the shape.
    assert "Schema:" in fake.messages.create_calls[0]["messages"][-1]["content"]


def test_a_transport_failure_becomes_an_llm_error(config):
    fake = _FakeClient(_Message("{}"))

    def boom(**kwargs):
        raise RuntimeError("connection reset")

    fake.messages.parse = boom
    with pytest.raises(LLMError):
        AnthropicLLMClient("k", config.llm, client=fake).complete(_request(config))


# -- JSON recovery --------------------------------------------------------

def test_parses_a_bare_object():
    payload = _analyst().model_dump()
    result = _parse_json(json.dumps(payload), AnalystOutput, "analyst")
    assert result.regime == "range"


def test_parses_through_a_markdown_fence():
    payload = json.dumps(_analyst().model_dump())
    fenced = f"```json\n{payload}\n```"
    assert _parse_json(fenced, AnalystOutput, "analyst").regime == "range"


def test_parses_with_surrounding_prose():
    payload = json.dumps(_analyst().model_dump())
    assert _parse_json(f"はい、こちらです:\n{payload}\n以上です。",
                       AnalystOutput, "analyst").regime == "range"


def test_refuses_output_with_no_json():
    with pytest.raises(LLMError):
        _parse_json("申し訳ありませんが判断できません。", AnalystOutput, "analyst")


def test_refuses_output_that_breaks_the_schema():
    with pytest.raises(LLMError):
        _parse_json('{"regime": "sideways", "confidence": 2}', AnalystOutput,
                    "analyst")


# -- model routing (spec 4.2) --------------------------------------------

def test_all_agents_share_one_model_by_default(config):
    router = ModelRouter(config)
    assert router.model_for("analyst") == config.llm.model
    assert router.model_for(CONTRARIAN_AGENT) == config.llm.model
    assert not router.uses_alternate_model()


def test_the_contrarian_can_be_routed_to_another_model(config):
    config.llm.contrarian_model = "some-other-model"
    router = ModelRouter(config)
    assert router.model_for(CONTRARIAN_AGENT) == "some-other-model"
    assert router.model_for("strategy:trend") == config.llm.model
    assert router.uses_alternate_model()


def test_a_missing_api_key_is_fatal_when_trading_live(config):
    from trade_agent.errors import ConfigError
    from trade_agent.llm.registry import build_llm_client

    class NoSecrets:
        def get_optional(self, name):
            return None

    config.system.paper_trading = False
    config.llm.provider = "anthropic"
    with pytest.raises(ConfigError):
        build_llm_client(config, secrets=NoSecrets())


def test_the_offline_stub_may_never_drive_live_trading(config):
    from trade_agent.errors import ConfigError
    from trade_agent.llm.registry import build_llm_client

    config.system.paper_trading = False
    config.llm.provider = "stub"
    with pytest.raises(ConfigError) as excinfo:
        build_llm_client(config)
    assert "must never drive real orders" in str(excinfo.value)


def test_a_missing_api_key_falls_back_to_the_stub_on_paper(config):
    from trade_agent.llm.registry import build_llm_client
    from trade_agent.llm.stub import StubLLMClient

    class NoSecrets:
        def get_optional(self, name):
            return None

    config.system.paper_trading = True
    config.llm.provider = "anthropic"
    assert isinstance(build_llm_client(config, secrets=NoSecrets()),
                      StubLLMClient)


def test_the_constitution_drops_the_boredom_rule_when_it_is_off():
    """A strategist cited this rule by name as grounds to decline, while
    `boredom.enabled` was false. Listing a rule the system does not run is not
    a documentation slip — the agents obey the prompt, not the config."""
    from trade_agent.agents.prompts import constitution
    from trade_agent.config import get_config

    config = get_config()
    assert config.boredom.enabled is False, "fixture drifted; see below"
    assert "退屈防止" not in constitution(config)

    assert "3. 収益目標" in constitution(config), (
        "with the rule gone the profit target must move up to 3; a gap in the "
        "numbering reads as an omitted principle")

    enabled = config.model_copy(deep=True)
    enabled.boredom.enabled = True
    text = constitution(enabled)
    assert "3. 退屈防止ルール" in text
    assert "4. 収益目標" in text


def test_the_principles_stay_numbered_either_way():
    """The numbers are load-bearing: the constitution says a higher number
    never outranks a lower one."""
    import re

    from trade_agent.agents.prompts import constitution
    from trade_agent.config import get_config

    config = get_config()
    for enabled in (False, True):
        variant = config.model_copy(deep=True)
        variant.boredom.enabled = enabled
        numbers = re.findall(r"^(\d)\. ", constitution(variant), re.MULTILINE)
        assert numbers == [str(n) for n in range(1, len(numbers) + 1)], (
            f"principles must be 1..N with no gap or repeat: {numbers}")
