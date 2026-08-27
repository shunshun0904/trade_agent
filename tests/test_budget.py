"""LLM cost accounting and the budget ladder (spec 11)."""

from decimal import Decimal

from trade_agent.llm.base import TokenUsage
from trade_agent.llm.budget import BudgetLadder, CostMeter

E = Decimal


def meter(config) -> CostMeter:
    return CostMeter(config.llm, config.cost)


def test_budget_is_total_minus_infrastructure(config):
    assert config.cost.total_budget_jpy == E(3000)
    assert config.cost.infra_cost_jpy == E(100)
    assert config.cost.llm_budget_jpy == E(2900)


def test_cache_reads_are_charged_at_a_tenth(config):
    m = meter(config)
    uncached = m.cost_jpy(TokenUsage(input_tokens=10000))
    cached = m.cost_jpy(TokenUsage(cache_read_tokens=10000))
    assert cached == uncached * config.llm.pricing.cache_read_multiplier


def test_cache_writes_cost_more_than_plain_input(config):
    m = meter(config)
    written = m.cost_jpy(TokenUsage(cache_write_tokens=1000))
    plain = m.cost_jpy(TokenUsage(input_tokens=1000))
    assert written > plain


def test_batch_calls_are_half_price(config):
    m = meter(config)
    usage = TokenUsage(input_tokens=5000, output_tokens=1000)
    assert m.cost_jpy(usage, batch=True) == m.cost_jpy(usage) / 2


def test_ladder_thresholds(config):
    m = meter(config)
    assert m.evaluate(E(0)).ladder is BudgetLadder.NORMAL
    assert m.evaluate(E(2319)).ladder is BudgetLadder.NORMAL
    # 80% of 2,900 = 2,320
    assert m.evaluate(E(2320)).ladder is BudgetLadder.DEGRADED
    assert m.evaluate(E(2899)).ladder is BudgetLadder.DEGRADED
    assert m.evaluate(E(2900)).ladder is BudgetLadder.STOPPED


def test_only_the_stopped_rung_blocks_llm_calls(config):
    m = meter(config)
    assert m.evaluate(E(0)).llm_allowed
    assert m.evaluate(E(2400)).llm_allowed
    assert not m.evaluate(E(3000)).llm_allowed


def test_degraded_rung_caps_the_daily_debates(config):
    m = meter(config)
    assert m.evaluate(E(0)).daily_debate_limit_override is None
    assert m.evaluate(E(2400)).daily_debate_limit_override == 1


def test_a_realistic_cycle_fits_the_monthly_budget(config):
    """Nine calls a cycle (1 analyst + 3 proposals + 3 critiques + judge +
    risk), eight cycles a day, thirty days must fit 2,900 JPY."""
    m = meter(config)
    per_call = m.cost_jpy(TokenUsage(input_tokens=2500, output_tokens=250,
                                     cache_read_tokens=2000))
    monthly = per_call * 9 * config.schedule.daily_full_debate_limit * 30
    assert monthly < config.cost.llm_budget_jpy, (
        f"projected {monthly} JPY/month exceeds the "
        f"{config.cost.llm_budget_jpy} JPY budget")
