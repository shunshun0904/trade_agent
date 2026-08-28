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
    """Two rungs, not three. The middle one cut the day to a single debate at
    80% spent — a count standing in for money. Daily pacing does that job
    continuously now, leaving only the floor: at 100% nothing calls a model."""
    m = meter(config)
    assert m.evaluate(E(0)).ladder is BudgetLadder.NORMAL
    assert m.evaluate(E(2320)).ladder is BudgetLadder.NORMAL      # was DEGRADED
    assert m.evaluate(E(2899)).ladder is BudgetLadder.NORMAL
    assert m.evaluate(E(2900)).ladder is BudgetLadder.STOPPED
    assert m.evaluate(E(3000)).ladder is BudgetLadder.STOPPED


def test_only_the_stopped_rung_blocks_llm_calls(config):
    m = meter(config)
    assert m.evaluate(E(0)).llm_allowed
    assert m.evaluate(E(2400)).llm_allowed
    assert not m.evaluate(E(3000)).llm_allowed


def test_the_allowance_paces_the_month_across_its_days(config):
    """`(budget - spent) / days left * multiplier`. A day that spends heavily
    leaves less to divide tomorrow, which is what lets busy and quiet days
    both be correct without any count being involved."""
    from datetime import datetime, timezone

    m = meter(config)
    budget = config.cost.llm_budget_jpy
    mid_month = datetime(2026, 8, 15, 3, 0, tzinfo=timezone.utc)   # 17 days left

    fresh = m.daily_allowance_jpy(E(0), mid_month)
    spent_half = m.daily_allowance_jpy(budget / 2, mid_month)
    assert spent_half < fresh, "spending must shrink what is left to divide"

    # Later in the month the same remaining sum is spread over fewer days.
    month_end = datetime(2026, 8, 30, 3, 0, tzinfo=timezone.utc)   # 2 days left
    assert m.daily_allowance_jpy(E(0), month_end) > fresh


def test_an_exhausted_budget_allows_nothing(config):
    from datetime import datetime, timezone

    m = meter(config)
    now = datetime(2026, 8, 15, 3, 0, tzinfo=timezone.utc)
    assert m.daily_allowance_jpy(config.cost.llm_budget_jpy, now) == 0
    assert m.daily_allowance_jpy(config.cost.llm_budget_jpy * 2, now) == 0


def test_a_realistic_cycle_fits_the_monthly_budget(config):
    """Nine calls a cycle (1 analyst + 3 proposals + 3 critiques + judge +
    risk), eight cycles a day, thirty days must fit 2,900 JPY."""
    m = meter(config)
    per_call = m.cost_jpy(TokenUsage(input_tokens=2500, output_tokens=250,
                                     cache_read_tokens=2000))
    # Eight cycles a day was the old fixed cap; it stays the yardstick here
    # because the question is whether a realistic day of trading fits, not
    # what the cap happens to be.
    monthly = per_call * 9 * 8 * 30
    assert monthly < config.cost.llm_budget_jpy, (
        f"projected {monthly} JPY/month exceeds the "
        f"{config.cost.llm_budget_jpy} JPY budget")
