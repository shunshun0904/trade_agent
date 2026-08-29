"""Lambda handlers (spec 9, 17.1)."""

from datetime import timedelta
from decimal import Decimal

import pytest

from trade_agent.handlers.decide import run as decide_run
from trade_agent.handlers.reflect import build_statistics, run as reflect_run
from trade_agent.handlers.screen import run as screen_run
from trade_agent.handlers.tick import run as tick_run
from trade_agent.models.trading import TradeRecord

E = Decimal


def _closed(index: int, pnl: Decimal, clock, *, probe=False, regime="range"):
    return TradeRecord(
        trade_id=f"t{index}", cycle_id=f"c{index}", pair="btc_jpy", probe=probe,
        qty_btc=E("0.0003"), entry_price=E(15000000), entry_order_id=f"o{index}",
        entry_at=clock.now() - timedelta(hours=index + 1),
        stop_loss=E(14900000), take_profit=E(15200000),
        exit_price=E(15000000) + pnl, exit_at=clock.now() - timedelta(hours=index),
        exit_reason="take_profit" if pnl > 0 else "stop_loss",
        fee_jpy=E("0.5"), gross_pnl_jpy=pnl, net_pnl_jpy=pnl, closed=True,
        regime=regime)


# -- tick -----------------------------------------------------------------

def test_tick_records_a_heartbeat_and_equity(ctx):
    result = tick_run(ctx)
    assert result["equity_jpy"] == 10000.0
    assert ctx.load_state().last_tick_at is not None


def test_tick_writes_the_equity_curve(ctx):
    tick_run(ctx)
    from trade_agent.timeutil import jst_date_str

    point = ctx.store.equity.get(jst_date_str(ctx.clock.now()))
    assert point is not None
    assert point.equity_jpy == E(10000)


def test_tick_engages_the_kill_switch_and_emails(ctx, market):
    # Drop the simulated account's cash so equity falls below the threshold.
    ctx.exchange.inner.account.jpy_free = E(7000)
    result = tick_run(ctx)
    assert result["kill_switch"] is True
    assert any("キルスイッチ" in subject for subject, _ in ctx.notifier.sent)


def test_tick_survives_a_market_data_failure(ctx, market):
    from trade_agent.errors import ExchangeError

    market.fail_next = ExchangeError("bitbank is down")
    result = tick_run(ctx)
    assert "error" in result
    assert ctx.load_state().consecutive_private_api_failures == 1


def test_repeated_api_failures_notify_the_owner(ctx, market, config):
    from trade_agent.errors import ExchangeError

    for _ in range(config.exchange.private_failure_threshold):
        market.fail_next = ExchangeError("bitbank is down")
        tick_run(ctx)
    assert any("APIの連続失敗" in subject for subject, _ in ctx.notifier.sent)


# -- screen ---------------------------------------------------------------

def test_screen_invokes_decide_when_a_trigger_fires(ctx, llm, clock):
    state = ctx.load_state()
    state.last_floor_run_at = clock.now()
    state.last_full_debate_at = clock.now() - timedelta(hours=2)
    ctx.save_state(state)

    market = ctx.exchange.market
    market.quiet()
    market.price = market.price  # keep the level
    # Force a breakout: the last price sits at the 24h high.
    market.candle_range = E(0)

    result = screen_run(ctx)
    assert result["debate"] is True
    assert result["invoked"] == "inline"
    assert llm.calls, "the inline decide cycle should have run"


def test_screen_stands_down_while_halted(ctx, llm):
    state = ctx.load_state()
    state.kill_switch = True
    ctx.save_state(state)
    result = screen_run(ctx)
    assert result["debate"] is False
    assert llm.calls == []


# -- decide ---------------------------------------------------------------

def test_decide_returns_a_structured_summary(ctx, llm):
    result = decide_run(ctx, {"trigger": "manual", "cycle_id": "h-1"})
    assert result["cycle_id"] == "h-1"
    assert result["traded"] is True
    assert result["llm_calls"] > 0


def test_decide_is_a_no_op_the_second_time(ctx, llm):
    decide_run(ctx, {"trigger": "manual", "cycle_id": "h-2"})
    again = decide_run(ctx, {"trigger": "manual", "cycle_id": "h-2"})
    assert "skipped" in again
    assert len(ctx.exchange.orders_sent) == 1


def test_decide_writes_the_daily_report_at_the_configured_hour(ctx, llm, clock):
    # 21:00 JST is 12:00 UTC.
    clock.current = clock.current.replace(hour=12, minute=5)
    decide_run(ctx, {"trigger": "floor", "cycle_id": "h-3"})
    report = ctx.store.reports.get()
    assert report is not None
    assert "equity" in report.report_text


def test_an_unknown_trigger_is_treated_as_manual(ctx, llm):
    result = decide_run(ctx, {"trigger": "nonsense", "cycle_id": "h-4"})
    assert result["trigger"] == "manual"


# -- reflect --------------------------------------------------------------

def test_reflect_refuses_to_generalise_from_too_few_trades(ctx, clock, llm):
    for index in range(5):
        ctx.store.trades.put(_closed(index, E(10), clock))
    result = reflect_run(ctx, {})
    assert "skipped" in result
    assert llm.calls == []


def test_reflect_stores_lessons_once_there_is_a_sample(ctx, clock, config, llm):
    config.llm.batch_api_for_reflect = False
    for index in range(config.reflect.min_trades_for_lessons):
        ctx.store.trades.put(_closed(index, E(10) if index % 2 else E(-8), clock))

    result = reflect_run(ctx, {})
    assert result["lessons"] >= 1
    stored = ctx.store.lessons.list(limit=10)
    assert stored and stored[0].evidence


def test_reflect_excludes_probe_trades_from_the_statistics(ctx, clock, config, llm):
    config.llm.batch_api_for_reflect = False
    for index in range(config.reflect.min_trades_for_lessons):
        ctx.store.trades.put(_closed(index, E(10), clock))
    for index in range(5):
        ctx.store.trades.put(_closed(100 + index, E(-50), clock, probe=True))

    result = reflect_run(ctx, {})
    assert result["trades"] == config.reflect.min_trades_for_lessons


def test_statistics_are_aggregates_not_narratives(clock):
    trades = [_closed(i, E(10) if i % 2 else E(-5), clock) for i in range(20)]
    stats = build_statistics(trades, 60)
    assert stats["trade_count"] == 20
    assert 0 <= stats["win_rate"] <= 1
    assert "by_regime" in stats and "by_exit_reason" in stats
    # No individual trade identifiers leak into what the model sees.
    assert "trade_id" not in str(stats)


def test_the_consensus_rate_counts_cycles_not_judge_calls(ctx, llm):
    """It read the judge call log, where a call only ever happened *after*
    consensus was reached — so it measured the share of judge calls that did
    not error, which is very nearly always 1.0. With the judge removed it
    would have reported None forever, and said nothing while doing it."""
    from trade_agent.handlers.decide import _consensus_rate
    from trade_agent.orchestrator.cycle import NO_CONSENSUS_PREFIX
    from trade_agent.storage.base import AuditEvent

    assert _consensus_rate(ctx) is None  # nothing has run yet

    now = ctx.clock.now()
    for index, detail in enumerate([
        "traded [buys 1/1, 2 call(s)]",
        f"{NO_CONSENSUS_PREFIX}: 0/1 buy proposals, 1 required [buys 0/1]",
        f"{NO_CONSENSUS_PREFIX}: 0/1 buy proposals, 1 required [buys 0/1]",
        "sizing rejected the plan: stop too wide [buys 1/1, 2 call(s)]",
    ]):
        ctx.store.audit.put(AuditEvent(
            event_id=f"cycle:cyc-{index}", at=now,
            actor="cycle:scheduled",
            action="traded" if index == 0 else "no_trade", detail=detail))

    # Two of four cycles produced a buy proposal: the one that traded, and the
    # one the sizing rejected afterwards.
    assert _consensus_rate(ctx) == 0.5


def test_the_consensus_prefix_is_the_one_the_cycle_writes(ctx, llm):
    """The reader and the writer must not hold separate copies of the string."""
    from trade_agent.orchestrator.cycle import NO_CONSENSUS_PREFIX

    from trade_agent.orchestrator.cycle import DecisionCycle

    llm.bias = "wait"
    outcome = DecisionCycle(ctx, cycle_id="cyc-consensus-prefix").run()
    assert not outcome.traded
    assert outcome.no_trade_reason.startswith(NO_CONSENSUS_PREFIX)
