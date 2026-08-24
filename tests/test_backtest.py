"""The offline replay harness (spec 13, Phase 0)."""

import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from trade_agent.backtest import BaselinePlanner, run_backtest
from trade_agent.models.market import Candle

E = Decimal


def _series(count=400, seed=11, start=E(15000000)):
    random.seed(seed)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    price = start
    out = []
    for index in range(count):
        close = price + E(random.randint(-50000, 50000))
        out.append(Candle(
            open=price, high=max(price, close) + E(random.randint(0, 20000)),
            low=min(price, close) - E(random.randint(0, 20000)),
            close=close, volume=E(random.randint(1, 12)),
            opened_at=base + timedelta(hours=index)))
        price = close
    return out


def test_the_harness_runs_and_reports(config):
    result = run_backtest(_series(), config)
    summary = result.summary()
    assert summary["candles"] == 400
    assert summary["trades"] >= 0
    assert summary["equity_end_jpy"] == summary["equity_start_jpy"] + \
        summary["net_pnl_jpy"]


def test_fees_are_charged_on_every_trade(config):
    result = run_backtest(_series(seed=3), config)
    for trade in result.trades:
        assert trade.fees != 0
        assert trade.net_pnl == (trade.exit - trade.entry) * trade.qty - trade.fees


def test_the_maker_rebate_is_modelled_as_a_credit(config):
    """bitbank pays the maker rather than charging them.

    A maker-in / maker-out round trip therefore carries a negative fee, and a
    target only a yen above the entry still nets positive. This is genuinely
    how the exchange works, and the harness has to say so — but it is exactly
    why the guard rejects a target that thin: it survives only while the
    rebate does, and the rebate is a fee schedule, not a strategy.
    """
    class Thin:
        def __call__(self, candles, cfg):
            last = candles[-1].close
            return {"entry": last,
                    "stop_loss": last * E("0.98"),
                    "take_profit": last + E(1)}

    assert config.exchange.maker_fee_rate < 0
    result = run_backtest(_series(seed=5), config, planner=Thin())
    makers = [t for t in result.trades if t.reason == "take_profit"]
    assert makers, "expected at least one maker exit in this series"
    assert all(t.fees < 0 for t in makers)


def test_a_taker_stop_costs_more_than_a_maker_exit(config):
    """The asymmetry the executor is built around: stops pay, targets earn."""
    result = run_backtest(_series(seed=5), config)
    stops = [t for t in result.trades if t.reason == "stop_loss"]
    targets = [t for t in result.trades if t.reason == "take_profit"]
    if stops and targets:
        assert min(t.fees for t in stops) > max(t.fees for t in targets)


def test_stops_win_a_tie_inside_one_bar(config):
    """A bar spanning both levels books as a stop — never as a win."""
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = [Candle(open=E(15000000), high=E(15000000), low=E(15000000),
                      close=E(15000000), volume=E(1),
                      opened_at=base + timedelta(hours=i)) for i in range(70)]
    # The fill bar dips below the limit, then a bar spans both stop and target.
    candles.append(Candle(open=E(15000000), high=E(15000000), low=E(14000000),
                          close=E(15000000), volume=E(1),
                          opened_at=base + timedelta(hours=70)))
    candles.append(Candle(open=E(15000000), high=E(16000000), low=E(13000000),
                          close=E(15000000), volume=E(1),
                          opened_at=base + timedelta(hours=71)))

    class Fixed:
        def __call__(self, series, cfg):
            if len(series) != 70:
                return None
            return {"entry": E(15000000), "stop_loss": E(14500000),
                    "take_profit": E(15500000)}

    result = run_backtest(candles, config, planner=Fixed(), warmup=60)
    assert result.trades
    assert result.trades[0].reason == "stop_loss"


def test_an_unfilled_entry_expires(config):
    """A PostOnly entry the market never trades through is cancelled, not
    counted as a fill."""
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = [Candle(open=E(15000000), high=E(15000000), low=E(15000000),
                      close=E(15000000), volume=E(1),
                      opened_at=base + timedelta(hours=i)) for i in range(120)]

    class Fixed:
        def __call__(self, series, cfg):
            return {"entry": E(14000000), "stop_loss": E(13900000),
                    "take_profit": E(14300000)}

    result = run_backtest(candles, config, planner=Fixed(), warmup=60)
    assert result.triggers > 0
    assert result.trades == []


def test_the_baseline_planner_declines_without_history(config):
    assert BaselinePlanner()([], config) is None
