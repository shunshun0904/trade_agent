import numpy as np
import pandas as pd
import pytest

from bbresearch import backtest as bt

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


def ev_lab():
    events = pd.DataFrame({
        "event_id": ["a", "b", "c", "d"],
        "t0": [T("2026-01-01 00:00"), T("2026-01-01 00:30"), T("2026-01-01 03:00"), T("2026-01-01 05:00")],
        "signal_type": "dip",
    })
    labels = pd.DataFrame({
        "event_id": ["a", "b", "c", "d"],
        "filled": [True, True, False, True],
        "t_f": [T("2026-01-01 00:05"), T("2026-01-01 00:35"), pd.NaT, T("2026-01-01 05:05")],
        "t_x": [T("2026-01-01 01:00"), T("2026-01-01 01:30"), pd.NaT, T("2026-01-01 06:00")],
        "P_e": [100.0, 100.0, 100.0, 100.0],
        "P_x": [102.0, 99.0, np.nan, 98.0],
        "exit_type": ["tp", "sl", None, "sl"],
        "f_e": -0.0002, "f_x": [-0.0002, 0.0012, np.nan, 0.0012],
        "ret_net": [0.0204, -0.0112, np.nan, -0.0212],
    })
    return events, labels


ACC = bt.Account(initial_capital=10_000, max_fraction=1.0, amount_digits=4, min_amount=0.0001)


def test_one_position_at_a_time_and_unfilled_orders():
    ev, lab = ev_lab()
    tr = bt.run_backtest(ev, lab, None, bt.Strategy("p", "all"), ACC, pd.Timedelta("15min"))
    # b は a の保有中なので無視。c は未約定の注文として残る
    assert list(tr["event_id"]) == ["a", "c", "d"]
    assert list(tr["filled"]) == [True, False, True]
    a = tr.iloc[0]
    assert a["amount"] == 100.0
    assert a["pnl"] == pytest.approx(100 * 2 - 100 * 100 * -0.0002 - 100 * 102 * -0.0002)
    s = bt.summarize(tr, ACC, T("2026-01-01"), T("2026-01-02"))
    assert s["n_orders"] == 3 and s["n_trades"] == 2 and s["fill_rate"] == pytest.approx(2 / 3)
    assert set(s["by_exit_type"]) == {"tp", "sl"}


def test_threshold_and_meta_sizing():
    ev, lab = ev_lab()
    p = pd.Series({"a": 0.4, "b": 0.7, "c": 0.9, "d": 0.6})
    tr = bt.run_backtest(ev, lab, p, bt.Strategy("t", "threshold", theta=0.55), ACC, pd.Timedelta("15min"))
    assert list(tr["event_id"]) == ["b", "c", "d"]
    tr = bt.run_backtest(ev, lab, p, bt.Strategy("m", "meta"), ACC, pd.Timedelta("15min"))
    assert "a" not in set(tr["event_id"])  # p < 0.5 → 発注量 0
    assert tr.set_index("event_id").loc["b", "frac"] == pytest.approx(float(bt.bet_size(0.7)))


def test_min_amount():
    ev, lab = ev_lab()
    acc = bt.Account(initial_capital=0.005, max_fraction=1.0, amount_digits=4, min_amount=0.0001)
    assert bt.run_backtest(ev, lab, None, bt.Strategy("p", "all"), acc, pd.Timedelta("15min")).empty


def test_bet_size_properties():
    p = np.array([0.3, 0.5, 0.6, 0.9, 0.999])
    m = bt.bet_size(p)
    assert m[0] == 0 and m[1] == 0 and 0 < m[2] < m[3] < m[4] <= 1
    assert bt.bet_size(0.6, step=0.25) == 0.25 or bt.bet_size(0.6, step=0.25) == 0.0


def test_deflated_sharpe():
    assert bt.deflated_sharpe(0.1, [0.1], 100, 0, 3) is None
    hi = bt.deflated_sharpe(0.2, [0.0, 0.05, 0.2], 365, 0, 3)
    lo = bt.deflated_sharpe(0.2, [0.0, 0.05, 0.2] * 30, 365, 0, 3)
    assert 0 <= lo < hi <= 1


def test_equity_curve_includes_trade_closing_after_period_end():
    trades = pd.DataFrame({
        "filled": [True], "t_x": [T("2026-01-03 01:00")], "pnl": [500.0],
    })
    curve = bt.equity_curve(trades, 10_000, T("2026-01-01"), T("2026-01-02"))
    assert curve.iloc[-1] == 10_500
    s = bt.summarize(trades.assign(t_f=T("2026-01-01 23:00"), exit_type="tp", ret_net=0.05,
                                   entry_fee=0.0, exit_fee=0.0),
                     ACC, T("2026-01-01"), T("2026-01-02"))
    assert s["total_return"] == pytest.approx(0.05)
