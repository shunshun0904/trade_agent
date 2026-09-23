import numpy as np
import pandas as pd
import pytest

from bbresearch.labeling import (
    BarrierParams, TradeTape, barriers, ceil_price, entry_price, fill_time, first_exit, floor_price,
    label_event, label_events, net_return,
)
from bbresearch.signals import ewm_sigma

M = 60_000  # 1分（ミリ秒）


def tape(rows):
    """rows: (ts_ms, side, price)"""
    ts, side, px = zip(*rows)
    return TradeTape(np.array(ts, dtype="int64"), np.array([s == "buy" for s in side]), np.array(px, dtype=float))


PRM = BarrierParams(delta=0.0, t_fill_min=15, k_up=1.0, k_dn=1.0, n_v=2, bar_minutes=15,
                    s_slip=0.001, fill_rule="strict", maker_fee=-0.0002, taker_fee=0.0012)
SIGMA = 0.01  # U = 101, L = 99（P_e = 100）


def test_rounding():
    assert floor_price(100.9, 1) == 100
    assert ceil_price(100.1, 1) == 101
    assert ceil_price(101.0, 1) == 101
    assert entry_price(100.7, 0.0, 1) == 100
    assert barriers(100, 0.01, 1, 1, 1) == (101, 99.0)


def test_fill_strict_vs_touch():
    t = tape([(1 * M, "sell", 100), (2 * M, "buy", 99), (3 * M, "sell", 99.5)])
    assert fill_time(t, 100, 0, 15 * M, "touch") == (1 * M, 0)
    # strict: 同値の sell と buy の約定は無視し、100 未満の sell で約定
    assert fill_time(t, 100, 0, 15 * M, "strict") == (3 * M, 2)
    assert fill_time(t, 99, 0, 15 * M, "strict") is None
    # 待機時間外の約定は数えない
    assert fill_time(t, 100, 0, 3 * M, "strict") is None


def test_take_profit():
    t = tape([(1 * M, "sell", 99.9), (5 * M, "buy", 101), (6 * M, "sell", 101.5), (7 * M, "buy", 101.5), (60 * M, "buy", 100)])
    out = label_event(t, 0, 100.0, SIGMA, PRM, 1)
    # 101 ちょうどの buy は U を超えていない。101.5 の sell も side が違う。
    assert out["exit_type"] == "tp" and out["t_x"] == 7 * M and out["P_x"] == 101
    assert out["f_x"] == PRM.maker_fee
    assert out["y"] == 1
    assert out["ret_net"] == pytest.approx(net_return(100, 101, -0.0002, -0.0002))


def test_stop_loss_any_side():
    t = tape([(1 * M, "sell", 99.9), (5 * M, "buy", 99.0), (60 * M, "buy", 100)])
    out = label_event(t, 0, 100.0, SIGMA, PRM, 1)
    assert out["exit_type"] == "sl" and out["t_x"] == 5 * M
    assert out["P_x"] == pytest.approx(99.0 * (1 - 0.001))
    assert out["f_x"] == PRM.taker_fee and out["y"] == 0


def test_time_exit_uses_last_price_before_tv():
    t_f = 1 * M
    t_v = t_f + 2 * 15 * M
    t = tape([(t_f, "sell", 99.9), (10 * M, "buy", 100.5), (t_v - 1, "sell", 100.2), (t_v, "buy", 102), (t_v + M, "buy", 90)])
    out = label_event(t, 0, 100.0, SIGMA, PRM, 1)
    assert out["exit_type"] == "time" and out["t_x"] == t_v
    assert out["P_x"] == pytest.approx(100.2 * (1 - 0.001))


def test_unfilled_and_incomplete():
    t = tape([(1 * M, "buy", 99), (2 * M, "sell", 100)])
    assert label_event(t, 0, 100.0, SIGMA, PRM, 1) == {"P_e": 100, "filled": False}
    # 約定したが t_v までのデータがない → 決済は未判定
    t2 = tape([(1 * M, "sell", 99.9), (2 * M, "buy", 100)])
    out = label_event(t2, 0, 100.0, SIGMA, PRM, 1)
    assert out["filled"] and "exit_type" not in out


def test_exit_ignores_trades_at_fill_time():
    # 約定と同時刻の約定は「t_f より後」ではない
    t = tape([(1 * M, "sell", 99.9), (1 * M, "sell", 98.0), (3 * M, "buy", 101.5), (60 * M, "buy", 100)])
    assert first_exit(t, 0, 1 * M, 101, 99, 31 * M, 0.001)[0] == "tp"


def test_label_events_uses_sigma_at_t0_only():
    idx = pd.date_range("2026-01-01", periods=200, freq="15min", tz="UTC")
    rng = np.random.default_rng(0)
    bars = pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 200))), "is_empty": False}, index=idx)
    sigma = ewm_sigma(bars, 20)
    t0 = idx[100] + pd.Timedelta("15min")
    ev = pd.DataFrame({"event_id": ["e1"], "t0": [t0]})
    t = tape([(t0.value // 10**6 + M, "sell", 1.0), (t0.value // 10**6 + 10**9, "buy", 1.0)])
    lab1 = label_events(ev, bars, t, PRM, 0.001, sigma)
    # 未来の足を大きく変えても、t0 のラベルに使う close と σ は変わらない
    bars2 = bars.copy()
    bars2.iloc[101:, 0] *= 3
    lab2 = label_events(ev, bars2, t, PRM, 0.001, ewm_sigma(bars2, 20))
    pd.testing.assert_frame_equal(lab1, lab2)


def test_tape_load_in_chunks_matches_single_load(tmp_path):
    from bbdata.download import load_transactions
    from synth import synth_trades, write_trades

    write_trades(tmp_path, "btc_jpy", synth_trades("2026-01-01", 5, seed=2, per_min=1))
    a = TradeTape.load(tmp_path, "btc_jpy", "2026-01-01", "2026-01-06", chunk_days=2)
    b = TradeTape.from_frame(load_transactions(tmp_path, "btc_jpy", "2026-01-01", "2026-01-06"))
    assert np.array_equal(a.ts, b.ts) and np.array_equal(a.is_buy, b.is_buy) and np.array_equal(a.price, b.price)
