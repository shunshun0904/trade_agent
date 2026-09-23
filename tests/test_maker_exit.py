import numpy as np
import pytest

from bbresearch.labeling import TradeTape
from bbresearch.maker_exit import MakerExitParams, label_event_maker

M = 60_000


def tape(rows):
    ts, side, px = zip(*rows)
    return TradeTape(np.array(ts, dtype="int64"), np.array([s == "buy" for s in side]), np.array(px, float))


# close 100、σ 0.01 → 2σ 下の指値 98。U = 98 × 1.01 = 98.98 → 呼値 0.01 で 98.98、L = 98 × 0.96 = 94.08
PRM = MakerExitParams(delta_sigma=2.0, t_fill_min=15, k_up=1.0, k_dn=4.0, n_v=4, grace_min=15, s_slip=0.001,
                      maker_fee=0.0, taker_fee=0.001)
T_V = 2 * M + 4 * 15 * M  # 約定 2 分 → t_v = 62 分


def run(rows):
    return label_event_maker(tape(rows), 0, 100.0, 0.01, PRM, 0.01)


def test_unfilled_when_price_stays_above_deep_limit():
    assert run([(1 * M, "sell", 98.5), (100 * M, "buy", 99)]) == {"P_e": 98.0, "filled": False}


def test_take_profit_is_maker():
    out = run([(2 * M, "sell", 97.9), (10 * M, "buy", 99.0), (200 * M, "buy", 99)])
    assert out["exit_type"] == "tp" and out["P_x"] == pytest.approx(98.98) and out["f_x"] == 0.0


def test_stop_is_market():
    out = run([(2 * M, "sell", 97.9), (10 * M, "sell", 94.0), (200 * M, "buy", 99)])
    assert out["exit_type"] == "sl" and out["P_x"] == pytest.approx(94.0 * 0.999) and out["f_x"] == 0.001


def test_time_exit_limit_filled_by_buyer_above_limit():
    rows = [(2 * M, "sell", 97.9), (T_V - 1, "sell", 98.2),       # t_v 直前 98.2 → 指値 98.21
            (T_V + 1 * M, "buy", 98.21),                           # 同値は約定とみなさない
            (T_V + 3 * M, "buy", 98.25), (200 * M, "buy", 99)]
    out = run(rows)
    assert out["exit_type"] == "time_maker" and out["P_x"] == pytest.approx(98.21)
    assert out["t_x"] == T_V + 3 * M and out["f_x"] == 0.0


def test_time_exit_falls_back_to_market_after_grace():
    rows = [(2 * M, "sell", 97.9), (T_V - 1, "sell", 98.2), (T_V + 5 * M, "sell", 98.1),
            (T_V + 20 * M, "buy", 99.0), (200 * M, "buy", 99)]
    out = run(rows)
    assert out["exit_type"] == "time_taker" and out["t_x"] == T_V + 15 * M
    assert out["P_x"] == pytest.approx(98.1 * 0.999) and out["f_x"] == 0.001


def test_stop_during_grace_window():
    rows = [(2 * M, "sell", 97.9), (T_V - 1, "sell", 98.2), (T_V + 2 * M, "sell", 94.0), (200 * M, "buy", 99)]
    assert run(rows)["exit_type"] == "sl"
