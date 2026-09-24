from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bbresearch.labeling import TradeTape
from bbresearch.val_rule import RollingProfile, ValRuleParams, render, run_val_rule, simulate

from synth import synth_trades, write_trades

CFG = Path(__file__).resolve().parents[1] / "configs" / "research.yaml"
H = 3_600_000


def rand_tape(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.integers(0, 72 * H, n)).astype("int64")
    px = np.round(100 * np.exp(np.cumsum(rng.normal(0, 0.001, n))), 2)
    return TradeTape(ts, rng.random(n) < 0.5, px, np.round(rng.uniform(0.001, 1, n), 4))


def test_rolling_histogram_equals_recount():
    tape = rand_tape()
    rp = RollingProfile(tape, 1e-4)
    for t in range(25 * H, 72 * H, 7 * 60_000):  # 7 分ずつ進める
        rp.advance(t, 24 * H)
    m = (tape.ts >= t - 24 * H) & (tape.ts < t)
    direct = np.zeros_like(rp.hist)
    np.add.at(direct, rp.fb[m] - rp.off, rp.qty[m])
    assert np.array_equal(rp.hist, direct)
    assert rp.last_price() == tape.price[np.flatnonzero(tape.ts < t)[-1]]


def test_levels_ignore_trades_at_or_after_t():
    tape = rand_tape(seed=1)
    t = 40 * H
    a = RollingProfile(tape, 1e-4)
    a.advance(t, 24 * H)
    # t 以降の約定の価格と数量を大きく変えても、t の水準は変わらない
    late = tape.ts >= t
    px2 = tape.price.copy()
    px2[late] *= 3
    amt2 = tape.amount.copy()
    amt2[late] *= 100
    b = RollingProfile(TradeTape(tape.ts, tape.is_buy, px2, amt2), 1e-4)
    b.advance(t, 24 * H)
    prm = ValRuleParams()
    p = a.last_price()
    assert a.levels(p, 0.004, prm) == b.levels(p, 0.004, prm)


def test_levels_are_ordered_and_price_based():
    tape = rand_tape(seed=2)
    rp = RollingProfile(tape, 1e-4)
    rp.advance(48 * H, 24 * H)
    lv = rp.levels(rp.last_price(), 0.004, ValRuleParams())
    assert lv["val"] < lv["poc"] < lv["vah"]


def test_simulate_orders_and_exits(tmp_path):
    write_trades(tmp_path, "btc_jpy", synth_trades("2026-01-01", 8, seed=3, per_min=4, vol=0.002))
    from bbdata.bars import build_bars

    bars = build_bars(tmp_path, "btc_jpy", "2026-01-01", "2026-01-08", "15min")
    tape = TradeTape.load(tmp_path, "btc_jpy", "2026-01-01", "2026-01-08")
    prm = ValRuleParams(min_reward=0.001)
    orders, trades = simulate(tape, bars, "2026-01-01", "2026-01-08", prm, (0.0, 0.001), 0.0005, 1.0)
    assert len(orders) > 0
    # 最初の 24 時間は準備期間
    assert orders["t"].min() >= pd.Timestamp("2026-01-02", tz="UTC")
    # 指値は現在値より下、利確は POC 以上、損切りは VAL − 0.5 × (POC − VAL)
    assert (orders["P_e"] < orders["price"]).all() and (orders["U"] >= orders["P_e"] * 1.001 - 1).all()
    assert np.allclose(orders["L"], orders["P_e"] - 0.5 * (orders["U"] - orders["P_e"]))
    if len(trades):
        assert set(trades["exit_type"]) <= {"tp", "sl", "time_maker", "time_taker"}
        # ポジションは同時に 1 つ
        assert (trades["t_order"].iloc[1:].to_numpy() >= trades["t_x"].iloc[:-1].to_numpy()).all()


def test_run_val_rule_end_to_end(tmp_path):
    write_trades(tmp_path / "data", "btc_jpy", synth_trades("2026-01-01", 10, seed=5, per_min=3, vol=0.002))
    cfg = yaml.safe_load(CFG.read_text())
    cfg["data"].update(root=str(tmp_path / "data"), start="2026-01-01", end="2026-01-10")
    cfg["experiment_log"] = str(tmp_path / "experiments.jsonl")
    cfg["pair_spec"] = {"name": "btc_jpy", "price_digits": 0, "amount_digits": 4, "unit_amount": "0.0001",
                        "status_min_amount": "0.0001", "maker_fee_rate_quote": "0", "taker_fee_rate_quote": "0.001"}
    rep = run_val_rule(cfg, {"params": {"min_reward": 0.001}, "n_control": 2})
    assert rep["n_orders"] > 0 and len(rep["control"]) == 2
    assert (tmp_path / "experiments.jsonl").exists()
    assert "年ごと" in render(rep)
