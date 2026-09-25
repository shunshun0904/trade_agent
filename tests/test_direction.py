from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bbdata.bars import build_bars
from bbresearch.direction import (BAR_MS, BIN_SIGMA, FINE, MIN_MS, PROFILE_COLUMNS, feature_table, flow_features,
                                  profile_features, render, run_direction, stack_features, targets)
from bbresearch.labeling import TradeTape
from bbresearch.pipeline import Market
from bbresearch.profile import value_area
from bbresearch.signals import ewm_sigma

from synth import synth_trades, write_trades

CFG = Path(__file__).resolve().parents[1] / "configs" / "research.yaml"
SPEC = {"name": "btc_jpy", "price_digits": 0, "amount_digits": 4, "unit_amount": "0.0001",
        "status_min_amount": "0.0001", "maker_fee_rate_quote": "0", "taker_fee_rate_quote": "0.001"}
H = 3_600_000


def make_market(root, days=5, seed=0, trades=None):
    trades = synth_trades("2026-01-01", days, seed=seed, per_min=3, vol=0.002) if trades is None else trades
    write_trades(root, "btc_jpy", trades)
    end = (pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    bars = build_bars(root, "btc_jpy", "2026-01-01", end, "15min")
    tape = TradeTape.load(root, "btc_jpy", "2026-01-01", end)
    return Market(bars=bars, tape=tape, spec=SPEC, tick=1.0, min_amount=0.0001), end


def test_flow_features_match_brute_force(tmp_path):
    m, _ = make_market(tmp_path)
    tape = m.tape
    t_start = int(pd.Timestamp("2026-01-03", tz="UTC").value // 1_000_000)
    grid = np.arange(t_start, t_start + 300 * MIN_MS, MIN_MS, dtype="int64")
    f = flow_features(tape, grid)
    rng = np.random.default_rng(0)
    for t in rng.choice(grid, 20, replace=False):
        def last_before(x):
            return tape.price[np.searchsorted(tape.ts, x, side="left") - 1]
        p = last_before(t)
        assert f.at[t, "price"] == p
        for k in (1, 5, 15, 60):
            assert np.isclose(f.at[t, f"m_ret_{k}"], np.log(p / last_before(t - k * MIN_MS)))
        for k in (1, 5, 15):
            s = (tape.ts >= t - k * MIN_MS) & (tape.ts < t)
            vol, buy = tape.amount[s].sum(), tape.amount[s & tape.is_buy].sum()
            assert np.isclose(f.at[t, f"m_logvol_{k}"], np.log1p(vol), rtol=1e-5)
            assert np.isclose(f.at[t, f"m_imb_{k}"], (2 * buy - vol) / vol, atol=1e-6)
            assert f.at[t, f"m_cnt_{k}"] == s.sum()


def test_targets_are_aligned():
    ts = np.array([0, 30_000, 60_000, 16 * MIN_MS, 17 * MIN_MS, 40 * MIN_MS], dtype="int64")
    px = np.array([100.0, 101.0, 102.0, 105.0, 101.0, 90.0])
    tape = TradeTape(ts, np.ones(6, bool), px, np.ones(6))
    y, r = targets(tape, np.array([MIN_MS, 2 * MIN_MS, 30 * MIN_MS], dtype="int64"))
    # t = 1 分: 直前は 101（30 秒）、t + 15 分 = 16 分の直前は 102 → 上げ
    assert np.isclose(r[0], np.log(102 / 101)) and y[0] == 1
    # t = 2 分: 直前は 102、17 分の直前は 105 → 上げ
    assert np.isclose(r[1], np.log(105 / 102)) and y[1] == 1
    # t = 30 分: 直前は 101、45 分は最後の約定より後 → 正解なし
    assert np.isnan(r[2]) and np.isnan(y[2])


def test_profile_features_match_direct_recount(tmp_path):
    m, _ = make_market(tmp_path)
    tape, bars = m.tape, m.bars
    t_start = int(pd.Timestamp("2026-01-03T05:00", tz="UTC").value // 1_000_000)
    grid = np.arange(t_start, t_start + 120 * MIN_MS, 7 * MIN_MS, dtype="int64")
    price = flow_features(tape, grid)["price"].to_numpy()
    pf = profile_features(tape, bars, grid, price, 96, workers=1)
    pf2 = profile_features(tape, bars, grid, price, 96, workers=2)
    pd.testing.assert_frame_equal(pf, pf2)
    sig = ewm_sigma(bars, 96)
    bar_end = np.array([t.value // 1_000_000 for t in bars.index]) + BAR_MS
    for i, t in enumerate(grid):
        j = np.searchsorted(bar_end, t, side="right") - 1
        s = sig.iloc[j]
        w = BIN_SIGMA * s
        lp = np.log(price[i])
        sel = (tape.ts >= t - 24 * H) & (tape.ts < t)
        fine = np.floor(np.log(tape.price[sel]) / FINE)
        centers = (fine + 0.5) * FINE
        k = np.floor((centers - lp) / w).astype(int)
        kmin = k.min()
        counts = np.bincount(k - kmin, weights=np.rint(tape.amount[sel] * 1e4))
        poc, lo, hi = value_area(counts)
        assert np.isclose(pf.iloc[i]["vp_poc_dist"], (poc + kmin + 0.5) * BIN_SIGMA)
        assert np.isclose(pf.iloc[i]["vp_val_dist"], (lo + kmin) * BIN_SIGMA)
        assert np.isclose(pf.iloc[i]["vp_vah_dist"], (hi + kmin + 1) * BIN_SIGMA)
        rel = np.arange(len(counts)) + kmin
        assert np.isclose(pf.iloc[i]["vp_up_1s"], counts[(rel >= 0) & (rel < 4)].sum() / counts.sum(), atol=1e-6)
        assert pf.iloc[i]["vp_val_dist"] <= pf.iloc[i]["vp_poc_dist"] <= pf.iloc[i]["vp_vah_dist"]
        assert pf.iloc[i]["tpo_val_dist"] <= pf.iloc[i]["tpo_vah_dist"]


def test_features_do_not_look_ahead(tmp_path):
    trades = synth_trades("2026-01-01", 5, seed=1, per_min=3, vol=0.002)
    t_cut = pd.Timestamp("2026-01-04T06:07", tz="UTC")
    cut_ms = t_cut.value // 1_000_000
    m1, end = make_market(tmp_path / "a", trades=trades)
    # t_cut 以降の約定の価格・数量・売買を大きく変える
    late = trades["executed_at"] >= cut_ms
    t2 = trades.copy()
    t2.loc[late, "price"] = np.round(t2.loc[late, "price"] * 1.3)
    t2.loc[late, "amount"] = t2.loc[late, "amount"] * 50
    t2.loc[late, "side"] = "buy"
    m2, _ = make_market(tmp_path / "b", trades=t2)
    f1 = feature_table(m1, "2026-01-01", end, 96, 1)
    f2 = feature_table(m2, "2026-01-01", end, 96, 1)
    x = [c for c in f1.columns if c not in ("y", "r")]
    upto = f1.index <= cut_ms
    assert upto.sum() > 1000
    pd.testing.assert_frame_equal(f1.loc[upto, x], f2.loc[upto, x])
    # 変えた後の特徴量は実際に変わる（テストが空振りしていないこと）
    after = f1.index > cut_ms + 20 * MIN_MS
    assert not np.allclose(f1.loc[after, "m_logvol_15"], f2.loc[after, "m_logvol_15"])
    # 正解は t + 15 分の価格を使うので、t_cut − 15 分より後は変わりうる
    assert f1.loc[f1.index + BAR_MS < cut_ms, "r"].equals(f2.loc[f2.index + BAR_MS < cut_ms, "r"])


def test_feature_table_columns(tmp_path):
    m, end = make_market(tmp_path)
    f = feature_table(m, "2026-01-01", end, 96, 1)
    assert f.index[0] == pd.Timestamp("2026-01-03T01:00", tz="UTC").value // 1_000_000
    assert (np.diff(f.index) == MIN_MS).all()
    assert set(PROFILE_COLUMNS) <= set(f.columns)
    assert any(c.startswith("b_") for c in f.columns)
    assert f[PROFILE_COLUMNS].notna().mean().min() > 0.9


def test_stack_features_use_only_past_15_minutes():
    t = np.arange(0, 60 * MIN_MS, MIN_MS, dtype="int64")
    pred = pd.Series(np.arange(60, dtype=float), index=t)
    s = stack_features(pred, np.array([30 * MIN_MS], dtype="int64"))
    assert s["B_last"].iloc[0] == 30 and s["B_mean"].iloc[0] == np.mean(np.arange(16, 31))
    assert s["B_slope"].iloc[0] == 14


def test_run_direction_end_to_end(tmp_path):
    write_trades(tmp_path / "data", "btc_jpy", synth_trades("2026-01-01", 9, seed=5, per_min=3, vol=0.002))
    cfg = yaml.safe_load(CFG.read_text())
    cfg["data"].update(root=str(tmp_path / "data"), start="2026-01-01", end="2026-01-09")
    cfg["experiment_log"] = str(tmp_path / "experiments.jsonl")
    cfg["pair_spec"] = SPEC
    rep = run_direction(cfg, {"train_end": "2026-01-07", "n_splits": 3, "thetas": [0.55], "n_boot": 20}, workers=2)
    assert rep["n"]["A_test"] > 100 and rep["n"]["B_test"] > 1000
    assert "auc" in rep["test"]["A"] and "auc" in rep["test"]["A+B"]
    assert len((tmp_path / "experiments.jsonl").read_text().splitlines()) == 3
    assert "二段構え" in render(rep)
