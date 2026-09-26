from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bbdata.bars import build_bars
from bbresearch.direction import (BAR_MS, BIN_SIGMA, FINE, MIN_MS, PROFILE_COLUMNS, feature_table, flow_features,
                                  level_summary, profile_features, render, run_direction, stack_features, taker_round_trip,
                                  targets)
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
    # 費用を超える値幅: 0.5% を超えて上がったときだけ 1。t = 1 分は +0.99%、t = 2 分は +2.9%
    y2, _ = targets(tape, np.array([MIN_MS, 2 * MIN_MS, 30 * MIN_MS], dtype="int64"), min_ret=0.02)
    assert y2[0] == 0 and y2[1] == 1 and np.isnan(y2[2])


def test_taker_round_trip_is_break_even():
    c = taker_round_trip(0.001, 0.0005)
    g = np.exp(c)
    assert abs(g * (1 - 0.0005) * (1 - 0.001) / ((1 + 0.0005) * (1 + 0.001)) - 1) < 1e-12
    assert 0.0029 < c < 0.0031


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
    assert s["B_slope"].iloc[0] == 14 and s["B_mean_recent"].iloc[0] == np.mean(np.arange(28, 31))
    # 60 分の窓: t−59 … t。窓の外（負の時刻）は NaN として無視する
    s60 = stack_features(pred, np.array([59 * MIN_MS, 30 * MIN_MS], dtype="int64"), 60)
    assert s60["B_mean"].iloc[0] == np.mean(np.arange(0, 60)) and s60["B_slope"].iloc[0] == 59
    assert s60["B_mean"].iloc[1] == np.mean(np.arange(0, 31)) and np.isnan(s60["B_slope"].iloc[1])


def test_level_summary_uses_only_past_window():
    t = np.arange(0, 120 * MIN_MS, MIN_MS, dtype="int64")
    lv = pd.DataFrame({"vp_poc_dist": np.arange(120, dtype=float), "tpo_va_pos": np.sin(np.arange(120))}, index=t)
    out = level_summary(lv, np.array([100 * MIN_MS], dtype="int64"), 60)
    assert out["L_vp_poc_dist_mean"].iloc[0] == np.mean(np.arange(41, 101))
    assert out["L_vp_poc_dist_chg"].iloc[0] == 100 - 41
    assert np.isclose(out["L_tpo_va_pos_mean"].iloc[0], np.sin(np.arange(41, 101)).mean())
    # t より後の値を変えても結果は変わらない
    lv2 = lv.copy()
    lv2.loc[lv2.index > 100 * MIN_MS] = 999.0
    pd.testing.assert_frame_equal(out, level_summary(lv2, np.array([100 * MIN_MS], dtype="int64"), 60))


def test_run_direction_end_to_end(tmp_path):
    write_trades(tmp_path / "data", "btc_jpy", synth_trades("2026-01-01", 9, seed=5, per_min=3, vol=0.002))
    cfg = yaml.safe_load(CFG.read_text())
    cfg["data"].update(root=str(tmp_path / "data"), start="2026-01-01", end="2026-01-09")
    cfg["experiment_log"] = str(tmp_path / "experiments.jsonl")
    cfg["pair_spec"] = SPEC
    rep = run_direction(cfg, {"train_end": "2026-01-07", "n_splits": 3, "thetas": [0.55], "n_boot": 20}, workers=2)
    assert rep["n"]["A_test"] > 100 and rep["n"]["B_test"] > 1000
    assert "auc" in rep["test"]["A"] and "auc" in rep["test"]["A+B"]
    assert rep["horizon_min"] == 15 and "A+B+L" not in rep["test"]
    assert len((tmp_path / "experiments.jsonl").read_text().splitlines()) == 3
    assert "二段構え" in render(rep)


def test_run_direction_one_hour_with_level_summary(tmp_path):
    write_trades(tmp_path / "data", "btc_jpy", synth_trades("2026-01-01", 9, seed=6, per_min=3, vol=0.002))
    cfg = yaml.safe_load(CFG.read_text())
    cfg["data"].update(root=str(tmp_path / "data"), start="2026-01-01", end="2026-01-09")
    cfg["experiment_log"] = str(tmp_path / "experiments.jsonl")
    cfg["pair_spec"] = SPEC
    spec = {"train_end": "2026-01-07", "n_splits": 3, "thetas": [0.55], "n_boot": 20, "horizon_min": 60,
            "level_summary": True}
    rep = run_direction(cfg, spec, workers=2)
    # 1 時間ごとの判断: 評価 2 日で 48 回弱
    assert 40 <= rep["n"]["A_test"] <= 48 and rep["horizon_min"] == 60
    assert set(rep["test"]) >= {"A", "A+B", "A+B+L", "B_on_bar"}
    fa, fab, fabl = (rep["variant_features"][k] for k in ("A", "A+B", "A+B+L"))
    assert len(fab) == len(fa) + 5 and len(fabl) == len(fab) + 2 * len(PROFILE_COLUMNS)
    assert set(rep["auc_diff_vs_A"]) == {"A+B", "A+B+L"}
    assert len((tmp_path / "experiments.jsonl").read_text().splitlines()) == 4
    md = render(rep)
    assert "60 分後" in md and "A+B+L" in md


def test_run_direction_cost_target(tmp_path):
    write_trades(tmp_path / "data", "btc_jpy", synth_trades("2026-01-01", 9, seed=7, per_min=3, vol=0.002))
    cfg = yaml.safe_load(CFG.read_text())
    cfg["data"].update(root=str(tmp_path / "data"), start="2026-01-01", end="2026-01-09")
    cfg["experiment_log"] = str(tmp_path / "experiments.jsonl")
    cfg["pair_spec"] = SPEC
    spec = {"train_end": "2026-01-07", "n_splits": 3, "thetas": [0.5], "top_fracs": [0.1], "n_boot": 10,
            "horizon_min": 60, "target_min_return": "taker_round_trip"}
    rep = run_direction(cfg, spec, workers=2)
    assert abs(rep["target_min_return"] - taker_round_trip(0.001, 0.0005)) < 1e-12
    assert 0 < rep["base_rate_test"] < 0.5   # 0.3% を超える上げは半分より少ない
    rows = rep["pnl_test"]["A"]
    assert rows[1]["top_frac"] == 0.1 and abs(rows[1]["coverage"] - 0.1) < 0.03
    md = render(rep)
    assert "0.30% を超えて上がるか" in md and "上位 10%" in md
    assert "two_sided" not in rep


def test_run_direction_two_sided(tmp_path):
    write_trades(tmp_path / "data", "btc_jpy", synth_trades("2026-01-01", 9, seed=8, per_min=3, vol=0.002))
    cfg = yaml.safe_load(CFG.read_text())
    cfg["data"].update(root=str(tmp_path / "data"), start="2026-01-01", end="2026-01-09")
    cfg["experiment_log"] = str(tmp_path / "experiments.jsonl")
    cfg["pair_spec"] = SPEC
    spec = {"train_end": "2026-01-07", "n_splits": 3, "thetas": [0.5], "top_fracs": [0.2], "diff_thetas": [0.0],
            "n_boot": 10, "horizon_min": 60, "target_min_return": "taker_round_trip", "two_sided": True,
            "level_summary": True}
    rep = run_direction(cfg, spec, workers=2)
    ts = rep["two_sided"]
    assert 0 < ts["base_rate_down_test"] < 0.5
    assert set(ts["test_down"]) == {"B_all_minutes", "A", "A+B", "A+B+L", "B_on_bar"}
    assert set(ts["pnl_diff"]) == {"A", "A+B", "A+B+L", "B_on_bar"}
    rows = ts["pnl_diff"]["A"]
    assert rows[0]["theta"] == 0.0 and rows[1]["top_frac"] == 0.2 and abs(rows[1]["coverage"] - 0.2) < 0.03
    assert -1 <= ts["corr_up_down_test"]["A"] <= 1 and "0.2" in ts["bottom_mean_r"]["A"]
    md = render(rep)
    assert "下げ側" in md and "上げ確率 − 下げ確率" in md and md.rstrip().endswith("0.0005")
    assert "two_sided/A" in (tmp_path / "experiments.jsonl").read_text()
