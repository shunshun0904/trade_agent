"""bbresearch/distbase.py: ボラだけの基準（HAR 型・GARCH-t 型）、分布の評価、先読みがないこと。"""
import math

import numpy as np
import pandas as pd
import pytest

from bbresearch import distbase as db
from bbresearch import nullsim

QS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def _cdf_crps(V, W, y, weight=None, lo=-1.0, hi=1.0):
    """CRPS = ∫ (F(z) − 1{y ≤ z})² w(z) dz を区間ごとに正確に積分する（被積分関数は区分的に一定。検算用）。"""
    b = np.unique(np.r_[lo, hi, V, y, 0.03, -0.03])
    mid = (b[:-1] + b[1:]) / 2
    F = np.array([W[V <= z].sum() for z in mid])
    g = (F - (y <= mid)) ** 2
    if weight is not None:
        g = g * weight(mid)
    return float(np.sum(g * np.diff(b)))


def test_crps_and_threshold_weighted_crps_match_numerical_integral():
    rng = np.random.default_rng(0)
    V = np.sort(rng.normal(0, 0.1, 40))
    W = rng.random(40)
    W /= W.sum()
    for y in (-0.15, 0.0, 0.07):
        m = db.dist_metrics(V[None, :], W[None, :], np.array([y]), QS, (0.0, 0.03), tw=0.03)
        assert math.isclose(m["crps"][0], _cdf_crps(V, W, y), abs_tol=1e-12)
        assert math.isclose(m["tw_up"][0], _cdf_crps(V, W, y, lambda z: (z > 0.03).astype(float)), abs_tol=1e-12)
        assert math.isclose(m["tw_dn"][0], _cdf_crps(V, W, y, lambda z: (z < -0.03).astype(float)), abs_tol=1e-12)
        assert math.isclose(m["p"][0, 0], W[V > 0].sum()) and math.isclose(m["mean"][0], (W * V).sum())
        assert math.isclose(m["pit"][0], W[V < y].sum())
        for j, q in enumerate(QS):
            assert m["q"][0, j] == V[np.searchsorted(np.cumsum(W), q - 1e-12)]


def test_per_row_values_and_shared_values_agree():
    rng = np.random.default_rng(1)
    z = np.sort(rng.standard_t(4, 300))
    sig = np.array([0.01, 0.02, 0.005])
    obs = np.array([0.003, -0.02, 0.0])
    V, W = db.scaled_dist(z, sig)
    a = db.dist_metrics(V, W, obs, QS, (0.0,), 0.003)
    for i in range(3):
        b = db.dist_metrics((sig[i] * z)[None, :], np.full((1, 300), 1 / 300), obs[i:i + 1], QS, (0.0,), 0.003)
        for k in ("crps", "mean", "pit", "tw_up", "tw_dn"):
            assert math.isclose(a[k][i], b[k][0], rel_tol=1e-9, abs_tol=1e-15)
        assert np.allclose(a["q"][i], b["q"][0])


def test_hourly_rv_sums_squared_minute_returns_within_the_bar():
    idx = pd.date_range("2024-01-01", periods=180, freq="1min", tz="UTC")
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.001, len(idx))
    mc = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    mc.iloc[70:75] = np.nan                      # 約定のない分は直前の値（リターン 0）
    hours = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    rv = db.hourly_rv(mc, hours)
    lr = np.diff(np.log(mc.ffill().to_numpy()))
    assert math.isclose(rv.iloc[0], np.sum(lr[:59] ** 2))           # 最初の分はリターンがない
    assert math.isclose(rv.iloc[1], np.sum(lr[59:119] ** 2))        # 分 60 のリターン = 終値 60 / 終値 59
    assert math.isclose(rv.iloc[2], np.sum(lr[119:179] ** 2))


def test_har_features_and_target_have_no_lookahead():
    idx = pd.date_range("2024-01-01", periods=600, freq="h", tz="UTC")
    rng = np.random.default_rng(3)
    rv = pd.Series(rng.lognormal(-10, 1, len(idx)), index=idx)
    X, y = db.har_features(rv), db.har_target(rv, 4)
    rv2 = rv.copy()
    rv2.iloc[400:] *= 5
    X2, y2 = db.har_features(rv2), db.har_target(rv2, 4)
    pd.testing.assert_frame_equal(X.iloc[:400], X2.iloc[:400])
    assert np.allclose(y.iloc[:396], y2.iloc[:396]) and not np.allclose(y.iloc[396:596], y2.iloc[396:596])
    assert math.isclose(y.iloc[10], np.log(rv.iloc[11:15].sum()))
    assert math.isclose(X["har_24"].iloc[100], np.log(rv.iloc[77:101].mean()))


def test_garch_filter_has_no_lookahead_and_h_step_variance_is_the_sum_of_forecasts():
    par = {"mu": 0.0, "omega": 2e-7, "alpha": 0.08, "beta": 0.9, "nu": 4.0}
    rng = np.random.default_rng(4)
    r = rng.normal(0, 0.004, 500)
    s = db.garch_next_var(par, r)
    r2 = r.copy()
    r2[300:] *= 3
    assert np.allclose(s[:300], db.garch_next_var(par, r2)[:300])
    # σ²_{t+1|t} = ω + α r_t² + β σ²_{t|t−1}
    assert math.isclose(s[10], par["omega"] + par["alpha"] * r[10] ** 2 + par["beta"] * s[9])
    vbar = par["omega"] / (1 - par["alpha"] - par["beta"])
    h = 24
    brute = [sum(vbar + (par["alpha"] + par["beta"]) ** k * (v - vbar) for k in range(h)) for v in s[:5]]
    assert np.allclose(db.garch_sigma_h(par, s[:5], h) ** 2, brute)
    # α + β = 1（学習期間の当てはめが境界に来た場合）: E_t[σ²_{t+k}] = σ²_{t+1|t} + (k − 1)ω
    ig = par | {"alpha": 0.08, "beta": 0.92}
    assert np.allclose(db.garch_sigma_h(ig, s[:5], h) ** 2, h * s[:5] + par["omega"] * h * (h - 1) / 2)


def test_rolling_windows_use_only_confirmed_returns():
    y = np.arange(100, dtype=float)
    w = db.rolling_windows(y, np.array([60, 61]), n=10, h=4)
    assert w[0].tolist() == list(range(46, 56)) and w[1].tolist() == list(range(47, 57))   # 位置 60 − 4 − 10 〜 60 − 4 − 1


def test_sign_magnitude_distribution():
    grid = np.array([0.5, 1.0, 2.0])
    p = np.array([[0.2, 0.5, 0.9], [0.5, 0.5, 0.5]])
    V, W = db.sm_dist(p, grid, np.array([0.01, 0.02]))
    assert np.all(np.diff(V, axis=1) >= 0) and np.allclose(W.sum(axis=1), 1.0)
    m = db.dist_metrics(V, W, np.array([0.0, 0.0]), QS, (0.0,), 0.003)
    assert np.allclose(m["p"][:, 0], p.mean(axis=1))
    assert math.isclose(m["mean"][0], 0.01 * np.mean(grid * (2 * p[0] - 1)))


def test_elementary_score_difference_equals_payoff_difference_and_corp_identity():
    rng = np.random.default_rng(5)
    y = rng.normal(0.0005, 0.01, 2000)
    xa, xb = y + rng.normal(0, 0.01, 2000), rng.normal(0, 0.003, 2000)
    th = 0.003
    S = lambda x: np.maximum(y - th, 0) - (x > th) * (y - th)   # noqa: E731  初等スコア
    assert math.isclose(S(xa).mean() - S(xb).mean(),
                        -(db.elementary_payoff(xa, y, th).mean() - db.elementary_payoff(xb, y, th).mean()), abs_tol=1e-15)
    p = np.clip(0.5 + 3 * (xa - xa.mean()), 0, 1)
    o = (y > 0).astype(float)
    c = db.corp(p, o)
    assert math.isclose(c["brier"], c["mcb"] - c["dsc"] + c["unc"], abs_tol=1e-12)
    assert c["mcb"] >= -1e-12 and c["dsc"] >= -1e-12


@pytest.fixture(scope="module")
def small():
    """期待リターン 0 の合成データ（10 か月、2023-07 から評価）と、小さく軽くした設定。"""
    idx = pd.date_range("2022-10-01", "2023-08-01", freq="h", tz="UTC", inclusive="left")
    cal = {"garch": {"mu": 0.0, "omega": 3e-7, "alpha": 0.08, "beta": 0.88, "nu": 4.5},
           "volume": {"a": 5.0, "b": 1.0, "phi": 0.6, "sd": 0.5}, "p0": 3e6,
           "ms2": {"sigma": [0.003, 0.01], "P": [[0.99, 0.01], [0.05, 0.95]]}}
    h1, mc = nullsim.simulate(idx, "garch_t", cal, 7)
    cfg = {"split": "2023-07-01", "horizons": [4, 24], "cost": 0.003, "quantiles": list(QS), "thresholds": [0.0, 0.003, -0.003],
           "logloss_eps": 1e-3, "elementary_theta": 0.003, "murphy_thetas": [0.0, 0.003], "top_share": 0.2,
           "forest": {"n_estimators": 20, "min_samples_leaf": 25, "max_features": "sqrt", "max_samples": 0.5,
                      "split_target": "bins", "n_bins": 8, "random_state": 0},
           "har": {"windows": [1, 4, 24, 168], "calendar": True, "rv_floor": 1e-10}, "rolling_n": [168, 720],
           "sign_magnitude": {"grid": 10, "valid_frac": 0.2, "early_stopping_rounds": 10,
                               "lgbm": {"n_estimators": 50, "learning_rate": 0.05, "num_leaves": 15,
                                                   "min_child_samples": 50, "random_state": 0}}}
    return h1, mc, cfg


def test_pipeline_runs_and_predictions_have_no_lookahead(small):
    h1, mc, cfg = small
    out = db.run_pipeline(h1, mc, cfg, n_jobs=1, log=lambda *a: None)
    # 評価期間の途中から先の価格を書き換えても、それより前の行の予測（分位点、確率、平均）は変わらない
    cut = pd.Timestamp("2023-07-15", tz="UTC")
    h2, mc2 = h1.copy(), mc.copy()
    later = h2.index >= cut
    h2.loc[later, ["open", "high", "low", "close"]] *= np.exp(np.linspace(0, 0.3, later.sum()))[:, None]
    mc2[mc2.index >= cut] *= 1.1
    out2 = db.run_pipeline(h2, mc2, cfg, n_jobs=1, log=lambda *a: None)
    for h in (4, 24):
        H, H2 = out["horizons"][h], out2["horizons"][h]
        before = H["index"] < cut
        assert before.sum() > 100 and set(H["models"]) == set(db.MODELS)
        for m in db.MODELS:
            for k in ("q", "p", "mean"):
                assert np.allclose(H["models"][m][k][before], H2["models"][m][k][before]), (h, m, k)
            assert np.all(np.isfinite(H["models"][m]["crps"]))
        # 位置を固定したフォレストの中央値は学習期間の中央値
        assert np.allclose(H["models"]["forest_fixed"]["q"][:, 3], H["median_train"])
        # 幅はフォレストと同じ
        wf = H["models"]["forest"]["q"][:, 6] - H["models"]["forest"]["q"][:, 0]
        wx = H["models"]["forest_fixed"]["q"][:, 6] - H["models"]["forest_fixed"]["q"][:, 0]
        assert np.allclose(wf, wx)
        row = db.summary_row(H["models"]["har"], H["y"], cfg)
        assert 0 < row["brier_0"] < 0.5 and "spread_series" in row
