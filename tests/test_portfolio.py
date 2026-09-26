import numpy as np
import pandas as pd

from bbresearch.portfolio import best_combination, min_var_weights, shrunk_cov, shrunk_mean, summary, walk_forward


def test_min_var_weights_matches_closed_form_when_unconstrained():
    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]])
    w, var = min_var_weights(cov, w_max=1.0)
    inv = np.linalg.inv(cov)
    w_ref = inv @ np.ones(3) / (np.ones(3) @ inv @ np.ones(3))
    assert np.allclose(w, w_ref, atol=1e-5) and abs(var - w_ref @ cov @ w_ref) < 1e-8


def test_min_var_weights_respects_cap_and_target():
    cov = np.diag([0.01, 0.04, 0.09])
    mu = np.array([0.0, 0.1, 0.3])
    w, _ = min_var_weights(cov, w_max=0.4)
    assert w.max() <= 0.4 + 1e-9 and abs(w.sum() - 1) < 1e-9
    w2, var2 = min_var_weights(cov, mu, target=0.15, w_max=1.0)
    assert mu @ w2 >= 0.15 - 1e-6 and var2 > min_var_weights(cov, w_max=1.0)[1]
    assert min_var_weights(cov, mu, target=0.5, w_max=1.0) is None  # 届かない
    assert min_var_weights(cov[:2, :2], w_max=0.4) is None  # 2 銘柄 × 40% では合計 1 にならない


def test_best_combination_prefers_uncorrelated_low_vol_assets():
    names = ["a", "b", "c", "d", "e"]
    sd = np.array([0.02, 0.02, 0.02, 0.05, 0.05])
    corr = np.full((5, 5), 0.9)
    np.fill_diagonal(corr, 1.0)
    corr[0, 1] = corr[1, 0] = 0.0  # a と b は無相関
    corr[0, 2] = corr[2, 0] = 0.0
    corr[1, 2] = corr[2, 1] = 0.0
    cov = corr * np.outer(sd, sd)
    mu = np.zeros(5)
    best = best_combination(cov, mu, names, k_max=3, target=None, w_max=0.4)
    assert set(best["names"]) == {"a", "b", "c"}
    assert abs(best["var"] - (0.02**2) / 3) < 1e-6


def test_walk_forward_uses_only_past_data_and_charges_costs():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=900, freq="D", tz="UTC")
    n = 5
    cols = ["btc_jpy", "a_jpy", "b_jpy", "c_jpy", "d_jpy"]
    r = rng.normal(0.0005, 0.03, size=(900, n)) + rng.normal(0, 0.02, size=(900, 1))  # 共通要因で相関あり
    close = pd.DataFrame(np.exp(np.cumsum(r, axis=0)) * 100, index=idx, columns=cols)
    res = walk_forward(close, "2021-01-01", k_max=3, w_max=0.5, target_mults=(None, 1.0), est_days=300, cost=0.0015)
    assert set(res.daily.columns) == {"minvar", "minvar_x1.0", "equal", "btc_jpy"}
    assert res.daily.index[0] >= pd.Timestamp("2021-01-01", tz="UTC")
    # 2021-06 以降の価格を変えても、2021-06 の重みは変わらない
    close2 = close.copy()
    close2.loc[close2.index >= "2021-06-01"] *= np.exp(rng.normal(0, 0.1, size=(int((close2.index >= "2021-06-01").sum()), n)))
    res2 = walk_forward(close2, "2021-01-01", k_max=3, w_max=0.5, target_mults=(None, 1.0), est_days=300, cost=0.0015)
    w1 = [w for w in res.weights["minvar"] if w["date"] <= "2021-06-01"]
    w2 = [w for w in res2.weights["minvar"] if w["date"] <= "2021-06-01"]
    assert w1 == w2 and len(w1) >= 6
    # 費用: 最初の月の初日は回転率 1 × cost だけ差し引かれている
    first = res.daily.index[0]
    w0 = pd.Series(res.weights["btc_jpy"][0]["weights"]).reindex(cols).fillna(0.0)
    gross = float((np.log(close).diff().loc[first, cols] * w0).sum())
    assert abs(res.daily.loc[first, "btc_jpy"] - (gross - 0.0015)) < 1e-12
    st = summary(res.daily["equal"])
    assert st["ann_vol"] > 0 and -1 < st["max_drawdown"] <= 0


def test_shrinkage_helpers():
    rng = np.random.default_rng(1)
    R = rng.normal(size=(200, 4)) * [0.01, 0.02, 0.03, 0.04]
    cov = shrunk_cov(R)
    assert cov.shape == (4, 4) and np.allclose(cov, cov.T) and np.linalg.eigvalsh(cov).min() > 0
    m = shrunk_mean(R, 0.5)
    assert np.allclose(m, 0.5 * R.mean(axis=0) + 0.5 * R.mean())


def test_per_asset_caps_and_zero_weights_are_pruned():
    names = ["btc_jpy", "a", "b", "c"]
    cov = np.diag([0.01, 0.04, 0.09, 0.16])
    mu = np.zeros(4)
    best = best_combination(cov, mu, names, k_max=3, target=None, w_max=0.4, caps={"btc_jpy": 0.6})
    w = dict(zip(best["names"], best["weights"]))
    assert abs(w["btc_jpy"] - 0.6) < 1e-6 and abs(sum(w.values()) - 1) < 1e-9
    assert all(v > 1e-6 for v in w.values())
    # 上限の合計が 1 に満たない組み合わせ（2 銘柄 × 40%）は使われない
    best2 = best_combination(cov[1:, 1:], mu[1:], names[1:], k_max=2, target=None, w_max=0.4)
    assert best2 is None
    assert min_var_weights(cov[:2, :2], w_max=np.array([0.6, 0.4]))[0].tolist() == [0.6, 0.4]


def test_vol_target_moves_into_cash():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2020-01-01", periods=800, freq="D", tz="UTC")
    cols = ["btc_jpy", "a_jpy", "b_jpy", "c_jpy"]
    r = rng.normal(0, 0.05, size=(800, 4)) + rng.normal(0, 0.03, size=(800, 1))  # 年率ボラ 100% 超
    close = pd.DataFrame(np.exp(np.cumsum(r, axis=0)) * 100, index=idx, columns=cols)
    res = walk_forward(close, "2021-01-01", k_max=3, w_max=0.5, target_mults=(None,), est_days=300,
                       vol_targets=(None, 0.2), caps={"btc_jpy": 0.6})
    assert {"minvar", "minvar_vt20", "equal", "btc_jpy"} <= set(res.daily.columns)
    full, vt = res.weights["minvar"], res.weights["minvar_vt20"]
    assert all(w["cash"] == 0.0 for w in full) and all(0.5 < w["cash"] < 1.0 for w in vt)
    # 暗号資産の相対の重みは同じ（縮めているだけ）
    for a, b in zip(full, vt):
        s = 1 - b["cash"]
        assert set(a["weights"]) == set(b["weights"])
        assert all(abs(b["weights"][k] - a["weights"][k] * s) < 2e-4 for k in a["weights"])
    assert summary(res.daily["minvar_vt20"])["ann_vol"] < summary(res.daily["minvar"])["ann_vol"] * 0.5


def test_short_vol_window_reacts_to_regime_change():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2020-01-01", periods=700, freq="D", tz="UTC")
    cols = ["btc_jpy", "a_jpy", "b_jpy"]
    sd = np.where(np.arange(700)[:, None] < 560, 0.01, 0.05)  # 最後の 140 日でボラが 5 倍
    r = rng.normal(0, 1, size=(700, 3)) * sd + rng.normal(0, 1, size=(700, 1)) * sd * 0.5
    close = pd.DataFrame(np.exp(np.cumsum(r, axis=0)) * 100, index=idx, columns=cols)
    res = walk_forward(close, "2021-01-01", k_max=3, w_max=0.5, target_mults=(None,), est_days=300,
                       vol_targets=(0.3,), vol_days=(None, 60), caps={"btc_jpy": 0.6})
    assert {"minvar_vt30", "minvar_vt30_w60"} <= set(res.daily.columns)
    long_w = {w["date"]: w["cash"] for w in res.weights["minvar_vt30"]}
    short_w = {w["date"]: w["cash"] for w in res.weights["minvar_vt30_w60"]}
    # ボラが上がった後の月は、短い窓のほうが早く JPY を増やす
    late = [d for d in long_w if d >= "2021-10-01"]
    assert late and all(short_w[d] >= long_w[d] - 1e-9 for d in late) and any(short_w[d] > long_w[d] + 0.05 for d in late)
