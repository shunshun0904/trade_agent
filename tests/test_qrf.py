import numpy as np

from bbresearch.qrf import (QUANTILES, QuantileForest, crps_weighted, evaluate_quantiles, prob_exceed, prob_exceed_rows,
                            rolling_empirical, weighted_quantiles)


def _hetero(n, seed):
    """x1 で散らばりが変わり、x2 で中心が変わるデータ。x3 は無関係。"""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 3))
    scale = 0.5 + 2.0 * (X[:, 0] > 0)
    y = 0.5 * X[:, 1] + scale * rng.normal(size=n)
    return X, y, scale


def test_quantile_forest_tracks_conditional_spread_and_center():
    X, y, _ = _hetero(6000, 0)
    Xt, yt, scale_t = _hetero(2000, 1)
    qf = QuantileForest(n_estimators=100, min_samples_leaf=40, max_features=1.0, random_state=0, n_jobs=1).fit(X, y)
    q = qf.predict_quantiles(Xt)
    width = q[:, QUANTILES.index(0.95)] - q[:, QUANTILES.index(0.05)]
    assert width[scale_t > 1].mean() > 2.0 * width[scale_t < 1].mean()   # 散らばりの違いを捉える
    med = q[:, QUANTILES.index(0.5)]
    assert np.corrcoef(med, 0.5 * Xt[:, 1])[0, 1] > 0.6                    # 中心の違いを捉える
    ev = evaluate_quantiles(yt, q)
    for qq in QUANTILES:
        assert abs(ev["coverage"][qq] - qq) < 0.06, (qq, ev["coverage"][qq])
    assert 0.84 < ev["interval90"] < 0.96
    # 無条件の分位点より良い
    uncond = np.tile(np.quantile(y, QUANTILES), (len(yt), 1))
    assert ev["pinball_mean"] < 0.9 * evaluate_quantiles(yt, uncond)["pinball_mean"]


def test_weights_sum_to_one_and_quantiles_are_monotone():
    X, y, _ = _hetero(500, 2)
    qf = QuantileForest(n_estimators=20, min_samples_leaf=10, max_features=1.0, random_state=0, n_jobs=1).fit(X, y)
    w = qf.weights(X[:7])
    assert np.allclose(w.sum(axis=1), 1.0)
    q = qf.predict_quantiles(X[:7])
    assert (np.diff(q, axis=1) >= 0).all()


def test_crps_matches_brute_force_and_prob_exceed():
    rng = np.random.default_rng(3)
    y = np.sort(rng.normal(size=50))
    w = rng.uniform(size=(4, 50))
    w /= w.sum(axis=1, keepdims=True)
    obs = rng.normal(size=4)
    got = crps_weighted(y, w, obs)
    for r in range(4):
        t1 = (w[r] * np.abs(y - obs[r])).sum()
        t2 = (w[r][:, None] * w[r][None, :] * np.abs(y[:, None] - y[None, :])).sum()
        assert abs(got[r] - (t1 - 0.5 * t2)) < 1e-12
    p = prob_exceed(y, w, 0.0)
    assert np.allclose(p, (w * (y > 0)[None, :]).sum(axis=1))
    q = weighted_quantiles(y, w, (0.5,))
    for r in range(4):
        cw = np.cumsum(w[r])
        assert q[r, 0] == y[np.searchsorted(cw, 0.5)]


def test_rolling_empirical_uses_only_settled_returns():
    tail = np.arange(100, dtype=float)
    test = np.arange(100, 130, dtype=float)
    q = rolling_empirical(tail, test, n=50, horizon=4, qs=(0.5,))
    # 行 0 の時点で確定しているのは学習末尾の 100 − 4 = 96 個目まで → 窓は [46, 96) の中央値 70.5
    assert abs(q[0, 0] - np.quantile(np.arange(46, 96), 0.5)) < 1e-9
    assert abs(q[10, 0] - np.quantile(np.arange(56, 106), 0.5)) < 1e-9


def test_bins_split_and_rowwise_exceed_probability():
    X, y, scale_t = _hetero(4000, 4)
    qf = QuantileForest(n_estimators=60, min_samples_leaf=40, max_features=1.0, random_state=0, n_jobs=1,
                        split_target="bins", n_bins=6).fit(X, y)
    q = qf.predict_quantiles(X[:500])
    assert (np.diff(q, axis=1) >= 0).all()
    width = q[:, -1] - q[:, 0]
    assert width[scale_t[:500] > 1].mean() > 2.0 * width[scale_t[:500] < 1].mean()   # 区間割合の分割でも散らばりを捉える
    zs, w = next(qf.predict_distribution(X[:5]))
    c = np.array([-1.0, 0.0, 0.5, 100.0, -100.0])
    p = prob_exceed_rows(zs, w, c)
    for i in range(5):
        assert abs(p[i] - (w[i] * (zs > c[i])).sum()) < 1e-12
