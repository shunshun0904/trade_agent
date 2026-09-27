import numpy as np
import pandas as pd

from bbresearch.portfolio import walk_forward
from bbresearch.qrf import QuantileForest
from bbresearch.volforecast import daily_ratio, hourly_vol_forecast, realized_hourly_vol


def _hourly(n=24 * 500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    vol = 0.003 * (1 + 0.9 * np.sign(np.sin(np.arange(n) / 400)))   # 高ボラと低ボラを行き来する
    close = 1e7 * np.exp(np.cumsum(rng.normal(0, vol)))
    opn = np.r_[close[0], close[:-1]]
    high = np.maximum(opn, close) * np.exp(np.abs(rng.normal(0, 0.001, n)))
    low = np.minimum(opn, close) * np.exp(-np.abs(rng.normal(0, 0.001, n)))
    return pd.DataFrame({"open": opn, "high": high, "low": low, "close": close, "volume": rng.lognormal(0, 1, n)}, index=idx), vol


def test_predict_std_matches_weighted_moments():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 3))
    y = X[:, 0] * 2 + rng.normal(size=600)
    qf = QuantileForest(n_estimators=20, min_samples_leaf=10, max_features=1.0, n_jobs=1).fit(X, y)
    sd = qf.predict_std(X[:7])
    w = qf.weights(X[:7])
    ys = qf._y
    ref = np.sqrt((w * ys**2).sum(axis=1) - ((w * ys).sum(axis=1)) ** 2)
    assert np.allclose(sd, ref)


def test_hourly_forecast_is_forward_only_and_tracks_regimes():
    df, vol = _hourly()
    fc = hourly_vol_forecast(df, horizon=24, start="2024-09-01", refit_months=2, n_estimators=30, verbose=False)
    got = fc.dropna()
    assert got.index[0] >= pd.Timestamp("2024-09-01", tz="UTC") + pd.Timedelta(hours=1)
    # 予測は足の終値の時刻に置く: 未来の足を書き換えても、それより前の予測は変わらない
    df2 = df.copy()
    cut = pd.Timestamp("2025-02-01", tz="UTC")
    df2.loc[df2.index >= cut, ["open", "high", "low", "close"]] *= 1.5
    fc2 = hourly_vol_forecast(df2, horizon=24, start="2024-09-01", refit_months=2, n_estimators=30, verbose=False)
    before = fc.index < cut - pd.Timedelta(hours=24)   # 目的変数が確定している範囲より前は学習も同じ
    assert np.allclose(fc[before].dropna(), fc2[before].dropna())
    # 高ボラ期の予測 > 低ボラ期の予測
    true_vol = pd.Series(vol, index=df.index + pd.Timedelta(hours=1)).reindex(got.index)
    assert got[true_vol > 0.004].mean() > 1.5 * got[true_vol < 0.002].mean()
    rv = realized_hourly_vol(df, 24)
    assert rv.index[0] == df.index[0] + pd.Timedelta(hours=1)


def test_daily_ratio_and_walk_forward_with_ratio():
    rng = np.random.default_rng(2)
    days = pd.date_range("2022-01-01", periods=900, freq="D", tz="UTC")
    close = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.03, (900, 4)), axis=0)) * 100, index=days,
                         columns=["btc_jpy", "a_jpy", "b_jpy", "c_jpy"])
    hourly_sigma = pd.Series(0.02, index=pd.date_range("2022-01-01 01:00", periods=900 * 24, freq="h", tz="UTC"))
    ratio = daily_ratio(hourly_sigma, close["btc_jpy"], days=365)
    assert ratio.dropna().index[0] >= days[int(365 * 0.8)]
    assert np.allclose(ratio.dropna(), 1.0)   # 予測が一定なら倍率は 1
    # 倍率 2 を毎日掛ければ、倍率なしより暗号資産の比率が下がる（JPY が増える）
    big = pd.Series(2.0, index=days)
    res = walk_forward(close, "2023-06-01", k_max=3, w_max=0.5, target_mults=(None,), est_days=365, cost=0.0,
                       vol_targets=(0.30,), cash_freq_days=(1,), vol_ratios={None: None, "x2": big})
    c_plain = np.mean([w["cash"] for w in res.weights["minvar_vt30_c1"]])
    c_big = np.mean([w["cash"] for w in res.weights["minvar_vt30_c1_rx2"]])
    assert c_big > c_plain
    assert len(res.weights["minvar_vt30_c1_rx2"]) > 300   # 日次の区切り
