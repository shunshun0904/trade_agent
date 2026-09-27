import numpy as np
import pandas as pd

from bbresearch.indicators import forward_return, indicator_table, rsi


def _ohlcv(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    close = 1e7 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    opn = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.001, n))
    high = np.maximum(opn, close) * np.exp(np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(opn, close) * np.exp(-np.abs(rng.normal(0, 0.003, n)))
    vol = rng.lognormal(0, 1, n)
    return pd.DataFrame({"open": opn, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def test_indicators_use_only_past_bars():
    df = _ohlcv()
    f = indicator_table(df)
    assert f.shape[0] == len(df) and f.shape[1] > 60
    cut = 400
    df2 = df.copy()
    df2.iloc[cut:, :4] *= 1.3              # 未来の足を書き換える
    df2.iloc[cut:, 4] *= 5.0
    f2 = indicator_table(df2)
    a, b = f.iloc[:cut], f2.iloc[:cut]
    diff = (a - b).abs().max()
    assert (diff.fillna(0) < 1e-9).all(), diff[diff > 1e-9]
    # 後半は変わる（先読みのテストが空振りしていないことの確認）
    assert (f.iloc[cut + 50:] - f2.iloc[cut + 50:]).abs().max().max() > 1e-6


def test_indicator_ranges_and_rsi_definition():
    df = _ohlcv()
    f = indicator_table(df).dropna()
    for col in ("o_rsi14", "o_stoch_k", "o_mfi14", "o_ultimate", "o_stoch_rsi"):
        assert f[col].between(-1e-6, 100 + 1e-6).all(), col
    assert f["o_willr14"].between(-100 - 1e-6, 1e-6).all()
    assert f["t_dc20"].between(-1e-6, 1 + 1e-6).all()
    assert f["c_hour"].between(0, 23).all() and f["c_weekday"].between(0, 6).all()
    # RSI: 上げ続ければ 100 に近づく
    up = pd.Series(np.arange(1, 200, dtype=float))
    assert rsi(up, 14).iloc[-1] > 99


def test_forward_return_definition():
    close = pd.Series([100.0, 110.0, 121.0, 133.1, 146.41])
    r = forward_return(close, 2)
    assert abs(r.iloc[0] - np.log(1.21)) < 1e-12 and np.isnan(r.iloc[-1]) and np.isnan(r.iloc[-2])
