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


def test_extended_indicators_no_lookahead_ranges_and_confirmation():
    """拡充後の全列で先読みがないこと、値の範囲、確認待ちの指標が確認前の値を使わないこと。"""
    df = _ohlcv(n=1500, seed=5)
    f = indicator_table(df)
    assert f.shape[1] >= 350
    cut = 1200
    df2 = df.copy()
    df2.iloc[cut:, :4] *= 1.3
    df2.iloc[cut:, 4] *= 5.0
    f2 = indicator_table(df2)
    diff = (f.iloc[:cut] - f2.iloc[:cut]).abs().max()
    assert (diff.fillna(0) < 1e-9).all(), diff[diff > 1e-9]
    # 範囲
    assert f["r_total"].dropna().between(-1, 1).all() and f["r_ma"].dropna().between(-1, 1).all()
    for col in [c for c in f.columns if c.startswith("p_cdl_")] + ["v_squeeze", "p_fib100_up", "c_weekend"]:
        assert set(f[col].dropna().unique()) <= {0.0, 1.0}, col
    for col in ("t_aroon_up25", "o_stoch_k_5_3_3", "o_stoch_rsi_k21", "o_mfi28", "o_ultimate_slow", "o_crsi", "v_rvi10"):
        assert f[col].dropna().between(-1e-6, 100 + 1e-6).all(), col
    assert f["t_st_dir10_3"].dropna().isin([1.0, -1.0]).all() and f["t_vstop_dir"].dropna().isin([1.0, -1.0]).all()
    assert f["p_vp24_va_pos"].notna().sum() > 1000 and f["p_vp168_at"].dropna().between(0, 1 + 1e-9).all()
    # 確認待ちの指標: 右側 5 本で確認するピボット高は、その 5 本が来るまで前の値のまま
    from bbresearch.ta import confirmed_extremes
    hv = np.zeros(30)
    hv[10] = 5.0                       # 10 本目が孤立した高値
    ph, _ = confirmed_extremes(hv, -hv, 5, 5)
    assert np.isnan(ph[:15]).all() and ph[15] == 5.0
