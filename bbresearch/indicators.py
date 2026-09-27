"""OHLCV の足からテクニカル指標（トレンド系・オシレーター系・ボラ・出来高・時刻）を計算する。2026-09-27 オーナー指示。

すべて numpy / pandas だけで書く。行 t の特徴量は足 t（とそれ以前）だけから計算する（足 t の終値で判断する想定）。
先読みがないことは tests/test_indicators.py で確かめる（未来の足を書き換えても過去の行が変わらない）。

値は価格の水準に依存しない形（比率・対数差・0〜100 の振れ）にそろえ、ペアや時期が違っても比べられるようにする。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder の平滑化（RSI・ATR・ADX が使う）。alpha = 1/n。"""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up, down = d.clip(lower=0.0), (-d).clip(lower=0.0)
    rs = wilder(up, n) / (wilder(down, n) + EPS)
    return 100 - 100 / (1 + rs)


def stochastic(df: pd.DataFrame, n: int = 14, d: int = 3) -> tuple[pd.Series, pd.Series]:
    lo, hi = df["low"].rolling(n, min_periods=n).min(), df["high"].rolling(n, min_periods=n).max()
    k = 100 * (df["close"] - lo) / (hi - lo + EPS)
    return k, k.rolling(d, min_periods=d).mean()


def williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    lo, hi = df["low"].rolling(n, min_periods=n).min(), df["high"].rolling(n, min_periods=n).max()
    return -100 * (hi - df["close"]) / (hi - lo + EPS)


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    m = sma(tp, n)
    mad = (tp - m).abs().rolling(n, min_periods=n).mean()
    return (tp - m) / (0.015 * mad + EPS)


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    flow = tp * df["volume"]
    up = flow.where(tp > tp.shift(1), 0.0)
    down = flow.where(tp < tp.shift(1), 0.0)
    ratio = up.rolling(n, min_periods=n).sum() / (down.rolling(n, min_periods=n).sum() + EPS)
    return 100 - 100 / (1 + ratio)


def adx(df: pd.DataFrame, n: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    up, down = df["high"].diff(), -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr = wilder(true_range(df), n)
    plus_di = 100 * wilder(plus_dm, n) / (atr + EPS)
    minus_di = 100 * wilder(minus_dm, n) / (atr + EPS)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    return wilder(dx, n), plus_di, minus_di


def aroon(df: pd.DataFrame, n: int = 25) -> tuple[pd.Series, pd.Series]:
    # 直近 n+1 本の中で高値・安値がどれだけ新しいか（100 = 今の足）
    hi_age = df["high"].rolling(n + 1, min_periods=n + 1).apply(lambda x: n - int(np.argmax(x)), raw=True)
    lo_age = df["low"].rolling(n + 1, min_periods=n + 1).apply(lambda x: n - int(np.argmin(x)), raw=True)
    return 100 * (n - hi_age) / n, 100 * (n - lo_age) / n


def parabolic_sar(df: pd.DataFrame, step: float = 0.02, cap: float = 0.2) -> pd.Series:
    """パラボリック SAR（Wilder）。逐次計算なので Python ループ。"""
    high, low = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(df)
    out = np.full(n, np.nan)
    if n < 2:
        return pd.Series(out, index=df.index)
    long, af = True, step
    sar, ep = low[0], high[0]
    for i in range(1, n):
        prev_sar = sar
        sar = prev_sar + af * (ep - prev_sar)
        if long:
            sar = min(sar, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < sar:  # 反転
                long, sar, ep, af = False, ep, low[i], step
            else:
                if high[i] > ep:
                    ep, af = high[i], min(cap, af + step)
        else:
            sar = max(sar, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > sar:
                long, sar, ep, af = True, ep, high[i], step
            else:
                if low[i] < ep:
                    ep, af = low[i], min(cap, af + step)
        out[i] = sar
    return pd.Series(out, index=df.index)


def linreg_slope(s: pd.Series, n: int) -> pd.Series:
    """直近 n 本の対数価格の回帰の傾き（1 本あたり）。"""
    x = np.arange(n) - (n - 1) / 2
    denom = float((x**2).sum())
    return s.rolling(n, min_periods=n).apply(lambda y: float((x * (y - y.mean())).sum() / denom), raw=True)


def indicator_table(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV（列 open, high, low, close, volume、index = 足の開始時刻 UTC、等間隔）→ 指標の表（同じ index）。

    列は 3 つの群に分かれる（列名の接頭辞）: t_ トレンド系、o_ オシレーター系、v_ ボラ・出来高、c_ 時刻。
    """
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    logc = np.log(c)
    r1 = logc.diff()
    f = pd.DataFrame(index=df.index)

    # ---- トレンド系
    for n in (5, 10, 20, 50, 100, 200):
        f[f"t_sma{n}"] = logc - np.log(sma(c, n))              # 移動平均からの対数距離
        f[f"t_ema{n}"] = logc - np.log(ema(c, n))
    f["t_sma20_50"] = np.log(sma(c, 20)) - np.log(sma(c, 50))
    f["t_sma50_200"] = np.log(sma(c, 50)) - np.log(sma(c, 200))
    macd = ema(c, 12) - ema(c, 26)
    signal = ema(macd, 9)
    f["t_macd"] = macd / c
    f["t_macd_hist"] = (macd - signal) / c
    a, pdi, mdi = adx(df, 14)
    f["t_adx14"], f["t_di_diff"] = a, pdi - mdi
    au, ad = aroon(df, 25)
    f["t_aroon_osc"] = au - ad
    f["t_sar"] = (c - parabolic_sar(df)) / c
    tenkan = (h.rolling(9, min_periods=9).max() + l.rolling(9, min_periods=9).min()) / 2
    kijun = (h.rolling(26, min_periods=26).max() + l.rolling(26, min_periods=26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((h.rolling(52, min_periods=52).max() + l.rolling(52, min_periods=52).min()) / 2).shift(26)
    f["t_ichi_tk"] = (tenkan - kijun) / c
    f["t_ichi_cloud"] = (c - (senkou_a + senkou_b) / 2) / c
    f["t_ichi_cloud_w"] = (senkou_a - senkou_b).abs() / c
    for n in (10, 30, 100):
        f[f"t_slope{n}"] = linreg_slope(logc, n)
    f["t_trix"] = ema(ema(ema(logc, 15), 15), 15).diff() * 1e3
    f["t_dc20"] = (c - l.rolling(20, min_periods=20).min()) / (h.rolling(20, min_periods=20).max() - l.rolling(20, min_periods=20).min() + EPS)
    f["t_dc55"] = (c - l.rolling(55, min_periods=55).min()) / (h.rolling(55, min_periods=55).max() - l.rolling(55, min_periods=55).min() + EPS)

    # ---- オシレーター系
    f["o_rsi7"], f["o_rsi14"], f["o_rsi28"] = rsi(c, 7), rsi(c, 14), rsi(c, 28)
    k, d = stochastic(df, 14, 3)
    f["o_stoch_k"], f["o_stoch_d"] = k, d
    rs = rsi(c, 14)
    f["o_stoch_rsi"] = 100 * (rs - rs.rolling(14, min_periods=14).min()) / (rs.rolling(14, min_periods=14).max() - rs.rolling(14, min_periods=14).min() + EPS)
    f["o_willr14"] = williams_r(df, 14)
    f["o_cci20"] = cci(df, 20)
    f["o_mfi14"] = mfi(df, 14)
    for n in (1, 3, 6, 12, 24, 72):
        f[f"o_roc{n}"] = logc - logc.shift(n)
    med = (h + l) / 2
    f["o_ao"] = (sma(med, 5) - sma(med, 34)) / c
    bp = c - pd.concat([l, c.shift(1)], axis=1).min(axis=1)
    tr = true_range(df)
    avg = [bp.rolling(n, min_periods=n).sum() / (tr.rolling(n, min_periods=n).sum() + EPS) for n in (7, 14, 28)]
    f["o_ultimate"] = 100 * (4 * avg[0] + 2 * avg[1] + avg[2]) / 7
    f["o_ppo"] = 100 * (ema(c, 12) - ema(c, 26)) / (ema(c, 26) + EPS)
    f["o_cmo14"] = 100 * (r1.clip(lower=0).rolling(14, min_periods=14).sum() - (-r1).clip(lower=0).rolling(14, min_periods=14).sum()) / (r1.abs().rolling(14, min_periods=14).sum() + EPS)

    # ---- ボラティリティ・出来高
    atr14 = wilder(tr, 14)
    f["v_atr14"] = atr14 / c
    f["v_atr_ratio"] = atr14 / (wilder(tr, 100) + EPS)
    for n in (24, 168):
        f[f"v_rv{n}"] = r1.rolling(n, min_periods=n).std()
    f["v_rv_ratio"] = f["v_rv24"] / (f["v_rv168"] + EPS)
    m20, s20 = sma(c, 20), c.rolling(20, min_periods=20).std()
    f["v_bb_pos"] = (c - m20) / (2 * s20 + EPS)                    # ボリンジャー %b を −1〜1 に
    f["v_bb_width"] = 4 * s20 / (m20 + EPS)
    f["v_kc_pos"] = (c - ema(c, 20)) / (2 * atr14 + EPS)           # ケルトナー
    f["v_range"] = (h - l) / c
    f["v_range_ratio"] = (h - l) / (sma(h - l, 24) + EPS)
    f["v_gap"] = (df["open"] - c.shift(1)) / c.shift(1)
    f["v_body"] = (c - df["open"]) / (h - l + EPS)                 # 実体の向きと大きさ
    f["v_upper_wick"] = (h - pd.concat([c, df["open"]], axis=1).max(axis=1)) / (h - l + EPS)
    f["v_lower_wick"] = (pd.concat([c, df["open"]], axis=1).min(axis=1) - l) / (h - l + EPS)
    f["v_vol_ratio24"] = v / (sma(v, 24) + EPS)
    f["v_vol_ratio168"] = v / (sma(v, 168) + EPS)
    obv = (np.sign(c.diff()).fillna(0.0) * v).cumsum()
    f["v_obv_slope"] = linreg_slope(obv, 24) / (sma(v, 24) + EPS)
    mfv = ((c - l) - (h - c)) / (h - l + EPS) * v
    f["v_cmf20"] = mfv.rolling(20, min_periods=20).sum() / (v.rolling(20, min_periods=20).sum() + EPS)
    tp = (h + l + c) / 3
    day = df.index.floor("D")
    cum_pv = (tp * v).groupby(day).cumsum()
    cum_v = v.groupby(day).cumsum()
    f["v_vwap_dist"] = (c - cum_pv / (cum_v + EPS)) / c            # 当日（UTC）の VWAP からの距離
    f["v_max_dd72"] = c / c.rolling(72, min_periods=72).max() - 1
    f["v_run_up72"] = c / c.rolling(72, min_periods=72).min() - 1

    # ---- 時刻（周期をそのまま入れる。木はそのまま分割できる）
    f["c_hour"] = df.index.hour.astype(float)
    f["c_weekday"] = df.index.dayofweek.astype(float)
    return f


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """足 t の終値で買い、horizon 本後の足の終値で売ったときの対数収益率（行 t に置く。末尾 horizon 行は NaN）。"""
    return np.log(close.shift(-horizon)) - np.log(close)
