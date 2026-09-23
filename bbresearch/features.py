"""Phase 4: 特徴量。

すべて開始時刻が t0 より前の足（= 終了時刻が t0 以前の足）から計算する。
足ごとの特徴量（その足の終了時点までのデータで計算）を先に作り、イベントの t0 で
「終了時刻が t0 の足」の行を引く。ローリング計算は min_periods = 窓長。

本番（Phase 7）でも bar_features → event_features を同じように呼ぶ。

vol_z は log(1 + volume) を使う（約定なしの足で log(0) にならないようにするため）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .signals import ewm_sigma, log_returns

RET_WINDOWS = (1, 4, 16, 96)
RV_WINDOWS = (16, 96)
PK_WINDOWS = (16, 96)
IMB_WINDOWS = (4, 16)
MA_WINDOWS = (16, 96)
Z_WINDOW = 96
LARGE_WINDOWS = (16,)

EVENT_ONLY = ["cusum_ratio", "sigma", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]


def _zscore(x: pd.Series, n: int) -> pd.Series:
    roll = x.rolling(n, min_periods=n)
    return (x - roll.mean()) / roll.std()


def bar_features(bars: pd.DataFrame, sigma_span: int = 96) -> pd.DataFrame:
    """各行が、その足の終了時点までのデータだけで計算された特徴量。"""
    r = log_returns(bars)
    lc = np.log(bars["close"])
    out = pd.DataFrame(index=bars.index)
    for k in RET_WINDOWS:
        out[f"ret_{k}"] = lc - lc.shift(k)
    for k in RV_WINDOWS:
        out[f"rv_{k}"] = r.rolling(k, min_periods=k).std()
    hl2 = np.log(bars["high"] / bars["low"]) ** 2
    for k in PK_WINDOWS:
        out[f"pk_vol_{k}"] = np.sqrt(hl2.rolling(k, min_periods=k).mean() / (4 * np.log(2)))
    out["vol_z"] = _zscore(np.log1p(bars["volume"]), Z_WINDOW)
    diff = bars["buy_volume"] - bars["sell_volume"]
    for k in IMB_WINDOWS:
        vol = bars["volume"].rolling(k, min_periods=k).sum()
        out[f"imb_{k}"] = (diff.rolling(k, min_periods=k).sum() / vol.where(vol > 0)).fillna(0.0).where(vol.notna())
    out["ntrades_z"] = _zscore(bars["n_trades"].astype("float64"), Z_WINDOW)
    if "large_volume" in bars:
        for k in LARGE_WINDOWS:
            vol = bars["volume"].rolling(k, min_periods=k).sum()
            out[f"large_share_{k}"] = bars["large_volume"].rolling(k, min_periods=k).sum() / vol.where(vol > 0)
    out["empty_share"] = bars["is_empty"].astype("float64").rolling(Z_WINDOW, min_periods=Z_WINDOW).mean()
    sigma = ewm_sigma(bars, sigma_span)
    for n in MA_WINDOWS:
        sma = bars["close"].rolling(n, min_periods=n).mean()
        out[f"dist_ma_{n}"] = (bars["close"] - sma) / (sigma * bars["close"])
    out["bar_sigma"] = sigma
    return out


def event_features(events: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """イベントごとの特徴量。index は event_id。"""
    step = feats.index[1] - feats.index[0]
    rows = feats.reindex(events["t0"] - step)
    rows.index = events["event_id"].to_numpy()
    rows = rows.drop(columns=["bar_sigma"])
    t0 = pd.DatetimeIndex(events["t0"])
    hour = t0.hour + t0.minute / 60.0
    rows["cusum_ratio"] = (events["s_value"].abs() / events["h"]).to_numpy()
    rows["sigma"] = events["sigma"].to_numpy()
    rows["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    rows["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    rows["dow_sin"] = np.sin(2 * np.pi * t0.dayofweek / 7)
    rows["dow_cos"] = np.cos(2 * np.pi * t0.dayofweek / 7)
    rows.index.name = "event_id"
    return rows
