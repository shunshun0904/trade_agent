"""1 時間足の指標からの翌日ボラ予測と、ポートフォリオの JPY 比率の日次調整に使う倍率。2026-09-27 オーナー決定。

- 分位点回帰フォレスト（分位点ビン分割、葉 25、√p）を、目的変数 = 24 本後までの対数収益率で学習し、
  予測分布の標準偏差 σ̂_24h をボラ予測にする（向きは使わない）。
- 前向き: refit_months ごとに、それより前のデータだけで学習し直し、次の期間を予測する。
- 倍率 = 予測した年率ボラ ÷ 長期（365 日）の実現ボラ。walk_forward の vol_ratios に渡すと、推定ボラにこの倍率を掛けて
  目標ボラに縮めるので、JPY の割合が日次で動く。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bbresearch.indicators import forward_return, indicator_table
from bbresearch.qrf import QuantileForest

HOURS_PER_YEAR = 24 * 365


def hourly_vol_forecast(df: pd.DataFrame, horizon: int = 24, start: str = "2022-01-01", refit_months: int = 3,
                        n_estimators: int = 150, min_samples_leaf: int = 25, max_features="sqrt",
                        split_target: str = "bins", verbose: bool = True) -> pd.Series:
    """1 時間足 → 各足の終値時点での「次の horizon 本の対数収益率の標準偏差」の予測（index = 足の終値の時刻）。

    start 以降を予測する。refit_months ごとに学習し直し、学習には目的変数が確定した行（t + horizon ≤ 学習期間の末）だけを使う。
    """
    feats = indicator_table(df)
    y = forward_return(df["close"], horizon)
    cols = list(feats.columns)
    ok = feats.notna().all(axis=1)
    X_all = feats[cols].to_numpy()
    y_all = y.to_numpy()
    idx = df.index
    out = pd.Series(np.nan, index=idx + pd.Timedelta(hours=1))   # 予測は足の終値の時刻に置く
    t = pd.Timestamp(start, tz="UTC")
    end = idx[-1]
    while t <= end:
        t_next = t + pd.DateOffset(months=refit_months)
        train = ok & (idx < t - pd.Timedelta(hours=horizon)) & ~np.isnan(y_all)
        pred = ok & (idx >= t) & (idx < t_next)
        if train.sum() >= 1000 and pred.any():
            qf = QuantileForest(n_estimators, min_samples_leaf, max_features, 0.5, split_target=split_target).fit(X_all[train], y_all[train])
            out.iloc[np.where(pred)[0]] = qf.predict_std(X_all[pred])
            if verbose:
                print(f"ボラ予測: {t.date()} 〜 {min(t_next, end).date()} を {int(train.sum()):,} 行で学習して予測")
        t = t_next
    return out


def realized_hourly_vol(df: pd.DataFrame, n: int, horizon: int = 24) -> pd.Series:
    """直近 n 本の 1 時間リターンの標準偏差 × √horizon（足の終値の時刻に置く）。フォレストなしの基準。"""
    r = np.log(df["close"]).diff()
    s = r.rolling(n, min_periods=n).std() * np.sqrt(horizon)
    return pd.Series(s.to_numpy(), index=df.index + pd.Timedelta(hours=1))


def daily_ratio(hourly_sigma: pd.Series, close_daily: pd.Series, days: int = 365, horizon: int = 24) -> pd.Series:
    """各日付 d について、d 以前の最後の 1 時間足の予測 σ̂ を、その予測自身の d より前の直近 days 日の平均で割る。

    同じ系列の長期平均で割るので、1 時間足ベースの σ̂ と日足ベースの推定ボラの尺度の違い（日中と終値間の差）が打ち消され、
    倍率の平均は 1 前後になる。walk_forward の推定ボラ（直前 days 日の日次共分散）に掛けて使う。
    """
    at_day = hourly_sigma.reindex(hourly_sigma.index.union(close_daily.index)).ffill().reindex(close_daily.index)
    long_run = at_day.rolling(days, min_periods=int(days * 0.8)).mean().shift(1)   # d より前の日だけ
    return (at_day / long_run).replace([np.inf, -np.inf], np.nan)
