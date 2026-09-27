"""約定の履歴から 1 時間足ごとの「約定フロー」の特徴量を作る。2026-09-27 オーナー決定（向きの情報源として約定を試す）。

行 t は足 [t, t+1h) とそれ以前の約定だけから計算する（足 t の終値の時刻に判断する想定）。列名の接頭辞は f_。
約定の side はテイカー側の方向（buy = 買い成行）と仮定する（bbresearch/labeling.py と同じ前提）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bbresearch.labeling import TradeTape

EPS = 1e-12
H_MS = 3_600_000


def hourly_aggregates(tape: TradeTape, index: pd.DatetimeIndex) -> pd.DataFrame:
    """各足 [t, t+1h) の約定の集計。index は足の開始時刻（UTC、等間隔でなくてもよい）。"""
    starts = index.tz_convert("UTC").as_unit("ms").asi8.astype("int64")   # asi8 の単位は index の分解能に依存するので ms にそろえる
    ends = starts + H_MS
    i0 = np.searchsorted(tape.ts, starts, side="left")
    i1 = np.searchsorted(tape.ts, ends, side="left")
    amount = tape.amount if tape.amount is not None else np.ones(len(tape.ts))
    notional = tape.price * amount
    buy = tape.is_buy.astype(float)
    logp = np.log(tape.price)
    dlogp = np.diff(logp, prepend=logp[0] if len(logp) else 0.0)
    sign = np.where(tape.is_buy, 1.0, -1.0)
    flip = np.r_[0.0, (sign[1:] != sign[:-1]).astype(float)] if len(sign) > 1 else np.zeros(len(sign))

    def seg_sum(x: np.ndarray) -> np.ndarray:
        c = np.r_[0.0, np.cumsum(x)]
        return c[i1] - c[i0]

    n = (i1 - i0).astype(float)
    out = pd.DataFrame(index=index)
    out["n"] = n
    out["vol"] = seg_sum(amount)
    out["buy_vol"] = seg_sum(amount * buy)
    out["buy_n"] = seg_sum(buy)
    out["notional"] = seg_sum(notional)
    out["signed_vol"] = seg_sum(amount * sign)
    out["rv_trade"] = seg_sum(dlogp**2)                 # 約定間の対数価格変化の二乗和（約定ベースの実現分散）
    out["flips"] = seg_sum(flip)                        # 売買の向きが切り替わった回数
    out["big_vol"] = 0.0                                # 大口: 足内で大きい順 10% の約定の出来高（後で埋める）
    # 大口の出来高: 各足の約定量の 90% 分位以上
    big = np.zeros(len(index))
    vw = np.zeros(len(index))
    for k in range(len(index)):
        a = amount[i0[k]:i1[k]]
        if len(a) >= 10:
            q = np.quantile(a, 0.9)
            big[k] = a[a >= q].sum()
            vw[k] = float((tape.price[i0[k]:i1[k]] * a).sum() / (a.sum() + EPS))
        elif len(a):
            vw[k] = float((tape.price[i0[k]:i1[k]] * a).sum() / (a.sum() + EPS))
    out["big_vol"] = big
    out["vwap"] = np.where(vw > 0, vw, np.nan)
    last_px = np.where(i1 > i0, tape.price[np.maximum(i1 - 1, 0)], np.nan)
    out["last"] = last_px
    return out


def flow_features(agg: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """集計 → 特徴量（水準に依存しない形）。close は同じ index の終値（VWAP との距離などに使う）。"""
    f: dict[str, pd.Series] = {}
    n, vol, bvol, bn, sv = agg["n"], agg["vol"], agg["buy_vol"], agg["buy_n"], agg["signed_vol"]
    r1 = np.log(close).diff()
    for w in (1, 4, 24, 168):
        rs = (lambda s: s.rolling(w, min_periods=w).sum()) if w > 1 else (lambda s: s)
        f[f"f_imb_vol{w}"] = (2 * rs(bvol) - rs(vol)) / (rs(vol) + EPS)          # 出来高の買い比率 −1〜1
        f[f"f_imb_n{w}"] = (2 * rs(bn) - rs(n)) / (rs(n) + EPS)                  # 件数の買い比率
    for w in (4, 24):
        f[f"f_imb_vol{w}_slope"] = f[f"f_imb_vol{w}"].diff(w)
    base_n, base_vol = n.rolling(168, min_periods=168).mean(), vol.rolling(168, min_periods=168).mean()
    for w in (1, 4, 24):
        f[f"f_intensity{w}"] = n.rolling(w, min_periods=w).mean() / (base_n + EPS)   # 約定数の相対的な多さ
        f[f"f_volratio{w}"] = vol.rolling(w, min_periods=w).mean() / (base_vol + EPS)
    size = vol / (n + EPS)
    f["f_size_ratio1"] = size / (size.rolling(168, min_periods=168).mean() + EPS)
    f["f_size_ratio24"] = size.rolling(24, min_periods=24).mean() / (size.rolling(168, min_periods=168).mean() + EPS)
    f["f_big_share1"] = agg["big_vol"] / (vol + EPS)
    f["f_big_share24"] = agg["big_vol"].rolling(24, min_periods=24).sum() / (vol.rolling(24, min_periods=24).sum() + EPS)
    sv_z_base = sv.rolling(168, min_periods=168).std()
    f["f_signed_z1"] = sv / (sv_z_base + EPS)
    f["f_signed_z24"] = sv.rolling(24, min_periods=24).sum() / (sv_z_base * np.sqrt(24) + EPS)
    f["f_flip_rate1"] = agg["flips"] / (n + EPS)                                    # 向きの切り替え頻度（低い = 一方向に連続）
    f["f_flip_rate24"] = agg["flips"].rolling(24, min_periods=24).sum() / (n.rolling(24, min_periods=24).sum() + EPS)
    rv_bar = r1**2
    f["f_rv_trade_ratio1"] = agg["rv_trade"] / (rv_bar.rolling(24, min_periods=24).mean() + EPS)    # 約定ベースの分散 ÷ 足の分散
    f["f_rv_trade24"] = np.sqrt(agg["rv_trade"].rolling(24, min_periods=24).sum())
    f["f_amihud24"] = (r1.abs() / (agg["notional"] + EPS)).rolling(24, min_periods=24).mean() * 1e9
    f["f_vwap_dist"] = (close - agg["vwap"]) / close
    f["f_vwap_dist24"] = (close - (agg["notional"].rolling(24, min_periods=24).sum() / (vol.rolling(24, min_periods=24).sum() + EPS))) / close
    # 価格インパクト（Kyle の λ の近似）: 直近 24 本の 1 時間リターンを符号付き出来高に回帰した傾き × 直近の符号付き出来高
    cov = r1.rolling(24, min_periods=24).cov(sv)
    var = sv.rolling(24, min_periods=24).var()
    lam = cov / (var + EPS)
    f["f_lambda24"] = lam * sv_z_base
    f["f_lambda_x_sv"] = lam * sv
    f["f_ret_sv_corr24"] = r1.rolling(24, min_periods=24).corr(sv)
    f["f_ret_sv_corr168"] = r1.rolling(168, min_periods=168).corr(sv)
    f["f_no_trades"] = (n == 0).astype(float)
    out = pd.DataFrame(f, index=agg.index)
    return out.replace([np.inf, -np.inf], np.nan)


def flow_table(tape: TradeTape, index: pd.DatetimeIndex, close: pd.Series) -> pd.DataFrame:
    return flow_features(hourly_aggregates(tape, index), close)
