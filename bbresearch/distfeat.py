"""収益率の分布のパラメータ推定と、その変化の特徴量（2026-09-23 追加、オーナー提案）。

15 分足の対数リターン r（signals.log_returns と同じ）について、各足の終了時点までの直近 k 本から
分布のパラメータを推定する。変化は「直近 k 本」と「その直前の k 本」（重ならない区間）の比較で表し、
平均の差は推定誤差で割った t 値にする（平均は短い窓ではほとんど推定できないため、差の大きさを
そのまま使わない）。

正規分布（k = 4, 16, 96）:
    dist_mu_t_k     平均の t 値 = μ̂_k / (σ̂_k / √k)
    dist_dmu_t_k    平均の変化の t 値 = (μ̂_k(t) − μ̂_k(t−k)) / √((σ̂_k(t)² + σ̂_k(t−k)²) / k)
    dist_dlogsd_k   標準偏差の変化 = ln(σ̂_k(t) / σ̂_k(t−k))
t 分布（k = 16, 96。自由度は尖度からのモーメント法: 超過尖度 κ に対し ν = 4 + 6/κ、κ ≤ 0 なら上限）:
    dist_tdf_k      自由度 ν̂_k（小さいほど裾が厚い）
    dist_dlogtdf_k  自由度の変化 = ln(ν̂_k(t) / ν̂_k(t−k))
利確バリアに先に届く理論確率（k = 16, 96。イベントごと）:
    dist_pup_k      収益がドリフト μ̂_k・分散 σ̂_k² のブラウン運動に従うとしたとき、上側バリア
                    （距離 a = ln(U/P_e)）に下側バリア（距離 b = −ln(L/P_e)）より先に届く確率
                    P = (1 − e^{θb}) / (e^{−θa} − e^{θb})、θ = 2μ/σ²（μ = 0 のとき b/(a+b)）。
                    時間切れのバリアは考慮しない。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .signals import log_returns

NORMAL_WINDOWS = (4, 16, 96)
T_WINDOWS = (16, 96)
TDF_MAX = 100.0


def t_df_from_kurtosis(excess_kurt):
    """t 分布の自由度のモーメント推定。超過尖度が 0 以下（正規分布より裾が薄い）なら上限。"""
    k = np.asarray(excess_kurt, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        nu = np.where(k > 0, 4.0 + 6.0 / k, TDF_MAX)
    return np.where(np.isnan(k), np.nan, np.minimum(nu, TDF_MAX))


def dist_bar_features(bars: pd.DataFrame, normal_windows=NORMAL_WINDOWS, t_windows=T_WINDOWS) -> pd.DataFrame:
    """各行が、その足の終了時点までのデータだけで計算された特徴量。"""
    r = log_returns(bars)
    out = pd.DataFrame(index=bars.index)
    for k in normal_windows:
        mu = r.rolling(k, min_periods=k).mean()
        sd = r.rolling(k, min_periods=k).std()
        mu_p, sd_p = mu.shift(k), sd.shift(k)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"dist_mu_t_{k}"] = mu / (sd / np.sqrt(k))
            out[f"dist_dmu_t_{k}"] = (mu - mu_p) / np.sqrt((sd**2 + sd_p**2) / k)
            out[f"dist_dlogsd_{k}"] = np.log(sd / sd_p)
        out[f"_mu_{k}"] = mu
        out[f"_sd_{k}"] = sd
    for k in t_windows:
        nu = pd.Series(t_df_from_kurtosis(r.rolling(k, min_periods=k).kurt()), index=bars.index)
        out[f"dist_tdf_{k}"] = nu
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"dist_dlogtdf_{k}"] = np.log(nu / nu.shift(k))
    return out.replace([np.inf, -np.inf], np.nan)


def prob_upper_first(mu, sigma, a, b):
    """ドリフト mu・拡散 sigma のブラウン運動が、+a に −b より先に届く確率（a, b > 0）。"""
    mu, sigma, a, b = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (mu, sigma, a, b)))
    out = np.full(mu.shape, np.nan)
    ok = np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0) & (a > 0) & (b > 0)
    theta = np.zeros(mu.shape)
    theta[ok] = 2.0 * mu[ok] / sigma[ok] ** 2
    small = ok & (np.abs(theta * np.maximum(a, b)) < 1e-9)
    out[small] = b[small] / (a[small] + b[small])
    big = ok & ~small
    th = np.clip(theta[big], -700.0 / np.maximum(a[big], b[big]), 700.0 / np.maximum(a[big], b[big]))
    num = -np.expm1(th * b[big])                      # 1 − e^{θb}
    den = np.exp(-th * a[big]) - np.exp(th * b[big])  # e^{−θa} − e^{θb}
    out[big] = num / den
    return np.clip(out, 0.0, 1.0)


def dist_event_features(events: pd.DataFrame, feats: pd.DataFrame, sigma: pd.Series, k_up: float, k_dn: float,
                        tick: float = 0.0, t_windows=T_WINDOWS) -> pd.DataFrame:
    """イベントごとの特徴量。feats は dist_bar_features の戻り値。index は event_id。

    バリアはラベルと同じく P_e を基準に U = P_e (1 + k_up σ_0)、L = P_e (1 − k_dn σ_0) とし、
    距離を対数で表す（P_e と呼値の丸めによる差は無視できる大きさなので、ここでは考慮しない）。
    """
    step = feats.index[1] - feats.index[0]
    bar_start = events["t0"] - step
    rows = feats.reindex(bar_start)
    s0 = sigma.reindex(bar_start).to_numpy(dtype=float)
    a = np.log1p(k_up * s0)
    with np.errstate(invalid="ignore"):
        b = -np.log1p(-k_dn * s0)
    out = rows[[c for c in rows.columns if not c.startswith("_")]].copy()
    for k in t_windows:
        out[f"dist_pup_{k}"] = prob_upper_first(rows[f"_mu_{k}"].to_numpy(), rows[f"_sd_{k}"].to_numpy(), a, b)
    out.index = events["event_id"].to_numpy()
    out.index.name = "event_id"
    return out
