"""予測の損失差の検定（評価の土台、候補 1。2026-09-28 オーナー決定）。

d_t = 損失(モデル) − 損失(基準) の平均が 0 かを検定する（Diebold & Mariano 1995 の形）。4 時間先・24 時間先を
1 時間ごとに出す予測では d_t の自己相関が強いので、長期分散の推定に頑健な方法を使う（先行研究の調査、付録 B）。

- fixed-b: Bartlett 核、帯域 M = ⌈1.3√P⌉（重み 1 − j/M）。臨界値は fixed-b の極限分布（Kiefer & Vogelsang 2005）。
  極限分布は長期分散によらないので、独立な正規乱数で同じ統計量を P 点で作り、その分布から p 値を出す。
- EWC: 余弦の重み B = ⌊0.4 P^{2/3}⌋ 個で長期分散を推定し、t 分布（自由度 B）で p 値を出す（Lazarus ほか 2018）。
- 定常ブートストラップ（Politis & Romano 1994。平均のブロック長は Politis & White 2004 の自動選択）。
M と B の決め方は Shin & Schor 2026（ForeComp）の推奨に合わせた。
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy import fft, stats


def nw_bandwidth(P: int, mult: float = 1.3) -> int:
    return int(math.ceil(mult * math.sqrt(P)))


def ewc_dof(P: int, mult: float = 0.4) -> int:
    return max(1, int(math.floor(mult * P ** (2.0 / 3.0))))


def _acov(e: np.ndarray, M: int) -> np.ndarray:
    """e（平均を引いたもの、形 (..., P)）の自己共分散 γ_0..γ_{M−1}（1/P で割る）。FFT で計算する。"""
    P = e.shape[-1]
    n = 1 << int(math.ceil(math.log2(2 * P)))
    f = fft.rfft(e, n=n, axis=-1)
    ac = fft.irfft(f * np.conj(f), n=n, axis=-1)[..., :M]
    return ac / P


def nw_lrv(d: np.ndarray, M: int) -> float:
    """Bartlett 核（重み 1 − j/M、j < M）の長期分散。"""
    e = np.asarray(d, float) - np.mean(d)
    g = _acov(e, M)
    w = 1.0 - np.arange(M) / M
    return float(g[0] + 2.0 * np.sum(w[1:] * g[1:]))


def fixedb_t(d: np.ndarray, M: int | None = None) -> tuple[float, int]:
    d = np.asarray(d, float)
    P = len(d)
    M = M or nw_bandwidth(P)
    lrv = nw_lrv(d, M)
    t = math.sqrt(P) * float(np.mean(d)) / math.sqrt(lrv) if lrv > 0 else float("nan")
    return t, M


@lru_cache(maxsize=16)
def fixedb_null(P: int, M: int, sims: int = 10000, seed: int = 0) -> np.ndarray:
    """独立な N(0,1) の P 点で作った fixed-b の t 統計量の分布（昇順）。極限分布の近似として使う。"""
    rng = np.random.default_rng(seed)
    out = []
    w = 1.0 - np.arange(M) / M
    batch = max(1, min(sims, int(4e6 // P)))
    done = 0
    while done < sims:
        k = min(batch, sims - done)
        d = rng.standard_normal((k, P))
        m = d.mean(axis=1, keepdims=True)
        g = _acov(d - m, M)
        lrv = g[:, 0] + 2.0 * (g[:, 1:] * w[1:]).sum(axis=1)
        out.append(math.sqrt(P) * m[:, 0] / np.sqrt(lrv))
        done += k
    return np.sort(np.concatenate(out))


def fixedb_pvalue(t: float, P: int, M: int, sims: int = 10000, seed: int = 0) -> float:
    """両側の p 値。"""
    if not np.isfinite(t):
        return float("nan")
    null = np.abs(fixedb_null(P, M, sims, seed))
    return float((1 + np.sum(null >= abs(t))) / (1 + len(null)))


def fixedb_cv975(b: float) -> float:
    """fixed-b（Bartlett）の両側 5% の臨界値の近似（Kiefer & Vogelsang 2005。Shin & Schor 2026 に掲載の式）。"""
    return 1.9600 + 2.9694 * b + 0.4160 * b**2 - 0.5324 * b**3


def ewc_t(d: np.ndarray, B: int | None = None) -> tuple[float, int]:
    """Λ_j = √(2/P) Σ_t cos(π j (t − ½)/P) d_t（j = 1..B）、長期分散 = Λ_j² の平均。t は自由度 B の t 分布に従う。"""
    d = np.asarray(d, float)
    P = len(d)
    B = B or ewc_dof(P)
    lam = fft.dct(d, type=2)[1:B + 1] / math.sqrt(2.0 * P)
    lrv = float(np.mean(lam**2))
    t = math.sqrt(P) * float(np.mean(d)) / math.sqrt(lrv) if lrv > 0 else float("nan")
    return t, B


def ewc_pvalue(t: float, B: int) -> float:
    return float(2.0 * stats.t.sf(abs(t), B)) if np.isfinite(t) else float("nan")


def sb_pvalue(d: np.ndarray, reps: int = 999, seed: int = 0) -> tuple[float, float]:
    """定常ブートストラップの両側 p 値（平均 0 の帰無）と、使った平均のブロック長。"""
    from arch.bootstrap import StationaryBootstrap, optimal_block_length

    d = np.asarray(d, float)
    block = float(optimal_block_length(d)["stationary"].iloc[0])
    block = max(1.0, block)
    bs = StationaryBootstrap(block, d, seed=seed)
    m = float(np.mean(d))
    means = np.array([float(np.mean(x[0][0])) for x in bs.bootstrap(reps)])
    p = float((1 + np.sum(np.abs(means - m) >= abs(m))) / (1 + reps))
    return p, block


def holm(pvals: list[float]) -> list[float]:
    """ホルム法で調整した p 値（同じ順序で返す）。"""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj.tolist()


def acf(d: np.ndarray, lags) -> dict[int, float]:
    e = np.asarray(d, float) - np.mean(d)
    g = _acov(e, max(lags) + 1)
    return {int(k): float(g[k] / g[0]) if g[0] > 0 else float("nan") for k in lags}


def hill(x: np.ndarray, share: float = 0.01) -> float:
    """|x| の上側の裾の指数（Hill 推定。上位 share の点を使う）。小さいほど裾が厚い（2 未満なら分散が有限と言いにくい）。"""
    a = np.sort(np.abs(np.asarray(x, float)))[::-1]
    k = max(10, int(len(a) * share))
    if len(a) <= k or a[k] <= 0:
        return float("nan")
    m = float(np.mean(np.log(a[:k] / a[k])))
    return float(1.0 / m) if m > 0 else float("nan")


def adf_pvalue(d: np.ndarray) -> float:
    from arch.unitroot import ADF
    from arch.utility.exceptions import InfeasibleTestException

    try:
        return float(ADF(np.asarray(d, float), trend="c").pvalue)
    except InfeasibleTestException:     # ほとんど一定の系列（例: どちらのモデルも一度も買わない損益の差）
        return float("nan")


def compare(d: np.ndarray, nw_mult: float = 1.3, ewc_mult: float = 0.4, fixedb_sims: int = 10000, seed: int = 0,
            bootstrap_reps: int = 0, acf_lags=(1, 4, 24, 168), hill_share: float = 0.01, adf: bool = False) -> dict:
    """損失差 d_t の平均と検定の一式。bootstrap_reps = 0 ならブートストラップを省く（帰無の監査では省く）。"""
    d = np.asarray(d, float)
    P = len(d)
    M = nw_bandwidth(P, nw_mult)
    B = ewc_dof(P, ewc_mult)
    if np.ptp(d) == 0:                  # 差が一定（例: 両方のモデルが一度も買わない）。検定はしない
        nan = float("nan")
        return {"n": P, "mean": float(np.mean(d)), "M": M, "b": M / P, "t_fb": nan, "p_fb": nan, "cv975_fb": fixedb_cv975(M / P),
                "B": B, "t_ewc": nan, "p_ewc": nan, "acf": {int(k): nan for k in acf_lags if k < P}, "hill": nan,
                **({"p_sb": nan, "sb_block": nan} if bootstrap_reps else {}), **({"adf_p": nan} if adf else {})}
    t_fb, _ = fixedb_t(d, M)
    t_ewc, _ = ewc_t(d, B)
    out = {"n": P, "mean": float(np.mean(d)), "M": M, "b": M / P, "t_fb": t_fb,
           "p_fb": fixedb_pvalue(t_fb, P, M, fixedb_sims, seed), "cv975_fb": fixedb_cv975(M / P),
           "B": B, "t_ewc": t_ewc, "p_ewc": ewc_pvalue(t_ewc, B),
           "acf": acf(d, [k for k in acf_lags if k < P]), "hill": hill(d, hill_share)}
    if bootstrap_reps:
        out["p_sb"], out["sb_block"] = sb_pvalue(d, bootstrap_reps, seed)
    if adf:
        out["adf_p"] = adf_pvalue(d)
    return out
