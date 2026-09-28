"""帰無の監査用の合成データ（評価の土台、候補 1。2026-09-28 オーナー決定）。

期待リターンが 0 で、ボラの動きと裾の厚さだけを BTC/JPY の学習期間（2021〜2023 年）に合わせた価格の系列を作り、
実データと同じ手順（特徴量 → 各モデルの学習 → 評価）を走らせる。向きの情報がないデータで「向きの上乗せ」がどれだけ出るかの
分布（帰無の分布）を作り、実データの値がその外にあるかを見る。手順に先読みがあれば、ここで上乗せが出る。

- garch_t: 1 時間のリターン R_t = σ_t ε_t、ε は分散 1 の t 分布、σ²_{t+1} = ω + α R_t² + β σ²_t（学習期間に当てはめた値）。
- ms2: ボラの 2 状態（マルコフ切り替え、状態ごとに正規分布。学習期間に当てはめた値）。
1 時間の中の 1 分ごとの動きは、終点を R_t に固定したブラウン橋（1 分の分散 σ_t²/60）。高値・安値・1 分の終値はこの経路から作る。
出来高は log(出来高) = a + b log(足の値幅 log(高値/安値)) + u_t、u は AR(1)（学習期間に当てはめた値）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bbresearch.distbase import fit_garch_t

MINUTES = 60


def fit_ms2(r: np.ndarray, seed: int = 0) -> dict:
    """平均 0、分散が 2 状態で切り替わるモデル（statsmodels の MarkovRegression）。100 倍して推定し、元の尺度で返す。"""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    mod = MarkovRegression(100 * np.asarray(r, float), k_regimes=2, trend="n", switching_variance=True)
    np.random.seed(seed)
    res = mod.fit(disp=False, search_reps=5)
    p = dict(zip(mod.param_names, np.asarray(res.params, float)))
    s2 = np.array([p["sigma2[0]"], p["sigma2[1]"]]) / 1e4
    P = np.array([[p["p[0->0]"], 1 - p["p[0->0]"]], [p["p[1->0]"], 1 - p["p[1->0]"]]])
    lo = int(np.argmin(s2))
    order = [lo, 1 - lo]   # 状態 0 を低ボラにそろえる
    return {"sigma": np.sqrt(s2[order]).tolist(), "P": P[np.ix_(order, order)].tolist(), "loglik": float(res.llf)}


def fit_volume(h1: pd.DataFrame) -> dict:
    """log(出来高) を log(値幅) に回帰し、残差を AR(1) とみなす。"""
    rng_ = np.log(h1["high"] / h1["low"])
    ok = (h1["volume"] > 0) & (rng_ > 0)
    x, y = np.log(rng_[ok].to_numpy()), np.log(h1["volume"][ok].to_numpy())
    b, a = np.polyfit(x, y, 1)
    u = y - (a + b * x)
    phi = float(np.corrcoef(u[1:], u[:-1])[0, 1])
    return {"a": float(a), "b": float(b), "phi": phi, "sd": float(u.std() * np.sqrt(max(1e-6, 1 - phi**2)))}


MAX_PERSISTENCE = 0.995   # 合成データの GARCH の α + β の上限


def stationary_garch(g: dict, var: float, cap: float = MAX_PERSISTENCE) -> dict:
    """合成データ用の GARCH の数値。α + β が cap を超えるときは、α : β の比を保って cap に縮める。
    ω は無条件分散が学習期間の分散 var に合うように決める（分散のターゲティング）。α + β = 1 のまま作ると
    分散が平均に戻らず、系列が現実から離れる（2026-09-28 の check で尖度 93.5、実データは 14.8）。"""
    a, b = g["alpha"], g["beta"]
    phi = a + b
    if phi > cap:
        a, b = a * cap / phi, b * cap / phi
    return {**g, "alpha": a, "beta": b, "omega": var * (1 - a - b), "fitted_alpha": g["alpha"], "fitted_beta": g["beta"],
            "fitted_omega": g["omega"]}


def calibrate(h1: pd.DataFrame, split, garch: dict | None = None, seed: int = 0) -> dict:
    """学習期間（split より前）の 1 時間足から、合成データの数値を決める。"""
    split = pd.Timestamp(split, tz="UTC") if isinstance(split, str) else split
    tr = h1[h1.index < split]
    r = np.log(tr["close"]).diff().dropna().to_numpy()
    g = dict(garch) if garch else fit_garch_t(r)
    return {"garch": stationary_garch(g, float(r.var())), "ms2": fit_ms2(r, seed), "volume": fit_volume(tr),
            "p0": float(h1["close"].iloc[0]), "train_sd": float(r.std()), "train_kurt": float(pd.Series(r).kurt())}


def _garch_path(g: dict, n: int, rng: np.random.Generator, burn: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    nu = g["nu"]
    eps = rng.standard_t(nu, size=n + burn) * np.sqrt((nu - 2) / nu)
    om, a, b = g["omega"], g["alpha"], g["beta"]
    s2 = np.empty(n + burn)
    R = np.empty(n + burn)
    prev = om / max(1e-12, 1 - a - b)
    for t in range(n + burn):
        s2[t] = prev
        R[t] = np.sqrt(prev) * eps[t]
        prev = om + a * R[t] ** 2 + b * prev
    return np.sqrt(s2[burn:]), R[burn:]


def _ms2_path(m: dict, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(m["P"])
    pi0 = P[1, 0] / (P[0, 1] + P[1, 0])          # 定常分布での状態 0 の確率
    u = rng.random(n)
    state = np.empty(n, dtype=int)
    state[0] = 0 if u[0] < pi0 else 1
    stay = np.diag(P)
    for t in range(1, n):
        s = state[t - 1]
        state[t] = s if u[t] < stay[s] else 1 - s
    sigma = np.asarray(m["sigma"])[state]
    return sigma, sigma * rng.standard_normal(n)


def simulate(index: pd.DatetimeIndex, kind: str, cal: dict, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """期待リターン 0 の合成データ。戻り値は 1 時間足（open, high, low, close, volume）と 1 分の終値。"""
    rng = np.random.default_rng(seed)
    n = len(index)
    if kind == "garch_t":
        sigma, R = _garch_path(cal["garch"], n, rng)
    elif kind == "ms2":
        sigma, R = _ms2_path(cal["ms2"], n, rng)
    else:
        raise ValueError(kind)
    steps = np.arange(1, MINUTES + 1) / MINUTES
    e = rng.standard_normal((n, MINUTES)) * (sigma[:, None] / np.sqrt(MINUTES))
    path = np.cumsum(e, axis=1)
    path = path - steps[None, :] * path[:, -1:] + steps[None, :] * R[:, None]   # 終点を R_t にしたブラウン橋
    log_open = np.log(cal["p0"]) + np.r_[0.0, np.cumsum(R)[:-1]]
    log_close = log_open + R                     # 1 分の経路の終点と同じ計算にして、高値・安値との大小を丸めでずらさない
    minute_log = log_open[:, None] + path
    high = np.maximum(log_open, minute_log.max(axis=1))
    low = np.minimum(log_open, minute_log.min(axis=1))
    v = cal["volume"]
    u = np.empty(n)
    u[0] = rng.normal(0, v["sd"] / np.sqrt(max(1e-6, 1 - v["phi"] ** 2)))
    shocks = rng.normal(0, v["sd"], n)
    for t in range(1, n):
        u[t] = v["phi"] * u[t - 1] + shocks[t]
    rng_ = np.maximum(high - low, 1e-6)
    volume = np.exp(v["a"] + v["b"] * np.log(rng_) + u)
    h1 = pd.DataFrame({"open": np.exp(log_open), "high": np.exp(high), "low": np.exp(low), "close": np.exp(log_close),
                       "volume": volume}, index=index)
    m_index = pd.DatetimeIndex((index.as_unit("ns").asi8[:, None] + np.arange(MINUTES)[None, :] * 60_000_000_000).ravel(),
                               tz="UTC")
    return h1, pd.Series(np.exp(minute_log.ravel()), index=m_index)


def path_stats(h1: pd.DataFrame) -> dict:
    """合成データと実データを比べる要約（1 時間の対数リターンの標準偏差・尖度、二乗の自己相関、出来高と値幅の相関）。"""
    r = np.log(h1["close"]).diff().dropna()
    r2 = r**2
    out = {"sd": float(r.std()), "kurt": float(r.kurt()), "mean": float(r.mean())}
    for k in (1, 24, 168):
        out[f"acf_r2_{k}"] = float(r2.autocorr(k))
    rng_ = np.log(h1["high"] / h1["low"])
    ok = (h1["volume"] > 0) & (rng_ > 0)
    out["corr_logvol_logrange"] = float(np.corrcoef(np.log(h1["volume"][ok]), np.log(rng_[ok]))[0, 1])
    return out
