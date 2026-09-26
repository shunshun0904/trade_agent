"""ボラティリティを抑えるポートフォリオ（2026-09-27 オーナー指示）。

オーナー決定: 目標リターン付きの最小分散、銘柄数の上限を決めて組み合わせを探索、日足・毎月再計算の前向き検証。

- 共分散は Ledoit-Wolf の縮小推定、期待リターンは標本平均を横断平均へ 50% 縮める（推定誤差を抑える）。
- 目標リターンは絶対値ではなく「候補全体の均等配分の期待リターン × 倍率」で与える（倍率 1.0 なら均等配分と同じ
  期待リターンで分散だけ下げる）。None なら目標なしの最小分散。
- 組み合わせ: 候補から k_max 銘柄以下のすべての組み合わせについて QP（SLSQP、ロング、上限 w_max）を解き、
  分散が最小のものを選ぶ。
- 前向き検証: 毎月初に、その時点より前の est_days 日で推定して翌月を保有する。売買費用は回転率 × cost。
- 上限は銘柄ごとに変えられる（caps。2026-09-27 オーナー決定: BTC 60%、他 40%）。
- 目標ボラ（vol_targets）: 推定ボラ sqrt(w'Σw × 365) が目標を超えるときは、その比率だけ暗号資産を減らして残りを
  JPY（リターン 0）で持つ。銘柄の選択と相対の重みは変えない（2026-09-27 オーナー決定）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365


def shrunk_cov(R: np.ndarray) -> np.ndarray:
    from sklearn.covariance import LedoitWolf

    return LedoitWolf().fit(R).covariance_


def shrunk_mean(R: np.ndarray, shrink: float = 0.5) -> np.ndarray:
    m = R.mean(axis=0)
    return (1 - shrink) * m + shrink * m.mean()


def min_var_weights(cov: np.ndarray, mu: np.ndarray | None = None, target: float | None = None,
                    w_max=0.4) -> tuple[np.ndarray, float] | None:
    """ロング・合計 1・上限 w_max（数値か銘柄ごとの配列）・（あれば）mu·w ≥ target の最小分散。実行不能なら None。"""
    from scipy.optimize import minimize

    n = len(cov)
    caps = np.full(n, float(w_max)) if np.isscalar(w_max) else np.asarray(w_max, dtype=float)
    if caps.sum() < 1 - 1e-9:
        return None
    if n == 1:
        w = np.array([1.0])
        if target is not None and mu is not None and mu @ w < target - 1e-12:
            return None
        return w, float(w @ cov @ w)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}]
    if target is not None and mu is not None:
        if mu.max() < target - 1e-12:  # 上限がなくても届かない
            return None
        cons.append({"type": "ineq", "fun": lambda w: mu @ w - target, "jac": lambda w: mu})
    x0 = np.full(n, 1.0 / n)
    x0 = np.minimum(x0, caps)
    x0 /= x0.sum()
    res = minimize(lambda w: w @ cov @ w, x0, jac=lambda w: 2 * cov @ w, method="SLSQP",
                   bounds=[(0.0, float(c)) for c in caps], constraints=cons, options={"maxiter": 200, "ftol": 1e-12})
    w = np.clip(res.x, 0.0, caps)
    w /= w.sum()
    if not res.success and abs(w.sum() - 1) > 1e-6:
        return None
    if target is not None and mu is not None and mu @ w < target - 1e-6:
        return None
    return w, float(w @ cov @ w)


def best_combination(cov: np.ndarray, mu: np.ndarray, names: list[str], k_max: int, target: float | None,
                     w_max: float = 0.4, caps: dict[str, float] | None = None) -> dict | None:
    """k_max 銘柄以下のすべての組み合わせから分散最小のものを選ぶ。caps は銘柄ごとの上限（なければ w_max）。

    重みが 0 の銘柄は結果から除く。
    """
    n = len(names)
    cap = np.array([(caps or {}).get(nm, w_max) for nm in names], dtype=float)
    k_min = int(np.ceil(1.0 / cap.max() - 1e-9))
    best = None
    for k in range(k_min, min(k_max, n) + 1):
        for idx in combinations(range(n), k):
            ii = list(idx)
            if cap[ii].sum() < 1 - 1e-9:
                continue
            sol = min_var_weights(cov[np.ix_(ii, ii)], mu[ii], target, cap[ii])
            if sol is None:
                continue
            w, var = sol
            if best is None or var < best["var"] - 1e-15:
                keep = w > 1e-6
                best = {"var": var, "names": [names[i] for i, k_ in zip(ii, keep) if k_], "weights": w[keep] / w[keep].sum()}
    return best


@dataclass
class WalkForwardResult:
    daily: pd.DataFrame                       # 戦略ごとの日次リターン（費用込み）
    weights: dict[str, list[dict]] = field(default_factory=dict)  # 戦略 → 月ごとの {date, weights}


def _month_starts(index: pd.DatetimeIndex, start) -> list[pd.Timestamp]:
    start = pd.Timestamp(start)
    if start.tzinfo is None and index.tz is not None:
        start = start.tz_localize(index.tz)
    idx = index[index >= start]
    return [d for i, d in enumerate(idx) if i == 0 or d.month != idx[i - 1].month]


def strategy_name(m, v) -> str:
    return ("minvar" if m is None else f"minvar_x{m}") + ("" if v is None else f"_vt{int(round(v * 100))}")


def walk_forward(close: pd.DataFrame, start, k_max: int = 5, w_max: float = 0.4,
                 target_mults: tuple = (None, 1.0, 1.5), est_days: int = 365, cost: float = 0.0015,
                 benchmark: str = "btc_jpy", caps: dict[str, float] | None = None,
                 vol_targets: tuple = (None,)) -> WalkForwardResult:
    """close は日足の終値（列 = ペア、index = 日付）。毎月初に直前 est_days 日で推定し、翌月を保有する。

    vol_targets の各 v（年率）について、推定ボラが v を超える月は暗号資産の比率を v / 推定ボラ に落とし、残りを JPY で持つ。
    """
    ret = np.log(close).diff()
    starts = _month_starts(close.index, start)
    names = {f"minvar_x{m}" if m is not None else "minvar": m for m in target_mults}
    strategies = [strategy_name(m, v) for m in target_mults for v in vol_targets] + ["equal", benchmark]
    daily = {s: pd.Series(0.0, index=close.index[close.index >= starts[0]]) for s in strategies}
    weights = {s: [] for s in strategies}
    prev_w = {s: pd.Series(dtype=float) for s in strategies}
    for j, d0 in enumerate(starts):
        d1 = starts[j + 1] if j + 1 < len(starts) else close.index[-1] + pd.Timedelta(days=1)
        est = ret[(ret.index < d0) & (ret.index >= d0 - pd.Timedelta(days=est_days))]
        avail = [c for c in close.columns if est[c].notna().sum() >= est_days * 0.9]
        if len(avail) < 3:
            continue
        R = est[avail].dropna().to_numpy()
        cov, mu = shrunk_cov(R), shrunk_mean(R)
        mu_ew = float(mu.mean())
        hold = ret[(ret.index >= d0) & (ret.index < d1)]
        chosen: dict[str, pd.Series] = {}
        for s, m in names.items():
            tgt = None if m is None else mu_ew * m
            best = best_combination(cov, mu, avail, k_max, tgt, w_max, caps)
            if best is None:  # 目標に届かない月は均等配分に落とす
                w_r = pd.Series(1.0 / len(avail), index=avail)
            else:
                w_r = pd.Series(best["weights"], index=best["names"])
            wv = w_r.reindex(avail).fillna(0.0).to_numpy()
            est_vol = float(np.sqrt(wv @ cov @ wv * DAYS_PER_YEAR))
            for v in vol_targets:
                scale = 1.0 if v is None or est_vol <= v else v / est_vol
                chosen[strategy_name(m, v)] = w_r * scale  # 残り 1 − scale は JPY
        chosen["equal"] = pd.Series(1.0 / len(avail), index=avail)
        chosen[benchmark] = pd.Series({benchmark: 1.0})
        for s, w in chosen.items():
            w = w.reindex(close.columns).fillna(0.0)
            turnover = float((w - prev_w[s].reindex(close.columns).fillna(0.0)).abs().sum())
            # 月内は重みを固定した日次リバランスとみなす（単純和）。初日に費用を引く
            r = (hold[w.index].fillna(0.0) * w).sum(axis=1)
            if len(r):
                r.iloc[0] -= turnover * cost
            daily[s].loc[r.index] = r
            weights[s].append({"date": d0.strftime("%Y-%m-%d"), "weights": {k: round(float(v), 4) for k, v in w[w > 1e-6].items()},
                               "cash": round(float(max(0.0, 1.0 - w.sum())), 4)})
            prev_w[s] = w
    return WalkForwardResult(daily=pd.DataFrame(daily), weights=weights)


def summary(r: pd.Series) -> dict:
    r = r.dropna()
    eq = np.exp(r.cumsum())
    dd = (eq / eq.cummax() - 1).min()
    years = len(r) / DAYS_PER_YEAR
    return {"days": int(len(r)), "ann_return": float(np.exp(r.sum()) ** (1 / years) - 1) if years > 0 else None,
            "ann_vol": float(r.std(ddof=1) * np.sqrt(DAYS_PER_YEAR)),
            "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(DAYS_PER_YEAR)) if r.std(ddof=1) > 0 else None,
            "max_drawdown": float(dd), "worst_day": float(r.min()), "total": float(np.exp(r.sum()) - 1)}
