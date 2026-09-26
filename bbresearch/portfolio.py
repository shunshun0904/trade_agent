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
- 目標ボラの推定窓（vol_days）: 銘柄の選択は est_days（365 日）のまま、縮める比率に使う Σ だけ直近 vol_days 日
  （例 90 日）で推定し直す。急変への追随を速めるため（2026-09-27 オーナー決定）。None なら est_days と同じ。
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


def strategy_name(m, v, vd=None, cf=None, tr=None) -> str:
    return (("minvar" if m is None else f"minvar_x{m}") + ("" if v is None else f"_vt{int(round(v * 100))}")
            + ("" if vd is None or v is None else f"_w{vd}") + ("" if cf is None or v is None else f"_c{cf}")
            + ("" if tr is None or v is None else f"_t{tr[0]}s{int(round(tr[1] * 100))}"))


def trend_on(close: pd.Series, t0, days: int) -> bool | None:
    """t0 より前の最後の終値が、その日までの days 日単純移動平均以上なら True（上昇トレンド）。データ不足なら None。"""
    past = close[close.index < t0].dropna()
    if len(past) < days:
        return None
    return bool(past.iloc[-1] >= past.iloc[-days:].mean())


def _segments(d0: pd.Timestamp, d1: pd.Timestamp, index: pd.DatetimeIndex, every_days: int | None) -> list[tuple]:
    """[d0, d1) を every_days 日ごとに区切る（None なら区切らない）。各区切りの開始は index にある日。"""
    if every_days is None:
        return [(d0, d1)]
    days = index[(index >= d0) & (index < d1)]
    bounds = [days[i] for i in range(0, len(days), every_days)]
    return [(b, bounds[i + 1] if i + 1 < len(bounds) else d1) for i, b in enumerate(bounds)]


def walk_forward(close: pd.DataFrame, start, k_max: int = 5, w_max: float = 0.4,
                 target_mults: tuple = (None, 1.0, 1.5), est_days: int = 365, cost: float = 0.0015,
                 benchmark: str = "btc_jpy", caps: dict[str, float] | None = None,
                 vol_targets: tuple = (None,), vol_days: tuple = (None,),
                 cash_freq_days: tuple = (None,), trends: tuple = (None,)) -> WalkForwardResult:
    """close は日足の終値（列 = ペア、index = 日付）。毎月初に直前 est_days 日で推定し、翌月を保有する。

    vol_targets の各 v（年率）について、推定ボラが v を超えるときは暗号資産の比率を v / 推定ボラ に落とし、残りを JPY で持つ。
    cash_freq_days に 7 を入れると、銘柄と相対の重みは月 1 回のまま、JPY に縮める比率だけ 7 日ごとに見直す
    （2026-09-27 オーナー決定）。None は月 1 回。
    trends の各要素は None か (days, scale)。(days, scale) は、区切りの開始時点で benchmark（BTC）の終値が days 日移動平均を
    下回っていれば暗号資産の重みを scale 倍にする（残りは JPY。scale=0 で全額 JPY）。判定は開始日より前の終値だけを使う。
    目標ボラ付きの戦略にだけ掛ける（2026-09-27 オーナー指示のトレンドフィルタ）。
    """
    ret = np.log(close).diff()
    starts = _month_starts(close.index, start)
    names = {f"minvar_x{m}" if m is not None else "minvar": m for m in target_mults}
    combos = [(m, v, vd, cf, tr) for m in target_mults for v in vol_targets
              for vd in (vol_days if v is not None else (None,)) for cf in (cash_freq_days if v is not None else (None,))
              for tr in (trends if v is not None else (None,))]
    strategies = list(dict.fromkeys(strategy_name(*c) for c in combos)) + ["equal", benchmark]
    daily = {s: pd.Series(0.0, index=close.index[close.index >= starts[0]]) for s in strategies}
    weights = {s: [] for s in strategies}
    prev_w = {s: pd.Series(dtype=float) for s in strategies}

    def apply(s: str, w: pd.Series, s0, s1) -> None:
        """[s0, s1) を重み w で保有し、初日に回転率 × cost を引く。"""
        w = w.reindex(close.columns).fillna(0.0)
        hold = ret[(ret.index >= s0) & (ret.index < s1)]
        turnover = float((w - prev_w[s].reindex(close.columns).fillna(0.0)).abs().sum())
        r = (hold[w.index].fillna(0.0) * w).sum(axis=1)
        if len(r):
            r.iloc[0] -= turnover * cost
        daily[s].loc[r.index] = r
        weights[s].append({"date": pd.Timestamp(s0).strftime("%Y-%m-%d"),
                           "weights": {k: round(float(v), 4) for k, v in w[w > 1e-6].items()},
                           "cash": round(float(max(0.0, 1.0 - w.sum())), 4)})
        prev_w[s] = w

    def vol_scale(wv: np.ndarray, avail: list[str], t0, v: float, vd, cov_full: np.ndarray) -> float:
        """t0 より前の直近 vd 日（None なら est_days 日）で推定した年率ボラに対する縮め率。"""
        days = est_days if vd is None else min(vd, est_days)
        win = ret[(ret.index < t0) & (ret.index >= t0 - pd.Timedelta(days=days))][avail].dropna()
        cov_v = shrunk_cov(win.to_numpy()) if len(win) >= 20 else cov_full
        est_vol = float(np.sqrt(wv @ cov_v @ wv * DAYS_PER_YEAR))
        return 1.0 if est_vol <= v else v / est_vol

    for j, d0 in enumerate(starts):
        d1 = starts[j + 1] if j + 1 < len(starts) else close.index[-1] + pd.Timedelta(days=1)
        est = ret[(ret.index < d0) & (ret.index >= d0 - pd.Timedelta(days=est_days))]
        avail = [c for c in close.columns if est[c].notna().sum() >= est_days * 0.9]
        if len(avail) < 3:
            continue
        R = est[avail].dropna().to_numpy()
        cov, mu = shrunk_cov(R), shrunk_mean(R)
        mu_ew = float(mu.mean())
        for s, m in names.items():
            tgt = None if m is None else mu_ew * m
            best = best_combination(cov, mu, avail, k_max, tgt, w_max, caps)
            if best is None:  # 目標に届かない月は均等配分に落とす
                w_r = pd.Series(1.0 / len(avail), index=avail)
            else:
                w_r = pd.Series(best["weights"], index=best["names"])
            wv = w_r.reindex(avail).fillna(0.0).to_numpy()
            for v in vol_targets:
                if v is None:
                    apply(strategy_name(m, None), w_r, d0, d1)
                    continue
                for vd in vol_days:
                    for cf in cash_freq_days:
                        for tr in trends:
                            name = strategy_name(m, v, vd, cf, tr)
                            for s0, s1 in _segments(d0, d1, close.index, cf):
                                sc = vol_scale(wv, avail, s0, v, vd, cov)
                                if tr is not None and trend_on(close[benchmark], s0, tr[0]) is False:
                                    sc *= tr[1]
                                apply(name, w_r * sc, s0, s1)  # 残りは JPY
        apply("equal", pd.Series(1.0 / len(avail), index=avail), d0, d1)
        apply(benchmark, pd.Series({benchmark: 1.0}), d0, d1)
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


def yearly(r: pd.Series) -> pd.DataFrame:
    """年ごとのリターン（その年の累積。年率化しない）、年率ボラ、その年の中での最大ドローダウン。"""
    r = r.dropna()
    rows = []
    for y, g in r.groupby(r.index.year):
        eq = np.exp(g.cumsum())
        rows.append({"year": int(y), "days": int(len(g)), "return": float(np.exp(g.sum()) - 1),
                     "ann_vol": float(g.std(ddof=1) * np.sqrt(DAYS_PER_YEAR)) if len(g) > 1 else None,
                     "max_drawdown": float((eq / eq.cummax() - 1).min())})
    return pd.DataFrame(rows).set_index("year")


def current_weights(close: pd.DataFrame, k_max: int = 5, w_max: float = 0.4, caps: dict[str, float] | None = None,
                    target_vol: float | None = 0.30, est_days: int = 365) -> dict:
    """直近 est_days 日で推定した今の重み（ペア → 割合）と JPY の割合、推定ボラ。scripts/rebalance.py が使う。"""
    ret = np.log(close).diff()
    est = ret[ret.index >= close.index[-1] - pd.Timedelta(days=est_days)]
    avail = [c for c in close.columns if est[c].notna().sum() >= est_days * 0.9]
    if len(avail) < 3:
        raise ValueError("候補が 3 銘柄に満たない")
    R = est[avail].dropna().to_numpy()
    cov, mu = shrunk_cov(R), shrunk_mean(R)
    best = best_combination(cov, mu, avail, k_max, None, w_max, caps)
    if best is None:
        raise ValueError("実行可能な組み合わせがない")
    wv = pd.Series(best["weights"], index=best["names"]).reindex(avail).fillna(0.0).to_numpy()
    vol = float(np.sqrt(wv @ cov @ wv * DAYS_PER_YEAR))
    scale = 1.0 if target_vol is None or vol <= target_vol else target_vol / vol
    return {"weights": {n: float(w) * scale for n, w in zip(best["names"], best["weights"])},
            "cash": 1.0 - scale, "est_vol": vol * scale, "est_vol_full": vol, "as_of": str(close.index[-1].date())}
