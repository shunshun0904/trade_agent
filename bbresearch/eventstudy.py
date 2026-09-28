"""候補 2（24 時間の新しい情報源）の事象研究。2026-09-28 オーナー決定の事前登録（docs/SPEC.md の「24 時間の新しい情報源」の節）。

- 判断の時刻 T: 0・8・16 時 UTC。信号は T に分かっている最新の値（古さの上限つき）。
- 結果: T の bitbank の終値（足 [T − 1h, T) の終値）から T + 24h の終値までの対数収益率。下側の裾の仮説では、それが HAR 型の
  5% 点を下回ったか（0 か 1）。
- 事象: 信号の順位が q 以下（下位）か 1 − q 以上（上位）。事象として数えた時刻から 24 時間の間の事象は数えない（結果が重ならない）。
- 統制: 同じ側の事象が前後 24 時間にない時刻（結果の期間が事象と重ならない）。直前 24 時間の値動き z の区分ごとに統制の平均を取り、
  事象の区分の割合で重みを付けて比べる（直前の値動きをそろえる）。
- 帰無: 信号の列を 30 日以上ずらした循環シフトで同じ計算をする。事象の固まり方と値動きの時系列の性質はそのまま残る。
- HAR 型（候補 1 と同じ: 1 分の実現分散、直近 1・4・24・168 本、時刻・曜日）は、毎月初めにその時点までに結果が出そろった行だけで
  当てはめ直す（拡大窓）。分位点は σ̂ × 標準化した残差の経験分位点。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bbresearch.distbase import HARVol, har_features, har_target

HOUR = pd.Timedelta(hours=1)


def _utc(x) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def grid_times(start, end, hours=(0, 8, 16), horizon: str = "24h") -> pd.DatetimeIndex:
    """[start, end) の判断の時刻のうち、T + horizon の終値が end までに分かるもの。"""
    s, e = _utc(start), _utc(end)
    t = pd.date_range(s.normalize(), e, freq="1h", inclusive="left")
    t = t[t.hour.isin(list(hours)) & (t >= s)]
    return t[t + pd.Timedelta(horizon) <= e]


def value_at(x: pd.Series, times: pd.DatetimeIndex, tolerance: str) -> np.ndarray:
    """各時刻 T に分かっている最新の値（index が T 以下で、T からの古さが tolerance 以内）。x の index は値が分かる時刻。"""
    x = x.dropna()
    x = x[~x.index.duplicated()].sort_index()
    if len(x) == 0:
        return np.full(len(times), np.nan)
    return x.reindex(times, method="ffill", tolerance=pd.Timedelta(tolerance)).to_numpy(float)


def forward_log_return(close_known: pd.Series, times: pd.DatetimeIndex, horizon: str = "24h") -> np.ndarray:
    """close_known の index は終値が分かる時刻（足が閉じる時刻）。T から T + horizon までの対数収益率。"""
    lc = np.log(close_known)
    return lc.reindex(times + pd.Timedelta(horizon)).to_numpy() - lc.reindex(times).to_numpy()


def prior_z(close_known: pd.Series, times: pd.DatetimeIndex, horizon: str = "24h", vol_hours: int = 720,
            min_hours: int = 480) -> np.ndarray:
    """直前 horizon の対数収益率 ÷（直前 vol_hours 本の 1 時間の対数収益率の標準偏差 × √(horizon の時間数)）。T までの値だけ。"""
    lc = np.log(close_known)
    k = pd.Timedelta(horizon) / HOUR
    sd = lc.diff().rolling(vol_hours, min_periods=min_hours).std() * np.sqrt(k)
    past = lc.reindex(times).to_numpy() - lc.reindex(times - pd.Timedelta(horizon)).to_numpy()
    return past / sd.reindex(times).to_numpy()


def zbins(z: np.ndarray, edges) -> np.ndarray:
    """z の区分（0〜len(edges)）。区分 i は edges[i−1] ≤ z < edges[i]。z が NaN なら −1。"""
    b = np.digitize(z, np.asarray(edges, float))
    return np.where(np.isfinite(z), b, -1)


def event_mask(rank: np.ndarray, q: float, side: str) -> np.ndarray:
    if side == "low":
        return np.isfinite(rank) & (rank <= q)
    if side == "high":
        return np.isfinite(rank) & (rank >= 1 - q)
    raise ValueError(side)


def select_episodes(ev: np.ndarray, t_ns: np.ndarray, gap_ns: int) -> np.ndarray:
    """事象を時刻の順に見て、前に数えた事象から gap 以上離れたものだけ数える。"""
    sel = np.zeros(len(ev), bool)
    last = None
    for i in np.flatnonzero(ev):
        if last is None or t_ns[i] - last >= gap_ns:
            sel[i] = True
            last = t_ns[i]
    return sel


def control_mask(ev: np.ndarray, defined: np.ndarray, t_ns: np.ndarray, gap_ns: int) -> np.ndarray:
    """信号があり、同じ側の事象が前後 gap 未満にない時刻。"""
    et = t_ns[ev]
    if len(et) == 0:
        return defined.copy()
    pos = np.searchsorted(et, t_ns)
    left = np.where(pos > 0, t_ns - et[np.maximum(pos - 1, 0)], np.iinfo(np.int64).max)
    right = np.where(pos < len(et), et[np.minimum(pos, len(et) - 1)] - t_ns, np.iinfo(np.int64).max)
    near = (left < gap_ns) | (right < gap_ns)
    return defined & ~ev & ~near


def matched_effect(y: np.ndarray, sel: np.ndarray, ctrl: np.ndarray, zb: np.ndarray, n_bins: int) -> dict:
    """事象の平均 − 統制の平均（z の区分ごとの統制の平均を、事象の区分の割合で重み付け）。統制のない区分の事象は除く。"""
    ok = np.isfinite(y) & (zb >= 0)
    ev, cm = sel & ok, ctrl & ok
    ev_b = np.bincount(zb[ev], minlength=n_bins)
    c_b = np.bincount(zb[cm], minlength=n_bins)
    c_sum = np.bincount(zb[cm], weights=y[cm], minlength=n_bins)
    keep = c_b > 0
    ev_keep = ev & keep[np.clip(zb, 0, n_bins - 1)]
    n_ev = int(ev_keep.sum())
    if n_ev == 0:
        return {"effect": np.nan, "ev_mean": np.nan, "ctrl_mean": np.nan, "n_ev": 0, "n_ctrl": int(cm.sum()),
                "n_dropped": int(ev.sum())}
    w = np.where(keep, ev_b, 0) / n_ev
    ctrl_mean = float((w * np.where(keep, c_sum / np.maximum(c_b, 1), 0.0)).sum())
    ev_mean = float(y[ev_keep].mean())
    return {"effect": ev_mean - ctrl_mean, "ev_mean": ev_mean, "ctrl_mean": ctrl_mean, "n_ev": n_ev, "n_ctrl": int(cm.sum()),
            "n_dropped": int(ev.sum()) - n_ev}


def event_study(rank: np.ndarray, y: np.ndarray, zb: np.ndarray, t_ns: np.ndarray, q: float, side: str, gap_ns: int,
                n_bins: int) -> dict:
    defined = np.isfinite(rank)
    ev = event_mask(rank, q, side)
    sel = select_episodes(ev, t_ns, gap_ns)
    out = matched_effect(y, sel, control_mask(ev, defined, t_ns, gap_ns), zb, n_bins)
    out["n_events"] = int(ev.sum())
    out["episodes"] = sel
    return out


def shift_null(rank: np.ndarray, y: np.ndarray, zb: np.ndarray, t_ns: np.ndarray, q: float, side: str, gap_ns: int,
               n_bins: int, reps: int, min_shift: int, seed: int) -> np.ndarray:
    """信号の列を k（min_shift 以上、長さ − min_shift 以下）だけ循環シフトしたときの差。"""
    rng = np.random.default_rng(seed)
    n = len(rank)
    ks = rng.integers(min_shift, n - min_shift + 1, size=reps)
    return np.array([event_study(np.roll(rank, int(k)), y, zb, t_ns, q, side, gap_ns, n_bins)["effect"] for k in ks])


def p_one_sided(obs: float, null: np.ndarray, direction: int) -> float:
    """仮説の向き（+1: 事象の後が高い、−1: 低い）の片側 p。(1 + 帰無で観測以上に極端な回数) / (1 + 回数)。"""
    null = null[np.isfinite(null)]
    if not np.isfinite(obs):
        return float("nan")
    k = int((null >= obs).sum()) if direction > 0 else int((null <= obs).sum())
    return (1 + k) / (1 + len(null))


def har_walkforward(rv: pd.Series, log_close_known: pd.Series, times: pd.DatetimeIndex, windows=(1, 4, 24, 168),
                    calendar: bool = True, floor: float = 1e-10, h: int = 24, min_train_days: int = 60,
                    quantiles=(0.05, 0.01)) -> dict:
    """HAR 型の σ̂ と分位点を、各月の初め（と最初の判断の時刻）に当てはめ直して出す。

    rv: 足 t（[t, t+1h)）の 1 分の実現分散（index は足の開始時刻、抜けのない 1 時間ごと）。行 t の特徴量は足 t が閉じる t + 1h に分かり、
    行 t の目的変数は足 t+1〜t+h の実現分散の和の対数、行 t の収益率は T = t + 1h から T + h までの対数収益率。
    当てはめには、時刻 R までに収益率が出そろった行（T + h ≤ R）だけを使う。"""
    X = har_features(rv, windows, floor, calendar)
    tgt = har_target(rv, h, floor).to_numpy()
    T_rows = X.index + HOUR
    r_rows = log_close_known.reindex(T_rows + pd.Timedelta(hours=h)).to_numpy() - log_close_known.reindex(T_rows).to_numpy()
    feat_ok = X.notna().all(axis=1).to_numpy()
    months = pd.date_range(times[0].normalize() + pd.offsets.MonthBegin(1), times[-1], freq="MS")
    refits = [times[0]] + [m for m in months if m > times[0]]
    sigma = np.full(len(times), np.nan)
    qv = {q: np.full(len(times), np.nan) for q in quantiles}
    fits = []
    for i, R in enumerate(refits):
        nxt = refits[i + 1] if i + 1 < len(refits) else times[-1] + HOUR
        train = feat_ok & np.isfinite(tgt) & np.isfinite(r_rows) & (T_rows + pd.Timedelta(hours=h) <= R)
        if train.sum() < min_train_days * 24:
            continue
        m = HARVol().fit(X.to_numpy()[train], tgt[train])
        z = r_rows[train] / m.sigma(X.to_numpy()[train])
        zq = {q: float(np.quantile(z, q)) for q in quantiles}
        sel = (times >= R) & (times < nxt)
        rows = X.reindex(times[sel] - HOUR)
        ok = rows.notna().all(axis=1).to_numpy()
        s = np.full(int(sel.sum()), np.nan)
        s[ok] = m.sigma(rows.to_numpy()[ok])
        sigma[sel] = s
        for q in quantiles:
            qv[q][sel] = s * zq[q]
        fits.append({"refit": str(R), "n_train": int(train.sum()), "r2": m.r2, **{f"z{q}": zq[q] for q in quantiles}})
    return {"sigma": sigma, "q": qv, "fits": fits}
