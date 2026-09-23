"""価格帯別出来高（ボリュームプロファイル）と TPO（Time Price Opportunity）の特徴量。

どちらも判断時刻 t0 の直前 window（既定 24 時間）だけから作る。区間は [t0 − window, t0)。
価格帯の刻み幅は w = bin_sigma × σ_0 × close(t0)（σ_0 は t0 時点の 15 分足の σ）で、
刻みの境界は close(t0) に揃える（close(t0) はある価格帯の下端になる）。距離はすべて σ 単位
（(価格 − close) / (σ_0 × close)）で表し、価格水準によらず比べられるようにする。

- 価格帯別出来高: 約定単位の数量を価格帯ごとに合計する。
- TPO: 15 分足を2本ずつまとめた 30 分の区間ごとに、その区間の安値〜高値にかかる価格帯へ 1 を数える
  （Market Profile の慣例に合わせて 30 分単位）。区間は t0 から過去へ数える。

特徴量（接頭辞 vp_ は出来高、tpo_ は TPO）:
    *_poc_dist      最も多い価格帯（POC）の中心と close の距離
    *_vah_dist      バリューエリア（POC から隣の多い側へ広げて全体の 70% に達するまで）の上端と close の距離
    *_val_dist      バリューエリアの下端と close の距離
    *_va_pos        バリューエリア内での close の位置（下端 0、上端 1。外側では 0 未満・1 超）
    *_at_price      close の価格帯の量 ÷ 量のある価格帯の平均（1 より大きいほど厚い）
    *_to_upper      close から利確バリア U までの価格帯の量が全体に占める割合
    *_to_lower      損切りバリア L から close までの価格帯の量が全体に占める割合
    vp_skew         出来高で重み付けした価格の歪度（close 基準）
    tpo_single_up   close から U までの価格帯のうち、TPO が 1 以下の割合（シングルプリントの多さ）
    tpo_single_dn   L から close までの価格帯のうち、TPO が 1 以下の割合
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .labeling import TradeTape

VALUE_AREA = 0.70
PROFILE_COLUMNS = [
    "vp_poc_dist", "vp_vah_dist", "vp_val_dist", "vp_va_pos", "vp_at_price", "vp_to_upper", "vp_to_lower", "vp_skew",
    "tpo_poc_dist", "tpo_vah_dist", "tpo_val_dist", "tpo_va_pos", "tpo_at_price", "tpo_to_upper", "tpo_to_lower",
    "tpo_single_up", "tpo_single_dn",
]


def value_area(counts: np.ndarray, share: float = VALUE_AREA) -> tuple[int, int, int]:
    """(POC, 下端, 上端) の配列インデックス。POC から、隣接する価格帯のうち量の多い側へ広げる。"""
    poc = int(np.argmax(counts))
    total = counts.sum()
    lo = hi = poc
    acc = counts[poc]
    while acc < share * total and (lo > 0 or hi < len(counts) - 1):
        down = counts[lo - 1] if lo > 0 else -1.0
        up = counts[hi + 1] if hi < len(counts) - 1 else -1.0
        if up >= down:
            hi += 1
            acc += counts[hi]
        else:
            lo -= 1
            acc += counts[lo]
    return poc, lo, hi


def _profile_stats(counts: np.ndarray, first_bin: int, u_bins: float, l_bins: float, bin_sigma: float,
                   prefix: str) -> dict[str, float]:
    """counts[i] は価格帯 first_bin + i（close を 0 番の下端とする相対番号）の量。"""
    out = {f"{prefix}_{k}": np.nan for k in
           ("poc_dist", "vah_dist", "val_dist", "va_pos", "at_price", "to_upper", "to_lower")}
    total = counts.sum()
    if total <= 0:
        return out
    poc, lo, hi = value_area(counts)
    centre = (np.arange(len(counts)) + first_bin + 0.5) * bin_sigma  # 価格帯の中心（σ 単位）
    val = (lo + first_bin) * bin_sigma          # バリューエリア下端（価格帯の下端）
    vah = (hi + first_bin + 1) * bin_sigma      # 上端（価格帯の上端）
    out[f"{prefix}_poc_dist"] = float(centre[poc])
    out[f"{prefix}_vah_dist"] = float(vah)
    out[f"{prefix}_val_dist"] = float(val)
    out[f"{prefix}_va_pos"] = float((0.0 - val) / (vah - val)) if vah > val else np.nan
    cur = -first_bin  # close を含む価格帯（相対番号 0）の配列インデックス
    occupied = counts[counts > 0]
    out[f"{prefix}_at_price"] = float(counts[cur] / occupied.mean()) if 0 <= cur < len(counts) else 0.0
    rel = np.arange(len(counts)) + first_bin
    out[f"{prefix}_to_upper"] = float(counts[(rel >= 0) & (rel < u_bins)].sum() / total)
    out[f"{prefix}_to_lower"] = float(counts[(rel < 0) & (rel >= -l_bins)].sum() / total)
    return out


def profile_features_at(
    tape: TradeTape, bars: pd.DataFrame, t0: pd.Timestamp, close: float, sigma0: float,
    k_up: float, k_dn: float, window: pd.Timedelta = pd.Timedelta(hours=24), bin_sigma: float = 0.25,
) -> dict[str, float]:
    """1 イベント分の特徴量。tape は amount を持つこと。"""
    out = {c: np.nan for c in PROFILE_COLUMNS}
    if not (np.isfinite(sigma0) and sigma0 > 0 and np.isfinite(close) and close > 0):
        return out
    w = bin_sigma * sigma0 * close
    t0_ms = int(t0.value // 1_000_000)
    i0 = int(np.searchsorted(tape.ts, t0_ms - int(window / pd.Timedelta(milliseconds=1)), side="left"))
    i1 = int(np.searchsorted(tape.ts, t0_ms, side="left"))
    u_bins = k_up / bin_sigma
    l_bins = k_dn / bin_sigma

    # ---- 価格帯別出来高
    if i1 > i0:
        px = tape.price[i0:i1]
        amt = tape.amount[i0:i1]
        b = np.floor((px - close) / w).astype(np.int64)
        first = int(b.min())
        counts = np.bincount(b - first, weights=amt).astype("float64")
        out.update(_profile_stats(counts, first, u_bins, l_bins, bin_sigma, "vp"))
        z = (px - close) / (sigma0 * close)
        tot = amt.sum()
        if tot > 0:
            m = float((amt * z).sum() / tot)
            var = float((amt * (z - m) ** 2).sum() / tot)
            out["vp_skew"] = float((amt * (z - m) ** 3).sum() / tot / var**1.5) if var > 0 else 0.0

    # ---- TPO（終了時刻が t0 以前の 15 分足を、t0 から過去へ 2 本ずつまとめる）
    step = bars.index[1] - bars.index[0]
    n_bars = int(window / step)
    j1 = int(bars.index.searchsorted(t0 - step, side="right"))  # 終了時刻が t0 以前の足まで
    j0 = max(0, j1 - n_bars)
    if j1 - j0 >= 2:
        hi = bars["high"].to_numpy()[j0:j1]
        lo = bars["low"].to_numpy()[j0:j1]
        # 末尾（t0 に近い側）から 2 本ずつ
        n = (j1 - j0) // 2 * 2
        hi2 = hi[len(hi) - n:].reshape(-1, 2).max(axis=1)
        lo2 = lo[len(lo) - n:].reshape(-1, 2).min(axis=1)
        ok = np.isfinite(hi2) & np.isfinite(lo2)
        if ok.any():
            bh = np.floor((hi2[ok] - close) / w).astype(np.int64)
            bl = np.floor((lo2[ok] - close) / w).astype(np.int64)
            first = int(bl.min())
            size = int(bh.max()) - first + 1
            diff = np.zeros(size + 1)
            np.add.at(diff, bl - first, 1.0)
            np.add.at(diff, bh - first + 1, -1.0)
            counts = np.cumsum(diff)[:-1]
            out.update(_profile_stats(counts, first, u_bins, l_bins, bin_sigma, "tpo"))
            rel = np.arange(size) + first
            up = (rel >= 0) & (rel < u_bins)
            dn = (rel < 0) & (rel >= -l_bins)
            # 範囲外（誰も付けていない価格帯）は TPO 0 として数える
            n_up, n_dn = int(np.ceil(u_bins)), int(np.ceil(l_bins))
            out["tpo_single_up"] = float(((counts[up] <= 1).sum() + (n_up - up.sum())) / n_up) if n_up else np.nan
            out["tpo_single_dn"] = float(((counts[dn] <= 1).sum() + (n_dn - dn.sum())) / n_dn) if n_dn else np.nan
    return out


def profile_event_features(
    events: pd.DataFrame, bars: pd.DataFrame, tape: TradeTape, sigma: pd.Series, k_up: float, k_dn: float,
    window: pd.Timedelta = pd.Timedelta(hours=24), bin_sigma: float = 0.25,
) -> pd.DataFrame:
    """イベントごとの価格帯別出来高・TPO 特徴量。index は event_id。

    close(t0) と σ_0 は、ラベルと同じくイベント足（終了時刻が t0 の足）の値を使う。
    """
    if tape.amount is None:
        raise ValueError("価格帯別出来高には約定の数量（TradeTape.amount）が必要です")
    step = bars.index[1] - bars.index[0]
    rows = []
    for ev in events.itertuples(index=False):
        bs = ev.t0 - step
        rows.append(profile_features_at(tape, bars, ev.t0, float(bars.at[bs, "close"]), float(sigma.at[bs]),
                                        k_up, k_dn, window, bin_sigma))
    out = pd.DataFrame(rows, columns=PROFILE_COLUMNS, index=events["event_id"].to_numpy())
    out.index.name = "event_id"
    return out
