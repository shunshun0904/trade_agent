"""1 分ごとの特徴量（研究用の direction.py と本番の Lambda（dashboard/app）で同じ関数を使う）。

どれも時刻 t より前のデータだけから作る:
- 1 分足から: 直近 1・5・15・60 分の対数リターン、1・5・15 分の出来高・売買の偏り・約定件数
- 15 分足から（t 以前に確定した最新の足）: features.bar_features と distfeat.dist_bar_features（接頭辞 b_）
- 価格帯別出来高・TPO（直近 24 時間、刻み 0.25σ）: POC・VAH・VAL までの距離（σ 単位）、バリューエリア内の
  位置、現在の価格帯の厚さ、上下 1σ 以内の量の割合、歪度、TPO のシングルプリントの割合
- 時刻・曜日

numpy と pandas だけに依存する（lightgbm・sklearn・requests は使わない）。
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .distfeat import dist_bar_features
from .features import bar_features
from .labeling import TradeTape
from .profile import value_area
from .signals import ewm_sigma

log = logging.getLogger(__name__)
MIN_MS = 60_000
BAR_MS = 15 * MIN_MS
H_MS = 3_600_000
FINE = 1e-4          # 対数価格の細かい刻み
LOG_RANGE = 0.25     # 現在値から ±25%（対数）の外の出来高は数えない
BIN_SIGMA = 0.25
PROFILE_COLUMNS = [
    "vp_poc_dist", "vp_vah_dist", "vp_val_dist", "vp_va_pos", "vp_at_price", "vp_up_1s", "vp_dn_1s", "vp_skew",
    "tpo_poc_dist", "tpo_vah_dist", "tpo_val_dist", "tpo_va_pos", "tpo_at_price", "tpo_up_1s", "tpo_dn_1s",
    "tpo_single_up", "tpo_single_dn",
]
_SHARED: dict = {}


# ---------------------------------------------------------------- 1 分足と流れの特徴量

def minute_grid(start_ms: int, end_ms: int) -> np.ndarray:
    return np.arange(start_ms, end_ms, MIN_MS, dtype="int64")


def minute_bars(tape: TradeTape, t0_ms: int, n: int) -> dict[str, np.ndarray]:
    """[t0 + i 分, t0 + (i+1) 分) の出来高・買い出来高・件数と、その分の最後の約定価格（約定なしは NaN）。"""
    lo = int(np.searchsorted(tape.ts, t0_ms, side="left"))
    hi = int(np.searchsorted(tape.ts, t0_ms + n * MIN_MS, side="left"))
    k = ((tape.ts[lo:hi] - t0_ms) // MIN_MS).astype(np.int64)
    amt = tape.amount[lo:hi]
    vol = np.bincount(k, weights=amt, minlength=n)
    buy = np.bincount(k, weights=np.where(tape.is_buy[lo:hi], amt, 0.0), minlength=n)
    cnt = np.bincount(k, minlength=n).astype(float)
    last = np.full(n, np.nan)
    if hi > lo:
        # 各分の最後の約定の位置
        ends = np.searchsorted(k, np.arange(n), side="right") - 1
        has = np.zeros(n, bool)
        has[np.unique(k)] = True
        last[has] = tape.price[lo:hi][ends[has]]
    return {"vol": vol, "buy": buy, "cnt": cnt, "last": last}


def flow_features(tape: TradeTape, grid: np.ndarray) -> pd.DataFrame:
    """時刻 t（grid）ごとに、t より前の約定だけから作る流れの特徴量。価格 p_t は t 直前の約定価格。"""
    back = 61  # pos − 60 ≥ 0 になるように（負の位置は配列の末尾を指してしまう）
    t0 = int(grid[0]) - back * MIN_MS
    n = len(grid) + back
    mb = minute_bars(tape, t0, n)
    last = pd.Series(mb["last"]).ffill().to_numpy()
    # grid[i] の直前の分（t − 1 分 〜 t）は mb の位置 back + i − 1
    pos = np.arange(len(grid)) + back - 1
    out = {}
    lp = np.log(last)
    for k in (1, 5, 15, 60):
        out[f"m_ret_{k}"] = lp[pos] - lp[pos - k]
    cs = {name: np.concatenate([[0.0], np.cumsum(mb[name])]) for name in ("vol", "buy", "cnt")}
    for k in (1, 5, 15):
        vol = cs["vol"][pos + 1] - cs["vol"][pos + 1 - k]
        buy = cs["buy"][pos + 1] - cs["buy"][pos + 1 - k]
        cnt = cs["cnt"][pos + 1] - cs["cnt"][pos + 1 - k]
        out[f"m_logvol_{k}"] = np.log1p(vol)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"m_imb_{k}"] = np.where(vol > 0, (2 * buy - vol) / vol, 0.0)
        out[f"m_cnt_{k}"] = cnt
    out["price"] = last[pos]
    return pd.DataFrame(out, index=grid)


def targets(tape: TradeTape, grid: np.ndarray, horizon_ms: int = BAR_MS, min_ret: float = 0.0
            ) -> tuple[np.ndarray, np.ndarray]:
    """(y, r)。r は t+h の直前の約定価格と t の直前の約定価格の対数比。

    min_ret = 0（上げ下げ）: y は r > 0 で 1、r < 0 で 0、r = 0 は NaN。
    min_ret > 0（費用を超える値幅で上がるか、2026-09-26 オーナー指示）: y は r > min_ret で 1、それ以外は 0。
    """
    i0 = np.searchsorted(tape.ts, grid, side="left") - 1
    i1 = np.searchsorted(tape.ts, grid + horizon_ms, side="left") - 1
    ok = (i0 >= 0) & (grid + horizon_ms <= tape.ts[-1])
    r = np.full(len(grid), np.nan)
    r[ok] = np.log(tape.price[i1[ok]] / tape.price[i0[ok]])
    if min_ret > 0:
        y = np.where(np.isnan(r), np.nan, np.where(r > min_ret, 1.0, 0.0))
    else:
        y = np.where(r > 0, 1.0, np.where(r < 0, 0.0, np.nan))
    return y, r


def taker_round_trip(taker_fee: float, slip: float) -> float:
    """成行で買って成行で売ったときに、損益ゼロになるために必要な対数リターン。"""
    return float(np.log((1 + slip) * (1 + taker_fee) / ((1 - slip) * (1 - taker_fee))))


# ---------------------------------------------------------------- 価格帯別出来高・TPO（1 分ごと）

def _coarse(counts_fine: np.ndarray, centers_log: np.ndarray, lp: float, w: float) -> tuple[np.ndarray, int]:
    k = np.floor((centers_log - lp) / w).astype(np.int64)
    kmin = int(k.min())
    return np.bincount(k - kmin, weights=counts_fine), kmin


def _levels_feats(counts: np.ndarray, kmin: int, prefix: str) -> dict[str, float]:
    """counts[i] は close 基準の相対番号 kmin + i の価格帯（刻み 0.25σ）。距離は σ 単位。"""
    out = {}
    tot = counts.sum()
    if tot <= 0:
        return out
    poc, lo, hi = value_area(counts)
    val = (lo + kmin) * BIN_SIGMA
    vah = (hi + kmin + 1) * BIN_SIGMA
    out[f"{prefix}_poc_dist"] = (poc + kmin + 0.5) * BIN_SIGMA
    out[f"{prefix}_vah_dist"] = vah
    out[f"{prefix}_val_dist"] = val
    out[f"{prefix}_va_pos"] = (0.0 - val) / (vah - val) if vah > val else np.nan
    cur = -kmin
    occ = counts[counts > 0]
    out[f"{prefix}_at_price"] = counts[cur] / occ.mean() if 0 <= cur < len(counts) else 0.0
    rel = np.arange(len(counts)) + kmin
    nb = int(round(1.0 / BIN_SIGMA))  # 1σ = 4 価格帯
    out[f"{prefix}_up_1s"] = counts[(rel >= 0) & (rel < nb)].sum() / tot
    out[f"{prefix}_dn_1s"] = counts[(rel < 0) & (rel >= -nb)].sum() / tot
    return out


def profile_chunk(args: tuple[int, int]) -> np.ndarray:
    """grid[a:b] の各時刻の PROFILE_COLUMNS（fork したワーカーで _SHARED を読む）。"""
    a, b = args
    grid = _SHARED["grid"]
    fb, qty, off, ts = _SHARED["fb"], _SHARED["qty"], _SHARED["off"], _SHARED["ts"]
    price = _SHARED["price"]
    sig, h2, l2, bar_end_ms = _SHARED["sigma"], _SHARED["h2"], _SHARED["l2"], _SHARED["bar_end_ms"]
    hist = np.zeros(_SHARED["n_fine"], dtype=np.int64)
    lo_i = hi_i = int(np.searchsorted(ts, grid[a] - 24 * H_MS, side="left"))
    r = int(LOG_RANGE / FINE)
    out = np.full((b - a, len(PROFILE_COLUMNS)), np.nan, dtype=np.float32)
    col = {c: i for i, c in enumerate(PROFILE_COLUMNS)}
    for row, t in enumerate(grid[a:b]):
        new_hi = int(np.searchsorted(ts, t, side="left"))
        new_lo = int(np.searchsorted(ts, t - 24 * H_MS, side="left"))
        if new_hi > hi_i:
            np.add.at(hist, fb[hi_i:new_hi] - off, qty[hi_i:new_hi])
            hi_i = new_hi
        if new_lo > lo_i:
            np.add.at(hist, fb[lo_i:new_lo] - off, -qty[lo_i:new_lo])
            lo_i = new_lo
        p = price[a + row]
        j = int(np.searchsorted(bar_end_ms, t, side="right")) - 1  # t 以前に確定した最新の 15 分足
        if not np.isfinite(p) or j < 95 or not (np.isfinite(sig[j]) and sig[j] > 0):
            continue
        s = float(sig[j])
        w = BIN_SIGMA * s
        lp = np.log(p)
        c = int(np.floor(lp / FINE)) - off
        x0, x1 = max(0, c - r), min(len(hist), c + r + 1)
        hseg = hist[x0:x1].astype(float)
        feats: dict[str, float] = {}
        if hseg.sum() > 0:
            centers = (np.arange(x0, x1) + off + 0.5) * FINE
            counts, kmin = _coarse(hseg, centers, lp, w)
            feats.update(_levels_feats(counts, kmin, "vp"))
            z = (centers - lp) / s
            tot = hseg.sum()
            m = (hseg * z).sum() / tot
            var = (hseg * (z - m) ** 2).sum() / tot
            feats["vp_skew"] = (hseg * (z - m) ** 3).sum() / tot / var**1.5 if var > 0 else 0.0
        # TPO: 直近 96 本（48 区間）。h2/l2[j] は足 j−1 と j をまとめた 30 分区間
        hh = np.log(h2[j - 94:j + 1:2])
        ll = np.log(l2[j - 94:j + 1:2])
        okk = np.isfinite(hh) & np.isfinite(ll)
        if okk.any():
            kh = np.floor((hh[okk] - lp) / w).astype(np.int64)
            kl = np.floor((ll[okk] - lp) / w).astype(np.int64)
            kmin = int(kl.min())
            size = int(kh.max()) - kmin + 1
            diff = np.zeros(size + 1)
            np.add.at(diff, kl - kmin, 1.0)
            np.add.at(diff, kh - kmin + 1, -1.0)
            counts = np.cumsum(diff)[:-1]
            feats.update(_levels_feats(counts, kmin, "tpo"))
            rel = np.arange(size) + kmin
            nb = int(round(1.0 / BIN_SIGMA))
            up = (rel >= 0) & (rel < nb)
            dn = (rel < 0) & (rel >= -nb)
            feats["tpo_single_up"] = ((counts[up] <= 1).sum() + (nb - up.sum())) / nb
            feats["tpo_single_dn"] = ((counts[dn] <= 1).sum() + (nb - dn.sum())) / nb
        for k, v in feats.items():
            out[row, col[k]] = v
    return out


def profile_features(tape: TradeTape, bars: pd.DataFrame, grid: np.ndarray, price: np.ndarray,
                     sigma_span: int = 96, workers: int = 1) -> pd.DataFrame:
    fb = np.floor(np.log(tape.price) / FINE).astype(np.int64)
    off = int(fb.min())
    step_ms = BAR_MS
    bar_start_ms = np.array([t.value // 1_000_000 for t in bars.index], dtype="int64")
    hi = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    h2 = np.full(len(bars), np.nan)
    l2 = np.full(len(bars), np.nan)
    h2[1:] = np.maximum(hi[1:], hi[:-1])
    l2[1:] = np.minimum(lo[1:], lo[:-1])
    _SHARED.clear()
    _SHARED.update(grid=grid, fb=fb, qty=np.rint(tape.amount * 1e4).astype(np.int64), off=off, ts=tape.ts,
                   n_fine=int(fb.max()) - off + 1, price=price, sigma=ewm_sigma(bars, sigma_span).to_numpy(),
                   h2=h2, l2=l2, bar_end_ms=bar_start_ms + step_ms)
    n = len(grid)
    parts = max(1, workers) * 4
    edges = np.linspace(0, n, parts + 1).astype(int)
    tasks = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
            chunks = list(ex.map(profile_chunk, tasks))
    else:
        chunks = [profile_chunk(t) for t in tasks]
    _SHARED.clear()
    return pd.DataFrame(np.vstack(chunks), columns=PROFILE_COLUMNS, index=grid)


# ---------------------------------------------------------------- 特徴量の表

WARMUP_HOURS = 49  # 24 時間の窓と、σ・ローリング特徴量（最大 192 本 = 48 時間）のための準備期間


def minute_features(tape: TradeTape, bars: pd.DataFrame, grid: np.ndarray, sigma_span: int = 96,
                    workers: int = 1) -> pd.DataFrame:
    """grid（分の開始時刻、ミリ秒）ごとの特徴量。正解は含まない（本番でも同じ関数を使う）。"""
    flow = flow_features(tape, grid)
    prof = profile_features(tape, bars, grid, flow["price"].to_numpy(), sigma_span, workers)
    bf = bar_features(bars, sigma_span).join(dist_bar_features(bars))
    bf = bf[[c for c in bf.columns if not c.startswith("_")]]
    bar_start_ms = np.array([t.value // 1_000_000 for t in bars.index], dtype="int64")
    j = np.searchsorted(bar_start_ms + BAR_MS, grid, side="right") - 1  # t 以前に確定した最新の足
    bft = bf.iloc[np.clip(j, 0, None)].to_numpy(dtype=np.float32)
    bft[j < 0] = np.nan
    X = pd.concat([flow.drop(columns=["price"]), prof,
                   pd.DataFrame(bft, columns=[f"b_{c}" for c in bf.columns], index=grid)], axis=1)
    tt = pd.to_datetime(grid, unit="ms", utc=True)
    hour = tt.hour + tt.minute / 60.0
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    X["dow_sin"], X["dow_cos"] = np.sin(2 * np.pi * tt.dayofweek / 7), np.cos(2 * np.pi * tt.dayofweek / 7)
    return X.astype("float32")


def feature_table(market, start, end, sigma_span: int = 96, workers: int = 1,
                  horizon_ms: int = BAR_MS, min_ret: float = 0.0) -> pd.DataFrame:
    """[start, end) の 1 分ごとの特徴量・正解。最初の WARMUP_HOURS 時間は準備期間。"""
    from bbdata.download import to_utc

    start, end = to_utc(start), to_utc(end)
    grid = minute_grid(int((start + pd.Timedelta(hours=WARMUP_HOURS)).value // 1_000_000),
                       int(end.value // 1_000_000))
    X = minute_features(market.tape, market.bars, grid, sigma_span, workers)
    y, r = targets(market.tape, grid, horizon_ms, min_ret)
    X["y"], X["r"] = y, r
    return X


