"""h 分後に上がるか下がるかを予測する二段構えのモデル（2026-09-25 オーナー指示。h は spec の horizon_min）。

    python -m bbresearch direction --config configs/research.yaml --spec configs/direction.yaml --out reports/direction

- モデル B（補強用）: 1 分ごとに、h 分後の価格が今より高いかを予測する。
- モデル A（判断用）: h 分ごと（h = 15 なら足の区切り、60 なら毎正時）に同じことを予測する。入力は B と同じ
  特徴量に、直近 h 分の B の予測の要約（平均・最新・変化・ばらつき・直近 1/4 の平均）を加えたもの
  （スタッキング、A+B）。B の予測は、その時点のデータで学習していないモデルの予測だけを使う（学習期間は
  Purged K-fold の out-of-fold、評価期間は学習期間だけで学習した B の予測）。
- A+B+L（spec の level_summary: true のとき）: さらに、直近 h 分の生の価格帯別出来高・TPO の水準
  （PROFILE_COLUMNS）の平均と変化（最新 − h 分前）を加える（2026-09-26 オーナー指示: 1 時間の判断に
  1 分ごとの TPO×価格帯別出来高の推移を使う）。

特徴量（オーナー決定: 全部）。どれも時刻 t より前のデータだけから作る:
- 1 分足から: 直近 1・5・15・60 分の対数リターン、1・5・15 分の出来高・売買の偏り・約定件数
- 15 分足から（t 以前に確定した最新の足）: features.bar_features と distfeat.dist_bar_features
- 価格帯別出来高・TPO（直近 24 時間、刻み 0.25σ）: POC・VAH・VAL までの距離（σ 単位）、バリューエリア内の
  位置、現在の価格帯の厚さ、上下 1σ 以内の量の割合、歪度、TPO のシングルプリントの割合
- 時刻・曜日

正解: 15 分後（t + 15 分の直前の約定価格）が t の直前の約定価格より高ければ 1、低ければ 0。同じなら除く。

期間（オーナー決定）: 学習・交差検証は [data.start, train_end)、評価は [train_end, data.end) で 1 回だけ。
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bbdata.download import to_utc

from .cv import PurgedKFold
from .distfeat import dist_bar_features
from .features import bar_features
from .labeling import TradeTape
from .model import metrics, params_hash
from .pipeline import barrier_params, load_market
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

def feature_table(market, start, end, sigma_span: int = 96, workers: int = 1,
                  horizon_ms: int = BAR_MS, min_ret: float = 0.0) -> pd.DataFrame:
    """[start, end) の 1 分ごとの特徴量・正解。最初の 49 時間は準備期間（24 時間の窓と σ のため）。"""
    start, end = to_utc(start), to_utc(end)
    grid = minute_grid(int((start + pd.Timedelta(hours=49)).value // 1_000_000), int(end.value // 1_000_000))
    tape, bars = market.tape, market.bars
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
    y, r = targets(tape, grid, horizon_ms, min_ret)
    X["y"], X["r"] = y, r
    return X.astype({c: "float32" for c in X.columns if c not in ("y", "r")})


# ---------------------------------------------------------------- モデル

def lgbm(n_rows: int, seed: int = 0, n_jobs: int = 4):
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=max(50, n_rows // 2000), subsample=0.5, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=seed, n_jobs=n_jobs, verbose=-1)


def oof_and_final(X: pd.DataFrame, y: np.ndarray, t_ms: np.ndarray, X_test: pd.DataFrame, n_splits: int,
                  horizon_ms: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """学習期間の out-of-fold 予測、学習期間全体で学習したモデルの評価期間の予測、fold ごとの指標。"""
    t0 = pd.Series(pd.to_datetime(t_ms, unit="ms", utc=True))
    tx = t0 + pd.Timedelta(milliseconds=horizon_ms)
    cv = PurgedKFold(n_splits=n_splits, embargo=pd.Timedelta(milliseconds=horizon_ms))
    oof = np.full(len(X), np.nan)
    folds = []
    for k, (tr, te) in enumerate(cv.split(t0, tx)):
        m = lgbm(len(tr), seed).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        folds.append({"fold": k, **{kk: v for kk, v in metrics(y[te], oof[te]).items() if kk != "calibration"}})
    final = lgbm(len(X), seed).fit(X, y)
    return oof, final.predict_proba(X_test)[:, 1], folds


def stack_features(pred_min: pd.Series, t_ms: np.ndarray, n_min: int = 15) -> pd.DataFrame:
    """判断時刻 t に、直前 n_min 分（t − (n_min−1) 分 〜 t）の B の予測の要約を付ける。"""
    s = pred_min.reindex(np.concatenate([t_ms - k * MIN_MS for k in range(n_min - 1, -1, -1)]))
    v = s.to_numpy().reshape(n_min, len(t_ms)).T  # 行: 時刻、列: t−(n_min−1) … t
    q = max(1, n_min // 4)
    with np.errstate(invalid="ignore"):
        return pd.DataFrame({"B_mean": np.nanmean(v, axis=1), "B_last": v[:, -1], "B_slope": v[:, -1] - v[:, 0],
                             "B_std": np.nanstd(v, axis=1), "B_mean_recent": np.nanmean(v[:, -q:], axis=1)},
                            index=t_ms)


def level_summary(levels: pd.DataFrame, t_ms: np.ndarray, n_min: int) -> pd.DataFrame:
    """判断時刻 t に、直前 n_min 分（t − (n_min−1) 分 〜 t）の生の水準（1 分ごと）の平均と変化（最新 − 最初）を付ける。

    levels の index は 1 分ごとの時刻（ミリ秒）。窓に含まれない時刻は NaN として扱う。
    """
    cols = list(levels.columns)
    idx = np.concatenate([t_ms - k * MIN_MS for k in range(n_min - 1, -1, -1)])
    v = levels.reindex(idx).to_numpy(dtype=float).reshape(n_min, len(t_ms), len(cols))  # (窓, 時刻, 列)
    out = {}
    with np.errstate(invalid="ignore"):
        for j, c in enumerate(cols):
            out[f"L_{c}_mean"] = np.nanmean(v[:, :, j], axis=0)
            out[f"L_{c}_chg"] = v[-1, :, j] - v[0, :, j]
    return pd.DataFrame(out, index=t_ms)


def pnl(p: np.ndarray, r: np.ndarray, theta: float, fees: tuple[float, float], slip: float,
        sel: np.ndarray | None = None) -> dict:
    """確率が theta 以上（sel を渡せばその時刻）のとき t で買い t+h で売る。taker: 成行往復、maker: 手数料・滑りなし（楽観的な上限）。"""
    sel = p >= theta if sel is None else sel
    g = np.exp(r[sel])
    taker = g * (1 - slip) * (1 - fees[1]) / ((1 + slip) * (1 + fees[1])) - 1
    maker = g - 1
    def st(x):
        if len(x) < 3:
            return {"n": int(len(x))}
        return {"n": int(len(x)), "mean": float(x.mean()), "t": float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))),
                "win": float((x > 0).mean())}
    return {"theta": theta, "coverage": float(sel.mean()), "taker": st(taker), "maker_upper": st(maker)}


def _peak_gb() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def run_direction(cfg: dict, spec: dict, workers: int | None = None) -> dict:
    market = load_market(cfg)
    workers = workers or max(1, os.cpu_count() or 1)
    start, end = to_utc(cfg["data"]["start"]), to_utc(cfg["data"]["end"])
    train_end = to_utc(spec["train_end"])
    bprm = barrier_params({**cfg, "pair_spec": market.spec}, market.spec)
    fees, slip = (bprm.maker_fee, bprm.taker_fee), bprm.s_slip
    h_min = int(spec.get("horizon_min", 15))
    h_ms = h_min * MIN_MS
    # 目的変数のしきい値: "taker_round_trip" なら成行往復の費用（手数料・滑りから計算）、数値ならその対数リターン、なければ 0
    tm = spec.get("target_min_return", 0.0)
    min_ret = taker_round_trip(fees[1], slip) if tm == "taker_round_trip" else float(tm)
    use_levels = bool(spec.get("level_summary", False))
    log.info("特徴量を計算 %s〜%s、ホライズン %d 分", start, end, h_min)
    F = feature_table(market, start, end, cfg["signal"]["sigma_span"], workers, h_ms, min_ret)
    del market  # 約定データは以降使わない（メモリを空ける）
    levels = F[PROFILE_COLUMNS] if use_levels else None  # 1 分ごとの生の水準（正解のない時刻も窓に使う）
    F = F[F["y"].notna()]
    log.info("特徴量 %d 行 × %d 列、最大メモリ %.1f GB", len(F), F.shape[1], _peak_gb())
    t_ms = F.index.to_numpy()
    feat_cols = [c for c in F.columns if c not in ("y", "r")]
    is_tr = t_ms < int(train_end.value // 1_000_000) - h_ms  # 正解が学習期間内で決まるものだけ
    is_te = t_ms >= int(train_end.value // 1_000_000)
    n_splits = int(spec.get("n_splits", 5))

    # モデル B（1 分ごと）
    log.info("モデル B: 学習 %d、評価 %d", int(is_tr.sum()), int(is_te.sum()))
    Xb_tr, Xb_te = F.loc[is_tr, feat_cols], F.loc[is_te, feat_cols]
    yb_tr, yb_te = F.loc[is_tr, "y"].to_numpy(int), F.loc[is_te, "y"].to_numpy(int)
    oof_b, test_b, folds_b = oof_and_final(Xb_tr, yb_tr, t_ms[is_tr], Xb_te, n_splits, h_ms)
    pred_b = pd.Series(np.concatenate([oof_b, test_b]), index=np.concatenate([t_ms[is_tr], t_ms[is_te]]))

    # モデル A（h 分ごと）。h の区切りの時刻だけ
    on_bar = (t_ms % h_ms) == 0
    a_tr, a_te = is_tr & on_bar, is_te & on_bar
    ta_tr, ta_te = t_ms[a_tr], t_ms[a_te]
    ya_tr, ya_te = F.loc[a_tr, "y"].to_numpy(int), F.loc[a_te, "y"].to_numpy(int)
    ra_te = F.loc[a_te, "r"].to_numpy()
    XA_tr, XA_te = F.loc[a_tr, feat_cols], F.loc[a_te, feat_cols]
    SB_tr, SB_te = stack_features(pred_b, ta_tr, h_min), stack_features(pred_b, ta_te, h_min)
    variants = {"A": (XA_tr, XA_te),
                "A+B": (pd.concat([XA_tr, SB_tr.set_axis(XA_tr.index)], axis=1),
                        pd.concat([XA_te, SB_te.set_axis(XA_te.index)], axis=1))}
    if use_levels:
        LS_tr, LS_te = level_summary(levels, ta_tr, h_min), level_summary(levels, ta_te, h_min)
        variants["A+B+L"] = (pd.concat([variants["A+B"][0], LS_tr.set_axis(XA_tr.index)], axis=1),
                             pd.concat([variants["A+B"][1], LS_te.set_axis(XA_te.index)], axis=1))
    log.info("モデル A: 学習 %d、評価 %d、変種 %s", len(XA_tr), len(XA_te), list(variants))
    oof, test, folds = {}, {}, {}
    for name, (Xtr, Xte) in variants.items():
        oof[name], test[name], folds[name] = oof_and_final(Xtr, ya_tr, ta_tr, Xte, n_splits, h_ms)
    log.info("学習を終了、最大メモリ %.1f GB", _peak_gb())

    thetas = list(spec.get("thetas", [0.55, 0.6]))
    top_fracs = list(spec.get("top_fracs", []))
    b_on_bar = pred_b.reindex(ta_te).to_numpy()

    def pnl_rows(p):
        rows = [pnl(p, ra_te, th, fees, slip) for th in thetas]
        for f in top_fracs:  # 評価期間の予測確率の上位 f（順位で選ぶ）。しきい値を評価期間で決めるので、説明用
            k = max(1, int(round(f * len(p))))
            top = np.zeros(len(p), bool)
            top[np.argsort(-p, kind="stable")[:k]] = True
            rows.append({**pnl(p, ra_te, float(p[top].min()), fees, slip, sel=top), "top_frac": f})
        return rows
    cv = {"B": {k: v for k, v in metrics(yb_tr, oof_b).items() if k != "calibration"}, "B_folds": folds_b}
    for name in variants:
        cv[name] = {k: v for k, v in metrics(ya_tr, oof[name]).items() if k != "calibration"}
        cv[f"{name}_folds"] = folds[name]
    res = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "horizon_min": h_min,
        "periods": {"train": [str(start), str(train_end)], "test": [str(train_end), str(end)]},
        "n": {"B_train": int(is_tr.sum()), "B_test": int(is_te.sum()), "A_train": int(len(XA_tr)), "A_test": int(len(XA_te))},
        "base_rate_test": float(ya_te.mean()) if len(ya_te) else None,
        "features": feat_cols,
        "variant_features": {name: list(Xtr.columns) for name, (Xtr, _) in variants.items()},
        "cv": cv,
        "test": {"B_all_minutes": metrics(yb_te, test_b), "B_on_bar": metrics(ya_te, b_on_bar),
                 **{name: metrics(ya_te, test[name]) for name in variants}},
        "target_min_return": min_ret,
        "pnl_test": {name: pnl_rows(p) for name, p in (*test.items(), ("B_on_bar", b_on_bar))},
        "unconditional_test": pnl(np.ones(len(ra_te)), ra_te, 0.0, fees, slip),
        "fees": {"maker": fees[0], "taker": fees[1], "s_slip": slip},
    }
    # AUC の差（各変種 − A）の日単位ブロック・ブートストラップ
    days = (ta_te // (24 * H_MS)).astype(np.int64)
    uniq = np.unique(days)
    rng = np.random.default_rng(0)
    from sklearn.metrics import roc_auc_score

    idx_by_day = {d: np.flatnonzero(days == d) for d in uniq}
    diffs = {name: [] for name in variants if name != "A"}
    for _ in range(int(spec.get("n_boot", 200))):
        pick = np.concatenate([idx_by_day[d] for d in rng.choice(uniq, size=len(uniq), replace=True)])
        yy = ya_te[pick]
        if len(np.unique(yy)) < 2:
            continue
        base = roc_auc_score(yy, test["A"][pick])
        for name in diffs:
            diffs[name].append(roc_auc_score(yy, test[name][pick]) - base)
    res["auc_diff_vs_A"] = {name: {"mean": float(np.mean(d)), "p05": float(np.percentile(d, 5)),
                                   "p95": float(np.percentile(d, 95))} for name, d in diffs.items() if d}
    # 実験ログ（試行として数える）
    th = params_hash({"direction": spec, "features": feat_cols, "fees": list(fees), "s_slip": slip})
    log_path = Path(cfg.get("experiment_log", "reports/experiments.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fh:
        for name in (*variants, "B_on_bar"):
            fh.write(json.dumps({"run_at": res["run_at"], "stage": "direction", "trial_hash": th,
                                 "code_version": os.environ.get("GITHUB_SHA"), "key": f"direction/{h_min}m/r>{min_ret:.4f}/{name}",
                                 "test_auc": res["test"][name].get("auc")}) + "\n")
    res["trial_hash"] = th
    return res


def _p(x, pct=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x * 100:+.3f}%" if pct else f"{x:.4f}"


def render(rep: dict) -> str:
    h = rep.get("horizon_min", 15)
    mr = rep.get("target_min_return", 0.0)
    what = f"{h} 分後に {mr * 100:.2f}% を超えて上がるか" if mr > 0 else f"{h} 分後の上げ下げ"
    md = [f"# {what}の予測（二段構え、{rep['run_at']}）\n",
          f"- 学習・交差検証: {rep['periods']['train'][0]} 〜 {rep['periods']['train'][1]}",
          f"- 評価（1 回だけ）: {rep['periods']['test'][0]} 〜 {rep['periods']['test'][1]}（これまでの分析で一部を見ている期間）",
          f"- 標本: B 学習 {rep['n']['B_train']:,} / 評価 {rep['n']['B_test']:,}、A 学習 {rep['n']['A_train']:,} / 評価 {rep['n']['A_test']:,}",
          f"- 目的変数: {what}（1 = はい）。評価期間で 1 の割合: {_p(rep['base_rate_test'])[1:]}、特徴量 {len(rep['features'])} 個"
          + (f"（A+B+L は {len(rep['variant_features']['A+B+L'])} 個）" if "A+B+L" in rep.get("variant_features", {}) else "") + "\n",
          "## 当たり具合（AUC、0.5 が当て推量）\n",
          "| モデル | 交差検証（学習期間） | 評価期間 | 評価期間 log loss |\n|---|---|---|---|"]
    cv, te = rep["cv"], rep["test"]
    rows = [("B（1 分ごと、全時刻）", "B", "B_all_minutes"), (f"B（{h} 分の区切りだけ）", None, "B_on_bar"),
            (f"A（{h} 分ごと）", "A", "A"), ("A+B（二段構え: A + 直近の B の要約）", "A+B", "A+B")]
    if "A+B+L" in te:
        rows.append(("A+B+L（さらに直近の TPO・価格帯別出来高の水準の推移）", "A+B+L", "A+B+L"))
    for name, cvk, tek in rows:
        c = cv.get(cvk, {}) if cvk else {}
        md.append(f"| {name} | {_p(c.get('auc'), False)} | {_p(te[tek].get('auc'), False)} | {_p(te[tek].get('log_loss'), False)} |")
    for name, d in rep.get("auc_diff_vs_A", {}).items():
        md.append(f"\n{name} と A の AUC の差（評価期間、日単位ブートストラップ）: 平均 {_p(d['mean'], False)}、"
                  f"5〜95% 区間 [{_p(d['p05'], False)}, {_p(d['p95'], False)}]")
    md.append(f"\n## 費用込みの損益（評価期間、確率がしきい値以上で買い {h} 分後に売る）\n")
    md.append("| モデル | しきい値 | 発注割合 | 取引 | 成行往復の平均 | t 値 | 勝率 | 手数料・滑りなしの平均（上限） |\n|---|---|---|---|---|---|---|---|")
    for name, rows in rep["pnl_test"].items():
        for r in rows:
            tk, mk = r["taker"], r["maker_upper"]
            th = f"上位 {r['top_frac'] * 100:.0f}%（{r['theta']:.3f}）" if "top_frac" in r else f"{r['theta']}"
            md.append(f"| {name} | {th} | {_p(r['coverage'])[1:]} | {tk.get('n', 0)} | {_p(tk.get('mean'))} | "
                      f"{_p(tk.get('t'), False)} | {_p(tk.get('win'))[1:]} | {_p(mk.get('mean'))} |")
    u = rep["unconditional_test"]
    md.append(f"| 常に買う | - | 100% | {u['taker'].get('n', 0)} | {_p(u['taker'].get('mean'))} | "
              f"{_p(u['taker'].get('t'), False)} | {_p(u['taker'].get('win'))[1:]} | {_p(u['maker_upper'].get('mean'))} |")
    md.append(f"\n手数料: メイカー {rep['fees']['maker']}、テイカー {rep['fees']['taker']}、成行の滑り {rep['fees']['s_slip']}")
    return "\n".join(md) + "\n"


def main(args) -> None:
    from .pipeline import load_config

    rep = run_direction(load_config(args.config), yaml.safe_load(Path(args.spec).read_text()), workers=args.workers)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    md = render(rep)
    (out / "report.md").write_text(md)
    print(md)
