"""価格帯別出来高と TPO の計算（標準ライブラリだけで書いたダッシュボード用の実装）。

研究用の bbresearch/profile.py と同じ定義にしている（テストで照合する）:
- 区間は [now − window, now)。画面は直近 3 時間（2026-09-26 オーナー決定。研究用の既定は 24 時間）。
- 価格帯の刻みは w = bin_sigma × σ × 基準価格（基準価格は now 直前の約定価格）。境界は基準価格に揃える。
- バリューエリアは POC から隣の多い側へ広げて全体の 70% に達するまで（同量なら上側）。
- TPO は足を now から過去へ block 本ずつまとめた区間ごとに、安値〜高値にかかる価格帯へ 1 を数える。
  画面は 1 分足を 5 本ずつ（5 分区間）。研究用は 15 分足を 2 本ずつ（30 分区間）。
σ は直近 180 本の 1 分足の対数リターンの標準偏差（2026-09-26 オーナー決定。研究用の 15 分足の EWMA とは異なる。表示用）。
"""
from __future__ import annotations

import math

BAR_MS = 15 * 60_000


def value_area(counts: list[float], share: float = 0.70) -> tuple[int, int, int]:
    poc = max(range(len(counts)), key=lambda i: (counts[i], -i))  # 同量なら最初（numpy.argmax と同じ）
    total = sum(counts)
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


def bars_at(trades: list[tuple[int, float, float]], start_ms: int, end_ms: int, step_ms: int,
            candles: list[list] | None = None, trade_from_ms: int | None = None) -> list[dict]:
    """[start, end) の step_ms ごとの足（約定なしの足は直前の close で埋める）。trades は (ts, price, amount) の時刻順。

    candles（公式 1 分足 [t, o, h, l, c, v]）を渡すと、trade_from_ms より前の分はそれで作る（約定がそろっていない
    時間帯の補い。step_ms が 1 分のときだけ）。
    """
    n = (end_ms - start_ms) // step_ms
    out = [{"t": start_ms + i * step_ms, "o": None, "h": None, "l": None, "c": None, "v": 0.0} for i in range(n)]
    cut = _approx_until(trade_from_ms, step_ms) if candles else None
    for ts, px, amt in trades:
        if not (start_ms <= ts < start_ms + n * step_ms) or (cut is not None and ts < cut):
            continue
        b = out[(ts - start_ms) // step_ms]
        if b["o"] is None:
            b["o"] = b["h"] = b["l"] = px
        b["h"] = max(b["h"], px)
        b["l"] = min(b["l"], px)
        b["c"] = px
        b["v"] += amt
    if cut is not None and step_ms == CANDLE_MS:
        for t, o, h, l, c, v in candles:
            if start_ms <= t < min(cut, start_ms + n * step_ms) and (t - start_ms) % step_ms == 0:
                out[(t - start_ms) // step_ms].update(o=o, h=h, l=l, c=c, v=v)
    last = None
    for b in out:
        if b["c"] is None:
            if last is not None:
                b["o"] = b["h"] = b["l"] = b["c"] = last
        else:
            last = b["c"]
    return out


def bars_15m(trades: list[tuple[int, float, float]], start_ms: int, end_ms: int) -> list[dict]:
    return bars_at(trades, start_ms, end_ms, BAR_MS)


def sigma_from_bars(bars: list[dict], n: int = 96) -> float | None:
    closes = [b["c"] for b in bars if b["c"] is not None]
    r = [math.log(b / a) for a, b in zip(closes[:-1], closes[1:])][-n:]
    if len(r) < n:
        return None
    m = sum(r) / len(r)
    var = sum((x - m) ** 2 for x in r) / (len(r) - 1)
    return math.sqrt(var) if var > 0 else None


def _levels(counts: list[float], first: int, ref: float, w: float) -> dict:
    poc, lo, hi = value_area(counts)
    return {"poc": ref + (poc + first + 0.5) * w, "val": ref + (lo + first) * w, "vah": ref + (hi + first + 1) * w}


def _approx_until(trade_from_ms: int | None, step_ms: int) -> int | None:
    """約定がそろっているのは trade_from_ms 以降。その分の開始時刻より前を公式 1 分足で補う。"""
    if trade_from_ms is None:
        return None
    return trade_from_ms // step_ms * step_ms


def volume_profile(trades, now_ms: int, ref: float, w: float, window_ms: int, candles: list[list] | None = None,
                   trade_from_ms: int | None = None) -> tuple[dict, list[dict]]:
    """価格帯別出来高。candles を渡すと、trade_from_ms の分より前は公式 1 分足の出来高を安値〜高値の価格帯に均等に配る。"""
    acc: dict[int, float] = {}
    cut = _approx_until(trade_from_ms, CANDLE_MS) if candles else None
    for ts, px, amt in trades:
        if now_ms - window_ms <= ts < now_ms and (cut is None or ts >= cut):
            k = math.floor(round((px - ref) / w, 9))
            acc[k] = acc.get(k, 0.0) + amt
    if cut is not None:
        for t, o, h, l, c, v in candles:
            if now_ms - window_ms <= t < min(cut, now_ms) and v > 0:
                k0, k1 = math.floor(round((l - ref) / w, 9)), math.floor(round((h - ref) / w, 9))
                for k in range(k0, k1 + 1):
                    acc[k] = acc.get(k, 0.0) + v / (k1 - k0 + 1)
    if not acc:
        return {}, []
    first, last = min(acc), max(acc)
    counts = [acc.get(k, 0.0) for k in range(first, last + 1)]
    rows = [{"lo": ref + k * w, "hi": ref + (k + 1) * w, "v": acc.get(k, 0.0)} for k in range(first, last + 1)]
    return _levels(counts, first, ref, w), rows


def tpo_profile(bars: list[dict], now_ms: int, ref: float, w: float, window_ms: int, step_ms: int = BAR_MS,
                block: int = 2) -> tuple[dict, list[dict]]:
    """bars は step_ms ごとの足。終了時刻が now 以前の足を、now から過去へ block 本ずつまとめる。"""
    done = [b for b in bars if b["t"] + step_ms <= now_ms and b["t"] >= now_ms - window_ms and b["h"] is not None]
    n = len(done) // block * block
    done = done[len(done) - n:]
    acc: dict[int, int] = {}
    for i in range(0, n, block):
        hi = max(b["h"] for b in done[i:i + block])
        lo = min(b["l"] for b in done[i:i + block])
        for k in range(math.floor(round((lo - ref) / w, 9)), math.floor(round((hi - ref) / w, 9)) + 1):
            acc[k] = acc.get(k, 0) + 1
    if not acc:
        return {}, []
    first, last = min(acc), max(acc)
    counts = [float(acc.get(k, 0)) for k in range(first, last + 1)]
    rows = [{"lo": ref + k * w, "hi": ref + (k + 1) * w, "n": acc.get(k, 0)} for k in range(first, last + 1)]
    return _levels(counts, first, ref, w), rows


WINDOW_H = 3.0        # 画面の窓（2026-09-26 オーナー決定）
CANDLE_MS = 60_000    # 画面のローソク足は 1 分足
TPO_BLOCK = 5         # TPO は 1 分足 5 本 = 5 分区間
SIGMA_N = 180         # σ は直近 180 本の 1 分足のリターンの標準偏差（2026-09-26 オーナー決定）


def compute(trades: list[tuple[int, float, float]], now_ms: int, window_h: float = WINDOW_H,
            bin_sigma: float = 0.25, candle_ms: int = CANDLE_MS, tpo_block: int = TPO_BLOCK,
            sigma_n: int = SIGMA_N, candles: list[list] | None = None, trade_from_ms: int | None = None) -> dict:
    """ダッシュボードに渡す一式。trades は (ts, price, amount) の時刻順で、σ の計算に十分前から含むこと。

    σ は candle_ms の足の対数リターン直近 sigma_n 本の標準偏差。価格帯の刻みは bin_sigma × σ × 基準価格。
    candles と trade_from_ms を渡すと、約定がそろっていない時間帯（trade_from_ms より前）を公式 1 分足で補う。
    """
    window_ms = int(window_h * 3_600_000)
    past = [t for t in trades if t[0] < now_ms]
    if not past:
        return {"error": "約定がない"}
    ref = past[-1][1]
    end = now_ms // candle_ms * candle_ms + candle_ms  # 形成中の足まで（表示用）
    back = max(window_ms + candle_ms * tpo_block, candle_ms * (sigma_n + 2))
    bars = bars_at(past, end - back, end, candle_ms, candles, trade_from_ms)
    sigma = sigma_from_bars([b for b in bars if b["t"] + candle_ms <= now_ms], sigma_n)
    if sigma is None:
        return {"error": "σ を計算するデータが足りない（約定か 1 分足が 3 時間分そろうまで待つ）"}
    w = bin_sigma * sigma * ref
    vp_lv, vp = volume_profile(past, now_ms, ref, w, window_ms, candles, trade_from_ms)
    tpo_lv, tpo = tpo_profile(bars, now_ms, ref, w, window_ms, candle_ms, tpo_block)
    shown = [b for b in bars if b["t"] >= now_ms - window_ms and b["c"] is not None]
    cut = _approx_until(trade_from_ms, candle_ms) if candles else None
    return {"now_ms": now_ms, "price": ref, "sigma": sigma, "sigma_n": sigma_n, "bin_width": w, "window_h": window_h,
            "candle_ms": candle_ms, "tpo_block_min": candle_ms * tpo_block // 60_000,
            "approx_until_ms": cut if cut is not None and cut > now_ms - window_ms else None,
            "vp_levels": vp_lv, "vp": vp, "tpo_levels": tpo_lv, "tpo": tpo,
            "candles": [[b["t"], b["o"], b["h"], b["l"], b["c"]] for b in shown],
            "n_trades_window": sum(1 for t in past if t[0] >= now_ms - window_ms)}


# ---------------------------------------------------------------- 1 分ごとの水準（画面の右の列）

MIN_MS = 60_000
SIGNAL_KEYS = ("vp_poc_dist", "vp_va_pos", "vp_at_price", "tpo_poc_dist", "tpo_va_pos", "tpo_single_up")


def level_features(levels: dict, rows: list[dict], key: str, ref: float, sigma: float, w: float, prefix: str) -> dict:
    """水準を現在値からの相対値にする（bbresearch/direction.py の _levels_feats と同じ定義）。

    poc_dist: (POC − 現在値) / (σ × 現在値)。va_pos: バリューエリア内の位置（0 = VAL、1 = VAH）。
    at_price: 現在値の価格帯の量 / 量のある価格帯の平均。single_up: 現在値から上 1σ（4 価格帯）のうち TPO が 1 以下の割合。
    """
    out: dict = {}
    if not levels or not rows:
        return out
    s = sigma * ref
    out[f"{prefix}_poc_dist"] = (levels["poc"] - ref) / s
    span = levels["vah"] - levels["val"]
    out[f"{prefix}_va_pos"] = (ref - levels["val"]) / span if span > 0 else None
    occ = [r[key] for r in rows if r[key] > 0]
    here = next((r[key] for r in rows if r["lo"] <= ref < r["hi"]), 0.0)
    out[f"{prefix}_at_price"] = here / (sum(occ) / len(occ)) if occ else 0.0
    if prefix == "tpo":
        nb = round(1.0 / (w / s))  # 1σ = 4 価格帯
        up = [r[key] for r in rows if ref <= r["lo"] < ref + nb * w]
        out["tpo_single_up"] = (sum(1 for n in up if n <= 1) + (nb - len(up))) / nb
    return out


def signal_at(trades: list[tuple[int, float, float]], bars1: list[dict], t_ms: int,
              window_h: float = WINDOW_H, bin_sigma: float = 0.25, candles: list[list] | None = None,
              trade_from_ms: int | None = None) -> dict | None:
    """時刻 t の水準（t より前の約定と、t 以前に確定した足だけを使う。compute と同じ定義）。

    bars1 は σ と TPO に使う 1 分足。t を含む十分な範囲（t − 3 時間 − 5 分より前から）。
    """
    window_ms = int(window_h * 3_600_000)
    lo = _bisect_ts(trades, t_ms - window_ms)
    hi = _bisect_ts(trades, t_ms)
    if hi == 0:
        return None
    ref = trades[hi - 1][1]
    done = [b for b in bars1 if b["t"] + CANDLE_MS <= t_ms]
    sigma = sigma_from_bars(done[-(SIGMA_N + 1):], SIGMA_N)
    if sigma is None:
        return None
    w = bin_sigma * sigma * ref
    vp_lv, vp = volume_profile(trades[lo:hi], t_ms, ref, w, window_ms, candles, trade_from_ms)
    tpo_lv, tpo = tpo_profile(bars1, t_ms, ref, w, window_ms, CANDLE_MS, TPO_BLOCK)
    row = {"t": t_ms, "price": ref, "sigma": sigma}
    row.update(level_features(vp_lv, vp, "v", ref, sigma, w, "vp"))
    row.update(level_features(tpo_lv, tpo, "n", ref, sigma, w, "tpo"))
    return row


def _bisect_ts(trades, t_ms: int) -> int:
    """時刻順の trades で、ts >= t_ms となる最初の位置。"""
    lo, hi = 0, len(trades)
    while lo < hi:
        mid = (lo + hi) // 2
        if trades[mid][0] < t_ms:
            lo = mid + 1
        else:
            hi = mid
    return lo


def signals(trades: list[tuple[int, float, float]], now_ms: int, n_min: int = 60, known: dict | None = None,
            window_h: float = WINDOW_H, candles: list[list] | None = None, trade_from_ms: int | None = None) -> dict:
    """直近 n_min 分（分の開始時刻ごと）の水準。known に計算済みの {t: row} を渡すと、その分は計算しない。"""
    t_last = now_ms // MIN_MS * MIN_MS
    ts = [t_last - k * MIN_MS for k in range(n_min - 1, -1, -1)]
    window_ms = int(window_h * 3_600_000)
    back = max(window_ms + CANDLE_MS * TPO_BLOCK, CANDLE_MS * (SIGMA_N + 2))
    bars1 = bars_at(trades, ts[0] - back, t_last + CANDLE_MS, CANDLE_MS, candles, trade_from_ms)
    known = known if known is not None else {}
    out = []
    for t in ts:
        if t not in known:
            known[t] = signal_at(trades, bars1, t, window_h, candles=candles, trade_from_ms=trade_from_ms)
        if known[t] is not None:
            out.append(known[t])
    for t in [k for k in known if k < ts[0]]:  # 窓の外は捨てる
        del known[t]
    return {"now_ms": now_ms, "window_min": n_min, "keys": list(SIGNAL_KEYS), "minutes": out}
