"""価格帯別出来高と TPO の計算（標準ライブラリだけで書いたダッシュボード用の実装）。

研究用の bbresearch/profile.py と同じ定義にしている（テストで照合する）:
- 区間は [now − window, now)。
- 価格帯の刻みは w = bin_sigma × σ × 基準価格（基準価格は now 直前の約定価格）。境界は基準価格に揃える。
- バリューエリアは POC から隣の多い側へ広げて全体の 70% に達するまで（同量なら上側）。
- TPO は 15 分足を now から過去へ 2 本ずつまとめた 30 分区間ごとに、安値〜高値にかかる価格帯へ 1 を数える。
σ は直近 96 本の 15 分足の対数リターンの標準偏差（研究用の EWMA とは異なる。表示用の近似）。
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


def bars_15m(trades: list[tuple[int, float, float]], start_ms: int, end_ms: int) -> list[dict]:
    """[start, end) の 15 分足（約定なしの足は直前の close で埋める）。trades は (ts, price, amount) の時刻順。"""
    n = (end_ms - start_ms) // BAR_MS
    out = [{"t": start_ms + i * BAR_MS, "o": None, "h": None, "l": None, "c": None, "v": 0.0} for i in range(n)]
    for ts, px, amt in trades:
        if not (start_ms <= ts < start_ms + n * BAR_MS):
            continue
        b = out[(ts - start_ms) // BAR_MS]
        if b["o"] is None:
            b["o"] = b["h"] = b["l"] = px
        b["h"] = max(b["h"], px)
        b["l"] = min(b["l"], px)
        b["c"] = px
        b["v"] += amt
    last = None
    for b in out:
        if b["c"] is None:
            if last is not None:
                b["o"] = b["h"] = b["l"] = b["c"] = last
        else:
            last = b["c"]
    return out


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


def volume_profile(trades, now_ms: int, ref: float, w: float, window_ms: int) -> tuple[dict, list[dict]]:
    acc: dict[int, float] = {}
    for ts, px, amt in trades:
        if now_ms - window_ms <= ts < now_ms:
            k = math.floor(round((px - ref) / w, 9))
            acc[k] = acc.get(k, 0.0) + amt
    if not acc:
        return {}, []
    first, last = min(acc), max(acc)
    counts = [acc.get(k, 0.0) for k in range(first, last + 1)]
    rows = [{"lo": ref + k * w, "hi": ref + (k + 1) * w, "v": acc.get(k, 0.0)} for k in range(first, last + 1)]
    return _levels(counts, first, ref, w), rows


def tpo_profile(bars: list[dict], now_ms: int, ref: float, w: float, window_ms: int) -> tuple[dict, list[dict]]:
    """bars は 15 分足。終了時刻が now 以前の足を、now から過去へ 2 本ずつまとめる。"""
    done = [b for b in bars if b["t"] + BAR_MS <= now_ms and b["t"] >= now_ms - window_ms and b["h"] is not None]
    n = len(done) // 2 * 2
    done = done[len(done) - n:]
    acc: dict[int, int] = {}
    for i in range(0, n, 2):
        hi = max(done[i]["h"], done[i + 1]["h"])
        lo = min(done[i]["l"], done[i + 1]["l"])
        for k in range(math.floor(round((lo - ref) / w, 9)), math.floor(round((hi - ref) / w, 9)) + 1):
            acc[k] = acc.get(k, 0) + 1
    if not acc:
        return {}, []
    first, last = min(acc), max(acc)
    counts = [float(acc.get(k, 0)) for k in range(first, last + 1)]
    rows = [{"lo": ref + k * w, "hi": ref + (k + 1) * w, "n": acc.get(k, 0)} for k in range(first, last + 1)]
    return _levels(counts, first, ref, w), rows


def compute(trades: list[tuple[int, float, float]], now_ms: int, window_h: float = 24.0,
            bin_sigma: float = 0.25) -> dict:
    """ダッシュボードに渡す一式。trades は (ts, price, amount) の時刻順で、σ の計算に 24 時間以上前から含むこと。"""
    window_ms = int(window_h * 3_600_000)
    past = [t for t in trades if t[0] < now_ms]
    if not past:
        return {"error": "約定がない"}
    ref = past[-1][1]
    end = now_ms // BAR_MS * BAR_MS + BAR_MS  # 形成中の足まで（表示用）
    bars = bars_15m(past, end - 2 * window_ms - BAR_MS, end)
    sigma = sigma_from_bars([b for b in bars if b["t"] + BAR_MS <= now_ms])
    if sigma is None:
        return {"error": "σ を計算するデータが足りない"}
    w = bin_sigma * sigma * ref
    vp_lv, vp = volume_profile(past, now_ms, ref, w, window_ms)
    tpo_lv, tpo = tpo_profile(bars, now_ms, ref, w, window_ms)
    shown = [b for b in bars if b["t"] >= now_ms - window_ms and b["c"] is not None]
    return {"now_ms": now_ms, "price": ref, "sigma": sigma, "bin_width": w, "window_h": window_h,
            "vp_levels": vp_lv, "vp": vp, "tpo_levels": tpo_lv, "tpo": tpo,
            "candles": [[b["t"], b["o"], b["h"], b["l"], b["c"]] for b in shown],
            "n_trades_window": sum(1 for t in past if t[0] >= now_ms - window_ms)}
