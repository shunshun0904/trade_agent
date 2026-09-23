"""Phase 2: 一次シグナル（対称 CUSUM フィルタ）。

López de Prado (2018) Snippet 2.4 を基にする。期待リターンは 0 とみなす。

    r_t  = ln(close_t / close_{t-1})        約定なしの足では 0
    σ_t  = r の EWMA 標準偏差（span = sigma_span、t までのデータのみ）
    h_t  = k_h × σ_t
    S⁺_t = max(0, S⁺_{t-1} + r_t)、S⁻_t = min(0, S⁻_{t-1} + r_t)
    S⁻_t ≤ -h_t → "dip"、S⁺_t ≥ h_t → "breakout"（出したほうを 0 に戻す）

σ は min_periods = sigma_span とし、σ が定まるまでの区間ではイベントを出さず、
S⁺ / S⁻ も累積しない（σ が定まった足から累積を始める）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_COLUMNS = ["event_id", "pair", "t0", "signal_type", "s_value", "h", "sigma"]


def log_returns(bars: pd.DataFrame) -> pd.Series:
    """足の対数リターン。約定なしの足と、先頭の足は 0。"""
    r = np.log(bars["close"]).diff()
    if "is_empty" in bars:
        r = r.where(~bars["is_empty"].astype(bool), 0.0)
    return r.fillna(0.0)


def ewm_sigma(bars: pd.DataFrame, span: int) -> pd.Series:
    """各足の終了時点までのデータで計算した σ（EWMA 標準偏差）。"""
    return log_returns(bars).ewm(span=span, min_periods=span, adjust=True).std()


def cusum_events(bars: pd.DataFrame, pair: str, sigma_span: int = 96, k_h: float = 2.0) -> pd.DataFrame:
    """events テーブルを返す。t0 はイベント足の終了時刻（= 足の開始時刻 + 足の長さ）。"""
    if len(bars.index) < 2:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    step = bars.index[1] - bars.index[0]
    r = log_returns(bars).to_numpy()
    sigma = ewm_sigma(bars, sigma_span).to_numpy()
    rows = []
    s_pos = s_neg = 0.0
    for i in range(len(r)):
        sig = sigma[i]
        if not np.isfinite(sig) or sig <= 0:
            continue
        h = k_h * sig
        s_pos = max(0.0, s_pos + r[i])
        s_neg = min(0.0, s_neg + r[i])
        t0 = bars.index[i] + step
        if s_neg <= -h:
            rows.append((pair, t0, "dip", s_neg, h, sig))
            s_neg = 0.0
        if s_pos >= h:
            rows.append((pair, t0, "breakout", s_pos, h, sig))
            s_pos = 0.0
    ev = pd.DataFrame(rows, columns=EVENT_COLUMNS[1:])
    ev.insert(0, "event_id", [f"{pair}-{t:%Y%m%dT%H%M}-{s}" for t, s in zip(ev["t0"], ev["signal_type"])])
    return ev
