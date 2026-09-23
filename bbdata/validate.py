"""自前集計の足と公式ロウソク足の突き合わせ。

公式ロウソク足の timestamp が足の開始時刻か終了時刻かは、ドキュメントに明記がない。
そのため ±max_shift 本ずらして比較し、最も一致するずれを確認できるようにしている。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OHLC = ["open", "high", "low", "close"]


def compare_with_official(bars: pd.DataFrame, candles: pd.DataFrame, max_shift: int = 1) -> pd.DataFrame:
    """ずれ（本数）ごとの一致率を返す。約定のあった足だけを比較する。"""
    if len(bars.index) < 2:
        raise ValueError("bars が2本以上必要です")
    step = bars.index[1] - bars.index[0]
    ours = bars.loc[~bars["is_empty"], OHLC + ["volume"]]
    rows = []
    for k in range(-max_shift, max_shift + 1):
        official = candles[OHLC + ["volume"]].copy()
        official.index = official.index + k * step
        j = ours.join(official, rsuffix="_official", how="inner")
        if j.empty:
            rows.append({"shift_bars": k, "n": 0})
            continue
        match = {c: np.isclose(j[c], j[f"{c}_official"], rtol=1e-12, atol=0.0) for c in OHLC}
        rel_err = (j["volume"] - j["volume_official"]).abs() / j["volume_official"].where(j["volume_official"] > 0)
        rows.append(
            {
                "shift_bars": k,
                "n": len(j),
                "close_match_rate": float(match["close"].mean()),
                "ohlc_match_rate": float(np.logical_and.reduce(list(match.values())).mean()),
                "volume_rel_err_median": float(rel_err.median()),
                "volume_rel_err_p99": float(rel_err.quantile(0.99)),
            }
        )
    return pd.DataFrame(rows).set_index("shift_bars")
