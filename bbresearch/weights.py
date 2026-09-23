"""サンプル重み: ラベル期間の重なりから平均ユニークネスを計算する（López de Prado 2018, 4.4 節）。

ラベル期間は [t_f, t_x]。時間軸は足の格子で離散化し、各足の同時ラベル数 c_t から
u_i = mean_{t ∈ [t_f, t_x]} 1 / c_t を求める。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def average_uniqueness(start: pd.Series, end: pd.Series, freq: str = "15min") -> pd.Series:
    """start / end は同じ index を持つ UTC 時刻。戻り値は同じ index の平均ユニークネス。"""
    if len(start) == 0:
        return pd.Series(dtype="float64", index=start.index)
    s = pd.DatetimeIndex(start).floor(freq)
    e = pd.DatetimeIndex(end).floor(freq)
    grid = pd.date_range(s.min(), e.max(), freq=freq)
    si = grid.get_indexer(s)
    ei = grid.get_indexer(e)
    # 差分配列で各足の同時ラベル数を数える
    delta = np.zeros(len(grid) + 1, dtype="int64")
    np.add.at(delta, si, 1)
    np.add.at(delta, ei + 1, -1)
    conc = np.cumsum(delta)[:-1]
    inv = np.where(conc > 0, 1.0 / np.maximum(conc, 1), 0.0)
    csum = np.concatenate([[0.0], np.cumsum(inv)])
    u = (csum[ei + 1] - csum[si]) / (ei - si + 1)
    return pd.Series(u, index=start.index)
