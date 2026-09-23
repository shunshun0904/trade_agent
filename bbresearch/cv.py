"""Purged K-fold（エンバーゴ付き）。López de Prado (2018) 7 章。

サンプルは t0 の昇順に並んでいること。テスト fold は連続した区間とする。
- パージ: 学習サンプルのラベル期間 [t0, t_x] が、テスト区間 [テストの最小 t0, テストの最大 t_x]
  と重なるものを除外する。
- エンバーゴ: テスト区間の終わり（最大 t_x）の直後 embargo の間に t0 がある学習サンプルを除外する。
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


class PurgedKFold:
    def __init__(self, n_splits: int = 5, embargo: pd.Timedelta = pd.Timedelta(0)):
        if n_splits < 2:
            raise ValueError("n_splits は 2 以上")
        self.n_splits = n_splits
        self.embargo = pd.Timedelta(embargo)

    def split(self, t0: pd.Series, t_x: pd.Series) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        t0 = pd.DatetimeIndex(t0)
        t_x = pd.DatetimeIndex(t_x)
        if not t0.is_monotonic_increasing:
            raise ValueError("サンプルは t0 の昇順に並べてください")
        idx = np.arange(len(t0))
        for test in np.array_split(idx, self.n_splits):
            if len(test) == 0:
                continue
            lo = t0[test[0]]
            hi = t_x[test].max()
            overlap = (t0 <= hi) & (t_x >= lo)
            embargoed = (t0 > hi) & (t0 <= hi + self.embargo)
            train = idx[~overlap & ~embargoed]
            train = np.setdiff1d(train, test)
            yield train, test
