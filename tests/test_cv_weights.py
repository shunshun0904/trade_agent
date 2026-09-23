import numpy as np
import pandas as pd

from bbresearch.cv import PurgedKFold
from bbresearch.weights import average_uniqueness


def test_average_uniqueness():
    t = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:15", "2026-01-01 02:00"], utc=True)
    e = pd.to_datetime(["2026-01-01 00:15", "2026-01-01 00:30", "2026-01-01 02:00"], utc=True)
    u = average_uniqueness(pd.Series(t), pd.Series(e))
    # 1本目: 00:00 は単独(1)、00:15 は2本重なる(1/2) → 0.75。2本目も同様。3本目は単独
    assert np.allclose(u.to_numpy(), [0.75, 0.75, 1.0])


def test_purged_kfold_removes_overlap_and_embargo():
    t0 = pd.Series(pd.date_range("2026-01-01", periods=20, freq="1h", tz="UTC"))
    t_x = t0 + pd.Timedelta("150min")  # 各ラベルは次の2本と重なる
    cv = PurgedKFold(n_splits=4, embargo=pd.Timedelta("2h"))
    for tr, te in cv.split(t0, t_x):
        lo, hi = t0[te[0]], t_x[te].max()
        for i in tr:
            assert t_x[i] < lo or t0[i] > hi + pd.Timedelta("2h")
        assert len(np.intersect1d(tr, te)) == 0
    folds = list(cv.split(t0, t_x))
    # 2番目の fold のテスト = 5..9。直前の 3, 4 はラベル期間が重なるので除外、10〜 もパージ・エンバーゴで除外
    tr, te = folds[1]
    assert list(te) == [5, 6, 7, 8, 9]
    assert 4 not in tr and 3 not in tr and 2 in tr
    assert 10 not in tr and 13 not in tr and 14 in tr
