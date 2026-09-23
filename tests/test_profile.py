import numpy as np
import pandas as pd
import pytest

from bbresearch.labeling import TradeTape
from bbresearch.profile import PROFILE_COLUMNS, profile_event_features, profile_features_at, value_area

T0 = pd.Timestamp("2026-01-02 00:00", tz="UTC")
MS = lambda t: int(t.value // 1_000_000)  # noqa: E731


def tape(rows):
    """rows: (時刻, 価格, 数量)"""
    ts, px, amt = zip(*rows)
    return TradeTape(np.array([MS(t) for t in ts], dtype="int64"), np.zeros(len(ts), bool),
                     np.array(px, float), np.array(amt, float))


def flat_bars(start="2026-01-01", periods=192, price=100.0):
    idx = pd.date_range(start, periods=periods, freq="15min", tz="UTC")
    return pd.DataFrame({"high": price, "low": price, "close": price}, index=idx)


def test_value_area_expands_toward_larger_side():
    assert value_area(np.array([1.0, 5, 10, 3, 1])) == (2, 1, 2)
    assert value_area(np.array([0.0, 0, 10, 0, 0])) == (2, 2, 2)


def test_volume_profile_in_sigma_units():
    # close = 100、σ = 0.01 → 1σ = 1 円、bin_sigma = 0.5 → 刻み 0.5 円
    rows = [(T0 - pd.Timedelta(hours=1), 101.2, 5.0),   # +1.2σ → 価格帯 [1.0σ, 1.5σ)（中心 +1.25σ）
            (T0 - pd.Timedelta(hours=2), 100.2, 1.0),   # 価格帯 0
            (T0 - pd.Timedelta(hours=3), 99.6, 1.0),    # 価格帯 -1
            (T0 - pd.Timedelta(hours=30), 50.0, 99.0),  # 24 時間より前 → 数えない
            (T0, 100.0, 99.0)]                           # t0 ちょうど → 数えない
    rows.sort()
    f = profile_features_at(tape(rows), flat_bars(), T0, 100.0, 0.01, k_up=1.0, k_dn=1.0, bin_sigma=0.5)
    assert f["vp_poc_dist"] == pytest.approx(1.25)
    assert f["vp_to_upper"] == pytest.approx(1 / 7)   # [0, +1σ) にあるのは 100.2 の 1
    assert f["vp_to_lower"] == pytest.approx(1 / 7)   # [-1σ, 0) にあるのは 99.6 の 1
    assert f["vp_at_price"] == pytest.approx(1 / (7 / 3))
    assert set(f) == set(PROFILE_COLUMNS)


def test_tpo_counts_30min_periods():
    bars = flat_bars()
    # t0 直前の 30 分（2 本）だけ 99〜103 に広げる。残りは 100 に張り付き
    bars.loc[T0 - pd.Timedelta("30min"):T0 - pd.Timedelta("15min"), ["low", "high"]] = [99.0, 103.0]
    rows = [(T0 - pd.Timedelta(hours=1), 100.0, 1.0)]
    f = profile_features_at(tape(rows), bars, T0, 100.0, 0.01, k_up=2.0, k_dn=1.0, bin_sigma=1.0)
    # POC は close の価格帯（48 区間すべてが触れている）
    assert f["tpo_poc_dist"] == pytest.approx(0.5)
    # close から +2σ（U）の間の価格帯 [0,1) と [1,2): 前者は 48、後者は 1 → シングルプリントは半分
    assert f["tpo_single_up"] == pytest.approx(0.5)
    # L 側 [-1,0) は直前の 30 分だけ → 1 以下
    assert f["tpo_single_dn"] == pytest.approx(1.0)


def test_no_lookahead():
    rng = np.random.default_rng(0)
    base = T0 - pd.Timedelta(hours=26)
    times = sorted(base + pd.to_timedelta(rng.integers(0, 26 * 3600, 3000), unit="s"))
    rows = [(t, 100 + rng.normal(0, 1), rng.uniform(0.01, 1)) for t in times]
    bars = flat_bars(start=str(base.floor("D").date()), periods=400)
    bars["high"] = bars["close"] + rng.uniform(0, 2, len(bars))
    bars["low"] = bars["close"] - rng.uniform(0, 2, len(bars))
    ev = pd.DataFrame({"event_id": ["e"], "t0": [T0]})
    sig = pd.Series(0.01, index=bars.index)
    a = profile_event_features(ev, bars, tape(rows), sig, 2.0, 2.0)
    # t0 以降の約定と足を変えても変わらない
    later = rows + [(T0 + pd.Timedelta(minutes=m), 150.0, 50.0) for m in range(0, 60, 5)]
    bars2 = bars.copy()
    bars2.loc[bars2.index >= T0, ["high", "low", "close"]] = [300.0, 1.0, 150.0]
    b = profile_event_features(ev, bars2, tape(later), sig, 2.0, 2.0)
    pd.testing.assert_frame_equal(a, b)
    assert a.notna().all(axis=None)
