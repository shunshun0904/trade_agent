import numpy as np
import pandas as pd

from bbresearch.signals import cusum_events, ewm_sigma, log_returns


def make_bars(closes, start="2026-01-01", empty=None):
    idx = pd.date_range(start, periods=len(closes), freq="15min", tz="UTC")
    b = pd.DataFrame({"close": np.asarray(closes, dtype=float)}, index=idx)
    b["is_empty"] = False if empty is None else empty
    return b


def test_log_returns_zero_on_empty_bars():
    b = make_bars([100, 110, 110, 121], empty=[False, False, True, False])
    r = log_returns(b)
    assert r.iloc[0] == 0 and r.iloc[2] == 0
    assert np.isclose(r.iloc[3], np.log(121 / 110))


def test_cusum_detects_dip_and_breakout_at_expected_times():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.001, 300)
    r[200] = -0.02   # 急落 → dip
    r[250] = 0.02    # 急騰 → breakout
    b = make_bars(100 * np.exp(np.cumsum(r)))
    ev = cusum_events(b, "btc_jpy", sigma_span=48, k_h=3.0)
    step = pd.Timedelta("15min")
    dips = ev[ev.signal_type == "dip"]
    brk = ev[ev.signal_type == "breakout"]
    assert (b.index[200] + step) in set(dips.t0)
    assert (b.index[250] + step) in set(brk.t0)
    # σ が定まるまで（最初の span 本）はイベントを出さない
    assert ev.t0.min() >= b.index[47] + step
    assert (ev.s_value.abs() >= ev.h).all()


def test_cusum_count_on_deterministic_series():
    # σ を一定に近づけるため、±x を交互に並べた後に一方向の小さな動きを加える
    r = np.tile([0.001, -0.001], 100).astype(float)
    r = np.concatenate([r, np.full(20, 0.0005)])
    b = make_bars(100 * np.exp(np.cumsum(r)))
    ev = cusum_events(b, "x", sigma_span=20, k_h=2.0)
    assert (ev.signal_type == "breakout").sum() >= 1
    assert (ev.t0 > b.index[199]).all()


def test_no_lookahead_appending_future_does_not_change_past():
    rng = np.random.default_rng(1)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 600)))
    b_full = make_bars(closes)
    b_part = b_full.iloc[:400]
    ev_full = cusum_events(b_full, "x", 96, 2.0)
    ev_part = cusum_events(b_part, "x", 96, 2.0)
    cut = b_part.index[-1] + pd.Timedelta("15min")
    pd.testing.assert_frame_equal(
        ev_full[ev_full.t0 <= cut].reset_index(drop=True), ev_part.reset_index(drop=True)
    )
    pd.testing.assert_series_equal(ewm_sigma(b_full, 96).iloc[:400], ewm_sigma(b_part, 96))
