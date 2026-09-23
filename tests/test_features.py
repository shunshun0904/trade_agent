import numpy as np
import pandas as pd

from bbdata.bars import aggregate_trades, finalize_bars
from bbdata.download import load_transactions
from bbresearch.features import bar_features, event_features
from bbresearch.signals import cusum_events

from synth import synth_trades, write_trades


def bars_from(tmp_path, days=4):
    tr = synth_trades("2026-01-01", days, seed=3, per_min=2)
    write_trades(tmp_path, "btc_jpy", tr)
    t = load_transactions(tmp_path, "btc_jpy", "2026-01-01", f"2026-01-0{1 + days}")
    return finalize_bars(aggregate_trades(t, "15min", "2026-01-01", f"2026-01-0{1 + days}", large_trade_amount=0.15))


def test_min_periods_gives_nan_until_window_full(tmp_path):
    f = bar_features(bars_from(tmp_path))
    assert f["ret_96"].iloc[:96].isna().all() and f["ret_96"].iloc[96:].notna().all()
    assert f["rv_16"].iloc[:15].isna().all() and np.isfinite(f["rv_16"].iloc[16])
    assert "large_share_16" in f


def test_no_lookahead_in_features(tmp_path):
    bars = bars_from(tmp_path)
    full = bar_features(bars)
    part = bar_features(bars.iloc[:250])
    pd.testing.assert_frame_equal(full.iloc[:250], part)


def test_event_features_use_bar_ending_at_t0(tmp_path):
    bars = bars_from(tmp_path)
    feats = bar_features(bars)
    ev = cusum_events(bars, "btc_jpy", 96, 1.0)
    assert len(ev) > 0
    X = event_features(ev, feats)
    e = ev.iloc[0]
    row = feats.loc[e.t0 - pd.Timedelta("15min")]
    assert X.loc[e.event_id, "ret_1"] == row["ret_1"]
    assert np.isclose(X.loc[e.event_id, "cusum_ratio"], abs(e.s_value) / e.h)
    assert "bar_sigma" not in X
