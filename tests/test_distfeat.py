import numpy as np
import pandas as pd
import pytest

from bbresearch.distfeat import dist_bar_features, dist_event_features, prob_upper_first, t_df_from_kurtosis


def test_prob_upper_first_properties():
    assert prob_upper_first(0.0, 0.01, 0.02, 0.02) == pytest.approx(0.5)
    assert prob_upper_first(0.0, 0.01, 0.01, 0.03) == pytest.approx(0.75)
    # 対称性: μ → −μ、a ↔ b で P → 1 − P
    p = prob_upper_first(0.001, 0.01, 0.02, 0.03)
    q = prob_upper_first(-0.001, 0.01, 0.03, 0.02)
    assert p + q == pytest.approx(1.0)
    # ドリフトが大きいほど上に先に届きやすい
    ps = prob_upper_first(np.array([-0.002, -0.001, 0, 0.001, 0.002]), 0.01, 0.02, 0.02)
    assert np.all(np.diff(ps) > 0)
    # 極端な値でもあふれない
    assert prob_upper_first(1.0, 1e-4, 0.02, 0.02) == pytest.approx(1.0)
    assert prob_upper_first(-1.0, 1e-4, 0.02, 0.02) == pytest.approx(0.0)
    assert np.isnan(prob_upper_first(0.0, 0.0, 0.02, 0.02))


def test_prob_upper_first_matches_simulation():
    rng = np.random.default_rng(0)
    mu, sd, a, b, n = 0.0004, 0.004, 0.01, 0.008, 4000
    steps = rng.normal(mu, sd, size=(n, 3000)).cumsum(axis=1)
    up = (steps >= a).argmax(axis=1)
    dn = (steps <= -b).argmax(axis=1)
    hit_up = np.where((steps >= a).any(axis=1), up, 10**9)
    hit_dn = np.where((steps <= -b).any(axis=1), dn, 10**9)
    sim = float((hit_up < hit_dn).mean())
    # 離散時間のため連続時間の式と少しずれる（バリアを飛び越える分）
    assert sim == pytest.approx(float(prob_upper_first(mu, sd, a, b)), abs=0.05)


def test_t_df_moment_estimate():
    assert t_df_from_kurtosis(1.0) == pytest.approx(10.0)
    assert t_df_from_kurtosis(-0.5) == 100.0
    rng = np.random.default_rng(1)
    x = rng.standard_t(12, size=400_000)
    excess = pd.Series(x).kurt()
    assert t_df_from_kurtosis(excess) == pytest.approx(12, rel=0.25)


def make_bars(n=600, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.standard_t(5, n) * 0.002))
    return pd.DataFrame({"close": close, "is_empty": False}, index=idx)


def test_no_lookahead_and_windows():
    bars = make_bars()
    full = dist_bar_features(bars)
    part = dist_bar_features(bars.iloc[:400])
    pd.testing.assert_frame_equal(full.iloc[:400], part)
    # 変化の特徴量は 2k 本そろうまで NaN
    assert full["dist_dmu_t_96"].iloc[:191].isna().all() and full["dist_dmu_t_96"].iloc[191:].notna().all()


def test_event_features():
    bars = make_bars()
    feats = dist_bar_features(bars)
    sigma = pd.Series(0.004, index=bars.index)
    t0 = bars.index[300] + pd.Timedelta("15min")
    ev = pd.DataFrame({"event_id": ["e"], "t0": [t0]})
    X = dist_event_features(ev, feats, sigma, k_up=2.0, k_dn=2.0)
    row = feats.loc[bars.index[300]]
    assert X.loc["e", "dist_mu_t_16"] == row["dist_mu_t_16"]
    assert not any(c.startswith("_") for c in X.columns)
    assert 0.0 <= X.loc["e", "dist_pup_96"] <= 1.0
