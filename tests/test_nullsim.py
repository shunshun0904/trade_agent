"""bbresearch/nullsim.py: 帰無の監査用の合成データ（期待リターン 0）と、その数値の当てはめ。"""
import numpy as np
import pandas as pd

from bbresearch import nullsim

CAL = {"garch": {"mu": 0.0, "omega": 2e-7, "alpha": 0.08, "beta": 0.9, "nu": 4.0},
       "volume": {"a": 6.0, "b": 1.0, "phi": 0.6, "sd": 0.5}, "p0": 3e6,
       "ms2": {"sigma": [0.003, 0.009], "P": [[0.995, 0.005], [0.02, 0.98]]}}
IDX = pd.date_range("2021-01-01", "2024-01-01", freq="h", tz="UTC", inclusive="left")


def test_paths_are_consistent_and_have_zero_mean():
    for kind in ("garch_t", "ms2"):
        h1, mc = nullsim.simulate(IDX, kind, CAL, 3)
        assert len(mc) == 60 * len(h1)
        assert (h1["low"] <= h1[["open", "close"]].min(axis=1)).all() and (h1["high"] >= h1[["open", "close"]].max(axis=1)).all()
        assert np.allclose(mc.iloc[59::60].to_numpy(), h1["close"].to_numpy(), rtol=1e-12)   # 1 時間の最後の分の終値 = 足の終値
        assert mc.index[60] == h1.index[1] and (h1["volume"] > 0).all()
        r = np.log(h1["close"]).diff().dropna()
        assert abs(r.mean()) < 3 * r.std() / np.sqrt(len(r))
        assert r.pow(2).autocorr(1) > 0.05                                # ボラが固まって動く
    # 同じ種なら同じ系列
    a, _ = nullsim.simulate(IDX[:500], "garch_t", CAL, 9)
    b, _ = nullsim.simulate(IDX[:500], "garch_t", CAL, 9)
    pd.testing.assert_frame_equal(a, b)


def test_calibration_recovers_the_generating_values():
    h1, _ = nullsim.simulate(IDX, "garch_t", CAL, 4)
    cal = nullsim.calibrate(h1, "2024-01-01")
    g = cal["garch"]
    assert abs(g["alpha"] - 0.08) < 0.03 and abs(g["beta"] - 0.9) < 0.04 and abs(g["nu"] - 4.0) < 1.0
    v = cal["volume"]
    assert abs(v["b"] - 1.0) < 0.1 and abs(v["phi"] - 0.6) < 0.1
    h2, _ = nullsim.simulate(IDX, "ms2", CAL, 5)
    m = nullsim.fit_ms2(np.log(h2["close"]).diff().dropna().to_numpy())
    assert abs(m["sigma"][0] / 0.003 - 1) < 0.1 and abs(m["sigma"][1] / 0.009 - 1) < 0.1
    assert abs(m["P"][0][0] - 0.995) < 0.005


def test_synthetic_garch_is_made_stationary_with_the_training_variance():
    g = {"mu": 0.0, "omega": 1e-7, "alpha": 0.0877, "beta": 0.9123, "nu": 3.5}   # 2026-09-28 の check と同じく α + β = 1
    var = 0.007 ** 2
    s = nullsim.stationary_garch(g, var)
    assert abs(s["alpha"] + s["beta"] - nullsim.MAX_PERSISTENCE) < 1e-12
    assert abs(s["alpha"] / s["beta"] - g["alpha"] / g["beta"]) < 1e-12
    assert abs(s["omega"] / (1 - s["alpha"] - s["beta"]) - var) < 1e-15
    cal = CAL | {"garch": s}
    h1, _ = nullsim.simulate(IDX, "garch_t", cal, 6)
    r = np.log(h1["close"]).diff().dropna()
    assert 0.6 < r.std() / 0.007 < 1.4
    # 境界にない値はそのまま（ω だけ分散に合わせる）
    s2 = nullsim.stationary_garch(CAL["garch"], 0.004 ** 2)
    assert s2["alpha"] == 0.08 and s2["beta"] == 0.9
