import math

import numpy as np
import pandas as pd
import yaml

from bbresearch.swing import (BARS_PER_DAY, bar_sigma, configs, crash_events, daily, decisions, deflate, eligible,
                              evaluate, hac_alpha, judge, panel, risk_parity, simulate, stagger, tsmom_signal,
                              xsmom_weights)

SMALL = {"min_bars": 60, "turnover_days": 5, "min_turnover_jpy": 1e6, "sigma_days": 5, "hold_days": [1, 2],
         "tsmom": {"kinds": ["ret", "ma"], "lookback_days": [3, 10]},
         "xsmom": {"lookback_days": [3], "top_k": 1},
         "rebound": {"window_bars": [6], "k_sigma": [2]}}


def _prices(n=1500, pairs=("a", "b", "c"), seed=0, drift=None):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
    r = rng.normal(0, 0.01, (n, len(pairs)))
    if drift is not None:
        r += drift
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=list(pairs))
    vol = pd.DataFrame(1e6, index=idx, columns=list(pairs))
    return close, vol


def _all_outputs(close, vol):
    elig = eligible(close, vol, SMALL["min_bars"], SMALL["turnover_days"], SMALL["min_turnover_jpy"])
    base = risk_parity(bar_sigma(close, SMALL["sigma_days"]), elig)
    out = {"elig": elig.astype(float), "base": base,
           "ma": tsmom_signal(close, 3, "ma").astype(float), "ret": tsmom_signal(close, 3, "ret").astype(float),
           "xs": xsmom_weights(close, elig, 3, 1), "crash": crash_events(close, 6, 2, 5).astype(float)}
    for c in configs(SMALL):
        out[c["key"]] = stagger(decisions(c, close, elig, base, SMALL), c["hold_days"] * BARS_PER_DAY)
    return out


def test_no_lookahead_in_signals_eligibility_weights_and_holdings():
    close, vol = _prices()
    t0 = close.index[900]
    before = _all_outputs(close, vol)
    c2, v2 = close.copy(), vol.copy()
    rng = np.random.default_rng(1)
    c2.loc[c2.index > t0] *= np.exp(rng.normal(0, 0.2, c2.loc[c2.index > t0].shape))
    v2.loc[v2.index > t0] = 0.0
    after = _all_outputs(c2, v2)
    for k in before:
        pd.testing.assert_frame_equal(before[k].loc[:t0], after[k].loc[:t0], check_names=False, obj=k)


def test_stagger_is_the_average_of_offset_subportfolios():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2021-01-01", periods=40, freq="4h", tz="UTC")
    w = pd.DataFrame(rng.random((40, 2)), index=idx, columns=["a", "b"])
    hold = 3
    subs = []
    for k in range(hold):
        s = pd.DataFrame(0.0, index=idx, columns=w.columns)
        last = None
        for i in range(len(idx)):
            if i % hold == k:
                last = w.iloc[i]
            if last is not None:
                s.iloc[i] = last
        subs.append(s)
    pd.testing.assert_frame_equal(stagger(w, hold), sum(subs) / hold)


def test_simulate_returns_and_costs_by_hand():
    idx = pd.date_range("2021-01-01", periods=4, freq="4h", tz="UTC")
    close = pd.DataFrame({"a": [100.0, 110.0, 99.0, 99.0]}, index=idx)
    w = pd.DataFrame({"a": [1.0, 1.0, 0.0, 0.0]}, index=idx)
    sim = simulate(w, close, {"a": 0.001})
    assert np.allclose(sim["net"].to_numpy(), [0.0, 0.1 - 0.001, -0.1, -0.001])
    assert np.allclose(sim["exposure"].to_numpy(), [0, 1, 1, 0])


def test_eligibility_needs_history_and_turnover():
    close, vol = _prices(n=400, pairs=("a",))
    vol.iloc[:200] = 1e3  # 最初は出来高が少ない
    elig = eligible(close, vol, min_bars=60, turnover_days=5, min_turnover=1e6)["a"]
    assert not elig.iloc[:200].any()
    first = elig.idxmax()
    # 24 時間の出来高が閾値を超えた足が 30 本の中央値を占めるまでは対象にならない
    assert close.index[200] < first <= close.index[200 + 5 * BARS_PER_DAY]
    # 取引が止まったら（直近 24 時間の出来高 0）、中央値が下がるのを待たずに外れる
    vol.iloc[350:] = 0.0
    elig = eligible(close, vol, min_bars=60, turnover_days=5, min_turnover=1e6)["a"]
    assert elig.iloc[349] and not elig.iloc[350 + BARS_PER_DAY - 1:].any()


def test_panel_marks_unlisted_and_fills_gaps():
    idx = pd.date_range("2021-01-01 08:00", periods=5, freq="4h", tz="UTC")
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0], "volume": [1.0] * 5}, index=idx).drop(idx[2])
    close, vol = panel({"a": df}, "2021-01-01", "2021-01-02 04:00")
    assert close["a"].iloc[:2].isna().all()                      # 最初の足より前
    assert close.loc[idx[2], "a"] == 2.0 and vol.loc[idx[2], "a"] == 0.0  # 抜けた足


def test_crash_events_fire_only_after_a_large_drop():
    close, _ = _prices(n=600, pairs=("a",), seed=3)
    close.iloc[500:] *= 0.8  # 1 本で -20%（1 本の σ は 1%。-5σ√6 ≈ -12% を超える）
    ev = crash_events(close, bars=6, k=5, sigma_days=5)["a"]
    assert not ev.iloc[:500].any()
    assert ev.iloc[500:506].all()


def test_hac_alpha_recovers_alpha_and_beta():
    rng = np.random.default_rng(4)
    idx = pd.date_range("2020-01-01", periods=3000, freq="D", tz="UTC")
    x = pd.Series(rng.normal(0, 0.02, 3000), index=idx)
    y = 0.002 + 0.5 * x + pd.Series(rng.normal(0, 0.01, 3000), index=idx)
    r = hac_alpha(y, x)
    assert abs(r["alpha"] - 0.002) < 0.0005 and abs(r["beta"] - 0.5) < 0.03 and r["t_alpha"] > 5
    r0 = hac_alpha(0.5 * x + pd.Series(rng.normal(0, 0.01, 3000), index=idx), x)
    assert abs(r0["t_alpha"]) < 3


def test_evaluate_finds_momentum_only_when_trends_persist():
    n = 6000
    rng = np.random.default_rng(5)
    # 60 日ごとに向きが変わるドリフト（トレンドが続く市場）
    regime = np.repeat(rng.choice([-1.0, 1.0], size=n // 360 + 1), 360)[:n]
    drift = np.column_stack([regime * 0.003] * 3)
    close_t, vol_t = _prices(n=n, seed=6, drift=drift)
    close_r, vol_r = _prices(n=n, seed=7)
    cost = {p: 0.001 for p in close_t.columns}
    trend = evaluate(close_t, vol_t, cost, SMALL)["results"]
    rw = evaluate(close_r, vol_r, cost, SMALL)["results"]
    long_tsmom = [k for k in trend if k.startswith("tsmom") and "L10d" in k]
    assert all(trend[k]["alpha_ann"] > 0 and trend[k]["t_alpha"] > 2 for k in long_tsmom)
    assert max(abs(rw[k]["t_alpha"]) for k in rw if k.startswith("tsmom")) < 3.5
    dsr = deflate(trend)
    j = judge(trend, dsr, "tsmom", min_share=2 / 3, min_dsr=0.95, min_year_share=0.6,
              full_years=[2021, 2022])
    assert j["supported"]


def test_daily_compounds_bars_within_utc_days():
    idx = pd.date_range("2021-01-01", periods=12, freq="4h", tz="UTC")
    r = pd.Series(0.01, index=idx)
    d = daily(r)
    assert len(d) == 2 and math.isclose(d.iloc[0], 1.01 ** 6 - 1)


def test_preregistered_config_has_45_trials():
    with open("configs/swing.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    keys = [c["key"] for c in configs(cfg)]
    assert len(keys) == len(set(keys)) == 45
