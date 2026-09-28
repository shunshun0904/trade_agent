"""bbresearch/eventstudy.py: 候補 2 の事象研究の部品（先読みがないこと、統制のそろえ方、循環シフト、HAR 型の当てはめ直し）。"""
import numpy as np
import pandas as pd
import pytest

from bbresearch import eventstudy as ev

NS_H = 3_600_000_000_000


def test_grid_times_hours_and_end():
    t = ev.grid_times("2020-01-01", "2020-01-04", (0, 8, 16), "24h")
    assert list(t.hour[:3]) == [0, 8, 16] and t[0] == pd.Timestamp("2020-01-01", tz="UTC")
    assert t[-1] == pd.Timestamp("2020-01-03", tz="UTC")            # 2020-01-03 08:00 の 24 時間後は終わりを超える


def test_value_at_latest_known_with_staleness():
    x = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2020-01-01 04:00", "2020-01-01 12:00", "2020-01-02 04:00"], utc=True))
    t = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 08:00", "2020-01-01 12:00", "2020-01-02 00:00"], utc=True)
    v = ev.value_at(x, pd.DatetimeIndex(t), "8h")
    assert np.isnan(v[0]) and v[1] == 1.0 and v[2] == 2.0            # 同じ時刻の値は使える（その時刻に分かる）
    assert np.isnan(v[3])                                            # 12 時間前の値は古すぎる


def _closes(n=24 * 60, seed=0):
    idx = pd.date_range("2020-01-01 01:00", periods=n, freq="h", tz="UTC")   # 足が閉じる時刻
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)


def test_forward_return_and_prior_z_no_lookahead():
    c = _closes()
    t = ev.grid_times("2020-01-02", "2020-02-28", (0, 8, 16), "24h")
    y = ev.forward_log_return(c, t, "24h")
    assert y[0] == pytest.approx(np.log(c[t[0] + pd.Timedelta("24h")] / c[t[0]]))
    z = ev.prior_z(c, t, "24h", 240, 120)
    i = 60
    past = np.log(c[t[i]] / c[t[i] - pd.Timedelta("24h")])
    sd = np.log(c).diff().loc[:t[i]].tail(240).std() * np.sqrt(24)
    assert z[i] == pytest.approx(past / sd)
    c2 = c.copy()
    c2[c2.index > t[i]] *= 1.5                                       # T より後の終値を変えても T の z は変わらない
    assert ev.prior_z(c2, t, "24h", 240, 120)[i] == pytest.approx(z[i])


def test_zbins():
    assert ev.zbins(np.array([-3, -2, -1.5, 0, 1, 1.5, 2.5, np.nan]), [-2, -1, 1, 2]).tolist() == [0, 1, 1, 2, 3, 3, 4, -1]


def test_episodes_and_controls():
    t = np.arange(16) * 8 * NS_H                                       # 8 時間ごと
    evm = np.zeros(16, bool)
    evm[[2, 3, 4, 5, 12]] = True
    sel = ev.select_episodes(evm, t, 24 * NS_H)
    assert np.flatnonzero(sel).tolist() == [2, 5, 12]                  # 2 の 24 時間後の 5 は数える
    defined = np.ones(16, bool)
    defined[15] = False
    ctrl = ev.control_mask(evm, defined, t, 24 * NS_H)
    assert np.flatnonzero(ctrl).tolist() == [8, 9]                      # 前後 24 時間未満に事象がなく、信号がある時刻だけ
    assert ev.control_mask(np.zeros(16, bool), defined, t, 24 * NS_H).tolist() == defined.tolist()


def test_matched_effect_weights_by_event_bins():
    y = np.array([1.0, 2.0, 10.0, 20.0, 0.0, 5.0, 7.0, np.nan])
    zb = np.array([0, 0, 1, 1, 0, 1, 2, 0])
    sel = np.array([True, False, True, False, False, False, True, True])
    ctrl = np.array([False, True, False, True, True, True, False, False])
    r = ev.matched_effect(y, sel, ctrl, zb, 3)
    # 事象: 区分 0 の 1.0、区分 1 の 10.0（区分 2 は統制がないので除く、NaN も除く）。統制: 区分 0 の平均 1.0、区分 1 の平均 12.5
    assert r["n_ev"] == 2 and r["n_dropped"] == 1
    assert r["ev_mean"] == pytest.approx(5.5) and r["ctrl_mean"] == pytest.approx(0.5 * 1.0 + 0.5 * 12.5)
    assert r["effect"] == pytest.approx(5.5 - 6.75)


def test_planted_effect_is_found_and_null_is_not():
    n = 3 * 365 * 3
    t = pd.date_range("2020-01-01", periods=n, freq="8h", tz="UTC")
    t_ns = t.as_unit("ns").asi8
    rng = np.random.default_rng(1)
    rank = np.clip(pd.Series(rng.normal(size=n)).rolling(9, min_periods=1).mean().rank(pct=True).to_numpy(), 0, 1)
    y = rng.normal(0, 0.03, n)
    zb = rng.integers(0, 5, n)
    low = ev.event_mask(rank, 0.1, "low")
    y_eff = y + np.where(low, 0.02, 0.0)
    obs = ev.event_study(rank, y_eff, zb, t_ns, 0.1, "low", 24 * NS_H, 5)
    null = ev.shift_null(rank, y_eff, zb, t_ns, 0.1, "low", 24 * NS_H, 5, 199, 90, 0)
    assert obs["effect"] == pytest.approx(0.02, abs=0.006) and ev.p_one_sided(obs["effect"], null, 1) < 0.01
    obs0 = ev.event_study(rank, y, zb, t_ns, 0.1, "low", 24 * NS_H, 5)
    null0 = ev.shift_null(rank, y, zb, t_ns, 0.1, "low", 24 * NS_H, 5, 199, 90, 0)
    assert ev.p_one_sided(obs0["effect"], null0, 1) > 0.05
    assert ev.p_one_sided(obs["effect"], null, -1) > 0.95                 # 向きが逆なら p は大きい


def test_p_one_sided():
    null = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, np.nan])
    assert ev.p_one_sided(1.5, null, 1) == pytest.approx(2 / 6)
    assert ev.p_one_sided(-1.5, null, -1) == pytest.approx(2 / 6)
    assert np.isnan(ev.p_one_sided(np.nan, null, 1))


def _har_data(days=200, seed=2):
    idx = pd.date_range("2020-01-01", periods=days * 24, freq="h", tz="UTC")       # 足の開始時刻
    rng = np.random.default_rng(seed)
    vol = np.exp(np.cumsum(rng.normal(0, 0.05, len(idx))) * 0.2) * 0.005
    r = rng.standard_t(5, len(idx)) * vol / np.sqrt(5 / 3)
    rv = pd.Series(r ** 2 * rng.uniform(0.7, 1.3, len(idx)) + 1e-8, index=idx)
    lc = pd.Series(np.log(100) + np.cumsum(r), index=idx + pd.Timedelta("1h"))      # 足が閉じる時刻の対数終値
    return rv, lc


def test_har_walkforward_calibration_and_no_lookahead():
    rv, lc = _har_data()
    t = ev.grid_times("2020-03-01", "2020-07-18", (0, 8, 16), "24h")
    out = ev.har_walkforward(rv, lc, t, min_train_days=40)
    assert np.isfinite(out["sigma"]).all() and len(out["fits"]) >= 4
    y = ev.forward_log_return(np.exp(lc), t, "24h")
    rate = np.mean(y < out["q"][0.05])
    assert 0.01 < rate < 0.12                                           # 5% 点を下回る割合はおおむね 5%
    first = pd.Timestamp(out["fits"][0]["refit"])
    assert out["fits"][0]["n_train"] == int(((rv.index + pd.Timedelta("25h") <= first) & (rv.index >= rv.index[167])).sum())
    i = 150
    T = t[i]
    rv2, lc2 = rv.copy(), lc.copy()
    rv2[rv2.index >= T] *= 3.0                                          # T 以降に始まる足と、T より後に分かる終値を変える
    lc2[lc2.index > T] += 0.2
    out2 = ev.har_walkforward(rv2, lc2, t, min_train_days=40)
    assert np.allclose(out["sigma"][:i + 1], out2["sigma"][:i + 1]) and np.allclose(out["q"][0.05][:i + 1], out2["q"][0.05][:i + 1])
    assert not np.allclose(out["sigma"][i + 1:], out2["sigma"][i + 1:])
