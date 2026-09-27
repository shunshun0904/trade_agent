"""OHLCV の足からテクニカル指標の表を作る。2026-09-27 オーナー指示（同日に TradingView / bitbank の指標に合わせて拡充）。

行 t の特徴量は足 t（とそれ以前）だけから計算する（足 t の終値で判断する想定）。先読みがないことは
tests/test_indicators.py で確かめる（未来の足を書き換えても過去の行が変わらない）。

値は価格の水準に依存しない形（比率・対数差・0〜100 の振れ）にそろえる。列名の接頭辞:
  t_ トレンド系（移動平均、MACD、DMI、一目均衡表 など）
  o_ オシレーター系（RSI、ストキャスティクス、CCI など）
  v_ ボラティリティ・出来高
  p_ 水準・パターン（ピボット、フラクタル、ジグザグ、VWAP、価格帯別出来高、ローソク足のパターン）
  r_ TradingView のテクニカル評価に準じた集計
  c_ 時刻
部品は bbresearch/ta.py。TradingView の一覧（Technicals）に合わせた対応表は docs/SPEC.md。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bbresearch import ta
from bbresearch.ta import EPS, rsi  # noqa: F401  （rsi はテストが参照する）

GROUPS = {"t": "トレンド", "o": "オシレーター", "v": "ボラ・出来高", "p": "水準・パターン", "r": "テクニカル評価", "c": "時刻"}

MA_TYPES = {
    "sma": lambda c, v, n: ta.sma(c, n), "ema": lambda c, v, n: ta.ema(c, n), "wma": lambda c, v, n: ta.wma(c, n),
    "dema": lambda c, v, n: ta.dema(c, n), "tema": lambda c, v, n: ta.tema(c, n), "hma": lambda c, v, n: ta.hma(c, n),
    "rma": lambda c, v, n: ta.rma(c, n), "lsma": lambda c, v, n: ta.lsma(c, n), "alma": lambda c, v, n: ta.alma(c, n),
    "mcg": lambda c, v, n: ta.mcginley(c, n), "vwma": lambda c, v, n: ta.vwma(c, v, n), "kama": lambda c, v, n: ta.kama(c, n),
}
MA_LENGTHS = (9, 20, 50, 100, 200)


def _dist(c: pd.Series, level) -> pd.Series:
    """終値から水準への相対距離（正 = 終値が上）。"""
    return (c - level) / c


def _trend_features(df: pd.DataFrame, f: dict) -> None:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    logc = np.log(c)
    mas: dict[str, pd.Series] = {}
    for kind, fn in MA_TYPES.items():
        for n in MA_LENGTHS:
            m = fn(c, v, n)
            mas[f"{kind}{n}"] = m
            f[f"t_{kind}{n}"] = logc - np.log(m)
    for n in (5, 10, 30):
        for kind in ("sma", "ema"):
            m = MA_TYPES[kind](c, v, n)
            mas[f"{kind}{n}"] = m
            f[f"t_{kind}{n}"] = logc - np.log(m)
    # 移動平均のクロス（速い − 遅い）
    for a, b in (("sma10", "sma20"), ("sma20", "sma50"), ("sma50", "sma200"), ("ema9", "ema20"), ("ema20", "ema50"),
                 ("ema50", "ema200"), ("hma9", "hma50")):
        f[f"t_x_{a}_{b}"] = np.log(mas[a]) - np.log(mas[b])
    f["t_x_ema12_26"] = np.log(ta.ema(c, 12)) - np.log(ta.ema(c, 26))
    # 傾きとリボン
    for name in ("ema20", "ema50", "sma200"):
        f[f"t_slope_{name}"] = np.log(mas[name]) - np.log(mas[name].shift(5))
    ribbon = [ta.ema(c, n) for n in (20, 30, 40, 50, 60)]
    f["t_ribbon_order"] = sum((ribbon[i] > ribbon[i + 1]).astype(float) for i in range(4)) - 2  # 全部順なら +2
    f["t_ribbon_width"] = (ribbon[0] - ribbon[-1]) / c
    # 移動平均チャネル（高値・安値の SMA）
    for n in (20, 50):
        hi_ma, lo_ma = ta.sma(h, n), ta.sma(l, n)
        f[f"t_mac{n}"] = (c - lo_ma) / (hi_ma - lo_ma + EPS)
    # MACD / PPO
    for fast, slow, sig in ((12, 26, 9), (5, 35, 5), (8, 17, 9), (19, 39, 9)):
        macd = ta.ema(c, fast) - ta.ema(c, slow)
        f[f"t_macd{fast}_{slow}"] = macd / c
        f[f"t_macdh{fast}_{slow}_{sig}"] = (macd - ta.ema(macd, sig)) / c
    for fast, slow in ((12, 26), (5, 35)):
        ppo = 100 * (ta.ema(c, fast) - ta.ema(c, slow)) / (ta.ema(c, slow) + EPS)
        f[f"o_ppo{fast}_{slow}"] = ppo
        f[f"o_ppoh{fast}_{slow}"] = ppo - ta.ema(ppo, 9)
    # DMI / ADX
    for n in (7, 14, 28):
        a, pdi, mdi, dx = ta.adx(df, n)
        f[f"t_adx{n}"], f[f"t_di_diff{n}"], f[f"t_dx{n}"] = a, pdi - mdi, dx
    # Aroon
    for n in (14, 25, 50):
        up, dn = ta.aroon(df, n)
        f[f"t_aroon_up{n}"], f[f"t_aroon_osc{n}"] = up, up - dn
    # パラボリック SAR
    for step, cap in ((0.02, 0.2), (0.01, 0.1), (0.03, 0.3)):
        f[f"t_sar{int(step * 100)}"] = _dist(c, ta.parabolic_sar(df, step, cap))
    # 一目均衡表（標準 9/26/52 と、暗号資産でよく使う倍の 20/60/120）
    for t_n, k_n, s_n in ((9, 26, 52), (20, 60, 120)):
        tenkan = (h.rolling(t_n, min_periods=t_n).max() + l.rolling(t_n, min_periods=t_n).min()) / 2
        kijun = (h.rolling(k_n, min_periods=k_n).max() + l.rolling(k_n, min_periods=k_n).min()) / 2
        span_a = ((tenkan + kijun) / 2).shift(k_n)
        span_b = ((h.rolling(s_n, min_periods=s_n).max() + l.rolling(s_n, min_periods=s_n).min()) / 2).shift(k_n)
        tag = f"{t_n}_{k_n}"
        f[f"t_ichi_tk_{tag}"] = (tenkan - kijun) / c
        f[f"t_ichi_kijun_{tag}"] = _dist(c, kijun)
        f[f"t_ichi_cloud_{tag}"] = _dist(c, (span_a + span_b) / 2)
        f[f"t_ichi_cloud_w_{tag}"] = (span_a - span_b) / c
        f[f"t_ichi_chikou_{tag}"] = logc - logc.shift(k_n)
    # スーパートレンド
    for n, m in ((10, 3.0), (7, 2.0), (14, 4.0)):
        st, d = ta.supertrend(df, n, m)
        f[f"t_st_dir{n}_{int(m)}"], f[f"t_st_dist{n}_{int(m)}"] = d, _dist(c, st)
    # 回帰
    for n in (10, 30, 100, 200):
        f[f"t_lr_slope{n}"] = ta.linreg_slope(logc, n)
    for n in (25, 100):
        f[f"t_lr_dist{n}"] = logc - ta.lsma(logc, n)
    for n in (30, 100):
        f[f"t_lr_r2{n}"] = ta.linreg_r2(logc, n)
    resid = logc - ta.lsma(logc, 100)
    f["t_lr_chan100"] = resid / (resid.rolling(100, min_periods=100).std() + EPS)
    # その他のトレンド系
    for n in (9, 15, 30):
        f[f"t_trix{n}"] = ta.trix(c, n)
    k, ks = ta.kst(c)
    f["t_kst"], f["t_kst_h"] = k, k - ks
    f["t_coppock"] = ta.coppock(c)
    for n in (20, 50):
        f[f"t_dpo{n}"] = ta.dpo(c, n) / c
    for n in (14, 28):
        vp, vm = ta.vortex(df, n)
        f[f"t_vi_diff{n}"], f[f"t_vi_plus{n}"] = vp - vm, vp
    f["t_mass25"] = ta.mass_index(df, 25)
    for n in (14, 28):
        f[f"t_chop{n}"] = ta.choppiness(df, n)
    f["t_chop_zone"] = ta.chop_zone(df)
    for n in (20, 55, 100):
        hi_n, lo_n = h.rolling(n, min_periods=n).max(), l.rolling(n, min_periods=n).min()
        f[f"t_dc{n}"] = (c - lo_n) / (hi_n - lo_n + EPS)
        f[f"t_dc_w{n}"] = (hi_n - lo_n) / c
    lo_stop, hi_stop = ta.chande_kroll_stop(df)
    f["t_ck_long"], f["t_ck_short"] = _dist(c, lo_stop), _dist(c, hi_stop)
    jaw, teeth, lips = ta.alligator(df)
    f["t_alli_jaw"], f["t_alli_teeth"], f["t_alli_lips"] = _dist(c, jaw), _dist(c, teeth), _dist(c, lips)
    f["t_alli_lt"], f["t_alli_tj"] = (lips - teeth) / c, (teeth - jaw) / c
    vs, vd = ta.volatility_stop(df, 20, 2.0)
    f["t_vstop_dir"], f["t_vstop_dist"] = vd, _dist(c, vs)
    f["t_median20"] = _dist(c, c.rolling(20, min_periods=20).median())
    f["t_median50"] = _dist(c, c.rolling(50, min_periods=50).median())


def _oscillator_features(df: pd.DataFrame, f: dict) -> None:
    c, h, l = df["close"], df["high"], df["low"]
    logc = np.log(c)
    for n in (7, 14, 21, 28):
        f[f"o_rsi{n}"] = ta.rsi(c, n)
    f["o_rsi14_slope"] = f["o_rsi14"].diff(3)
    for n, d, sk in ((14, 3, 1), (14, 3, 3), (5, 3, 3), (21, 5, 5), (9, 3, 1)):
        k, dd = ta.stochastic(df, n, d, sk)
        tag = f"{n}_{d}_{sk}"
        f[f"o_stoch_k_{tag}"], f[f"o_stoch_kd_{tag}"] = k, k - dd
    f["o_stoch_k"], f["o_stoch_d"] = f["o_stoch_k_14_3_1"], f["o_stoch_k_14_3_1"] - f["o_stoch_kd_14_3_1"]
    for n in (14, 21):
        k, dd = ta.stoch_rsi(c, n, n, 3, 3)
        f[f"o_stoch_rsi_k{n}"], f[f"o_stoch_rsi_kd{n}"] = k, k - dd
    f["o_stoch_rsi"] = f["o_stoch_rsi_k14"]
    for n in (14, 28):
        f[f"o_willr{n}"] = ta.williams_r(df, n)
    for n in (14, 20, 50):
        f[f"o_cci{n}"] = ta.cci(df, n)
    f["o_woodies_cci6"] = ta.cci(df, 6)
    f["o_ultimate"] = ta.ultimate(df)
    f["o_ultimate_slow"] = ta.ultimate(df, 14, 28, 56)
    med = (h + l) / 2
    ao = (ta.sma(med, 5) - ta.sma(med, 34)) / c
    f["o_ao"], f["o_ao_slope"] = ao, ao.diff()
    for n in (9, 14):
        f[f"o_cmo{n}"] = ta.cmo(c, n)
    for n in (1, 3, 6, 12, 24, 72, 168):
        f[f"o_roc{n}"] = logc - logc.shift(n)
    f["o_mom10_slope"] = (c - c.shift(10)).diff() / c
    f["o_crsi"] = ta.connors_rsi(c)
    fi, fs = ta.fisher(df, 9)
    f["o_fisher9"], f["o_fisher9_h"] = fi, fi - fs
    r, rs = ta.rvi(df, 10)
    f["o_rvi10"], f["o_rvi10_h"] = r, r - rs
    t, ts = ta.tsi(c)
    f["o_tsi"], f["o_tsi_h"] = t, t - ts
    s, ss = ta.smi_ergodic(c)
    f["o_smi"], f["o_smi_h"] = s, s - ss
    for n in (14, 28):
        f[f"o_mfi{n}"] = ta.mfi(df, n)
    f["o_bop"] = ta.balance_of_power(df)
    f["o_bop14"] = ta.sma(f["o_bop"], 14)
    bull, bear = ta.bull_bear_power(df, 13)
    f["o_bull13"], f["o_bear13"], f["o_bbp13"] = bull / c, bear / c, (bull + bear) / c
    f["o_dpo20"] = ta.dpo(c, 20) / c


def _volatility_volume_features(df: pd.DataFrame, f: dict) -> None:
    c, h, l, v, o = df["close"], df["high"], df["low"], df["volume"], df["open"]
    logc = np.log(c)
    r1 = logc.diff()
    tr = ta.true_range(df)
    for n in (7, 14, 28):
        f[f"v_atr{n}"] = ta.atr(df, n) / c
    f["v_atr_ratio"] = ta.atr(df, 14) / (ta.atr(df, 100) + EPS)
    for n in (10, 24, 72, 168):
        f[f"v_rv{n}"] = r1.rolling(n, min_periods=n).std()
    f["v_rv_ratio24_168"] = f["v_rv24"] / (f["v_rv168"] + EPS)
    f["v_rv_ratio10_72"] = f["v_rv10"] / (f["v_rv72"] + EPS)
    hl = np.log(h / l)
    f["v_parkinson24"] = np.sqrt((hl**2).rolling(24, min_periods=24).mean() / (4 * np.log(2)))
    f["v_gk24"] = np.sqrt((0.5 * hl**2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2).rolling(24, min_periods=24).mean().clip(lower=0))
    f["v_rvi10"] = ta.relative_volatility_index(c, 10)
    f["v_ulcer14"] = ta.ulcer_index(c, 14)
    f["v_chaikin_vol10"] = ta.ema(h - l, 10) / (ta.ema(h - l, 10).shift(10) + EPS) - 1
    for n, k in ((20, 2.0), (20, 3.0), (50, 2.0), (10, 1.5)):
        m, s = ta.sma(c, n), c.rolling(n, min_periods=n).std()
        tag = f"{n}_{int(k * 10)}"
        f[f"v_bb_pos{tag}"] = (c - m) / (k * s + EPS)
        f[f"v_bb_w{tag}"] = 2 * k * s / (m + EPS)
    f["v_bb_pos"], f["v_bb_width"] = f["v_bb_pos20_20"], f["v_bb_w20_20"]
    for n, k in ((20, 1.0), (20, 2.0), (50, 2.0)):
        f[f"v_kc_pos{n}_{int(k)}"] = (c - ta.ema(c, n)) / (k * ta.atr(df, n) + EPS)
    f["v_kc_w20"] = 4 * ta.atr(df, 20) / c
    f["v_squeeze"] = (f["v_bb_w20_20"] < f["v_kc_w20"]).astype(float)   # ボリンジャーがケルトナーの内側
    f["v_std20"] = c.rolling(20, min_periods=20).std() / c
    f["v_range"] = (h - l) / c
    f["v_range_ratio"] = (h - l) / (ta.sma(h - l, 24) + EPS)
    f["v_tr_ratio"] = tr / (ta.atr(df, 14) + EPS)
    f["v_gap"] = (o - c.shift(1)) / c.shift(1)
    f["v_body"] = (c - o) / (h - l + EPS)
    f["v_upper_wick"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / (h - l + EPS)
    f["v_lower_wick"] = (pd.concat([c, o], axis=1).min(axis=1) - l) / (h - l + EPS)
    for n in (24, 168):
        f[f"v_max_dd{n}"] = c / c.rolling(n, min_periods=n).max() - 1
        f[f"v_run_up{n}"] = c / c.rolling(n, min_periods=n).min() - 1
        f[f"v_age_high{n}"] = ta.rolling_age_of_max(h.to_numpy(dtype=float), n) / n
        f[f"v_age_low{n}"] = ta.rolling_age_of_min(l.to_numpy(dtype=float), n) / n
    f["v_max_dd72"] = c / c.rolling(72, min_periods=72).max() - 1
    f["v_run_up72"] = c / c.rolling(72, min_periods=72).min() - 1
    # 出来高
    for n in (24, 168):
        f[f"v_vol_ratio{n}"] = v / (ta.sma(v, n) + EPS)
    f["v_vol_z168"] = (v - ta.sma(v, 168)) / (v.rolling(168, min_periods=168).std() + EPS)
    f["v_vol_osc5_10"] = 100 * (ta.sma(v, 5) - ta.sma(v, 10)) / (ta.sma(v, 10) + EPS)
    up_v = v.where(c > o, 0.0)
    dn_v = v.where(c < o, 0.0)
    f["v_updown_vol24"] = (up_v.rolling(24, min_periods=24).sum() - dn_v.rolling(24, min_periods=24).sum()) / (v.rolling(24, min_periods=24).sum() + EPS)
    f["v_net_vol24"] = (np.sign(c - o) * v).rolling(24, min_periods=24).sum() / (v.rolling(24, min_periods=24).sum() + EPS)
    ob = ta.obv(df)
    for n in (24, 168):
        f[f"v_obv_slope{n}"] = ta.linreg_slope(ob, n) / (ta.sma(v, n) + EPS)
    f["v_ad_slope24"] = ta.linreg_slope(ta.accumulation_distribution(df), 24) / (ta.sma(v, 24) + EPS)
    f["v_pvt_slope24"] = ta.linreg_slope(ta.pvt(df), 24) / (ta.sma(v, 24) + EPS)
    for n in (20, 50):
        f[f"v_cmf{n}"] = ta.chaikin_money_flow(df, n)
    f["v_chaikin_osc"] = ta.chaikin_oscillator(df) / (ta.sma(v, 10) + EPS)
    for n in (2, 13):
        f[f"v_efi{n}"] = ta.elder_force(df, n) / (ta.sma(v, 13) * c + EPS)
    k, ks = ta.klinger(df)
    f["v_klinger"], f["v_klinger_h"] = k / (ta.sma(v, 55) + EPS), (k - ks) / (ta.sma(v, 55) + EPS)
    f["v_eom14"] = ta.ease_of_movement(df, 14) / c
    f["v_vwma_diff20"] = np.log(ta.vwma(c, v, 20)) - np.log(ta.sma(c, 20))
    f["v_pv_corr24"] = r1.rolling(24, min_periods=24).corr(np.log(v + 1).diff())


def _level_pattern_features(df: pd.DataFrame, f: dict) -> None:
    c, h, l, v, o = df["close"], df["high"], df["low"], df["volume"], df["open"]
    a14 = ta.atr(df, 14)
    # ピボットポイント（前日 / 前週）
    for freq, tag in (("D", "d"), ("W", "w")):
        p = ta.period_pivots(df, freq)
        cols = ["pp", "r1", "s1", "r2", "s2", "fib_r1", "fib_s1", "woodie_pp", "cam_r3", "cam_s3", "dm_pp"] if freq == "D" else ["pp", "r1", "s1"]
        for k in cols:
            f[f"p_piv{tag}_{k}"] = (c - p[k]) / (a14 + EPS)   # ATR 単位の距離
        f[f"p_piv{tag}_pos"] = (c - p["s1"]) / (p["r1"] - p["s1"] + EPS)
    # フラクタル（2 本）とピボット高安（5 本）: 確認できたものだけ
    hv, lv = h.to_numpy(dtype=float), l.to_numpy(dtype=float)
    for left, right, tag in ((2, 2, "fr"), (5, 5, "ph")):
        ph, pl = ta.confirmed_extremes(hv, lv, left, right)
        val_h, age_h = ta.last_value_and_age(ph)
        val_l, age_l = ta.last_value_and_age(pl)
        f[f"p_{tag}_high_dist"] = (c.to_numpy() - val_h) / c.to_numpy()
        f[f"p_{tag}_low_dist"] = (c.to_numpy() - val_l) / c.to_numpy()
        f[f"p_{tag}_high_age"] = np.log1p(age_h)
        f[f"p_{tag}_low_age"] = np.log1p(age_l)
    # ジグザグ（3% と 8% の反転で確定）
    for dev in (0.03, 0.08):
        piv, leg = ta.zigzag_confirmed(df, dev)
        f[f"p_zz{int(dev * 100)}_dist"] = (c.to_numpy() - piv) / c.to_numpy()
        f[f"p_zz{int(dev * 100)}_leg"] = leg
    # 自動フィボナッチ（直近 n 本の高値・安値の間での押し戻しの位置）
    for n in (100, 500):
        hi_n, lo_n = h.rolling(n, min_periods=n).max(), l.rolling(n, min_periods=n).min()
        age_h = ta.rolling_age_of_max(hv, n)
        age_l = ta.rolling_age_of_min(lv, n)
        up_swing = age_h < age_l   # 高値のほうが新しい = 上昇の波
        pos = np.where(up_swing, (hi_n - c) / (hi_n - lo_n + EPS), (c - lo_n) / (hi_n - lo_n + EPS))
        f[f"p_fib{n}_retrace"] = pos
        f[f"p_fib{n}_up"] = up_swing.astype(float)
    # VWAP（日・週・月、バンド、直近 168 本の安値 / 高値を起点）
    day = df.index.floor("D")
    naive = df.index.tz_convert(None) if df.index.tz is not None else df.index
    week = naive.to_period("W").start_time
    month = naive.to_period("M").start_time
    f["p_vwap_d"] = _dist(c, ta.period_vwap(df, day))
    f["p_vwap_w"] = _dist(c, ta.period_vwap(df, week))
    f["p_vwap_m"] = _dist(c, ta.period_vwap(df, month))
    f["p_vwap_d_band"] = ta.vwap_band_pos(df, day)
    f["p_avwap_low168"] = _dist(c, ta.anchored_vwap_from_extreme(df, 168, True))
    f["p_avwap_high168"] = _dist(c, ta.anchored_vwap_from_extreme(df, 168, False))
    # 価格帯別出来高（直近 24 本 / 168 本）
    for n in (24, 168):
        vp = ta.volume_profile_window(df, n)
        f[f"p_vp{n}_poc"] = _dist(c, vp["poc"])
        f[f"p_vp{n}_va_pos"] = (c - vp["val"]) / (vp["vah"] - vp["val"] + EPS)
        f[f"p_vp{n}_va_w"] = (vp["vah"] - vp["val"]) / c
        f[f"p_vp{n}_at"] = vp["at"]
    # ローソク足のパターン（1 = 出現）
    body = (c - o).abs()
    rng = h - l + EPS
    up_bar, dn_bar = (c > o), (c < o)
    upper = h - pd.concat([c, o], axis=1).max(axis=1)
    lower = pd.concat([c, o], axis=1).min(axis=1) - l
    pat = {
        "doji": body <= 0.1 * rng,
        "hammer": (lower >= 2 * body) & (upper <= 0.3 * body + 0.05 * rng) & (body > 0.05 * rng),
        "shooting_star": (upper >= 2 * body) & (lower <= 0.3 * body + 0.05 * rng) & (body > 0.05 * rng),
        "bull_engulf": dn_bar.shift(1, fill_value=False) & up_bar & (o <= c.shift(1)) & (c >= o.shift(1)),
        "bear_engulf": up_bar.shift(1, fill_value=False) & dn_bar & (o >= c.shift(1)) & (c <= o.shift(1)),
        "bull_harami": dn_bar.shift(1, fill_value=False) & up_bar & (o >= c.shift(1)) & (c <= o.shift(1)),
        "bear_harami": up_bar.shift(1, fill_value=False) & dn_bar & (o <= c.shift(1)) & (c >= o.shift(1)),
        "piercing": dn_bar.shift(1, fill_value=False) & up_bar & (o < c.shift(1)) & (c > (o.shift(1) + c.shift(1)) / 2) & (c < o.shift(1)),
        "dark_cloud": up_bar.shift(1, fill_value=False) & dn_bar & (o > c.shift(1)) & (c < (o.shift(1) + c.shift(1)) / 2) & (c > o.shift(1)),
        "three_soldiers": up_bar & up_bar.shift(1, fill_value=False) & up_bar.shift(2, fill_value=False) & (c > c.shift(1)) & (c.shift(1) > c.shift(2)),
        "three_crows": dn_bar & dn_bar.shift(1, fill_value=False) & dn_bar.shift(2, fill_value=False) & (c < c.shift(1)) & (c.shift(1) < c.shift(2)),
        "inside": (h < h.shift(1)) & (l > l.shift(1)),
        "outside": (h > h.shift(1)) & (l < l.shift(1)),
        "bull_marubozu": up_bar & (body >= 0.9 * rng),
        "bear_marubozu": dn_bar & (body >= 0.9 * rng),
        "morning_star": dn_bar.shift(2, fill_value=False) & (body.shift(1) <= 0.3 * body.shift(2)) & up_bar & (c > (o.shift(2) + c.shift(2)) / 2),
        "evening_star": up_bar.shift(2, fill_value=False) & (body.shift(1) <= 0.3 * body.shift(2)) & dn_bar & (c < (o.shift(2) + c.shift(2)) / 2),
    }
    for k, s in pat.items():
        f[f"p_cdl_{k}"] = s.astype(float)
    sign = np.sign(c.diff()).fillna(0.0).to_numpy()
    run = np.zeros(len(sign))
    for i in range(1, len(sign)):
        run[i] = run[i - 1] + sign[i] if sign[i] == np.sign(run[i - 1]) or run[i - 1] == 0 else sign[i]
    f["p_run"] = run


def _rating_features(df: pd.DataFrame, f: dict) -> None:
    """TradingView のテクニカル評価に準じた集計（近似。各指標の票 +1 / 0 / −1 の平均）。"""
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    votes_ma = []
    for n in (10, 20, 30, 50, 100, 200):
        for m in (ta.ema(c, n), ta.sma(c, n)):
            votes_ma.append(np.sign(c - m))
    kijun = (h.rolling(26, min_periods=26).max() + l.rolling(26, min_periods=26).min()) / 2
    votes_ma += [np.sign(c - kijun), np.sign(c - ta.vwma(c, v, 20)), np.sign(c - ta.hma(c, 9))]
    ma_rating = pd.concat(votes_ma, axis=1).mean(axis=1)

    def vote(buy, sell):
        return pd.Series(np.where(buy, 1.0, np.where(sell, -1.0, 0.0)), index=c.index)

    r = ta.rsi(c, 14)
    k, d = ta.stochastic(df, 14, 3, 3)
    cc = ta.cci(df, 20)
    a, pdi, mdi, _ = ta.adx(df, 14)
    ao = ta.sma((h + l) / 2, 5) - ta.sma((h + l) / 2, 34)
    mom = c - c.shift(10)
    macd = ta.ema(c, 12) - ta.ema(c, 26)
    sk, sd = ta.stoch_rsi(c, 14, 14, 3, 3)
    wr = ta.williams_r(df, 14)
    bull, bear = ta.bull_bear_power(df, 13)
    uo = ta.ultimate(df)
    uptrend = c > ta.ema(c, 50)
    votes_o = [
        vote((r < 30) & (r > r.shift(1)), (r > 70) & (r < r.shift(1))),
        vote((k < 20) & (d < 20) & (k > d), (k > 80) & (d > 80) & (k < d)),
        vote((cc < -100) & (cc > cc.shift(1)), (cc > 100) & (cc < cc.shift(1))),
        vote((a > 20) & (pdi > mdi), (a > 20) & (pdi < mdi)),
        vote(((ao > 0) & (ao > ao.shift(1)) & (ao.shift(1) < ao.shift(2))) | ((ao > 0) & (ao.shift(1) <= 0)),
             ((ao < 0) & (ao < ao.shift(1)) & (ao.shift(1) > ao.shift(2))) | ((ao < 0) & (ao.shift(1) >= 0))),
        vote(mom > mom.shift(1), mom < mom.shift(1)),
        vote(macd > ta.ema(macd, 9), macd < ta.ema(macd, 9)),
        vote(~uptrend & (sk < 20) & (sd < 20) & (sk > sd), uptrend & (sk > 80) & (sd > 80) & (sk < sd)),
        vote((wr < -80) & (wr > wr.shift(1)), (wr > -20) & (wr < wr.shift(1))),
        vote(uptrend & (bear < 0) & (bear > bear.shift(1)), ~uptrend & (bull > 0) & (bull < bull.shift(1))),
        vote(uo > 70, uo < 30),
    ]
    osc_rating = pd.concat(votes_o, axis=1).mean(axis=1)
    f["r_ma"], f["r_osc"], f["r_total"] = ma_rating, osc_rating, (ma_rating + osc_rating) / 2
    f["r_n_overbought"] = ((r > 70).astype(float) + (k > 80).astype(float) + (cc > 100).astype(float)
                           + (wr > -20).astype(float) + (ta.mfi(df, 14) > 80).astype(float))
    f["r_n_oversold"] = ((r < 30).astype(float) + (k < 20).astype(float) + (cc < -100).astype(float)
                         + (wr < -80).astype(float) + (ta.mfi(df, 14) < 20).astype(float))


def _time_features(df: pd.DataFrame, f: dict) -> None:
    hour = df.index.hour.astype(float)
    f["c_hour"] = hour
    f["c_weekday"] = df.index.dayofweek.astype(float)
    f["c_hour_sin"], f["c_hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["c_dom"] = df.index.day.astype(float)
    f["c_weekend"] = (df.index.dayofweek >= 5).astype(float)
    f["c_jp_session"] = ((hour >= 0) & (hour < 6)).astype(float)      # 09〜15 JST
    f["c_us_session"] = ((hour >= 13) & (hour < 21)).astype(float)    # 米国株の時間帯


def indicator_table(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV（列 open, high, low, close, volume、index = 足の開始時刻 UTC、等間隔）→ 指標の表（同じ index）。"""
    f: dict = {}   # 列を dict に集めて最後に 1 回で表にする（列ごとの挿入は遅い）
    _trend_features(df, f)
    _oscillator_features(df, f)
    _volatility_volume_features(df, f)
    _level_pattern_features(df, f)
    _rating_features(df, f)
    _time_features(df, f)
    out = pd.DataFrame({k: (v.to_numpy() if isinstance(v, pd.Series) else np.asarray(v, dtype=float)) for k, v in f.items()},
                       index=df.index)
    return out.replace([np.inf, -np.inf], np.nan)


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """足 t の終値で買い、horizon 本後の足の終値で売ったときの対数収益率（行 t に置く。末尾 horizon 行は NaN）。"""
    return np.log(close.shift(-horizon)) - np.log(close)
