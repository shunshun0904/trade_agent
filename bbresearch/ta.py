"""テクニカル指標の部品（numpy / pandas だけ）。2026-09-27 オーナー指示で TradingView の組み込み指標に合わせて拡充。

すべて「行 t の値は足 t 以前だけから決まる」ように書く。先に描画される（未来にずらす）指標は、その分だけ過去にずらして
「時刻 t に画面に見えている値」を使う（一目均衡表の雲、アリゲーター）。右側の確認が要る指標（フラクタル、ピボット高安、
ジグザグ）は確認できた時点の値だけを使う。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

EPS = 1e-12


# ------------------------------------------------------------------ 移動平均
def _conv(s: pd.Series, w: np.ndarray) -> pd.Series:
    """重み w（w[-1] が最新の足に掛かる）の畳み込み。窓に NaN があれば NaN。"""
    x = s.to_numpy(dtype=float)
    n = len(w)
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = np.convolve(x, w[::-1], mode="valid")
    return pd.Series(out, index=s.index)


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder の平滑化（SMMA / RMA）。alpha = 1/n。"""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return _conv(s, w / w.sum())


def dema(s: pd.Series, n: int) -> pd.Series:
    e = ema(s, n)
    return 2 * e - ema(e, n)


def tema(s: pd.Series, n: int) -> pd.Series:
    e1 = ema(s, n)
    e2 = ema(e1, n)
    e3 = ema(e2, n)
    return 3 * e1 - 3 * e2 + e3


def hma(s: pd.Series, n: int) -> pd.Series:
    return wma(2 * wma(s, max(1, n // 2)) - wma(s, n), max(1, int(round(np.sqrt(n)))))


def _lin_weights(n: int) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    sxx = float((xc**2).sum())
    slope_w = xc / sxx                                  # Σ w_k y_k = 傾き
    end_w = 1.0 / n + xc * (n - 1 - x.mean()) / sxx      # 最後の点での回帰値
    return slope_w, end_w, sxx


def lsma(s: pd.Series, n: int) -> pd.Series:
    """最小二乗移動平均（回帰直線の最後の点の値）。"""
    _, end_w, _ = _lin_weights(n)
    return _conv(s, end_w)


def linreg_slope(s: pd.Series, n: int) -> pd.Series:
    """直近 n 本の回帰の傾き（1 本あたり）。"""
    slope_w, _, _ = _lin_weights(n)
    return _conv(s, slope_w)


def linreg_r2(s: pd.Series, n: int) -> pd.Series:
    """直近 n 本の回帰の決定係数（トレンドの直線らしさ）。"""
    slope_w, _, sxx = _lin_weights(n)
    slope = _conv(s, slope_w)
    var_y = s.rolling(n, min_periods=n).var(ddof=0)
    return (slope**2 * sxx / n) / (var_y + EPS)


def alma(s: pd.Series, n: int, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    m = offset * (n - 1)
    sd = n / sigma
    w = np.exp(-((np.arange(n) - m) ** 2) / (2 * sd * sd))
    return _conv(s, w / w.sum())


def mcginley(s: pd.Series, n: int) -> pd.Series:
    x = s.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    md = np.nan
    for i, c in enumerate(x):
        if np.isnan(c):
            continue
        if np.isnan(md):
            md = c
        else:
            md = md + (c - md) / (n * (c / md) ** 4)
        out[i] = md
    return pd.Series(out, index=s.index)


def kama(s: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman の適応移動平均。"""
    x = s.to_numpy(dtype=float)
    change = np.abs(x - np.r_[np.full(n, np.nan), x[:-n]])
    vol = pd.Series(np.abs(np.diff(x, prepend=np.nan)), index=s.index).rolling(n, min_periods=n).sum().to_numpy()
    er = change / (vol + EPS)
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    out = np.full(len(x), np.nan)
    k = np.nan
    for i in range(len(x)):
        if np.isnan(sc[i]) or np.isnan(x[i]):
            continue
        k = x[i] if np.isnan(k) else k + sc[i] * (x[i] - k)
        out[i] = k
    return pd.Series(out, index=s.index)


def vwma(c: pd.Series, v: pd.Series, n: int) -> pd.Series:
    return sma(c * v, n) / (sma(v, n) + EPS)


# ------------------------------------------------------------------ レンジ・ボラ
def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return rma(true_range(df), n)


def rolling_age_of_max(x: np.ndarray, n: int) -> np.ndarray:
    """直近 n 本の最大値が何本前か（0 = 今の足）。先頭 n−1 本は NaN。"""
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        win = sliding_window_view(x, n)
        out[n - 1:] = n - 1 - np.argmax(win, axis=1)
    return out


def rolling_age_of_min(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        win = sliding_window_view(x, n)
        out[n - 1:] = n - 1 - np.argmin(win, axis=1)
    return out


# ------------------------------------------------------------------ オシレーター
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up, down = d.clip(lower=0.0), (-d).clip(lower=0.0)
    rs = rma(up, n) / (rma(down, n) + EPS)
    return 100 - 100 / (1 + rs)


def stochastic(df: pd.DataFrame, n: int = 14, d: int = 3, smooth_k: int = 1) -> tuple[pd.Series, pd.Series]:
    lo, hi = df["low"].rolling(n, min_periods=n).min(), df["high"].rolling(n, min_periods=n).max()
    k = 100 * (df["close"] - lo) / (hi - lo + EPS)
    if smooth_k > 1:
        k = sma(k, smooth_k)
    return k, sma(k, d)


def stoch_rsi(close: pd.Series, n_rsi: int = 14, n_stoch: int = 14, k: int = 3, d: int = 3) -> tuple[pd.Series, pd.Series]:
    r = rsi(close, n_rsi)
    lo, hi = r.rolling(n_stoch, min_periods=n_stoch).min(), r.rolling(n_stoch, min_periods=n_stoch).max()
    kk = sma(100 * (r - lo) / (hi - lo + EPS), k)
    return kk, sma(kk, d)


def williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    lo, hi = df["low"].rolling(n, min_periods=n).min(), df["high"].rolling(n, min_periods=n).max()
    return -100 * (hi - df["close"]) / (hi - lo + EPS)


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    m = sma(tp, n)
    mad = (tp - m).abs().rolling(n, min_periods=n).mean()
    return (tp - m) / (0.015 * mad + EPS)


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    flow = tp * df["volume"]
    up = flow.where(tp > tp.shift(1), 0.0)
    down = flow.where(tp < tp.shift(1), 0.0)
    ratio = up.rolling(n, min_periods=n).sum() / (down.rolling(n, min_periods=n).sum() + EPS)
    return 100 - 100 / (1 + ratio)


def adx(df: pd.DataFrame, n: int = 14) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """(ADX, +DI, −DI, DX)。"""
    up, down = df["high"].diff(), -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    a = atr(df, n)
    plus_di = 100 * rma(plus_dm, n) / (a + EPS)
    minus_di = 100 * rma(minus_dm, n) / (a + EPS)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    return rma(dx, n), plus_di, minus_di, dx


def aroon(df: pd.DataFrame, n: int = 25) -> tuple[pd.Series, pd.Series]:
    hi_age = rolling_age_of_max(df["high"].to_numpy(dtype=float), n + 1)
    lo_age = rolling_age_of_min(df["low"].to_numpy(dtype=float), n + 1)
    return (pd.Series(100 * (n - hi_age) / n, index=df.index), pd.Series(100 * (n - lo_age) / n, index=df.index))


def ultimate(df: pd.DataFrame, n1: int = 7, n2: int = 14, n3: int = 28) -> pd.Series:
    c, l = df["close"], df["low"]
    bp = c - pd.concat([l, c.shift(1)], axis=1).min(axis=1)
    tr = true_range(df)
    avg = [bp.rolling(n, min_periods=n).sum() / (tr.rolling(n, min_periods=n).sum() + EPS) for n in (n1, n2, n3)]
    return 100 * (4 * avg[0] + 2 * avg[1] + avg[2]) / 7


def cmo(close: pd.Series, n: int = 9) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(n, min_periods=n).sum()
    down = (-d).clip(lower=0).rolling(n, min_periods=n).sum()
    return 100 * (up - down) / (up + down + EPS)


def fisher(df: pd.DataFrame, n: int = 9) -> tuple[pd.Series, pd.Series]:
    hl2 = (df["high"] + df["low"]) / 2
    lo, hi = hl2.rolling(n, min_periods=n).min(), hl2.rolling(n, min_periods=n).max()
    raw = (2 * (hl2 - lo) / (hi - lo + EPS) - 1).to_numpy()
    out = np.full(len(raw), np.nan)
    v = 0.0
    f = 0.0
    for i, r in enumerate(raw):
        if np.isnan(r):
            continue
        v = 0.66 * r + 0.67 * v
        v = min(0.999, max(-0.999, v))
        f = 0.5 * np.log((1 + v) / (1 - v)) + 0.5 * f
        out[i] = f
    fs = pd.Series(out, index=df.index)
    return fs, fs.shift(1)


def rvi(df: pd.DataFrame, n: int = 10) -> tuple[pd.Series, pd.Series]:
    """Relative Vigor Index（(C−O) の対称加重を (H−L) の対称加重で割る）。"""
    def sw(s):
        return (s + 2 * s.shift(1) + 2 * s.shift(2) + s.shift(3)) / 6
    num = sw(df["close"] - df["open"]).rolling(n, min_periods=n).sum()
    den = sw(df["high"] - df["low"]).rolling(n, min_periods=n).sum()
    r = num / (den + EPS)
    return r, sw(r)


def tsi(close: pd.Series, long: int = 25, short: int = 13, signal: int = 13) -> tuple[pd.Series, pd.Series]:
    m = close.diff()
    num = ema(ema(m, long), short)
    den = ema(ema(m.abs(), long), short)
    t = 100 * num / (den + EPS)
    return t, ema(t, signal)


def smi_ergodic(close: pd.Series, long: int = 20, short: int = 5, signal: int = 5) -> tuple[pd.Series, pd.Series]:
    """SMI Ergodic = TSI(short, long) とそのシグナル（TradingView の既定 5, 20, 5）。"""
    m = close.diff()
    num = ema(ema(m, long), short)
    den = ema(ema(m.abs(), long), short)
    t = num / (den + EPS)
    return t, ema(t, signal)


def connors_rsi(close: pd.Series, n_rsi: int = 3, n_streak: int = 2, n_rank: int = 100) -> pd.Series:
    d = np.sign(close.diff()).fillna(0.0).to_numpy()
    streak = np.zeros(len(d))
    for i in range(1, len(d)):
        if d[i] > 0:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1
        elif d[i] < 0:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1
    r1 = rsi(close, n_rsi)
    r2 = rsi(pd.Series(streak, index=close.index), n_streak)
    roc = close.pct_change()
    r3 = roc.rolling(n_rank, min_periods=n_rank).rank(pct=True) * 100
    return (r1 + r2 + r3) / 3


def coppock(close: pd.Series, wma_n: int = 10, long: int = 14, short: int = 11) -> pd.Series:
    roc = 100 * (close / close.shift(long) - 1) + 100 * (close / close.shift(short) - 1)
    return wma(roc, wma_n)


def kst(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    def r(n):
        return 100 * (close / close.shift(n) - 1)
    k = sma(r(10), 10) + 2 * sma(r(15), 10) + 3 * sma(r(20), 10) + 4 * sma(r(30), 15)
    return k, sma(k, 9)


def dpo(close: pd.Series, n: int = 20) -> pd.Series:
    """Detrended Price Oscillator（先読みしない形: 過去にずらした終値 − 移動平均）。"""
    return close.shift(n // 2 + 1) - sma(close, n)


def trix(close: pd.Series, n: int = 15) -> pd.Series:
    return 1e4 * ema(ema(ema(np.log(close), n), n), n).diff()


def vortex(df: pd.DataFrame, n: int = 14) -> tuple[pd.Series, pd.Series]:
    vm_p = (df["high"] - df["low"].shift(1)).abs()
    vm_m = (df["low"] - df["high"].shift(1)).abs()
    tr = true_range(df).rolling(n, min_periods=n).sum()
    return vm_p.rolling(n, min_periods=n).sum() / (tr + EPS), vm_m.rolling(n, min_periods=n).sum() / (tr + EPS)


def mass_index(df: pd.DataFrame, n: int = 25) -> pd.Series:
    r = df["high"] - df["low"]
    e1 = ema(r, 9)
    e2 = ema(e1, 9)
    return (e1 / (e2 + EPS)).rolling(n, min_periods=n).sum()


def choppiness(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr_sum = true_range(df).rolling(n, min_periods=n).sum()
    rng = df["high"].rolling(n, min_periods=n).max() - df["low"].rolling(n, min_periods=n).min()
    return 100 * np.log10(tr_sum / (rng + EPS)) / np.log10(n)


def chop_zone(df: pd.DataFrame, n_ema: int = 34, periods: int = 30) -> pd.Series:
    """TradingView の Chop Zone の角度（度。EMA が上向きなら正）。"""
    avg = (df["high"] + df["low"] + df["close"]) / 3
    hh = df["high"].rolling(periods, min_periods=periods).max()
    ll = df["low"].rolling(periods, min_periods=periods).min()
    span = 25 / (hh - ll + EPS) * ll
    e = ema(df["close"], n_ema)
    y2 = (e.shift(1) - e) / avg * span
    c = np.sqrt(1 + y2**2)
    angle = np.degrees(np.arccos(np.clip(1 / c, -1, 1)))
    return angle.where(y2 <= 0, -angle) * -1  # y2 < 0（EMA 上昇）で正


def balance_of_power(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]) / (df["high"] - df["low"] + EPS)


def bull_bear_power(df: pd.DataFrame, n: int = 13) -> tuple[pd.Series, pd.Series]:
    e = ema(df["close"], n)
    return df["high"] - e, df["low"] - e


def elder_force(df: pd.DataFrame, n: int = 13) -> pd.Series:
    return ema(df["close"].diff() * df["volume"], n)


def accumulation_distribution(df: pd.DataFrame) -> pd.Series:
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + EPS)
    return (clv * df["volume"]).cumsum()


def chaikin_oscillator(df: pd.DataFrame, fast: int = 3, slow: int = 10) -> pd.Series:
    ad = accumulation_distribution(df)
    return ema(ad, fast) - ema(ad, slow)


def chaikin_money_flow(df: pd.DataFrame, n: int = 20) -> pd.Series:
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + EPS)
    return (clv * df["volume"]).rolling(n, min_periods=n).sum() / (df["volume"].rolling(n, min_periods=n).sum() + EPS)


def klinger(df: pd.DataFrame, fast: int = 34, slow: int = 55, signal: int = 13) -> tuple[pd.Series, pd.Series]:
    hlc = df["high"] + df["low"] + df["close"]
    sv = df["volume"] * np.sign(hlc.diff()).fillna(0.0)
    k = ema(sv, fast) - ema(sv, slow)
    return k, ema(k, signal)


def ease_of_movement(df: pd.DataFrame, n: int = 14, divisor: float = 1e4) -> pd.Series:
    mid = (df["high"] + df["low"]) / 2
    box = df["volume"] / divisor / (df["high"] - df["low"] + EPS)
    return sma(mid.diff() / (box + EPS), n)


def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["close"].diff()).fillna(0.0) * df["volume"]).cumsum()


def pvt(df: pd.DataFrame) -> pd.Series:
    return (df["close"].pct_change().fillna(0.0) * df["volume"]).cumsum()


def relative_volatility_index(close: pd.Series, n: int = 10, n_std: int = 10) -> pd.Series:
    sd = close.rolling(n_std, min_periods=n_std).std()
    d = close.diff()
    up = rma(sd.where(d > 0, 0.0), n)
    down = rma(sd.where(d < 0, 0.0), n)
    return 100 * up / (up + down + EPS)


def ulcer_index(close: pd.Series, n: int = 14) -> pd.Series:
    dd = 100 * (close / close.rolling(n, min_periods=n).max() - 1)
    return np.sqrt((dd**2).rolling(n, min_periods=n).mean())


# ------------------------------------------------------------------ 逐次計算のトレンド系
def parabolic_sar(df: pd.DataFrame, step: float = 0.02, cap: float = 0.2) -> pd.Series:
    high, low = df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float)
    n = len(df)
    out = np.full(n, np.nan)
    if n < 2:
        return pd.Series(out, index=df.index)
    long, af = True, step
    sar, ep = low[0], high[0]
    for i in range(1, n):
        prev_sar = sar
        sar = prev_sar + af * (ep - prev_sar)
        if long:
            sar = min(sar, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < sar:
                long, sar, ep, af = False, ep, low[i], step
            elif high[i] > ep:
                ep, af = high[i], min(cap, af + step)
        else:
            sar = max(sar, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > sar:
                long, sar, ep, af = True, ep, high[i], step
            elif low[i] < ep:
                ep, af = low[i], min(cap, af + step)
        out[i] = sar
    return pd.Series(out, index=df.index)


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """(スーパートレンドの線, 向き +1/−1)。"""
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    a = atr(df, n).to_numpy()
    close = df["close"].to_numpy()
    upper, lower = hl2 + mult * a, hl2 - mult * a
    n_ = len(close)
    st = np.full(n_, np.nan)
    d = np.full(n_, np.nan)
    fu, fl, direction = np.nan, np.nan, 1
    for i in range(n_):
        if np.isnan(a[i]):
            continue
        if np.isnan(fu):
            fu, fl = upper[i], lower[i]
        else:
            fu = upper[i] if (upper[i] < fu or close[i - 1] > fu) else fu
            fl = lower[i] if (lower[i] > fl or close[i - 1] < fl) else fl
        if direction == -1 and close[i] > fu:
            direction = 1
        elif direction == 1 and close[i] < fl:
            direction = -1
        st[i] = fl if direction == 1 else fu
        d[i] = direction
    return pd.Series(st, index=df.index), pd.Series(d, index=df.index)


def volatility_stop(df: pd.DataFrame, n: int = 20, mult: float = 2.0) -> tuple[pd.Series, pd.Series]:
    """ATR のトレーリングストップ（TradingView の Volatility Stop）。(ストップ, 向き)。"""
    close = df["close"].to_numpy()
    a = (atr(df, n) * mult).to_numpy()
    n_ = len(close)
    stop = np.full(n_, np.nan)
    d = np.full(n_, np.nan)
    up, mx, mn = True, -np.inf, np.inf
    for i in range(n_):
        if np.isnan(a[i]):
            continue
        mx, mn = max(mx, close[i]), min(mn, close[i])
        s = (mx - a[i]) if up else (mn + a[i])
        if up and close[i] < s:
            up, mn = False, close[i]
            s = mn + a[i]
        elif not up and close[i] > s:
            up, mx = True, close[i]
            s = mx - a[i]
        if up:
            mn = np.inf
        else:
            mx = -np.inf
        stop[i], d[i] = s, 1 if up else -1
    return pd.Series(stop, index=df.index), pd.Series(d, index=df.index)


def chande_kroll_stop(df: pd.DataFrame, p: int = 10, x: float = 1.0, q: int = 9) -> tuple[pd.Series, pd.Series]:
    a = atr(df, p)
    first_high = df["high"].rolling(p, min_periods=p).max() - x * a
    first_low = df["low"].rolling(p, min_periods=p).min() + x * a
    return first_low.rolling(q, min_periods=q).min(), first_high.rolling(q, min_periods=q).max()  # (long stop, short stop)


def alligator(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(顎, 歯, 唇)。未来にずらして描く分は過去にずらし、時刻 t に見えている値にする。"""
    hl2 = (df["high"] + df["low"]) / 2
    return rma(hl2, 13).shift(8), rma(hl2, 8).shift(5), rma(hl2, 5).shift(3)


def confirmed_extremes(x_high: np.ndarray, x_low: np.ndarray, left: int, right: int) -> tuple[np.ndarray, np.ndarray]:
    """左右 left / right 本より高い（低い）点。確認できる right 本後の行に値を置く（それまでは NaN）。"""
    n = len(x_high)
    w = left + right + 1
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    if n >= w:
        wh = sliding_window_view(x_high, w)
        wl = sliding_window_view(x_low, w)
        is_h = wh[:, left] >= wh.max(axis=1)
        is_l = wl[:, left] <= wl.min(axis=1)
        idx = np.arange(w - 1, n)                 # 確認できる行（窓の右端）
        ph[idx[is_h]] = wh[is_h, left]
        pl[idx[is_l]] = wl[is_l, left]
    return ph, pl


def last_value_and_age(marks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN でない点を前に引き延ばし、その点から何本たったかも返す。"""
    s = pd.Series(marks)
    val = s.ffill().to_numpy()
    pos = pd.Series(np.where(~np.isnan(marks), np.arange(len(marks)), np.nan)).ffill().to_numpy()
    return val, np.arange(len(marks)) - pos


def zigzag_confirmed(df: pd.DataFrame, dev: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """割合 dev の反転で確定するジグザグ。(確定した最後の転換点の価格, その向き +1=安値からの上昇脚 −1=高値からの下降脚)。

    転換点は、反対方向に dev 以上動いた時点で初めて確定する（それまでは前の転換点を使う）。
    """
    high, low = df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float)
    n = len(high)
    piv = np.full(n, np.nan)
    leg = np.full(n, np.nan)
    if n == 0:
        return piv, leg
    trend = 0            # +1: 上昇脚（安値が確定済み、高値を探す）
    ext_i = 0
    last_piv = np.nan
    for i in range(n):
        if trend == 0:
            if high[i] >= low[0] * (1 + dev):
                trend, ext_i, last_piv = 1, i, low[0]
            elif low[i] <= high[0] * (1 - dev):
                trend, ext_i, last_piv = -1, i, high[0]
        elif trend == 1:
            if high[i] > high[ext_i]:
                ext_i = i
            elif low[i] <= high[ext_i] * (1 - dev):
                last_piv, trend, ext_i = high[ext_i], -1, i
        else:
            if low[i] < low[ext_i]:
                ext_i = i
            elif high[i] >= low[ext_i] * (1 + dev):
                last_piv, trend, ext_i = low[ext_i], 1, i
        piv[i], leg[i] = last_piv, (trend if trend != 0 else np.nan)
    return piv, leg


# ------------------------------------------------------------------ 水準（ピボット、VWAP、価格帯別出来高）
def period_pivots(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """前の期間（freq = "D" / "W"）の OHLC から各種ピボットを作り、足の index に並べる（当期間の始値は当期間の最初の足）。"""
    naive = df.index.tz_convert(None) if df.index.tz is not None else df.index
    key = naive.floor("D") if freq == "D" else naive.to_period("W").start_time
    g = df.groupby(key)
    o, h, l, c = g["open"].first(), g["high"].max(), g["low"].min(), g["close"].last()
    prev = pd.DataFrame({"h": h.shift(1), "l": l.shift(1), "c": c.shift(1), "o_cur": o})
    p = prev.reindex(key)
    p.index = df.index
    H, L, C, O = p["h"], p["l"], p["c"], p["o_cur"]
    r = H - L
    pp = (H + L + C) / 3
    out = pd.DataFrame(index=df.index)
    out["pp"], out["r1"], out["s1"], out["r2"], out["s2"] = pp, 2 * pp - L, 2 * pp - H, pp + r, pp - r
    out["fib_r1"], out["fib_s1"], out["fib_r2"], out["fib_s2"] = pp + 0.382 * r, pp - 0.382 * r, pp + 0.618 * r, pp - 0.618 * r
    out["woodie_pp"] = (H + L + 2 * O) / 4
    out["cam_r3"], out["cam_s3"], out["cam_r4"], out["cam_s4"] = C + r * 1.1 / 4, C - r * 1.1 / 4, C + r * 1.1 / 2, C - r * 1.1 / 2
    x = np.where(C < O, H + 2 * L + C, np.where(C > O, 2 * H + L + C, H + L + 2 * C))
    out["dm_pp"], out["dm_r1"], out["dm_s1"] = x / 4, x / 2 - L, x / 2 - H
    return out


def period_vwap(df: pd.DataFrame, key) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = (tp * df["volume"]).groupby(key).cumsum()
    v = df["volume"].groupby(key).cumsum()
    return pv / (v + EPS)


def vwap_band_pos(df: pd.DataFrame, key) -> pd.Series:
    """当期間の VWAP からの距離を、出来高加重の標準偏差で割ったもの（VWAP バンドの位置）。"""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    v = df["volume"]
    cv = v.groupby(key).cumsum()
    m = (tp * v).groupby(key).cumsum() / (cv + EPS)
    m2 = (tp * tp * v).groupby(key).cumsum() / (cv + EPS)
    sd = np.sqrt((m2 - m * m).clip(lower=0))
    return (df["close"] - m) / (sd + EPS)


def anchored_vwap_from_extreme(df: pd.DataFrame, n: int, at_min: bool) -> pd.Series:
    """直近 n 本の安値（高値）の足を起点にした VWAP。"""
    tp = ((df["high"] + df["low"] + df["close"]) / 3).to_numpy()
    v = df["volume"].to_numpy()
    cpv = np.r_[0.0, np.cumsum(tp * v)]
    cv = np.r_[0.0, np.cumsum(v)]
    src = df["low"].to_numpy() if at_min else df["high"].to_numpy()
    age = rolling_age_of_min(src, n) if at_min else rolling_age_of_max(src, n)
    i = np.arange(len(v))
    anchor = i - age
    out = np.full(len(v), np.nan)
    ok = ~np.isnan(age)
    a = anchor[ok].astype(int)
    out[ok] = (cpv[i[ok] + 1] - cpv[a]) / (cv[i[ok] + 1] - cv[a] + EPS)
    return pd.Series(out, index=df.index)


def volume_profile_window(df: pd.DataFrame, n: int, bins: int = 40, va: float = 0.7, chunk: int = 1024) -> pd.DataFrame:
    """直近 n 本の価格帯別出来高（各足の出来高を安値〜高値に均等に配る）。POC・バリューエリア・今の価格帯の出来高の割合。"""
    high, low, vol, close = (df[c].to_numpy(dtype=float) for c in ("high", "low", "volume", "close"))
    T = len(close)
    poc = np.full(T, np.nan)
    val = np.full(T, np.nan)
    vah = np.full(T, np.nan)
    at = np.full(T, np.nan)
    if T < n:
        return pd.DataFrame({"poc": poc, "val": val, "vah": vah, "at": at}, index=df.index)
    wh, wl, wv = (sliding_window_view(x, n) for x in (high, low, vol))
    for s in range(0, T - n + 1, chunk):
        H, L, V = wh[s:s + chunk], wl[s:s + chunk], wv[s:s + chunk]
        top, bot = H.max(axis=1), L.min(axis=1)
        width = (top - bot) / bins + EPS
        edges = bot[:, None] + width[:, None] * np.arange(bins + 1)[None, :]           # (m, bins+1)
        lo = np.maximum(L[:, :, None], edges[:, None, :-1])                          # (m, n, bins)
        hi = np.minimum(H[:, :, None], edges[:, None, 1:])
        frac = np.clip(hi - lo, 0, None) / (H - L + EPS)[:, :, None]
        prof = (frac * V[:, :, None]).sum(axis=1)                                     # (m, bins)
        tot = prof.sum(axis=1) + EPS
        ipoc = np.argmax(prof, axis=1)
        # バリューエリア: POC から出来高の多い側へ広げる
        m = len(prof)
        lo_i, hi_i = ipoc.copy(), ipoc.copy()
        acc = prof[np.arange(m), ipoc].copy()
        for _ in range(bins):
            need = acc < va * tot
            if not need.any():
                break
            left = np.where(lo_i > 0, prof[np.arange(m), np.maximum(lo_i - 1, 0)], -1.0)
            right = np.where(hi_i < bins - 1, prof[np.arange(m), np.minimum(hi_i + 1, bins - 1)], -1.0)
            go_left = need & (left >= right) & (left >= 0)
            go_right = need & ~go_left & (right >= 0)
            acc[go_left] += left[go_left]
            lo_i[go_left] -= 1
            acc[go_right] += right[go_right]
            hi_i[go_right] += 1
        idx = np.arange(s + n - 1, s + n - 1 + m)
        poc[idx] = bot + width * (ipoc + 0.5)
        val[idx] = bot + width * lo_i
        vah[idx] = bot + width * (hi_i + 1)
        cb = np.clip(((close[idx] - bot) / width).astype(int), 0, bins - 1)
        at[idx] = prof[np.arange(m), cb] / prof.max(axis=1)
    return pd.DataFrame({"poc": poc, "val": val, "vah": vah, "at": at}, index=df.index)
