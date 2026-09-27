"""スイング（4 時間足で判断し、1〜3 日保有する）の方向の研究。2026-09-28 オーナー決定。

仮説（数値は configs/swing.yaml に事前に固定し、全期間で 1 回だけ評価する）:
- H1 時系列モメンタム（tsmom）: ペアごとに、直近 L 日のリターンが正（ret）、または終値が L 日移動平均以上（ma）なら保有する。
- H2 横断モメンタム（xsmom）: 直近 L 日のリターンで順位を付け、上位 k 銘柄のうちリターンが正のものを 1/k ずつ保有する。
- H3 急落後の反発（rebound）: 直近 N 本の対数リターンが -k・σ・√N 以下の足で買う（σ は急落の前の 30 日の 1 本あたりの標準偏差）。

共通の約束:
- 判断は足 t の終値の時点で、足 t までのデータだけを使う（index は足の開始時刻、終値は開始の 4 時間後）。
  約定は足 t の終値に片側の費用（手数料 + 滑り）を足したものとみなす。
- 対象は時点ごとに決める: 履歴が min_bars 本以上、直近 turnover_days 日の「24 時間の出来高（JPY）」の中央値が閾値以上。
- 配分はリスク均等（H1・H3 と比較用の常時保有）: 対象ペアの直近 sigma_days 日の実現ボラの逆数に比例させ、合計を 1 にする。
  信号が出ていない枠は JPY（リターン 0）。H2 は 1/k ずつの等分。
- 保有期間 H 本: 判断の時刻を 1 本ずつずらした H 個の部分ポートフォリオ（それぞれ H 本ごとに判断し直す）の平均を持つ。
  これは直近 H 回の判断の重みの平均に等しい（判断のタイミングの運を消す）。
- 足の間は重みを一定とみなす（足の中の値動きによる重みのずれと、その調整の費用は無視する）。
- 方向の情報の有無は、同じ配分で常に保有する場合（同じ H）に対するアルファで測る（相場全体の上昇の分を除く）。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from bbdata.download import to_utc
from bbresearch.backtest import deflated_sharpe

BARS_PER_DAY = 6
BARS_PER_YEAR = BARS_PER_DAY * 365
HAC_LAGS = 5  # 日次の回帰の Newey-West の次数（保有は最大 3 日）


# ------------------------------------------------------------------ データ

def panel(candles: dict[str, pd.DataFrame], start, end) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ペアごとの公式 4 時間足から、終値と出来高（JPY）の表を作る（index = 4 時間ごとの足の開始時刻、列 = ペア）。

    最初の足より前は NaN。その後で足が抜けている時刻は、終値を前の足から引き継ぎ、出来高を 0 とする。
    """
    idx = pd.date_range(to_utc(start), to_utc(end), freq="4h", inclusive="left")
    close, vol = {}, {}
    for p, df in candles.items():
        if df.empty:
            continue
        listed = idx >= df.index.min()
        close[p] = df["close"].reindex(idx).ffill().where(listed)
        vol[p] = (df["volume"] * df["close"]).reindex(idx).fillna(0.0).where(listed)
    return pd.DataFrame(close, index=idx), pd.DataFrame(vol, index=idx)


def eligible(close: pd.DataFrame, vol_jpy: pd.DataFrame, min_bars: int, turnover_days: int,
             min_turnover: float) -> pd.DataFrame:
    """時点ごとの対象ペア。履歴 min_bars 本以上、直近 turnover_days 日の 24 時間出来高（JPY）の中央値 ≥ min_turnover。"""
    hist = close.notna().cumsum()
    v24 = vol_jpy.rolling(BARS_PER_DAY, min_periods=BARS_PER_DAY).sum()
    w = turnover_days * BARS_PER_DAY
    med = v24.rolling(w, min_periods=int(w * 0.8)).median()
    return (hist >= min_bars) & (med >= min_turnover) & close.notna()


def bar_sigma(close: pd.DataFrame, days: int) -> pd.DataFrame:
    """足 t までの直近 days 日の、1 本あたりの対数リターンの標準偏差。"""
    r = np.log(close).diff()
    w = days * BARS_PER_DAY
    return r.rolling(w, min_periods=int(w * 0.8)).std()


def risk_parity(sigma: pd.DataFrame, elig: pd.DataFrame) -> pd.DataFrame:
    """対象ペアに、ボラの逆数に比例した重み（合計 1）。対象がない時点はすべて 0。"""
    inv = (1.0 / sigma).where(elig & (sigma > 0))
    return inv.div(inv.sum(axis=1), axis=0).fillna(0.0)


# ------------------------------------------------------------------ 信号

def tsmom_signal(close: pd.DataFrame, days: int, kind: str) -> pd.DataFrame:
    """時系列モメンタム。ret: 直近 days 日のリターンが正。ma: 終値が days 日の単純移動平均以上。"""
    n = days * BARS_PER_DAY
    if kind == "ret":
        return close / close.shift(n) - 1 > 0
    if kind == "ma":
        return close >= close.rolling(n, min_periods=n).mean()
    raise ValueError(f"未知の kind: {kind}")


def xsmom_weights(close: pd.DataFrame, elig: pd.DataFrame, days: int, k: int) -> pd.DataFrame:
    """横断モメンタム。対象ペアを直近 days 日のリターンで順位付けし、上位 k のうちリターンが正のものに 1/k ずつ。"""
    n = days * BARS_PER_DAY
    mom = (close / close.shift(n) - 1).where(elig)
    rank = mom.rank(axis=1, ascending=False, method="first")
    return ((rank <= k) & (mom > 0)).astype(float) / k


def crash_events(close: pd.DataFrame, bars: int, k: float, sigma_days: int) -> pd.DataFrame:
    """直近 bars 本の対数リターンが -k・σ・√bars 以下の足。σ は急落の窓より前の sigma_days 日で測る。"""
    lr = np.log(close)
    move = lr - lr.shift(bars)
    sig = bar_sigma(close, sigma_days).shift(bars)
    return move <= -k * sig * math.sqrt(bars)


# ------------------------------------------------------------------ 保有と損益

def stagger(w_dec: pd.DataFrame, hold: int) -> pd.DataFrame:
    """直近 hold 回の判断の重みの平均（まだ判断していない部分ポートフォリオは JPY）。"""
    return w_dec.fillna(0.0).rolling(hold, min_periods=1).sum() / hold


def simulate(w: pd.DataFrame, close: pd.DataFrame, cost: dict[str, float]) -> pd.DataFrame:
    """足 t の終値で重みを w[t] にして足 t+1 の間持つ。index は足（その足の間のリターン）。

    取引量は重みの変化の絶対値、費用はペアごとの片側の率 × 取引量で、取引した次の足に計上する。
    """
    r = close.pct_change(fill_method=None).fillna(0.0)
    w = w.reindex(index=close.index, columns=close.columns).fillna(0.0)
    held = w.shift(1).fillna(0.0)
    trade = w.diff().abs()
    trade.iloc[0] = w.iloc[0].abs()
    c = pd.Series({p: cost[p] for p in close.columns})
    fee = (trade * c).sum(axis=1).shift(1).fillna(0.0)
    gross = (held * r).sum(axis=1)
    return pd.DataFrame({"gross": gross, "cost": fee, "net": gross - fee, "exposure": held.sum(axis=1),
                         "turnover": trade.sum(axis=1).shift(1).fillna(0.0)})


def daily(ret: pd.Series) -> pd.Series:
    """足ごとの単純リターンを UTC の日ごとに複利でまとめる。"""
    return (1 + ret).resample("D").prod() - 1


def stats(ret: pd.Series) -> dict:
    """足ごとの単純リターンの年率リターン・ボラ・シャープ・最大ドローダウン。"""
    ret = ret.dropna()
    eq = (1 + ret).cumprod()
    years = len(ret) / BARS_PER_YEAR
    sd = ret.std(ddof=1)
    return {"bars": int(len(ret)), "ann_return": float(eq.iloc[-1] ** (1 / years) - 1),
            "ann_vol": float(sd * math.sqrt(BARS_PER_YEAR)),
            "sharpe": float(ret.mean() / sd * math.sqrt(BARS_PER_YEAR)) if sd > 0 else None,
            "max_drawdown": float((eq / eq.cummax() - 1).min()), "total": float(eq.iloc[-1] - 1)}


def hac_alpha(y: pd.Series, x: pd.Series, lags: int = HAC_LAGS) -> dict:
    """y = a + b x + e の OLS。a の t 値は Newey-West（Bartlett 核）の標準誤差による。active = y - b x（= a + e）。"""
    df = pd.concat([y, x], axis=1).dropna()
    Y = df.iloc[:, 0].to_numpy()
    X = np.column_stack([np.ones(len(df)), df.iloc[:, 1].to_numpy()])
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    e = Y - X @ beta
    xe = X * e[:, None]
    S = xe.T @ xe
    for lag in range(1, lags + 1):
        g = xe[lag:].T @ xe[:-lag]
        S += (1 - lag / (lags + 1)) * (g + g.T)
    inv = np.linalg.inv(X.T @ X)
    se = math.sqrt((inv @ S @ inv)[0, 0])
    return {"alpha": float(beta[0]), "beta": float(beta[1]), "t_alpha": float(beta[0] / se) if se > 0 else None,
            "active": pd.Series(Y - beta[1] * X[:, 1], index=df.index)}


# ------------------------------------------------------------------ 事前登録した組み合わせの評価

def configs(cfg: dict) -> list[dict]:
    """configs/swing.yaml の値から、評価するすべての組み合わせ（= 試行）を作る。"""
    out = []
    t = cfg["tsmom"]
    for kind in t["kinds"]:
        for days in t["lookback_days"]:
            for hold in cfg["hold_days"]:
                out.append({"key": f"tsmom/{kind}/L{days}d/H{hold}d", "family": "tsmom", "kind": kind,
                            "lookback_days": days, "hold_days": hold})
    x = cfg["xsmom"]
    for days in x["lookback_days"]:
        for hold in cfg["hold_days"]:
            out.append({"key": f"xsmom/L{days}d/k{x['top_k']}/H{hold}d", "family": "xsmom", "lookback_days": days,
                        "top_k": x["top_k"], "hold_days": hold})
    b = cfg["rebound"]
    for bars in b["window_bars"]:
        for k in b["k_sigma"]:
            for hold in cfg["hold_days"]:
                out.append({"key": f"rebound/N{bars}/k{k}/H{hold}d", "family": "rebound", "window_bars": bars,
                            "k_sigma": k, "hold_days": hold})
    return out


def decisions(c: dict, close: pd.DataFrame, elig: pd.DataFrame, base: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """組み合わせ c の、足ごとの判断の重み（部分ポートフォリオ 1 個ぶん。合計 1 以下）。"""
    if c["family"] == "tsmom":
        return base.where(tsmom_signal(close, c["lookback_days"], c["kind"]), 0.0)
    if c["family"] == "xsmom":
        return xsmom_weights(close, elig, c["lookback_days"], c["top_k"])
    if c["family"] == "rebound":
        ev = crash_events(close, c["window_bars"], c["k_sigma"], cfg["sigma_days"]) & elig
        return base.where(ev, 0.0)
    raise ValueError(c["family"])


def yearly_active(active: pd.Series) -> dict[int, float]:
    """日次のアクティブリターン（y - b x）の年ごとの合計。"""
    return {int(y): float(g.sum()) for y, g in active.groupby(active.index.year)}


def evaluate(close: pd.DataFrame, vol_jpy: pd.DataFrame, cost: dict[str, float], cfg: dict,
             cost_mult: float = 1.0, only: set[str] | None = None) -> dict:
    """事前登録したすべての組み合わせを、同じ期間・同じ費用で評価する。

    評価期間は、対象ペアが 1 つ以上ある最初の足から close の最後の足まで。比較の基準は、同じ配分（リスク均等）を
    常に保有し、同じ H でずらした部分ポートフォリオの平均（常時保有）。
    """
    cost = {p: v * cost_mult for p, v in cost.items()}
    elig = eligible(close, vol_jpy, cfg["min_bars"], cfg["turnover_days"], cfg["min_turnover_jpy"])
    base = risk_parity(bar_sigma(close, cfg["sigma_days"]), elig)
    has = elig.any(axis=1)
    if not has.any():
        raise ValueError("対象ペアがない")
    start = has.idxmax()
    win = close.index >= start

    bench = {}
    for hold in cfg["hold_days"]:
        sim = simulate(stagger(base, hold * BARS_PER_DAY), close, cost)[win]
        bench[hold] = sim
    results = {}
    for c in configs(cfg):
        if only is not None and c["key"] not in only:
            continue
        hold = c["hold_days"] * BARS_PER_DAY
        sim = simulate(stagger(decisions(c, close, elig, base, cfg), hold), close, cost)[win]
        y, x = daily(sim["net"]), daily(bench[c["hold_days"]]["net"])
        reg = hac_alpha(y, x)
        act = reg.pop("active")
        sd = act.std(ddof=1)
        results[c["key"]] = c | stats(sim["net"]) | {
            "exposure": float(sim["exposure"].mean()),
            "turnover_per_year": float(sim["turnover"].sum() / (len(sim) / BARS_PER_YEAR)),
            "cost_per_year": float(sim["cost"].sum() / (len(sim) / BARS_PER_YEAR)),
            "alpha_ann": reg["alpha"] * 365, "beta": reg["beta"], "t_alpha": reg["t_alpha"],
            "active_sr_daily": float(act.mean() / sd) if sd > 0 else None,
            "active_skew": float(skew(act)), "active_kurt": float(kurtosis(act, fisher=False)),
            "n_days": int(len(act)), "yearly_active": yearly_active(act),
            "yearly_return": {int(yy): float(np.prod(1 + g) - 1) for yy, g in y.groupby(y.index.year)},
        }
    return {"start": start, "end": close.index[-1], "results": results,
            "benchmarks": {f"always/H{h}d": stats(b["net"]) | {"yearly_return": {
                int(yy): float(np.prod(1 + g) - 1) for yy, g in daily(b["net"]).groupby(daily(b["net"]).index.year)}}
                for h, b in bench.items()},
            "eligible_count": elig[win].sum(axis=1)}


def deflate(results: dict) -> dict[str, float | None]:
    """すべての試行のアクティブの日次シャープを並べ、組み合わせごとのデフレートシャープ（試行数 = 組み合わせの数）。"""
    srs = [r["active_sr_daily"] for r in results.values() if r["active_sr_daily"] is not None]
    return {k: (deflated_sharpe(r["active_sr_daily"], srs, r["n_days"], r["active_skew"], r["active_kurt"])
                if r["active_sr_daily"] is not None else None) for k, r in results.items()}


def judge(results: dict, dsr: dict, family: str, min_share: float, min_dsr: float, min_year_share: float,
          full_years: list[int]) -> dict:
    """事前登録した判定。(1) アルファが正の組み合わせの割合、(2) 最良（アクティブのシャープが最大）の DSR、
    (3) 最良の年ごとのアクティブリターンが正の年の割合（full_years だけで数える）。3 つとも満たせば支持。"""
    fam = {k: r for k, r in results.items() if r["family"] == family}
    share = float(np.mean([r["alpha_ann"] > 0 for r in fam.values()]))
    best = max(fam, key=lambda k: fam[k]["active_sr_daily"] if fam[k]["active_sr_daily"] is not None else -np.inf)
    ya = fam[best]["yearly_active"]
    years = [y for y in full_years if y in ya]
    pos = float(np.mean([ya[y] > 0 for y in years])) if years else 0.0
    d = dsr.get(best)
    ok = share >= min_share and d is not None and d >= min_dsr and pos >= min_year_share
    return {"family": family, "n": len(fam), "share_alpha_pos": share, "best": best, "best_dsr": d,
            "best_year_share": pos, "supported": bool(ok)}


def event_study(close: pd.DataFrame, events: pd.DataFrame, elig: pd.DataFrame, hold: int) -> dict:
    """急落の足の後 hold 本の対数リターン（ペアをまとめる）と、対象の全時点の同じリターンの平均の比較。

    急落は連続した足をひとまとめにした「局面」の最初の足で数える（重なりを減らすため）。記述統計で、判定には使わない。
    """
    lr = np.log(close)
    fwd = lr.shift(-hold) - lr
    ev = events & elig
    first = ev & ~ev.shift(1, fill_value=False)
    e = fwd[first].stack().dropna()
    u = fwd[elig].iloc[::hold].stack().dropna()
    se = e.std(ddof=1) / math.sqrt(len(e)) if len(e) > 1 else None
    return {"episodes": int(len(e)), "mean_fwd": float(e.mean()) if len(e) else None,
            "t": float(e.mean() / se) if se else None, "uncond_mean_fwd": float(u.mean()) if len(u) else None}
