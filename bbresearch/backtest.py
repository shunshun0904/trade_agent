"""Phase 6: イベント駆動バックテスト。

約定・決済の判定は Phase 3 の labeling の関数で済ませてある（labels テーブル）。
ここでは、その上に「同時に持てるポジションは1つ」「発注量」「最小数量」の規則を載せて
資金の推移を計算する。

ルール（SPEC.md §5 Phase 6）:
1. ペアごとに同時に持てるポジションは1つ。エントリー注文の待機中（t0 〜 約定または T_fill 経過）
   と保有中（〜 t_x）は新しいイベントを無視する。
2. 発注量はメタモデルの予測確率から決める（López de Prado 2018, 10.3 節）。
   比較用に、p ≥ θ で固定額を発注するしきい値方式と、常に発注する一次シグナルのみの方式がある。
3. 発注量が max(unit_amount, min_amount) 未満なら発注しない。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew


def bet_size(p: np.ndarray | float, step: float = 0.0) -> np.ndarray:
    """予測確率から発注量の比率 [0, 1] を求める（ロングのみなので負は 0）。

    z = (p − 1/2) / sqrt(p (1 − p))、m = 2 Φ(z) − 1。step > 0 なら step 刻みに切り捨てる。
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    z = (p - 0.5) / np.sqrt(p * (1 - p))
    m = np.clip(2 * norm.cdf(z) - 1, 0.0, 1.0)
    if step > 0:
        m = np.floor(m / step + 1e-9) * step
    return m


@dataclass(frozen=True)
class Strategy:
    name: str
    mode: str                 # "meta"（確率から発注量）/ "threshold"（p ≥ θ で固定額）/ "all"（常に固定額）
    theta: float = 0.5
    size_step: float = 0.0


@dataclass(frozen=True)
class Account:
    initial_capital: float = 1_000_000.0   # 円
    max_fraction: float = 1.0              # 発注量比率 1 のときに使う資金の割合
    amount_digits: int = 4
    min_amount: float = 0.0001             # max(unit_amount, min_amount)


def _floor_amount(x: float, digits: int) -> float:
    q = 10**digits
    return math.floor(x * q + 1e-9) / q


def run_backtest(
    events: pd.DataFrame, labels: pd.DataFrame, proba: pd.Series | None, strategy: Strategy,
    account: Account, t_fill: pd.Timedelta,
) -> pd.DataFrame:
    """取引の一覧を返す。events と labels は event_id で結合できること。

    未約定の注文も「発注したが約定しなかった」行として残す（約定率の計算に使う）。
    t_x が決まらない（データ末尾で決済が判定できない）イベント以降は打ち切る。
    """
    df = events.merge(labels, on="event_id", how="inner").sort_values("t0")
    if proba is not None:
        df["p"] = df["event_id"].map(proba)
    else:
        df["p"] = np.nan
    equity = account.initial_capital
    busy_until = pd.Timestamp.min.tz_localize("UTC")
    rows = []
    for r in df.itertuples(index=False):
        if r.t0 < busy_until:
            continue
        if strategy.mode == "all":
            frac = 1.0
        elif np.isnan(r.p):
            continue
        elif strategy.mode == "threshold":
            frac = 1.0 if r.p >= strategy.theta else 0.0
        elif strategy.mode == "meta":
            frac = float(bet_size(r.p, strategy.size_step))
        else:
            raise ValueError(strategy.mode)
        if frac <= 0:
            continue
        amount = _floor_amount(equity * account.max_fraction * frac / r.P_e, account.amount_digits)
        if amount < account.min_amount:
            continue
        if not r.filled:
            busy_until = r.t0 + t_fill
            rows.append({"event_id": r.event_id, "t0": r.t0, "signal_type": r.signal_type, "p": r.p,
                         "frac": frac, "amount": amount, "filled": False})
            continue
        if pd.isna(r.t_x):
            break
        busy_until = r.t_x
        entry_fee = amount * r.P_e * r.f_e
        exit_fee = amount * r.P_x * r.f_x
        pnl = amount * (r.P_x - r.P_e) - entry_fee - exit_fee
        equity += pnl
        rows.append({
            "event_id": r.event_id, "t0": r.t0, "signal_type": r.signal_type, "p": r.p, "frac": frac,
            "amount": amount, "filled": True, "t_f": r.t_f, "t_x": r.t_x, "exit_type": r.exit_type,
            "P_e": r.P_e, "P_x": r.P_x, "ret_net": r.ret_net, "pnl": pnl,
            "entry_fee": entry_fee, "exit_fee": exit_fee, "equity": equity,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 評価

def equity_curve(trades: pd.DataFrame, initial: float, start, end) -> pd.Series:
    """日次の資産推移（決済時点で損益を計上）。

    期間の終わり直前に建てて期間を越えて決済した取引も含めるよう、最後の決済日まで伸ばす。
    """
    if trades.empty or "pnl" not in trades:
        f = None
        last = pd.Timestamp(end)
    else:
        f = trades[trades["filled"]]
        last = max(pd.Timestamp(end), f["t_x"].max()) if len(f) else pd.Timestamp(end)
    days = pd.date_range(pd.Timestamp(start).floor("D"), last.ceil("D"), freq="D")
    if f is None or f.empty:
        return pd.Series(initial, index=days, dtype="float64")
    daily = f.groupby(f["t_x"].dt.floor("D"))["pnl"].sum()
    return initial + daily.reindex(days, fill_value=0.0).cumsum()


def sharpe(daily_ret: pd.Series, periods: int = 365) -> float | None:
    sd = daily_ret.std()
    if not np.isfinite(sd) or sd == 0:
        return None
    return float(daily_ret.mean() / sd * math.sqrt(periods))


def max_drawdown(curve: pd.Series) -> float:
    return float((curve / curve.cummax() - 1).min()) if len(curve) else 0.0


def summarize(trades: pd.DataFrame, account: Account, start, end) -> dict:
    curve = equity_curve(trades, account.initial_capital, start, end)
    daily = curve.pct_change().dropna()
    out: dict = {
        "n_orders": int(len(trades)),
        "n_trades": 0,
        "total_return": float(curve.iloc[-1] / account.initial_capital - 1),
        "max_drawdown": max_drawdown(curve),
        "sharpe": sharpe(daily),
    }
    if trades.empty:
        return out
    out["fill_rate"] = float(trades["filled"].mean())
    f = trades[trades["filled"]]
    out["n_trades"] = int(len(f))
    if f.empty:
        return out
    wins, losses = f[f["pnl"] > 0], f[f["pnl"] <= 0]
    held = (f["t_x"] - f["t_f"]).sum()
    out.update(
        win_rate=float(len(wins) / len(f)),
        avg_win=float(wins["pnl"].mean()) if len(wins) else None,
        avg_loss=float(losses["pnl"].mean()) if len(losses) else None,
        avg_ret_net=float(f["ret_net"].mean()),
        time_in_market=float(held / (pd.Timestamp(end) - pd.Timestamp(start))),
        maker_fee_total=float(f["entry_fee"].sum() + f.loc[f["exit_type"] == "tp", "exit_fee"].sum()),
        taker_fee_total=float(f.loc[f["exit_type"] != "tp", "exit_fee"].sum()),
        by_exit_type={
            k: {"n": int(len(g)), "pnl": float(g["pnl"].sum()), "avg_ret_net": float(g["ret_net"].mean())}
            for k, g in f.groupby("exit_type")
        },
        daily_ret_skew=float(skew(daily)) if len(daily) > 2 else None,
        daily_ret_kurt=float(kurtosis(daily, fisher=False)) if len(daily) > 3 else None,
        n_days=int(len(daily)),
    )
    return out


def buy_and_hold(bars: pd.DataFrame, start, end, maker_fee: float, taker_fee: float) -> dict:
    """期間の最初の close で買い、最後の close で売る（エントリーはメイカー、決済はテイカー）。"""
    b = bars.loc[(bars.index >= start) & (bars.index < end), "close"].dropna()
    if b.empty:
        return {}
    ret = b.iloc[-1] * (1 - taker_fee) / (b.iloc[0] * (1 + maker_fee)) - 1
    daily = b.resample("D").last().pct_change().dropna()
    return {"total_return": float(ret), "sharpe": sharpe(daily), "max_drawdown": max_drawdown(b)}


def deflated_sharpe(sr: float, sr_trials: list[float], n_obs: int, sk: float, ku: float) -> float | None:
    """Deflated Sharpe Ratio（Bailey & López de Prado 2014）。

    sr と sr_trials は年率化していない（観測1期間あたりの）シャープレシオ。ku は通常の尖度（正規で 3）。
    試行数 N = len(sr_trials)。N < 2 のときは期待最大値の補正ができないため None。
    """
    n = len(sr_trials)
    if n < 2 or n_obs < 3:
        return None
    var = float(np.var(sr_trials, ddof=1))
    g = 0.5772156649
    sr0 = math.sqrt(var) * ((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * math.e)))
    denom = 1 - sk * sr + (ku - 1) / 4 * sr**2
    if denom <= 0:
        return None
    return float(norm.cdf((sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom)))
