"""ルール A: 価格帯別出来高のバリューエリア下端（VAL）での反発（2026-09-24 オーナー決定、数値は検証前に固定）。

    python -m bbresearch val-rule --config configs/research.yaml --rule configs/val_rule.yaml --out reports/val_rule

1 分ごとに、直近 24 時間 [t − 24h, t) の約定から価格帯別出来高を作る（刻み 0.25σ、バリューエリア 70%）。
ポジションも注文もないときに、次をすべて満たせば VAL に post_only 買い指値を出す。
- 現在値（t 直前の約定価格）がバリューエリアの中（VAL < 価格 ≤ VAH）
- POC と VAL の差が VAL の min_reward（0.3%）以上
注文は valid_min（60 分）有効。約定しなければ取り消し、その時点で計算し直す。
約定したら、利確は約定時点の POC に post_only 売り指値（メイカー）、損切りは VAL − stop_frac × (POC − VAL)
以下の約定で成行、時間切れは hold_min（4 時間）後に指値（grace_min 分で成行に切り替え）。

実装: 約定を対数価格の細かい刻み（fine_log_step、0.01%）に割り当て、整数の数量で 24 時間窓の出来高を
1 分ずつ差分更新する（浮動小数の誤差をためない）。各時点で、それを close 基準の 0.25σ の刻みにまとめ直す。
σ は t 以前に終わった最新の 15 分足の σ（signals.ewm_sigma）。現在値から ±window_log_range の外の出来高は数えない。

比較対象: ルールが指値を出した同じ時刻に、指値の深さと利確・損切りの距離（現在値に対する比率）を
ルールの注文どうしで入れ替えて（VAL とは無関係な価格に）指値を出した場合の、注文ごとの損益。
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bbdata.download import to_utc

from . import backtest as bt
from .labeling import TradeTape, ceil_price, fill_time, first_exit, floor_price, net_return
from .maker_exit import maker_time_exit
from .model import params_hash
from .pipeline import barrier_params, daily_sr, load_market
from .profile import value_area
from .signals import ewm_sigma

log = logging.getLogger(__name__)
MIN_MS = 60_000


@dataclass(frozen=True)
class ValRuleParams:
    window_hours: float = 24.0
    bin_sigma: float = 0.25
    value_area: float = 0.70
    min_reward: float = 0.003
    valid_min: int = 60
    stop_frac: float = 0.5
    hold_min: int = 240
    grace_min: int = 15
    sigma_span: int = 96
    fine_log_step: float = 1e-4
    window_log_range: float = 0.25


class RollingProfile:
    """[t − window, t) の約定の出来高を、対数価格の細かい刻みで保持する（整数の数量で差分更新）。"""

    def __init__(self, tape: TradeTape, fine_log_step: float, amount_scale: float = 1e4) -> None:
        self.tape = tape
        self.step = fine_log_step
        lp = np.log(tape.price)
        self.fb = np.floor(lp / fine_log_step).astype(np.int64)
        self.off = int(self.fb.min()) if len(self.fb) else 0
        self.qty = np.rint(tape.amount * amount_scale).astype(np.int64)
        self.hist = np.zeros(int(self.fb.max()) - self.off + 1 if len(self.fb) else 1, dtype=np.int64)
        self.lo = 0  # 窓の最初の約定の位置
        self.hi = 0  # 窓の後ろ（t 以降の最初の約定の位置）

    def advance(self, t_ms: int, window_ms: int) -> None:
        ts = self.tape.ts
        new_hi = int(np.searchsorted(ts, t_ms, side="left"))
        new_lo = int(np.searchsorted(ts, t_ms - window_ms, side="left"))
        if new_hi < self.hi or new_lo < self.lo:
            raise ValueError("時刻は進める方向にしか動かせない")
        if new_hi > self.hi:
            np.add.at(self.hist, self.fb[self.hi:new_hi] - self.off, self.qty[self.hi:new_hi])
        if new_lo > self.lo:
            np.add.at(self.hist, self.fb[self.lo:new_lo] - self.off, -self.qty[self.lo:new_lo])
        self.lo, self.hi = new_lo, new_hi

    def last_price(self) -> float | None:
        return float(self.tape.price[self.hi - 1]) if self.hi > self.lo else None

    def levels(self, price: float, sigma: float, prm: ValRuleParams) -> dict | None:
        """現在値を基準に 0.25σ 刻みにまとめ直し、(POC, VAL, VAH) の価格を返す。"""
        if not (np.isfinite(sigma) and sigma > 0):
            return None
        lp = np.log(price)
        c = int(np.floor(lp / self.step)) - self.off
        r = int(prm.window_log_range / self.step)
        a, b = max(0, c - r), min(len(self.hist), c + r + 1)
        h = self.hist[a:b]
        if h.sum() <= 0:
            return None
        w = prm.bin_sigma * sigma  # 対数価格での刻み幅（σ は対数リターンの標準偏差）
        centers = (np.arange(a, b) + self.off + 0.5) * self.step
        k = np.floor((centers - lp) / w).astype(np.int64)
        kmin = int(k.min())
        counts = np.bincount(k - kmin, weights=h.astype(float))
        poc, lo, hi = value_area(counts, prm.value_area)
        return {"poc": float(np.exp(lp + (poc + kmin + 0.5) * w)),
                "val": float(np.exp(lp + (lo + kmin) * w)),
                "vah": float(np.exp(lp + (hi + kmin + 1) * w))}


def _exit(tape: TradeTape, idx: int, tf: int, u: float, l: float, prm: ValRuleParams, fees, s_slip, tick):
    t_v = tf + prm.hold_min * MIN_MS
    ex = first_exit(tape, idx, tf, u, l, t_v, s_slip)
    if ex is None:
        return None
    et, tx, px = ex
    if et == "time":
        ex2 = maker_time_exit(tape, t_v, l, prm.grace_min * MIN_MS, s_slip, tick)
        if ex2 is None:
            return None
        et, tx, px = ex2
    f_x = fees[0] if et in ("tp", "time_maker") else fees[1]
    return et, tx, px, f_x


def simulate(tape: TradeTape, bars: pd.DataFrame, start, end, prm: ValRuleParams, fees: tuple[float, float],
             s_slip: float, tick: float, fill_rule: str = "strict") -> tuple[pd.DataFrame, pd.DataFrame]:
    """ルール A を [start, end) で逐次に実行する。戻り値は (注文の一覧, 取引の一覧)。"""
    start, end = to_utc(start), to_utc(end)
    sigma = ewm_sigma(bars, prm.sigma_span)
    step = bars.index[1] - bars.index[0]
    bar_start_ms = np.array([t.value // 1_000_000 for t in bars.index], dtype="int64")
    step_ms = int(step / pd.Timedelta(milliseconds=1))
    sig_v = sigma.to_numpy()
    window_ms = int(prm.window_hours * 3_600_000)
    rp = RollingProfile(tape, prm.fine_log_step)
    t = int((start + pd.Timedelta(hours=prm.window_hours)).ceil("min").value // 1_000_000)  # 最初の 24 時間は準備
    t_end = int(end.value // 1_000_000)
    orders, trades = [], []
    while t < t_end:
        rp.advance(t, window_ms)
        price = rp.last_price()
        j = int(np.searchsorted(bar_start_ms, t - step_ms, side="right")) - 1  # t 以前に終わった最新の足
        s0 = sig_v[j] if j >= 0 else np.nan
        lv = rp.levels(price, s0, prm) if price is not None else None
        if lv is None or not (lv["val"] < price <= lv["vah"]) or lv["poc"] < lv["val"] * (1 + prm.min_reward):
            t += MIN_MS
            continue
        p_e = floor_price(lv["val"], tick)
        u = ceil_price(lv["poc"], tick)
        l = p_e - prm.stop_frac * (u - p_e)
        valid_end = min(t + prm.valid_min * MIN_MS, t_end)
        f = fill_time(tape, p_e, t, valid_end, fill_rule)
        order = {"t": t, "price": price, "P_e": p_e, "U": u, "L": l, "poc": lv["poc"], "vah": lv["vah"],
                 "filled": f is not None}
        orders.append(order)
        if f is None:
            t = valid_end
            continue
        tf, idx = f
        ex = _exit(tape, idx, tf, u, l, prm, fees, s_slip, tick)
        if ex is None:
            break  # データの末尾で決済を判定できない
        et, tx, px, f_x = ex
        r = net_return(p_e, px, fees[0], f_x)
        trades.append({"t_order": t, "t_f": tf, "t_x": tx, "P_e": p_e, "P_x": px, "exit_type": et,
                       "f_e": fees[0], "f_x": f_x, "ret_net": r})
        t = int(np.ceil((tx + 1) / MIN_MS) * MIN_MS)
    o = pd.DataFrame(orders)
    tr = pd.DataFrame(trades)
    for df, cols in ((o, ["t"]), (tr, ["t_order", "t_f", "t_x"])):
        for c in cols:
            if c in df:
                df[c] = pd.to_datetime(df[c], unit="ms", utc=True)
    return o, tr


def control_orders(tape: TradeTape, orders: pd.DataFrame, prm: ValRuleParams, fees, s_slip: float, tick: float,
                   rng: np.random.Generator, fill_rule: str = "strict") -> np.ndarray:
    """比較対象: ルールの注文と同じ時刻に、深さ・利確・損切りの比率を注文どうしで入れ替えた指値の、注文ごとの損益
    （約定したものだけ）。VAL とは関係のない価格になる。"""
    if orders.empty:
        return np.array([])
    depth = 1 - orders["P_e"] / orders["price"]
    up = orders["U"] / orders["P_e"] - 1
    dn = 1 - orders["L"] / orders["P_e"]
    perm = rng.permutation(len(orders))
    out = []
    for (t, price), d, uu, dd in zip(orders[["t", "price"]].itertuples(index=False), depth.to_numpy()[perm],
                                     up.to_numpy()[perm], dn.to_numpy()[perm]):
        t_ms = int(t.value // 1_000_000)
        p_e = floor_price(price * (1 - d), tick)
        f = fill_time(tape, p_e, t_ms, t_ms + prm.valid_min * MIN_MS, fill_rule)
        if f is None:
            continue
        tf, idx = f
        ex = _exit(tape, idx, tf, ceil_price(p_e * (1 + uu), tick), p_e * (1 - dd), prm, fees, s_slip, tick)
        if ex is None:
            continue
        et, tx, px, f_x = ex
        out.append(net_return(p_e, px, fees[0], f_x))
    return np.array(out)


def _stats(x) -> dict:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {"n": 0}
    sd = x.std(ddof=1) if len(x) > 1 else np.nan
    return {"n": int(len(x)), "mean": float(x.mean()), "t": float(x.mean() / sd * np.sqrt(len(x))) if sd > 0 else None,
            "win": float((x > 0).mean()), "sum": float(x.sum())}


def run_val_rule(cfg: dict, rule_cfg: dict, market=None) -> dict:
    market = market or load_market(cfg)
    prm = ValRuleParams(**rule_cfg.get("params", {}))
    bprm = barrier_params(cfg, market.spec)
    fees = (bprm.maker_fee, bprm.taker_fee)
    start = to_utc(rule_cfg.get("start", cfg["data"]["start"]))
    end = to_utc(rule_cfg.get("end", cfg["data"]["end"]))
    orders, trades = simulate(market.tape, market.bars, start, end, prm, fees, bprm.s_slip, market.tick, bprm.fill_rule)
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    controls = [control_orders(market.tape, orders, prm, fees, bprm.s_slip, market.tick, rng, bprm.fill_rule)
                for _ in range(int(rule_cfg.get("n_control", 5)))]

    b = cfg["backtest"]
    account = bt.Account(initial_capital=float(b["initial_capital"]), max_fraction=float(b["max_fraction"]),
                         amount_digits=int(market.spec["amount_digits"]), min_amount=market.min_amount)
    # 資金推移（全額を投じて複利）
    eq = float(account.initial_capital)
    rows = []
    for tr in trades.itertuples(index=False):
        amt = np.floor(eq * account.max_fraction / tr.P_e * 10**account.amount_digits) / 10**account.amount_digits
        pnl = amt * (tr.P_x * (1 - tr.f_x) - tr.P_e * (1 + tr.f_e))
        eq += pnl
        rows.append({"filled": True, "t_f": tr.t_f, "t_x": tr.t_x, "pnl": pnl, "ret_net": tr.ret_net,
                     "exit_type": tr.exit_type, "entry_fee": amt * tr.P_e * tr.f_e, "exit_fee": amt * tr.P_x * tr.f_x})
    summ = bt.summarize(pd.DataFrame(rows), account, start, end)

    years = {}
    if not orders.empty:
        for y in range(start.year, end.year + 1):
            ys, ye = max(start, pd.Timestamp(f"{y}-01-01", tz="UTC")), min(end, pd.Timestamp(f"{y + 1}-01-01", tz="UTC"))
            o = orders[(orders["t"] >= ys) & (orders["t"] < ye)]
            tr = trades[(trades["t_order"] >= ys) & (trades["t_order"] < ye)] if not trades.empty else trades
            years[y] = {"orders": int(len(o)), "fill_rate": float(o["filled"].mean()) if len(o) else None,
                        "trades": _stats(tr["ret_net"] if len(tr) else []),
                        "exit_types": tr["exit_type"].value_counts().to_dict() if len(tr) else {},
                        "buy_and_hold": bt.buy_and_hold(market.bars, ys, ye, fees[0], fees[1]).get("total_return")}
    th = params_hash({"val_rule": asdict(prm), "period": [str(start), str(end)], "fees": list(fees), "s_slip": bprm.s_slip})
    log_path = Path(cfg.get("experiment_log", "reports/experiments.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fh:
        fh.write(json.dumps({"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stage": "val_rule",
                             "trial_hash": th, "code_version": os.environ.get("GITHUB_SHA"), "key": "val_rule/A",
                             "daily_sr": daily_sr(summ), "total_return": summ["total_return"],
                             "n_trades": summ["n_trades"]}) + "\n")
    return {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "params": asdict(prm),
        "period": [str(start), str(end)], "fees": {"maker": fees[0], "taker": fees[1], "s_slip": bprm.s_slip},
        "n_orders": int(len(orders)), "fill_rate": float(orders["filled"].mean()) if len(orders) else None,
        "trades": _stats(trades["ret_net"] if len(trades) else []),
        "exit_types": trades["exit_type"].value_counts().to_dict() if len(trades) else {},
        "exit_means": trades.groupby("exit_type")["ret_net"].mean().to_dict() if len(trades) else {},
        "backtest": summ, "buy_and_hold": bt.buy_and_hold(market.bars, start, end, fees[0], fees[1]),
        "control": [_stats(c) for c in controls], "years": years, "trial_hash": th,
    }


def _p(x, pct=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x * 100:+.3f}%" if pct else f"{x:.3g}"


def render(rep: dict) -> str:
    t, s = rep["trades"], rep["backtest"]
    md = [f"# ルール A: VAL での反発（{rep['run_at']}）\n",
          f"- 期間: {rep['period'][0]} 〜 {rep['period'][1]}（検証前に固定した数値で1回だけ評価）",
          f"- 数値: `{rep['params']}`",
          f"- 手数料: メイカー {rep['fees']['maker']}、テイカー {rep['fees']['taker']}、成行の滑り {rep['fees']['s_slip']}\n",
          "## 全体\n",
          f"- 注文 {rep['n_orders']}、約定率 {_p(rep['fill_rate'])[1:]}、取引 {t.get('n', 0)}",
          f"- 1 取引あたり平均 {_p(t.get('mean'))}（t = {_p(t.get('t'), False)}）、勝率 {_p(t.get('win'))[1:]}",
          f"- 決済の内訳: {rep['exit_types']}、種別ごとの平均: " + ", ".join(f"{k} {_p(v)}" for k, v in rep["exit_means"].items()),
          f"- 資金推移（全額・複利）: 総損益 {_p(s['total_return'])}、シャープ {_p(s.get('sharpe'), False)}、"
          f"最大ドローダウン {_p(s['max_drawdown'])}、保有時間の割合 {_p(s.get('time_in_market'))[1:]}",
          f"- バイ・アンド・ホールド: 総損益 {_p(rep['buy_and_hold'].get('total_return'))}、シャープ "
          f"{_p(rep['buy_and_hold'].get('sharpe'), False)}\n",
          "## 比較対象（同じ時刻、VAL と無関係な深さの指値。注文ごとの損益）\n",
          "| 回 | 取引 | 平均 | t 値 | 勝率 |\n|---|---|---|---|---|"]
    for i, c in enumerate(rep["control"], 1):
        md.append(f"| {i} | {c.get('n', 0)} | {_p(c.get('mean'))} | {_p(c.get('t'), False)} | {_p(c.get('win'))[1:]} |")
    md.append(f"| ルール A | {t.get('n', 0)} | {_p(t.get('mean'))} | {_p(t.get('t'), False)} | {_p(t.get('win'))[1:]} |")
    md.append("\n## 年ごと\n")
    md.append("| 年 | 注文 | 約定率 | 取引 | 平均 | t 値 | 勝率 | 合計 | バイ・アンド・ホールド |\n|---|---|---|---|---|---|---|---|---|")
    for y, v in rep["years"].items():
        tr = v["trades"]
        md.append(f"| {y} | {v['orders']} | {_p(v['fill_rate'])[1:]} | {tr.get('n', 0)} | {_p(tr.get('mean'))} | "
                  f"{_p(tr.get('t'), False)} | {_p(tr.get('win'))[1:]} | {_p(tr.get('sum'))} | {_p(v['buy_and_hold'])} |")
    return "\n".join(md) + "\n"


def main(args) -> None:
    from .pipeline import load_config

    rep = run_val_rule(load_config(args.config), yaml.safe_load(Path(args.rule).read_text()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    md = render(rep)
    (out / "report.md").write_text(md)
    print(md)
