"""深い指値・短い保有・売りも指値の決済条件の探索と、確認期間での1回だけの評価（2026-09-23 オーナー指示）。

    python -m bbresearch maker-search --config configs/research.yaml --grid configs/maker_search.yaml --out reports/maker_search

期間（オーナー決定）:
- 探索: [data.start, search_end)
- 確認: [search_end, ホールドアウト開始)。探索で選んだ1つだけを1回評価する。
  この期間はこれまでの分析（ホライズン・エントリー方法の比較）で見ているので、完全な未使用データではない。
- ホールドアウト（使用済み）は使わない。最終確認は 2026-09-22 以降のデータが 3 か月たまってから行う。

一次シグナル（CUSUM）で全イベントに発注する方式でバックテストする（メタモデルは使わない）。
選択の基準（探索の前に決めたもの）: 探索期間で取引数が min_trades 以上のうち、日次シャープレシオが最大。
すべての組み合わせを試行として実験ログに記録し、DSR の試行数に数える。
"""
from __future__ import annotations

import copy
import json
import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bbdata.download import to_utc

from . import backtest as bt
from .maker_exit import MakerExitParams, label_events_maker
from .model import params_hash
from .pipeline import barrier_params, daily_sr, load_market, read_past_trials
from .search import expand_grid
from .signals import cusum_events, ewm_sigma

log = logging.getLogger(__name__)
_SHARED: dict = {}
PARAM_KEYS = ("delta_sigma", "t_fill_min", "k_up", "k_dn", "n_v", "grace_min")


def _periods(cfg: dict, grid_cfg: dict):
    start, end = to_utc(cfg["data"]["start"]), to_utc(cfg["data"]["end"])
    holdout_start = end - pd.Timedelta(days=int(cfg["split"]["holdout_days"]))
    search_end = to_utc(grid_cfg["search_end"])
    assert start < search_end < holdout_start
    return {"search": (start, search_end), "validate": (search_end, holdout_start)}


def _evaluate(combo: dict, sig: str, period: str) -> dict:
    """1 つの組み合わせ・シグナルを、指定した期間の全イベントに発注する方式で評価する。"""
    m = _SHARED["market"]
    ps, pe = _SHARED["periods"][period]
    events = _SHARED["events"][combo["k_h"]]
    ev = events[(events["signal_type"] == sig) & (events["t0"] >= ps) & (events["t0"] < pe)]
    prm = MakerExitParams(**{k: combo[k] for k in PARAM_KEYS}, bar_minutes=15, s_slip=_SHARED["s_slip"],
                          fill_rule=_SHARED["fill_rule"], maker_fee=_SHARED["fees"][0], taker_fee=_SHARED["fees"][1])
    lab = label_events_maker(ev, m.bars, m.tape, prm, m.tick, _SHARED["sigma"])
    tr = bt.run_backtest(ev, lab, None, bt.Strategy("primary", "all"), _SHARED["account"],
                         pd.Timedelta(minutes=prm.t_fill_min))
    summ = bt.summarize(tr, _SHARED["account"], ps, pe)
    f = lab[lab["filled"] & lab["ret_net"].notna()]
    r = f["ret_net"].to_numpy(dtype=float)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return {
        "n_events": int(len(ev)), "fill_rate": float(lab["filled"].mean()) if len(lab) else None,
        "n_filled": int(len(r)), "mean_ret_net": float(r.mean()) if len(r) else None,
        "t_stat": float(r.mean() / sd * np.sqrt(len(r))) if len(r) > 2 and sd > 0 else None,
        "exit_types": f["exit_type"].value_counts().to_dict(),
        "bt_n_trades": summ["n_trades"], "bt_total_return": summ["total_return"], "bt_sharpe": summ["sharpe"],
        "bt_max_drawdown": summ["max_drawdown"], "summary": summ,
    }


def _task(args):
    combo, sig = args
    return combo, sig, _evaluate(combo, sig, "search")


def run_maker_search(cfg: dict, grid_cfg: dict, workers: int | None = None) -> dict:
    periods = _periods(cfg, grid_cfg)
    market = load_market(cfg)
    base = copy.deepcopy(cfg)
    base["pair_spec"] = market.spec
    prm0 = barrier_params(base, market.spec)
    s = base["signal"]
    grid = grid_cfg["grid"]
    combos = expand_grid(grid)
    sigma = ewm_sigma(market.bars, s["sigma_span"])
    b = base["backtest"]
    _SHARED.clear()
    _SHARED.update(
        market=market, periods=periods, sigma=sigma, s_slip=prm0.s_slip, fill_rule=prm0.fill_rule,
        fees=(prm0.maker_fee, prm0.taker_fee),
        account=bt.Account(initial_capital=float(b["initial_capital"]), max_fraction=float(b["max_fraction"]),
                           amount_digits=int(market.spec["amount_digits"]), min_amount=market.min_amount),
        events={kh: cusum_events(market.bars, base["data"]["pair"], s["sigma_span"], kh)
                for kh in sorted({c["k_h"] for c in combos})},
    )
    signals = grid_cfg.get("signals", ["dip", "breakout"])
    tasks = [(c, sig) for c in combos for sig in signals]
    log.info("組み合わせ %d 通り × %d シグナル", len(combos), len(signals))
    workers = workers or max(1, os.cpu_count() or 1)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
            results = list(ex.map(_task, tasks, chunksize=4))
    else:
        results = [_task(t) for t in tasks]

    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log_path = Path(base.get("experiment_log", "reports/experiments.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    period_key = [str(x) for x in periods["search"]]
    with log_path.open("a") as fh:
        for combo, sig, res in results:
            summ = res.pop("summary")
            th = params_hash({"maker_exit": combo, "signal": sig, "search_period": period_key,
                              "fees": list(_SHARED["fees"]), "s_slip": _SHARED["s_slip"]})
            rows.append({**combo, "signal": sig, **{k: v for k, v in res.items() if k != "exit_types"},
                         "exit_types": json.dumps(res["exit_types"]), "trial_hash": th, "daily_sr": daily_sr(summ),
                         "n_days": summ.get("n_days"), "skew": summ.get("daily_ret_skew"), "kurt": summ.get("daily_ret_kurt")})
            fh.write(json.dumps({"run_at": run_at, "stage": "maker_search", "trial_hash": th,
                                 "code_version": os.environ.get("GITHUB_SHA"), "key": f"maker/{sig}",
                                 "daily_sr": daily_sr(summ), "total_return": summ["total_return"],
                                 "n_trades": summ["n_trades"]}, ensure_ascii=False) + "\n")
    screen = pd.DataFrame(rows)

    # 選択（探索期間のみ）
    min_trades = int(grid_cfg.get("min_trades", 200))
    elig = screen[(screen["bt_n_trades"] >= min_trades) & screen["bt_sharpe"].notna()]
    all_sr = [v for v in read_past_trials(log_path, "", None).values() if v is not None]
    selected = None
    if len(elig):
        best = elig.sort_values("bt_sharpe", ascending=False).iloc[0]
        combo = {k: (best[k].item() if hasattr(best[k], "item") else best[k]) for k in grid}
        sr = best["daily_sr"]
        dsr = (bt.deflated_sharpe(sr, all_sr, int(best["n_days"]), best["skew"], best["kurt"])
               if sr is not None and best["skew"] is not None else None)
        selected = {"combo": combo, "signal": best["signal"], "search": {
            k: (best[k].item() if hasattr(best[k], "item") else best[k])
            for k in ("n_filled", "mean_ret_net", "t_stat", "bt_n_trades", "bt_total_return", "bt_sharpe", "bt_max_drawdown")},
            "search_dsr": dsr, "n_trials_total": len(all_sr)}
        # 確認期間で1回だけ評価する
        val = _evaluate(combo, best["signal"], "validate")
        val_summ = val.pop("summary")
        selected["validate"] = {k: v for k, v in val.items()}
        selected["validate"]["daily_sr"] = daily_sr(val_summ)
        vs, ve = periods["validate"]
        selected["validate_buy_and_hold"] = bt.buy_and_hold(market.bars, vs, ve, prm0.maker_fee, prm0.taker_fee)
    return {"report": {"run_at": run_at, "grid": grid, "periods": {k: [str(a), str(b)] for k, (a, b) in periods.items()},
                       "n_combos": len(combos), "signals": signals, "min_trades": min_trades,
                       "n_positive_mean": int((screen["mean_ret_net"] > 0).sum()), "n_rows": int(len(screen)),
                       "selected": selected},
            "screen": screen}


def _f(x, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x * 100:+.3f}%" if pct else (f"{x:.4g}" if isinstance(x, float) else str(x))


def render(res: dict, top: int = 25) -> str:
    rep, screen = res["report"], res["screen"]
    keys = list(rep["grid"])
    md = [f"# 深い指値・短い保有・売りも指値の決済条件の探索（{rep['run_at']}）\n",
          f"- 探索期間: {rep['periods']['search'][0]} 〜 {rep['periods']['search'][1]}",
          f"- 確認期間: {rep['periods']['validate'][0]} 〜 {rep['periods']['validate'][1]}"
          "（これまでの分析で見ている期間。完全な未使用データではない）",
          f"- 組み合わせ {rep['n_combos']} 通り × {len(rep['signals'])} シグナル。手数料込みの平均損益がプラス: "
          f"{rep['n_positive_mean']} / {rep['n_rows']}",
          "- 一次シグナルの全イベントに発注する方式（メタモデルなし）。選択基準: 探索期間で取引 "
          f"{rep['min_trades']} 件以上のうちシャープレシオ最大\n",
          f"## 探索期間の上位 {top}（シャープレシオ順）\n"]
    cols = keys + ["signal", "n_filled", "fill_rate", "mean_ret_net", "t_stat", "bt_n_trades", "bt_total_return", "bt_sharpe"]
    md.append("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for _, r in screen.sort_values("bt_sharpe", ascending=False).head(top).iterrows():
        md.append("| " + " | ".join(_f(r[c], c in ("mean_ret_net",)) if c not in keys + ["signal"] else str(r[c])
                                    for c in cols) + " |")
    sel = rep["selected"]
    md.append("\n## 選択と確認期間での評価\n")
    if not sel:
        md.append("条件を満たす組み合わせがない")
        return "\n".join(md) + "\n"
    md.append(f"- 選択: `{sel['signal']}` `{sel['combo']}`")
    s, v = sel["search"], sel["validate"]
    md.append(f"- 探索期間: 取引 {s['bt_n_trades']}、平均 {_f(s['mean_ret_net'], True)}（t = {_f(s['t_stat'])}）、"
              f"総損益 {_f(s['bt_total_return'], True)}、シャープ {_f(s['bt_sharpe'])}、DSR {_f(sel['search_dsr'])}"
              f"（試行 {sel['n_trials_total']}）")
    md.append(f"- 確認期間: 取引 {v['bt_n_trades']}、平均 {_f(v['mean_ret_net'], True)}（t = {_f(v['t_stat'])}）、"
              f"総損益 {_f(v['bt_total_return'], True)}、シャープ {_f(v['bt_sharpe'])}、最大ドローダウン "
              f"{_f(v['bt_max_drawdown'], True)}、決済の内訳 {v['exit_types']}")
    bh = sel["validate_buy_and_hold"]
    md.append(f"- 確認期間のバイ・アンド・ホールド: 総損益 {_f(bh.get('total_return'), True)}、シャープ {_f(bh.get('sharpe'))}")
    return "\n".join(md) + "\n"


def main(args) -> None:
    from .pipeline import load_config

    res = run_maker_search(load_config(args.config), yaml.safe_load(Path(args.grid).read_text()), workers=args.workers)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res["screen"].to_csv(out / "screen.csv", index=False)
    (out / "report.json").write_text(json.dumps(res["report"], ensure_ascii=False, indent=2, default=str))
    md = render(res)
    (out / "report.md").write_text(md)
    print(md)
