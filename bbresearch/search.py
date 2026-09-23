"""SPEC §7 の探索範囲でのパラメータ探索（開発期間のみ。ホールドアウトは一切評価しない）。

    python -m bbresearch search --config configs/research.yaml --grid configs/search.yaml --out reports/search

1. 一次スクリーニング: グリッドの組み合わせごとに、開発期間のイベントとラベルを作り直し、
   一次シグナルのみの方式（全イベントに発注）でバックテストする。約定した取引の ret_net の t 値で順位を付ける。
2. 上位の組み合わせだけ、メタモデルを含むパイプライン（開発期間のみ）で評価する。
3. あらかじめ決めた基準（開発期間のシャープレシオ最大、取引数の下限つき）で1つ選び、
   selected.yaml に書き出す。ホールドアウトでの確認は、その設定で研究パイプラインを実行して行う。

すべての組み合わせを試行として実験ログに記録し、Deflated Sharpe Ratio の試行数に数える。
"""
from __future__ import annotations

import copy
import itertools
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
from .labeling import label_events
from .model import params_hash
from .pipeline import Market, barrier_params, daily_sr, load_market, read_past_trials, run, trial_hash
from .signals import cusum_events, ewm_sigma

log = logging.getLogger(__name__)

# fork したワーカーが参照するデータ（コピーオンライトで共有する）
_SHARED: dict = {}


def expand_grid(grid: dict[str, list]) -> list[dict[str, object]]:
    """{"label.k_up": [1, 2], ...} → 組み合わせのリスト（キーは "節.名前"）。"""
    keys = sorted(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def apply_combo(cfg: dict, combo: dict[str, object]) -> dict:
    out = copy.deepcopy(cfg)
    for key, val in combo.items():
        section, name = key.split(".", 1)
        out[section][name] = val
    return out


def _dev_window(cfg: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = to_utc(cfg["data"]["start"]), to_utc(cfg["data"]["end"])
    return start, end - pd.Timedelta(days=int(cfg["split"]["holdout_days"]))


def screen_one(cfg: dict) -> list[dict]:
    """1つの組み合わせを、開発期間の一次シグナルのみの方式で評価する（dip / breakout の2行）。"""
    market: Market = _SHARED["market"]
    s = cfg["signal"]
    start, holdout_start = _dev_window(cfg)
    prm = barrier_params(cfg, market.spec)
    sigma = _SHARED["sigma"][s["sigma_span"]]
    events = _SHARED["events"][(s["sigma_span"], s["k_h"])]
    labels = label_events(events, market.bars, market.tape, prm, market.tick, sigma)
    b = cfg["backtest"]
    account = bt.Account(initial_capital=float(b["initial_capital"]), max_fraction=float(b["max_fraction"]),
                         amount_digits=int(market.spec["amount_digits"]), min_amount=market.min_amount)
    rows = []
    for sig in ("dip", "breakout"):
        ev = events[events["signal_type"] == sig]
        lab = labels[labels["event_id"].isin(ev["event_id"])]
        tr = bt.run_backtest(ev, lab, None, bt.Strategy("primary", "all"), account,
                             pd.Timedelta(minutes=prm.t_fill_min))
        summ = bt.summarize(tr, account, start, holdout_start)
        filled = lab[lab["filled"] & lab["ret_net"].notna()]
        r = filled["ret_net"].to_numpy(dtype=float)
        t_stat = float(r.mean() / r.std(ddof=1) * np.sqrt(len(r))) if len(r) > 2 and r.std(ddof=1) > 0 else None
        rows.append({
            "signal": sig, "n_events": int(len(ev)), "fill_rate": float(lab["filled"].mean()) if len(lab) else None,
            "n_filled": int(len(r)), "mean_ret_net": float(r.mean()) if len(r) else None, "t_stat": t_stat,
            "win_rate": float((r > 0).mean()) if len(r) else None,
            "bt_n_trades": summ["n_trades"], "bt_total_return": summ["total_return"], "bt_sharpe": summ["sharpe"],
            "summary": summ,
        })
    return rows


def _screen_task(args: tuple[dict, dict]) -> tuple[dict, list[dict]]:
    combo, cfg = args
    return combo, screen_one(cfg)


def run_search(cfg: dict, search_cfg: dict, out_dir: Path, workers: int | None = None) -> dict:
    grid = search_cfg["grid"]
    combos = expand_grid(grid)
    base = copy.deepcopy(cfg)
    start, holdout_start = _dev_window(base)
    log.info("組み合わせ %d 通り", len(combos))

    market = load_market(base)
    base["pair_spec"] = market.spec  # ワーカーやステージ2で API を呼び直さない
    sigma_spans = sorted({apply_combo(base, c)["signal"]["sigma_span"] for c in combos})
    khs = sorted({apply_combo(base, c)["signal"]["k_h"] for c in combos})
    _SHARED.clear()
    _SHARED["market"] = market
    _SHARED["sigma"] = {sp: ewm_sigma(market.bars, sp) for sp in sigma_spans}
    _SHARED["events"] = {}
    for sp in sigma_spans:
        for kh in khs:
            ev = cusum_events(market.bars, base["data"]["pair"], sp, kh)
            _SHARED["events"][(sp, kh)] = ev[ev["t0"] < holdout_start].reset_index(drop=True)

    # ---- ステージ1: 一次スクリーニング
    tasks = [(c, apply_combo(base, c)) for c in combos]
    workers = workers or max(1, (os.cpu_count() or 1))
    results: list[tuple[dict, list[dict]]] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
            for i, res in enumerate(ex.map(_screen_task, tasks, chunksize=4), 1):
                results.append(res)
                if i % 50 == 0:
                    log.info("スクリーニング %d / %d", i, len(tasks))
    else:
        results = [_screen_task(t) for t in tasks]

    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log_path = Path(base.get("experiment_log", "reports/experiments.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    screen_rows = []
    with log_path.open("a") as f:
        for combo, rows in results:
            ccfg = apply_combo(base, combo)
            th = trial_hash(ccfg, barrier_params(ccfg, market.spec))
            for row in rows:
                summ = row.pop("summary")
                screen_rows.append({**combo, **row, "trial_hash": th,
                                    "daily_sr": daily_sr(summ), "n_days": summ.get("n_days"),
                                    "skew": summ.get("daily_ret_skew"), "kurt": summ.get("daily_ret_kurt")})
                f.write(json.dumps({"run_at": run_at, "stage": "screen", "trial_hash": th,
                                    "code_version": os.environ.get("GITHUB_SHA"), "key": f"{row['signal']}/primary",
                                    "daily_sr": daily_sr(summ), "total_return": summ["total_return"],
                                    "n_trades": summ["n_trades"]}, ensure_ascii=False) + "\n")
    screen = pd.DataFrame(screen_rows)

    # ---- ステージ2: 上位の組み合わせをメタモデル込みで評価（開発期間のみ）
    s1 = search_cfg.get("stage1", {})
    s2 = search_cfg.get("stage2", {})
    eligible = screen[(screen["n_filled"] >= int(s1.get("min_filled", 200))) & screen["t_stat"].notna()]
    top = eligible.sort_values("t_stat", ascending=False).head(int(s2.get("top_n", 5)))
    grid_keys = sorted(grid)
    stage2 = []
    seen = set()
    for _, row in top.iterrows():
        combo = {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in grid_keys}
        key = json.dumps(combo, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        ccfg = apply_combo(base, combo)
        ccfg["backtest"]["n_random"] = int(s2.get("n_random", 0))
        rep = run(ccfg, out_dir / "stage2", market=market, evaluate_holdout=False)
        stage2.append({"combo": combo, "trial_hash": rep["trial_hash"], "backtest": rep["backtest"]["dev"],
                       "cv": {sig: {k: v.get("oof", {}) if isinstance(v, dict) else v
                                    for k, v in c.items()} for sig, c in rep["cv"].items()}})

    # ---- DSR: この探索と実験ログのすべての試行を試行数に数えて計算し直す
    past = read_past_trials(log_path, current_trial_hash="", current_config_hash=None)
    all_sr = [v for v in past.values() if v is not None]
    n_trials = len(all_sr)
    candidates = []
    min_trades = int(s2.get("min_trades", 100))
    for item in stage2:
        for key, summ in item["backtest"].items():
            sr = daily_sr(summ)
            dsr = None
            if sr is not None and summ.get("daily_ret_skew") is not None:
                dsr = bt.deflated_sharpe(sr, all_sr, summ["n_days"], summ["daily_ret_skew"], summ["daily_ret_kurt"])
            summ["dsr"] = dsr
            if summ.get("n_trades", 0) >= min_trades and summ.get("sharpe") is not None:
                candidates.append({"combo": item["combo"], "strategy": key, "sharpe": summ["sharpe"],
                                   "total_return": summ["total_return"], "n_trades": summ["n_trades"], "dsr": dsr})
    selected = max(candidates, key=lambda c: c["sharpe"]) if candidates else None

    out_dir.mkdir(parents=True, exist_ok=True)
    screen.to_csv(out_dir / "screen.csv", index=False)
    report = {
        "run_at": run_at, "n_combos": len(combos), "grid": grid, "search_config": search_cfg,
        "dev_period": [str(start), str(holdout_start)], "n_trials_total": n_trials,
        "stage2": stage2, "selected": selected, "search_hash": params_hash({"cfg": cfg, "search": search_cfg}),
    }
    if selected:
        sel_cfg = apply_combo(cfg, selected["combo"])
        (out_dir / "selected.yaml").write_text(yaml.safe_dump(
            {"selected_at": run_at, "strategy": selected["strategy"], "combo": selected["combo"],
             "dev_sharpe": selected["sharpe"], "dev_total_return": selected["total_return"],
             "dev_n_trades": selected["n_trades"], "dev_dsr": selected["dsr"],
             "signal": sel_cfg["signal"], "label": sel_cfg["label"]}, allow_unicode=True, sort_keys=False))
    return {"report": report, "screen": screen}


def render(res: dict, top_rows: int = 30) -> str:
    rep, screen = res["report"], res["screen"]
    grid_keys = sorted(rep["grid"])
    md = [f"# パラメータ探索結果（{rep['run_at']}）\n"]
    md.append(f"- 開発期間のみ: {rep['dev_period'][0]} 〜 {rep['dev_period'][1]}（ホールドアウトは評価していない）")
    md.append(f"- 組み合わせ {rep['n_combos']} 通り × 2 シグナル。試行数（DSR 用、実験ログ全体）: {rep['n_trials_total']}")
    md.append("- グリッド: " + ", ".join(f"`{k}` = {v}" for k, v in rep["grid"].items()) + "\n")
    md.append(f"## ステージ1: 一次シグナルのみ（t 値の上位 {top_rows}）\n")
    cols = grid_keys + ["signal", "n_filled", "fill_rate", "win_rate", "mean_ret_net", "t_stat", "bt_total_return", "bt_sharpe"]
    md.append("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for _, r in screen.sort_values("t_stat", ascending=False).head(top_rows).iterrows():
        md.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    pos = screen[screen["mean_ret_net"] > 0]
    md.append(f"\n手数料込みの平均 ret_net がプラスの組み合わせ: {len(pos)} / {len(screen)}\n")
    md.append("## ステージ2: メタモデル込み（開発期間）\n")
    md.append("| 組み合わせ | strategy | trades | total_return | sharpe | win_rate | DSR |\n|---|---|---|---|---|---|---|")
    for item in rep["stage2"]:
        combo = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in item["combo"].items())
        for key, s in item["backtest"].items():
            md.append(f"| {combo} | {key} | {s['n_trades']} | {_fmt(s['total_return'])} | {_fmt(s.get('sharpe'))} | "
                      f"{_fmt(s.get('win_rate'))} | {_fmt(s.get('dsr'))} |")
    md.append("\n## 選択（開発期間のシャープレシオ最大、取引数の下限つき）\n")
    sel = rep["selected"]
    if sel:
        md.append(f"- 組み合わせ: `{sel['combo']}`、strategy: `{sel['strategy']}`")
        md.append(f"- 開発期間: sharpe {_fmt(sel['sharpe'])}、total_return {_fmt(sel['total_return'])}、"
                  f"取引 {sel['n_trades']}、DSR {_fmt(sel['dsr'])}")
        md.append("- ホールドアウトでの確認はまだ行っていない（selected.yaml の設定で研究パイプラインを実行する）")
    else:
        md.append("- 条件を満たす候補なし")
    return "\n".join(md) + "\n"


def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{x:.4g}"
    return str(x)
