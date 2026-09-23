"""研究パイプライン: 足 → CUSUM → 約定考慮トリプルバリア → 特徴量 → メタモデル → バックテスト。

    python -m bbresearch run --config configs/research.yaml

期間の分け方:
- 開発期間 [start, holdout_start): Purged K-fold の out-of-fold 予測でバックテストする
- ホールドアウト [holdout_start, end): 開発期間の全データで学習したモデルで予測する。
  モデル選択には一切使わない（レポートに並べるだけ）
開発期間の学習サンプルは t_x < holdout_start のものに限る（境界をまたぐラベルのリーク防止）。
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bbdata.bars import build_bars
from bbdata.client import PublicClient
from bbdata.download import to_utc

from . import backtest as bt
from .features import bar_features, event_features
from .labeling import BarrierParams, TradeTape, label_events
from .model import cross_validate, fit_final, metrics, params_hash, predict_by_fold, predict_proba, save_model
from .signals import cusum_events, ewm_sigma
from .weights import average_uniqueness

log = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def fetch_pair_spec(pair: str) -> dict:
    api = PublicClient(base_url="https://api.bitbank.cc/v1", min_interval=0.5)
    spec = next(p for p in api.get("/spot/pairs")["pairs"] if p["name"] == pair)
    status = next((s for s in api.get("/spot/status")["statuses"] if s["pair"] == pair), {})
    return {**spec, "status_min_amount": status.get("min_amount")}


def random_events(bars: pd.DataFrame, sigma: pd.Series, n: int, start, end, rng: np.random.Generator) -> pd.DataFrame:
    """同じ件数のランダムなエントリー時刻（足の終了時刻）を作る。"""
    step = bars.index[1] - bars.index[0]
    cand = sigma[(sigma.index >= start) & (sigma.index + step < end)].dropna().index
    if len(cand) == 0 or n <= 0:
        return pd.DataFrame(columns=["event_id", "t0", "signal_type", "s_value", "h", "sigma"])
    pick = np.sort(rng.choice(len(cand), size=min(n, len(cand)), replace=False))
    bs = cand[pick]
    return pd.DataFrame({
        "event_id": [f"rand-{t:%Y%m%dT%H%M}" for t in bs], "t0": bs + step, "signal_type": "random",
        "s_value": np.nan, "h": np.nan, "sigma": sigma.loc[bs].to_numpy(),
    })


def run(cfg: dict, out_dir: Path) -> dict:
    d, s, lab_cfg, m_cfg, b_cfg = cfg["data"], cfg["signal"], cfg["label"], cfg["model"], cfg["backtest"]
    pair = d["pair"]
    start, end = to_utc(d["start"]), to_utc(d["end"])
    holdout_start = end - pd.Timedelta(days=int(cfg["split"]["holdout_days"]))
    root = d.get("root", "data")
    seed = int(cfg.get("seed", 0))

    spec = cfg.get("pair_spec") or fetch_pair_spec(pair)
    tick = 10.0 ** -int(spec["price_digits"])
    min_amount = max(float(spec.get("unit_amount") or 0), float(spec.get("status_min_amount") or 0))
    # 手数料率は設定で上書きしない限り /spot/pairs の値を使う（ハードコードしない。SPEC §3.5）
    lab = dict(lab_cfg)
    if lab.get("maker_fee") is None:
        lab["maker_fee"] = float(spec["maker_fee_rate_quote"])
    if lab.get("taker_fee") is None:
        lab["taker_fee"] = float(spec["taker_fee_rate_quote"])
    prm = BarrierParams(**lab)

    log.info("足を構築 %s %s〜%s", pair, start, end)
    bars = build_bars(root, pair, start, end, "15min", large_trade_amount=d.get("large_trade_amount"))
    sigma = ewm_sigma(bars, s["sigma_span"])
    events = cusum_events(bars, pair, s["sigma_span"], s["k_h"])
    log.info("イベント %d 件", len(events))

    tape = TradeTape.load(root, pair, start, end)
    labels = label_events(events, bars, tape, prm, tick, sigma)
    X_all = event_features(events, bar_features(bars, s["sigma_span"]))

    ev = events.merge(labels, on="event_id")
    ev["period"] = np.where(ev["t0"] < holdout_start, "dev", "holdout")
    t_fill = pd.Timedelta(minutes=prm.t_fill_min)
    embargo = pd.Timedelta(minutes=prm.bar_minutes * int(m_cfg.get("embargo_bars", prm.n_v)))
    account = bt.Account(
        initial_capital=float(b_cfg["initial_capital"]), max_fraction=float(b_cfg["max_fraction"]),
        amount_digits=int(spec["amount_digits"]), min_amount=min_amount,
    )
    strategies = [bt.Strategy("primary", "all")]
    strategies += [bt.Strategy(f"threshold_{th}", "threshold", theta=float(th)) for th in b_cfg["thetas"]]
    strategies += [bt.Strategy("meta", "meta", size_step=float(b_cfg.get("size_step", 0.0)))]
    periods = {"dev": (start, holdout_start), "holdout": (holdout_start, end)}

    report: dict = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg, "config_hash": params_hash(cfg), "pair_spec": spec,
        "fees_used": {"maker": prm.maker_fee, "taker": prm.taker_fee},
        "n_bars": len(bars), "n_empty_bars": int(bars["is_empty"].sum()), "n_trades_raw": len(tape),
        "events": {}, "cv": {}, "backtest": {}, "baselines": {},
    }
    trials: list[dict] = []
    rng = np.random.default_rng(seed)

    for sig in ("dip", "breakout"):
        e = ev[ev["signal_type"] == sig].sort_values("t0").reset_index(drop=True)
        report["events"][sig] = {
            p: {
                "n_events": int((e["period"] == p).sum()),
                "fill_rate": float(e.loc[e["period"] == p, "filled"].mean()) if (e["period"] == p).any() else None,
                "exit_types": e.loc[(e["period"] == p) & e["filled"], "exit_type"].value_counts().to_dict(),
                "mean_ret_net_filled": float(e.loc[(e["period"] == p) & e["filled"], "ret_net"].mean())
                if ((e["period"] == p) & e["filled"]).any() else None,
                "y_rate": float(e.loc[(e["period"] == p) & e["filled"], "y"].mean())
                if ((e["period"] == p) & e["filled"]).any() else None,
            }
            for p in periods
        }
        train = e[(e["period"] == "dev") & e["filled"] & e["t_x"].notna() & (e["t_x"] < holdout_start)]
        if len(train) < int(m_cfg.get("min_train", 100)) or train["y"].nunique() < 2:
            log.warning("%s: 学習サンプル不足（%d 件）。モデルを学習しない", sig, len(train))
            report["cv"][sig] = {"skipped": f"学習サンプル {len(train)} 件"}
            probas = {}
        else:
            X = X_all.loc[train["event_id"]]
            y = train.set_index("event_id")["y"].astype(int)
            w = average_uniqueness(train["t_f"], train["t_x"]).set_axis(X.index) if m_cfg.get("use_weights", True) else None
            t0s, txs = train["t0"].reset_index(drop=True), train["t_x"].reset_index(drop=True)
            probas = {}
            report["cv"][sig] = {"n_train": int(len(train)), "base_rate": float(y.mean())}
            for kind in m_cfg["kinds"]:
                params = (m_cfg.get("params") or {}).get(kind)
                res = cross_validate(kind, X, y, w, t0s, txs, int(m_cfg["n_splits"]), embargo, seed, params)
                # 開発期間: ラベルのあるイベントは OOF、ないイベントは該当 fold のモデル
                dev_e = e[e["period"] == "dev"]
                p_dev = predict_by_fold(res, X_all.loc[dev_e["event_id"]], dev_e["t0"])
                p_dev.update(res.oof)
                final = fit_final(kind, X, y, w, seed, params)
                ho_e = e[e["period"] == "holdout"]
                p_ho = pd.Series(predict_proba(final, X_all.loc[ho_e["event_id"]]) if len(ho_e) else [],
                                 index=ho_e["event_id"].to_numpy(), dtype="float64")
                probas[kind] = pd.concat([p_dev, p_ho])
                ho_lab = ho_e[ho_e["filled"] & ho_e["y"].notna()]
                report["cv"][sig][kind] = {
                    "oof": res.overall, "folds": res.fold_metrics,
                    "holdout": metrics(ho_lab["y"].astype(int).to_numpy(), probas[kind].loc[ho_lab["event_id"]].to_numpy())
                    if len(ho_lab) else {},
                }
                if hasattr(final, "feature_importances_"):
                    report["cv"][sig][kind]["feature_importance"] = dict(
                        sorted(zip(X.columns, map(int, final.feature_importances_)), key=lambda kv: -kv[1]))
                save_model(final, out_dir / "models" / f"{pair}_{sig}_{kind}", {
                    "features": list(X.columns), "params_hash": params_hash({"cfg": cfg, "kind": kind}),
                    "train_period": [str(start), str(holdout_start)], "signal_type": sig, "kind": kind,
                    "n_train": len(train), "data_version": {"n_raw_trades": len(tape), "end": str(end)},
                })

        for pname, (ps, pe) in periods.items():
            e_p = e[e["period"] == pname]
            lab_p = e_p[labels.columns]
            ev_p = e_p[events.columns]
            for strat in strategies:
                kinds = [None] if strat.mode == "all" else list(probas)
                for kind in kinds:
                    tr = bt.run_backtest(ev_p, lab_p, None if kind is None else probas[kind], strat, account, t_fill)
                    summ = bt.summarize(tr, account, ps, pe)
                    key = f"{sig}/{strat.name}" + (f"/{kind}" if kind else "")
                    report["backtest"].setdefault(pname, {})[key] = summ
                    if pname == "dev":
                        trials.append({"key": key, **summ})
            # ランダムエントリー: 一次シグナルのみの方式の発注回数と同じ数の候補時刻を、期間内の足の終了時刻から
            # 一様に選ぶ。待機中・保有中に当たった候補は見送るので、実際の発注回数はそれ以下になる
            n_orders = report["backtest"][pname][f"{sig}/primary"]["n_orders"]
            rand_ret, rand_orders = [], []
            for _ in range(int(b_cfg.get("n_random", 20))):
                rev = random_events(bars, sigma, n_orders, ps, pe, rng)
                rlab = label_events(rev, bars, tape, prm, tick, sigma)
                tr = bt.run_backtest(rev, rlab, None, bt.Strategy("random", "all"), account, t_fill)
                rand_ret.append(bt.summarize(tr, account, ps, pe)["total_return"])
                rand_orders.append(len(tr))
            report["baselines"].setdefault(pname, {})[f"{sig}/random"] = {
                "n_candidates": n_orders,
                "n_orders_mean": float(np.mean(rand_orders)) if rand_orders else None,
                "total_return_mean": float(np.mean(rand_ret)) if rand_ret else None,
                "total_return_p05": float(np.percentile(rand_ret, 5)) if rand_ret else None,
                "total_return_p95": float(np.percentile(rand_ret, 95)) if rand_ret else None,
            }

    for pname, (ps, pe) in periods.items():
        report["baselines"].setdefault(pname, {})["buy_and_hold"] = bt.buy_and_hold(
            bars, ps, pe, prm.maker_fee, prm.taker_fee)

    # Deflated Sharpe Ratio: 今回の試行と、実験ログに記録された過去の試行を試行数に数える。
    # 試行は「戦略を決める設定（trial_hash）× 戦略（key）」で数え、同じ組み合わせの再実行は1試行とする
    trial_hash = params_hash({
        **{k: cfg[k] for k in ("data", "split", "signal", "label", "model")},
        "backtest": {k: v for k, v in b_cfg.items() if k != "n_random"},
        "fees": report["fees_used"],
    })
    report["trial_hash"] = trial_hash
    log_path = Path(cfg.get("experiment_log", "reports/experiments.jsonl"))
    past: dict[tuple, float | None] = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            p = json.loads(line)
            th = p.get("trial_hash")
            if th is None:  # trial_hash を記録する前の行は config_hash で同じ設定かを判断する
                if p.get("config_hash") == report["config_hash"]:
                    continue
                th = "config:" + str(p.get("config_hash"))
            if th == trial_hash:  # 今回と同じ設定の試行は今回の結果で数える
                continue
            past[(th, p.get("key"))] = p.get("daily_sr")
    sr_d = lambda t: t["sharpe"] / np.sqrt(365) if t.get("sharpe") is not None else None  # noqa: E731
    all_sr = [x for x in [sr_d(t) for t in trials] + list(past.values()) if x is not None]
    report["n_trials_total"] = len(all_sr)
    for t in trials:
        sr = sr_d(t)
        if sr is None or t.get("daily_ret_skew") is None:
            continue
        report["backtest"]["dev"][t["key"]]["dsr"] = bt.deflated_sharpe(
            sr, all_sr, t["n_days"], t["daily_ret_skew"], t["daily_ret_kurt"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        for t in trials:
            f.write(json.dumps({"run_at": report["run_at"], "config_hash": report["config_hash"],
                                "trial_hash": trial_hash, "code_version": os.environ.get("GITHUB_SHA"),
                                "key": t["key"], "daily_sr": sr_d(t), "total_return": t["total_return"],
                                "n_trades": t["n_trades"]}, ensure_ascii=False) + "\n")
    return report


def render(rep: dict) -> str:
    md = [f"# 研究パイプライン結果（{rep['run_at']}）\n"]
    c = rep["config"]
    md.append(f"- pair: `{c['data']['pair']}`、期間 {c['data']['start']} 〜 {c['data']['end']}（ホールドアウト {c['split']['holdout_days']} 日）")
    md.append(f"- 足 {rep['n_bars']} 本（約定なし {rep['n_empty_bars']}）、約定 {rep['n_trades_raw']} 件、config_hash `{rep['config_hash']}`")
    md.append(f"- 手数料率: メイカー {rep['fees_used']['maker']}、テイカー {rep['fees_used']['taker']}")
    md.append(f"- 試行数（DSR 用、過去の実験ログを含む）: {rep['n_trials_total']}\n")
    md.append("## イベントとラベル\n")
    md.append("| signal | period | events | fill_rate | y_rate | mean_ret_net | exit_types |\n|---|---|---|---|---|---|---|")
    for sig, per in rep["events"].items():
        for p, v in per.items():
            md.append(f"| {sig} | {p} | {v['n_events']} | {_f(v['fill_rate'])} | {_f(v['y_rate'])} | {_f(v['mean_ret_net_filled'], 5)} | {v['exit_types']} |")
    md.append("\n## メタモデル（OOF / ホールドアウト）\n")
    md.append("| signal | model | n_train | base_rate | OOF logloss | OOF AUC | OOF brier | HO n | HO AUC | HO logloss |\n|---|---|---|---|---|---|---|---|---|---|")
    for sig, v in rep["cv"].items():
        if "skipped" in v:
            md.append(f"| {sig} | - | {v['skipped']} | | | | | | | |")
            continue
        for kind in c["model"]["kinds"]:
            o, h = v[kind]["oof"], v[kind]["holdout"]
            md.append(f"| {sig} | {kind} | {v['n_train']} | {_f(v['base_rate'])} | {_f(o.get('log_loss'))} | {_f(o.get('auc'))} | {_f(o.get('brier'))} | {h.get('n', 0)} | {_f(h.get('auc'))} | {_f(h.get('log_loss'))} |")
    for pname, rows in rep["backtest"].items():
        md.append(f"\n## バックテスト（{pname}）\n")
        md.append("| strategy | orders | trades | fill_rate | total_return | max_dd | sharpe | win_rate | avg_ret_net | time_in_mkt | maker_fee | taker_fee | DSR |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k, v in rows.items():
            md.append(f"| {k} | {v['n_orders']} | {v['n_trades']} | {_f(v.get('fill_rate'))} | {_f(v['total_return'], 4)} | {_f(v['max_drawdown'], 4)} | {_f(v['sharpe'])} | {_f(v.get('win_rate'))} | {_f(v.get('avg_ret_net'), 5)} | {_f(v.get('time_in_market'))} | {_f(v.get('maker_fee_total'), 0)} | {_f(v.get('taker_fee_total'), 0)} | {_f(v.get('dsr'))} |")
        md.append("\nexit_type 別:\n")
        for k, v in rows.items():
            if v.get("by_exit_type"):
                md.append(f"- {k}: " + ", ".join(f"{et} n={x['n']} pnl={x['pnl']:.0f} avg_ret={x['avg_ret_net']:.5f}" for et, x in v["by_exit_type"].items()))
        md.append("\n比較対象:\n")
        for k, v in rep["baselines"].get(pname, {}).items():
            md.append(f"- {k}: " + ", ".join(f"{a}={_f(b, 4)}" for a, b in v.items()))
    return "\n".join(md) + "\n"


def _f(x, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    if isinstance(x, (int, np.integer)):
        return str(x)
    return f"{x:.{nd}f}"
