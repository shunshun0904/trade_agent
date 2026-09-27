"""ポートフォリオの JPY 比率の日次調整（ボラ予測）の前向き検証。2026-09-27 オーナー決定。

    python scripts/vol_daily.py [--config configs/vol_daily.yaml] [--out reports/vol_daily]

比べる戦略（すべて目標ボラ 30%、銘柄と相対の重みは月 1 回、トレンドフィルタあり / なし）:
  月 1 回（現行）、週 1 回（採用済み）、日次（倍率なし）、日次 × 直近 24 本の実現ボラ、日次 × 直近 168 本、日次 × フォレストの予測。
倍率 = 予測した年率ボラ ÷ 直近 365 日の実現ボラ。判断は日付 d より前のデータだけで行う。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bbdata.client import PublicClient
from bbresearch.portfolio import summary, walk_forward, yearly
from bbresearch.volforecast import daily_ratio, hourly_vol_forecast, realized_hourly_vol
from scripts.dist_forecast import fetch_hourly
from scripts.rebalance import fetch_daily


def forecast_quality(sig: pd.Series, df: pd.DataFrame, horizon: int) -> dict:
    """予測 σ̂ と実現（次の horizon 本の対数収益率の絶対値）の対応。QLIKE と log 差の RMSE（値が小さいほど良い）、順位相関。"""
    r = np.log(df["close"]).shift(-horizon) - np.log(df["close"])
    real = pd.Series(np.abs(r.to_numpy()), index=df.index + pd.Timedelta(hours=1))
    d = pd.concat([sig.rename("s"), real.rename("a")], axis=1).dropna()
    d = d[(d["s"] > 0) & (d["a"] > 0)]
    x = (d["a"] ** 2) / (d["s"] ** 2)
    return {"n": int(len(d)), "qlike": float(np.mean(x - np.log(x) - 1)),
            "rmse_log": float(np.sqrt(np.mean((np.log(d["a"] * np.sqrt(np.pi / 2)) - np.log(d["s"])) ** 2))),
            "spearman": float(d["s"].rank().corr(d["a"].rank()))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vol_daily.yaml")
    ap.add_argument("--out", default="reports/vol_daily")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    horizon = int(cfg["horizon_hours"])
    api = PublicClient(min_interval=0.3)
    years = range(cfg["history_from_year"], datetime.now(timezone.utc).year + 1)
    close = pd.concat([fetch_daily(api, p, years) for p in cfg["pool"]], axis=1)
    close = close[close.index < pd.Timestamp.now(tz="UTC").normalize()]
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    hourly = fetch_hourly(api, "btc_jpy", cfg["hourly_start"], today)

    fp = cfg["forest"]
    sig_f = hourly_vol_forecast(hourly, horizon, cfg["start"], int(cfg["refit_months"]), fp["n_estimators"],
                                fp["min_samples_leaf"], fp["max_features"], fp["split_target"])
    sigs = {"forest": sig_f}
    for n in cfg["realized_windows"]:
        sigs[f"rv{n}"] = realized_hourly_vol(hourly, int(n), horizon)
    quality = {k: forecast_quality(v[v.index >= pd.Timestamp(cfg["start"], tz="UTC")], hourly, horizon) for k, v in sigs.items()}
    ratios = {None: None}
    for k, v in sigs.items():
        ratios[k] = daily_ratio(v, close["btc_jpy"], cfg["est_days"], horizon)
    tr = tuple(cfg["trend"]) if cfg.get("trend") else None
    res = walk_forward(close, cfg["start"], cfg["k_max"], cfg["w_max"], (None,), cfg["est_days"], cfg["cost"],
                       caps=cfg["caps"], vol_targets=(cfg["target_vol"],), cash_freq_days=(None, 7, 1),
                       trends=(None, tr) if tr else (None,), vol_ratios=ratios)
    keep = [s for s in res.daily.columns if ("_c1" in s or "_r" not in s)]   # 倍率は日次のときだけ意味がある
    stats = {s: summary(res.daily[s]) for s in keep}
    for s in keep:
        stats[s]["cash_mean"] = float(np.mean([w.get("cash", 0.0) for w in res.weights[s]])) if res.weights.get(s) else 0.0
        stats[s]["rebalances_per_year"] = len(res.weights[s]) / max(1e-9, stats[s]["days"] / 365)
        # 実際に動いた回数（重みが 1% 以上変わった区切り）
        ws = res.weights[s]
        moves = sum(1 for a, b in zip(ws, ws[1:]) if abs(a["cash"] - b["cash"]) >= 0.01 or set(a["weights"]) != set(b["weights"]))
        stats[s]["moves_per_year"] = moves / max(1e-9, stats[s]["days"] / 365)
    yr = {s: yearly(res.daily[s]) for s in keep}

    def label(s: str) -> str:
        if s in ("equal", "btc_jpy"):
            return {"equal": "均等配分", "btc_jpy": "BTC 単独"}[s]
        if s == "minvar":
            return "最小分散（JPY 調整なし）"
        parts = []
        parts.append("月 1 回" if "_c" not in s else ("週 1 回" if "_c7" in s else "日次"))
        if "_rforest" in s:
            parts.append("フォレストのボラ予測")
        elif "_rrv" in s:
            parts.append("直近 " + s.split("_rrv")[1].split("_")[0] + " 本の実現ボラ")
        parts.append("トレンドフィルタあり" if "_t200" in s else "トレンドフィルタなし")
        return "、".join(parts)

    md = [f"# JPY 比率の日次調整（ボラ予測）の前向き検証（{cfg['start']} 〜 {close.index[-1].date()}、目標ボラ {cfg['target_vol']:.0%}）\n",
          f"- 銘柄と相対の重みは月 1 回（直前 {cfg['est_days']} 日、{cfg['k_max']} 銘柄以内、BTC {cfg['caps']['btc_jpy']:.0%} まで）。JPY の割合だけ月 1 回 / 週 1 回 / 日次で見直す。売買費用は回転率 × {cfg['cost']:.2%}",
          f"- 倍率 = 予測した年率ボラ ÷ 直近 {cfg['est_days']} 日の実現ボラ。予測は BTC の 1 時間足 373 指標 → 分位点回帰フォレスト（{fp['split_target']} 分割、葉 {fp['min_samples_leaf']}、木 {fp['n_estimators']}、{cfg['refit_months']} か月ごとに学習し直し）の予測分布の標準偏差（{horizon} 本）。基準は直近 n 本の実現ボラ",
          "- 判断は日付より前のデータだけ（フォレストの学習は目的変数が確定した行だけ）\n",
          "## ボラ予測の精度（1 時間足ごと、次の 24 本の |収益率| に対して。値が小さいほど良い。順位相関は大きいほど良い）\n",
          "| 予測 | 本数 | QLIKE | log 差の RMSE | 順位相関 |\n|---|---|---|---|---|"]
    for k, q in quality.items():
        name = "フォレスト" if k == "forest" else f"直近 {k[2:]} 本の実現ボラ"
        md.append(f"| {name} | {q['n']:,} | {q['qlike']:.4f} | {q['rmse_log']:.4f} | {q['spearman']:.3f} |")
    md.append("\n## 前向き検証\n")
    md.append("| 戦略 | 年率ボラ | 年率リターン | シャープ | 最大ドローダウン | 累積 | JPY 比率の平均 | 見直し / 年 | 1% 以上動いた回数 / 年 |\n|---|---|---|---|---|---|---|---|---|")
    order = sorted(keep, key=lambda s: (s in ("equal", "btc_jpy"), "_t200" in s, "_c1" not in s, "_c7" not in s, s))
    for s in order:
        st = stats[s]
        md.append(f"| {label(s)} | {st['ann_vol']:.1%} | {st['ann_return']:.1%} | {st['sharpe']:.2f} | {st['max_drawdown']:.1%} | "
                  f"{st['total']:+.0%} | {st['cash_mean']:.0%} | {st['rebalances_per_year']:.0f} | {st['moves_per_year']:.0f} |")
    years_all = sorted(set().union(*[set(v.index) for v in yr.values()]))
    for title, col, fmt in [("年ごとの年率ボラ（目標 30% にどれだけ近いか）", "ann_vol", "{:.1%}"), ("年ごとのリターン", "return", "{:+.1%}"),
                            ("年ごとの最大ドローダウン", "max_drawdown", "{:.1%}")]:
        md.append(f"\n## {title}\n")
        md.append("| 年 | " + " | ".join(label(s) for s in order) + " |")
        md.append("|---|" + "---|" * len(order))
        for y in years_all:
            md.append(f"| {y} | " + " | ".join(fmt.format(yr[s].loc[y, col]) if y in yr[s].index else "-" for s in order) + " |")
    # 目標との乖離: 年率ボラの年ごとの目標からの絶対差の平均
    md.append("\n## 目標ボラからのぶれ（年ごとの実現ボラと目標の差の絶対値の平均。小さいほど狙いどおり）\n")
    md.append("| 戦略 | ぶれ |\n|---|---|")
    for s in order:
        if s in ("equal", "btc_jpy", "minvar"):
            continue
        dev = float(np.mean([abs(v - cfg["target_vol"]) for v in yr[s]["ann_vol"].dropna()]))
        md.append(f"| {label(s)} | {dev:.1%} |")
    md.append("\n注意: 日次の調整は見直し回数が年 365 回になる。費用は回転率にしか掛けていないので、残高が小さいと最小注文単位で動かせない日が多い。")
    text = "\n".join(md) + "\n"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(text)
    (out / "metrics.json").write_text(json.dumps({"stats": stats, "quality": quality,
                                                  "yearly": {s: v.reset_index().to_dict("records") for s, v in yr.items()}}, ensure_ascii=False, indent=1))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
