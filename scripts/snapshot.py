"""1 日 1 回の記録（残高・評価額・目標の重み）と、モニター画面の生成。2026-09-27 オーナー指示。

    python scripts/snapshot.py [--config configs/rebalance.yaml] [--out docs/monitor]

残高は環境変数 BITBANK_API_KEY / BITBANK_API_SECRET で読む（参照系のみ、発注しない）。
オーナー決定: 少額なので金額は伏せない。記録は docs/monitor/daily.jsonl、画面は docs/monitor/index.html。
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from bbdata.client import PublicClient
from bbresearch.monitor import append_jsonl, load_jsonl, render
from bbresearch.rebalance import portfolio_value
from scripts.rebalance import fetch_daily, target_for_today


def build_page(out: Path, target_vol: float) -> None:
    (out / "index.html").write_text(render(load_jsonl(out / "daily.jsonl"), load_jsonl(out / "rebalances.jsonl"), target_vol))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rebalance.yaml")
    ap.add_argument("--out", default="docs/monitor")
    ap.add_argument("--page-only", action="store_true", help="記録せず画面だけ作り直す")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.page_only:
        build_page(out, cfg.get("target_vol") or 0.30)
        return 0
    key, secret = os.environ.get("BITBANK_API_KEY"), os.environ.get("BITBANK_API_SECRET")
    if not key or not secret:
        print("BITBANK_API_KEY / BITBANK_API_SECRET がない")
        return 1
    from bblive.private_client import PrivateClient

    pool: list[str] = cfg["pool"]
    api = PublicClient(min_interval=0.5)
    years = range(cfg.get("history_from_year", 2019), datetime.now(timezone.utc).year + 1)
    close = pd.concat([fetch_daily(api, p, years) for p in pool], axis=1)
    close = close[close.index < pd.Timestamp.now(tz="UTC").normalize()]
    # 目標は運用と同じ規則（直近の月初の相対の重み + 今日のボラとトレンドで JPY の割合）で出す
    tw = target_for_today(cfg, close, load_jsonl(out / "rebalances.jsonl"), "weekly")
    assets = {a["asset"]: float(a.get("onhand_amount") or a.get("free_amount") or 0.0)
              for a in PrivateClient(key, secret).assets()}
    prices = {p: float(api.ticker(p)["last"]) for p in pool}
    holdings = {p.split("_")[0]: assets.get(p.split("_")[0], 0.0) for p in pool}
    jpy = assets.get("jpy", 0.0)
    total, fractions = portfolio_value(holdings, prices, jpy)
    row = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_jpy": round(total, 0), "jpy": round(jpy, 0),
        "values_jpy": {p: round(holdings[p.split("_")[0]] * prices[p], 0) for p in pool if holdings[p.split("_")[0]] > 0},
        "fractions": {p: round(f, 4) for p, f in fractions.items() if f > 0},
        "cash_frac": round(1 - sum(fractions.values()), 4) if total > 0 else 1.0,
        "target_weights": {p: round(w, 4) for p, w in tw["weights"].items()},
        "target_cash": round(tw["cash"], 4), "est_vol": round(tw["est_vol"], 4), "trend": tw.get("trend"),
        "btc_price": prices.get("btc_jpy"),
    }
    append_jsonl(out / "daily.jsonl", row)
    build_page(out, cfg.get("target_vol") or 0.30)
    print(f"記録: {row['date']}、JPY {row['cash_frac']:.0%}（目標 {row['target_cash']:.0%}）、"
          + "、".join(f"{p} {f:.0%}" for p, f in row["fractions"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
