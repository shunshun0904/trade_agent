"""月初のリバランス計画（ドライラン。発注はしない）。2026-09-27 オーナー指示。

    python scripts/rebalance.py [--config configs/rebalance.yaml]

1. 公式日足（公開 API）から今の目標の重みを計算する（bbresearch.portfolio.current_weights）。
2. 認証付き API で残高を読む（環境変数 BITBANK_API_KEY / BITBANK_API_SECRET。参照系のみ）。
3. 目標に近づける成行注文の一覧を計算し、割合（%）だけを出力する。数量・金額・残高は出力しない（公開リポジトリ）。

発注する処理はこのスクリプトには入っていない（発注を行うコードの追加はオーナーの明示的な承認が要る。SPEC 参照）。
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from bbdata.client import BitbankAPIError, PublicClient
from bbresearch.monitor import append_jsonl
from bbresearch.portfolio import current_weights
from bbresearch.rebalance import PairSpec, describe, plan_orders, portfolio_value


def fetch_daily(api: PublicClient, pair: str, years: range) -> pd.Series:
    rows = []
    for y in years:
        try:
            rows += api.candlestick(pair, "1day", str(y))
        except BitbankAPIError:
            continue
    if not rows:
        return pd.Series(dtype=float, name=pair)
    df = pd.DataFrame(rows, columns=["o", "h", "l", "c", "v", "ts"])
    s = pd.Series(df["c"].astype(float).to_numpy(), index=pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True),
                  name=pair)
    return s[~s.index.duplicated()].sort_index()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rebalance.yaml")
    ap.add_argument("--history", default="docs/monitor/rebalances.jsonl", help="計画を追記する記録（空なら記録しない）")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    pool: list[str] = cfg["pool"]
    api = PublicClient(min_interval=0.5)
    years = range(cfg.get("history_from_year", 2019), datetime.now(timezone.utc).year + 1)
    close = pd.concat([fetch_daily(api, p, years) for p in pool], axis=1)
    close = close[close.index < pd.Timestamp.now(tz="UTC").normalize()]
    tw = current_weights(close, cfg.get("k_max", 5), cfg.get("w_max", 0.4), cfg.get("caps") or {},
                         cfg.get("target_vol"), cfg.get("est_days", 365))
    print(f"目標（{tw['as_of']} までのデータ）: " + "、".join(f"{p} {w:.0%}" for p, w in tw["weights"].items())
          + f"、JPY {tw['cash']:.0%}（推定ボラ {tw['est_vol']:.1%}、縮める前 {tw['est_vol_full']:.1%}）")

    key, secret = os.environ.get("BITBANK_API_KEY"), os.environ.get("BITBANK_API_SECRET")
    if not key or not secret:
        print("残高を読むキーがないので、目標の重みだけを出力した")
        return 0
    from bblive.private_client import PrivateClient

    priv = PrivateClient(key, secret)
    assets = {a["asset"]: float(a.get("free_amount") or 0.0) for a in priv.assets()}
    spec_api = PublicClient(base_url="https://api.bitbank.cc/v1", min_interval=0.5)
    specs = {}
    for p in spec_api.get("/spot/pairs")["pairs"]:
        if p["name"] in pool:
            specs[p["name"]] = PairSpec(p["name"], float(p["unit_amount"]), int(p["amount_digits"]),
                                        float(p["market_max_amount"]) if p.get("market_max_amount") else None)
    prices = {p: float(api.ticker(p)["last"]) for p in pool}
    holdings = {p.split("_")[0]: assets.get(p.split("_")[0], 0.0) for p in pool}
    total, current = portfolio_value(holdings, prices, assets.get("jpy", 0.0))
    if total <= 0:
        print("運用に充てる残高がない")
        return 1
    print("現在: " + "、".join(f"{p} {w:.0%}" for p, w in current.items() if w >= 0.005)
          + f"、JPY {1 - sum(current.values()):.0%}")
    orders = plan_orders(tw["weights"], holdings, prices, assets.get("jpy", 0.0), specs,
                         cfg.get("min_trade_frac", 0.01), cfg.get("buy_buffer", 0.003))
    print("計画（ドライラン、発注しない）:")
    print(describe(orders, tw["cash"]))
    if args.history:
        append_jsonl(Path(args.history), {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "as_of": tw["as_of"],
            "target_weights": {p: round(w, 4) for p, w in tw["weights"].items()}, "target_cash": round(tw["cash"], 4),
            "est_vol": round(tw["est_vol"], 4), "current": {p: round(w, 4) for p, w in current.items() if w > 0},
            "orders": [{"pair": o.pair, "side": o.side, "frac": round(o.frac, 4)} for o in orders], "dry_run": True})
    return 0


if __name__ == "__main__":
    sys.exit(main())
