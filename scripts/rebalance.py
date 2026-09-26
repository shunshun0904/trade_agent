"""リバランス計画（ドライラン。発注はしない）。2026-09-27 オーナー指示。

    python scripts/rebalance.py [--config configs/rebalance.yaml] [--mode auto|monthly|weekly]

1. 公式日足（公開 API）から今の目標の重みを計算する。
   - monthly（毎月 1 日）: 銘柄と相対の重みを選び直す（bbresearch.portfolio.current_weights）。
   - weekly（月曜）: 直近の monthly の相対の重みをそのまま使い、JPY の割合（目標ボラとトレンドフィルタ）だけ見直す
     （bbresearch.portfolio.scale_weights）。monthly の記録がなければ monthly と同じ計算をする。
   - auto: UTC の日付が 1 日なら monthly、それ以外は weekly。
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
from bbresearch.monitor import append_jsonl, load_jsonl
from bbresearch.portfolio import current_weights, scale_weights
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


def last_monthly(history: list[dict]) -> dict | None:
    """記録の中で最後の monthly（kind がない古い記録も monthly とみなす）。"""
    for r in reversed(history):
        if r.get("kind", "monthly") == "monthly":
            return r
    return None


def target_for_today(cfg: dict, close: pd.DataFrame, history: list[dict], mode: str = "auto") -> dict:
    """今日の目標の重み。mode は auto / monthly / weekly。戻り値に kind（実際に使った方式）を入れる。"""
    if mode == "auto":
        mode = "monthly" if datetime.now(timezone.utc).day == 1 else "weekly"
    common = dict(target_vol=cfg.get("target_vol"), est_days=cfg.get("est_days", 365),
                  trend_days=cfg.get("trend_days"), trend_scale=cfg.get("trend_scale", 0.0))
    base = last_monthly(history) if mode == "weekly" else None
    if base is not None:
        rel = base.get("relative")
        if not rel:  # 古い記録: 目標の重みを JPY 以外の合計で割って相対の重みに戻す
            tot = sum(base["target_weights"].values())
            rel = {p: w / tot for p, w in base["target_weights"].items()} if tot > 0 else None
        if rel and all(p in close.columns for p in rel):
            out = scale_weights(close, rel, **common)
            out.update(relative=rel, kind="weekly", based_on=base.get("date"))
            return out
    out = current_weights(close, cfg.get("k_max", 5), cfg.get("w_max", 0.4), cfg.get("caps") or {}, **common)
    out["kind"] = "monthly"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rebalance.yaml")
    ap.add_argument("--history", default="docs/monitor/rebalances.jsonl", help="計画を追記する記録（空なら記録しない）")
    ap.add_argument("--mode", default="auto", choices=["auto", "monthly", "weekly"])
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    pool: list[str] = cfg["pool"]
    api = PublicClient(min_interval=0.5)
    years = range(cfg.get("history_from_year", 2019), datetime.now(timezone.utc).year + 1)
    close = pd.concat([fetch_daily(api, p, years) for p in pool], axis=1)
    close = close[close.index < pd.Timestamp.now(tz="UTC").normalize()]
    history = load_jsonl(Path(args.history)) if args.history else []
    tw = target_for_today(cfg, close, history, args.mode)
    trend = {None: "判定なし", True: "上昇（移動平均以上）", False: "下降（移動平均を下回る → 暗号資産を縮める）"}[tw.get("trend")]
    print(f"方式: {'月初（銘柄と相対の重みを選び直す）' if tw['kind'] == 'monthly' else '週次（' + str(tw.get('based_on')) + ' の相対の重みで JPY の割合だけ見直す）'}")
    print(f"目標（{tw['as_of']} までのデータ）: " + "、".join(f"{p} {w:.0%}" for p, w in tw["weights"].items())
          + f"、JPY {tw['cash']:.0%}（推定ボラ {tw['est_vol']:.1%}、縮める前 {tw['est_vol_full']:.1%}、BTC のトレンド: {trend}）")

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
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "as_of": tw["as_of"], "kind": tw["kind"],
            "relative": {p: round(w, 4) for p, w in tw["relative"].items()}, "trend": tw.get("trend"),
            "target_weights": {p: round(w, 4) for p, w in tw["weights"].items()}, "target_cash": round(tw["cash"], 4),
            "est_vol": round(tw["est_vol"], 4), "current": {p: round(w, 4) for p, w in current.items() if w > 0},
            "orders": [{"pair": o.pair, "side": o.side, "frac": round(o.frac, 4)} for o in orders], "dry_run": True})
    return 0


if __name__ == "__main__":
    sys.exit(main())
