"""候補 2 の H2a の前向きのドライラン: 判断の時刻（0・8・16 時 UTC）に bitbank の気配を記録する。発注はしない。2026-09-28 オーナー決定。

    python scripts/ext_quotes.py              # 1 行の JSON を標準出力に出す（ワークフローが reports/ext_study/quotes.jsonl に追記する）

公開の ticker の最良の売り気配（sell、成行で買うときの価格）・買い気配（buy、成行で売るときの価格）・最後の約定（last）を記録する。
Actions の schedule は遅れるので、記録した時刻と判断の時刻（それ以前で最も近い 0・8・16 時）の差（delay_min）も残す。
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

HOURS = (0, 8, 16)


def slot_of(now: pd.Timestamp, hours=HOURS) -> pd.Timestamp:
    """now 以前で最も近い判断の時刻。"""
    day = now.normalize()
    past = [day + pd.Timedelta(hours=h) for h in hours if day + pd.Timedelta(hours=h) <= now]
    return max(past) if past else day - pd.Timedelta(days=1) + pd.Timedelta(hours=max(hours))


def record(api, pair: str = "btc_jpy", now: pd.Timestamp | None = None) -> dict:
    t = api.ticker(pair)
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    slot = slot_of(now)
    return {"slot": slot.isoformat(), "fetched_at": now.isoformat(timespec="seconds"),
            "delay_min": round((now - slot) / pd.Timedelta(minutes=1), 1), "pair": pair,
            "sell": float(t["sell"]), "buy": float(t["buy"]), "last": float(t["last"]),
            "ticker_ts": pd.Timestamp(int(t["timestamp"]), unit="ms", tz="UTC").isoformat()}


def main(argv=None, api=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="btc_jpy")
    args = ap.parse_args(argv)
    if api is None:
        from bbdata.client import PublicClient
        api = PublicClient(min_interval=0.3)
    print(json.dumps(record(api, args.pair), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
