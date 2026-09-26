"""bitbank の全ペアについて、前日（UTC）1 日分の約定数・出来高（JPY 換算）を数える（公開 API、認証不要）。

    python scripts/activity.py [YYYYMMDD]

ペアごとに 1 リクエスト（/{pair}/transactions/{日付}）。結果は Markdown の表で標準出力と reports/activity/report.md に書く。
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bbdata.client import BitbankAPIError, PublicClient


def main() -> None:
    day = sys.argv[1] if len(sys.argv) > 1 else (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y%m%d")
    api = PublicClient(base_url="https://api.bitbank.cc/v1", min_interval=0.5)
    pairs = [p["name"] for p in api.get("/spot/pairs")["pairs"] if p["name"].endswith("_jpy")]
    pub = PublicClient(min_interval=0.5)
    rows = []
    for pair in pairs:
        try:
            txs = pub.transactions(pair, day)
        except BitbankAPIError as exc:
            rows.append((pair, 0, 0.0, 0.0, f"取得できず: {exc}"))
            continue
        n = len(txs)
        notional = sum(float(t["price"]) * float(t["amount"]) for t in txs)
        amount = sum(float(t["amount"]) for t in txs)
        rows.append((pair, n, notional, amount, ""))
    rows.sort(key=lambda r: -r[1])
    md = [f"# bitbank の約定の活発さ（{day} UTC、1 日分）\n",
          "| ペア | 約定数 | 1 分あたり | 出来高（JPY、百万） | 出来高（枚） | 備考 |", "|---|---|---|---|---|---|"]
    for pair, n, notional, amount, note in rows:
        md.append(f"| {pair} | {n:,} | {n / 1440:.1f} | {notional / 1e6:,.0f} | {amount:,.2f} | {note} |")
    text = "\n".join(md) + "\n"
    out = Path("reports/activity")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
