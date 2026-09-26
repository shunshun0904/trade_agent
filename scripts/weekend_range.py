"""曜日別の値幅（公式ロウソク足、2017 年以降、認証不要）。土日に取引したいオーナーの問いに答える（2026-09-27）。

    python scripts/weekend_range.py [pair]

- 日足（JST の暦日に揃えるため 4 時間足から作り直す。公式の日足は UTC 区切り）: (高値 − 安値) / 始値 の曜日別分布
- 4 時間足: JST の曜日 × 時間帯ごとの値幅の中央値
リクエストは年ごとに 4hour を 1 回（10 年で 10 回）。結果は reports/weekend_range/report.md。
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bbdata.client import BitbankAPIError, PublicClient

WD = ["月", "火", "水", "木", "金", "土", "日"]


def main() -> None:
    pair = sys.argv[1] if len(sys.argv) > 1 else "btc_jpy"
    api = PublicClient(min_interval=0.5)
    rows = []
    for year in range(2017, datetime.now(timezone.utc).year + 1):
        try:
            rows += api.candlestick(pair, "4hour", str(year))
        except BitbankAPIError as exc:
            print(f"{year}: {exc}")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume", "ts"]).astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float, "ts": "int64"})
    df["t"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("Asia/Tokyo")
    df = df.sort_values("t").drop_duplicates("ts")
    df = df[df["t"] >= "2020-01-01"]  # 直近 6 年弱（市場の厚みが今に近い期間）
    df["date"] = df["t"].dt.date
    df["wd"] = df["t"].dt.dayofweek
    df["hour"] = df["t"].dt.hour
    df["range"] = (df["high"] - df["low"]) / df["open"]
    df["absret"] = (df["close"] / df["open"] - 1).abs()

    # JST 暦日の日足
    day = df.groupby("date").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                 close=("close", "last"), volume=("volume", "sum"), wd=("wd", "first"), n=("ts", "size"))
    day = day[day["n"] == 6]
    day["range"] = (day["high"] - day["low"]) / day["open"]
    day["absret"] = (day["close"] / day["open"] - 1).abs()
    md = [f"# {pair} の曜日別の値幅（JST、{day.index.min()} 〜 {day.index.max()}、4 時間足から集計）\n",
          "## 日足（JST の暦日）: (高値 − 安値) / 始値\n",
          "| 曜日 | 日数 | 中央値 | 25% | 75% | 平均 | 1% 以上の日 | 2% 以上の日 | 3% 以上の日 | 終値の変化（絶対値）の中央値 | 出来高の中央値（枚） |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for w in range(7):
        d = day[day["wd"] == w]
        r = d["range"]
        md.append(f"| {WD[w]} | {len(d)} | {r.median():.2%} | {r.quantile(.25):.2%} | {r.quantile(.75):.2%} | {r.mean():.2%} | "
                  f"{(r >= .01).mean():.0%} | {(r >= .02).mean():.0%} | {(r >= .03).mean():.0%} | {d['absret'].median():.2%} | {d['volume'].median():,.0f} |")
    wk = day[day["wd"] < 5]["range"]
    we = day[day["wd"] >= 5]["range"]
    md.append(f"\n平日の中央値 {wk.median():.2%}、土日の中央値 {we.median():.2%}（土日 / 平日 = {we.median() / wk.median():.2f}）\n")
    md.append("## 4 時間足: 曜日 × 開始時刻（JST）の値幅の中央値\n")
    hours = sorted(df["hour"].unique())
    md.append("| 曜日 | " + " | ".join(f"{h:02d}〜{(h + 4) % 24:02d} 時" for h in hours) + " |")
    md.append("|---|" + "---|" * len(hours))
    for w in range(7):
        cells = []
        for h in hours:
            r = df[(df["wd"] == w) & (df["hour"] == h)]["range"]
            cells.append(f"{r.median():.2%}" if len(r) else "-")
        md.append(f"| {WD[w]} | " + " | ".join(cells) + " |")
    md.append("\n## 4 時間足: 曜日 × 開始時刻（JST）で値幅が 0.6%（成行往復の費用 0.3% の 2 倍）以上だった割合\n")
    md.append("| 曜日 | " + " | ".join(f"{h:02d}〜{(h + 4) % 24:02d} 時" for h in hours) + " |")
    md.append("|---|" + "---|" * len(hours))
    for w in range(7):
        cells = []
        for h in hours:
            r = df[(df["wd"] == w) & (df["hour"] == h)]["range"]
            cells.append(f"{(r >= .006).mean():.0%}" if len(r) else "-")
        md.append(f"| {WD[w]} | " + " | ".join(cells) + " |")
    # 年ごとの土日の値幅（最近ほど小さいかを見る）
    day["year"] = pd.to_datetime(day.index).year
    md.append("\n## 年ごとの日足の値幅の中央値（平日 / 土日）\n")
    md.append("| 年 | 平日 | 土日 |\n|---|---|---|")
    for y, g in day.groupby("year"):
        md.append(f"| {y} | {g[g['wd'] < 5]['range'].median():.2%} | {g[g['wd'] >= 5]['range'].median():.2%} |")
    text = "\n".join(md) + "\n"
    out = Path("reports/weekend_range")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
