"""候補 2（24 時間の新しい情報源）の外部データの確認。2026-09-28 オーナー決定。

    python scripts/ext_check.py [--out reports/ext]

Binance・Bybit の公開 API とアーカイブ、ドル円の候補（Dukascopy、FRED）に GitHub Actions から届くか、履歴がどこまであるかを調べる。
認証は使わず、発注もしない。リクエストは 40 件ほど（公開 API に大量のリクエストを送らない）。
2 回目（2026-09-28）: Binance のアーカイブの始まりの月と、代わりの取引所（Deribit、OKX、Kraken Futures、BitMEX）に届くかだけを加えた。
3 回目（2026-09-28）: オーナー決定で使う BitMEX と Deribit の資金調達率の履歴がどこまであるかを加えた。
結果は reports/ext/check.md・check.json。
"""
from __future__ import annotations

import argparse
import json
import lzma
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

OUT = Path("reports/ext")
UA = {"User-Agent": "trade_agent research (github.com/shunshun0904/trade_agent)"}
MS_2020_04_01 = 1585699200000   # 2020-04-01 00:00 UTC
MS_2020_04_02 = 1585785600000
MS_2021_01_01 = 1609459200000
MS_2021_01_02 = 1609545600000

# (名前, 方法, URL, 読み方, 説明)
PROBES = [
    ("binance_spot_klines", "GET", "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&startTime=0&limit=2",
     "binance_klines", "Binance 現物 BTCUSDT の 1 時間足（最も古い足）"),
    ("binance_data_api_klines", "GET", "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1h&startTime=0&limit=2",
     "binance_klines", "Binance の市場データ専用の窓口（同じ現物の足）"),
    ("binance_fut_funding", "GET", "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&startTime=0&limit=2",
     "binance_funding", "Binance 永久先物 BTCUSDT の資金調達率（最も古い値）"),
    ("binance_fut_klines", "GET", "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1h&startTime=0&limit=2",
     "binance_klines", "Binance 永久先物の 1 時間足"),
    ("binance_fut_premium", "GET", "https://fapi.binance.com/fapi/v1/premiumIndexKlines?symbol=BTCUSDT&interval=1h&startTime=0&limit=2",
     "binance_klines", "Binance 永久先物のプレミアム指数（先物と現物の価格差）の 1 時間足"),
    ("binance_fut_oi_hist", "GET", "https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=1h&limit=2",
     "binance_oi", "Binance 永久先物の建玉の履歴（直近の分だけの窓口）"),
    ("vision_funding", "HEAD", "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2019-10.zip",
     "head", "Binance のアーカイブ: 資金調達率（2019-10 の月次ファイル）"),
    ("vision_um_klines", "HEAD", "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2019-10.zip",
     "head", "Binance のアーカイブ: 永久先物の 1 時間足（2019-10）"),
    ("vision_spot_klines", "HEAD", "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2019-10.zip",
     "head", "Binance のアーカイブ: 現物の 1 時間足（2019-10）"),
    ("vision_premium", "HEAD", "https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines/BTCUSDT/1h/BTCUSDT-1h-2020-01.zip",
     "head", "Binance のアーカイブ: プレミアム指数の 1 時間足（2020-01）"),
    ("vision_metrics_2020", "HEAD", "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2020-09-01.zip",
     "head", "Binance のアーカイブ: 建玉などの 5 分ごとの記録（2020-09-01）"),
    ("vision_metrics_2021", "HEAD", "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2021-12-01.zip",
     "head", "Binance のアーカイブ: 建玉などの 5 分ごとの記録（2021-12-01）"),
    ("vision_metrics_2023", "HEAD", "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2023-01-01.zip",
     "head", "Binance のアーカイブ: 建玉などの 5 分ごとの記録（2023-01-01）"),
    ("bybit_funding_2020", "GET", f"https://api.bybit.com/v5/market/funding/history?category=linear&symbol=BTCUSDT&startTime={MS_2020_04_01}"
     f"&endTime={MS_2020_04_02}&limit=10", "bybit_funding", "Bybit 永久先物 BTCUSDT の資金調達率（2020-04-01 の分）"),
    ("bybit_funding_latest", "GET", "https://api.bybit.com/v5/market/funding/history?category=linear&symbol=BTCUSDT&limit=2",
     "bybit_funding", "Bybit の資金調達率（最新）"),
    ("bybit_oi_2021", "GET", f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol=BTCUSDT&intervalTime=1h"
     f"&startTime={MS_2021_01_01}&endTime={MS_2021_01_02}&limit=5", "bybit_oi", "Bybit の建玉の 1 時間ごとの履歴（2021-01-01 の分）"),
    ("bybit_kline_2020", "GET", f"https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=60&start={MS_2020_04_01}"
     f"&end={MS_2020_04_02}&limit=5", "bybit_kline", "Bybit 永久先物の 1 時間足（2020-04-01 の分）"),
    ("bybit_archive", "HEAD", "https://public.bybit.com/trading/BTCUSDT/", "head", "Bybit の約定のアーカイブ（一覧のページ）"),
    ("dukascopy_hour_2020_01", "GET", "https://datafeed.dukascopy.com/datafeed/USDJPY/2020/00/BID_candles_hour_1.bi5",
     "dukascopy", "Dukascopy のドル円（USDJPY）: 2020 年 1 月の 1 時間足の月次ファイル"),
    ("dukascopy_min_2020_01_02", "GET", "https://datafeed.dukascopy.com/datafeed/USDJPY/2020/00/02/BID_candles_min_1.bi5",
     "dukascopy", "Dukascopy のドル円: 2020-01-02 の 1 分足の日次ファイル"),
    ("fred_dexjpus", "GET", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXJPUS", "fred",
     "FRED のドル円（DEXJPUS、日次。ニューヨーク正午の値）"),
    # 2 回目（2026-09-28）: Binance のアーカイブがどの月からあるか。Binance の窓口と Bybit は Actions から届かなかった（451・403）ので、
    # 代わりをオーナーと相談するために、他の取引所の公開 API に届くかだけを確かめる（データは研究に使わない）
] + [(f"vision_funding_{m}", "HEAD", f"https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{m}.zip",
      "head", f"Binance のアーカイブ: 資金調達率（{m}）") for m in ("2019-09", "2019-12", "2020-01", "2020-06", "2021-01", "2022-01", "2024-01")] + [
    (f"vision_um_klines_{m}", "HEAD", f"https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{m}.zip",
     "head", f"Binance のアーカイブ: 永久先物の 1 時間足（{m}）") for m in ("2019-12", "2020-01")] + [
    (f"vision_premium_{m}", "HEAD", f"https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines/BTCUSDT/1h/BTCUSDT-1h-{m}.zip",
     "head", f"Binance のアーカイブ: プレミアム指数の 1 時間足（{m}）") for m in ("2019-09", "2019-12")] + [
    ("deribit_funding", "GET", "https://www.deribit.com/api/v2/public/get_funding_rate_history?instrument_name=BTC-PERPETUAL"
     "&start_timestamp=1577836800000&end_timestamp=1577840400000", "reach", "Deribit の資金調達率（届くかだけ）"),
    ("okx_funding", "GET", "https://www.okx.com/api/v5/public/funding-rate-history?instId=BTC-USDT-SWAP&limit=1", "reach",
     "OKX の資金調達率（届くかだけ）"),
    ("kraken_futures_funding", "GET", "https://futures.kraken.com/derivatives/api/v4/historicalfundingrates?symbol=PF_XBTUSD", "reach",
     "Kraken Futures の資金調達率（届くかだけ）"),
    ("bitmex_funding", "GET", "https://www.bitmex.com/api/v1/funding?symbol=XBTUSD&count=1&reverse=false", "reach",
     "BitMEX の資金調達率（届くかだけ）"),
    # 3 回目（2026-09-28）: オーナー決定で Bybit の代わりに BitMEX と Deribit を使う。資金調達率の履歴がどこまであるか
    ("bitmex_funding_2019", "GET", "https://www.bitmex.com/api/v1/funding?symbol=XBTUSD&count=3&reverse=false&startTime=2019-06-01T00:00:00Z",
     "bitmex_funding", "BitMEX XBTUSD の資金調達率（2019-06-01 以降の最初の 3 件）"),
    ("bitmex_funding_2020", "GET", "https://www.bitmex.com/api/v1/funding?symbol=XBTUSD&count=3&reverse=false&startTime=2020-01-01T00:00:00Z",
     "bitmex_funding", "BitMEX XBTUSD の資金調達率（2020-01-01 以降の最初の 3 件）"),
    ("deribit_funding_2019", "GET", "https://www.deribit.com/api/v2/public/get_funding_rate_history?instrument_name=BTC-PERPETUAL"
     "&start_timestamp=1559347200000&end_timestamp=1559358000000", "deribit_funding", "Deribit BTC-PERPETUAL の資金調達率（2019-06-01 の 3 時間）"),
    ("deribit_funding_2020", "GET", "https://www.deribit.com/api/v2/public/get_funding_rate_history?instrument_name=BTC-PERPETUAL"
     "&start_timestamp=1577836800000&end_timestamp=1577847600000", "deribit_funding", "Deribit BTC-PERPETUAL の資金調達率（2020-01-01 の 3 時間）"),
]


def ms_iso(ms) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def decode_bi5_candles(raw: bytes, point: float = 1000.0) -> list[tuple]:
    """Dukascopy の足のファイル（LZMA で圧縮、1 件 24 バイト: 秒のずれ、始値、終値、安値、高値（整数 ÷ point）、出来高）。"""
    data = lzma.decompress(raw) if raw else b""
    out = []
    for off in range(0, len(data) - len(data) % 24, 24):
        t, o, c, lo, hi, v = struct.unpack(">5if", data[off:off + 24])
        out.append((t, o / point, c / point, lo / point, hi / point, v))
    return out


def summarize(kind: str, resp: requests.Response) -> str:
    if kind == "head":
        size = resp.headers.get("Content-Length")
        return f"{int(size):,} バイト" if size and size.isdigit() else "-"
    if resp.status_code != 200:
        return resp.text[:120].replace("\n", " ")
    if kind == "binance_klines":
        rows = resp.json()
        return f"最も古い足 {ms_iso(rows[0][0])}（UTC）" if rows else "空"
    if kind == "binance_funding":
        rows = resp.json()
        return f"最も古い値 {ms_iso(rows[0]['fundingTime'])}、率 {rows[0]['fundingRate']}" if rows else "空"
    if kind == "binance_oi":
        rows = resp.json()
        return f"{len(rows)} 件、最初 {ms_iso(rows[0]['timestamp'])}" if rows else "空"
    if kind in ("bybit_funding", "bybit_oi", "bybit_kline"):
        body = resp.json()
        rows = (body.get("result") or {}).get("list") or []
        if body.get("retCode") != 0:
            return f"retCode {body.get('retCode')}: {body.get('retMsg')}"
        if not rows:
            return "空（この期間の履歴なし）"
        key = {"bybit_funding": "fundingRateTimestamp", "bybit_oi": "timestamp"}.get(kind)
        ts = [int(r[key]) if key else int(r[0]) for r in rows]
        return f"{len(rows)} 件、{ms_iso(min(ts))} 〜 {ms_iso(max(ts))}"
    if kind == "dukascopy":
        recs = decode_bi5_candles(resp.content)
        if not recs:
            return f"{len(resp.content):,} バイト、足 0 本"
        t, o, c, lo, hi, v = recs[0]
        return f"{len(resp.content):,} バイト、足 {len(recs)} 本、最初の足: ずれ {t} 秒、始値 {o:.3f}、高値 {hi:.3f}、安値 {lo:.3f}、終値 {c:.3f}"
    if kind == "bitmex_funding":
        rows = resp.json()
        return (f"{len(rows)} 件、最初 {rows[0]['timestamp'][:16]}、率 {rows[0]['fundingRate']}、間隔 {rows[0].get('fundingInterval', '-')}"
                if rows else "空")
    if kind == "deribit_funding":
        rows = (resp.json() or {}).get("result") or []
        return (f"{len(rows)} 件、最初 {ms_iso(rows[0]['timestamp'])}、8 時間の率 {rows[0].get('interest_8h')}、1 時間の率 {rows[0].get('interest_1h')}"
                if rows else "空")
    if kind == "reach":
        return f"{len(resp.content):,} バイト"
    if kind == "fred":
        lines = [x for x in resp.text.strip().splitlines() if x]
        return f"{len(lines) - 1:,} 行、最初 {lines[1]}、最後 {lines[-1]}" if len(lines) > 1 else "空"
    return "-"


def probe(session: requests.Session, name: str, method: str, url: str, kind: str) -> dict:
    t0 = time.monotonic()
    try:
        resp = session.request(method, url, timeout=30, allow_redirects=True)
        return {"name": name, "status": resp.status_code, "ok": resp.status_code == 200, "detail": summarize(kind, resp),
                "seconds": round(time.monotonic() - t0, 2)}
    except Exception as exc:  # 届かない（DNS、TLS、タイムアウト）も結果として記録する
        return {"name": name, "status": None, "ok": False, "detail": f"{type(exc).__name__}: {str(exc)[:120]}",
                "seconds": round(time.monotonic() - t0, 2)}


def main(argv: list[str] | None = None, session: requests.Session | None = None, pause: float = 0.5) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    s = session or requests.Session()
    s.headers.update(UA)
    rows = []
    for name, method, url, kind, desc in PROBES:
        r = probe(s, name, method, url, kind) | {"url": url, "description": desc}
        rows.append(r)
        print(f"{name}: {r['status']} {r['detail']}")
        time.sleep(pause)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out / "check.json").write_text(json.dumps({"run_at": run_at, "probes": rows}, ensure_ascii=False, indent=1))
    md = ["# 候補 2: 外部データの確認（GitHub Actions から届くか、履歴がどこまであるか）\n",
          f"- 実行 {run_at}（UTC）。認証なしの公開 API とアーカイブだけ。発注はしない。",
          "- 状態 200 以外（451・403 など）は、その場所（Actions の機械の国）からは使えないことを示す。\n",
          "| 名前 | 内容 | 状態 | 結果 |", "|---|---|---|---|"]
    md += [f"| {r['name']} | {r['description']} | {r['status'] if r['status'] is not None else '届かない'} | {r['detail']} |" for r in rows]
    (out / "check.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
