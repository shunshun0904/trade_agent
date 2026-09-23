"""コマンドライン。日付はすべて UTC、--start は含み --end は含まない。

例:
    python -m bbdata download   --pairs btc_jpy --start 2025-01-01 --end 2026-09-24 --candles 15min
    python -m bbdata build-bars --pairs btc_jpy --start 2025-01-01 --end 2026-09-24 --freqs 15min 1h
    python -m bbdata validate   --pair  btc_jpy --start 2026-09-01 --end 2026-09-08
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime

from .bars import build_bars, save_bars
from .client import PublicClient
from .download import download_candles, download_transactions, load_candles
from .validate import compare_with_official

# 自前の足の freq と公式 candle-type の対応
FREQ_TO_CANDLE = {"1min": "1min", "5min": "5min", "15min": "15min", "30min": "30min", "1h": "1hour"}


def _date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _common() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--root", default="data", help="保存先ディレクトリ")
    p.add_argument("--start", type=_date, required=True, help="YYYY-MM-DD（UTC、含む）")
    p.add_argument("--end", type=_date, required=True, help="YYYY-MM-DD（UTC、含まない）")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bbdata")
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = _common()

    d = sub.add_parser("download", parents=[common], help="約定履歴（と公式ロウソク足）を取得")
    d.add_argument("--pairs", nargs="+", default=["btc_jpy"])
    d.add_argument("--candles", nargs="*", default=[], help="検証用に取得する candle-type（例: 15min 1hour）")
    d.add_argument("--overwrite", action="store_true")
    d.add_argument("--min-interval", type=float, default=0.3, help="リクエスト間隔（秒）")

    b = sub.add_parser("build-bars", parents=[common], help="保存済み約定から足を作る")
    b.add_argument("--pairs", nargs="+", default=["btc_jpy"])
    b.add_argument("--freqs", nargs="+", default=["15min", "1h"])
    b.add_argument("--large-trade-amount", type=float, default=None, help="大口約定とみなす数量（base 通貨）")
    b.add_argument("--chunk-days", type=int, default=7)

    v = sub.add_parser("validate", parents=[common], help="公式ロウソク足と突き合わせる")
    v.add_argument("--pair", default="btc_jpy")
    v.add_argument("--freq", default="15min", choices=sorted(FREQ_TO_CANDLE))
    v.add_argument("--max-shift", type=int, default=1)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.cmd == "download":
        client = PublicClient(min_interval=args.min_interval)
        for pair in args.pairs:
            download_transactions(client, pair, args.start, args.end, args.root, overwrite=args.overwrite)
            for ct in args.candles:
                download_candles(client, pair, ct, args.start, args.end, args.root, overwrite=args.overwrite)

    elif args.cmd == "build-bars":
        for pair in args.pairs:
            for freq in args.freqs:
                bars = build_bars(
                    args.root, pair, args.start, args.end, freq,
                    large_trade_amount=args.large_trade_amount, chunk_days=args.chunk_days,
                )
                path = save_bars(bars, args.root, pair, freq)
                logging.info("%s %s: %d 本（約定なし %d 本）→ %s",
                             pair, freq, len(bars), int(bars["is_empty"].sum()), path)

    elif args.cmd == "validate":
        candle_type = FREQ_TO_CANDLE[args.freq]
        candles = load_candles(args.root, args.pair, candle_type, args.start, args.end)
        if candles.empty:
            download_candles(PublicClient(), args.pair, candle_type, args.start, args.end, args.root)
            candles = load_candles(args.root, args.pair, candle_type, args.start, args.end)
        bars = build_bars(args.root, args.pair, args.start, args.end, args.freq)
        print(compare_with_official(bars, candles, max_shift=args.max_shift).to_string())


if __name__ == "__main__":
    main()
