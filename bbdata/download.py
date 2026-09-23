"""過去データの取得・保存・読み込み。

保存レイアウト:
    {root}/raw/transactions/{pair}/{YYYYMMDD}.parquet
    {root}/raw/candles/{pair}/{candle_type}/{YYYYMMDD または YYYY}.parquet
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

from .client import BitbankAPIError, PublicClient

log = logging.getLogger(__name__)

# 公式ドキュメント記載の candle-type と期間指定の対応
DAILY_CANDLE_TYPES = frozenset({"1min", "5min", "15min", "30min", "1hour"})
YEARLY_CANDLE_TYPES = frozenset({"4hour", "8hour", "12hour", "1day", "1week", "1month"})

# YYYYMMDD の日付境界がどのタイムゾーンかは公式ドキュメントに記載がない。
# そのため前後 PAD_DAYS 日を余分に取得し、読み込み時に executed_at で切り出す。
PAD_DAYS = 1
# 直近の日付は確定していない可能性があるため、この日数以内は毎回取り直す
MUTABLE_DAYS = 2

TRADE_DTYPES = {
    "transaction_id": "int64",
    "side": "object",
    "price": "float64",
    "amount": "float64",
    "executed_at": "int64",
}


def daterange(start: date, end: date) -> Iterator[date]:
    """start 以上 end 以下の日付を返す。"""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def to_utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _is_final(d: date, today: date) -> bool:
    return d <= today - timedelta(days=MUTABLE_DAYS)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


# ---------------------------------------------------------------- transactions

def trades_path(root: str | Path, pair: str, d: date) -> Path:
    return Path(root) / "raw" / "transactions" / pair / f"{d:%Y%m%d}.parquet"


def transactions_to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(TRADE_DTYPES))
    return df.astype(TRADE_DTYPES)


def download_transactions(
    client: PublicClient,
    pair: str,
    start: date,
    end: date,
    root: str | Path,
    overwrite: bool = False,
    today: date | None = None,
) -> list[date]:
    """[start, end) の日付について約定履歴を取得して保存する。

    前後 PAD_DAYS 日も取得する。確定済みの日付で既にファイルがあればスキップする。
    戻り値は今回取得した日付のリスト。
    """
    today = today or _utc_today()
    first = start - timedelta(days=PAD_DAYS)
    # 日付境界が UTC より進んだタイムゾーンの場合に備え、UTC の翌日まで要求する
    last = min(end - timedelta(days=1) + timedelta(days=PAD_DAYS), today + timedelta(days=1))
    fetched: list[date] = []
    for d in daterange(first, last):
        path = trades_path(root, pair, d)
        if path.exists() and not overwrite and _is_final(d, today):
            continue
        try:
            rows = client.transactions(pair, f"{d:%Y%m%d}")
        except BitbankAPIError as exc:
            log.warning("%s %s をスキップ: %s", pair, d, exc)
            continue
        df = transactions_to_frame(rows)
        _write_parquet(df, path)
        fetched.append(d)
        log.info("%s %s: %d 件", pair, d, len(df))
    return fetched


def load_transactions(root: str | Path, pair: str, start, end) -> pd.DataFrame:
    """[start, end)（UTC）の約定を返す。

    前後 PAD_DAYS 日のファイルも読み、transaction_id で重複を除去してから
    executed_at で切り出す。列 ts（UTC の datetime）を追加する。
    """
    start, end = to_utc(start), to_utc(end)
    frames, missing = [], []
    first = (start - pd.Timedelta(days=PAD_DAYS)).date()
    last = (end + pd.Timedelta(days=PAD_DAYS)).date()
    for d in daterange(first, last):
        p = trades_path(root, pair, d)
        if p.exists():
            frames.append(pd.read_parquet(p))
        elif start.date() <= d < end.date():
            missing.append(d)
    if missing:
        log.warning("%s: 未取得の日付 %d 件（例: %s）", pair, len(missing), missing[0])

    frames = [f for f in frames if not f.empty]
    if frames:
        df = pd.concat(frames, ignore_index=True).astype(TRADE_DTYPES)
    else:
        df = transactions_to_frame([])
    df = df.drop_duplicates("transaction_id")
    df["ts"] = pd.to_datetime(df["executed_at"], unit="ms", utc=True)
    df = df[(df["ts"] >= start) & (df["ts"] < end)]
    return df.sort_values(["executed_at", "transaction_id"]).reset_index(drop=True)


# --------------------------------------------------------------------- candles

def candles_path(root: str | Path, pair: str, candle_type: str, period: str) -> Path:
    return Path(root) / "raw" / "candles" / pair / candle_type / f"{period}.parquet"


def _candle_periods(candle_type: str, start: date, end: date, today: date) -> list[tuple[str, bool]]:
    """(period 文字列, 確定済みか) のリスト。end は含まない。"""
    if candle_type in DAILY_CANDLE_TYPES:
        first = start - timedelta(days=PAD_DAYS)
        last = min(end - timedelta(days=1) + timedelta(days=PAD_DAYS), today + timedelta(days=1))
        return [(f"{d:%Y%m%d}", _is_final(d, today)) for d in daterange(first, last)]
    if candle_type in YEARLY_CANDLE_TYPES:
        last_year = min((end - timedelta(days=1)).year, today.year)
        return [(str(y), y < today.year) for y in range(start.year, last_year + 1)]
    raise ValueError(f"未知の candle_type: {candle_type}")


def download_candles(
    client: PublicClient,
    pair: str,
    candle_type: str,
    start: date,
    end: date,
    root: str | Path,
    overwrite: bool = False,
    today: date | None = None,
) -> list[str]:
    """公式ロウソク足を取得して保存する（自前集計の検証用）。"""
    today = today or _utc_today()
    fetched = []
    for period, final in _candle_periods(candle_type, start, end, today):
        path = candles_path(root, pair, candle_type, period)
        if path.exists() and not overwrite and final:
            continue
        try:
            rows = client.candlestick(pair, candle_type, period)
        except BitbankAPIError as exc:
            log.warning("%s %s %s をスキップ: %s", pair, candle_type, period, exc)
            continue
        df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume", "ts_ms"])
        df = df.astype({c: "float64" for c in ["open", "high", "low", "close", "volume"]} | {"ts_ms": "int64"})
        _write_parquet(df, path)
        fetched.append(period)
    return fetched


def load_candles(root: str | Path, pair: str, candle_type: str, start, end) -> pd.DataFrame:
    """[start, end)（UTC）の公式ロウソク足。index は ts_ms を UTC 時刻にしたもの。"""
    start, end = to_utc(start), to_utc(end)
    base = Path(root) / "raw" / "candles" / pair / candle_type
    frames = [pd.read_parquet(p) for p in sorted(base.glob("*.parquet"))] if base.exists() else []
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.concat(frames, ignore_index=True).drop_duplicates("ts_ms")
    df.index = pd.to_datetime(df.pop("ts_ms"), unit="ms", utc=True)
    df.index.name = "ts"
    df = df.sort_index()
    return df[(df.index >= start) & (df.index < end)]
