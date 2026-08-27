"""Historical candle download (spec 13, Phase 0).

bitbank serves candles a day (or a year) at a time, so a backfill is a loop
over calendar buckets. Results are cached on disk: re-running a backtest should
not re-download a month of history, and the public endpoint deserves the same
courtesy as the private one.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

from ..errors import ExchangeError
from ..models.market import Candle
from ..timeutil import iso, parse_iso

log = logging.getLogger(__name__)

INTRADAY = {"1min", "5min", "15min", "30min", "1hour"}


def backfill(exchange, *, candle_type: str, days: int, out_dir: str | Path,
             end: date | None = None) -> list[Candle]:
    """Download `days` of history, caching one file per bucket."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    end = end or date.today()
    candles: list[Candle] = []

    for key in _buckets(candle_type, days, end):
        path = out / f"{candle_type}-{key}.json"
        if path.exists():
            candles.extend(_load(path))
            continue
        try:
            rows = exchange.get_candles(candle_type, key)
        except ExchangeError as exc:
            log.warning("skipping %s: %s", key, exc)
            continue
        _save(path, rows)
        candles.extend(rows)

    candles.sort(key=lambda c: c.opened_at)
    return _dedupe(candles)


def load_cached(out_dir: str | Path, candle_type: str) -> list[Candle]:
    rows: list[Candle] = []
    for path in sorted(Path(out_dir).glob(f"{candle_type}-*.json")):
        rows.extend(_load(path))
    rows.sort(key=lambda c: c.opened_at)
    return _dedupe(rows)


def _buckets(candle_type: str, days: int, end: date) -> list:
    if candle_type in INTRADAY:
        return [end - timedelta(days=offset) for offset in range(days)]
    years = {(end - timedelta(days=offset)).year for offset in range(days)}
    return sorted(years, reverse=True)


def _save(path: Path, candles: list[Candle]) -> None:
    path.write_text(json.dumps([{
        "open": str(c.open), "high": str(c.high), "low": str(c.low),
        "close": str(c.close), "volume": str(c.volume),
        "opened_at": iso(c.opened_at),
    } for c in candles], ensure_ascii=False))


def _load(path: Path) -> list[Candle]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return []
    return [Candle(open=r["open"], high=r["high"], low=r["low"], close=r["close"],
                   volume=r["volume"], opened_at=parse_iso(r["opened_at"]))
            for r in raw]


def _dedupe(candles: list[Candle]) -> list[Candle]:
    seen: set = set()
    out: list[Candle] = []
    for candle in candles:
        if candle.opened_at not in seen:
            seen.add(candle.opened_at)
            out.append(candle)
    return out
