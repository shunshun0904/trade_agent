"""候補 2（24 時間の新しい情報源）の外部データ。2026-09-28 オーナー決定。

GitHub Actions（米国）から届く窓口だけを使う（SPEC §9 E1。Binance の api・fapi と Bybit は届かない）:
- Binance のアーカイブ data.binance.vision: 資金調達率・永久先物の 1 時間足・プレミアム指数・現物の 1 時間足は月ごと、
  建玉などの 5 分ごとの記録（metrics）は日ごとの zip（中身は CSV）。月ごとのファイルは 2020-01 から。
- BitMEX: /api/v1/funding（XBTUSD、8 時間ごと）。
- Deribit: /api/v2/public/get_funding_rate_history（BTC-PERPETUAL、1 時間ごと。interest_8h は 8 時間あたりに直した率）。
- Dukascopy: ドル円（USDJPY）の 1 時間足。月ごとの BID と ASK のファイル（LZMA 圧縮、1 件 24 バイト）。
- FRED: DEXJPUS（ドル円、日次、ニューヨーク正午）。
認証は使わず、発注もしない。取ったものは .cache/ext にため、確定した月・日は取り直さない。時刻はすべて UTC、足の index は開始時刻。
"""
from __future__ import annotations

import io
import json
import lzma
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

VISION = "https://data.binance.vision/data"
UA = {"User-Agent": "trade_agent research (github.com/shunshun0904/trade_agent)"}
BI5_CANDLE = np.dtype([("t", ">i4"), ("o", ">i4"), ("c", ">i4"), ("l", ">i4"), ("h", ">i4"), ("v", ">f4")])
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
              "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
FUNDING_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]
METRICS_COLS = ["create_time", "symbol", "sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
                "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]


class Fetcher:
    """キャッシュ付きの取得。404 は None（その月・日のファイルがない）。429・5xx と接続の失敗は待って取り直す（Retry-After が
    あればその秒数、なければ 2, 4, 8, … 秒。最長 60 秒）。strict でなければ、取り直しても取れないものや 404 以外の拒否は
    failures に記録して None を返す（check で使う。どのファイルが取れなかったかを報告に出す）。"""

    def __init__(self, cache: Path | str = ".cache/ext", session: requests.Session | None = None, pause: float = 0.25,
                 retries: int = 6, strict: bool = True):
        self.cache = Path(cache)
        self.session = session or requests.Session()
        self.session.headers.update(UA)
        self.pause, self.retries, self.strict = pause, retries, strict
        self.n_requests = 0
        self.failures: list[dict] = []

    def _fail(self, url: str, status) -> None:
        if self.strict:
            raise RuntimeError(f"{url}: {status}（{self.retries} 回まで取り直した）")
        self.failures.append({"url": url, "status": str(status)})

    def get(self, url: str, rel: str, final: bool = True, pause: float | None = None) -> bytes | None:
        """final: 確定したファイル（ためたものがあれば取り直さない）。pause: 要求の後に待つ秒（窓口ごとの制限に合わせる）。"""
        path = self.cache / rel
        miss = path.with_suffix(path.suffix + ".404")
        if final and path.exists():
            return path.read_bytes()
        if final and miss.exists():
            return None
        last = None
        for attempt in range(self.retries):
            self.n_requests += 1
            try:
                resp = self.session.get(url, timeout=60)
            except requests.RequestException as exc:
                last = f"{type(exc).__name__}"
                time.sleep(min(60, 2 ** (attempt + 1)))
                continue
            time.sleep(self.pause if pause is None else pause)
            if resp.status_code == 200:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(resp.content)
                return resp.content
            if resp.status_code == 404:
                if final:
                    miss.parent.mkdir(parents=True, exist_ok=True)
                    miss.write_bytes(b"")
                return None
            last = f"HTTP {resp.status_code}"
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = (getattr(resp, "headers", None) or {}).get("Retry-After")
                time.sleep(min(60, float(wait)) if wait and str(wait).isdigit() else min(60, 2 ** (attempt + 1)))
                continue
            self._fail(url, f"{last} {resp.text[:200]}")
            return None
        self._fail(url, last)
        return None


def _empty(columns: list[str] | None = None, name: str | None = None):
    """取れなかったときの空の表・系列（index は UTC の時刻。後の計算で他の系列とそろえられるように）。"""
    idx = pd.DatetimeIndex([], tz="UTC")
    if columns is None:
        return pd.Series(dtype=float, index=idx, name=name)
    return pd.DataFrame({c: pd.Series(dtype=float) for c in columns}, index=idx)


def months(start: str, end_exclusive: str) -> list[str]:
    """[start, end) の月（YYYY-MM）。"""
    s = pd.Timestamp(start).to_period("M")
    e = pd.Timestamp(end_exclusive).to_period("M")
    return [str(p) for p in pd.period_range(s, e - 1, freq="M")] if e > s else []


def month_is_final(month: str, today: date | None = None) -> bool:
    today = today or datetime.now(timezone.utc).date()
    first_next = (pd.Timestamp(month + "-01") + pd.offsets.MonthBegin(1)).date()
    return first_next + timedelta(days=2) <= today


def read_zip_csv(raw: bytes, names: list[str] | None = None, positional: bool = False) -> pd.DataFrame:
    """zip の中の最初の CSV。1 行目の最初の欄に英字があれば見出しとして扱う（Binance のファイルは見出しのあるものとないものがあり、
    日時の文字列で始まるものもある）。見出しがなければ names を付ける。positional なら見出しがあっても names に置き換える（列の順が決まっている書式）。"""
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        text = zf.read(zf.namelist()[0]).decode("utf-8")
    first = text.split("\n", 1)[0].split(",")[0].strip()
    has_header = any(ch.isalpha() for ch in first)
    df = pd.read_csv(io.StringIO(text), header=0 if has_header else None)
    if names and (positional or not has_header):
        df = df.iloc[:, :len(names)]
        df.columns = names[:df.shape[1]]
    return df


def _ms_to_utc(x: pd.Series) -> pd.DatetimeIndex:
    v = pd.to_numeric(x).astype("int64")
    v = np.where(v > 10**14, v // 1000, v)            # 2025 年以降の現物のファイルはマイクロ秒
    return pd.DatetimeIndex(pd.to_datetime(v, unit="ms", utc=True))


def binance_klines(f: Fetcher, market: str, kind: str, symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
    """月ごとのアーカイブの足。market は "spot" か "futures/um"、kind は "klines" か "premiumIndexKlines"。index は足の開始時刻。"""
    parts = []
    for m in months(start, end):
        url = f"{VISION}/{market}/monthly/{kind}/{symbol}/{interval}/{symbol}-{interval}-{m}.zip"
        raw = f.get(url, f"binance/{market.replace('/', '_')}/{kind}/{symbol}/{interval}/{m}.zip", month_is_final(m))
        if raw:
            parts.append(read_zip_csv(raw, KLINE_COLS, positional=True))
    if not parts:
        return _empty(["open", "high", "low", "close", "volume"])
    df = pd.concat(parts, ignore_index=True)
    df.index = _ms_to_utc(df["open_time"])
    out = df[["open", "high", "low", "close", "volume"]].astype(float)
    out = out[~out.index.duplicated()].sort_index()
    return out[(out.index >= pd.Timestamp(start, tz="UTC")) & (out.index < pd.Timestamp(end, tz="UTC"))]


def binance_funding(f: Fetcher, symbol: str, start: str, end: str) -> pd.Series:
    """資金調達率（決済の時刻 → 率）。月ごとのアーカイブ（列 calc_time, funding_interval_hours, last_funding_rate）。"""
    parts = []
    for m in months(start, end):
        url = f"{VISION}/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{m}.zip"
        raw = f.get(url, f"binance/funding/{symbol}/{m}.zip", month_is_final(m))
        if raw:
            parts.append(read_zip_csv(raw, FUNDING_COLS, positional=True))
    if not parts:
        return _empty(name="binance")
    df = pd.concat(parts, ignore_index=True)
    s = pd.Series(df["last_funding_rate"].astype(float).to_numpy(), index=_ms_to_utc(df["calc_time"]).floor("min"), name="binance")
    s = s[~s.index.duplicated()].sort_index()
    return s[(s.index >= pd.Timestamp(start, tz="UTC")) & (s.index < pd.Timestamp(end, tz="UTC"))]


def binance_metrics(f: Fetcher, symbol: str, start: str, end: str) -> pd.DataFrame:
    """建玉などの 5 分ごとの記録（日ごとのアーカイブ）。index は create_time（UTC）。"""
    parts = []
    today = datetime.now(timezone.utc).date()
    for d in pd.date_range(start, pd.Timestamp(end) - pd.Timedelta(days=1), freq="D"):
        day = d.strftime("%Y-%m-%d")
        url = f"{VISION}/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{day}.zip"
        raw = f.get(url, f"binance/metrics/{symbol}/{day}.zip", d.date() + timedelta(days=2) <= today)
        if raw:
            parts.append(read_zip_csv(raw, METRICS_COLS))
    if not parts:
        return _empty(["sum_open_interest", "sum_open_interest_value"])
    df = pd.concat(parts, ignore_index=True)
    df.index = pd.DatetimeIndex(pd.to_datetime(df["create_time"], utc=True))
    keep = [c for c in ("sum_open_interest", "sum_open_interest_value") if c in df.columns]
    out = df[keep].astype(float)
    return out[~out.index.duplicated()].sort_index()


def bitmex_funding(f: Fetcher, start: str, end: str, symbol: str = "XBTUSD", pause: float = 2.5) -> pd.Series:
    """BitMEX の資金調達率（8 時間ごと、決済の時刻 → 率）。年ごとに JSON でためる。認証なしの要求は 1 分に 30 件まで
    （BitMEX の REST API の説明）なので、1 件ごとに pause 秒待つ。"""
    out = []
    for year in range(pd.Timestamp(start).year, (pd.Timestamp(end) - pd.Timedelta(seconds=1)).year + 1):   # end は含まない
        final = date(year + 1, 1, 3) <= datetime.now(timezone.utc).date()
        rel = f"bitmex/funding/{symbol}/{year}.json"
        path = f.cache / rel
        if final and path.exists():
            rows = json.loads(path.read_text())
        else:
            rows, cursor = [], f"{year}-01-01T00:00:00Z"
            while True:
                url = (f"https://www.bitmex.com/api/v1/funding?symbol={symbol}&count=500&reverse=false&startTime={cursor}"
                       f"&endTime={year + 1}-01-01T00:00:00Z")
                raw = f.get(url, f"bitmex/tmp/{symbol}-{cursor[:19].replace(':', '')}.json", final=False, pause=pause)
                if raw is None:                       # 取れなかった（strict でないとき）。その年はためない
                    break
                batch = json.loads(raw)
                rows += batch
                if len(batch) < 500:
                    break
                cursor = (pd.Timestamp(batch[-1]["timestamp"]) + pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            if raw is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(rows))
        out += rows
    if not out:
        return _empty(name="bitmex")
    s = pd.Series([float(r["fundingRate"]) for r in out], index=pd.DatetimeIndex(pd.to_datetime([r["timestamp"] for r in out], utc=True)),
                  name="bitmex")
    s = s[~s.index.duplicated()].sort_index()
    return s[(s.index >= pd.Timestamp(start, tz="UTC")) & (s.index < pd.Timestamp(end, tz="UTC"))]


def deribit_funding(f: Fetcher, start: str, end: str, instrument: str = "BTC-PERPETUAL") -> pd.DataFrame:
    """Deribit の資金調達率（1 時間ごとの記録。interest_8h は 8 時間あたりに直した率、interest_1h は 1 時間の率。2 つの関係は
    check で確かめる）。月ごとに取り、1 回の応答が月の終わりまで届かなければ（件数の上限がある場合）、最後の記録の後から取り足す。"""
    out = []
    for m in months(start, end):
        s0 = int(pd.Timestamp(m + "-01", tz="UTC").value // 10**6)
        s1 = int((pd.Timestamp(m + "-01", tz="UTC") + pd.offsets.MonthBegin(1)).value // 10**6)
        cursor, fin = s0, month_is_final(m)
        while cursor < s1:
            url = (f"https://www.deribit.com/api/v2/public/get_funding_rate_history?instrument_name={instrument}"
                   f"&start_timestamp={cursor}&end_timestamp={s1}")
            rel = f"deribit/funding/{instrument}/{m}.json" if cursor == s0 else f"deribit/funding/{instrument}/{m}-{cursor}.json"
            raw = f.get(url, rel, fin)
            rows = (json.loads(raw).get("result") or []) if raw else []
            out += rows
            if not rows:
                break
            last = max(int(r["timestamp"]) for r in rows)
            if last >= s1 - 3_600_000 or last < cursor:
                break
            cursor = last + 1
    if not out:
        return _empty(["interest_8h", "interest_1h"])
    df = pd.DataFrame({"interest_8h": [float(r["interest_8h"]) for r in out], "interest_1h": [float(r["interest_1h"]) for r in out]},
                      index=pd.DatetimeIndex(pd.to_datetime([int(r["timestamp"]) for r in out], unit="ms", utc=True)))
    df = df[~df.index.duplicated()].sort_index()
    return df[(df.index >= pd.Timestamp(start, tz="UTC")) & (df.index < pd.Timestamp(end, tz="UTC"))]


def decode_bi5_candles(raw: bytes, base: pd.Timestamp, point: float = 1000.0) -> pd.DataFrame:
    """Dukascopy の足のファイル。1 件 24 バイト（ビッグエンディアン）: 基準からの秒、始値、終値、安値、高値（整数 ÷ point）、出来高。"""
    data = lzma.decompress(raw) if raw else b""
    n = len(data) // 24
    if n == 0:
        return _empty(["open", "high", "low", "close"])
    rec = np.frombuffer(data[:n * 24], dtype=BI5_CANDLE)
    idx = base + pd.to_timedelta(rec["t"].astype("int64"), unit="s")
    return pd.DataFrame({"open": rec["o"] / point, "close": rec["c"] / point, "low": rec["l"] / point, "high": rec["h"] / point},
                        index=pd.DatetimeIndex(idx))


def dukascopy_hourly(f: Fetcher, start: str, end: str, symbol: str = "USDJPY", pause: float = 1.5) -> pd.DataFrame:
    """ドル円の 1 時間足（BID と ASK の中値）。月ごとのファイル。月の区切りは 0 始まり（1 月 = 00）。2026-09-28 の run 36440860628
    では最初のファイルで取り直しても取れなかった（429 か 5xx が続いた）ので、1 件ごとに pause 秒待つ。"""
    parts = []
    for m in months(start, end):
        y, mo = int(m[:4]), int(m[5:])
        base = pd.Timestamp(m + "-01", tz="UTC")
        sides = {}
        for side in ("BID", "ASK"):
            url = f"https://datafeed.dukascopy.com/datafeed/{symbol}/{y}/{mo - 1:02d}/{side}_candles_hour_1.bi5"
            raw = f.get(url, f"dukascopy/{symbol}/{m}-{side}.bi5", month_is_final(m), pause=pause)
            if raw:
                sides[side] = decode_bi5_candles(raw, base)
        if len(sides) == 2:
            parts.append((sides["BID"] + sides["ASK"]) / 2)
    if not parts:
        return _empty(["open", "high", "low", "close"])
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated()]
    return out[(out.index >= pd.Timestamp(start, tz="UTC")) & (out.index < pd.Timestamp(end, tz="UTC"))]


def fred_series(f: Fetcher, series: str = "DEXJPUS") -> pd.Series:
    """FRED の日次（欠損の "." は除く）。毎回取り直す（1 回の要求）。"""
    raw = f.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}", f"fred/{series}.csv", final=False)
    if raw is None:
        return _empty(name=series)
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = ["date", "value"]
    df = df[pd.to_numeric(df["value"], errors="coerce").notna()]
    return pd.Series(df["value"].astype(float).to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(df["date"], utc=True)), name=series)


# ------------------------------------------------------------------ 信号（判断の時刻より前のデータだけ）

def available(x, bar: str = "1h"):
    """足の開始時刻の index を、その足が確定する時刻（開始 + 足の長さ）に移す。信号はすべて、分かる時刻を index にする。"""
    y = x.copy()
    y.index = y.index + pd.Timedelta(bar)
    return y


def trailing_rank(x: pd.Series, window: str = "180D", min_periods: int = 100) -> pd.Series:
    """各時点の値が、その時点より前の window の値の中で何割の位置か（0〜1）。自分自身は含めない（先読みなし）。"""
    vals = x.to_numpy(dtype=float)
    idx = x.index
    out = np.full(len(x), np.nan)
    left = idx.searchsorted(idx - pd.Timedelta(window), side="left")
    for i in range(len(x)):
        past = vals[left[i]:i]
        past = past[np.isfinite(past)]
        if len(past) >= min_periods and np.isfinite(vals[i]):
            out[i] = (np.sum(past < vals[i]) + 0.5 * np.sum(past == vals[i])) / len(past)
    return pd.Series(out, index=idx, name=f"{x.name}_rank")


def jpy_premium(bitbank_close: pd.Series, binance_close: pd.Series, usdjpy_close: pd.Series) -> pd.Series:
    """log(bitbank の BTC/JPY) − log(Binance の BTC/USDT × ドル円)。足の開始時刻でそろえ、ドル円は直前の値で埋める（週末は動かない）。"""
    fx = usdjpy_close.reindex(bitbank_close.index.union(usdjpy_close.index)).ffill().reindex(bitbank_close.index)
    return (np.log(bitbank_close) - np.log(binance_close.reindex(bitbank_close.index)) - np.log(fx)).rename("jpy_premium")
