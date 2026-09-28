"""bbresearch/extdata.py: 外部データの読み取りと信号（ネットワークにはアクセスしない）。"""
import io
import json
import lzma
import struct
import zipfile

import numpy as np
import pandas as pd
import pytest

from bbresearch import extdata as xd


def zip_csv(text: str, name: str = "x.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, text)
    return buf.getvalue()


def ms(ts: str) -> int:
    return int(pd.Timestamp(ts, tz="UTC").value // 10**6)


class Resp:
    def __init__(self, status=200, content=b""):
        self.status_code, self.content = status, content
        self.text = content.decode("latin1")


class Session:
    """URL → 応答の表で答える。表にない URL は 404。"""

    def __init__(self, table=None, fn=None):
        self.headers, self.table, self.fn, self.calls = {}, table or {}, fn, []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if self.fn:
            r = self.fn(url)
            if r is not None:
                return r
        v = self.table.get(url)
        if v is None:
            return Resp(404)
        return v if isinstance(v, Resp) else Resp(200, v)


def kline_rows(start: str, n: int, price0: float = 100.0, us: bool = False) -> list[str]:
    rows = []
    for i in range(n):
        t = ms(start) + i * 3_600_000
        t_out = t * 1000 if us else t
        p = price0 + i
        rows.append(f"{t_out},{p},{p + 1},{p - 1},{p + 0.5},10,{t_out + 3_599_999},1000,5,4,400,0")
    return rows


def test_read_zip_csv_header_detection():
    with_header = zip_csv("open_time,open,high,low,close,volume\n1577836800000,1,2,0.5,1.5,10\n")
    no_header = zip_csv("1577836800000,1,2,0.5,1.5,10\n1577840400000,1.5,2,1,1.8,11\n")
    date_first = zip_csv("2020-09-01 00:05:00,BTCUSDT,100,1000000\n2020-09-01 00:10:00,BTCUSDT,101,1010000\n")
    a = xd.read_zip_csv(with_header, ["t", "o", "h", "l", "c", "v"])
    assert list(a.columns) == ["open_time", "open", "high", "low", "close", "volume"] and len(a) == 1
    b = xd.read_zip_csv(no_header, ["t", "o", "h", "l", "c", "v"])
    assert list(b.columns) == ["t", "o", "h", "l", "c", "v"] and len(b) == 2
    c = xd.read_zip_csv(with_header, ["t", "o", "h", "l", "c", "v"], positional=True)
    assert list(c.columns) == ["t", "o", "h", "l", "c", "v"] and len(c) == 1
    d = xd.read_zip_csv(date_first, xd.METRICS_COLS)          # 日時で始まる行は見出しではない
    assert len(d) == 2 and d["sum_open_interest"].tolist() == [100, 101]


def test_ms_to_utc_handles_microseconds():
    idx = xd._ms_to_utc(pd.Series([ms("2025-01-01"), ms("2025-01-01 01:00") * 1000]))
    assert list(idx) == [pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-01-01 01:00", tz="UTC")]


def test_months_and_final():
    assert xd.months("2020-01-01", "2020-04-01") == ["2020-01", "2020-02", "2020-03"]
    assert xd.months("2020-01-15", "2020-02-01") == ["2020-01"]
    today = pd.Timestamp("2026-09-02").date()
    assert xd.month_is_final("2026-07", today) and not xd.month_is_final("2026-08", today)
    assert xd.month_is_final("2026-08", pd.Timestamp("2026-09-03").date())


def test_binance_klines_mixed_formats_cache_and_missing_month(tmp_path):
    base = f"{xd.VISION}/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-"
    jan = zip_csv("\n".join(kline_rows("2020-01-31 22:00", 2)) + "\n")
    feb = zip_csv(",".join(xd.KLINE_COLS) + "\n" + "\n".join(kline_rows("2020-02-01", 3, 200, us=True)) + "\n")
    sess = Session({base + "2020-01.zip": jan, base + "2020-02.zip": feb})
    f = xd.Fetcher(tmp_path, session=sess, pause=0)
    out = xd.binance_klines(f, "spot", "klines", "BTCUSDT", "1h", "2020-01-31 23:00", "2020-04-01")
    assert list(out.index) == [pd.Timestamp("2020-01-31 23:00", tz="UTC")] + list(pd.date_range("2020-02-01", periods=3, freq="h", tz="UTC"))
    assert out["close"].tolist() == [101.5, 200.5, 201.5, 202.5] and list(out.columns) == ["open", "high", "low", "close", "volume"]
    n = len(sess.calls)
    assert n == 3                                                  # 2020-03 は 404
    again = xd.binance_klines(f, "spot", "klines", "BTCUSDT", "1h", "2020-01-31 23:00", "2020-04-01")
    assert len(sess.calls) == n and again.equals(out)             # 確定した月（404 も）は取り直さない


def test_fetcher_retries_and_errors(tmp_path):
    seq = [Resp(503), Resp(429), Resp(200, b"ok")]
    sess = Session(fn=lambda url: seq.pop(0) if "retry" in url else Resp(403, b"blocked"))
    f = xd.Fetcher(tmp_path, session=sess, pause=0)
    xd.time.sleep, orig = (lambda s: None), xd.time.sleep
    try:
        assert f.get("https://x/retry", "a.bin") == b"ok" and f.n_requests == 3
        with pytest.raises(RuntimeError, match="HTTP 403"):
            f.get("https://x/forbidden", "b.bin")
    finally:
        xd.time.sleep = orig
    sess2 = Session()
    f2 = xd.Fetcher(tmp_path / "c", session=sess2, pause=0)
    assert f2.get("https://x/missing", "m.bin", final=False) is None
    assert f2.get("https://x/missing", "m.bin", final=False) is None and len(sess2.calls) == 2   # 確定していなければ 404 も取り直す
    assert not (tmp_path / "c" / "m.bin.404").exists()


def test_binance_funding_floors_to_minute(tmp_path):
    url = f"{xd.VISION}/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2020-01.zip"
    text = "calc_time,funding_interval_hours,last_funding_rate\n" + "\n".join(
        f"{ms('2020-01-01') + i * 8 * 3_600_000 + 7},8,{0.0001 * (i + 1)}" for i in range(3)) + "\n"
    f = xd.Fetcher(tmp_path, session=Session({url: zip_csv(text)}), pause=0)
    s = xd.binance_funding(f, "BTCUSDT", "2020-01-01", "2020-02-01")
    assert list(s.index) == list(pd.date_range("2020-01-01", periods=3, freq="8h", tz="UTC"))
    assert s.name == "binance" and np.allclose(s.to_numpy(), [0.0001, 0.0002, 0.0003])


def test_binance_metrics_daily(tmp_path):
    def day(d):
        return f"{xd.VISION}/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-{d}.zip"
    head = ",".join(xd.METRICS_COLS) + "\n"
    rows = lambda d, v: "".join(f"{d} 00:{m:02d}:00,BTCUSDT,{v + m},{(v + m) * 10},1,1,1,1\n" for m in (5, 10))  # noqa: E731
    sess = Session({day("2020-09-01"): zip_csv(head + rows("2020-09-01", 100)), day("2020-09-03"): zip_csv(rows("2020-09-03", 200))})
    f = xd.Fetcher(tmp_path, session=sess, pause=0)
    m = xd.binance_metrics(f, "BTCUSDT", "2020-09-01", "2020-09-04")
    assert len(sess.calls) == 3 and len(m) == 4                    # 2020-09-02 は 404、2020-09-03 は見出しなし
    assert m["sum_open_interest"].tolist() == [105, 110, 205, 210]
    assert m.index[0] == pd.Timestamp("2020-09-01 00:05", tz="UTC")


def test_bitmex_pagination(tmp_path):
    def rows(t0, n):
        return [{"timestamp": (t0 + pd.Timedelta(hours=8 * i)).strftime("%Y-%m-%dT%H:%M:%S.000Z"), "fundingRate": 0.0001}
                for i in range(n)]
    first = rows(pd.Timestamp("2020-01-01 04:00"), 500)
    second = rows(pd.Timestamp("2020-01-01 04:00") + pd.Timedelta(hours=8 * 500), 10)

    def fn(url):
        if "startTime=2020-01-01T00:00:00Z" in url:
            return Resp(200, json.dumps(first).encode())
        nxt = (pd.Timestamp(first[-1]["timestamp"]) + pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        if f"startTime={nxt}" in url:
            return Resp(200, json.dumps(second).encode())
        raise AssertionError(url)
    sess = Session(fn=fn)
    f = xd.Fetcher(tmp_path, session=sess, pause=0)
    s = xd.bitmex_funding(f, "2020-01-01", "2021-01-01", pause=0)
    assert len(sess.calls) == 2 and len(s) == 510 and s.index[0] == pd.Timestamp("2020-01-01 04:00", tz="UTC")
    assert s.name == "bitmex" and s.index.is_monotonic_increasing
    xd.bitmex_funding(f, "2020-01-01", "2021-01-01", pause=0)
    assert len(sess.calls) == 2                                    # 確定した年はためたものを使う


def test_deribit_month_and_continuation(tmp_path):
    s0, s1 = ms("2020-01-01"), ms("2020-02-01")
    hours = [s0 + i * 3_600_000 for i in range(1, 745)]            # 01:00 〜 2 月 1 日 00:00
    part1, part2 = hours[:500], hours[500:]

    def rec(t):
        return {"timestamp": t, "interest_8h": 0.0001, "interest_1h": 0.0001 / 8, "index_price": 7000, "prev_index_price": 7000}

    def fn(url):
        if f"start_timestamp={s0}&end_timestamp={s1}" in url:
            return Resp(200, json.dumps({"result": [rec(t) for t in part1]}).encode())      # 上限で途中までしか返らない場合
        if f"start_timestamp={part1[-1] + 1}&end_timestamp={s1}" in url:
            return Resp(200, json.dumps({"result": [rec(t) for t in part2]}).encode())
        raise AssertionError(url)
    sess = Session(fn=fn)
    f = xd.Fetcher(tmp_path, session=sess, pause=0)
    d = xd.deribit_funding(f, "2020-01-01", "2020-02-01")
    assert len(sess.calls) == 2 and len(d) == 743                   # 2 月 1 日 00:00 は期間の外
    assert list(d.columns) == ["interest_8h", "interest_1h"] and d.index[0] == pd.Timestamp("2020-01-01 01:00", tz="UTC")


def test_dukascopy_mid_and_bi5(tmp_path):
    def bi5(prices, extra=0):
        raw = b"".join(struct.pack(">5if", i * 3600, p, p + extra, p - 50, p + 50, 1.0) for i, p in enumerate(prices))
        return lzma.compress(raw, format=lzma.FORMAT_ALONE)
    base = "https://datafeed.dukascopy.com/datafeed/USDJPY/2020/00/"
    sess = Session({base + "BID_candles_hour_1.bi5": bi5([108000, 108100]), base + "ASK_candles_hour_1.bi5": bi5([108010, 108110])})
    f = xd.Fetcher(tmp_path, session=sess, pause=0)
    fx = xd.dukascopy_hourly(f, "2020-01-01", "2020-02-01")
    assert list(fx.index) == list(pd.date_range("2020-01-01", periods=2, freq="h", tz="UTC"))
    assert np.allclose(fx["close"], [108.005, 108.105]) and np.allclose(fx["high"], [108.055, 108.155])
    c = xd.decode_bi5_candles(bi5([108631], extra=5), pd.Timestamp("2020-01-01", tz="UTC"))
    assert c.iloc[0].to_dict() == {"open": 108.631, "close": 108.636, "low": 108.581, "high": 108.681}
    assert xd.decode_bi5_candles(b"", pd.Timestamp("2020-01-01", tz="UTC")).empty


def test_fred_skips_missing(tmp_path):
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXJPUS"
    sess = Session({url: b"observation_date,DEXJPUS\n2020-01-01,\n2020-01-02,108.5\n2020-01-03,.\n2020-01-06,108.1\n"})
    s = xd.fred_series(xd.Fetcher(tmp_path, session=sess, pause=0))
    assert s.tolist() == [108.5, 108.1] and s.index[0] == pd.Timestamp("2020-01-02", tz="UTC")


def test_trailing_rank_excludes_current_and_future():
    idx = pd.date_range("2020-01-01", periods=400, freq="8h", tz="UTC")
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=400), index=idx, name="f")
    r = xd.trailing_rank(x, "30D", min_periods=50)
    assert r.iloc[:50].isna().all() and r.iloc[50:].notna().all()
    i = 300
    past = x.iloc[:i][x.index[:i] >= x.index[i] - pd.Timedelta("30D")]
    assert r.iloc[i] == pytest.approx((past < x.iloc[i]).mean())
    y = x.copy()
    y.iloc[i + 1:] = 100.0                                         # 先の値を書き換えても、それまでの順位は変わらない
    assert r.iloc[:i + 1].equals(xd.trailing_rank(y, "30D", min_periods=50).iloc[:i + 1])
    z = pd.Series([1.0] * 5 + [1.0], index=idx[:6])
    assert xd.trailing_rank(z, "30D", min_periods=5).iloc[-1] == 0.5  # 同じ値は半分として数える


def test_available_and_jpy_premium():
    idx = pd.date_range("2020-01-03 22:00", periods=4, freq="h", tz="UTC")
    bb = pd.Series([11_000_000.0, 11_100_000, 11_200_000, 11_300_000], index=idx)
    bn = pd.Series([100_000.0] * 4, index=idx)
    fx = pd.Series([110.0, 109.0], index=idx[:2])                   # 週末でドル円の足がない時刻は直前の値
    p = xd.jpy_premium(bb, bn, fx)
    assert np.allclose(p.to_numpy(), np.log(bb.to_numpy() / (100_000.0 * np.array([110.0, 109.0, 109.0, 109.0]))))
    assert list(xd.available(p).index) == list(idx + pd.Timedelta("1h"))
