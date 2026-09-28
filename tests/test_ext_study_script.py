"""scripts/ext_study.py を合成の外部データで最後まで通す（ネットワークにはアクセスしない）。"""
import json
import lzma
import re
import struct

import numpy as np
import pandas as pd
import pytest

import scripts.ext_study as es
from bbresearch import extdata as xd
from tests.test_extdata import Resp, Session, ms, zip_csv

START, END = "2020-01-01", "2020-07-01"
H = pd.date_range(START, END, freq="h", tz="UTC", inclusive="left")
RNG = np.random.default_rng(3)
USDT = pd.Series(7000 * np.exp(np.cumsum(RNG.normal(0, 0.006, len(H)))), index=H)
FX = pd.Series(108 + np.cumsum(RNG.normal(0, 0.02, len(H))), index=H)
PREM = pd.Series(np.zeros(len(H)), index=H)
for i in range(1, len(H)):
    PREM.iloc[i] = 0.98 * PREM.iloc[i - 1] + RNG.normal(0, 0.0005)
BB = USDT * FX * np.exp(PREM)
FUND = pd.Series(RNG.normal(1e-4, 1e-4, len(H[::8])), index=H[::8])
BITMEX = [{"timestamp": (t + pd.Timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%S.000Z"), "fundingRate": float(v)}
          for t, v in FUND.items()]
CFG = {"owner_approved": None, "pair": "btc_jpy", "binance_symbol": "BTCUSDT", "start": START, "end": END, "window": "60D",
       "min_days": 20, "check_thresholds": [0.05, 0.10], "metrics_start": START, "cache": ".cache/ext"}


def kline_zip(series, m, scale=1.0):
    g = series[series.index.strftime("%Y-%m") == m]
    rows = [f"{ms(str(t))},{v * scale},{v * scale * 1.001},{v * scale * 0.999},{v * scale},1,{ms(str(t)) + 3_599_999},1,1,1,1,0"
            for t, v in g.items()]
    return zip_csv("\n".join(rows) + "\n")


def bi5(m, side):
    g = FX[FX.index.strftime("%Y-%m") == m] + (0.005 if side == "ASK" else -0.005)
    base = pd.Timestamp(m + "-01", tz="UTC")
    raw = b"".join(struct.pack(">5if", int((t - base).total_seconds()), round(v * 1000), round(v * 1000), round(v * 1000) - 10,
                               round(v * 1000) + 10, 1.0) for t, v in g.items())
    return lzma.compress(raw, format=lzma.FORMAT_ALONE)


def respond(url):
    if m := re.search(r"/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-(\d{4}-\d{2})\.zip$", url):
        return Resp(200, kline_zip(USDT, m[1]))
    if m := re.search(r"/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-(\d{4}-\d{2})\.zip$", url):
        return Resp(200, kline_zip(USDT, m[1], 1.0005))
    if m := re.search(r"/premiumIndexKlines/BTCUSDT/1h/BTCUSDT-1h-(\d{4}-\d{2})\.zip$", url):
        return Resp(200, kline_zip(PREM * 0.1 + 1e-4, m[1]))
    if m := re.search(r"fundingRate/BTCUSDT/BTCUSDT-fundingRate-(\d{4}-\d{2})\.zip$", url):
        g = FUND[FUND.index.strftime("%Y-%m") == m[1]]
        return Resp(200, zip_csv("calc_time,funding_interval_hours,last_funding_rate\n" +
                                 "".join(f"{ms(str(t)) + 3},8,{v}\n" for t, v in g.items())))
    if m := re.search(r"metrics/BTCUSDT/BTCUSDT-metrics-(\d{4}-\d{2}-\d{2})\.zip$", url):
        day = pd.Timestamp(m[1], tz="UTC")
        if day < pd.Timestamp("2020-02-01", tz="UTC"):
            return Resp(404)
        t = pd.date_range(day, periods=288, freq="5min")
        oi = 1e4 * np.exp(np.cumsum(RNG.normal(0, 0.001, 288))) * (1 + (day.dayofyear % 30) / 100)
        return Resp(200, zip_csv(",".join(xd.METRICS_COLS) + "\n" +
                                 "".join(f"{a:%Y-%m-%d %H:%M:%S},BTCUSDT,{b},{b * 7000},1,1,1,1\n" for a, b in zip(t, oi))))
    if "bitmex.com/api/v1/funding" in url:
        s = pd.Timestamp(re.search(r"startTime=([^&]+)", url)[1])
        e = pd.Timestamp(re.search(r"endTime=([^&]+)", url)[1])
        rows = [r for r in BITMEX if s <= pd.Timestamp(r["timestamp"]) < e][:500]
        return Resp(200, json.dumps(rows).encode())
    if "deribit.com" in url:
        s = int(re.search(r"start_timestamp=(\d+)", url)[1])
        e = int(re.search(r"end_timestamp=(\d+)", url)[1])
        rows = [{"timestamp": ms(str(t)), "interest_8h": float(FUND.asof(t)), "interest_1h": float(FUND.asof(t)) / 8}
                for t in H if s < ms(str(t)) <= e]
        return Resp(200, json.dumps({"result": rows}).encode())
    if m := re.search(r"dukascopy.com/datafeed/USDJPY/(\d{4})/(\d{2})/(BID|ASK)_candles_hour_1\.bi5$", url):
        return Resp(200, bi5(f"{m[1]}-{int(m[2]) + 1:02d}", m[3]))
    if "fredgraph.csv?id=DEXJPUS" in url:
        days = pd.bdate_range(START, END, inclusive="left")
        vals = [FX.asof(pd.Timestamp(d.date()).tz_localize("America/New_York").tz_convert("UTC") + pd.Timedelta(hours=11)) for d in days]  # 正午に閉じる足
        return Resp(200, ("observation_date,DEXJPUS\n" + "".join(f"{d:%Y-%m-%d},{v:.4f}\n" for d, v in zip(days, vals))).encode())
    raise AssertionError(url)


def hourly(start, end):
    idx = H[(H >= pd.Timestamp(start, tz="UTC")) & (H < pd.Timestamp(end, tz="UTC"))]
    c = BB.reindex(idx)
    return pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1.0})


@pytest.fixture()
def run_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_check_runs_and_reports(run_dir):
    sess = Session(fn=respond)
    f = xd.Fetcher(run_dir / "cache", session=sess, pause=0)
    xd_bitmex = xd.bitmex_funding
    try:
        xd.bitmex_funding = lambda fetcher, s, e: xd_bitmex(fetcher, s, e, pause=0)
        chk = es.main_check(CFG, fetcher=f, hourly=hourly)
    finally:
        xd.bitmex_funding = xd_bitmex
    rep = json.loads((run_dir / "reports/ext_study/check.json").read_text())
    md = (run_dir / "reports/ext_study/check.md").read_text()
    cov = {r["name"]: r for r in rep["coverage"]}
    assert cov["bitbank BTC/JPY 1 時間足"]["missing"] == 0 and cov["Binance 現物 BTCUSDT 1 時間足"]["missing"] == 0
    assert cov["Binance 資金調達率（0・8・16 時）"]["missing"] == 0 and cov["BitMEX 資金調達率（4・12・20 時）"]["missing"] == 0
    assert cov["Dukascopy ドル円 1 時間足（中値）"]["missing"] == 0
    assert cov["Deribit 資金調達率（1 時間ごと）"]["missing"] == 1        # 最初の 00:00 は窓口が返さない
    metrics = [r for r in rep["coverage"] if r["name"].startswith("Binance 建玉など")][0]
    assert metrics["first"].startswith("2020-02-01 00:00") and metrics["missing"] == 31 * 288
    assert rep["funding_binance_gaps"] == {"8h": len(FUND) - 1}
    assert {r["signal"] for r in rep["event_counts"]} == {"funding_binance", "funding_bitmex", "funding_deribit", "jpy_premium",
                                                           "carry", "oi_change"}
    assert all(r["low0.1_ep"] <= r["low0.1"] and r["low0.05"] <= r["low0.1"] for r in rep["event_counts"])
    assert rep["fx_dukascopy_vs_fred"]["n"] > 100 and rep["fx_dukascopy_vs_fred"]["median_abs_diff"] < 1e-4
    assert rep["deribit_interest"]["corr_8x1h"] == pytest.approx(1.0)
    assert rep["funding_corr_daily"]["binance"]["bitmex"] > 0.9
    assert "## 事象の数" in md and "| funding_binance | 2020 |" in md and chk["requests"] == f.n_requests
    n = len(sess.calls)
    es.main_check(CFG, fetcher=xd.Fetcher(run_dir / "cache", session=sess, pause=0), hourly=hourly)
    assert len(sess.calls) - n <= 2                                   # 2 回目は FRED と BitMEX の未確定の年だけ（2020 年は確定）


def test_signals_use_only_data_known_at_their_time(run_dir):
    f = xd.Fetcher(run_dir / "cache", session=Session(fn=respond), pause=0)
    d = es.load_all(CFG, f, hourly)
    T = pd.Timestamp("2020-05-10 08:00", tz="UTC")
    rng = np.random.default_rng(9)

    def scramble(x, strict):
        x = x.copy()
        m = (x.index > T) if strict else (x.index >= T)            # 1 時間足は開始時刻 ≥ T、決済の値は時刻 > T を書き換える
        if isinstance(x, pd.DataFrame):
            num = x.select_dtypes("number").columns
            x.loc[m, num] = x.loc[m, num].to_numpy() * rng.uniform(0.5, 1.5, (int(m.sum()), len(num)))
        else:
            x[m] = x[m].to_numpy() * rng.uniform(0.5, 1.5, int(m.sum())) - 0.001
        return x
    d2 = {k: scramble(v, strict=k.startswith("funding_")) if k != "fred" else v for k, v in d.items()}
    d2["metrics"] = scramble(d["metrics"], strict=False)
    a, b = es.signals(d, CFG), es.signals(d2, CFG)
    for name in a:
        before = a[name].index <= T
        assert before.sum() > 100, name
        pd.testing.assert_series_equal(a[name][before], b[name][b[name].index <= T], obj=name)
        assert not a[name][~before].equals(b[name][b[name].index > T]), name   # 書き換えは後の値には効いている
    # 信号の時刻は値が分かる時刻: 1 時間足から作るものは足が閉じる時刻（正時）、資金調達率は決済の時刻
    assert (a["jpy_premium"].index.minute == 0).all() and a["jpy_premium"].dropna().index[0] > pd.Timestamp(START, tz="UTC")
    assert set(a["funding_binance"].index.hour) == {0, 8, 16} and set(a["funding_deribit"].index.hour) == {0, 8, 16}


def test_main_reads_config(run_dir, monkeypatch):
    called = {}
    monkeypatch.setattr(es, "main_check", lambda cfg: called.setdefault("cfg", cfg))
    (run_dir / "c.yaml").write_text("start: '2020-01-01'\nend: '2020-02-01'\n")
    assert es.main(["--mode", "check", "--config", str(run_dir / "c.yaml")]) == 0
    assert called["cfg"]["start"] == "2020-01-01"
