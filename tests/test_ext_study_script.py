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
    if g.empty:                                                     # 合成データの期間の外の月は、実際のアーカイブと同じく 404
        return None
    rows = [f"{ms(str(t))},{v * scale},{v * scale * 1.001},{v * scale * 0.999},{v * scale},1,{ms(str(t)) + 3_599_999},1,1,1,1,0"
            for t, v in g.items()]
    return zip_csv("\n".join(rows) + "\n")


def bi5(m, side):
    g = FX[FX.index.strftime("%Y-%m") == m] + (0.005 if side == "ASK" else -0.005)
    base = pd.Timestamp(m + "-01", tz="UTC")
    raw = b"".join(struct.pack(">5if", int((t - base).total_seconds()), round(v * 1000), round(v * 1000), round(v * 1000) - 10,
                               round(v * 1000) + 10, 1.0) for t, v in g.items())
    return lzma.compress(raw, format=lzma.FORMAT_ALONE)


def found(body):
    return Resp(404) if body is None else Resp(200, body)


def respond(url):
    if m := re.search(r"/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-(\d{4}-\d{2})\.zip$", url):
        return found(kline_zip(USDT, m[1]))
    if m := re.search(r"/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-(\d{4}-\d{2})\.zip$", url):
        return found(kline_zip(USDT, m[1], 1.0005))
    if m := re.search(r"/premiumIndexKlines/BTCUSDT/1h/BTCUSDT-1h-(\d{4}-\d{2})\.zip$", url):
        return found(kline_zip(PREM * 0.1 + 1e-4, m[1]))
    if m := re.search(r"fundingRate/BTCUSDT/BTCUSDT-fundingRate-(\d{4}-\d{2})\.zip$", url):
        g = FUND[FUND.index.strftime("%Y-%m") == m[1]]
        if g.empty:
            return Resp(404)
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
    monkeypatch.setattr(xd.time, "sleep", lambda s: None)          # 窓口ごとの待ち時間を飛ばす
    return tmp_path


def test_check_runs_and_reports(run_dir):
    sess = Session(fn=respond)
    f = xd.Fetcher(run_dir / "cache", session=sess, pause=0)
    chk = es.main_check(CFG, fetcher=f, hourly=hourly)
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


def test_check_reports_unreachable_source(run_dir, monkeypatch):
    """ドル円の窓口が 503 を返し続けても check は最後まで進み、取れなかったファイルを報告に出す。"""
    sess = Session(fn=lambda url: Resp(503) if "dukascopy" in url else respond(url))
    f = xd.Fetcher(run_dir / "cache", session=sess, pause=0, retries=2, strict=False)
    rep = es.main_check(CFG, fetcher=f, hourly=hourly)
    assert len(rep["failures"]) == 12 and all("dukascopy" in x["url"] and x["status"] == "HTTP 503" for x in rep["failures"])
    cov = {r["name"]: r for r in rep["coverage"]}
    assert cov["Dukascopy ドル円 1 時間足（中値）"]["n"] == 0 and rep["fx_dukascopy_vs_fred"]["n"] == 0
    assert not any(r["signal"] == "jpy_premium" for r in rep["event_counts"])
    md = (run_dir / "reports/ext_study/check.md").read_text()
    assert "取れなかったファイル 12 件" in md and "USDJPY/2020/00/BID_candles_hour_1.bi5: HTTP 503" in md


# ------------------------------------------------------------------ eval（事前登録どおりの評価）

EVAL = {"grid_hours": [0, 8, 16], "horizon_hours": 24, "q": 0.10, "gap_hours": 24, "staleness_hours": {"funding": 8, "hourly": 2},
        "zbins": [-2, -1, 1, 2], "z_vol_hours": 240, "z_min_hours": 120, "cost": 0.003, "alpha": 0.05, "tail_ratio": 2.0,
        "tail_quantile": 0.05, "null_reps": 199, "min_shift_days": 10, "seed": 5,
        "har": {"windows": [1, 4, 24, 168], "calendar": False, "rv_floor": 1e-10, "min_train_days": 20, "quantiles": [0.05, 0.01]},
        "primary": [{"id": "H1a", "signal": "funding_binance", "side": "low", "kind": "mean", "direction": 1},
                    {"id": "H1b", "signal": "funding_binance", "side": "high", "kind": "mean", "direction": -1},
                    {"id": "H2a", "signal": "jpy_premium", "side": "low", "kind": "mean", "direction": 1},
                    {"id": "H2b", "signal": "jpy_premium", "side": "high", "kind": "mean", "direction": -1},
                    {"id": "H3a", "signal": "carry", "side": "high", "kind": "tail", "direction": 1},
                    {"id": "H3b", "signal": "oi_change", "side": "high", "kind": "tail", "direction": 1}],
        "venue_check": ["funding_bitmex", "funding_deribit"], "forward_months": 6}


def minute_closes(bb: pd.Series, seed: int = 4) -> pd.Series:
    """1 時間足の終値を分に広げ、小さな揺れを足した 1 分の終値（HAR 型の実現分散に使う）。"""
    idx = pd.date_range(START, END, freq="1min", tz="UTC", inclusive="left")
    base = bb.reindex(idx, method="ffill")
    return base * np.exp(np.random.default_rng(seed).normal(0, 2e-4, len(idx)))


def planted(cfg, bump=0.03):
    """Binance の資金調達率が下位 10% の判断の時刻（数えた事象）の後 24 時間に、bitbank の価格を bump だけ上げる。"""
    from bbresearch import eventstudy as ev
    f = xd.Fetcher(".cache/plant", session=Session(fn=respond), pause=0)
    d = es.load_all(cfg, f, hourly)
    E = cfg["eval"]
    times = ev.grid_times(cfg["start"], cfg["end"], E["grid_hours"], "24h")
    rank = ev.value_at(es.signals(d, cfg)["funding_binance"], times, "8h")
    sel = ev.select_episodes(ev.event_mask(rank, E["q"], "low"), times.as_unit("ns").asi8, 24 * 3_600_000_000_000)
    drift = pd.Series(0.0, index=H)
    for T in times[sel]:
        ramp = (H >= T) & (H < T + pd.Timedelta("24h"))
        drift[ramp] += bump * ((H[ramp] - T) / pd.Timedelta("1h") + 1) / 24     # 足 T の終値（T + 1h に分かる）から上がり始める
        drift[H >= T + pd.Timedelta("24h")] += bump
    return BB * np.exp(drift), int(sel.sum())


def test_eval_requires_owner_approval(run_dir):
    with pytest.raises(SystemExit, match="owner_approved"):
        es.main_eval(CFG | {"eval": EVAL})


def test_eval_finds_planted_effect_and_reports(run_dir):
    cfg = CFG | {"owner_approved": "2026-09-28", "eval": EVAL}
    bb2, n_sel = planted(cfg)

    def hourly2(start, end):
        idx = H[(H >= pd.Timestamp(start, tz="UTC")) & (H < pd.Timestamp(end, tz="UTC"))]
        c = bb2.reindex(idx)
        return pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1.0})
    f = xd.Fetcher(run_dir / "cache", session=Session(fn=respond), pause=0)
    m = es.main_eval(cfg, fetcher=f, hourly=hourly2, minute_close=minute_closes(bb2))
    P = {r["id"]: r for r in m["primary"]}
    assert set(P) == {"H1a", "H1b", "H2a", "H2b", "H3a", "H3b"} and P["H1a"]["n_ev"] == n_sel
    assert P["H1a"]["effect"] == pytest.approx(0.03, abs=0.01) and P["H1a"]["ev_mean"] > 0.003
    assert P["H1a"]["p_holm"] < 0.05 and P["H1a"]["venue_ok"] and P["H1a"]["supported"]
    assert all(P[k]["p_holm"] >= P[k]["p"] for k in P if np.isfinite(P[k]["p"]))
    assert P["H2a"]["n_ev"] == 0 and np.isnan(P["H2a"]["p_holm"]) and not P["H2a"]["supported"]   # 上げを足したので割安の事象がない
    assert P["H3a"]["kind"] == "tail" and 0 <= P["H3a"]["ev_mean"] <= 1
    assert m["diag"]["har_refits"] >= 2 and 0 < m["diag"]["tail_rate_all"] < 0.2
    md = (run_dir / "reports/ext_study/report.md").read_text()
    assert "結論: 主な検定 6 つのうち支持は" in md and "| H1a | 資金調達率（Binance） | 下位 10% |" in md and "| 支持 |" in md
    log = [json.loads(x) for x in (run_dir / "reports/experiments.jsonl").read_text().splitlines()]
    assert [x["key"] for x in log] == [f"ext_study/{k}" for k in ("H1a", "H1b", "H2a", "H2b", "H3a", "H3b")]
    assert json.loads((run_dir / "reports/ext_study/metrics.json").read_text())["owner_approved"] == "2026-09-28"


def test_eval_without_effect_supports_nothing(run_dir):
    cfg = CFG | {"owner_approved": "2026-09-28", "eval": EVAL}
    f = xd.Fetcher(run_dir / "cache", session=Session(fn=respond), pause=0)
    m = es.main_eval(cfg, fetcher=f, hourly=hourly, minute_close=minute_closes(BB))
    assert not any(r["supported"] for r in m["primary"])
    assert "結論: 主な検定 6 つのうち支持は 0 個（前向きの確認はしない）" in (run_dir / "reports/ext_study/report.md").read_text()


def test_minutes_summary():
    idx = pd.date_range("2020-01-01", periods=3 * 1440, freq="1min", tz="UTC")
    mc = pd.Series(100.0, index=idx)
    mc[1440:1440 * 2] = np.nan                                      # 2 日目は約定なし
    mc[1440 * 2 + 10:1440 * 2 + 70] = np.nan
    h = pd.DataFrame({"close": 100.0}, index=pd.date_range("2020-01-01", periods=72, freq="h", tz="UTC"))
    s = es.minutes_summary(mc, h)
    r = s["by_year"][0]
    assert r["days_without_trades"] == 1 and r["longest_gap_min"] == 1440 and r["share_with_trades"] == pytest.approx((2 * 1440 - 60) / (3 * 1440))
    assert s["hourly_close_match"][2020] == 1.0


# ------------------------------------------------------------------ forward（支持した仮説の前向きのドライラン）

FWD = {"hypotheses": ["H2a"], "start": "2020-05-01", "months": 2, "entry_window_min": 15, "fee_round_trip": 0.002,
       "quotes": "quotes.jsonl", "max_quote_delay_min": 60}


def synthetic_trades(bb: pd.Series) -> pd.DataFrame:
    """毎時 1 分に買いの約定（終値 × 1.0005）、2 分に売りの約定（終値 × 0.9995）。index の足の終値は次の正時に分かる。"""
    rows = []
    for i, (t, c) in enumerate(bb.items()):
        nxt = t + pd.Timedelta("1h")                                    # 足 t の終値が分かる時刻
        rows.append((2 * i, "buy", c * 1.0005, 0.01, int((nxt + pd.Timedelta("1min")).value // 10**6)))
        rows.append((2 * i + 1, "sell", c * 0.9995, 0.01, int((nxt + pd.Timedelta("2min")).value // 10**6)))
    return pd.DataFrame(rows, columns=["transaction_id", "side", "price", "amount", "executed_at"])


def test_forward_complete_period_with_trades_and_quotes(run_dir):
    cfg = CFG | {"owner_approved": "2026-09-28", "eval": EVAL, "forward": FWD}
    slots = pd.date_range("2020-05-01", "2020-07-01", freq="8h", tz="UTC", inclusive="left")
    with open("quotes.jsonl", "w") as fh:
        for i, s in enumerate(slots):
            c = float(BB.asof(s - pd.Timedelta("1h")))                 # s に分かっている終値
            delay = 90 if i % 10 == 0 else 5                            # 10 回に 1 回は遅すぎて使わない
            fh.write(json.dumps({"slot": s.isoformat(), "delay_min": delay, "sell": c * 1.001, "buy": c * 0.999}) + "\n")
    f = xd.Fetcher(run_dir / "cache", session=Session(fn=respond), pause=0)
    o = es.main_forward(cfg, fetcher=f, hourly=hourly, trades=synthetic_trades(BB), today=pd.Timestamp("2020-07-05", tz="UTC"))
    assert o["complete"] and o["end"].startswith("2020-07-01")
    h = o["hypotheses"][0]
    assert h["id"] == "H2a" and h["n_ev"] > 0 and h["criterion_met"] == (h["effect"] > 0 and h["ev_mean"] > 0.003)
    ex, qu = h["exec"], h["quotes"]
    assert ex["n"] == h["n_ev"] and ex["mean_exec"] == pytest.approx(ex["mean_close"] + np.log(0.9995 / 1.0005), abs=1e-9)
    assert 0 < qu["n"] < h["n_ev"] and qu["mean_quote"] == pytest.approx(qu["mean_close"] + np.log(0.999 / 1.001), abs=1e-9)
    assert ex["mean_exec_net"] == pytest.approx(ex["mean_exec"] - 0.002)
    md = (run_dir / "reports/ext_study/forward.md").read_text()
    assert "そろった。判定する" in md and "## H2a（JPY の内外価格差、下位 10%）" in md


def test_forward_interim_and_not_started(run_dir):
    cfg = CFG | {"owner_approved": "2026-09-28", "eval": EVAL, "forward": FWD}
    f = xd.Fetcher(run_dir / "cache", session=Session(fn=respond), pause=0)
    o = es.main_forward(cfg, fetcher=f, hourly=hourly, trades=synthetic_trades(BB), today=pd.Timestamp("2020-06-10", tz="UTC"))
    assert not o["complete"] and o["end"].startswith("2020-06-01")
    assert all(pd.Timestamp(e["t"]) < pd.Timestamp("2020-05-31", tz="UTC") for e in o["hypotheses"][0]["events"])
    assert "途中経過" in (run_dir / "reports/ext_study/forward.md").read_text()
    o2 = es.main_forward(cfg, fetcher=f, hourly=hourly, trades=synthetic_trades(BB), today=pd.Timestamp("2020-05-20", tz="UTC"))
    assert o2["hypotheses"] == [] and o2["note"] == "確定した月がまだない"
    with pytest.raises(SystemExit, match="owner_approved"):
        es.main_forward(CFG | {"eval": EVAL, "forward": FWD}, fetcher=f, hourly=hourly, trades=synthetic_trades(BB))


def test_first_taker_price_window():
    tr = pd.DataFrame({"side": ["sell", "buy", "buy"], "price": [99.0, 101.0, 102.0],
                       "executed_at": [ms("2020-01-01 00:00:30"), ms("2020-01-01 00:03"), ms("2020-01-01 00:40")]})
    t = pd.DatetimeIndex(pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:30", "2020-01-01 01:00"], utc=True))
    out = es.first_taker_price(tr, t, "buy", 15)
    assert out[0] == 101.0 and out[1] == 102.0 and np.isnan(out[2])


def test_forward_uses_only_needed_history_and_gives_same_result(run_dir, monkeypatch):
    """前向きの計算は窓の長さ分の過去だけを読む。全期間を読んだときと結果が同じ。"""
    cfg = CFG | {"owner_approved": "2026-09-28", "eval": EVAL, "forward": FWD, "window": "30D", "min_days": 10}
    today = pd.Timestamp("2020-07-05", tz="UTC")
    f = xd.Fetcher(run_dir / "cache", session=Session(fn=respond), pause=0)
    short = es.main_forward(cfg, fetcher=f, hourly=hourly, trades=synthetic_trades(BB), today=today)
    assert short["data_start"] == "2020-03-25"                        # 2020-05-01 − (30 日 + 7 日。z の分母 240 本 + 24 時間より長い)
    monkeypatch.setattr(es, "forward_lookback", lambda c: pd.Timedelta(days=120))   # 合成データの最初（2020-01-01）の近くから読む
    full = es.main_forward(cfg, fetcher=f, hourly=hourly, trades=synthetic_trades(BB), today=today)
    a, b = short["hypotheses"][0], full["hypotheses"][0]
    assert a["n_ev"] > 0 and a["events"] == b["events"] and a["effect"] == pytest.approx(b["effect"])


def test_forward_stops_when_needed_series_is_missing(run_dir):
    cfg = CFG | {"owner_approved": "2026-09-28", "eval": EVAL, "forward": FWD}
    sess = Session(fn=lambda url: Resp(503) if "dukascopy" in url and "/2020/04/" in url else respond(url))
    f = xd.Fetcher(run_dir / "cache", session=sess, pause=0, retries=1, strict=False)
    with pytest.raises(SystemExit, match="ドル円の 1 時間足"):
        es.main_forward(cfg, fetcher=f, hourly=hourly, trades=synthetic_trades(BB), today=pd.Timestamp("2020-07-05", tz="UTC"))
