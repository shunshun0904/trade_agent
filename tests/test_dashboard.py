"""ダッシュボード（dashboard/app）のテスト。ネットワークと S3 には触れない。"""
import json
import sys
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app"
sys.path.insert(0, str(APP))

import app as dash  # noqa: E402
import profile_core as core  # noqa: E402

from bbresearch.labeling import TradeTape  # noqa: E402
from bbresearch.profile import profile_features_at  # noqa: E402
from bbresearch.profile import value_area as value_area_np  # noqa: E402

H = 3_600_000


def synth(n=20000, hours=50, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.integers(0, hours * H, n)).astype("int64") + 1_767_225_600_000  # 2026-01-01
    px = np.round(10_000_000 * np.exp(np.cumsum(rng.normal(0, 0.0008, n))))
    amt = np.round(rng.uniform(0.0001, 0.5, n), 4)
    return ts, px, amt


def test_value_area_matches_research_implementation():
    rng = np.random.default_rng(1)
    for _ in range(50):
        c = rng.integers(0, 20, rng.integers(1, 30)).astype(float)
        if c.sum() == 0:
            continue
        assert core.value_area(list(c)) == value_area_np(c)


def test_profile_matches_research_implementation():
    ts, px, amt = synth()
    now = int(ts[0] + 49 * H) // core.BAR_MS * core.BAR_MS  # 足の境界
    trades = list(zip(ts.tolist(), px.tolist(), amt.tolist()))
    res = core.compute(trades, now, window_h=24.0, candle_ms=core.BAR_MS, tpo_block=2, sigma_n=96)  # 研究用と同じ定義
    ref, sigma = res["price"], res["sigma"]
    # 研究用の実装に同じ σ と基準価格を渡して比べる（研究用は σ 単位の距離を返す）
    idx = pd.date_range(pd.Timestamp(ts[0] // core.BAR_MS * core.BAR_MS, unit="ms", tz="UTC"),
                        pd.Timestamp(now, unit="ms", tz="UTC"), freq="15min", inclusive="left")
    bars = pd.DataFrame(core.bars_15m(trades, int(idx[0].value // 10**6), now)).set_index(idx)
    bars = bars.rename(columns={"h": "high", "l": "low", "c": "close"})
    tape = TradeTape(ts, np.zeros(len(ts), bool), px.astype(float), amt)
    f = profile_features_at(tape, bars, pd.Timestamp(now, unit="ms", tz="UTC"), ref, sigma, 1.0, 1.0)
    unit = sigma * ref
    for key, lv in (("vp", res["vp_levels"]), ("tpo", res["tpo_levels"])):
        assert f[f"{key}_poc_dist"] == pytest.approx((lv["poc"] - ref) / unit, abs=1e-6)
        assert f[f"{key}_vah_dist"] == pytest.approx((lv["vah"] - ref) / unit, abs=1e-6)
        assert f[f"{key}_val_dist"] == pytest.approx((lv["val"] - ref) / unit, abs=1e-6)
    assert res["vp_levels"]["val"] < res["vp_levels"]["poc"] < res["vp_levels"]["vah"]


def test_compute_three_hours_matches_brute_force():
    ts, px, amt = synth(seed=9)
    now = int(ts[0] + 49 * H) + 37_000
    trades = list(zip(ts.tolist(), px.tolist(), amt.tolist()))
    res = core.compute(trades, now)
    assert res["window_h"] == 3.0 and res["candle_ms"] == 60_000 and res["tpo_block_min"] == 5 and res["sigma_n"] == 180
    # σ は直近 180 本の 1 分足の対数リターンの標準偏差
    m1 = now // 60_000 * 60_000
    closes = []
    for k in range(181, 0, -1):
        sel = [p for t, p, a in trades if m1 - k * 60_000 <= t < m1 - (k - 1) * 60_000]
        closes.append(sel[-1] if sel else closes[-1])
    r = np.diff(np.log(closes))
    assert res["sigma"] == pytest.approx(r.std(ddof=1), rel=1e-9)
    assert 180 <= len(res["candles"]) <= 181 and res["candles"][0][0] >= now - 3 * H  # 3 時間 + 形成中の 1 分
    w, ref = res["bin_width"], res["price"]
    # 価格帯別出来高: 3 時間の約定を刻みで数え直す
    acc = {}
    for t, p, a in trades:
        if now - 3 * H <= t < now:
            k = int(np.floor(round((p - ref) / w, 9)))
            acc[k] = acc.get(k, 0.0) + a
    assert sum(r["v"] for r in res["vp"]) == pytest.approx(sum(acc.values()))
    assert all(r["v"] == pytest.approx(acc.get(int(np.floor(round((r["lo"] - ref) / w, 9))), 0.0)) for r in res["vp"])
    # TPO: 確定した 1 分足を now から過去へ 5 本ずつ（36 区間）
    m1 = now // 60_000 * 60_000
    blocks = 0
    for k in range(36):
        b_end = m1 - k * 5 * 60_000
        sel = [(p) for t, p, a in trades if b_end - 5 * 60_000 <= t < b_end]
        if sel:
            blocks += 1
    assert sum(r["n"] for r in res["tpo"]) >= blocks  # 各区間は少なくとも 1 価格帯に 1 を足す
    assert res["vp_levels"]["val"] <= res["vp_levels"]["poc"] <= res["vp_levels"]["vah"]


def test_compute_ignores_trades_at_or_after_now():
    ts, px, amt = synth(seed=2)
    now = int(ts[0] + 49 * H)
    base = list(zip(ts.tolist(), px.tolist(), amt.tolist()))
    later = [(t, p * 3 if t >= now else p, a * 50 if t >= now else a) for t, p, a in base]
    a, b = core.compute(base, now), core.compute(later, now)
    for k in ("vp_levels", "tpo_levels", "price", "sigma"):
        assert a[k] == b[k]


class FakeStore:
    def __init__(self, data=None):
        self.data = data

    def load(self):
        return self.data

    def save(self, data):
        self.data = json.loads(json.dumps(data))


def tx(i, ms, px=10_000_000.0, amt=0.01):
    return {"transaction_id": i, "side": "buy", "price": str(px), "amount": str(amt), "executed_at": ms}


def test_refresh_fetches_days_first_then_latest_only():
    now = 1_767_225_600_000 + 60 * H  # 2026-01-03 12:00 UTC
    calls = []

    def get(path):
        calls.append(path)
        if path.endswith("/transactions"):
            return {"transactions": [tx(1000 + k, now - 1000 * k) for k in range(60)]}
        return {"transactions": [tx(int(path[-2:]) * 10, now - 5 * H)]}

    c = dash.refresh(None, now, get)
    assert sorted(calls) == ["/btc_jpy/transactions", "/btc_jpy/transactions/20260101",
                             "/btc_jpy/transactions/20260102", "/btc_jpy/transactions/20260103"]
    assert c["skipped"] == []
    calls.clear()
    c2 = dash.refresh(c, now + 5_000, get)       # 10 秒以内は取りに行かない
    assert calls == [] and c2 is c
    def get_far(path):  # 60 件がすべて保存済みの最新より新しい（取りこぼしの可能性）→ 当日を取り直す
        calls.append(path)
        if path.endswith("/transactions"):
            return {"transactions": [tx(5000 + k, now + 60_000 - 100 * k) for k in range(60)]}
        return {"transactions": []}

    dash.refresh(c, now + 60_000, get_far)
    assert "/btc_jpy/transactions" in calls and "/btc_jpy/transactions/20260103" in calls


def test_refresh_skips_days_without_data_and_reports_them():
    now = 1_767_225_600_000 + 60 * H  # 2026-01-03 12:00 UTC

    def get(path):
        if path.endswith("/transactions"):
            return {"transactions": [tx(1000 + k, now - 1000 * k) for k in range(60)]}
        if path.endswith("20260103"):  # 当日分がまだない
            raise dash.NoData(f"{path}: HTTP 404")
        return {"transactions": [tx(int(path[-2:]) * 10, now - 5 * H)]}

    c = dash.refresh(None, now, get)
    assert len(c["skipped"]) == 1 and "20260103" in c["skipped"][0]
    assert len(c["rows"]) == 62  # 2 日分 + 最新 60 件
    # HTTP 404 以外の失敗は上に伝える
    def boom(path):
        raise urllib.error.HTTPError(path, 500, "server error", {}, None)

    with pytest.raises(urllib.error.HTTPError):
        dash.refresh(None, now, boom)


def test_http_get_json_turns_404_into_nodata(monkeypatch):
    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"success": 0, "data": {"code": 10000}}).encode()

    def open404(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 404, "not found", {}, None)

    monkeypatch.setattr(dash.urllib.request, "urlopen", open404)
    with pytest.raises(dash.NoData):
        dash.http_get_json("/btc_jpy/transactions/20991231")
    monkeypatch.setattr(dash.urllib.request, "urlopen", lambda req, timeout: Resp())
    with pytest.raises(dash.NoData):
        dash.http_get_json("/btc_jpy/transactions/20991231")


def test_refresh_incremental_without_gap():
    now = 1_767_225_600_000 + 60 * H
    rows = [[i, now - 1000 * (100 - i), 1.0, 0.1] for i in range(100)]
    cache = {"rows": rows, "from_ms": now - dash.KEEP_MS, "fetched_ms": now - 60_000}
    calls = []

    def get(path):
        calls.append(path)
        return {"transactions": [tx(i, now - 1000 * (100 - i)) for i in range(90, 105)]}

    c = dash.refresh(cache, now, get)
    assert calls == ["/btc_jpy/transactions"]
    assert [r[0] for r in c["rows"]][-5:] == [100, 101, 102, 103, 104]


def test_handler_routes_and_errors():
    page = dash.handler({"path": "/"}, None, store=FakeStore())
    assert page["statusCode"] == 200 and page["headers"]["Content-Type"].startswith("text/html")
    assert "api/profile" in page["body"]

    def boom(path):
        raise RuntimeError("down")

    err = dash.handler({"path": "/api/profile"}, None, store=FakeStore(), get=boom, now_ms=1_767_225_600_000)
    assert err["statusCode"] == 502 and "down" in json.loads(err["body"])["error"]


def test_handler_profile_end_to_end():
    ts, px, amt = synth(n=8000, seed=3)
    now = int(ts[-1]) + 1
    rows = [[i, int(t), float(p), float(a)] for i, (t, p, a) in enumerate(zip(ts, px, amt))]
    store = FakeStore({"rows": rows, "from_ms": now - dash.KEEP_MS, "fetched_ms": now})
    out = dash.handler({"path": "/api/profile"}, None, store=store, now_ms=now)
    body = json.loads(out["body"])
    assert out["statusCode"] == 200, body
    assert body["vp"] and body["tpo"] and body["candles"] and body["pair"] == "btc_jpy"


def test_page_gets_refresh_interval_and_pauses_when_hidden():
    assert "__REFRESH_SECONDS__" not in dash.PAGE
    import re

    # React 版（dashboard/web）が置き換え用に埋めた文字列 Number("__REFRESH_SECONDS__") || 10（縮小後は +"..."||10）
    assert re.search(rf'"{dash.REFRESH_SECONDS}"\s*\)?\s*\|\|\s*30', dash.PAGE)
    assert "visibilitychange" in dash.PAGE and "api/signals" in dash.PAGE


def test_memory_cache_limits_s3_writes(monkeypatch):
    ts, px, amt = synth(n=6000, seed=4)
    now = int(ts[-1]) + 1
    rows = [[i, int(t), float(p), float(a)] for i, (t, p, a) in enumerate(zip(ts, px, amt))]
    saves = []

    class CountingStore(FakeStore):
        def save(self, data):
            saves.append(data["fetched_ms"])
            super().save(data)

    store = CountingStore({"rows": rows, "from_ms": now - dash.KEEP_MS, "fetched_ms": now - 60_000})
    monkeypatch.setattr(dash, "_MEM", {})
    monkeypatch.setattr(dash, "Store", lambda bucket: store)
    latest = lambda path: {"transactions": [tx(len(rows) - 1, int(ts[-1]))]}  # noqa: E731
    for k in range(7):  # 10 秒ごとに 7 回
        out = dash.handler({"path": "/api/profile"}, None, get=latest, now_ms=now + k * 10_000)
        assert out["statusCode"] == 200
    assert len(saves) == 2  # 最初と 60 秒後だけ


def test_signal_at_matches_profile_definitions():
    ts, px, amt = synth(n=20000, seed=6)
    trades = [(int(t), float(p), float(a)) for t, p, a in zip(ts, px, amt)]
    now = int(ts[-1]) + 1
    t = now // core.MIN_MS * core.MIN_MS - 7 * core.MIN_MS
    end = now // core.BAR_MS * core.BAR_MS + core.BAR_MS
    bars1 = core.bars_at(trades, t - 4 * H, t + core.MIN_MS, core.CANDLE_MS)
    row = core.signal_at(trades, bars1, t)
    # 同じ時刻の compute（画面の左の列）と同じ水準になる
    ref = core.compute(trades, t)
    assert row["price"] == ref["price"] and abs(row["sigma"] - ref["sigma"]) < 1e-12
    s = ref["sigma"] * ref["price"]
    assert abs(row["vp_poc_dist"] - (ref["vp_levels"]["poc"] - ref["price"]) / s) < 1e-9
    assert abs(row["tpo_poc_dist"] - (ref["tpo_levels"]["poc"] - ref["price"]) / s) < 1e-9
    va = ref["vp_levels"]
    assert abs(row["vp_va_pos"] - (ref["price"] - va["val"]) / (va["vah"] - va["val"])) < 1e-9
    occ = [r["v"] for r in ref["vp"] if r["v"] > 0]
    here = next(r["v"] for r in ref["vp"] if r["lo"] <= ref["price"] < r["hi"])
    assert abs(row["vp_at_price"] - here / (sum(occ) / len(occ))) < 1e-9
    assert 0 <= row["tpo_single_up"] <= 1


def test_signals_use_only_past_data_and_cache_by_minute():
    ts, px, amt = synth(n=20000, seed=7)
    trades = [(int(t), float(p), float(a)) for t, p, a in zip(ts, px, amt)]
    now = int(ts[-1]) + 1
    known = {}
    res = core.signals(trades, now, n_min=20, known=known)
    assert len(res["minutes"]) == 20 and [m["t"] for m in res["minutes"]] == sorted(m["t"] for m in res["minutes"])
    assert res["minutes"][-1]["t"] == now // core.MIN_MS * core.MIN_MS
    # t 以降の約定を変えても、t までの水準は変わらない
    t_cut = res["minutes"][10]["t"]
    changed = [(t, p * 1.5 if t >= t_cut else p, a * 10 if t >= t_cut else a) for t, p, a in trades]
    res2 = core.signals(changed, now, n_min=20)
    assert res2["minutes"][:11] == res["minutes"][:11]
    assert res2["minutes"][11:] != res["minutes"][11:]
    # 1 分進めると、計算済みの分は使い回し、新しい分だけ足す
    res3 = core.signals(trades, now + core.MIN_MS, n_min=20, known=known)
    assert res3["minutes"][:-1] == res["minutes"][1:] and len(known) == 20


def test_handler_signals_route():
    ts, px, amt = synth(n=8000, seed=8)
    now = int(ts[-1]) + 1
    rows = [[i, int(t), float(p), float(a)] for i, (t, p, a) in enumerate(zip(ts, px, amt))]
    store = FakeStore({"rows": rows, "from_ms": now - dash.KEEP_MS, "fetched_ms": now})
    out = dash.handler({"path": "/api/signals"}, None, store=store, now_ms=now)
    body = json.loads(out["body"])
    assert out["statusCode"] == 200, body
    assert len(body["minutes"]) == 60 and set(body["keys"]) <= set(body["minutes"][-1])
