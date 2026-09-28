"""scripts/ext_check.py を偽の応答で最後まで通す（ネットワークにはアクセスしない）。"""
import json
import lzma
import struct

import scripts.ext_check as ec


class FakeResp:
    def __init__(self, status=200, body=None, content=b"", headers=None):
        self.status_code = status
        self._body = body
        self.content = content
        self.text = json.dumps(body) if body is not None else content.decode("latin1")
        self.headers = headers or {}

    def json(self):
        return self._body


class FakeSession:
    def __init__(self):
        self.headers = {}
        self.calls = []

    def request(self, method, url, timeout=None, allow_redirects=True):
        self.calls.append((method, url))
        if "://api.binance.com" in url:
            return FakeResp(451, {"code": 0, "msg": "Service unavailable from a restricted location"})
        if "data-api.binance.vision" in url or "fapi/v1/klines" in url or "premiumIndexKlines" in url:
            return FakeResp(200, [[1502942400000, "4261.48", "4313.62", "4261.32", "4308.83", "47.18"]])
        if "fundingRate?" in url:
            return FakeResp(200, [{"symbol": "BTCUSDT", "fundingTime": 1568102400000, "fundingRate": "0.00010000"}])
        if "openInterestHist" in url:
            return FakeResp(200, [{"timestamp": 1790000000000, "sumOpenInterest": "1"}])
        if "data.binance.vision" in url or "public.bybit.com" in url:
            return FakeResp(200, headers={"Content-Length": "12345"})
        if "api.bybit.com" in url:
            if "funding" in url:
                return FakeResp(200, {"retCode": 0, "result": {"list": [{"fundingRateTimestamp": "1585699200000"}]}})
            if "open-interest" in url:
                return FakeResp(200, {"retCode": 0, "result": {"list": []}})
            return FakeResp(200, {"retCode": 0, "result": {"list": [["1585699200000", "6400", "6450", "6390", "6420", "10", "1"]]}})
        if "dukascopy" in url:
            raw = struct.pack(">5if", 0, 108610, 108700, 108500, 108750, 1.5) * 3
            return FakeResp(200, content=lzma.compress(raw, format=lzma.FORMAT_ALONE))
        if "bitmex.com" in url and "startTime" in url:
            return FakeResp(200, [{"timestamp": "2019-06-01T04:00:00.000Z", "fundingRate": 0.0001, "fundingInterval": "2000-01-01T08:00:00.000Z"}])
        if "deribit.com" in url and "1559347200000" in url:
            return FakeResp(200, {"result": []})
        if "deribit.com" in url and "1577836800000" in url and "end_timestamp=1577847600000" in url:
            return FakeResp(200, {"result": [{"timestamp": 1577836800000, "interest_8h": 0.0001, "interest_1h": 0.00001}]})
        if any(h in url for h in ("deribit.com", "okx.com", "kraken.com", "bitmex.com")):
            return FakeResp(200, content=b"{}")
        if "fred" in url:
            return FakeResp(200, content=b"observation_date,DEXJPUS\n1971-01-04,357.73\n2026-09-25,149.10\n")
        raise AssertionError(url)


def test_probes_run_and_report(tmp_path):
    sess = FakeSession()
    assert ec.main(["--out", str(tmp_path)], session=sess, pause=0) == 0
    rep = json.loads((tmp_path / "check.json").read_text())
    rows = {r["name"]: r for r in rep["probes"]}
    assert len(sess.calls) == len(ec.PROBES) and set(rows) == {p[0] for p in ec.PROBES}
    assert rows["binance_spot_klines"]["status"] == 451 and not rows["binance_spot_klines"]["ok"]
    assert "2019-09-10" in rows["binance_fut_funding"]["detail"]
    assert "空" in rows["bybit_oi_2021"]["detail"]
    assert "始値 108.610" in rows["dukascopy_hour_2020_01"]["detail"]
    assert "2026-09-25,149.10" in rows["fred_dexjpus"]["detail"]
    assert "2019-06-01T04:00" in rows["bitmex_funding_2019"]["detail"]
    assert rows["deribit_funding_2019"]["detail"] == "空" and "2020-01-01 00:00" in rows["deribit_funding_2020"]["detail"]
    md = (tmp_path / "check.md").read_text()
    assert "| binance_spot_klines |" in md and "451" in md


def test_unreachable_host_is_recorded():
    class Broken(FakeSession):
        def request(self, *a, **k):
            raise ConnectionError("blocked")
    r = ec.probe(Broken(), "x", "GET", "https://example.invalid", "binance_klines")
    assert r["status"] is None and not r["ok"] and "ConnectionError" in r["detail"]
