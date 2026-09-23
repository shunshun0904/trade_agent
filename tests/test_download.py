from datetime import date

import pandas as pd
import pytest

from bbdata.client import BitbankAPIError, PublicClient
from bbdata.download import (
    download_candles,
    download_transactions,
    load_candles,
    load_transactions,
    trades_path,
)

BASE = "https://public.bitbank.cc"


class FakeResp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


class FakeSession:
    """URL ごとに返すレスポンスを決める。リストなら先頭から順に返す。"""

    def __init__(self, routes, default=None):
        self.routes = routes
        self.default = default
        self.calls = []

    def get(self, url, timeout):
        self.calls.append(url)
        r = self.routes.get(url, self.default)
        if r is None:
            raise AssertionError(f"想定外のURL: {url}")
        return r.pop(0) if isinstance(r, list) else r


def ok(data):
    return FakeResp(200, {"success": 1, "data": data})


def client(session):
    return PublicClient(session=session, min_interval=0.0, sleep=lambda s: None)


def tx(i, ms, side="buy", price="100", amount="0.1"):
    return {"transaction_id": i, "side": side, "price": price, "amount": amount, "executed_at": ms}


def test_retry_on_429_then_success():
    url = f"{BASE}/btc_jpy/transactions/20260101"
    s = FakeSession({url: [FakeResp(429, {}), FakeResp(503, {}), ok({"transactions": [tx(1, 0)]})]})
    assert client(s).transactions("btc_jpy", "20260101") == [tx(1, 0)]
    assert len(s.calls) == 3


def test_error_response_is_not_retried():
    url = f"{BASE}/btc_jpy/transactions/20260101"
    s = FakeSession({url: FakeResp(404, {"success": 0, "data": {"code": 10000}})})
    with pytest.raises(BitbankAPIError) as e:
        client(s).transactions("btc_jpy", "20260101")
    assert e.value.code == 10000 and len(s.calls) == 1


def test_download_pads_days_and_skips_final_files(tmp_path):
    t = lambda d: int(pd.Timestamp(d, tz="UTC").value // 1_000_000)
    routes = {
        f"{BASE}/btc_jpy/transactions/20260101": ok({"transactions": [tx(1, t("2026-01-01T23:00"))]}),
        f"{BASE}/btc_jpy/transactions/20260102": ok({"transactions": [tx(2, t("2026-01-02T01:00"))]}),
        f"{BASE}/btc_jpy/transactions/20260103": ok({"transactions": [tx(2, t("2026-01-02T01:00")),
                                                                      tx(3, t("2026-01-03T12:00"))]}),
    }
    today = date(2026, 3, 1)
    s = FakeSession(routes)
    got = download_transactions(client(s), "btc_jpy", date(2026, 1, 2), date(2026, 1, 3), tmp_path, today=today)
    assert got == [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)]  # 前後1日を含む
    assert trades_path(tmp_path, "btc_jpy", date(2026, 1, 2)).exists()

    # 2回目は確定済みなので API を叩かない
    s2 = FakeSession({})
    assert download_transactions(client(s2), "btc_jpy", date(2026, 1, 2), date(2026, 1, 3), tmp_path, today=today) == []

    df = load_transactions(tmp_path, "btc_jpy", "2026-01-02", "2026-01-03")
    assert df["transaction_id"].tolist() == [2]  # 重複除去と期間切り出し


def test_recent_days_are_refetched(tmp_path):
    s = FakeSession({}, default=ok({"transactions": []}))
    today = date(2026, 1, 5)
    download_transactions(client(s), "btc_jpy", date(2026, 1, 4), date(2026, 1, 5), tmp_path, today=today)
    n_first = len(s.calls)
    download_transactions(client(s), "btc_jpy", date(2026, 1, 4), date(2026, 1, 5), tmp_path, today=today)
    # 取得範囲は 1/3〜1/5。1/3 は確定済み（today-2）なのでスキップ、1/4 と 1/5 は取り直す
    assert s.calls[n_first:] == [f"{BASE}/btc_jpy/transactions/20260104", f"{BASE}/btc_jpy/transactions/20260105"]


def test_api_error_day_is_skipped(tmp_path):
    bad = FakeResp(400, {"success": 0, "data": {"code": 10000}})
    s = FakeSession({}, default=bad)
    got = download_transactions(client(s), "btc_jpy", date(2026, 1, 2), date(2026, 1, 3), tmp_path,
                                today=date(2026, 3, 1))
    assert got == []


def test_candles_daily_and_yearly(tmp_path):
    row = ["1", "2", "0.5", "1.5", "10", int(pd.Timestamp("2026-01-02T00:15Z").value // 1_000_000)]
    block = ok({"candlestick": [{"type": "15min", "ohlcv": [row]}], "timestamp": 0})
    s = FakeSession({}, default=block)
    got = download_candles(client(s), "btc_jpy", "15min", date(2026, 1, 2), date(2026, 1, 3), tmp_path,
                           today=date(2026, 3, 1))
    assert got == ["20260101", "20260102", "20260103"]
    c = load_candles(tmp_path, "btc_jpy", "15min", "2026-01-02", "2026-01-03")
    assert len(c) == 1 and c.iloc[0]["close"] == 1.5

    s2 = FakeSession({}, default=ok({"candlestick": [{"type": "1day", "ohlcv": []}], "timestamp": 0}))
    got = download_candles(client(s2), "btc_jpy", "1day", date(2025, 6, 1), date(2026, 2, 1), tmp_path,
                           today=date(2026, 3, 1))
    assert got == ["2025", "2026"]
