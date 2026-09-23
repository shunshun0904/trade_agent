import gzip
import json

from bblive.orderbook import OrderBook
from bblive.recorder import HourlyWriter, Recorder


def test_orderbook_follows_documented_procedure():
    ob = OrderBook()
    # ドキュメントの例: diff{s=3}, diff{s=5}, diff{s=6}, diff{s=8}, whole{sequenceId=5}
    ob.on_diff({"a": [["101", "1"]], "b": [], "s": "3"})
    ob.on_diff({"a": [["102", "1"]], "b": [], "s": "5"})
    ob.on_diff({"a": [["103", "2"]], "b": [["99", "1"]], "s": "6"})
    ob.on_diff({"a": [["101", "0"]], "b": [], "s": "8"})
    ob.on_whole({"asks": [["101", "5"], ["104", "1"]], "bids": [["98", "3"]], "sequenceId": "5"})
    # whole で置き換えたあと、s=6 と s=8 だけを再適用する（s=3, 5 は無視）
    assert ob.asks == {103.0: 2.0, 104.0: 1.0}
    assert ob.bids == {98.0: 3.0, 99.0: 1.0}
    assert ob.best() == (99.0, 103.0)
    # whole 以降の、sequenceId 以下の diff は適用しない
    ob.on_diff({"a": [["150", "1"]], "b": [], "s": "4"})
    assert 150.0 not in ob.asks


def test_hourly_writer_rotates_by_utc_hour(tmp_path):
    w = HourlyWriter(tmp_path)
    t = 1767225600000  # 2026-01-01 00:00:00 UTC
    w.write(t + 1, "transactions_btc_jpy", {"data": {"x": 1}})
    w.write(t + 3_599_999, "transactions_btc_jpy", {"data": {"x": 2}})
    w.write(t + 3_600_000, "depth_diff_btc_jpy", {"data": {"x": 3}})
    w.close()
    a = [json.loads(x) for x in gzip.open(tmp_path / "20260101" / "00.jsonl.gz", "rt")]
    b = [json.loads(x) for x in gzip.open(tmp_path / "20260101" / "01.jsonl.gz", "rt")]
    assert [x["m"]["data"]["x"] for x in a] == [1, 2] and b[0]["room"] == "depth_diff_btc_jpy"


def test_recorder_keeps_only_subscribed_rooms(tmp_path):
    r = Recorder("btc_jpy", tmp_path, hours=0.0)
    r.on_message({"room_name": "transactions_btc_jpy", "message": {"data": {"transactions": []}}})
    r.on_message({"room_name": "ticker_xrp_jpy", "message": {"data": {}}})
    r.writer.close()
    assert r.counts == {"transactions_btc_jpy": 1}
    assert "circuit_break_info_btc_jpy" in r.rooms


def test_run_returns_immediately_when_deadline_passed(tmp_path):
    import asyncio

    stats = asyncio.run(Recorder("btc_jpy", tmp_path, hours=0.0).run())
    assert stats["counts"] == {} and stats["disconnects"] == []
