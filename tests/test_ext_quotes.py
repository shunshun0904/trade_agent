"""scripts/ext_quotes.py: 判断の時刻の気配の記録（ネットワークにはアクセスしない）。"""
import json

import pandas as pd

import scripts.ext_quotes as eq


class FakeAPI:
    def ticker(self, pair):
        assert pair == "btc_jpy"
        return {"sell": "15000100", "buy": "14999900", "last": "15000000", "timestamp": 1790000000000}


def test_slot_of():
    ts = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
    assert eq.slot_of(ts("2026-10-01 00:00")) == ts("2026-10-01 00:00")
    assert eq.slot_of(ts("2026-10-01 00:37")) == ts("2026-10-01 00:00")
    assert eq.slot_of(ts("2026-10-01 15:59")) == ts("2026-10-01 08:00")
    assert eq.slot_of(ts("2026-10-01 23:10")) == ts("2026-10-01 16:00")


def test_record_and_main(capsys):
    r = eq.record(FakeAPI(), now=pd.Timestamp("2026-10-01 08:17:30", tz="UTC"))
    assert r["slot"] == "2026-10-01T08:00:00+00:00" and r["delay_min"] == 17.5
    assert (r["sell"], r["buy"], r["last"]) == (15000100.0, 14999900.0, 15000000.0)
    assert eq.main([], api=FakeAPI()) == 0
    line = json.loads(capsys.readouterr().out)
    assert set(line) == {"slot", "fetched_at", "delay_min", "pair", "sell", "buy", "last", "ticker_ts"}
