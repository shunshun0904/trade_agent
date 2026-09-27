"""scripts/swing.py を偽の公開 API で最後まで通す（ネットワークにはアクセスしない）。"""
import json

import numpy as np
import pandas as pd
import pytest

import scripts.swing as sw

PAIRS = {"btc_jpy": "0.001", "eth_jpy": "0.0012", "xrp_jpy": "0.0012"}


class FakeAPI:
    """1 時間足の合成データから 4 時間足（UTC 0 時起点）を作って返す。"""

    def __init__(self, *args, **kwargs):
        idx = pd.date_range("2019-06-01", "2021-01-01", freq="1h", tz="UTC", inclusive="left")
        rng = np.random.default_rng(0)
        self.h1 = {}
        for i, p in enumerate(PAIRS):
            c = 100 * (i + 1) * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx))))
            df = pd.DataFrame({"open": np.r_[c[0], c[:-1]], "close": c, "volume": 1e5 / (i + 1)}, index=idx)
            df["high"] = df[["open", "close"]].max(axis=1) * 1.001
            df["low"] = df[["open", "close"]].min(axis=1) * 0.999
            self.h1[p] = df[["open", "high", "low", "close", "volume"]]

    def get(self, path):
        assert path == "/spot/pairs"
        return {"pairs": [{"name": p, "taker_fee_rate_quote": f, "is_enabled": True} for p, f in PAIRS.items()]}

    @staticmethod
    def _rows(df):
        ts = (df.index.as_unit("ms").asi8).tolist()
        return [[str(o), str(h), str(lo), str(c), str(v), t] for (o, h, lo, c, v), t in
                zip(df[["open", "high", "low", "close", "volume"]].to_numpy(), ts)]

    def candlestick(self, pair, candle_type, period):
        h1 = self.h1[pair]
        if candle_type == "1hour":
            day = pd.Timestamp(period, tz="UTC")
            return self._rows(h1[(h1.index >= day) & (h1.index < day + pd.Timedelta(days=1))])
        assert candle_type == "4hour"
        h4 = sw.to_4h(h1, 0)
        return self._rows(h4[h4.index.year == int(period)])

    def depth(self, pair):
        return {"asks": [["100.1", "1"]], "bids": [["99.9", "1"]]}


@pytest.fixture()
def cfg():
    return {"owner_approved": None, "data": {"start": "2019-06-01", "end": "2021-01-01", "exclude": []},
            "min_bars": 120, "turnover_days": 10, "min_turnover_jpy": 1e6, "sigma_days": 10, "hold_days": [1, 2, 3],
            "tsmom": {"kinds": ["ret", "ma"], "lookback_days": [5, 10]}, "xsmom": {"lookback_days": [5], "top_k": 2},
            "rebound": {"window_bars": [6], "k_sigma": [2]}, "cost": {"slippage": 0.001, "stress_mult": 2.0},
            "judge": {"min_share_alpha_pos": 0.667, "min_dsr": 0.95, "min_year_share": 0.6, "full_years": [2020]}}


@pytest.fixture()
def fake(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sw, "PublicClient", FakeAPI)
    monkeypatch.setattr(sw, "CHECK_DAYS", ("2020-09-14", "2020-09-21"))


def test_check_matches_the_4h_boundary_and_reports_pairs(fake, cfg, tmp_path):
    sw.main_check(cfg)
    out = json.loads((tmp_path / "reports/swing/check.json").read_text())
    by_off = {b["offset_hours"]: b for b in out["boundary"]}
    assert by_off[0]["ohlc_match"] == 1.0
    assert all(by_off[o]["n"] > 30 and by_off[o]["ohlc_match"] < 0.5 for o in (-3, -2, -1, 1, 2, 3))
    assert {p["pair"] for p in out["pairs"]} == set(PAIRS)
    assert all(abs(p["half_spread"] - 0.001) < 1e-9 for p in out["pairs"])
    assert "4 時間足の境界" in (tmp_path / "reports/swing/check.md").read_text()


def test_eval_refuses_without_approval_then_runs_once_approved(fake, cfg, tmp_path):
    with pytest.raises(SystemExit):
        sw.main_eval(cfg)
    cfg["owner_approved"] = "2026-09-28"
    sw.main_eval(cfg)
    m = json.loads((tmp_path / "reports/swing/metrics.json").read_text())
    n = 2 * 2 * 3 + 1 * 3 + 1 * 1 * 3
    assert len(m["results"]) == n and len(m["verdicts"]) == 3
    assert m["cost"]["btc_jpy"] == pytest.approx(0.002) and m["cost"]["eth_jpy"] == pytest.approx(0.0022)
    log = [json.loads(line) for line in (tmp_path / "reports/experiments.jsonl").read_text().splitlines()]
    assert len(log) == n and all(r["stage"] == "swing" for r in log)
    assert "判定" in (tmp_path / "reports/swing/report.md").read_text()
