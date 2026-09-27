"""scripts/swing_forward.py を偽の公開 API で通す（ネットワークにはアクセスしない）。"""
import hashlib
import json
from datetime import datetime, timezone

import pytest

import scripts.swing as sw
import scripts.swing_forward as fw
from test_swing_script import FakeAPI


def _utc(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


@pytest.fixture()
def setup(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sw, "PublicClient", FakeAPI)
    cfg = {"owner_approved": "2026-09-28", "data": {"start": "2019-06-01", "end": "2021-01-01", "exclude": []},
           "min_bars": 120, "turnover_days": 10, "min_turnover_jpy": 1e6, "sigma_days": 10, "hold_days": [1, 2, 3],
           "tsmom": {"kinds": ["ret", "ma"], "lookback_days": [5, 10]}, "xsmom": {"lookback_days": [5], "top_k": 2},
           "rebound": {"window_bars": [6], "k_sigma": [2]}, "cost": {"slippage": 0.001, "stress_mult": 2.0}}
    (tmp_path / "frozen.py").write_text("x = 1\n")
    fwd = {"forward_start": "2020-09-01T00:00:00Z", "judge_at": "2020-12-01T00:00:00Z", "family": "tsmom",
           "frozen": {"frozen.py": hashlib.sha256(b"x = 1\n").hexdigest()}, "backtest_mean_alpha_ann": 0.25}
    return cfg, fwd, tmp_path


def test_forward_uses_only_bars_between_start_and_now(setup):
    cfg, fwd, _ = setup
    m = fw.forward(cfg, fwd, _utc("2020-11-01T00:00:00"))
    assert m["bars"] == 61 * 6  # 2020-09-01 〜 2020-10-31 の 4 時間足
    assert m["last_bar"].startswith("2020-10-31 20:00")
    assert len(m["results"]) == 12 and "verdict" not in m
    assert m["summary"]["days"] == 61 and m["summary"]["n"] == 12


def test_forward_stops_at_judge_at_and_gives_a_verdict(setup):
    cfg, fwd, _ = setup
    m = fw.forward(cfg, fwd, _utc("2020-12-20T00:00:00"))
    assert m["bars"] == 91 * 6  # 2020-09-01 〜 2020-11-30（判定日より後の足は使わない）
    assert m["verdict"] in ("前向きでも正", "前向きでは正にならなかった")


def test_forward_skips_the_forming_bar(setup):
    cfg, fwd, _ = setup
    m = fw.forward(cfg, fwd, _utc("2020-11-01T02:00:00"))  # 00:00 の足はまだ形成中
    assert m["last_bar"].startswith("2020-10-31 20:00")


def test_forward_refuses_when_a_frozen_file_changed(setup):
    cfg, fwd, tmp = setup
    (tmp / "frozen.py").write_text("x = 2\n")
    with pytest.raises(SystemExit):
        fw.forward(cfg, fwd, _utc("2020-11-01T00:00:00"))


def test_forward_before_start_writes_a_report_without_results(setup):
    cfg, fwd, tmp = setup
    m = fw.forward(cfg, fwd, _utc("2020-08-15T00:00:00"))
    assert m["bars"] == 0 and m["results"] == {} and m["holdings"]
    fw.write(fwd, m)
    fw.write(fwd, fw.forward(cfg, fwd, _utc("2020-11-01T00:00:00")))
    hist = [json.loads(x) for x in (tmp / "reports/swing_forward/history.jsonl").read_text().splitlines()]
    assert len(hist) == 2 and hist[0]["bars"] == 0 and hist[1]["days"] == 61
    report = (tmp / "reports/swing_forward/report.md").read_text()
    assert "途中経過" in report and "今の保有" in report
    total = sum(h["weight"] for h in m["holdings"])
    assert 0.0 <= total <= 1.0 + 1e-9 and all(0.0 <= h["share_on"] <= 1.0 for h in m["holdings"])


def test_frozen_hashes_match_the_files_in_the_repository():
    """評価に使ったコードと数値を変えると、ここで気づく（前向きの検証が別物になるため）。"""
    import yaml
    with open("configs/swing_forward.yaml", encoding="utf-8") as f:
        fwd = yaml.safe_load(f)
    for path, h in fwd["frozen"].items():
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == h, path
