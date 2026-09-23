"""合成データでパラメータ探索を通す（ネットワークなし）。"""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from bbresearch.search import apply_combo, expand_grid, render, run_search

from synth import synth_trades, write_trades

CFG = Path(__file__).resolve().parents[1] / "configs" / "research.yaml"


def test_expand_and_apply():
    combos = expand_grid({"label.k_up": [1, 2], "signal.k_h": [3.0]})
    assert combos == [{"label.k_up": 1, "signal.k_h": 3.0}, {"label.k_up": 2, "signal.k_h": 3.0}]
    cfg = {"label": {"k_up": 9}, "signal": {"k_h": 0}}
    out = apply_combo(cfg, combos[1])
    assert out == {"label": {"k_up": 2}, "signal": {"k_h": 3.0}} and cfg["label"]["k_up"] == 9


@pytest.mark.parametrize("workers", [1, 2])
def test_search_uses_dev_period_only(tmp_path, workers):
    write_trades(tmp_path / "data", "btc_jpy", synth_trades("2026-01-01", 20, seed=11, per_min=3, vol=0.0015))
    cfg = yaml.safe_load(CFG.read_text())
    cfg["data"].update(root=str(tmp_path / "data"), start="2026-01-02", end="2026-01-20")
    cfg["split"]["holdout_days"] = 5
    cfg["model"].update(min_train=30, n_splits=3)
    cfg["experiment_log"] = str(tmp_path / "experiments.jsonl")
    cfg["pair_spec"] = {"name": "btc_jpy", "price_digits": 0, "amount_digits": 4, "unit_amount": "0.0001",
                        "status_min_amount": "0.0001", "maker_fee_rate_quote": "0", "taker_fee_rate_quote": "0.001"}
    search_cfg = {"grid": {"signal.k_h": [1.0, 1.5], "label.k_up": [1.0, 2.0]},
                  "stage1": {"min_filled": 10}, "stage2": {"top_n": 2, "min_trades": 5, "n_random": 0}}
    res = run_search(cfg, search_cfg, tmp_path / "out", workers=workers)
    rep, screen = res["report"], res["screen"]
    assert len(screen) == 4 * 2
    assert 1 <= len(rep["stage2"]) <= 2
    for item in rep["stage2"]:
        assert all(k.split("/")[0] in ("dip", "breakout") for k in item["backtest"])
    # 試行数: ステージ1の 8 と、ステージ2の戦略（primary はステージ1と同じ試行）
    lines = (tmp_path / "experiments.jsonl").read_text().splitlines()
    assert len(lines) >= 8 and rep["n_trials_total"] >= 8
    assert "ステージ2" in render(res)
    if rep["selected"]:
        assert (tmp_path / "out" / "selected.yaml").exists()


def test_search_screen_ignores_holdout_events(tmp_path, monkeypatch):
    import bbresearch.search as search

    write_trades(tmp_path / "data", "btc_jpy", synth_trades("2026-01-01", 12, seed=3, per_min=2, vol=0.0015))
    cfg = yaml.safe_load(CFG.read_text())
    cfg["data"].update(root=str(tmp_path / "data"), start="2026-01-02", end="2026-01-12")
    cfg["split"]["holdout_days"] = 4
    cfg["experiment_log"] = str(tmp_path / "experiments.jsonl")
    cfg["pair_spec"] = {"name": "btc_jpy", "price_digits": 0, "amount_digits": 4, "unit_amount": "0.0001",
                        "status_min_amount": "0.0001", "maker_fee_rate_quote": "0", "taker_fee_rate_quote": "0.001"}
    seen = []
    orig = search.label_events

    def spy(events, *a, **k):
        seen.append(events["t0"].max())
        return orig(events, *a, **k)

    monkeypatch.setattr(search, "label_events", spy)
    search_cfg = {"grid": {"signal.k_h": [1.0]}, "stage1": {"min_filled": 10**9}, "stage2": {"top_n": 0}}
    run_search(cfg, search_cfg, tmp_path / "out", workers=1)
    assert seen and max(seen) < pd.Timestamp("2026-01-08", tz="UTC")
