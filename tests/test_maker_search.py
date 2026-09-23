"""合成データで決済条件の探索を通す（ネットワークなし）。"""
from pathlib import Path

import pytest
import yaml

from bbresearch.maker_search import render, run_maker_search

from synth import synth_trades, write_trades

CFG = Path(__file__).resolve().parents[1] / "configs" / "research.yaml"


@pytest.mark.parametrize("workers", [1, 2])
def test_maker_search_selects_on_search_period_and_validates_once(tmp_path, workers):
    write_trades(tmp_path / "data", "btc_jpy", synth_trades("2026-01-01", 24, seed=9, per_min=3, vol=0.0015))
    cfg = yaml.safe_load(CFG.read_text())
    cfg["data"].update(root=str(tmp_path / "data"), start="2026-01-02", end="2026-01-24")
    cfg["split"]["holdout_days"] = 4
    cfg["experiment_log"] = str(tmp_path / "experiments.jsonl")
    cfg["pair_spec"] = {"name": "btc_jpy", "price_digits": 0, "amount_digits": 4, "unit_amount": "0.0001",
                        "status_min_amount": "0.0001", "maker_fee_rate_quote": "0", "taker_fee_rate_quote": "0.001"}
    grid_cfg = {"search_end": "2026-01-14", "min_trades": 5, "signals": ["dip"],
                "grid": {"k_h": [1.0], "delta_sigma": [0.5, 1.0], "t_fill_min": [15], "k_up": [1.0],
                         "k_dn": [4.0], "n_v": [2], "grace_min": [15]}}
    res = run_maker_search(cfg, grid_cfg, workers=workers)
    rep = res["report"]
    assert len(res["screen"]) == 2
    assert len((tmp_path / "experiments.jsonl").read_text().splitlines()) == 2
    sel = rep["selected"]
    assert sel is not None and sel["validate"]["n_events"] > 0
    assert rep["periods"]["validate"][0].startswith("2026-01-14")
    assert rep["periods"]["validate"][1].startswith("2026-01-20")  # ホールドアウト開始の手前まで
    assert "確認期間" in render(res)
