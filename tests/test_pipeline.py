"""合成データでパイプライン全体を通す（ネットワークなし）。"""
from pathlib import Path

import yaml

from bbresearch.pipeline import render, run

from synth import synth_trades, write_trades

CFG = Path(__file__).resolve().parents[1] / "configs" / "research.yaml"


def test_pipeline_end_to_end(tmp_path):
    write_trades(tmp_path / "data", "btc_jpy", synth_trades("2026-01-01", 30, seed=7, per_min=3, vol=0.0015))
    cfg = yaml.safe_load(CFG.read_text())
    cfg["data"].update(root=str(tmp_path / "data"), start="2026-01-02", end="2026-01-30")
    cfg["split"]["holdout_days"] = 7
    cfg["signal"]["k_h"] = 1.0
    cfg["model"].update(min_train=30, n_splits=3)
    cfg["backtest"]["n_random"] = 2
    cfg["experiment_log"] = str(tmp_path / "experiments.jsonl")
    cfg["pair_spec"] = {"name": "btc_jpy", "price_digits": 0, "amount_digits": 4, "unit_amount": "0.0001",
                        "status_min_amount": "0.0001"}
    rep = run(cfg, tmp_path / "out")
    md = render(rep)
    assert "バックテスト（dev）" in md and "バックテスト（holdout）" in md
    assert rep["events"]["dip"]["dev"]["n_events"] > 0
    assert "dip/primary" in rep["backtest"]["dev"]
    assert (tmp_path / "experiments.jsonl").exists()
    trained = [s for s, v in rep["cv"].items() if "skipped" not in v]
    for s in trained:
        assert (tmp_path / "out" / "models" / f"btc_jpy_{s}_lgbm.joblib").exists()
