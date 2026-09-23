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
                        "status_min_amount": "0.0001",
                        "maker_fee_rate_quote": "0", "taker_fee_rate_quote": "0.001"}
    rep = run(cfg, tmp_path / "out")
    md = render(rep)
    assert "バックテスト（dev）" in md and "バックテスト（holdout）" in md
    assert rep["events"]["dip"]["dev"]["n_events"] > 0
    assert "dip/primary" in rep["backtest"]["dev"]
    assert (tmp_path / "experiments.jsonl").exists()
    trained = [s for s, v in rep["cv"].items() if "skipped" not in v]
    for s in trained:
        assert (tmp_path / "out" / "models" / f"btc_jpy_{s}_lgbm.joblib").exists()


def test_random_events_cover_the_whole_period():
    import numpy as np
    import pandas as pd

    from bbresearch.pipeline import random_events

    idx = pd.date_range("2026-01-01", periods=96 * 20, freq="15min", tz="UTC")
    bars = pd.DataFrame({"close": 100.0}, index=idx)
    sigma = pd.Series(0.01, index=idx)
    start, end = idx[0], idx[-1]
    rev = random_events(bars, sigma, 400, start, end, np.random.default_rng(0))
    assert len(rev) == 400
    assert rev["t0"].is_monotonic_increasing
    assert (rev["t0"] > start).all() and (rev["t0"] <= end).all()
    # 期間の前半だけに偏らない
    mid = start + (end - start) / 2
    assert 0.35 < (rev["t0"] < mid).mean() < 0.65
