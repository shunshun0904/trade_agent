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
    cfg["split"].update(holdout_days=7, evaluate_holdout=True)
    cfg["features"].update(profile=True, dist=True, dist_windows=[4, 16, 96])
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
        # 価格帯別出来高・TPO の特徴量がモデルに入っている
        assert any(k.startswith(("vp_", "tpo_")) for k in rep["cv"][s]["lgbm"]["feature_importance"])
        assert any(k.startswith("dist_") for k in rep["cv"][s]["lgbm"]["feature_importance"])

    # ホールドアウトを使用済みにした設定では、ホールドアウトを一切評価しない
    no_ho = {**cfg, "split": {**cfg["split"], "evaluate_holdout": False},
             "experiment_log": str(tmp_path / "no_ho.jsonl")}
    rep_no = run(no_ho, tmp_path / "out_no")
    assert set(rep_no["backtest"]) == {"dev"} and not rep_no["evaluate_holdout"]

    # 同じ設定の再実行は試行数を増やさない。設定を変えると過去の試行として数える
    n_lines = len((tmp_path / "experiments.jsonl").read_text().splitlines())
    rep2 = run(cfg, tmp_path / "out")
    assert rep2["n_trials_total"] == rep["n_trials_total"]
    assert rep2["trial_hash"] == rep["trial_hash"]
    cfg["backtest"]["n_random"] = 1  # 戦略を変えない設定は同じ試行
    assert run(cfg, tmp_path / "out")["n_trials_total"] == rep["n_trials_total"]
    cfg["signal"]["k_h"] = 1.5
    rep3 = run(cfg, tmp_path / "out")
    assert rep3["trial_hash"] != rep["trial_hash"]
    assert rep3["n_trials_total"] > rep["n_trials_total"]
    assert len((tmp_path / "experiments.jsonl").read_text().splitlines()) > n_lines


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


def test_legacy_log_rows_with_same_config_are_not_extra_trials(tmp_path):
    import json

    write_trades(tmp_path / "data", "btc_jpy", synth_trades("2026-01-01", 20, seed=5, per_min=3, vol=0.0015))
    cfg = yaml.safe_load(CFG.read_text())
    cfg["data"].update(root=str(tmp_path / "data"), start="2026-01-02", end="2026-01-20")
    cfg["split"]["holdout_days"] = 5
    cfg["signal"]["k_h"] = 1.0
    cfg["model"].update(min_train=30, n_splits=3)
    cfg["backtest"]["n_random"] = 1
    log = tmp_path / "experiments.jsonl"
    cfg["experiment_log"] = str(log)
    cfg["pair_spec"] = {"name": "btc_jpy", "price_digits": 0, "amount_digits": 4, "unit_amount": "0.0001",
                        "status_min_amount": "0.0001", "maker_fee_rate_quote": "0", "taker_fee_rate_quote": "0.001"}
    first = run(cfg, tmp_path / "out")
    # trial_hash を持たない古い形式の行（同じ config_hash）と、別設定の古い行を足す
    log.write_text(json.dumps({"config_hash": first["config_hash"], "key": "dip/primary", "daily_sr": 0.1}) + "\n"
                   + json.dumps({"config_hash": "other", "key": "dip/primary", "daily_sr": 0.2}) + "\n")
    again = run(cfg, tmp_path / "out")
    assert again["n_trials_total"] == first["n_trials_total"] + 1
