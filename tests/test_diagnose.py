from pathlib import Path

import yaml

from bbresearch.diagnose import render, run_diagnose

from synth import synth_trades, write_trades

CFG = Path(__file__).resolve().parents[1] / "configs" / "research.yaml"


def test_diagnose_runs_on_dev_only(tmp_path):
    write_trades(tmp_path / "data", "btc_jpy", synth_trades("2026-01-01", 15, seed=4, per_min=3, vol=0.0015))
    cfg = yaml.safe_load(CFG.read_text())
    cfg["data"].update(root=str(tmp_path / "data"), start="2026-01-02", end="2026-01-15")
    cfg["split"]["holdout_days"] = 4
    cfg["pair_spec"] = {"name": "btc_jpy", "price_digits": 0, "amount_digits": 4, "unit_amount": "0.0001",
                        "status_min_amount": "0.0001", "maker_fee_rate_quote": "0", "taker_fee_rate_quote": "0.001"}
    rep = run_diagnose(cfg, {"sets": {"a": {"signal.k_h": 1.0, "label.k_up": 1.0, "label.k_dn": 1.0}}})
    d = rep["sets"]["a"]["signals"]["dip"]
    assert d["n_events"] > 0
    base, gross, ms = d["pnl"]["base"], d["pnl"]["gross"], d["pnl"]["maker_stop"]
    # 同じ取引で費用だけを変えている: 手数料・滑りなしのほうが平均は高い
    assert base["n"] == gross["n"] == ms["n"] and gross["mean"] > base["mean"]
    assert ms["mean"] >= base["mean"]
    assert d["market_entry"]["n"] >= d["pnl"]["base"]["n"]
    assert "事後の対数リターン" in render(rep)
