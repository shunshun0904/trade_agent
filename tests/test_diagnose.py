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


def test_horizon_analysis(tmp_path):
    import numpy as np

    from bbresearch.diagnose import _non_overlapping, render_horizon, run_horizon

    assert list(_non_overlapping(np.array([0, 1, 5, 6, 12]), 5)) == [0, 2, 4]
    write_trades(tmp_path / "data", "btc_jpy", synth_trades("2026-01-01", 15, seed=4, per_min=3, vol=0.0015))
    cfg = yaml.safe_load(CFG.read_text())
    cfg["data"].update(root=str(tmp_path / "data"), start="2026-01-02", end="2026-01-15")
    cfg["split"]["holdout_days"] = 4
    cfg["pair_spec"] = {"name": "btc_jpy", "price_digits": 0, "amount_digits": 4, "unit_amount": "0.0001",
                        "status_min_amount": "0.0001", "maker_fee_rate_quote": "0", "taker_fee_rate_quote": "0.001"}
    rep = run_horizon(cfg, {"sets": {"a": {"signal.k_h": 1.0}}, "horizons": [1, 16]})
    r = rep["sets"]["a"]["signals"]["dip"]
    assert r[1]["event"]["n"] >= r[16]["event"]["n"] > 0
    # 成行往復: 手数料と滑りだけで −0.3%。価格が動くので、ちょうど −0.3% にはならない
    # （買値と売値が同じ約定を指すと全件ちょうど −0.3% になる。時刻の単位の誤りで起きた）
    fees_only = (1 - 0.0005) * (1 - 0.001) / ((1 + 0.0005) * (1 + 0.001)) - 1
    assert r[16]["market_rt"]["n"] > 0 and abs(r[16]["market_rt"]["mean"] - fees_only) > 1e-6
    assert r[16]["maker_in"]["n"] > 0
    assert np.isfinite(r[16]["baseline_mean"]) and r[16]["t_nonoverlap"] is not None
    assert "成行往復" in render_horizon(rep)


def test_entry_orders_use_only_bars_up_to_placement():
    import numpy as np
    import pandas as pd

    from bbresearch.diagnose import _entry_orders

    idx = pd.date_range("2026-01-01", periods=10, freq="15min", tz="UTC")
    bars = pd.DataFrame({"close": [100, 99, 98, 99, 100, 101, 100, 99, 98, 97.0],
                         "volume_imbalance": [0, 0, -1, 0.5, 0, 0, 0, 0, 0, 0.0]}, index=idx)
    e = pd.DataFrame({"t0": [idx[2] + pd.Timedelta("15min")]})
    pos = np.array([2])
    sig = np.array([0.01])
    # 1 本待つ: 足 3 の終値 99 ≥ 足 2 の終値 98 → 足 3 の終了時刻に 99 で指値
    (q, t_ms, px), = _entry_orders(e, pos, bars, sig, {"wait_bars": 1, "require_up": True}, 1.0)
    assert q == 3 and px == 99 and t_ms == (idx[3] + pd.Timedelta("15min")).value // 10**6
    # 下げが止まっていなければ出さない（足 2 基準で 0 本待ちは c[q] ≥ c[p] が自明に成り立つので 4 本後で確認）
    bars2 = bars.copy()
    bars2["close"] = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91.0]
    assert _entry_orders(e, pos, bars2, sig, {"wait_bars": 1, "require_up": True}, 1.0) == [None]
    # 深い指値: 1σ 下（98 × 0.99 = 97.02 → 呼値 1 で切り捨て 97）
    (_, _, px2), = _entry_orders(e, pos, bars, sig, {"delta_sigma": 1.0}, 1.0)
    assert px2 == 97
    # 足の買いが売りを上回ったときだけ
    assert _entry_orders(e, pos, bars, sig, {"wait_bars": 1, "require_imbalance": True}, 1.0)[0] is not None
    assert _entry_orders(e, pos, bars, sig, {"wait_bars": 0, "require_imbalance": True}, 1.0) == [None]


def test_entry_set_runs(tmp_path):
    from bbresearch.diagnose import render_entry, run_entry

    write_trades(tmp_path / "data", "btc_jpy", synth_trades("2026-01-01", 12, seed=6, per_min=3, vol=0.0015))
    cfg = yaml.safe_load(CFG.read_text())
    cfg["data"].update(root=str(tmp_path / "data"), start="2026-01-02", end="2026-01-12")
    cfg["split"]["holdout_days"] = 3
    cfg["pair_spec"] = {"name": "btc_jpy", "price_digits": 0, "amount_digits": 4, "unit_amount": "0.0001",
                        "status_min_amount": "0.0001", "maker_fee_rate_quote": "0", "taker_fee_rate_quote": "0.001"}
    rules = {"immediate": {}, "wait1_up": {"wait_bars": 1, "require_up": True}, "deep": {"delta_sigma": 1.0}}
    rep = run_entry(cfg, {"sets": {"a": {"signal.k_h": 1.0}}, "entry_rules": rules})
    d = rep["sets"]["a"]["signals"]["dip"]
    assert d["immediate"]["placed"] >= d["wait1_up"]["placed"] > 0
    assert d["deep"]["fill_rate"] <= d["immediate"]["fill_rate"]
    assert "トリプルバリア" in render_entry(rep)
