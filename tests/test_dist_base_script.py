"""scripts/dist_base.py を偽の公開 API と合成の約定で最後まで通す（ネットワークにはアクセスしない）。"""
import json

import numpy as np
import pandas as pd
import pytest

import bbdata.client
import scripts.dist_base as sb
from bbresearch import nullsim

IDX = pd.date_range("2022-10-01", "2023-08-01", freq="h", tz="UTC", inclusive="left")
CAL = {"garch": {"mu": 0.0, "omega": 3e-7, "alpha": 0.08, "beta": 0.88, "nu": 4.5},
       "volume": {"a": 5.0, "b": 1.0, "phi": 0.6, "sd": 0.5}, "p0": 3e6,
       "ms2": {"sigma": [0.003, 0.01], "P": [[0.99, 0.01], [0.05, 0.95]]}}
H1, MC = nullsim.simulate(IDX, "garch_t", CAL, 11)


class FakeAPI:
    def __init__(self, *args, **kwargs):
        pass

    def candlestick(self, pair, candle_type, period):
        assert pair == "btc_jpy" and candle_type == "1hour"
        day = pd.Timestamp(period, tz="UTC")
        d = H1[(H1.index >= day) & (H1.index < day + pd.Timedelta(days=1))]
        ts = d.index.as_unit("ms").asi8.tolist()
        return [[str(o), str(h), str(lo), str(c), str(v), t] for (o, h, lo, c, v), t in
                zip(d[["open", "high", "low", "close", "volume"]].to_numpy(), ts)]


def _write_tape(root):
    """1 分に 1 件、その分の終値で約定したことにする。"""
    ts = MC.index.as_unit("ms").asi8 + 30_000
    df = pd.DataFrame({"transaction_id": np.arange(len(MC), dtype="int64"), "side": np.where(np.arange(len(MC)) % 2, "buy", "sell"),
                       "price": MC.to_numpy(), "amount": 0.01, "executed_at": ts})
    for day, g in df.groupby(MC.index.floor("D")):
        p = root / "raw" / "transactions" / "btc_jpy" / f"{day:%Y%m%d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        g.to_parquet(p, index=False)


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bbdata.client, "PublicClient", FakeAPI)
    _write_tape(tmp_path / "data")
    return {"owner_approved": None, "pair": "btc_jpy", "start": "2022-10-01", "split": "2023-07-01", "end": "2023-08-01",
            "horizons": [4, 24], "trades_root": "data", "cost": 0.003, "quantiles": [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
            "thresholds": [0.0, 0.003, -0.003], "logloss_eps": 0.001, "elementary_theta": 0.003,
            "murphy_thetas": [0.0, 0.003, 0.006], "top_share": 0.2,
            "forest": {"n_estimators": 20, "min_samples_leaf": 25, "max_features": "sqrt", "max_samples": 0.5,
                       "split_target": "bins", "n_bins": 8, "random_state": 0},
            "har": {"windows": [1, 4, 24, 168], "calendar": True, "rv_floor": 1e-10}, "rolling_n": [168, 720],
            "sign_magnitude": {"grid": 10, "valid_frac": 0.2, "early_stopping_rounds": 10,
                               "lgbm": {"n_estimators": 50, "learning_rate": 0.05, "num_leaves": 15,
                                                    "min_child_samples": 50, "random_state": 0}},
            "tests": {"nw_mult": 1.3, "ewc_mult": 0.4, "fixedb_sims": 500, "bootstrap_reps": 49, "mcs_size": 0.1,
                      "mcs_reps": 30, "mcs_block": 24, "acf_lags": [1, 4, 24, 168], "hill_share": 0.01, "seed": 0},
            "null_audit": {"models": ["garch_t", "ms2"], "reps": 1, "seed0": 1000, "shards": 2},
            "judge": {"primary": "brier_0", "comparisons": [["forest", "forest_fixed"], ["sign_magnitude", "har"]],
                      "alpha": 0.05, "null_alpha": 0.05, "width_pair": ["forest", "har"]}}


def test_check_null_and_eval_end_to_end(cfg):
    sb.main_check(cfg)
    chk = json.loads((sb.OUT / "check.json").read_text())
    assert chk["hourly"]["missing"] == 0 and chk["hourly"]["n"] == len(IDX)
    assert all(y["share_with_trades"] == 1.0 for y in chk["minutes"])
    assert chk["close_tape_vs_official"]["median"] < 1e-9           # 約定から作った終値 = 足の終値
    assert set(chk["har_train"]) == {"4", "24"} and "ms2" in chk["null_calibration"]
    assert "評価期間" in (sb.OUT / "check.md").read_text()

    with pytest.raises(SystemExit):                                   # 承認の前は評価しない
        sb.main_eval(cfg)
    sb.main_null(cfg, 0)
    sb.main_null(cfg, 1)
    sb.main_null_summary(cfg, sb.OUT)
    null = json.loads((sb.OUT / "null.json").read_text())
    assert null["n_reps"] == 2 and [r["kind"] for r in null["reps"]] == ["garch_t", "ms2"]
    assert null["code_hash"] == sb.code_hash()
    assert "forest|forest_fixed|brier_0" in null["reps"][0]["summary"]["horizons"]["4"]["comps"]

    cfg["owner_approved"] = "2026-09-28"                              # 承認日はハッシュに入れない
    sb.main_eval(cfg)
    m = json.loads((sb.OUT / "metrics.json").read_text())
    assert len(m["judge"]["direction"]) == 4 and len(m["judge"]["width"]) == 2
    assert all(r["n_null"] == 2 for r in m["judge"]["direction"])
    report = (sb.OUT / "report.md").read_text()
    assert "判定（事前登録）" in report and "フォレスト（位置を固定）" in report
    lines = (sb.OUT.parent / "experiments.jsonl").read_text().splitlines()
    assert len(lines) == 2 * 8 and json.loads(lines[0])["stage"] == "dist_base"

    cfg["null_audit"]["reps"] = 2                                           # 設定が変わったら、監査をやり直すまで評価しない
    with pytest.raises(SystemExit):
        sb.main_eval(cfg)
    cfg["null_audit"]["reps"] = 1
    null["code_hash"] = "000000000000"                                # コードが変わっても同じ
    (sb.OUT / "null.json").write_text(json.dumps(null))
    with pytest.raises(SystemExit):
        sb.main_eval(cfg)


def test_config_file_has_the_keys_the_script_reads(cfg):
    """configs/dist_base.yaml の構造が、上の試験用の設定と同じであること（YAML の null などの読み違いを防ぐ）。"""
    import yaml
    from pathlib import Path

    root = Path(sb.__file__).resolve().parents[1]
    real = yaml.safe_load((root / "configs" / "dist_base.yaml").read_text(encoding="utf-8"))
    assert set(real) == set(cfg)
    for k, v in cfg.items():
        if isinstance(v, dict):
            assert set(real[k]) == set(v), k
    assert real["owner_approved"] is None                              # 承認の前
    wf = yaml.safe_load((root / ".github" / "workflows" / "dist_base.yml").read_text(encoding="utf-8"))
    assert len(wf["jobs"]["null-shard"]["strategy"]["matrix"]["shard"]) == real["null_audit"]["shards"]
    assert real["null_audit"]["reps"] * len(real["null_audit"]["models"]) == 40
