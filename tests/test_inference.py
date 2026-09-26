"""本番（dashboard/app/inference.py）の確率が、研究用の関数と LightGBM の予測に一致することを確かめる。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard" / "app"))

import app as dash  # noqa: E402
import inference  # noqa: E402

from bbdata.bars import build_bars  # noqa: E402
from bbresearch.labeling import TradeTape  # noqa: E402
from bbresearch.minute_features import MIN_MS, feature_table, minute_grid  # noqa: E402
from bbresearch.pipeline import Market  # noqa: E402

from synth import synth_trades, write_trades  # noqa: E402

lgb = pytest.importorskip("lightgbm")
SPEC = {"name": "btc_jpy", "price_digits": 0, "amount_digits": 4, "unit_amount": "0.0001",
        "status_min_amount": "0.0001", "maker_fee_rate_quote": "0", "taker_fee_rate_quote": "0.001"}


def _market(root, trades, end):
    write_trades(root, "btc_jpy", trades)
    bars = build_bars(root, "btc_jpy", "2026-01-01", end, "15min")
    tape = TradeTape.load(root, "btc_jpy", "2026-01-01", end)
    return Market(bars=bars, tape=tape, spec=SPEC, tick=1.0, min_amount=0.0001)


def _export(model, feats):
    return {"model": model.booster_.dump_model(), "features": feats, "horizon_min": 60,
            "target_min_return": 0.003, "sigma_span": 96, "train_end": "2026-01-04", "large_trade_amount": None}


def test_lambda_probabilities_match_research_pipeline(tmp_path):
    trades = synth_trades("2026-01-01", 4, seed=11, per_min=3, vol=0.002)
    m = _market(tmp_path, trades, "2026-01-05")
    F = feature_table(m, "2026-01-01", "2026-01-05", 96, 1, 60 * MIN_MS, 0.003)
    F = F[F["y"].notna()]
    feats = [c for c in F.columns if c not in ("y", "r")]
    model = lgb.LGBMClassifier(n_estimators=30, num_leaves=8, min_child_samples=20, verbose=-1, random_state=0)
    model.fit(F[feats], F["y"].astype(int))
    export = _export(model, feats)

    # Lambda 側: 保存データの行（id, ts, price, amount, is_buy）から同じ時刻の確率を出す
    t = trades.sort_values("executed_at")
    rows = [[int(i), int(ts), float(p), float(a), s == "buy"]
            for i, ts, p, a, s in zip(t["transaction_id"], t["executed_at"], t["price"], t["amount"], t["side"])]
    grid = F.index[-60:].tolist()
    pred = inference.predict_minutes(rows, grid, export)
    ref = model.predict_proba(F.loc[grid, feats])[:, 1]
    got = np.array([pred[g] for g in grid], dtype=float)
    assert np.allclose(got, ref, atol=1e-6)


def test_predict_minutes_returns_none_without_history():
    rows = [[i, 1_767_225_600_000 + i * 20_000, 10_000_000.0 + i, 0.01, i % 2 == 0] for i in range(300)]
    grid = [1_767_225_600_000 + 90 * MIN_MS]
    model = {"model": {"objective": "binary sigmoid:1", "tree_info": [], "num_class": 1, "feature_names": []},
             "features": ["b_rv_96", "b_dist_ma_96", "vp_poc_dist"], "sigma_span": 96}  # 96 本の足と 24 時間の窓が要る
    assert inference.predict_minutes(rows, grid, model) == {grid[0]: None}


def test_handler_adds_probability_when_model_is_loaded(monkeypatch, tmp_path):
    trades = synth_trades("2026-01-01", 3, seed=12, per_min=3, vol=0.002)
    m = _market(tmp_path, trades, "2026-01-04")
    F = feature_table(m, "2026-01-01", "2026-01-04", 96, 1, 60 * MIN_MS, 0.003)
    F = F[F["y"].notna()]
    feats = [c for c in F.columns if c not in ("y", "r")]
    model = lgb.LGBMClassifier(n_estimators=10, num_leaves=4, min_child_samples=20, verbose=-1).fit(F[feats], F["y"].astype(int))
    (tmp_path / "model_B.json").write_text(json.dumps(_export(model, feats)))
    monkeypatch.setattr(inference, "MODEL_PATHS", (tmp_path / "model_B.json",))

    t = trades.sort_values("executed_at")
    rows = [[int(i), int(ts), float(p), float(a), s == "buy"]
            for i, ts, p, a, s in zip(t["transaction_id"], t["executed_at"], t["price"], t["amount"], t["side"])]
    now = int(t["executed_at"].iloc[-1]) + 1

    class Store:
        def __init__(self, d): self.d = d
        def load(self): return self.d
        def save(self, d): self.d = d

    mem = {}
    monkeypatch.setattr(dash, "_MEM", mem)
    monkeypatch.setattr(dash, "Store", lambda bucket: Store({"rows": rows, "from_ms": now - dash.KEEP_MS, "fetched_ms": now}))
    last = rows[-1]
    latest = lambda path: {"transactions": [{"transaction_id": last[0], "executed_at": last[1], "price": str(last[2]),  # noqa: E731
                                             "amount": str(last[3]), "side": "buy"}]}
    out = dash.handler({"path": "/api/signals"}, None, get=latest, now_ms=now)
    body = json.loads(out["body"])
    assert out["statusCode"] == 200, body
    ps = [r["p_big_1h"] for r in body["minutes"]]
    assert len(ps) == 60 and all(p is not None and 0 <= p <= 1 for p in ps)
    assert body["model"]["horizon_min"] == 60
    # 2 回目は計算済みの分を使い回す（新しい分だけ計算する）
    n_before = len(mem["pred"])
    out2 = dash.handler({"path": "/api/signals"}, None, get=latest, now_ms=now + MIN_MS)
    assert out2["statusCode"] == 200 and len(mem["pred"]) == n_before


def test_refresh_discards_cache_rows_without_side():
    now = 1_767_225_600_000 + 60 * 3_600_000
    old = {"rows": [[1, now - 1000, 1.0, 1.0]], "from_ms": now - dash.KEEP_MS, "fetched_ms": now - 3_600_000}
    days = []

    def get(path):
        days.append(path)
        return {"transactions": [{"transaction_id": 5, "executed_at": now - 500, "price": "1", "amount": "1", "side": "buy"}]}

    res = dash.refresh(old, now, get)
    assert any("/transactions/" in d for d in days)  # 日付指定で取り直した
    assert res["rows"] == [[5, now - 500, 1.0, 1.0, True]]
