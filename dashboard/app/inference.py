"""1 時間後に 0.3% を超えて上がる確率（ボラティリティの目安）を、研究用と同じ関数で計算する。

- 特徴量: bbresearch.minute_features.minute_features（研究用の direction.py と同じ）
- 足: bbdata.bars.aggregate_trades + finalize_bars（研究用の build_bars と同じ集計）
- 木の評価: bbresearch.treeeval（LightGBM を同梱しない。JSON の木を純粋 Python で評価する）
モデルは reports/direction_deploy/model_B.json（direction_deploy.yml が学習して書き出す）。numpy と pandas が要る。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

MODEL_PATHS = (Path(__file__).parent / "model_B.json",
               Path(__file__).resolve().parents[2] / "reports" / "direction_deploy" / "model_B.json")


def load_model() -> dict | None:
    for p in MODEL_PATHS:
        if p.exists():
            return json.loads(p.read_text())
    return None


def predict_minutes(rows: list[list], grid_ms: list[int], model: dict) -> dict[int, float | None]:
    """rows は [id, ts, price, amount, is_buy] の時刻順。grid_ms（分の開始）ごとの確率。足りなければ None。"""
    import numpy as np
    import pandas as pd

    from bbdata.bars import aggregate_trades, finalize_bars
    from bbresearch.labeling import TradeTape
    from bbresearch.minute_features import BAR_MS, minute_features
    from bbresearch.treeeval import predict_proba

    if not rows or not grid_ms:
        return {t: None for t in grid_ms}
    df = pd.DataFrame(rows, columns=["transaction_id", "executed_at", "price", "amount", "is_buy"])
    df["side"] = np.where(df["is_buy"].astype(bool), "buy", "sell")
    df["ts"] = pd.to_datetime(df["executed_at"], unit="ms", utc=True)  # load_transactions と同じ列
    tape = TradeTape(ts=df["executed_at"].to_numpy("int64"), is_buy=df["is_buy"].to_numpy(bool),
                     price=df["price"].to_numpy("float64"), amount=df["amount"].to_numpy("float64"))
    t0 = int(df["executed_at"].iloc[0]) // BAR_MS * BAR_MS
    t1 = max(grid_ms) // BAR_MS * BAR_MS + BAR_MS
    bars = finalize_bars(aggregate_trades(df, "15min", pd.Timestamp(t0, unit="ms", tz="UTC"),
                                          pd.Timestamp(t1, unit="ms", tz="UTC"), model.get("large_trade_amount")))
    grid = np.array(sorted(grid_ms), dtype="int64")
    X = minute_features(tape, bars, grid, int(model.get("sigma_span", 96)), workers=1)
    feats = model["features"]
    missing = [c for c in feats if c not in X.columns]
    if missing:
        raise RuntimeError(f"特徴量が足りない: {missing[:5]}")
    Xm = X[feats].to_numpy(dtype="float64")
    out: dict[int, float | None] = {}
    for t, x in zip(grid.tolist(), Xm.tolist()):
        # 準備期間が足りない行（特徴量の半分より多くが NaN）は出さない
        if sum(1 for v in x if v is None or (isinstance(v, float) and math.isnan(v))) * 2 > len(x):
            out[t] = None
        else:
            out[t] = predict_proba(model["model"], x)
    return out
