"""テスト用の合成データ（ネットワークに触れない）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bbdata.download import trades_path


def synth_trades(start: str, days: int, seed: int = 0, per_min: int = 4, vol: float = 0.0008) -> pd.DataFrame:
    """ランダムウォークの約定。side はランダム、価格は呼値 1 円。"""
    rng = np.random.default_rng(seed)
    n = days * 24 * 60 * per_min
    t0 = pd.Timestamp(start, tz="UTC").value // 1_000_000
    ts = t0 + np.sort(rng.integers(0, days * 86_400_000, size=n))
    steps = rng.normal(0, vol / np.sqrt(per_min), size=n)
    price = np.round(10_000_000 * np.exp(np.cumsum(steps)))
    return pd.DataFrame({
        "transaction_id": np.arange(1, n + 1, dtype="int64"),
        "side": np.where(rng.random(n) < 0.5, "buy", "sell").astype(object),
        "price": price.astype("float64"),
        "amount": rng.uniform(0.001, 0.2, size=n),
        "executed_at": ts.astype("int64"),
    })


def write_trades(root, pair: str, trades: pd.DataFrame) -> None:
    """UTC の日付ごとに保存する（取得時と同じレイアウト）。"""
    day = pd.to_datetime(trades["executed_at"], unit="ms", utc=True).dt.date
    for d, g in trades.groupby(day):
        p = trades_path(root, pair, d)
        p.parent.mkdir(parents=True, exist_ok=True)
        g.to_parquet(p, index=False)
