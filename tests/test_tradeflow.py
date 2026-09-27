import numpy as np
import pandas as pd

from bbresearch.labeling import TradeTape
from bbresearch.tradeflow import flow_table, hourly_aggregates

from synth import synth_trades


def _tape(days=12, seed=0):
    t = synth_trades("2026-01-01", days, seed=seed, per_min=3, vol=0.002)
    t["amount"] = np.random.default_rng(seed).lognormal(-3, 1, len(t))
    return TradeTape.from_frame(t)


def test_hourly_aggregates_match_brute_force_and_no_lookahead():
    tape = _tape()
    idx = pd.date_range("2026-01-01", periods=12 * 24, freq="h", tz="UTC")
    agg = hourly_aggregates(tape, idx)
    k = 100
    t0 = idx[k].value // 1_000_000
    m = (tape.ts >= t0) & (tape.ts < t0 + 3_600_000)
    assert agg["n"].iloc[k] == m.sum()
    assert abs(agg["buy_vol"].iloc[k] - tape.amount[m & tape.is_buy].sum()) < 1e-9
    assert abs(agg["signed_vol"].iloc[k] - (tape.amount[m] * np.where(tape.is_buy[m], 1, -1)).sum()) < 1e-9
    close = pd.Series(agg["last"].ffill().to_numpy(), index=idx)
    f = flow_table(tape, idx, close)
    assert f.shape[1] >= 30 and f.iloc[200:].notna().all().all()
    # 未来の約定を書き換えても過去の行は変わらない
    cut = idx[200].value // 1_000_000
    price2 = tape.price.copy()
    price2[tape.ts >= cut] *= 1.5
    amount2 = tape.amount.copy()
    amount2[tape.ts >= cut] *= 3
    tape2 = TradeTape(ts=tape.ts, is_buy=~tape.is_buy if False else tape.is_buy, price=price2, amount=amount2)
    close2 = close.copy()
    close2[idx >= idx[200]] *= 1.5
    f2 = flow_table(tape2, idx, close2)
    diff = (f.iloc[:200] - f2.iloc[:200]).abs().max()
    assert (diff.fillna(0) < 1e-9).all(), diff[diff > 1e-9]
    assert f["f_imb_vol24"].dropna().between(-1, 1).all() and f["f_flip_rate1"].dropna().between(0, 1).all()
