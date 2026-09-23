import numpy as np
import pandas as pd
import pytest

from bbdata.bars import aggregate_trades, build_bars, finalize_bars
from bbdata.download import trades_path, transactions_to_frame
from bbdata.validate import compare_with_official

T0 = pd.Timestamp("2026-01-01T00:00:00Z")


def ms(ts: pd.Timestamp) -> int:
    return int(ts.value // 1_000_000)


def make_trades(rows):
    """rows: (秒オフセット, side, price, amount)"""
    df = transactions_to_frame(
        [
            {"transaction_id": i, "side": s, "price": str(p), "amount": str(a),
             "executed_at": ms(T0 + pd.Timedelta(seconds=sec))}
            for i, (sec, s, p, a) in enumerate(rows)
        ]
    )
    df["ts"] = pd.to_datetime(df["executed_at"], unit="ms", utc=True)
    return df


TRADES = [
    (10, "buy", 100.0, 1.0),
    (300, "sell", 101.0, 2.0),
    (899, "buy", 99.0, 1.0),
    (900, "sell", 102.0, 0.5),   # ちょうど 00:15:00 → 2本目
    # 00:30〜00:45 は約定なし
    (2700, "buy", 103.0, 1.0),
]


def bars_1h():
    raw = aggregate_trades(make_trades(TRADES), "15min", T0, T0 + pd.Timedelta(hours=1), large_trade_amount=2.0)
    return finalize_bars(raw)


def test_ohlcv_and_flow():
    b = bars_1h()
    assert len(b) == 4
    first = b.iloc[0]
    assert (first.open, first.high, first.low, first.close) == (100.0, 101.0, 99.0, 99.0)
    assert first.volume == 4.0 and first.n_trades == 3 and first.n_buy == 2 and first.n_sell == 1
    assert first.buy_volume == 2.0 and first.sell_volume == 2.0
    assert first.vwap == pytest.approx((100 * 1 + 101 * 2 + 99 * 1) / 4)
    assert first.volume_imbalance == 0.0
    assert first.max_buy_price == 100.0 and first.min_sell_price == 101.0
    assert first.large_volume == 2.0


def test_left_closed_boundary():
    second = bars_1h().iloc[1]
    assert second.name == T0 + pd.Timedelta(minutes=15)
    assert second.open == second.close == 102.0
    assert second.volume_imbalance == -1.0
    assert np.isnan(second.max_buy_price)


def test_empty_bar_filled_from_past_only():
    third = bars_1h().iloc[2]
    assert third.is_empty
    assert (third.open, third.high, third.low, third.close, third.vwap) == (102.0,) * 5
    assert third.volume == 0.0 and third.n_trades == 0 and third.volume_imbalance == 0.0
    assert np.isnan(third.min_sell_price)  # 約定判定用の列は埋めない


def test_leading_bars_without_prior_trade_stay_nan():
    raw = aggregate_trades(make_trades([(1000, "buy", 1.0, 1.0)]), "15min", T0, T0 + pd.Timedelta(minutes=30))
    b = finalize_bars(raw)
    assert np.isnan(b.iloc[0].close) and b.iloc[1].close == 1.0


def test_freq_must_divide_day(tmp_path):
    with pytest.raises(ValueError):
        build_bars(tmp_path, "btc_jpy", T0, T0 + pd.Timedelta(days=1), freq="7min")


def _write_day(root, day, rows):
    df = transactions_to_frame(rows)
    p = trades_path(root, "btc_jpy", day)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def test_build_bars_dedup_padding_and_chunking(tmp_path):
    t = lambda h: ms(T0 + pd.Timedelta(hours=h))
    # 1/1 の約定の一部が 12/31 のファイルに入っている（日付境界が UTC でない場合を想定）
    _write_day(tmp_path, pd.Timestamp("2025-12-31").date(), [
        {"transaction_id": 1, "side": "buy", "price": "10", "amount": "1", "executed_at": t(-1)},
        {"transaction_id": 2, "side": "buy", "price": "11", "amount": "1", "executed_at": t(1)},
    ])
    _write_day(tmp_path, pd.Timestamp("2026-01-01").date(), [
        {"transaction_id": 2, "side": "buy", "price": "11", "amount": "1", "executed_at": t(1)},  # 重複
        {"transaction_id": 3, "side": "sell", "price": "12", "amount": "2", "executed_at": t(30)},
    ])
    _write_day(tmp_path, pd.Timestamp("2026-01-02").date(), [
        {"transaction_id": 4, "side": "sell", "price": "13", "amount": "1", "executed_at": t(40)},
    ])
    end = T0 + pd.Timedelta(days=2)
    a = build_bars(tmp_path, "btc_jpy", T0, end, "1h", chunk_days=1)
    b = build_bars(tmp_path, "btc_jpy", T0, end, "1h", chunk_days=7)
    pd.testing.assert_frame_equal(a, b)
    assert a["n_trades"].sum() == 3  # id=1 は期間外、id=2 の重複は1件に
    assert a.loc[T0 + pd.Timedelta(hours=1), "close"] == 11.0
    assert a.loc[T0 + pd.Timedelta(hours=29), "close"] == 11.0  # 日をまたいで前方埋め
    assert np.isnan(a.iloc[0]["close"])


def test_compare_with_official_detects_shift():
    b = bars_1h()
    official = b[["open", "high", "low", "close", "volume"]][~b["is_empty"]].copy()
    official.index = official.index + pd.Timedelta(minutes=15)  # 終了時刻表記だった場合
    res = compare_with_official(b, official, max_shift=1)
    assert res.loc[-1, "ohlc_match_rate"] == 1.0
    assert res.loc[0, "ohlc_match_rate"] < 1.0
