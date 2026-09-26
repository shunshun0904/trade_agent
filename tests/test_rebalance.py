import numpy as np
import pandas as pd
import pytest

from bbresearch.portfolio import current_weights
from bbresearch.rebalance import Order, PairSpec, describe, floor_amount, plan_orders, portfolio_value

SPECS = {"btc_jpy": PairSpec("btc_jpy", 0.0001, 4), "trx_jpy": PairSpec("trx_jpy", 1.0, 0, market_max_amount=100000.0),
         "bnb_jpy": PairSpec("bnb_jpy", 0.001, 3), "eth_jpy": PairSpec("eth_jpy", 0.0001, 4)}
PRICES = {"btc_jpy": 13_000_000.0, "trx_jpy": 50.0, "bnb_jpy": 150_000.0, "eth_jpy": 600_000.0}


def test_portfolio_value_and_current_fractions():
    total, cur = portfolio_value({"btc": 0.01, "eth": 0.5}, PRICES, 100_000.0)
    assert total == pytest.approx(130_000 + 300_000 + 100_000)
    assert cur["btc_jpy"] == pytest.approx(130_000 / 530_000) and cur["trx_jpy"] == 0.0


def test_plan_sells_first_then_buys_within_available_jpy():
    holdings = {"btc": 0.0, "trx": 0.0, "bnb": 0.0, "eth": 1.0}  # ETH 60 万円だけ持っている
    target = {"btc_jpy": 0.42, "trx_jpy": 0.37, "bnb_jpy": 0.13}  # JPY 8%
    orders = plan_orders(target, holdings, PRICES, 0.0, SPECS, min_trade_frac=0.01, buy_buffer=0.003)
    sides = [o.side for o in orders]
    assert sides == ["sell", "buy", "buy", "buy"] and orders[0].pair == "eth_jpy"
    assert orders[0].amount == 1.0  # 全部売る
    total = 600_000.0
    jpy_after = 600_000.0 * (1 - 0.003)
    spent = sum(o.amount * PRICES[o.pair] for o in orders if o.side == "buy")
    assert spent <= jpy_after  # 売った JPY を超えて買わない
    # 買いの数量は桁で切り捨て、最小数量以上
    for o in orders:
        if o.side == "buy":
            assert o.amount >= SPECS[o.pair].unit_amount
            assert o.amount == floor_amount(o.amount, SPECS[o.pair].amount_digits)
    # 割合は目標に近い（手数料のぶんだけ少なめ）
    got = {o.pair: o.amount * PRICES[o.pair] / total for o in orders if o.side == "buy"}
    for p, w in target.items():
        assert w * 0.98 <= got[p] <= w


def test_plan_skips_small_differences_and_respects_holdings():
    holdings = {"btc": 0.0323, "trx": 7400.0, "bnb": 0.87, "eth": 0.0}  # ほぼ目標どおり（総額 100 万円）
    jpy = 80_000.0
    target = {"btc_jpy": 0.42, "trx_jpy": 0.37, "bnb_jpy": 0.13}
    orders = plan_orders(target, holdings, PRICES, jpy, SPECS, min_trade_frac=0.01)
    assert orders == []  # 1% 未満の差は動かさない
    # 売る数量は保有を超えない。市場注文の上限も守る
    orders = plan_orders({"btc_jpy": 0.0, "trx_jpy": 0.0, "bnb_jpy": 0.0}, holdings, PRICES, jpy, SPECS)
    by = {o.pair: o for o in orders}
    assert all(o.side == "sell" for o in orders)
    assert by["btc_jpy"].amount <= holdings["btc"] and by["trx_jpy"].amount <= 7400.0
    assert by["trx_jpy"].amount == 7400.0 and by["bnb_jpy"].amount == 0.87


def test_describe_has_only_fractions():
    text = describe([Order("btc_jpy", "buy", 0.0123, 0.2), Order("trx_jpy", "sell", 1234.0, 0.05)], 0.3)
    assert "0.0123" not in text and "1234" not in text
    assert "btc_jpy buy 総額の 20.0%" in text and "JPY 目標 30%" in text


def test_current_weights_scales_to_target_vol():
    rng = np.random.default_rng(11)
    idx = pd.date_range("2024-01-01", periods=400, freq="D", tz="UTC")
    cols = ["btc_jpy", "a_jpy", "b_jpy", "c_jpy"]
    r = rng.normal(0, 0.04, size=(400, 4)) + rng.normal(0, 0.03, size=(400, 1))
    close = pd.DataFrame(np.exp(np.cumsum(r, axis=0)) * 100, index=idx, columns=cols)
    tw = current_weights(close, k_max=3, w_max=0.5, caps={"btc_jpy": 0.6}, target_vol=0.3)
    assert 0 < tw["cash"] < 1 and abs(sum(tw["weights"].values()) + tw["cash"] - 1) < 1e-9
    assert tw["est_vol"] == pytest.approx(0.3) and tw["est_vol_full"] > 0.3
    full = current_weights(close, k_max=3, w_max=0.5, caps={"btc_jpy": 0.6}, target_vol=None)
    assert full["cash"] == 0.0 and set(full["weights"]) == set(tw["weights"])


def _close(seed=11, n=400):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    cols = ["btc_jpy", "a_jpy", "b_jpy", "c_jpy"]
    r = rng.normal(0, 0.04, size=(n, 4)) + rng.normal(0, 0.03, size=(n, 1))
    return pd.DataFrame(np.exp(np.cumsum(r, axis=0)) * 100, index=idx, columns=cols)


def test_scale_weights_applies_trend_filter_from_past_closes_only():
    from bbresearch.portfolio import scale_weights

    close = _close()
    rel = {"btc_jpy": 0.5, "a_jpy": 0.5}
    plain = scale_weights(close, rel, target_vol=0.3, est_days=365)
    assert plain["trend"] is None and plain["est_vol"] == pytest.approx(0.3)
    up = close.copy()
    up.iloc[-1, 0] = up["btc_jpy"].iloc[-200:].mean() * 1.5  # 最後の終値を 200 日線の上に
    on = scale_weights(up, rel, target_vol=0.3, est_days=365, trend_days=200, trend_scale=0.0)
    assert on["trend"] is True and on["cash"] < 1.0
    down = close.copy()
    down.iloc[-1, 0] = down["btc_jpy"].iloc[-200:].mean() * 0.5
    off = scale_weights(down, rel, target_vol=0.3, est_days=365, trend_days=200, trend_scale=0.0)
    assert off["trend"] is False and off["cash"] == 1.0 and all(w == 0 for w in off["weights"].values())
    half = scale_weights(down, rel, target_vol=0.3, est_days=365, trend_days=200, trend_scale=0.5)
    assert half["weights"]["btc_jpy"] == pytest.approx(on["weights"]["btc_jpy"] * 0.5, rel=0.2)


def test_target_for_today_weekly_keeps_the_monthly_relative_weights():
    from scripts.rebalance import target_for_today

    close = _close()
    cfg = {"target_vol": 0.3, "est_days": 365, "k_max": 3, "w_max": 0.5, "caps": {"btc_jpy": 0.6}, "trend_days": None}
    monthly = target_for_today(cfg, close, [], "monthly")
    assert monthly["kind"] == "monthly" and abs(sum(monthly["relative"].values()) - 1) < 1e-9
    # weekly は直近の monthly の相対の重みを使う（銘柄を選び直さない）
    fake_hist = [{"date": "2026-10-01", "kind": "monthly", "relative": {"a_jpy": 0.7, "b_jpy": 0.3},
                  "target_weights": {"a_jpy": 0.35, "b_jpy": 0.15}, "target_cash": 0.5}]
    weekly = target_for_today(cfg, close, fake_hist, "weekly")
    assert weekly["kind"] == "weekly" and weekly["based_on"] == "2026-10-01" and set(weekly["weights"]) == {"a_jpy", "b_jpy"}
    assert weekly["weights"]["a_jpy"] / weekly["weights"]["b_jpy"] == pytest.approx(0.7 / 0.3)
    # 古い記録（relative なし）は目標の重みから相対の重みを戻す
    old = [{"date": "2026-09-01", "target_weights": {"a_jpy": 0.35, "b_jpy": 0.15}, "target_cash": 0.5}]
    weekly2 = target_for_today(cfg, close, old, "weekly")
    assert weekly2["kind"] == "weekly" and weekly2["weights"]["a_jpy"] / weekly2["weights"]["b_jpy"] == pytest.approx(0.7 / 0.3)
    # monthly の記録がなければ monthly と同じ
    assert target_for_today(cfg, close, [], "weekly")["kind"] == "monthly"
