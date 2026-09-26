"""月初のリバランスの計算（発注はしない。scripts/rebalance.py が呼ぶ）。2026-09-27 オーナー指示。

- 資金の範囲: 候補銘柄の残高 + JPY の全額（オーナー決定）。候補外の銘柄は触らない。
- 目標: bbresearch.portfolio の最小分散（BTC 60%・他 40% まで、5 銘柄以内）を目標ボラ 30% に縮め、残りは JPY。
- 注文は成行（オーナー決定）。売りを先に出し、その JPY で買う。
- 小さすぎる差（総額の min_trade_frac 未満）は動かさない。数量は amount_digits で切り捨て、unit_amount 未満は出さない。
- 買いは手数料と滑りのぶん（buy_buffer）だけ少なめにし、JPY が足りない場合は比例して減らす。

この関数は数量を返すが、呼び出し側は数量・金額をログに出さない（公開リポジトリ）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PairSpec:
    pair: str
    unit_amount: float      # 最小数量
    amount_digits: int      # 数量の小数桁
    market_max_amount: float | None = None


@dataclass(frozen=True)
class Order:
    pair: str
    side: str        # buy / sell
    amount: float    # 基軸通貨の数量
    frac: float      # 総額に対する割合（ログに出してよい）


def floor_amount(x: float, digits: int) -> float:
    q = 10 ** digits
    return math.floor(x * q + 1e-9) / q


def portfolio_value(holdings: dict[str, float], prices: dict[str, float], jpy: float) -> tuple[float, dict[str, float]]:
    """総額（JPY）と、ペアごとの現在の割合。holdings は基軸通貨（例 btc）→ 数量、prices はペア → 価格。"""
    values = {pair: holdings.get(pair.split("_")[0], 0.0) * px for pair, px in prices.items()}
    total = jpy + sum(values.values())
    if total <= 0:
        return 0.0, {}
    return total, {pair: v / total for pair, v in values.items()}


def plan_orders(target: dict[str, float], holdings: dict[str, float], prices: dict[str, float], jpy: float,
                specs: dict[str, PairSpec], min_trade_frac: float = 0.01, buy_buffer: float = 0.003) -> list[Order]:
    """目標の割合（ペア → 割合。JPY は含めない）に近づける成行注文の一覧（売りが先）。"""
    total, current = portfolio_value(holdings, prices, jpy)
    if total <= 0:
        return []
    sells: list[Order] = []
    buys: list[tuple[str, float]] = []
    for pair, px in prices.items():
        delta = (target.get(pair, 0.0) - current.get(pair, 0.0)) * total  # JPY
        if abs(delta) < min_trade_frac * total or px <= 0:
            continue
        spec = specs[pair]
        if delta < 0:
            amt = min(floor_amount(-delta / px, spec.amount_digits), holdings.get(pair.split("_")[0], 0.0))
            if spec.market_max_amount:
                amt = min(amt, spec.market_max_amount)
            if amt >= spec.unit_amount:
                sells.append(Order(pair, "sell", amt, amt * px / total))
        else:
            buys.append((pair, delta))
    # 売った後の JPY（手数料ぶんを引く）で買う。足りなければ比例して減らす
    jpy_after = jpy + sum(o.amount * prices[o.pair] for o in sells) * (1 - buy_buffer)
    need = sum(d for _, d in buys)
    scale = min(1.0, jpy_after / need) if need > 0 else 1.0
    out = list(sells)
    for pair, delta in buys:
        spec = specs[pair]
        amt = floor_amount(delta * scale * (1 - buy_buffer) / prices[pair], spec.amount_digits)
        if spec.market_max_amount:
            amt = min(amt, spec.market_max_amount)
        if amt >= spec.unit_amount:
            out.append(Order(pair, "buy", amt, amt * prices[pair] / total))
    return out


def describe(orders: list[Order], total_frac_cash: float) -> str:
    """ログ用（割合だけ。数量・金額は含めない）。"""
    if not orders:
        return f"注文なし（目標との差が小さい）。JPY 目標 {total_frac_cash:.0%}"
    lines = [f"{o.pair} {o.side} 総額の {o.frac:.1%}" for o in orders]
    return "\n".join(lines) + f"\nJPY 目標 {total_frac_cash:.0%}"
