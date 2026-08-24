"""Paper exchange — Phase 1 (spec 13).

Real public market data, simulated private endpoints. Nothing here can reach
bitbank's order API, which is the point: Phase 1 must be structurally unable to
place an order, not merely configured not to.

Fill model, deliberately pessimistic:
  * a resting PostOnly buy fills only if the market actually traded *below* the
    limit price after the order was placed (touching it is not enough);
  * a market sell fills at the best bid, minus the taker fee;
  * partial fills are not simulated — an order is unfilled or fully filled.

The pessimism matters: Phase 1's exit criterion is "expected value positive
after fees" (spec 13), and an optimistic simulator would manufacture that.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Callable

from pydantic import BaseModel, ConfigDict

from ..errors import ExchangeError, InsufficientFunds
from ..models.market import Candle
from ..models.trading import OrderIntent, OrderType, Side
from ..money import ZERO, dec
from ..timeutil import UTC, to_ms
from .base import Balance, PairSettings


class PaperOrder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str
    pair: str
    side: Side
    order_type: OrderType
    price: Decimal | None
    amount: Decimal
    post_only: bool
    status: str = "UNFILLED"
    executed_amount: Decimal = ZERO
    average_price: Decimal | None = None
    fee_quote: Decimal = ZERO
    ordered_at: datetime
    filled_at: datetime | None = None


class PaperAccount(BaseModel):
    """Simulated balances plus the open simulated orders."""

    model_config = ConfigDict(extra="forbid")

    jpy_free: Decimal
    jpy_locked: Decimal = ZERO
    btc_free: Decimal = ZERO
    btc_locked: Decimal = ZERO
    orders: dict[str, PaperOrder] = {}
    next_order_id: int = 1


class PaperExchange:
    """Satisfies :class:`ExchangeClient` for everything the system calls.

    `load_account` / `save_account` let Lambda persist simulated balances
    between invocations; the default keeps them in memory for tests.
    """

    def __init__(self, public, *, pair: str, maker_fee_rate: Decimal,
                 taker_fee_rate: Decimal, min_order_btc: Decimal,
                 initial_jpy: Decimal,
                 load_account: Callable[[], PaperAccount | None] | None = None,
                 save_account: Callable[[PaperAccount], None] | None = None,
                 now: Callable[[], datetime] | None = None):
        self.public = public
        self.pair = pair
        self.maker_fee_rate = dec(maker_fee_rate)
        self.taker_fee_rate = dec(taker_fee_rate)
        self.min_order_btc = dec(min_order_btc)
        self._load = load_account
        self._save = save_account
        self._now = now or (lambda: datetime.now(UTC))
        loaded = load_account() if load_account else None
        self.account = loaded or PaperAccount(jpy_free=dec(initial_jpy))

    # -- public passthrough -------------------------------------------------

    def get_ticker(self) -> dict:
        return self.public.get_ticker()

    def get_depth(self) -> dict:
        return self.public.get_depth()

    def get_candles(self, candle_type: str, day: date | int) -> list[Candle]:
        return self.public.get_candles(candle_type, day)

    def get_pair_settings(self) -> PairSettings:
        return self.public.get_pair_settings()

    def get_transactions(self, day: date | None = None) -> list[dict]:
        return self.public.get_transactions(day)

    def get_circuit_break_info(self) -> dict:
        return self.public.get_circuit_break_info()

    # -- simulated private --------------------------------------------------

    def get_balances(self) -> dict[str, Balance]:
        self._settle()
        acc = self.account
        return {
            "jpy": Balance(asset="jpy", free=acc.jpy_free, locked=acc.jpy_locked,
                           onhand=acc.jpy_free + acc.jpy_locked),
            "btc": Balance(asset="btc", free=acc.btc_free, locked=acc.btc_locked,
                           onhand=acc.btc_free + acc.btc_locked),
        }

    def create_order(self, intent: OrderIntent) -> dict:
        if intent.qty_btc < self.min_order_btc:
            raise ExchangeError(
                f"amount {intent.qty_btc} below minimum {self.min_order_btc}")
        acc = self.account
        now = self._now()
        if intent.side is Side.BUY:
            if intent.order_type is not OrderType.LIMIT or intent.price is None:
                raise ExchangeError("paper entries must be limit orders")
            cost = intent.price * intent.qty_btc
            if cost > acc.jpy_free:
                raise InsufficientFunds(
                    f"need {cost} JPY, have {acc.jpy_free}")
            acc.jpy_free -= cost
            acc.jpy_locked += cost
        else:
            if intent.qty_btc > acc.btc_free:
                raise InsufficientFunds(
                    f"need {intent.qty_btc} BTC, have {acc.btc_free}")
            acc.btc_free -= intent.qty_btc
            acc.btc_locked += intent.qty_btc

        order_id = str(acc.next_order_id)
        acc.next_order_id += 1
        acc.orders[order_id] = PaperOrder(
            order_id=order_id, pair=intent.pair, side=intent.side,
            order_type=intent.order_type, price=intent.price,
            amount=intent.qty_btc, post_only=intent.post_only, ordered_at=now)
        self._settle()
        return self._as_exchange_dict(acc.orders[order_id])

    def cancel_order(self, exchange_order_id: str) -> dict:
        acc = self.account
        order = acc.orders.get(str(exchange_order_id))
        if order is None:
            raise ExchangeError(f"unknown order {exchange_order_id}", code=50008)
        if order.status == "UNFILLED":
            self._release(order)
            order.status = "CANCELED_UNFILLED"
            self._persist()
        return self._as_exchange_dict(order)

    def get_order(self, exchange_order_id: str) -> dict:
        self._settle()
        order = self.account.orders.get(str(exchange_order_id))
        if order is None:
            raise ExchangeError(f"unknown order {exchange_order_id}", code=50008)
        return self._as_exchange_dict(order)

    def get_active_orders(self) -> list[dict]:
        self._settle()
        return [self._as_exchange_dict(o) for o in self.account.orders.values()
                if o.status in {"UNFILLED", "PARTIALLY_FILLED"}]

    def get_trades_for_order(self, exchange_order_id: str) -> list[dict]:
        order = self.account.orders.get(str(exchange_order_id))
        if order is None or order.status != "FULLY_FILLED":
            return []
        return [{
            "trade_id": int(order.order_id),
            "pair": order.pair,
            "order_id": int(order.order_id),
            "side": str(order.side),
            "type": str(order.order_type),
            "amount": str(order.executed_amount),
            "price": str(order.average_price),
            "maker_taker": "maker" if order.post_only else "taker",
            "fee_amount_quote": str(order.fee_quote),
            "executed_at": to_ms(order.filled_at or order.ordered_at),
        }]

    # -- fill simulation ----------------------------------------------------

    def _settle(self) -> None:
        """Advance every resting order against the market since it was placed."""
        open_orders = [o for o in self.account.orders.values() if o.status == "UNFILLED"]
        if not open_orders:
            return
        candles = self._recent_candles(min(o.ordered_at for o in open_orders))
        ticker = None
        changed = False
        for order in open_orders:
            window = [c for c in candles if c.opened_at >= order.ordered_at]
            if order.order_type is OrderType.MARKET:
                if ticker is None:
                    ticker = self.public.get_ticker()
                price = dec(ticker["buy"] if order.side is Side.SELL else ticker["sell"])
                self._fill(order, price, taker=True)
                changed = True
                continue
            if not window or order.price is None:
                continue
            if order.side is Side.BUY and min(c.low for c in window) < order.price:
                self._fill(order, order.price, taker=False)
                changed = True
            elif order.side is Side.SELL and max(c.high for c in window) > order.price:
                self._fill(order, order.price, taker=False)
                changed = True
        if changed:
            self._persist()

    def _recent_candles(self, since: datetime) -> list[Candle]:
        """1-minute candles from `since` to now, across a day boundary if needed."""
        now = self._now()
        days = {since.date(), now.date()}
        rows: list[Candle] = []
        for day in sorted(days):
            try:
                rows.extend(self.public.get_candles("1min", day))
            except ExchangeError:
                continue
        cutoff = since - timedelta(minutes=1)
        return [c for c in rows if c.opened_at >= cutoff]

    def _fill(self, order: PaperOrder, price: Decimal, *, taker: bool) -> None:
        acc = self.account
        rate = self.taker_fee_rate if taker else self.maker_fee_rate
        notional = price * order.amount
        fee = notional * rate
        if order.side is Side.BUY:
            reserved = (order.price or price) * order.amount
            acc.jpy_locked -= reserved
            # Refund the difference between the reserve and the actual cost.
            acc.jpy_free += reserved - notional - fee
            acc.btc_free += order.amount
        else:
            acc.btc_locked -= order.amount
            acc.jpy_free += notional - fee
        order.status = "FULLY_FILLED"
        order.executed_amount = order.amount
        order.average_price = price
        order.fee_quote = fee
        order.filled_at = self._now()

    def _release(self, order: PaperOrder) -> None:
        acc = self.account
        if order.side is Side.BUY:
            reserved = (order.price or ZERO) * order.amount
            acc.jpy_locked -= reserved
            acc.jpy_free += reserved
        else:
            acc.btc_locked -= order.amount
            acc.btc_free += order.amount

    def _persist(self) -> None:
        if self._save:
            self._save(self.account)

    def _as_exchange_dict(self, order: PaperOrder) -> dict:
        return {
            "order_id": int(order.order_id),
            "pair": order.pair,
            "side": str(order.side),
            "type": str(order.order_type),
            "start_amount": str(order.amount),
            "remaining_amount": str(order.amount - order.executed_amount),
            "executed_amount": str(order.executed_amount),
            "price": str(order.price) if order.price is not None else None,
            "post_only": order.post_only,
            "average_price": str(order.average_price or 0),
            "ordered_at": to_ms(order.ordered_at),
            "status": order.status,
        }
