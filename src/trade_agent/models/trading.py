"""Orders, positions and trades.

Spot, long only: an "entry" is always a buy and an "exit" is always a sell
(spec 2). Nothing here has a short branch, deliberately — a shorting bug would
be silent otherwise.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ..money import ZERO, dec


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    LIMIT = "limit"
    MARKET = "market"


class OrderPurpose(StrEnum):
    ENTRY = "entry"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"


class OrderStatus(StrEnum):
    """Local lifecycle. `PENDING` is written *before* the exchange call so a
    crash between write and call is recoverable (spec 8)."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    UNKNOWN = "unknown"

    @property
    def is_terminal(self) -> bool:
        return self in {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED}

    @property
    def is_open(self) -> bool:
        return self in {OrderStatus.PENDING, OrderStatus.SUBMITTED,
                        OrderStatus.PARTIALLY_FILLED, OrderStatus.UNKNOWN}


# bitbank order status -> local status
EXCHANGE_STATUS_MAP = {
    "INACTIVE": OrderStatus.SUBMITTED,
    "UNFILLED": OrderStatus.SUBMITTED,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
    "FULLY_FILLED": OrderStatus.FILLED,
    "CANCELED_UNFILLED": OrderStatus.CANCELED,
    "CANCELED_PARTIALLY_FILLED": OrderStatus.CANCELED,
    "REJECTED": OrderStatus.REJECTED,
}


def new_client_order_id() -> str:
    """Client-side unique id — the idempotency key for the whole system."""
    return uuid.uuid4().hex


class OrderIntent(_Model):
    """What we are about to ask the exchange for."""

    client_order_id: str = Field(default_factory=new_client_order_id)
    cycle_id: str
    pair: str
    side: Side
    order_type: OrderType
    qty_btc: Decimal
    price: Decimal | None = None
    post_only: bool = True
    purpose: OrderPurpose = OrderPurpose.ENTRY
    probe: bool = False
    trade_id: str | None = Field(
        default=None, description="the trade this order belongs to; set on exits")

    def notional_jpy(self) -> Decimal:
        return (self.price or ZERO) * self.qty_btc


class OrderRecord(_Model):
    """Persisted order state. One row per client_order_id, forever."""

    client_order_id: str
    cycle_id: str
    pair: str
    side: Side
    order_type: OrderType
    qty_btc: Decimal
    price: Decimal | None = None
    post_only: bool = True
    purpose: OrderPurpose = OrderPurpose.ENTRY
    probe: bool = False
    trade_id: str | None = None
    status: OrderStatus = OrderStatus.PENDING
    exchange_order_id: str | None = None
    executed_qty_btc: Decimal = ZERO
    average_price: Decimal | None = None
    fee_jpy: Decimal = ZERO
    created_at: datetime
    updated_at: datetime
    submitted_at: datetime | None = None
    closed_at: datetime | None = None
    error: str | None = None

    @classmethod
    def from_intent(cls, intent: OrderIntent, now: datetime) -> "OrderRecord":
        return cls(
            client_order_id=intent.client_order_id,
            cycle_id=intent.cycle_id,
            pair=intent.pair,
            side=intent.side,
            order_type=intent.order_type,
            qty_btc=intent.qty_btc,
            price=intent.price,
            post_only=intent.post_only,
            purpose=intent.purpose,
            probe=intent.probe,
            trade_id=intent.trade_id,
            created_at=now,
            updated_at=now,
        )

    @property
    def remaining_qty_btc(self) -> Decimal:
        return max(ZERO, self.qty_btc - self.executed_qty_btc)


class Position(_Model):
    """The single open long position (spec 6: at most one)."""

    trade_id: str
    cycle_id: str
    pair: str
    qty_btc: Decimal
    entry_price: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    opened_at: datetime
    probe: bool = False
    entry_fee_jpy: Decimal = ZERO
    judge_output_id: str | None = None
    exit_order_id: str | None = Field(
        default=None, description="client_order_id of the in-flight exit, if any")
    exit_reason: str | None = None

    def unrealized_pnl_jpy(self, price) -> Decimal:
        return (dec(price) - self.entry_price) * self.qty_btc

    def risk_jpy(self) -> Decimal:
        return (self.entry_price - self.stop_loss) * self.qty_btc


class TradeRecord(_Model):
    """A completed (or in-flight) round trip. This is the audit record: every
    number a report quotes must be derivable from these rows (spec 14)."""

    trade_id: str
    cycle_id: str
    pair: str
    side: Side = Side.BUY
    probe: bool = False
    qty_btc: Decimal
    entry_price: Decimal
    entry_order_id: str
    entry_at: datetime
    stop_loss: Decimal
    take_profit: Decimal
    exit_price: Decimal | None = None
    exit_order_id: str | None = None
    exit_at: datetime | None = None
    exit_reason: str | None = None
    fee_jpy: Decimal = ZERO
    gross_pnl_jpy: Decimal | None = None
    net_pnl_jpy: Decimal | None = None
    judge_output_id: str | None = None
    regime: str | None = None
    closed: bool = False

    def is_win(self) -> bool | None:
        if self.net_pnl_jpy is None:
            return None
        return self.net_pnl_jpy > 0

    def r_multiple(self) -> Decimal | None:
        """Realised profit measured in units of the risk originally taken."""
        if self.net_pnl_jpy is None:
            return None
        risk = (self.entry_price - self.stop_loss) * self.qty_btc
        if risk <= 0:
            return None
        return self.net_pnl_jpy / risk


class ExecutionPlan(_Model):
    """The single adopted plan handed from the decision cycle to the executor."""

    cycle_id: str
    trade_id: str
    entry: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    qty_btc: Decimal
    risk_jpy: Decimal
    probe: bool = False
    regime: str | None = None
    thesis: str = ""
    judge_output_id: str | None = None
    consensus: Decimal | None = None
    client_order_id: str | None = Field(
        default=None,
        description="deterministic idempotency key; derived from cycle_id so a "
                    "repeat of the same cycle cannot place a second order")
