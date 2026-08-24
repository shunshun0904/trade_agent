"""Order placement (spec 8).

The ordering of operations is the whole design:

    1. take the execution lock (conditional write, leased)
    2. reconcile anything already pending against the exchange
    3. re-fetch the price and abandon the cycle if it has moved
    4. write the order as `pending` (conditional create on the client id)
    5. only then call the exchange
    6. record the exchange id

Step 4 before step 5 is what makes a crash survivable: on restart there is a
`pending` row with no exchange id, and step 2 decides whether that order
reached the exchange. The code never places a second order to "make sure" —
that is the failure mode this whole sequence exists to prevent.

A note on matching: bitbank's create-order API takes no client-side order id,
so a pending row cannot be matched to an exchange order by id. Reconciliation
matches on (pair, side, amount, price, time window) instead, and when it cannot
decide it stops and asks rather than guessing (spec 6: when in doubt, do
nothing).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

from ..config import Config
from ..errors import DuplicateOrder, ExchangeError, LockNotAcquired
from ..models.trading import (
    EXCHANGE_STATUS_MAP,
    ExecutionPlan,
    OrderIntent,
    OrderPurpose,
    OrderRecord,
    OrderStatus,
    OrderType,
    Position,
    Side,
    TradeRecord,
)
from ..money import ZERO, dec, deviation_pct, quantize_price
from ..storage.base import LOCK_EXECUTION
from ..timeutil import Clock, from_ms

log = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    placed: bool
    reason: str = ""
    order: OrderRecord | None = None
    trade: TradeRecord | None = None
    requote_deviation_pct: Decimal | None = None
    reconciled: list[str] = field(default_factory=list)


class Executor:
    def __init__(self, *, exchange, store, config: Config, clock: Clock,
                 notifier=None, owner: str = "executor"):
        self.exchange = exchange
        self.store = store
        self.config = config
        self.clock = clock
        self.notifier = notifier
        self.owner = owner

    # -- locking -----------------------------------------------------------

    def acquire_lock(self, name: str = LOCK_EXECUTION) -> bool:
        return self.store.locks.acquire(
            name, self.owner, self.config.execution.lock_lease_seconds,
            self.clock.now())

    def release_lock(self, name: str = LOCK_EXECUTION) -> None:
        self.store.locks.release(name, self.owner)

    # -- reconciliation ----------------------------------------------------

    def reconcile_pending(self) -> list[str]:
        """Bring every locally-open order back in line with the exchange.

        Runs at the start of every function invocation (spec 8). Returns a
        human-readable note per order it touched.
        """
        notes: list[str] = []
        for record in self.store.orders.list_open():
            try:
                notes.extend(self._reconcile_one(record))
            except ExchangeError as exc:
                log.warning("could not reconcile %s: %s", record.client_order_id, exc)
                notes.append(f"{record.client_order_id}: reconcile failed ({exc})")
        return notes

    def _reconcile_one(self, record: OrderRecord) -> list[str]:
        now = self.clock.now()
        if record.exchange_order_id:
            before = record.status
            self._refresh(record)
            if record.status != before:
                return [f"{record.client_order_id}: {before} -> {record.status}"]
            return []

        # No exchange id: we either never called, or crashed mid-call.
        match = self._find_matching_exchange_order(record)
        if match is not None:
            record.exchange_order_id = str(match["order_id"])
            record.status = EXCHANGE_STATUS_MAP.get(match.get("status", ""),
                                                    OrderStatus.SUBMITTED)
            record.updated_at = now
            self.store.orders.update(record)
            self._refresh(record)
            return [f"{record.client_order_id}: adopted exchange order "
                    f"{record.exchange_order_id}"]

        age = now - record.created_at
        if age < timedelta(minutes=self.config.execution.pending_order_stale_minutes):
            return [f"{record.client_order_id}: pending, no match yet "
                    f"({int(age.total_seconds())}s old)"]

        # Old, and nothing on the exchange looks like it. Most likely the call
        # never went out. Mark it and tell the owner; do not re-place it.
        record.status = OrderStatus.UNKNOWN
        record.error = "no matching exchange order found within the stale window"
        record.updated_at = now
        self.store.orders.update(record)
        self._notify("pending order could not be reconciled",
                     f"client_order_id={record.client_order_id} "
                     f"({record.side} {record.qty_btc} @ {record.price}) has no "
                     "matching order on the exchange. No replacement was placed. "
                     "Check the exchange manually before resuming.")
        return [f"{record.client_order_id}: unresolved -> UNKNOWN"]

    def _find_matching_exchange_order(self, record: OrderRecord) -> dict | None:
        """Match a pending row to an exchange order by shape and timing.

        Ambiguity is treated as no match: two identical orders in the window
        mean we cannot tell which is ours, and adopting the wrong one is worse
        than leaving the row unresolved.
        """
        window_start = record.created_at - timedelta(minutes=2)
        candidates: list[dict] = []
        for order in self.exchange.get_active_orders():
            if self._shape_matches(record, order, window_start):
                candidates.append(order)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            log.warning("ambiguous reconcile for %s: %d candidates",
                        record.client_order_id, len(candidates))
        return None

    @staticmethod
    def _shape_matches(record: OrderRecord, order: dict,
                       window_start: datetime) -> bool:
        if str(order.get("side")) != str(record.side):
            return False
        if dec(order.get("start_amount") or 0) != record.qty_btc:
            return False
        if record.price is not None:
            price = order.get("price")
            if price is None or dec(price) != record.price:
                return False
        ordered_at = order.get("ordered_at")
        if ordered_at is None:
            return False
        return from_ms(ordered_at) >= window_start

    def _refresh(self, record: OrderRecord) -> OrderRecord:
        """Pull the exchange's view of an order into the local record."""
        if not record.exchange_order_id:
            return record
        data = self.exchange.get_order(record.exchange_order_id)
        record.status = EXCHANGE_STATUS_MAP.get(data.get("status", ""),
                                                OrderStatus.UNKNOWN)
        record.executed_qty_btc = dec(data.get("executed_amount") or 0)
        average = data.get("average_price")
        if average is not None and dec(average) > 0:
            record.average_price = dec(average)
        record.fee_jpy = self._fee_for(record)
        record.updated_at = self.clock.now()
        if record.status.is_terminal:
            record.closed_at = record.updated_at
        self.store.orders.update(record)
        return record

    def _fee_for(self, record: OrderRecord) -> Decimal:
        """Actual fee from the exchange when available, modelled otherwise."""
        if record.exchange_order_id:
            try:
                trades = self.exchange.get_trades_for_order(record.exchange_order_id)
            except ExchangeError:
                trades = []
            if trades:
                return sum((dec(t.get("fee_amount_quote") or 0) for t in trades), ZERO)
        rate = (self.config.exchange.maker_fee_rate if record.post_only
                else self.config.exchange.taker_fee_rate)
        price = record.average_price or record.price or ZERO
        return price * record.executed_qty_btc * rate

    # -- placing an entry --------------------------------------------------

    def place_entry(self, plan: ExecutionPlan) -> ExecutionResult:
        """Place the adopted plan's entry order, or refuse and say why."""
        cfg = self.config
        notes = self.reconcile_pending()

        if any(o.status.is_open for o in self.store.orders.list_open()):
            return ExecutionResult(False, "an order is already open", reconciled=notes)

        held = self._unclosed_entry()
        if held is not None:
            # An entry that filled but whose trade has not been booked closed is
            # a position, even if the state item has not caught up yet. Spec 6
            # allows exactly one at a time.
            return ExecutionResult(
                False, f"entry {held} has filled and its trade is still open",
                reconciled=notes)

        deviation = self._requote_deviation(plan.entry)
        if deviation is None:
            return ExecutionResult(False, "could not re-fetch the price",
                                   reconciled=notes)
        if deviation > cfg.execution.requote_max_deviation_pct:
            return ExecutionResult(
                False,
                f"price moved {deviation:.3f}% since the decision "
                f"(limit {cfg.execution.requote_max_deviation_pct}%); skipping",
                requote_deviation_pct=deviation, reconciled=notes)

        intent = OrderIntent(
            **({"client_order_id": plan.client_order_id}
               if plan.client_order_id else {}),
            cycle_id=plan.cycle_id,
            pair=cfg.exchange.pair,
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            qty_btc=plan.qty_btc,
            price=quantize_price(plan.entry, cfg.exchange.price_digits),
            post_only=True,
            purpose=OrderPurpose.ENTRY,
            probe=plan.probe,
            trade_id=plan.trade_id,
        )
        record = self.submit(intent)
        return ExecutionResult(True, "entry order placed", order=record,
                               requote_deviation_pct=deviation, reconciled=notes)

    def _unclosed_entry(self) -> str | None:
        """A filled entry order whose trade row is not closed yet."""
        for record in self.store.orders.list_recent(50):
            if record.purpose is not OrderPurpose.ENTRY:
                continue
            if record.executed_qty_btc <= 0 or not record.trade_id:
                continue
            trade = self.store.trades.get(record.trade_id)
            if trade is None or not trade.closed:
                return record.client_order_id
        return None

    def submit(self, intent: OrderIntent) -> OrderRecord:
        """Persist-then-send. The conditional create is the idempotency key."""
        now = self.clock.now()
        record = OrderRecord.from_intent(intent, now)
        self.store.orders.put_pending(record)  # raises DuplicateOrder

        if self.config.system.paper_trading and not hasattr(self.exchange, "account"):
            # Live exchange wired up while paper trading is on: refuse rather
            # than send. The paper exchange has an `account` attribute; the
            # real client does not.
            record.status = OrderStatus.REJECTED
            record.error = "paper trading is enabled; refusing to send a live order"
            record.updated_at = now
            self.store.orders.update(record)
            return record

        try:
            response = self.exchange.create_order(intent)
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
            record.error = str(exc)
            record.status = OrderStatus.REJECTED
            record.updated_at = self.clock.now()
            self.store.orders.update(record)
            raise

        record.exchange_order_id = str(response.get("order_id"))
        record.status = EXCHANGE_STATUS_MAP.get(response.get("status", ""),
                                                OrderStatus.SUBMITTED)
        record.executed_qty_btc = dec(response.get("executed_amount") or 0)
        record.submitted_at = self.clock.now()
        record.updated_at = record.submitted_at
        self.store.orders.update(record)
        return record

    # -- lifecycle ---------------------------------------------------------

    def expire_stale_entries(self) -> list[str]:
        """Cancel PostOnly entries that never filled (spec 8: 60 minutes)."""
        notes: list[str] = []
        timeout = timedelta(minutes=self.config.execution.post_only_timeout_minutes)
        now = self.clock.now()
        for record in self.store.orders.list_open():
            if record.purpose is not OrderPurpose.ENTRY:
                continue
            reference = record.submitted_at or record.created_at
            if now - reference < timeout:
                continue
            if record.executed_qty_btc > 0:
                # A partial fill is a position, not a stale order. Cancel the
                # remainder and let the position manager size the exits to what
                # actually filled (spec 8).
                notes.append(self._cancel(record, "partially filled and expired"))
            else:
                notes.append(self._cancel(record, "unfilled past the PostOnly timeout"))
        return notes

    def cancel(self, record: OrderRecord, reason: str) -> str:
        return self._cancel(record, reason)

    def _cancel(self, record: OrderRecord, reason: str) -> str:
        if record.exchange_order_id:
            try:
                self.exchange.cancel_order(record.exchange_order_id)
            except ExchangeError as exc:
                log.warning("cancel failed for %s: %s", record.client_order_id, exc)
                return f"{record.client_order_id}: cancel failed ({exc})"
            self._refresh(record)
        if record.status.is_open:
            record.status = OrderStatus.CANCELED
            record.updated_at = self.clock.now()
            record.closed_at = record.updated_at
            self.store.orders.update(record)
        return f"{record.client_order_id}: canceled ({reason})"

    def _requote_deviation(self, planned_entry: Decimal) -> Decimal | None:
        try:
            ticker = self.exchange.get_ticker()
        except ExchangeError as exc:
            log.warning("requote check failed: %s", exc)
            return None
        return deviation_pct(planned_entry, dec(ticker["last"]))

    def _notify(self, subject: str, body: str) -> None:
        if self.notifier is not None:
            self.notifier.send(subject, body)


def build_position(plan: ExecutionPlan, record: OrderRecord,
                   now: datetime) -> Position:
    """Turn a filled (or partially filled) entry into the tracked position.

    Sizing follows what actually executed, not what was requested — spec 8's
    partial-fill rule.
    """
    qty = record.executed_qty_btc or record.qty_btc
    entry = record.average_price or record.price or plan.entry
    return Position(
        trade_id=plan.trade_id,
        cycle_id=plan.cycle_id,
        pair=record.pair,
        qty_btc=qty,
        entry_price=entry,
        stop_loss=plan.stop_loss,
        take_profit=plan.take_profit,
        opened_at=now,
        probe=plan.probe,
        entry_fee_jpy=record.fee_jpy,
        judge_output_id=plan.judge_output_id,
    )
