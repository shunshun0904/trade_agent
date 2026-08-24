"""Position lifecycle and the self-managed OCO (spec 8).

bitbank has `stop_loss` and `take_profit` order types, but on a spot account a
resting take-profit sell and a stop-loss sell would each need to reserve the
same BTC, so only one of the two can exist on the exchange at a time. There is
no native OCO. The stops are therefore evaluated locally on every 5-minute tick
(spec 9), which is also why the dead-man's-switch alarm on that tick is a
safety requirement rather than an operational nicety (spec 17.3): if the tick
stops running, nothing is watching the stop.

Priority when both levels are breached inside one tick interval: the stop wins.
We cannot tell from a 5-minute bar which came first, and assuming the profitable
one would systematically overstate results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from ..config import Config
from ..errors import ExchangeError
from ..models.trading import (
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
from ..money import ZERO, dec, quantize_price
from ..risk.rules import RiskEngine
from ..timeutil import Clock
from .executor import Executor, build_position

log = logging.getLogger(__name__)


@dataclass
class PositionUpdate:
    opened: Position | None = None
    closed: TradeRecord | None = None
    exit_submitted: OrderRecord | None = None
    notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []


class PositionManager:
    def __init__(self, *, exchange, store, config: Config, clock: Clock,
                 executor: Executor, notifier=None):
        self.exchange = exchange
        self.store = store
        self.config = config
        self.clock = clock
        self.executor = executor
        self.risk = RiskEngine(config)
        self.notifier = notifier

    def step(self, state, *, last_price: Decimal, best_bid: Decimal,
             best_ask: Decimal, plan: ExecutionPlan | None = None) -> PositionUpdate:
        """One monitoring pass. Safe to call as often as the tick fires."""
        update = PositionUpdate()
        position: Position | None = state.open_position

        if position is None:
            position = self._promote_filled_entry(state, plan, update)
            if position is None:
                return update

        if position.exit_order_id:
            self._settle_exit(state, position, update)
            return update

        self._maybe_exit(state, position, update, last_price=last_price,
                         best_bid=best_bid, best_ask=best_ask)
        return update

    # -- entry -> position -------------------------------------------------

    def _promote_filled_entry(self, state, plan: ExecutionPlan | None,
                              update: PositionUpdate) -> Position | None:
        """An entry that has filled (even partially) becomes the position.

        This scans recent entries rather than *open* ones on purpose: an order
        that filled completely is terminal at the exchange, and if only open
        orders were considered, a fast fill would leave a position with no stop
        attached to it.
        """
        for record in self.store.orders.list_recent(50):
            if record.purpose is not OrderPurpose.ENTRY:
                continue
            if record.status.is_open:
                self.executor._refresh(record)
            if record.executed_qty_btc <= 0:
                continue
            booked = self.store.trades.get(record.trade_id or "")
            if booked is not None and booked.closed:
                continue
            plan_for_record = plan or self._plan_from_record(record)
            if plan_for_record is None:
                update.notes.append(
                    f"{record.client_order_id}: filled but no plan on hand; "
                    "cannot set stops")
                continue
            position = build_position(plan_for_record, record, self.clock.now())
            state.open_position = position
            state.last_entry_at = position.opened_at
            update.opened = position
            update.notes.append(
                f"position opened: {position.qty_btc} BTC @ {position.entry_price}")
            self._write_trade(position, record)
            self._notify("約定(新規建て)",
                         f"{position.qty_btc} BTC @ {position.entry_price} JPY\n"
                         f"SL {position.stop_loss} / TP {position.take_profit}\n"
                         f"probe={position.probe}")
            return position
        return None

    def _plan_from_record(self, record: OrderRecord) -> ExecutionPlan | None:
        """Recover the plan from the trade row written when the order went out.

        This is what lets a cold `tick` invocation attach stops to a position
        opened by an earlier `decide` invocation.
        """
        if not record.trade_id:
            return None
        trade = self.store.trades.get(record.trade_id)
        if trade is None:
            return None
        return ExecutionPlan(
            cycle_id=trade.cycle_id, trade_id=trade.trade_id,
            entry=trade.entry_price, stop_loss=trade.stop_loss,
            take_profit=trade.take_profit, qty_btc=trade.qty_btc,
            risk_jpy=(trade.entry_price - trade.stop_loss) * trade.qty_btc,
            probe=trade.probe, regime=trade.regime,
            judge_output_id=trade.judge_output_id)

    # -- exits -------------------------------------------------------------

    def _maybe_exit(self, state, position: Position, update: PositionUpdate, *,
                    last_price: Decimal, best_bid: Decimal,
                    best_ask: Decimal) -> None:
        cfg = self.config
        if last_price <= position.stop_loss:
            intent = OrderIntent(
                cycle_id=position.cycle_id, pair=position.pair, side=Side.SELL,
                order_type=OrderType.MARKET, qty_btc=position.qty_btc,
                price=None, post_only=False, purpose=OrderPurpose.STOP_LOSS,
                probe=position.probe, trade_id=position.trade_id)
            reason = "stop_loss"
        elif last_price >= position.take_profit:
            # Maker exit: rest at or above the best ask so the order does not
            # cross. Spec 8 allows taker execution for stops only.
            price = quantize_price(max(dec(position.take_profit), dec(best_ask)),
                                   cfg.exchange.price_digits)
            intent = OrderIntent(
                cycle_id=position.cycle_id, pair=position.pair, side=Side.SELL,
                order_type=OrderType.LIMIT, qty_btc=position.qty_btc,
                price=price, post_only=True, purpose=OrderPurpose.TAKE_PROFIT,
                probe=position.probe, trade_id=position.trade_id)
            reason = "take_profit"
        else:
            return

        try:
            record = self.executor.submit(intent)
        except ExchangeError as exc:
            log.error("exit order failed for %s: %s", position.trade_id, exc)
            update.notes.append(f"exit order failed: {exc}")
            self._notify("決済注文の発注に失敗",
                         f"trade_id={position.trade_id} reason={reason}\n{exc}")
            return

        position.exit_order_id = record.client_order_id
        position.exit_reason = reason
        state.open_position = position
        update.exit_submitted = record
        update.notes.append(f"exit submitted ({reason}) @ {intent.price or 'market'}")

    def _settle_exit(self, state, position: Position,
                     update: PositionUpdate) -> None:
        record = self.store.orders.get(position.exit_order_id or "")
        if record is None:
            position.exit_order_id = None
            state.open_position = position
            update.notes.append("exit order row vanished; will re-evaluate")
            return

        self.executor._refresh(record)
        if record.status is OrderStatus.FILLED:
            trade = self._close_trade(state, position, record)
            update.closed = trade
            update.notes.append(
                f"position closed: {trade.net_pnl_jpy} JPY ({position.exit_reason})")
            return

        if record.status.is_open and record.purpose is OrderPurpose.TAKE_PROFIT:
            age = self.clock.now() - (record.submitted_at or record.created_at)
            if age > timedelta(minutes=self.config.execution.tp_exit_timeout_minutes):
                # The maker exit did not fill and the move has faded. Stand
                # down and keep the position; the stop still protects it.
                self.executor.cancel(record, "take-profit exit did not fill")
                position.exit_order_id = None
                position.exit_reason = None
                state.open_position = position
                update.notes.append("take-profit exit expired; position retained")
            return

        if record.status.is_terminal and record.status is not OrderStatus.FILLED:
            if record.executed_qty_btc > 0:
                trade = self._close_trade(state, position, record)
                update.closed = trade
                update.notes.append("exit partially filled and canceled; booked")
            else:
                position.exit_order_id = None
                position.exit_reason = None
                state.open_position = position
                update.notes.append("exit order canceled; position retained")

    def force_close(self, state, position: Position, reason: str) -> PositionUpdate:
        """Liquidate now, at market (spec 6: the kill switch closes everything).

        This is the one place a taker sell is issued outside a stop, and it is
        justified by the same reasoning spec 8 uses for stops: when the goal is
        to be flat, certainty of execution beats the fee.
        """
        update = PositionUpdate()
        if position.exit_order_id:
            update.notes.append("an exit order is already in flight")
            return update
        intent = OrderIntent(
            cycle_id=position.cycle_id, pair=position.pair, side=Side.SELL,
            order_type=OrderType.MARKET, qty_btc=position.qty_btc, price=None,
            post_only=False, purpose=OrderPurpose.STOP_LOSS, probe=position.probe,
            trade_id=position.trade_id)
        try:
            record = self.executor.submit(intent)
        except Exception as exc:  # noqa: BLE001 - reported, never raised into the tick
            log.error("forced liquidation failed for %s: %s", position.trade_id, exc)
            update.notes.append(f"forced liquidation failed: {exc}")
            self._notify("強制決済に失敗",
                         f"trade_id={position.trade_id} reason={reason}\n{exc}")
            return update
        position.exit_order_id = record.client_order_id
        position.exit_reason = reason
        state.open_position = position
        update.exit_submitted = record
        update.notes.append(f"forced liquidation submitted ({reason})")
        return update

    # -- bookkeeping -------------------------------------------------------

    def _write_trade(self, position: Position, entry_record: OrderRecord) -> None:
        trade = self.store.trades.get(position.trade_id)
        if trade is None:
            trade = TradeRecord(
                trade_id=position.trade_id, cycle_id=position.cycle_id,
                pair=position.pair, probe=position.probe,
                qty_btc=position.qty_btc, entry_price=position.entry_price,
                entry_order_id=entry_record.client_order_id,
                entry_at=position.opened_at, stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                fee_jpy=entry_record.fee_jpy,
                judge_output_id=position.judge_output_id)
        else:
            trade.qty_btc = position.qty_btc
            trade.entry_price = position.entry_price
            trade.entry_at = position.opened_at
            trade.fee_jpy = entry_record.fee_jpy
        self.store.trades.put(trade)

    def _close_trade(self, state, position: Position,
                     exit_record: OrderRecord) -> TradeRecord:
        now = self.clock.now()
        exit_qty = exit_record.executed_qty_btc or position.qty_btc
        exit_price = exit_record.average_price or exit_record.price or ZERO
        gross = (dec(exit_price) - position.entry_price) * exit_qty
        fees = position.entry_fee_jpy + exit_record.fee_jpy
        net = gross - fees

        trade = self.store.trades.get(position.trade_id) or TradeRecord(
            trade_id=position.trade_id, cycle_id=position.cycle_id,
            pair=position.pair, probe=position.probe, qty_btc=exit_qty,
            entry_price=position.entry_price, entry_order_id="",
            entry_at=position.opened_at, stop_loss=position.stop_loss,
            take_profit=position.take_profit)
        trade.exit_price = dec(exit_price)
        trade.exit_order_id = exit_record.client_order_id
        trade.exit_at = now
        trade.exit_reason = position.exit_reason
        trade.fee_jpy = fees
        trade.gross_pnl_jpy = gross
        trade.net_pnl_jpy = net
        trade.closed = True
        self.store.trades.put(trade)

        state.open_position = None
        self.risk.apply_trade_result(state, trade, now)

        self._notify(
            "約定(決済)",
            f"trade_id={trade.trade_id} reason={trade.exit_reason}\n"
            f"{exit_qty} BTC {position.entry_price} -> {trade.exit_price} JPY\n"
            f"損益 {net} JPY (手数料 {fees} JPY 込)\n"
            f"equity {state.equity_jpy} JPY / 連敗 {state.losing_streak}")
        return trade

    def _notify(self, subject: str, body: str) -> None:
        if self.notifier is not None:
            self.notifier.send(subject, body)
