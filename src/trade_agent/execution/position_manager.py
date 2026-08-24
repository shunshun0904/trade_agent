"""Position lifecycle and the self-managed OCO (spec 8).

Exits come from two places, and which one is in force depends on what the
exchange accepted (see `protection.py`):

* **exchange legs** — a `stop` sell and a limit sell placed as a hand-rolled
  OCO. When these hold, a stop survives even if this process stops running.
* **local evaluation** — the 5-minute tick compares price against the levels
  and sends the closing order itself. This is the fallback, and it is also the
  backstop: it stays armed for whichever level the exchange is *not* holding,
  so a rejected leg cannot leave a position naked.

Local evaluation never acts on a level an exchange leg is still protecting;
otherwise the same position would be sold twice.

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
from ..errors import ExchangeError, InsufficientFunds
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
from .protection import LOCAL, ProtectionManager, ProtectionResult, weighted_exit

log = logging.getLogger(__name__)


@dataclass
class PositionUpdate:
    opened: Position | None = None
    closed: TradeRecord | None = None
    exit_submitted: OrderRecord | None = None
    protection: str | None = None
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
        self.protection = ProtectionManager(
            exchange=exchange, store=store, config=config, clock=clock,
            executor=executor, notifier=notifier)

    def step(self, state, *, last_price: Decimal, best_bid: Decimal,
             best_ask: Decimal, plan: ExecutionPlan | None = None) -> PositionUpdate:
        """One monitoring pass. Safe to call as often as the tick fires."""
        update = PositionUpdate()
        position: Position | None = state.open_position

        if position is None:
            position = self._promote_filled_entry(
                state, plan, update, last_price=last_price)
            if position is None:
                return update

        if position.protection != LOCAL:
            if self._poll_exchange_legs(state, position, update,
                                        last_price=last_price):
                return update
            position = state.open_position
            if position is None:
                return update

        if position.exit_order_id:
            self._settle_exit(state, position, update)
            return update

        self._maybe_exit(state, position, update, last_price=last_price,
                         best_bid=best_bid, best_ask=best_ask)
        return update

    # -- exchange-held legs -------------------------------------------------

    def _poll_exchange_legs(self, state, position: Position,
                            update: PositionUpdate, *,
                            last_price: Decimal) -> bool:
        """Returns True when the position closed on an exchange leg."""
        result = self.protection.poll(position, last_price=last_price)
        update.notes.extend(result.notes)
        update.protection = result.protection

        if result.protection != position.protection:
            position.protection = result.protection
            state.open_position = position

        if not result.closed:
            return False

        trade = self._close_trade(state, position, result.filled,
                                  reason=result.exit_reason)
        update.closed = trade
        update.notes.append(
            f"position closed on the exchange leg: {trade.net_pnl_jpy} JPY "
            f"({trade.exit_reason})")
        return True

    # -- entry -> position -------------------------------------------------

    def _promote_filled_entry(self, state, plan: ExecutionPlan | None,
                              update: PositionUpdate, *,
                              last_price: Decimal) -> Position | None:
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
            self._arm_protection(state, position, update, last_price=last_price)
            self._notify("約定(新規建て)",
                         f"{position.qty_btc} BTC @ {position.entry_price} JPY\n"
                         f"SL {position.stop_loss} / TP {position.take_profit}\n"
                         f"probe={position.probe}")
            return position
        return None

    def _arm_protection(self, state, position: Position, update: PositionUpdate,
                        *, last_price: Decimal) -> None:
        """Place the protective legs the moment the position exists.

        Ordering matters: the trade row and the position are already persisted,
        so a crash between here and the next tick leaves a recoverable state
        rather than an untracked position.
        """
        try:
            result = self.protection.arm(position, last_price=last_price)
        except Exception as exc:  # noqa: BLE001 - never lose the position over this
            log.exception("arming protection failed for %s", position.trade_id)
            update.notes.append(f"arming protection failed ({exc}); local only")
            position.protection = LOCAL
            state.open_position = position
            return

        update.notes.extend(result.notes)
        update.protection = result.protection
        position.protection = result.protection
        position.stop_order_id = (result.stop_order.client_order_id
                                  if result.stop_order else None)
        position.take_profit_order_id = (
            result.take_profit_order.client_order_id
            if result.take_profit_order else None)
        state.open_position = position

        if result.close_immediately:
            update.notes.extend(
                self.force_close(state, position, result.exit_reason or "stop_loss").notes)

    def _rearm_if_unprotected(self, state, position: Position,
                              update: PositionUpdate) -> None:
        """Put the exchange legs back after a retained position lost them."""
        if self.protection.mode == LOCAL or position.stop_order_id:
            return
        try:
            ticker = self.exchange.get_ticker()
        except ExchangeError as exc:
            update.notes.append(f"could not re-arm protection: {exc}")
            return
        self._arm_protection(state, position, update,
                             last_price=dec(ticker["last"]))

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
        if self._exchange_holds(position, OrderPurpose.STOP_LOSS) and \
                self._exchange_holds(position, OrderPurpose.TAKE_PROFIT):
            return
        if last_price <= position.stop_loss and \
                not self._exchange_holds(position, OrderPurpose.STOP_LOSS):
            intent = OrderIntent(
                cycle_id=position.cycle_id, pair=position.pair, side=Side.SELL,
                order_type=OrderType.MARKET, qty_btc=position.qty_btc,
                price=None, post_only=False, purpose=OrderPurpose.STOP_LOSS,
                probe=position.probe, trade_id=position.trade_id)
            reason = "stop_loss"
        elif last_price >= position.take_profit and \
                not self._exchange_holds(position, OrderPurpose.TAKE_PROFIT):
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

        # An exchange leg reserves the very BTC this order wants to sell, so
        # the legs come off before the local exit goes on. Between the two the
        # position is briefly unprotected, which is why this only ever runs on
        # a level the exchange is *not* already holding — a stop that is live
        # keeps its balance and does the work itself.
        # The leg ids are kept until the exit order is actually accepted: if
        # the first attempt is refused for want of balance, they are the only
        # handle on whatever is still holding it.
        if position.stop_order_id or position.take_profit_order_id:
            update.notes.extend(self.protection.disarm(
                position, f"freeing the balance for a local {reason} exit"))

        try:
            record = self.executor.submit(intent)
        except InsufficientFunds as exc:
            # Something is still holding the balance. Most likely a leg our
            # records believe is finished but the exchange does not. Ask the
            # exchange to let go, then try once more.
            update.notes.append(f"exit refused for want of balance ({exc}); "
                                "force-releasing the protective legs")
            update.notes.extend(self.protection.force_release(
                position, f"local {reason} exit"))
            retry = intent.model_copy(deep=True)
            retry.client_order_id = f"{intent.client_order_id}-retry"
            try:
                record = self.executor.submit(retry)
            except ExchangeError as retry_exc:
                log.error("exit order failed for %s after force-release: %s",
                          position.trade_id, retry_exc)
                update.notes.append(f"exit order failed: {retry_exc}")
                self._notify(
                    "決済注文の発注に失敗",
                    f"trade_id={position.trade_id} reason={reason}\n{retry_exc}\n\n"
                    "保護注文を取り消したうえで再試行しましたが失敗しました。"
                    "建玉が無保護の可能性があります。取引所の画面を確認してください。")
                self._forget_legs(state, position)
                return
        except ExchangeError as exc:
            log.error("exit order failed for %s: %s", position.trade_id, exc)
            update.notes.append(f"exit order failed: {exc}")
            self._notify("決済注文の発注に失敗",
                         f"trade_id={position.trade_id} reason={reason}\n{exc}")
            # The legs were just taken off to make room for an order that never
            # went out. Put protection back rather than leaving it naked.
            self._arm_protection(state, position, update, last_price=last_price)
            return

        self._forget_legs(state, position)
        position.exit_order_id = record.client_order_id
        position.exit_reason = reason
        state.open_position = position
        update.exit_submitted = record
        update.notes.append(f"exit submitted ({reason}) @ {intent.price or 'market'}")

    def _forget_legs(self, state, position: Position) -> None:
        """Drop the protective legs from the position and fall back to local."""
        position.stop_order_id = None
        position.take_profit_order_id = None
        position.protection = LOCAL
        state.open_position = position

    def _exchange_holds(self, position: Position, purpose: OrderPurpose) -> bool:
        """Is a live exchange leg already covering this level?

        The backstop exists precisely for when the answer is no, so this has to
        check the leg's actual status rather than trusting the mode label.
        """
        order_id = (position.stop_order_id if purpose is OrderPurpose.STOP_LOSS
                    else position.take_profit_order_id)
        if not order_id:
            return False
        record = self.store.orders.get(order_id)
        return record is not None and record.status.is_protective

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
                # The balance is free again, so the stop can go back on.
                self._rearm_if_unprotected(state, position, update)
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
                self._rearm_if_unprotected(state, position, update)

    def force_close(self, state, position: Position, reason: str) -> PositionUpdate:
        """Liquidate now, at market (spec 6: the kill switch closes everything).

        This is the one place a taker sell is issued outside a stop, and it is
        justified by the same reasoning spec 8 uses for stops: when the goal is
        to be flat, certainty of execution beats the fee.
        """
        update = PositionUpdate()
        # Cancel the protective legs first, or the market sell and a surviving
        # stop would both be trying to sell the same BTC.
        update.notes.extend(self.protection.disarm(position, reason))
        position.stop_order_id = None
        position.take_profit_order_id = None
        position.protection = LOCAL
        state.open_position = position

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
        self._forget_legs(state, position)
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
                     exit_records: OrderRecord | list[OrderRecord], *,
                     reason: str | None = None) -> TradeRecord:
        """Book the exit from what actually executed.

        Takes a list because an OCO close can span both legs: if the market
        gaps through the stop and the target in the same interval, the exit is
        the volume-weighted combination of the two, not whichever one is
        noticed first.
        """
        if isinstance(exit_records, OrderRecord):
            exit_records = [exit_records]
        now = self.clock.now()
        exit_qty, exit_price, exit_fees = weighted_exit(exit_records)
        if exit_qty <= 0:
            exit_qty = position.qty_btc
            exit_price = dec(exit_records[0].price or ZERO)
            exit_fees = sum((r.fee_jpy for r in exit_records), ZERO)
        gross = (dec(exit_price) - position.entry_price) * exit_qty
        fees = position.entry_fee_jpy + exit_fees
        net = gross - fees

        trade = self.store.trades.get(position.trade_id) or TradeRecord(
            trade_id=position.trade_id, cycle_id=position.cycle_id,
            pair=position.pair, probe=position.probe, qty_btc=exit_qty,
            entry_price=position.entry_price, entry_order_id="",
            entry_at=position.opened_at, stop_loss=position.stop_loss,
            take_profit=position.take_profit)
        trade.exit_price = dec(exit_price)
        trade.exit_order_id = ", ".join(r.client_order_id for r in exit_records)
        trade.exit_at = now
        trade.exit_reason = reason or position.exit_reason
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
