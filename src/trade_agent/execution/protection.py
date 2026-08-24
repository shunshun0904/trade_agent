"""Hand-rolled OCO (spec 8).

bitbank has no OCO endpoint, so the two protective legs are placed separately
and the survivor is cancelled when one fills:

    arm    place the stop leg, then the take-profit leg
    poll   on every tick, look for a fill; cancel the sibling
    disarm cancel both (kill switch, or re-sizing after a partial fill)

Two facts shape the whole design.

**A spot balance can only back one sell.** A resting limit sell at the target
reserves the BTC; a stop sell wants the same BTC. Whether bitbank refuses the
second order outright, or accepts it and rejects the stop later when it
triggers, could not be verified from the build environment
(docs/OPEN-QUESTIONS.md A-1). So this module never assumes either: it places
the *stop first* — if only one leg can exist it must be the protective one —
and keeps local evaluation armed as a backstop for whichever leg the exchange
turns out not to be holding.

**Cancelling races against filling.** Between reading a leg and cancelling its
sibling, the sibling may fill. bitbank answers 50010 (`OrderNotCancelable`),
which this module reads not as an error but as "the sibling won". Both legs are
then re-read and the exit is booked from what actually executed, never from
what was expected to.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal

from ..config import Config
from ..errors import (
    ExchangeError,
    InsufficientFunds,
    OrderNotCancelable,
    StopOrderRefused,
)
from ..models.trading import (
    OrderIntent,
    OrderPurpose,
    OrderRecord,
    OrderStatus,
    OrderType,
    Position,
    Side,
)
from ..money import ZERO, dec, quantize_price
from ..timeutil import Clock

log = logging.getLogger(__name__)

LOCAL = "local"
EXCHANGE_OCO = "exchange_oco"
STOP_ONLY = "exchange_stop_only"


@dataclass
class ProtectionResult:
    """What protection is now in force, and what happened getting there."""

    protection: str = LOCAL
    stop_order: OrderRecord | None = None
    take_profit_order: OrderRecord | None = None
    filled: list[OrderRecord] = field(default_factory=list)
    exit_reason: str | None = None
    close_immediately: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def closed(self) -> bool:
        return bool(self.filled)

    def executed_qty(self) -> Decimal:
        return sum((o.executed_qty_btc for o in self.filled), ZERO)


class ProtectionManager:
    def __init__(self, *, exchange, store, config: Config, clock: Clock,
                 executor, notifier=None):
        self.exchange = exchange
        self.store = store
        self.config = config
        self.clock = clock
        self.executor = executor
        self.notifier = notifier

    @property
    def mode(self) -> str:
        return self.config.execution.oco_mode

    # -- arming ------------------------------------------------------------

    def arm(self, position: Position, *, last_price: Decimal) -> ProtectionResult:
        """Place the protective legs for a position that has just opened."""
        result = ProtectionResult()
        if self.mode == LOCAL:
            result.notes.append("protection: local evaluation (configured)")
            return result

        position.protection_generation += 1

        if dec(last_price) <= position.stop_loss:
            # bitbank refuses a trigger that would fire immediately (60018), and
            # placing one would be wrong anyway: the stop is already breached.
            result.close_immediately = True
            result.exit_reason = "stop_loss"
            result.notes.append(
                f"market {last_price} is already through the stop "
                f"{position.stop_loss}; closing rather than arming")
            return result

        stop = self._place_stop(position, result)
        if stop is None:
            result.protection = LOCAL
            return result
        result.stop_order = stop

        take_profit = self._place_take_profit(position, result)
        if take_profit is None:
            result.protection = STOP_ONLY
            return result

        result.take_profit_order = take_profit
        result.protection = EXCHANGE_OCO
        if not self._both_legs_backed(position, result):
            # Both orders were accepted, but the balance only backs one. Left
            # alone, the resting target holds the BTC and the stop is rejected
            # at the moment it triggers — the position would look protected and
            # not be. Give the balance back to the stop.
            self._retract_take_profit(take_profit, result)
            result.take_profit_order = None
            result.protection = STOP_ONLY
        return result

    def _place_stop(self, position: Position,
                    result: ProtectionResult) -> OrderRecord | None:
        """The protective leg. It goes first, always."""
        cfg = self.config.execution
        order_type = (OrderType.STOP_LIMIT if cfg.stop_order_type == "stop_limit"
                      else OrderType.STOP)
        price = None
        if order_type is OrderType.STOP_LIMIT:
            price = quantize_price(
                position.stop_loss * (Decimal(1) - cfg.stop_limit_offset_pct
                                      / Decimal(100)),
                self.config.exchange.price_digits)

        intent = OrderIntent(
            client_order_id=leg_order_id(position.trade_id, "stop",
                                         position.protection_generation),
            cycle_id=position.cycle_id, pair=position.pair, side=Side.SELL,
            order_type=order_type, qty_btc=position.qty_btc, price=price,
            post_only=False, purpose=OrderPurpose.STOP_LOSS,
            probe=position.probe,
            trigger_price=quantize_price(position.stop_loss,
                                         self.config.exchange.price_digits),
            trade_id=position.trade_id)
        try:
            return self.executor.submit(intent)
        except (StopOrderRefused, InsufficientFunds, ExchangeError) as exc:
            # Falling back to local evaluation is a real reduction in safety:
            # the stop now depends on the tick continuing to run.
            log.error("could not place the exchange stop for %s: %s",
                      position.trade_id, exc)
            result.notes.append(f"stop leg refused ({exc}); falling back to local")
            self._notify(
                "取引所側の損切り注文を出せませんでした",
                f"trade_id={position.trade_id}\n{exc}\n\n"
                "ローカル監視(5分tick)に切り替えました。tickが停止すると"
                "損切りも停止するため、tickのハートビートアラームを確認してください。")
            return None

    def _place_take_profit(self, position: Position,
                           result: ProtectionResult) -> OrderRecord | None:
        intent = OrderIntent(
            client_order_id=leg_order_id(position.trade_id, "tp",
                                         position.protection_generation),
            cycle_id=position.cycle_id, pair=position.pair, side=Side.SELL,
            order_type=OrderType.LIMIT, qty_btc=position.qty_btc,
            price=quantize_price(position.take_profit,
                                 self.config.exchange.price_digits),
            post_only=True, purpose=OrderPurpose.TAKE_PROFIT,
            probe=position.probe, trade_id=position.trade_id)
        try:
            return self.executor.submit(intent)
        except InsufficientFunds:
            # Expected on a spot account whose stop already reserves the BTC.
            # Not a failure: the position is protected, and the target is the
            # leg that can safely be evaluated locally.
            result.notes.append(
                "take-profit leg refused for want of balance (the stop holds "
                "it); target evaluated locally")
            return None
        except ExchangeError as exc:
            log.warning("could not place the take-profit leg for %s: %s",
                        position.trade_id, exc)
            result.notes.append(f"take-profit leg refused ({exc}); "
                                "target evaluated locally")
            return None

    def _both_legs_backed(self, position: Position,
                          result: ProtectionResult) -> bool:
        """Does the account actually reserve balance for both sells?

        Acceptance is not the same as backing. If the exchange defers a stop's
        reservation to trigger time, both orders sit there looking healthy
        while only one of them can ever execute — and on a falling market the
        one that cannot is the stop.

        Costs one balance call, and it is the difference between a protected
        position and a position that only appears protected.
        """
        needed = position.qty_btc * Decimal(2)
        try:
            balances = self.exchange.get_balances()
        except ExchangeError as exc:
            # Unverifiable. Assume the worse case rather than the convenient
            # one: the stop keeps the balance and the target goes local.
            result.notes.append(
                f"could not verify the balance reservation ({exc}); keeping the "
                "stop as the only exchange-held leg")
            return False
        base = balances.get(self.config.exchange.base_asset)
        locked = base.locked if base else ZERO
        if locked >= needed:
            return True
        result.notes.append(
            f"only {locked} BTC is reserved against {needed} BTC of sell "
            "orders, so the exchange defers the stop's reservation: the "
            "resting target would starve the stop at trigger time")
        log.warning("OCO legs are not both backed (locked=%s, needed=%s); "
                    "retracting the take-profit leg", locked, needed)
        return False

    def _retract_take_profit(self, take_profit: OrderRecord,
                             result: ProtectionResult) -> None:
        """Cancel the target so the stop owns the balance outright."""
        try:
            self.executor.cancel(take_profit, "returning the balance to the stop")
            result.notes.append(
                "take-profit leg retracted; the stop holds the balance and the "
                "target is evaluated locally")
        except OrderNotCancelable:
            result.notes.append("take-profit leg filled before it could be retracted")
        except ExchangeError as exc:
            log.error("could not retract the take-profit leg: %s", exc)
            result.notes.append(f"FAILED to retract the take-profit leg: {exc}")
            self._notify(
                "利確注文を取り消せず、損切りが機能しない可能性があります",
                f"client_order_id={take_profit.client_order_id}\n{exc}\n\n"
                "取引所が損切りの残高確保をトリガー時まで遅延させる挙動のため、"
                "利確の指値売りが残っていると損切りが発動時に拒否されます。\n"
                "bitbank の注文一覧から利確注文を手動で取り消してください。")

    # -- polling -----------------------------------------------------------

    def poll(self, position: Position, *,
             last_price: Decimal) -> ProtectionResult:
        """Look for a filled leg and cancel its sibling."""
        result = ProtectionResult(protection=position.protection)
        stop = self._leg(position.stop_order_id)
        take_profit = self._leg(position.take_profit_order_id)
        result.stop_order, result.take_profit_order = stop, take_profit

        for leg in (stop, take_profit):
            if leg is not None and leg.status.is_open:
                self.executor._refresh(leg)  # noqa: SLF001 - same package

        filled = [leg for leg in (stop, take_profit)
                  if leg is not None and leg.executed_qty_btc > 0]
        if not filled:
            self._check_still_protected(position, stop, take_profit, result)
            return result

        survivor = next((leg for leg in (stop, take_profit)
                         if leg is not None and leg not in filled), None)
        if survivor is not None:
            self._cancel_sibling(survivor, result)
            self.executor._refresh(survivor)  # noqa: SLF001
            if survivor.executed_qty_btc > 0 and survivor not in filled:
                # It filled while we were cancelling. Both legs executed.
                filled.append(survivor)
                result.notes.append(
                    f"{survivor.purpose} also filled during cancellation")

        result.filled = filled
        result.exit_reason = _reason_for(filled, stop)
        self._check_oversell(position, filled, result)
        return result

    def _cancel_sibling(self, survivor: OrderRecord,
                        result: ProtectionResult) -> None:
        try:
            self.executor.cancel(survivor, "OCO sibling filled")
            result.notes.append(f"cancelled the {survivor.purpose} leg")
        except OrderNotCancelable:
            # bitbank 50010 — it filled between our read and our cancel.
            result.notes.append(
                f"the {survivor.purpose} leg could not be cancelled; it filled "
                "in the same interval")
        except ExchangeError as exc:
            # The dangerous case: one leg filled, the other is still live and
            # could sell BTC we no longer hold. Say so loudly.
            log.error("could not cancel the surviving OCO leg %s: %s",
                      survivor.client_order_id, exc)
            result.notes.append(f"FAILED to cancel the {survivor.purpose} leg: {exc}")
            self._notify(
                "OCO の残注文を取り消せませんでした",
                f"client_order_id={survivor.client_order_id}\n{exc}\n\n"
                "取引所側に決済注文が残っている可能性があります。"
                "bitbank の注文一覧を確認し、残っていれば手動で取り消してください。")

    def _check_oversell(self, position: Position, filled: list[OrderRecord],
                        result: ProtectionResult) -> None:
        total = sum((leg.executed_qty_btc for leg in filled), ZERO)
        if total <= position.qty_btc:
            return
        # A spot balance should make this impossible. If it happens anyway the
        # account is short BTC it never held, and a human has to look.
        log.error("OCO sold %s BTC against a %s BTC position", total,
                  position.qty_btc)
        result.notes.append(
            f"OVERSOLD: {total} BTC executed against a {position.qty_btc} BTC "
            "position")
        self._notify(
            "OCO で建玉より多く売却しました",
            f"trade_id={position.trade_id}\n"
            f"建玉 {position.qty_btc} BTC に対し {total} BTC が約定しました。\n"
            "bitbank の残高と約定履歴を確認してください。")

    def _check_still_protected(self, position: Position,
                               stop: OrderRecord | None,
                               take_profit: OrderRecord | None,
                               result: ProtectionResult) -> None:
        """A stop that died without filling leaves the position naked."""
        if position.protection == LOCAL or stop is None:
            return
        if stop.status.is_protective:
            return
        result.protection = LOCAL
        result.notes.append(
            f"the exchange stop is {stop.status}; reverting to local evaluation")
        log.error("exchange stop for %s is no longer protective (%s)",
                  position.trade_id, stop.status)
        self._notify(
            "取引所側の損切り注文が消えました",
            f"trade_id={position.trade_id} の損切り注文が {stop.status} です。\n"
            "ローカル監視に切り替えました。建玉は保有したままです。")

    # -- disarming ---------------------------------------------------------

    def force_release(self, position: Position, reason: str) -> list[str]:
        """Cancel the legs at the exchange whatever our records say.

        `disarm` trusts the local status and skips legs it believes are already
        finished. That is right in the normal case and wrong in the one that
        matters: if our view of a leg is stale, the exchange is still holding
        the balance and every local exit will fail for want of funds. This asks
        the exchange directly.
        """
        notes: list[str] = []
        for order_id in (position.stop_order_id, position.take_profit_order_id):
            leg = self._leg(order_id)
            if leg is None or not leg.exchange_order_id:
                continue
            try:
                self.exchange.cancel_order(leg.exchange_order_id)
                notes.append(f"{leg.client_order_id}: force-released ({reason})")
            except OrderNotCancelable:
                notes.append(f"{leg.client_order_id}: already filled")
            except ExchangeError as exc:
                notes.append(f"{leg.client_order_id}: force-release failed ({exc})")
            self.executor._refresh(leg)  # noqa: SLF001 - same package
        return notes

    def disarm(self, position: Position, reason: str) -> list[str]:
        """Cancel both legs. Used before a forced liquidation (spec 6)."""
        notes: list[str] = []
        for order_id in (position.stop_order_id, position.take_profit_order_id):
            leg = self._leg(order_id)
            if leg is None or not leg.status.is_open:
                continue
            try:
                notes.append(self.executor.cancel(leg, reason))
            except OrderNotCancelable:
                notes.append(f"{leg.client_order_id}: already filled")
            except ExchangeError as exc:
                notes.append(f"{leg.client_order_id}: cancel failed ({exc})")
        return notes

    # -- helpers -----------------------------------------------------------

    def _leg(self, client_order_id: str | None) -> OrderRecord | None:
        return self.store.orders.get(client_order_id) if client_order_id else None

    def _notify(self, subject: str, body: str) -> None:
        if self.notifier is not None:
            self.notifier.send(subject, body)


def leg_order_id(trade_id: str, leg: str, generation: int = 1) -> str:
    """Deterministic id per leg, so re-arming cannot duplicate an order.

    The conditional create in the order table then makes `arm` idempotent for
    free: a second attempt within the same generation raises DuplicateOrder
    instead of placing a second stop. `generation` is what lets a *deliberate*
    re-arm — after an exit order was cancelled — get a fresh id rather than
    colliding with the leg it replaces.
    """
    import hashlib

    return hashlib.sha256(
        f"{trade_id}:{leg}:{generation}".encode()).hexdigest()[:32]


def _reason_for(filled: list[OrderRecord], stop: OrderRecord | None) -> str:
    if len(filled) > 1:
        return "stop_loss_and_take_profit"
    leg = filled[0]
    if stop is not None and leg.client_order_id == stop.client_order_id:
        return "stop_loss"
    return "take_profit"


def weighted_exit(filled: list[OrderRecord]) -> tuple[Decimal, Decimal, Decimal]:
    """(quantity, volume-weighted price, total fee) across every filled leg.

    A leg with no usable price is a bug somewhere upstream, and valuing it at
    zero would book the entire position as a total loss. Fall back through the
    prices the order does carry, and refuse to return a zero price for a
    non-zero quantity.
    """
    qty = sum((leg.executed_qty_btc for leg in filled), ZERO)
    if qty <= 0:
        return ZERO, ZERO, ZERO
    notional = ZERO
    for leg in filled:
        price = leg.average_price or leg.price or leg.trigger_price
        if price is None or price <= 0:
            raise ValueError(
                f"order {leg.client_order_id} executed {leg.executed_qty_btc} "
                "with no recorded price; refusing to book it at zero")
        notional += dec(price) * leg.executed_qty_btc
    fees = sum((leg.fee_jpy for leg in filled), ZERO)
    return qty, notional / qty, fees
