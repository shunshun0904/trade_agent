"""Exchange interface.

`PaperExchange` and `BitbankClient` both satisfy this, which is what makes the
Phase 1 paper-trading requirement (spec 13, 17.3) a wiring choice rather than
an `if paper:` scattered through the executor.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ..models.market import Candle
from ..models.trading import OrderIntent


class PairSettings(BaseModel):
    """The exchange's own view of the pair's constants.

    Spec 2 asks for the minimum order size and fees to be re-checked against
    the exchange at implementation time and kept in config. bitbank exposes
    them on the unauthenticated `GET /v1/spot/pairs`, so we do better than a
    one-time check: `trade-agent verify-pair` diffs config against live values.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    min_order_btc: Decimal
    price_digits: int
    amount_digits: int
    maker_fee_rate: Decimal
    taker_fee_rate: Decimal
    limit_max_amount: Decimal | None = None
    is_enabled: bool = True
    stop_order: bool = False
    stop_order_and_cancel: bool = False


class Balance(BaseModel):
    asset: str
    free: Decimal
    locked: Decimal
    onhand: Decimal


@runtime_checkable
class ExchangeClient(Protocol):
    """Only the calls this system actually makes.

    Deliberately absent: anything under /user/withdrawal. The API key carries
    no withdrawal permission (spec 12) and the code has no path to ask for one.
    """

    def get_ticker(self) -> dict: ...

    def get_depth(self) -> dict: ...

    def get_candles(self, candle_type: str, day: date | int) -> list[Candle]: ...

    def get_pair_settings(self) -> PairSettings: ...

    def get_balances(self) -> dict[str, Balance]: ...

    def create_order(self, intent: OrderIntent) -> dict: ...

    def cancel_order(self, exchange_order_id: str) -> dict: ...

    def get_order(self, exchange_order_id: str) -> dict: ...

    def get_active_orders(self) -> list[dict]: ...

    def get_trades_for_order(self, exchange_order_id: str) -> list[dict]: ...
