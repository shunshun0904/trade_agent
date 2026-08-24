"""Shared fixtures.

The whole system runs offline here: an in-memory store, a scripted market and
the offline LLM. That is a design property, not a testing convenience — spec 14
asks for behaviours like "72 hours plus one minute triggers the 3-day rule",
which only become testable if time and the market are both injectable.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from trade_agent.config import load_config
from trade_agent.exchange.base import Balance, PairSettings
from trade_agent.exchange.paper import PaperExchange
from trade_agent.llm.budget import CostMeter
from trade_agent.llm.registry import ModelRouter
from trade_agent.llm.stub import StubLLMClient
from trade_agent.models.market import Candle
from trade_agent.money import dec
from trade_agent.notify.notifier import NullNotifier
from trade_agent.orchestrator.context import AppContext
from trade_agent.risk.rules import RiskEngine
from trade_agent.storage.memory import MemoryStore
from trade_agent.timeutil import UTC, FrozenClock

START = datetime(2026, 3, 2, 0, 0, tzinfo=UTC)  # 09:00 JST, a Monday
BASE_PRICE = Decimal("15000000")


class FakeMarket:
    """A scripted BTC/JPY market.

    `price` is the current last price; candles are generated around it so the
    indicators have real history to chew on. Tests move the market by assigning
    to `price`.
    """

    def __init__(self, clock, price: Decimal = BASE_PRICE):
        self.clock = clock
        self.price = dec(price)
        self.spread = Decimal("2000")
        self.calls: list[str] = []
        self.fail_next: Exception | None = None
        # Candle shape knobs. `hold_still()` flattens them so a resting limit
        # order provably cannot fill.
        self.candle_drift = Decimal("5000")
        self.candle_range = Decimal("4000")

    def hold_still(self) -> None:
        """Perfectly flat: a resting limit order provably cannot fill."""
        self.candle_drift = Decimal("0")
        self.candle_range = Decimal("0")

    def quiet(self) -> None:
        """Range-bound and directionless, but not degenerate.

        Closes do not move, so RSI sits at 50 and the VWAP deviation is zero,
        while the intrabar range keeps the 24h high above and the 24h low below
        the last price — so no breakout condition fires either.
        """
        self.candle_drift = Decimal("0")
        self.candle_range = Decimal("20000")

    # -- public surface ---------------------------------------------------

    def get_ticker(self) -> dict:
        self._maybe_fail("ticker")
        return {"last": str(self.price), "buy": str(self.best_bid),
                "sell": str(self.best_ask), "high": str(self.price * Decimal("1.02")),
                "low": str(self.price * Decimal("0.98")),
                "open": str(self.price), "vol": "120",
                "timestamp": int(self.clock.now().timestamp() * 1000)}

    @property
    def best_bid(self) -> Decimal:
        return self.price - self.spread / 2

    @property
    def best_ask(self) -> Decimal:
        return self.price + self.spread / 2

    def get_depth(self) -> dict:
        self._maybe_fail("depth")
        asks = [[str(self.best_ask + Decimal(i) * 1000), "0.05"] for i in range(20)]
        bids = [[str(self.best_bid - Decimal(i) * 1000), "0.05"] for i in range(20)]
        return {"asks": asks, "bids": bids, "asks_over": "0", "bids_under": "0",
                "timestamp": int(self.clock.now().timestamp() * 1000)}

    def get_candles(self, candle_type: str, day) -> list[Candle]:
        self._maybe_fail("candles")
        self.calls.append(f"candles:{candle_type}")
        step = {"1min": 1, "5min": 5, "15min": 15, "30min": 30,
                "1hour": 60}.get(candle_type, 60)
        count = {"1min": 240, "5min": 288}.get(candle_type, 24)
        if isinstance(day, int):
            day = self.clock.now().date()
        start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        out: list[Candle] = []
        for index in range(count):
            opened = start + timedelta(minutes=step * index)
            if opened > self.clock.now():
                break
            drift = Decimal(index % 7 - 3) * self.candle_drift
            close = self.price + drift
            out.append(Candle(
                open=close, high=close + self.candle_range,
                low=close - self.candle_range, close=close,
                volume=Decimal("2") + Decimal(index % 3),
                opened_at=opened))
        return out

    def get_pair_settings(self) -> PairSettings:
        return PairSettings(name="btc_jpy", min_order_btc=Decimal("0.0001"),
                            price_digits=0, amount_digits=8,
                            maker_fee_rate=Decimal("-0.0002"),
                            taker_fee_rate=Decimal("0.0012"))

    def get_transactions(self, day: date | None = None) -> list[dict]:
        return []

    def get_circuit_break_info(self) -> dict:
        return {"mode": "NONE", "fee_type": "NORMAL"}

    def _maybe_fail(self, what: str) -> None:
        if self.fail_next is not None:
            error, self.fail_next = self.fail_next, None
            raise error


class RecordingExchange:
    """Paper exchange over the scripted market, with the calls recorded."""

    def __init__(self, market: FakeMarket, exchange: PaperExchange):
        self.market = market
        self.inner = exchange
        self.orders_sent: list = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def create_order(self, intent):
        self.orders_sent.append(intent)
        return self.inner.create_order(intent)

    def get_balances(self) -> dict[str, Balance]:
        return self.inner.get_balances()


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(current=START)


@pytest.fixture
def config():
    cfg = load_config(use_env=False).model_copy(deep=True)
    cfg.storage.backend = "memory"
    cfg.llm.provider = "stub"
    cfg.system.paper_trading = True
    cfg.notify.enabled = False
    return cfg


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore()


@pytest.fixture
def market(clock) -> FakeMarket:
    return FakeMarket(clock)


@pytest.fixture
def exchange(market, config, clock) -> RecordingExchange:
    paper = PaperExchange(
        market, pair=config.exchange.pair,
        maker_fee_rate=config.exchange.maker_fee_rate,
        taker_fee_rate=config.exchange.taker_fee_rate,
        min_order_btc=config.exchange.min_order_btc,
        initial_jpy=config.capital.initial_equity_jpy,
        now=clock.now)
    return RecordingExchange(market, paper)


@pytest.fixture
def llm(config) -> StubLLMClient:
    return StubLLMClient(cost_meter=CostMeter(config.llm, config.cost))


@pytest.fixture
def notifier() -> NullNotifier:
    return NullNotifier()


@pytest.fixture
def ctx(config, clock, store, exchange, llm, notifier) -> AppContext:
    context = AppContext(
        config=config, clock=clock, store=store, exchange=exchange,
        secrets=_NoSecrets(), notifier=notifier,
        cost_meter=CostMeter(config.llm, config.cost),
        risk=RiskEngine(config), router=ModelRouter(config), owner="test")
    context._llm = llm
    return context


@pytest.fixture
def snapshot(ctx):
    return ctx.snapshot_builder().build()


class _NoSecrets:
    def get(self, name: str) -> str:
        return "test-token"

    def get_optional(self, name: str):
        return None
