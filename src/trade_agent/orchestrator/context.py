"""Dependency wiring.

One place decides which exchange, which store and which LLM client a Lambda
gets. That matters for two of the spec's safety properties:

* Phase 1 (spec 13/17.3) is enforced structurally — with `paper_trading` on,
  the object handed to the executor physically cannot place a live order.
* The `mcp` function is built with `needs_trading_credentials=False`, so it
  never even asks SSM for the bitbank key (spec 12/16.3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

from ..config import Config, get_config
from ..data.snapshot import SnapshotBuilder
from ..errors import ConfigError
from ..exchange.bitbank import BitbankClient
from ..exchange.ratelimit import RateLimiter
from ..execution.executor import Executor
from ..execution.position_manager import PositionManager
from ..guards.deterministic import DeterministicGuard  # noqa: F401  (re-export)
from ..llm.budget import CostMeter
from ..llm.registry import ModelRouter, build_llm_client
from ..models.state import SystemState
from ..notify.notifier import build_notifier
from ..risk.rules import RiskEngine
from ..storage.base import Store
from ..storage.memory import MemoryStore
from ..storage.secrets import SecretProvider, default_provider
from ..timeutil import Clock, jst_date_str, jst_month_str

log = logging.getLogger(__name__)


@dataclass
class AppContext:
    config: Config
    clock: Clock
    store: Store
    exchange: object
    secrets: SecretProvider
    notifier: object
    cost_meter: CostMeter
    risk: RiskEngine
    router: ModelRouter
    owner: str = "trade-agent"
    _llm: object | None = None

    # -- lazily built collaborators ---------------------------------------

    @property
    def llm(self):
        """Built on demand so a tick that never talks to a model never
        constructs a client or reads the API key."""
        if self._llm is None:
            self._llm = build_llm_client(self.config, secrets=self.secrets,
                                         cost_meter=self.cost_meter)
        return self._llm

    def snapshot_builder(self, position=None) -> SnapshotBuilder:
        return SnapshotBuilder(self.exchange, self.config, self.clock)

    def executor(self, owner: str | None = None) -> Executor:
        return Executor(exchange=self.exchange, store=self.store,
                        config=self.config, clock=self.clock,
                        notifier=self.notifier, owner=owner or self.owner)

    def position_manager(self, executor: Executor | None = None) -> PositionManager:
        return PositionManager(exchange=self.exchange, store=self.store,
                               config=self.config, clock=self.clock,
                               executor=executor or self.executor(),
                               notifier=self.notifier)

    # -- state -------------------------------------------------------------

    def load_state(self) -> SystemState:
        """Load system state, creating it on first run and rolling the daily
        and monthly counters when the JST calendar has moved on."""
        now = self.clock.now()
        today, month = jst_date_str(now), jst_month_str(now)
        state = self.store.state.load()
        if state is None:
            state = SystemState.initial(
                self.config.capital.initial_equity_jpy, now, today, month)
            return self.store.state.save(state)

        changed = False
        if state.daily.jst_date != today:
            from ..models.state import DailyCounters

            state.daily = DailyCounters(jst_date=today,
                                        start_equity_jpy=state.equity_jpy)
            changed = True
        if state.monthly.jst_month != month:
            from ..models.state import MonthlyCounters

            # A new month restores the probe budget and the LLM budget together.
            state.monthly = MonthlyCounters(jst_month=month)
            changed = True
        if changed:
            state = self.store.state.save(state)
        return state

    def save_state(self, state: SystemState) -> SystemState:
        state.updated_at = self.clock.now()
        return self.store.state.save(state)

    def budget_state(self, state: SystemState):
        return self.cost_meter.evaluate(state.monthly.llm_cost_jpy)


class _NoExchange:
    """Stands in where a context was built without an exchange.

    Returning None would surface as `AttributeError: 'NoneType' object has no
    attribute 'ticker'` from somewhere deep in a call stack. This says what
    actually happened.
    """

    def __getattr__(self, name):
        raise ConfigError(
            f"this context was built without an exchange, so {name!r} is not "
            "available. Build it with needs_exchange=True if it needs one.")

    def __bool__(self) -> bool:
        return False


def build_context(*, config: Config | None = None, clock: Clock | None = None,
                  store: Store | None = None, exchange=None, secrets=None,
                  notifier=None, owner: str = "trade-agent",
                  needs_trading_credentials: bool = True,
                  needs_exchange: bool = True,
                  offline: bool = False) -> AppContext:
    """Wire up the collaborators a function actually needs.

    `needs_exchange=False` matters for the mcp function specifically. Building
    an exchange is not free: it constructs an HTTP client, and under paper
    trading it also reads the simulated account from S3 before returning. None
    of the MCP tools touch the exchange — all seven read DynamoDB — so on the
    one component reachable from the public internet that work buys nothing
    and can only add ways for a cold start to fail.
    """
    config = config or get_config()
    clock = clock or Clock()
    secrets = secrets or default_provider()
    store = store if store is not None else _build_store(config)
    notifier = notifier if notifier is not None else build_notifier(config)
    if exchange is None:
        exchange = _build_exchange(
            config, store, secrets, clock,
            needs_trading_credentials=needs_trading_credentials,
            offline=offline) if needs_exchange else _NoExchange()
    return AppContext(
        config=config, clock=clock, store=store, exchange=exchange,
        secrets=secrets, notifier=notifier,
        cost_meter=CostMeter(config.llm, config.cost),
        risk=RiskEngine(config), router=ModelRouter(config), owner=owner)


def _build_store(config: Config) -> Store:
    if config.storage.backend == "memory":
        return MemoryStore()
    from ..storage.dynamo import DynamoStore

    return DynamoStore(config.storage)


def _build_exchange(config: Config, store: Store, secrets: SecretProvider,
                    clock: Clock, *, needs_trading_credentials: bool,
                    offline: bool = False):
    if offline:
        if not config.system.paper_trading:
            raise ConfigError("offline mode requires paper trading")
        from ..exchange.synthetic import SyntheticMarket

        return _wrap_paper(config, store, clock,
                           SyntheticMarket(pair=config.exchange.pair,
                                           now=clock.now))

    credentials = None
    if needs_trading_credentials and not config.system.paper_trading:
        key = secrets.get_optional(config.secrets.ssm_bitbank_api_key)
        secret = secrets.get_optional(config.secrets.ssm_bitbank_api_secret)
        if not key or not secret:
            raise ConfigError(
                "live trading is enabled but the bitbank credentials are not "
                "readable from SSM")
        credentials = (key, secret)

    public = BitbankClient(
        public_base_url=config.exchange.public_base_url,
        private_base_url=config.exchange.private_base_url,
        pair=config.exchange.pair,
        credentials=credentials,
        rate_limiter=RateLimiter(config.exchange.rate_limit.query_per_second,
                                 config.exchange.rate_limit.update_per_second),
        timeout=config.exchange.http_timeout_seconds,
        max_retries=config.exchange.max_retries,
        retry_base_delay=config.exchange.retry_base_delay_seconds,
    )
    if not config.system.paper_trading:
        return public
    return _wrap_paper(config, store, clock, public)


def _wrap_paper(config: Config, store: Store, clock: Clock, public):
    """Simulated balances and orders over a real (or synthetic) price feed."""
    from ..exchange.paper import PaperAccount, PaperExchange

    def load_account() -> PaperAccount | None:
        raw = store.blobs.get_json("paper/account.json")
        return PaperAccount.model_validate(raw) if raw else None

    def save_account(account: PaperAccount) -> None:
        store.blobs.put_json("paper/account.json",
                             account.model_dump(mode="json"))

    return PaperExchange(
        public, pair=config.exchange.pair,
        maker_fee_rate=config.exchange.maker_fee_rate,
        taker_fee_rate=config.exchange.taker_fee_rate,
        min_order_btc=config.exchange.min_order_btc,
        initial_jpy=Decimal(config.capital.initial_equity_jpy),
        load_account=load_account, save_account=save_account,
        now=clock.now)
