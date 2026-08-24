"""Repository interfaces and the records they hold.

Spec 10 asks for `trades`, `agent_calls`, `lessons`, `equity_curve`,
`daily_reports` and `system_state`. Orders live in their own `orders` table
rather than inside the `system_state` item: the idempotency guarantee in spec 8
is a *conditional write on the order's own key*, which needs one item per
order. The spec allows equivalent entities, and this is the shape that makes
the guarantee real.

Agent input/output JSON never goes into DynamoDB — the item limit is 400KB, so
bodies go to S3 and the table keeps the pointer (spec 10).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ..models.state import SystemState
from ..models.trading import OrderRecord, TradeRecord
from ..money import ZERO

LOCK_DECIDE = "decide-cycle"
LOCK_EXECUTION = "execution"


class _Rec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentCallRecord(_Rec):
    """One LLM call. The audit trail for "why did it do that" (spec 14)."""

    cycle_id: str
    agent: str
    sequence: int = 0
    called_at: datetime
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_jpy: Decimal = ZERO
    retries: int = 0
    ok: bool = True
    error: str | None = None
    duration_ms: int = 0
    io_s3_key: str | None = None
    batch: bool = False
    # Spec 4.1: proof that phase-1 proposals were made blind. Lists the agent
    # names whose output was visible in this call's prompt.
    saw_agents: list[str] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.cycle_id}#{self.sequence:02d}#{self.agent}"


class StoredLesson(_Rec):
    lesson_id: str
    created_at: datetime
    text: str
    regime_tag: str
    evidence: str
    confidence: float
    trades_analysed: int = 0
    source_cycle_id: str | None = None


class EquityPoint(_Rec):
    jst_date: str
    equity_jpy: Decimal
    realized_pnl_jpy: Decimal = ZERO
    cumulative_llm_cost_jpy: Decimal = ZERO
    infra_cost_jpy: Decimal = ZERO
    kill_switch: bool = False
    trades: int = 0
    probe_trades: int = 0
    updated_at: datetime | None = None


class DailyReport(_Rec):
    jst_date: str
    created_at: datetime
    headline: str
    report_text: str
    equity_jpy: Decimal
    realized_pnl_jpy: Decimal
    llm_cost_month_jpy: Decimal
    consensus_rate: float | None = None
    hours_since_last_entry: float | None = None


class AuditEvent(_Rec):
    """Operator actions taken through MCP (spec 16.2) and system halts."""

    event_id: str
    at: datetime
    actor: str
    action: str
    detail: str = ""


@runtime_checkable
class StateRepository(Protocol):
    def load(self) -> SystemState | None: ...
    def save(self, state: SystemState) -> SystemState: ...


@runtime_checkable
class OrderRepository(Protocol):
    def put_pending(self, record: OrderRecord) -> None:
        """Conditional create. Raises DuplicateOrder if the id already exists."""

    def update(self, record: OrderRecord) -> None: ...
    def get(self, client_order_id: str) -> OrderRecord | None: ...
    def list_open(self) -> list[OrderRecord]: ...

    def list_recent(self, limit: int = 50) -> list[OrderRecord]:
        """Most recently created first, open or not.

        A *filled* entry is terminal but still needs attention: until its
        position is tracked it has no stop attached to it.
        """


@runtime_checkable
class LockRepository(Protocol):
    def acquire(self, name: str, owner: str, ttl_seconds: int,
                now: datetime) -> bool: ...

    def release(self, name: str, owner: str) -> None: ...


@runtime_checkable
class TradeRepository(Protocol):
    def put(self, trade: TradeRecord) -> None: ...
    def get(self, trade_id: str) -> TradeRecord | None: ...
    def list_recent(self, limit: int = 50, *,
                    include_probe: bool = True) -> list[TradeRecord]: ...

    def list_between(self, start: datetime, end: datetime) -> list[TradeRecord]: ...


@runtime_checkable
class AgentCallRepository(Protocol):
    def put(self, record: AgentCallRecord) -> None: ...
    def list_for_cycle(self, cycle_id: str) -> list[AgentCallRecord]: ...


@runtime_checkable
class LessonRepository(Protocol):
    def put(self, lesson: StoredLesson) -> None: ...
    def list(self, *, regime: str | None = None,
             limit: int = 20) -> list[StoredLesson]: ...


@runtime_checkable
class EquityRepository(Protocol):
    def put(self, point: EquityPoint) -> None: ...
    def get(self, jst_date: str) -> EquityPoint | None: ...
    def list_recent(self, limit: int = 30) -> list[EquityPoint]: ...


@runtime_checkable
class ReportRepository(Protocol):
    def put(self, report: DailyReport) -> None: ...
    def get(self, jst_date: str | None = None) -> DailyReport | None: ...


@runtime_checkable
class AuditRepository(Protocol):
    def put(self, event: AuditEvent) -> None: ...
    def list_recent(self, limit: int = 50) -> list[AuditEvent]: ...


@runtime_checkable
class BlobStore(Protocol):
    def put_json(self, key: str, payload: dict) -> str: ...
    def get_json(self, key: str) -> dict | None: ...


class Store:
    """Bundle of repositories handed to every layer that needs persistence."""

    def __init__(self, *, state, orders, locks, trades, agent_calls, lessons,
                 equity, reports, audit, blobs):
        self.state: StateRepository = state
        self.orders: OrderRepository = orders
        self.locks: LockRepository = locks
        self.trades: TradeRepository = trades
        self.agent_calls: AgentCallRepository = agent_calls
        self.lessons: LessonRepository = lessons
        self.equity: EquityRepository = equity
        self.reports: ReportRepository = reports
        self.audit: AuditRepository = audit
        self.blobs: BlobStore = blobs
