"""In-memory store.

Used by the test suite and by `trade-agent` CLI dry runs. It implements the
same conditional-write semantics as DynamoDB — `put_pending` refuses an
existing key, `acquire` refuses a live lease — so idempotency tests exercise
the real contract rather than a permissive stub.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta

from ..errors import DuplicateOrder
from ..models.state import SystemState
from ..models.trading import OrderRecord, TradeRecord
from .base import (
    AgentCallRecord,
    AuditEvent,
    DailyReport,
    EquityPoint,
    Store,
    StoredLesson,
)


class _StateRepo:
    def __init__(self) -> None:
        self._state: SystemState | None = None

    def load(self) -> SystemState | None:
        return self._state.model_copy(deep=True) if self._state else None

    def save(self, state: SystemState) -> SystemState:
        state = state.model_copy(deep=True)
        state.version += 1
        self._state = state
        return state.model_copy(deep=True)


class _OrderRepo:
    def __init__(self) -> None:
        self._orders: dict[str, OrderRecord] = {}

    def put_pending(self, record: OrderRecord) -> None:
        if record.client_order_id in self._orders:
            raise DuplicateOrder(
                f"order {record.client_order_id} already recorded")
        self._orders[record.client_order_id] = record.model_copy(deep=True)

    def update(self, record: OrderRecord) -> None:
        self._orders[record.client_order_id] = record.model_copy(deep=True)

    def get(self, client_order_id: str) -> OrderRecord | None:
        found = self._orders.get(client_order_id)
        return found.model_copy(deep=True) if found else None

    def list_open(self) -> list[OrderRecord]:
        return [o.model_copy(deep=True) for o in self._orders.values()
                if o.status.is_open]

    def list_recent(self, limit: int = 50) -> list[OrderRecord]:
        rows = sorted(self._orders.values(), key=lambda o: o.created_at,
                      reverse=True)
        return [o.model_copy(deep=True) for o in rows[:limit]]

    def list_all(self) -> list[OrderRecord]:
        return [o.model_copy(deep=True) for o in self._orders.values()]


class _LockRepo:
    def __init__(self) -> None:
        self._leases: dict[str, tuple[str, datetime]] = {}

    def acquire(self, name: str, owner: str, ttl_seconds: int,
                now: datetime) -> bool:
        held = self._leases.get(name)
        if held is not None:
            holder, expires = held
            if now < expires and holder != owner:
                return False
        self._leases[name] = (owner, now + timedelta(seconds=ttl_seconds))
        return True

    def release(self, name: str, owner: str) -> None:
        held = self._leases.get(name)
        if held and held[0] == owner:
            del self._leases[name]


class _TradeRepo:
    def __init__(self) -> None:
        self._trades: dict[str, TradeRecord] = {}

    def put(self, trade: TradeRecord) -> None:
        self._trades[trade.trade_id] = trade.model_copy(deep=True)

    def get(self, trade_id: str) -> TradeRecord | None:
        found = self._trades.get(trade_id)
        return found.model_copy(deep=True) if found else None

    def list_recent(self, limit: int = 50, *,
                    include_probe: bool = True) -> list[TradeRecord]:
        rows = sorted(self._trades.values(), key=lambda t: t.entry_at, reverse=True)
        if not include_probe:
            rows = [t for t in rows if not t.probe]
        return [t.model_copy(deep=True) for t in rows[:limit]]

    def list_between(self, start: datetime, end: datetime) -> list[TradeRecord]:
        rows = [t for t in self._trades.values() if start <= t.entry_at <= end]
        rows.sort(key=lambda t: t.entry_at)
        return [t.model_copy(deep=True) for t in rows]


class _AgentCallRepo:
    def __init__(self) -> None:
        self._calls: dict[str, AgentCallRecord] = {}

    def put(self, record: AgentCallRecord) -> None:
        self._calls[record.key] = record.model_copy(deep=True)

    def list_for_cycle(self, cycle_id: str) -> list[AgentCallRecord]:
        rows = [c for c in self._calls.values() if c.cycle_id == cycle_id]
        rows.sort(key=lambda c: (c.sequence, c.agent))
        return [c.model_copy(deep=True) for c in rows]

    def list_all(self) -> list[AgentCallRecord]:
        return [c.model_copy(deep=True) for c in self._calls.values()]


class _LessonRepo:
    def __init__(self) -> None:
        self._lessons: dict[str, StoredLesson] = {}

    def put(self, lesson: StoredLesson) -> None:
        self._lessons[lesson.lesson_id] = lesson.model_copy(deep=True)

    def list(self, *, regime: str | None = None,
             limit: int = 20) -> list[StoredLesson]:
        rows = list(self._lessons.values())
        if regime:
            rows = [x for x in rows if x.regime_tag in {regime, "all"}]
        rows.sort(key=lambda x: x.created_at, reverse=True)
        return [x.model_copy(deep=True) for x in rows[:limit]]


class _EquityRepo:
    def __init__(self) -> None:
        self._points: dict[str, EquityPoint] = {}

    def put(self, point: EquityPoint) -> None:
        self._points[point.jst_date] = point.model_copy(deep=True)

    def get(self, jst_date: str) -> EquityPoint | None:
        found = self._points.get(jst_date)
        return found.model_copy(deep=True) if found else None

    def list_recent(self, limit: int = 30) -> list[EquityPoint]:
        rows = sorted(self._points.values(), key=lambda p: p.jst_date, reverse=True)
        return [p.model_copy(deep=True) for p in rows[:limit]]


class _ReportRepo:
    def __init__(self) -> None:
        self._reports: dict[str, DailyReport] = {}

    def put(self, report: DailyReport) -> None:
        self._reports[report.jst_date] = report.model_copy(deep=True)

    def get(self, jst_date: str | None = None) -> DailyReport | None:
        if not self._reports:
            return None
        key = jst_date or max(self._reports)
        found = self._reports.get(key)
        return found.model_copy(deep=True) if found else None


class _AuditRepo:
    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def put(self, event: AuditEvent) -> None:
        self._events.append(event.model_copy(deep=True))

    def list_recent(self, limit: int = 50) -> list[AuditEvent]:
        return [e.model_copy(deep=True) for e in
                sorted(self._events, key=lambda e: e.at, reverse=True)[:limit]]


class _BlobStore:
    def __init__(self) -> None:
        self.objects: dict[str, dict] = {}

    def put_json(self, key: str, payload: dict) -> str:
        self.objects[key] = copy.deepcopy(payload)
        return key

    def get_json(self, key: str) -> dict | None:
        found = self.objects.get(key)
        return copy.deepcopy(found) if found else None


class MemoryStore(Store):
    def __init__(self) -> None:
        super().__init__(
            state=_StateRepo(),
            orders=_OrderRepo(),
            locks=_LockRepo(),
            trades=_TradeRepo(),
            agent_calls=_AgentCallRepo(),
            lessons=_LessonRepo(),
            equity=_EquityRepo(),
            reports=_ReportRepo(),
            audit=_AuditRepo(),
            blobs=_BlobStore(),
        )
