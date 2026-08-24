"""DynamoDB implementation of the repositories (spec 10, 17.1).

Capacity: every table is provisioned at 1 RCU / 1 WCU. The AWS Always-Free
allowance is 25 RCU + 25 WCU *provisioned* across the account, so this keeps
the whole system inside the permanently free tier rather than the 12-month
on-demand promotion. At roughly 300 tick writes and a handful of cycle writes
per day, one unit per table is two orders of magnitude more than needed.

No global secondary indexes: each would cost another provisioned unit pair.
The two listings that are not natural key queries (`orders.list_open`,
`trades.list_*`) use a filtered Scan, which is cheap while those tables hold
hundreds of items — a Scan over a table this size costs a fraction of one RCU.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from ..config import StorageConfig
from ..errors import DuplicateOrder
from ..models.state import SystemState
from ..models.trading import OrderRecord, TradeRecord
from ..timeutil import iso, parse_iso
from .base import (
    AgentCallRecord,
    AuditEvent,
    DailyReport,
    EquityPoint,
    Store,
    StoredLesson,
)

STATE_KEY = "state"
EQUITY_PARTITION = "EQUITY"
REPORT_PARTITION = "REPORT"
AUDIT_PARTITION = "AUDIT"


def _encode(value: Any) -> Any:
    """Model value -> DynamoDB attribute value.

    Decimals pass through untouched (DynamoDB stores them exactly, which is why
    money never becomes a float anywhere in this system). Empty strings become
    None because DynamoDB treats "" as absent in key positions.
    """
    if isinstance(value, datetime):
        return iso(value)
    if isinstance(value, Decimal):
        return value
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return Decimal(repr(value))
    return str(value)


def _to_item(model, **extra) -> dict:
    data = {k: _encode(v) for k, v in model.model_dump().items() if v is not None}
    data.update(extra)
    return data


def _restore_datetimes(model_cls, data: dict) -> dict:
    """ISO strings back into datetimes for the fields the model declares as such."""
    out = dict(data)
    for name, field in model_cls.model_fields.items():
        value = out.get(name)
        if isinstance(value, str) and "datetime" in str(field.annotation):
            try:
                out[name] = parse_iso(value)
            except ValueError:
                pass
    return out


def _from_item(model_cls, item: dict | None):
    if item is None:
        return None
    payload = {k: v for k, v in item.items() if k in model_cls.model_fields}
    return model_cls.model_validate(_restore_datetimes(model_cls, payload))


class _Table:
    def __init__(self, resource, name: str):
        self.name = name
        self.table = resource.Table(name)


class _StateRepo(_Table):
    def load(self) -> SystemState | None:
        item = self.table.get_item(Key={"pk": STATE_KEY}).get("Item")
        if item is None:
            return None
        position = item.get("open_position")
        state = _from_item(SystemState, item)
        if state is not None and position:
            from ..models.trading import Position
            state.open_position = Position.model_validate(
                _restore_datetimes(Position, position))
        return state

    def save(self, state: SystemState) -> SystemState:
        """Optimistic concurrency: the write fails if another invocation has
        advanced `version` since this one read the item."""
        expected = state.version
        state = state.model_copy(deep=True)
        state.version = expected + 1
        item = _to_item(state, pk=STATE_KEY)
        if state.open_position is not None:
            item["open_position"] = _to_item(state.open_position)
        kwargs: dict[str, Any] = {"Item": item}
        if expected == 0:
            kwargs["ConditionExpression"] = (
                "attribute_not_exists(pk) OR version = :v")
            kwargs["ExpressionAttributeValues"] = {":v": 0}
        else:
            kwargs["ConditionExpression"] = "version = :v"
            kwargs["ExpressionAttributeValues"] = {":v": expected}
        self.table.put_item(**kwargs)
        return state


class _OrderRepo(_Table):
    def put_pending(self, record: OrderRecord) -> None:
        from botocore.exceptions import ClientError

        try:
            self.table.put_item(
                Item=_to_item(record, pk=record.client_order_id),
                ConditionExpression="attribute_not_exists(pk)")
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise DuplicateOrder(
                    f"order {record.client_order_id} already recorded") from exc
            raise

    def update(self, record: OrderRecord) -> None:
        self.table.put_item(Item=_to_item(record, pk=record.client_order_id))

    def get(self, client_order_id: str) -> OrderRecord | None:
        item = self.table.get_item(Key={"pk": client_order_id}).get("Item")
        return _from_item(OrderRecord, item)

    def list_open(self) -> list[OrderRecord]:
        rows = _scan(self.table, filter_expression="#s IN (:a, :b, :c, :d)",
                     names={"#s": "status"},
                     values={":a": "pending", ":b": "submitted",
                             ":c": "partially_filled", ":d": "unknown"})
        return [r for r in (_from_item(OrderRecord, i) for i in rows) if r]

    def list_recent(self, limit: int = 50) -> list[OrderRecord]:
        rows = [r for r in (_from_item(OrderRecord, i)
                            for i in _scan(self.table)) if r]
        rows.sort(key=lambda o: o.created_at, reverse=True)
        return rows[:limit]


class _LockRepo(_Table):
    def acquire(self, name: str, owner: str, ttl_seconds: int,
                now: datetime) -> bool:
        """Lease-based lock (spec 8): take it when it is free, expired, or
        already ours. A crashed holder's lease simply runs out."""
        from botocore.exceptions import ClientError

        expires = now + timedelta(seconds=ttl_seconds)
        try:
            self.table.put_item(
                Item={"pk": name, "owner": owner, "expires_at": iso(expires),
                      "ttl": int(expires.timestamp()) + 3600,
                      "acquired_at": iso(now)},
                ConditionExpression=(
                    "attribute_not_exists(pk) OR expires_at < :now "
                    "OR #o = :owner"),
                ExpressionAttributeNames={"#o": "owner"},
                ExpressionAttributeValues={":now": iso(now), ":owner": owner})
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise

    def release(self, name: str, owner: str) -> None:
        from botocore.exceptions import ClientError

        try:
            self.table.delete_item(
                Key={"pk": name},
                ConditionExpression="#o = :owner",
                ExpressionAttributeNames={"#o": "owner"},
                ExpressionAttributeValues={":owner": owner})
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
                raise


class _TradeRepo(_Table):
    def put(self, trade: TradeRecord) -> None:
        self.table.put_item(Item=_to_item(trade, pk=trade.trade_id))

    def get(self, trade_id: str) -> TradeRecord | None:
        return _from_item(TradeRecord,
                          self.table.get_item(Key={"pk": trade_id}).get("Item"))

    def list_recent(self, limit: int = 50, *,
                    include_probe: bool = True) -> list[TradeRecord]:
        rows = [t for t in (_from_item(TradeRecord, i) for i in _scan(self.table)) if t]
        if not include_probe:
            rows = [t for t in rows if not t.probe]
        rows.sort(key=lambda t: t.entry_at, reverse=True)
        return rows[:limit]

    def list_between(self, start: datetime, end: datetime) -> list[TradeRecord]:
        rows = [t for t in (_from_item(TradeRecord, i) for i in _scan(self.table)) if t]
        rows = [t for t in rows if start <= t.entry_at <= end]
        rows.sort(key=lambda t: t.entry_at)
        return rows


class _AgentCallRepo(_Table):
    def put(self, record: AgentCallRecord) -> None:
        self.table.put_item(
            Item=_to_item(record, pk=record.cycle_id, sk=record.key))

    def list_for_cycle(self, cycle_id: str) -> list[AgentCallRecord]:
        from boto3.dynamodb.conditions import Key

        response = self.table.query(KeyConditionExpression=Key("pk").eq(cycle_id))
        rows = [r for r in (_from_item(AgentCallRecord, i)
                            for i in response.get("Items", [])) if r]
        rows.sort(key=lambda c: (c.sequence, c.agent))
        return rows


class _LessonRepo(_Table):
    def put(self, lesson: StoredLesson) -> None:
        self.table.put_item(Item=_to_item(
            lesson, pk=lesson.regime_tag,
            sk=f"{iso(lesson.created_at)}#{lesson.lesson_id}"))

    def list(self, *, regime: str | None = None,
             limit: int = 20) -> list[StoredLesson]:
        from boto3.dynamodb.conditions import Key

        partitions = ["all"] if regime is None else [regime, "all"]
        rows: list[StoredLesson] = []
        for partition in dict.fromkeys(partitions):
            response = self.table.query(
                KeyConditionExpression=Key("pk").eq(partition),
                ScanIndexForward=False, Limit=limit)
            rows.extend(r for r in (_from_item(StoredLesson, i)
                                    for i in response.get("Items", [])) if r)
        rows.sort(key=lambda x: x.created_at, reverse=True)
        return rows[:limit]


class _EquityRepo(_Table):
    def put(self, point: EquityPoint) -> None:
        self.table.put_item(
            Item=_to_item(point, pk=EQUITY_PARTITION, sk=point.jst_date))

    def get(self, jst_date: str) -> EquityPoint | None:
        item = self.table.get_item(
            Key={"pk": EQUITY_PARTITION, "sk": jst_date}).get("Item")
        return _from_item(EquityPoint, item)

    def list_recent(self, limit: int = 30) -> list[EquityPoint]:
        from boto3.dynamodb.conditions import Key

        response = self.table.query(
            KeyConditionExpression=Key("pk").eq(EQUITY_PARTITION),
            ScanIndexForward=False, Limit=limit)
        return [r for r in (_from_item(EquityPoint, i)
                            for i in response.get("Items", [])) if r]


class _ReportRepo(_Table):
    def put(self, report: DailyReport) -> None:
        self.table.put_item(
            Item=_to_item(report, pk=REPORT_PARTITION, sk=report.jst_date))

    def get(self, jst_date: str | None = None) -> DailyReport | None:
        from boto3.dynamodb.conditions import Key

        if jst_date:
            item = self.table.get_item(
                Key={"pk": REPORT_PARTITION, "sk": jst_date}).get("Item")
            return _from_item(DailyReport, item)
        response = self.table.query(
            KeyConditionExpression=Key("pk").eq(REPORT_PARTITION),
            ScanIndexForward=False, Limit=1)
        items = response.get("Items", [])
        return _from_item(DailyReport, items[0]) if items else None


class _AuditRepo(_Table):
    def put(self, event: AuditEvent) -> None:
        self.table.put_item(Item=_to_item(
            event, pk=AUDIT_PARTITION, sk=f"{iso(event.at)}#{event.event_id}"))

    def list_recent(self, limit: int = 50) -> list[AuditEvent]:
        from boto3.dynamodb.conditions import Key

        response = self.table.query(
            KeyConditionExpression=Key("pk").eq(AUDIT_PARTITION),
            ScanIndexForward=False, Limit=limit)
        return [r for r in (_from_item(AuditEvent, i)
                            for i in response.get("Items", [])) if r]


class S3BlobStore:
    """Agent input/output bodies and daily backups (spec 10)."""

    def __init__(self, bucket: str, client=None):
        self.bucket = bucket
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("s3")
        return self._client

    def put_json(self, key: str, payload: dict) -> str:
        import json

        self.client.put_object(
            Bucket=self.bucket, Key=key,
            Body=json.dumps(payload, ensure_ascii=False, default=str).encode(),
            ContentType="application/json")
        return key

    def get_json(self, key: str) -> dict | None:
        import json

        from botocore.exceptions import ClientError

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"NoSuchKey", "404"}:
                return None
            raise
        return json.loads(response["Body"].read())


def _scan(table, *, filter_expression: str | None = None,
          names: dict | None = None, values: dict | None = None) -> list[dict]:
    kwargs: dict[str, Any] = {}
    if filter_expression:
        kwargs["FilterExpression"] = filter_expression
        if names:
            kwargs["ExpressionAttributeNames"] = names
        if values:
            kwargs["ExpressionAttributeValues"] = values
    items: list[dict] = []
    while True:
        response = table.scan(**kwargs)
        items.extend(response.get("Items", []))
        token = response.get("LastEvaluatedKey")
        if not token:
            return items
        kwargs["ExclusiveStartKey"] = token


class DynamoStore(Store):
    def __init__(self, config: StorageConfig, *, resource=None, s3_client=None):
        if resource is None:
            import boto3

            resource = boto3.resource(
                "dynamodb", region_name=os.environ.get("AWS_REGION", "ap-northeast-1"))
        table = config.table
        super().__init__(
            state=_StateRepo(resource, table("state")),
            orders=_OrderRepo(resource, table("orders")),
            locks=_LockRepo(resource, table("locks")),
            trades=_TradeRepo(resource, table("trades")),
            agent_calls=_AgentCallRepo(resource, table("agent-calls")),
            lessons=_LessonRepo(resource, table("lessons")),
            equity=_EquityRepo(resource, table("equity-curve")),
            reports=_ReportRepo(resource, table("daily-reports")),
            audit=_AuditRepo(resource, table("audit")),
            blobs=S3BlobStore(config.s3_bucket, s3_client),
        )
