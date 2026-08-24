"""Shared handler plumbing.

Every Lambda in this system follows the same shape: build the context, do the
work, return a JSON-serialisable summary, and never let an unexpected
exception escape silently — spec 16.4 requires the owner to hear about a
process crash, and a Lambda that dies quietly is exactly the failure the
dead-man's switch exists to catch (spec 17.3).
"""

from __future__ import annotations

import json
import logging
import os
from decimal import Decimal
from typing import Callable

from ..config import get_config
from ..orchestrator.context import AppContext, build_context

log = logging.getLogger()


def configure_logging() -> None:
    level = os.environ.get("TA_LOG_LEVEL") or get_config().system.log_level
    logging.getLogger().setLevel(level)


def json_default(value):
    if isinstance(value, Decimal):
        return float(value)
    return str(value)


def dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, default=json_default)


def run_handler(name: str, work: Callable[[AppContext], dict], *,
                ctx: AppContext | None = None,
                needs_trading_credentials: bool = True) -> dict:
    configure_logging()
    context = ctx or build_context(owner=f"{name}-lambda",
                                   needs_trading_credentials=needs_trading_credentials)
    try:
        result = work(context)
    except Exception as exc:  # noqa: BLE001 - reported, then re-raised
        log.exception("%s failed", name)
        _notify_crash(context, name, exc)
        raise
    log.info("%s: %s", name, dumps(result))
    return result


def _notify_crash(ctx: AppContext, name: str, exc: Exception) -> None:
    if ctx.notifier is None:
        return
    try:
        ctx.notifier.send(
            f"{name} が異常終了",
            f"Lambda {name} が例外で終了しました。\n\n{type(exc).__name__}: {exc}\n\n"
            "CloudWatch Logs を確認してください。5分tickが継続しているかどうかは"
            "デッドマンスイッチのアラームで検知されます。")
    except Exception:  # noqa: BLE001
        log.exception("could not send the crash notification")
