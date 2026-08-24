"""Email notification via SES (spec 16.4, 17.1).

The owner-facing MCP server is pull-only — claude.ai cannot receive a push
(spec 16.1) — so email is the only channel that reaches the owner when
something goes wrong while they are not looking. Spec 16.4 marks it
non-removable, and this module is deliberately boring for that reason.

Sender and recipient come from configuration and cannot be overridden per call
(spec 12): there is no code path that emails an arbitrary address.
"""

from __future__ import annotations

import logging
import os
from enum import StrEnum

from ..config import Config

log = logging.getLogger(__name__)


class NotifyEvent(StrEnum):
    """The events spec 16.4 requires an email for."""

    KILL_SWITCH = "kill_switch"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    LOSING_STREAK = "losing_streak"
    FILL = "fill"
    API_FAILURE = "api_failure"
    PROCESS_ERROR = "process_error"
    BUDGET_EXHAUSTED = "budget_exhausted"


SUBJECT_PREFIX = "[trade-agent]"


class Notifier:
    def __init__(self, config: Config, client=None):
        self.config = config
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "ses", region_name=self.config.notify.ses_region)
        return self._client

    def send(self, subject: str, body: str,
             event: NotifyEvent | None = None) -> bool:
        cfg = self.config.notify
        if not cfg.enabled:
            log.info("notification suppressed (disabled): %s", subject)
            return False
        if not cfg.from_address or not cfg.to_address:
            log.warning("notification not sent, addresses unconfigured: %s", subject)
            return False

        env = self.config.system.environment
        full_subject = f"{SUBJECT_PREFIX}[{env}] {subject}"
        footer = (
            "\n\n--\n"
            "この通知は自動送信です。状況の照会と操作は Claude のカスタムコネクタ"
            f"({self.config.mcp.server_name})から行ってください。\n"
            f"event={event or 'generic'} phase={self.config.system.phase} "
            f"paper={self.config.system.paper_trading}\n")
        try:
            self.client.send_email(
                Source=cfg.from_address,
                Destination={"ToAddresses": [cfg.to_address]},
                Message={
                    "Subject": {"Data": full_subject, "Charset": "UTF-8"},
                    "Body": {"Text": {"Data": body + footer, "Charset": "UTF-8"}},
                })
            return True
        except Exception as exc:  # noqa: BLE001 - a failed email must not stop trading
            log.error("failed to send notification %r: %s", full_subject, exc)
            return False


class NullNotifier:
    """Records instead of sending. Used in tests and local runs."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send(self, subject: str, body: str,
             event: NotifyEvent | None = None) -> bool:
        self.sent.append((subject, body))
        log.info("notification (not sent): %s", subject)
        return True


def build_notifier(config: Config):
    if os.environ.get("TA_DISABLE_EMAIL") == "1" or not config.notify.enabled:
        return NullNotifier()
    if not config.notify.from_address or not config.notify.to_address:
        log.warning("SES addresses are not configured; emergency email is inert")
        return NullNotifier()
    return Notifier(config)
