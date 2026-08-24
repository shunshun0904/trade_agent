"""Exception hierarchy.

Every failure mode the system is expected to survive gets its own type so
handlers can apply the spec's "when in doubt, do nothing" rule (spec 6)
without string-matching messages.
"""

from __future__ import annotations


class TradeAgentError(Exception):
    """Base class for all errors raised by this package."""


class ConfigError(TradeAgentError):
    """Configuration missing or internally inconsistent."""


class ExchangeError(TradeAgentError):
    """Any failure talking to the exchange."""

    def __init__(self, message: str, *, code: int | None = None, status: int | None = None):
        super().__init__(message)
        self.code = code
        self.status = status


class ExchangeRateLimited(ExchangeError):
    """HTTP 429 from the exchange."""


class InsufficientFunds(ExchangeError):
    """Balance too small for the requested order."""


class LockNotAcquired(TradeAgentError):
    """Another invocation holds the cycle lock (spec 8 distributed lock)."""


class DuplicateOrder(TradeAgentError):
    """A pending row for this client_order_id already exists (spec 8)."""


class GuardRejection(TradeAgentError):
    """Deterministic guard refused an agent's output (spec 5)."""

    def __init__(self, message: str, *, violations: list[str] | None = None,
                 retry_target: str | None = None):
        super().__init__(message)
        self.violations = violations or []
        self.retry_target = retry_target


class LLMError(TradeAgentError):
    """LLM transport failure or unparseable output."""


class BudgetExhausted(TradeAgentError):
    """Monthly LLM budget reached; no further LLM calls this month (spec 11)."""


class KillSwitchActive(TradeAgentError):
    """The kill switch is engaged; only `resume_trading` clears it (spec 6)."""
