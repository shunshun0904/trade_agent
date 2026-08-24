"""Bearer authentication for the MCP endpoint (spec 16.3).

The endpoint is a Lambda Function URL with `AuthType: NONE`, because Anthropic's
servers connect to it from outside AWS and cannot sign SigV4. That makes the
application-layer check the only thing between the internet and the pause /
resume controls, so it is `AuthType: NONE` at the infrastructure layer and
never at this one: a request without a valid token gets 401, always.

Comparison is constant-time. A timing oracle on a token that can stop trading
is not a theoretical concern worth taking on.
"""

from __future__ import annotations

import hmac
import logging

from ..config import Config
from ..errors import ConfigError

log = logging.getLogger(__name__)

WWW_AUTHENTICATE = 'Bearer realm="trade-agent", error="invalid_token"'


class AuthResult:
    def __init__(self, ok: bool, reason: str = ""):
        self.ok = ok
        self.reason = reason

    def __bool__(self) -> bool:
        return self.ok


def extract_bearer(headers: dict) -> str | None:
    """Header names arrive lower-cased through Function URLs, but not every
    client is consistent, so match case-insensitively."""
    for name, value in (headers or {}).items():
        if name.lower() != "authorization":
            continue
        parts = str(value).split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
    return None


def authenticate(config: Config, headers: dict, secrets) -> AuthResult:
    try:
        expected = secrets.get(config.mcp.ssm_bearer_token_param)
    except ConfigError as exc:
        # Fail closed. An unreadable token means no one gets in, including us.
        log.error("MCP bearer token unavailable: %s", exc)
        return AuthResult(False, "server is not configured for authentication")

    if not expected:
        return AuthResult(False, "server is not configured for authentication")

    presented = extract_bearer(headers)
    if not presented:
        return AuthResult(False, "missing bearer token")
    if not hmac.compare_digest(presented, expected):
        return AuthResult(False, "invalid bearer token")
    return AuthResult(True)
