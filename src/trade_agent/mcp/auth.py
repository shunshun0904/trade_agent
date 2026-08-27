"""Bearer authentication for the MCP endpoint (spec 16.3).

The endpoint is a Lambda Function URL with `AuthType: NONE`, because Anthropic's
servers connect to it from outside AWS and cannot sign SigV4. That makes the
application-layer check the only thing between the internet and the pause /
resume controls, so it is `AuthType: NONE` at the infrastructure layer and
never at this one: a request without a valid token gets 401, always.

The token may arrive two ways, and the second one exists for a reason worth
recording. `Authorization: Bearer <token>` is the right shape and is what
`curl` and Claude Code use. But claude.ai's "Add custom connector" dialog
offers only OAuth or no authentication — for an individual account there is
nowhere to enter a static header — so a server like this one, which has a
shared secret and no authorization server, cannot be registered with a header
at all. The fallback is to carry the token in the path: `/mcp/<token>`.

That is genuinely weaker than a header. A URL is the part of a request most
likely to be written down — proxy logs, browser history, a screenshot of a
settings page — whereas a header rarely is. It is used here because the
alternative is not "use a header instead", it is "stand up an OAuth 2.1
authorization server on the one internet-facing component of a trading
system". The mitigations are that nothing in this codebase logs the request
path, and that the token is 256 bits of URL-safe randomness, so the path
cannot be guessed and does not survive as a recognisable secret in a URL.

Comparison is constant-time whichever way the token arrived. A timing oracle
on a token that can stop trading is not a theoretical concern worth taking on.
"""

from __future__ import annotations

import hmac
import logging
from urllib.parse import unquote

from ..config import Config
from ..errors import ConfigError

log = logging.getLogger(__name__)

WWW_AUTHENTICATE = 'Bearer realm="trade-agent", error="invalid_token"'

# The single path shape that carries a token. Anything else is treated as an
# ordinary path with no credential, so a stray request to `/` is a clean 401
# rather than a comparison against whatever the path happened to contain.
PATH_PREFIX = "/mcp/"


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


def extract_path_token(path: str | None) -> str | None:
    """The token from a `/mcp/<token>` request path, if it is shaped like one.

    Percent-decoded, because a client that reaches this endpoint from a
    configured URL may have re-encoded it on the way.
    """
    if not path or not path.startswith(PATH_PREFIX):
        return None
    token = path[len(PATH_PREFIX):].strip("/")
    return unquote(token) or None


def authenticate(config: Config, headers: dict, secrets,
                 path: str | None = None) -> AuthResult:
    try:
        expected = secrets.get(config.mcp.ssm_bearer_token_param)
    except ConfigError as exc:
        # Fail closed. An unreadable token means no one gets in, including us.
        log.error("MCP bearer token unavailable: %s", exc)
        return AuthResult(False, "server is not configured for authentication")

    if not expected:
        return AuthResult(False, "server is not configured for authentication")

    # Header first: it is the better shape, and a client that can send one
    # should not be graded on its URL.
    presented = extract_bearer(headers) or extract_path_token(path)
    if not presented:
        return AuthResult(False, "missing bearer token")
    if not hmac.compare_digest(presented, expected):
        return AuthResult(False, "invalid bearer token")
    return AuthResult(True)
