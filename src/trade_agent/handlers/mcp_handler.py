"""`mcp` Lambda — Function URL entry point (spec 16, 17.1).

This function is built with `needs_trading_credentials=False`: it never asks
SSM for the bitbank key, and its IAM role has no permission to read it either
(spec 12). Defence in depth on the one component that is reachable from the
public internet.

It is also built with `needs_exchange=False`. Every MCP tool reads DynamoDB
and nothing else, so constructing an exchange would add an HTTP client and —
under paper trading — an S3 read to every cold start, in exchange for nothing.
Work not done here cannot fail here.
"""

from __future__ import annotations

import base64
import logging

from ..orchestrator.context import AppContext, build_context
from ..mcp.server import handle_request
from .common import configure_logging

log = logging.getLogger(__name__)

_CACHED_CONTEXT: AppContext | None = None


def handler(event=None, context=None, *, ctx: AppContext | None = None) -> dict:
    configure_logging()
    app = ctx or _context()
    event = event or {}

    method = (event.get("requestContext", {})
              .get("http", {})
              .get("method", event.get("httpMethod", "POST")))
    headers = event.get("headers") or {}
    # Function URLs send payload v2 (`rawPath`); `path` is the v1 spelling.
    # The path can carry the bearer token when the client has no way to set a
    # header — see mcp/auth.py. Never log it.
    path = event.get("rawPath") or event.get("path")
    body = event.get("body")
    if body and event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode()

    status, response_headers, response_body = handle_request(
        app, method=method, headers=headers, body=body, path=path)
    return {
        "statusCode": status,
        "headers": response_headers,
        "body": response_body,
    }


def _context() -> AppContext:
    """Reused across warm invocations; the MCP surface is read-mostly and the
    per-request work is a DynamoDB read, not a rebuild."""
    global _CACHED_CONTEXT
    if _CACHED_CONTEXT is None:
        _CACHED_CONTEXT = build_context(owner="mcp-lambda",
                                        needs_trading_credentials=False,
                                        needs_exchange=False)
    return _CACHED_CONTEXT
