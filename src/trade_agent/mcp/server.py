"""Streamable-HTTP MCP server, stateless (spec 16.1).

Stateless is a requirement rather than a simplification: the server lives on a
Lambda Function URL, so two consecutive requests from Claude may land on
different containers. Every request therefore carries everything needed to
answer it, and no session state is kept between them.

Transport shape: JSON-RPC 2.0 over POST, responses as `application/json`.
Notifications (a request with no `id`) get HTTP 202 and no body, per the
Streamable HTTP transport.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..config import Config
from ..orchestrator.context import AppContext
from .auth import WWW_AUTHENTICATE, authenticate
from .tools import TOOLS, ToolError, call_tool

log = logging.getLogger(__name__)

JSONRPC = "2.0"
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

SERVER_INSTRUCTIONS = (
    "BTC/JPY自動売買システムの状況照会と操作を行うサーバー。\n"
    "読取ツールは自由に呼び出してよい。pause_trading と resume_trading は"
    "オーナーが明示的に指示したときだけ、confirm=true を付けて呼び出すこと。\n"
    "特に resume_trading は、キルスイッチが発動した理由をオーナーが把握した上で"
    "の指示がない限り呼び出してはならない。"
)


class MCPServer:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.config: Config = ctx.config

    def handle(self, message: dict) -> dict | None:
        """Returns the JSON-RPC response, or None for a notification."""
        if not isinstance(message, dict) or message.get("jsonrpc") != JSONRPC:
            return _error(message.get("id") if isinstance(message, dict) else None,
                          INVALID_REQUEST, "expected a JSON-RPC 2.0 request")

        method = message.get("method")
        request_id = message.get("id")
        params = message.get("params") or {}

        if request_id is None:
            # A notification: acknowledge by doing nothing, and never answer.
            log.info("mcp notification: %s", method)
            return None

        try:
            if method == "initialize":
                return _ok(request_id, self._initialize(params))
            if method == "ping":
                return _ok(request_id, {})
            if method == "tools/list":
                return _ok(request_id, {"tools": TOOLS})
            if method == "tools/call":
                return _ok(request_id, self._call(params))
            return _error(request_id, METHOD_NOT_FOUND, f"unknown method: {method}")
        except ToolError as exc:
            return _ok(request_id, _tool_failure(str(exc)))
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as an error
            log.exception("mcp request failed")
            return _error(request_id, INTERNAL_ERROR, str(exc))

    def _initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": self.config.mcp.protocol_version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": self.config.mcp.server_name,
                           "version": _version()},
            "instructions": SERVER_INSTRUCTIONS,
        }

    def _call(self, params: dict) -> dict:
        name = params.get("name")
        if not name:
            raise ToolError("tools/call requires a tool name")
        arguments = params.get("arguments") or {}
        result = call_tool(self.ctx, name, arguments)
        text = json.dumps(result, ensure_ascii=False, indent=1)
        return {"content": [{"type": "text", "text": text}],
                "structuredContent": result,
                "isError": False}


def handle_request(ctx: AppContext, *, method: str, headers: dict,
                   body: str | None) -> tuple[int, dict, str]:
    """HTTP-level entry point. Returns (status, headers, body)."""
    json_headers = {"Content-Type": "application/json"}

    if method.upper() == "GET":
        # No server-initiated streaming: this server never pushes (spec 16.1).
        return 405, {**json_headers, "Allow": "POST"}, json.dumps(
            {"error": "this MCP server accepts POST only"})
    if method.upper() != "POST":
        return 405, {**json_headers, "Allow": "POST"}, json.dumps(
            {"error": "method not allowed"})

    auth = authenticate(ctx.config, headers, ctx.secrets)
    if not auth:
        return 401, {**json_headers, "WWW-Authenticate": WWW_AUTHENTICATE}, \
            json.dumps({"error": auth.reason})

    try:
        payload: Any = json.loads(body or "")
    except (ValueError, TypeError):
        return 400, json_headers, json.dumps(
            _error(None, PARSE_ERROR, "request body was not valid JSON"))

    server = MCPServer(ctx)
    if isinstance(payload, list):
        responses = [r for r in (server.handle(item) for item in payload)
                     if r is not None]
        if not responses:
            return 202, {}, ""
        return 200, json_headers, json.dumps(responses, ensure_ascii=False)

    response = server.handle(payload)
    if response is None:
        return 202, {}, ""
    return 200, json_headers, json.dumps(response, ensure_ascii=False)


def _ok(request_id, result: dict) -> dict:
    return {"jsonrpc": JSONRPC, "id": request_id, "result": result}


def _error(request_id, code: int, message: str) -> dict:
    return {"jsonrpc": JSONRPC, "id": request_id,
            "error": {"code": code, "message": message}}


def _tool_failure(message: str) -> dict:
    """A refused tool call is a result with isError, not a protocol error —
    that is how the model gets to read and act on the reason."""
    return {"content": [{"type": "text", "text": message}], "isError": True}


def _version() -> str:
    from .. import __version__

    return __version__
