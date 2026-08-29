"""The owner-facing MCP server (spec 16)."""

import json
from decimal import Decimal

import pytest

from trade_agent.mcp.auth import extract_bearer, extract_path_token
from trade_agent.mcp.server import handle_request
from trade_agent.mcp.tools import READ_ONLY, TOOLS, ToolError, call_tool

E = Decimal
AUTH = {"authorization": "Bearer test-token"}


def _rpc(ctx, method: str, params=None, request_id=1, headers=None, path=None):
    body = json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method,
                       "params": params or {}})
    status, _, text = handle_request(ctx, method="POST",
                                     headers=AUTH if headers is None else headers,
                                     body=body, path=path)
    return status, (json.loads(text) if text else None)


# -- transport & auth -----------------------------------------------------

def test_a_request_without_a_token_is_rejected(ctx):
    status, payload = _rpc(ctx, "tools/list", headers={})
    assert status == 401
    assert "missing bearer token" in payload["error"]


def test_a_wrong_token_is_rejected(ctx):
    status, _ = _rpc(ctx, "tools/list", headers={"authorization": "Bearer nope"})
    assert status == 401


def test_the_header_name_is_matched_case_insensitively():
    assert extract_bearer({"Authorization": "Bearer abc"}) == "abc"
    assert extract_bearer({"AUTHORIZATION": "bearer abc"}) == "abc"
    assert extract_bearer({"authorization": "Basic abc"}) is None


# claude.ai's custom-connector dialog offers OAuth or nothing — an individual
# account has nowhere to put a static header — so the token can also ride in
# the path. See mcp/auth.py for why that tradeoff was taken.

def test_a_token_in_the_path_authenticates(ctx):
    status, payload = _rpc(ctx, "tools/list", headers={}, path="/mcp/test-token")
    assert status == 200
    assert payload["result"]["tools"]


def test_a_wrong_token_in_the_path_is_rejected(ctx):
    status, _ = _rpc(ctx, "tools/list", headers={}, path="/mcp/nope")
    assert status == 401


def test_the_path_prefix_is_required(ctx):
    """A bare `/<token>` must not authenticate: without a fixed prefix, every
    request would compare its whole path against the secret."""
    status, _ = _rpc(ctx, "tools/list", headers={}, path="/test-token")
    assert status == 401


def test_a_header_beats_a_bad_path(ctx):
    status, _ = _rpc(ctx, "tools/list", headers=AUTH, path="/mcp/nope")
    assert status == 200


def test_path_token_extraction():
    assert extract_path_token("/mcp/abc") == "abc"
    assert extract_path_token("/mcp/abc/") == "abc"
    assert extract_path_token("/mcp/a%2Bb") == "a+b"   # percent-decoded
    assert extract_path_token("/mcp/") is None
    assert extract_path_token("/") is None
    assert extract_path_token("") is None
    assert extract_path_token(None) is None


# -- the mcp function's cold start ----------------------------------------
#
# All of this runs before authentication, so anything that throws here is a
# 502 for every caller — including one with no credential at all. Keeping the
# public endpoint's start-up free of I/O is what makes that unlikely.

def test_the_mcp_context_is_built_without_an_exchange(monkeypatch):
    """No MCP tool touches the exchange, so building one would add an
    HTTP client and (under paper trading) an S3 read to every cold start of the
    one internet-facing function, in exchange for nothing."""
    from trade_agent.orchestrator import context as ctx_mod

    def fail(*args, **kwargs):
        raise AssertionError("the mcp context must not build an exchange")

    monkeypatch.setattr(ctx_mod, "_build_exchange", fail)
    built = ctx_mod.build_context(
        owner="mcp-lambda", needs_trading_credentials=False,
        needs_exchange=False, config=_offline_config(),
        store=_MemoryStore(), secrets=_NoSecrets(), notifier=object())
    assert built.exchange is not None


def test_touching_a_missing_exchange_says_so(monkeypatch):
    """A bare None would surface as an AttributeError on NoneType from deep in
    a call stack; this states what actually happened."""
    from trade_agent.errors import ConfigError
    from trade_agent.orchestrator.context import _NoExchange

    with pytest.raises(ConfigError, match="without an exchange"):
        _NoExchange().ticker()


def test_every_read_only_tool_works_without_an_exchange(ctx):
    """The invariant behind needs_exchange=False, checked against the tools
    themselves rather than against a comment."""
    from trade_agent.orchestrator.context import _NoExchange

    ctx.exchange = _NoExchange()
    for name in sorted(READ_ONLY):
        call_tool(ctx, name, {})


def test_the_handler_asks_for_a_context_with_no_exchange(monkeypatch):
    """Checked at the call site, because that is what actually ships — the
    option existing is no use if the handler stops passing it."""
    from trade_agent.handlers import mcp_handler

    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here; only the arguments matter")

    monkeypatch.setattr(mcp_handler, "build_context", spy)
    monkeypatch.setattr(mcp_handler, "_CACHED_CONTEXT", None)
    with pytest.raises(RuntimeError):
        mcp_handler._context()

    assert seen.get("needs_exchange") is False
    assert seen.get("needs_trading_credentials") is False


def _offline_config():
    from trade_agent.config import get_config

    config = get_config()
    config.storage.backend = "memory"
    return config


class _MemoryStore:
    def __getattr__(self, name):
        from trade_agent.storage.memory import MemoryStore

        return getattr(MemoryStore(), name)


class _NoSecrets:
    def get(self, name):
        return "test-token"

    def get_optional(self, name):
        return None


def test_get_is_not_supported(ctx):
    status, _, _ = handle_request(ctx, method="GET", headers=AUTH, body=None)
    assert status == 405


def test_initialize_advertises_tools(ctx):
    status, payload = _rpc(ctx, "initialize")
    assert status == 200
    assert payload["result"]["capabilities"]["tools"] is not None
    assert payload["result"]["serverInfo"]["name"] == ctx.config.mcp.server_name


def test_a_notification_gets_no_body(ctx):
    body = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
    status, _, text = handle_request(ctx, method="POST", headers=AUTH, body=body)
    assert status == 202
    assert text == ""


def test_unknown_methods_are_a_protocol_error(ctx):
    _, payload = _rpc(ctx, "tools/destroy_everything")
    assert payload["error"]["code"] == -32601


def test_malformed_json_is_a_parse_error(ctx):
    status, _, text = handle_request(ctx, method="POST", headers=AUTH,
                                     body="{not json")
    assert status == 400
    assert json.loads(text)["error"]["code"] == -32700


# -- tool surface ---------------------------------------------------------

def test_exactly_the_specified_tools_are_exposed(ctx):
    """Spec 16 names seven; `get_cycles` is an eighth, added deliberately.

    The reason a cycle declined to trade was reaching the owner through one
    daily report — whichever cycle happened to cross 21:00 JST — while every
    other cycle's reason expired with its CloudWatch line. "Why is it not
    trading" could not be answered from the connector at all.

    The list stays pinned so a tool cannot appear by accident; adding one is a
    change to the documented surface and should read as such in the diff.
    """
    _, payload = _rpc(ctx, "tools/list")
    names = {t["name"] for t in payload["result"]["tools"]}
    assert names == {
        "get_status", "get_daily_report", "get_trades", "get_agent_log",
        "get_lessons", "get_cycles", "pause_trading", "resume_trading"}


def test_get_cycles_reports_why_each_cycle_ended(ctx, llm, clock):
    from trade_agent.models.state import CycleTrigger
    from trade_agent.orchestrator.cycle import DecisionCycle

    for n in range(3):
        DecisionCycle(ctx, trigger=CycleTrigger.MANUAL,
                      cycle_id=f"cyc-list-{n}").run()

    result = call_tool(ctx, "get_cycles", {})
    assert result["count"] >= 3
    assert len(result["cycles"]) >= 3

    first = result["cycles"][0]
    assert first["outcome"] in {"traded", "no_trade"}
    assert first["detail"]
    assert first["trigger"] == "manual"
    # The tally is the point: which of the eight reasons is actually firing.
    assert isinstance(result["no_trade_reasons"], dict)
    if result["traded"] < result["count"]:
        assert sum(result["no_trade_reasons"].values()) >= 1


def test_get_cycles_can_filter_to_the_ones_that_did_not_trade(ctx, llm, clock):
    from trade_agent.models.state import CycleTrigger
    from trade_agent.orchestrator.cycle import DecisionCycle

    DecisionCycle(ctx, trigger=CycleTrigger.MANUAL, cycle_id="cyc-f-1").run()
    result = call_tool(ctx, "get_cycles", {"outcome": "no_trade"})
    assert all(c["outcome"] == "no_trade" for c in result["cycles"])
    assert result["traded"] == 0


def test_get_cycles_is_read_only(ctx):
    """It answers a diagnostic question; it must not need confirm."""
    assert "get_cycles" in READ_ONLY
    call_tool(ctx, "get_cycles", {})   # no confirm=true, must not raise


def test_no_tool_can_place_or_cancel_an_order(ctx):
    """Spec 16.3: the most this channel can do is stop and start."""
    text = json.dumps(TOOLS)
    for forbidden in ("place_order", "create_order", "cancel", "withdraw",
                      "api_key", "secret"):
        assert forbidden not in text


def test_get_status_reports_the_safety_state(ctx):
    _, payload = _rpc(ctx, "tools/call", {"name": "get_status", "arguments": {}})
    result = payload["result"]["structuredContent"]
    assert result["safety"]["kill_switch"] is False
    assert result["cost"]["llm_budget_jpy"] == 2900.0
    assert result["boredom_rule"]["threshold_hours"] == 72
    assert result["paper_trading"] is True


# -- operational tools ----------------------------------------------------

def test_pause_requires_confirmation(ctx):
    _, payload = _rpc(ctx, "tools/call",
                      {"name": "pause_trading", "arguments": {}})
    assert payload["result"]["isError"] is True
    assert "confirm=true" in payload["result"]["content"][0]["text"]
    assert ctx.load_state().owner_paused is False


def test_pause_with_confirmation_halts_new_entries(ctx):
    _, payload = _rpc(ctx, "tools/call",
                      {"name": "pause_trading",
                       "arguments": {"confirm": True, "reason": "旅行中"}})
    assert payload["result"]["structuredContent"]["owner_paused"] is True
    assert ctx.load_state().owner_paused is True


def test_resume_requires_confirmation(ctx):
    state = ctx.load_state()
    state.kill_switch = True
    ctx.save_state(state)
    _, payload = _rpc(ctx, "tools/call",
                      {"name": "resume_trading", "arguments": {}})
    assert payload["result"]["isError"] is True
    assert ctx.load_state().kill_switch is True


def test_resume_clears_the_kill_switch(ctx):
    state = ctx.load_state()
    state.kill_switch = True
    state.kill_switch_reason = "drawdown"
    ctx.save_state(state)

    _, payload = _rpc(ctx, "tools/call",
                      {"name": "resume_trading", "arguments": {"confirm": True}})
    result = payload["result"]["structuredContent"]
    assert result["kill_switch_was_engaged"] is True
    assert ctx.load_state().kill_switch is False


def test_resume_does_not_clear_a_market_driven_brake(ctx, clock):
    from datetime import timedelta

    state = ctx.load_state()
    state.kill_switch = True
    state.losing_streak = 3
    state.losing_streak_until = clock.now() + timedelta(hours=12)
    ctx.save_state(state)

    _rpc(ctx, "tools/call",
         {"name": "resume_trading", "arguments": {"confirm": True}})
    after = ctx.load_state()
    assert after.kill_switch is False
    assert after.losing_streak_until is not None  # time-based, not owner-cleared


def test_operational_calls_are_audited(ctx):
    _rpc(ctx, "tools/call",
         {"name": "pause_trading", "arguments": {"confirm": True, "reason": "点検"}})
    events = ctx.store.audit.list_recent()
    assert any(e.action == "pause_trading" and "点検" in e.detail for e in events)


def test_read_only_tools_need_no_confirmation(ctx):
    for name in READ_ONLY:
        result = call_tool(ctx, name, {})
        assert isinstance(result, dict)


def test_an_unknown_tool_is_refused(ctx):
    with pytest.raises(ToolError):
        call_tool(ctx, "sell_everything", {"confirm": True})


def test_the_agent_log_proves_the_proposals_were_blind(ctx, llm):
    from trade_agent.models.state import CycleTrigger
    from trade_agent.orchestrator.cycle import DecisionCycle

    DecisionCycle(ctx, trigger=CycleTrigger.MANUAL, cycle_id="cyc-log").run()
    result = call_tool(ctx, "get_agent_log", {"cycle_id": "cyc-log"})
    assert result["found"] is True
    assert result["independent_proposals_verified"] is True
    assert result["total_cost_jpy"] > 0


def test_trades_separate_probe_from_strategy(ctx, clock):
    from trade_agent.models.trading import TradeRecord

    for index, probe in enumerate([False, True]):
        ctx.store.trades.put(TradeRecord(
            trade_id=f"t{index}", cycle_id="c", pair="btc_jpy", probe=probe,
            qty_btc=E("0.0001"), entry_price=E(15000000), entry_order_id="o",
            entry_at=clock.now(), stop_loss=E(14900000), take_profit=E(15200000),
            exit_price=E(15100000), exit_at=clock.now(), net_pnl_jpy=E(10),
            closed=True))

    result = call_tool(ctx, "get_trades", {"days": 1})
    assert result["summary"]["strategy_trades"] == 1
    assert result["summary"]["probe_trades"] == 1


def test_owner_facing_timestamps_are_actually_jst(ctx):
    """A field named `*_jst` must not quietly contain UTC."""
    _, payload = _rpc(ctx, "tools/call", {"name": "get_status", "arguments": {}})
    as_of = payload["result"]["structuredContent"]["as_of_jst"]
    assert as_of.endswith("+09:00"), as_of
    assert not as_of.endswith("Z")


def test_get_agent_log_defaults_to_the_most_recent_cycle(ctx):
    """Cycle ids are minted inside a Lambda and appear in no report, so
    without this the owner had an id-only tool and no source of ids."""
    from trade_agent.storage.base import AuditEvent

    now = ctx.clock.now()
    ctx.store.audit.put(AuditEvent(
        event_id="cycle:cyc-20260829T090000Z-scheduled", at=now,
        actor="cycle:scheduled", action="no_trade",
        detail="consensus not reached: 0/3 buy proposals, 1 required"))

    assert call_tool(ctx, "get_agent_log", {})["cycle_id"] == \
        "cyc-20260829T090000Z-scheduled"


def test_get_agent_log_says_so_when_nothing_has_run(ctx):
    """The old message asked for an argument the caller could not supply,
    which read as user error rather than as an empty system."""
    result = call_tool(ctx, "get_agent_log", {})
    assert result["found"] is False
    assert "議論が1回も" in result["message"]


def test_the_latest_cycle_ignores_unrelated_audit_events(ctx):
    """The audit table carries kill-switch flips and owner actions too; only
    rows a cycle wrote name a cycle."""
    from datetime import timedelta

    from trade_agent.storage.base import AuditEvent

    now = ctx.clock.now()
    ctx.store.audit.put(AuditEvent(
        event_id="cycle:cyc-20260829T090000Z-scheduled", at=now,
        actor="cycle:scheduled", action="no_trade", detail="0/3"))
    # Strictly newer, so the filter is what selects the cycle rather than the
    # sort happening to be stable on equal timestamps.
    ctx.store.audit.put(AuditEvent(
        event_id="pause:1", at=now + timedelta(minutes=5), actor="owner",
        action="pause_trading", detail="owner asked"))

    from trade_agent.mcp.tools import _latest_cycle_id
    assert _latest_cycle_id(ctx) == "cyc-20260829T090000Z-scheduled"
