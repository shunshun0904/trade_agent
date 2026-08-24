"""MCP tool surface (spec 16.2).

Seven tools: five read-only, two operational. What is deliberately absent is as
important as what is here — there is no place-order tool, no cancel tool, no
credential accessor. The most an owner (or a confused model acting on their
behalf) can do through this channel is stop the system and start it again
(spec 16.3).

Both operational tools require `confirm=true`. A model that decides on its own
to call `resume_trading` while the kill switch is engaged should hit a wall,
and an explicit confirmation flag is that wall.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from decimal import Decimal

from ..money import jpy
from ..orchestrator.context import AppContext
from ..storage.base import AuditEvent
from ..timeutil import iso_jst, jst_date_str

TOOLS = [
    {
        "name": "get_status",
        "description": (
            "現在の稼働状況を返す。equity、建玉、キルスイッチと連敗ブレーキの状態、"
            "最終約定からの経過時間、当月のLLM API費。"),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_daily_report",
        "description": "指定日の日次レポート(A6生成)を返す。日付を省略すると最新。",
        "inputSchema": {
            "type": "object",
            "properties": {"date": {"type": "string",
                                    "description": "JSTの日付 YYYY-MM-DD"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_trades",
        "description": "約定一覧を返す。probeトレード(退屈防止ルール)は区別して表示する。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "遡る日数(既定7)"},
                "limit": {"type": "integer", "description": "最大件数(既定20)"},
                "include_probe": {"type": "boolean", "description": "既定true"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_agent_log",
        "description": (
            "指定トレード(またはサイクル)の判断過程を返す。3案・相互批判・裁定・"
            "検査結果と、各エージェント呼び出しのトークン数とコスト。"),
        "inputSchema": {
            "type": "object",
            "properties": {
                "trade_id": {"type": "string"},
                "cycle_id": {"type": "string"},
                "include_bodies": {"type": "boolean",
                                   "description": "入出力JSON本体も返す(既定false)"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_lessons",
        "description": "教訓データベースを検索する。レジームタグで絞り込める。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "regime": {"type": "string",
                           "enum": ["trend_up", "trend_down", "range", "volatile", "all"]},
                "limit": {"type": "integer"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "pause_trading",
        "description": (
            "新規建てを停止する。建玉の損切り・利確監視は継続する。"
            "confirm=true が必須。"),
        "inputSchema": {
            "type": "object",
            "properties": {
                "confirm": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["confirm"],
            "additionalProperties": False,
        },
    },
    {
        "name": "resume_trading",
        "description": (
            "キルスイッチおよび pause を解除して取引を再開する。"
            "confirm=true が必須。オーナーの明示的な指示なしに呼び出してはならない。"),
        "inputSchema": {
            "type": "object",
            "properties": {
                "confirm": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["confirm"],
            "additionalProperties": False,
        },
    },
]

READ_ONLY = {"get_status", "get_daily_report", "get_trades", "get_agent_log",
             "get_lessons"}


class ToolError(Exception):
    pass


def call_tool(ctx: AppContext, name: str, arguments: dict) -> dict:
    handler = _HANDLERS.get(name)
    if handler is None:
        raise ToolError(f"unknown tool: {name}")
    if name not in READ_ONLY and not arguments.get("confirm"):
        # Spec 16.2: an operational call without confirm is refused outright.
        raise ToolError(
            f"{name} は操作系ツールのため confirm=true が必須です。"
            "オーナーの明示的な指示を確認してから再実行してください。")
    return handler(ctx, arguments or {})


# -- read-only ------------------------------------------------------------

def _get_status(ctx: AppContext, args: dict) -> dict:
    now = ctx.clock.now()
    state = ctx.load_state()
    budget = ctx.budget_state(state)
    position = state.open_position
    idle = state.hours_since_last_entry(now)

    return {
        "as_of_jst": iso_jst(now),
        "phase": ctx.config.system.phase,
        "paper_trading": ctx.config.system.paper_trading,
        "equity_jpy": float(jpy(state.equity_jpy)),
        "initial_equity_jpy": float(ctx.config.capital.initial_equity_jpy),
        "return_pct": float(_return_pct(ctx, state)),
        "position": None if position is None else {
            "trade_id": position.trade_id,
            "qty_btc": float(position.qty_btc),
            "entry_price": float(position.entry_price),
            "stop_loss": float(position.stop_loss),
            "take_profit": float(position.take_profit),
            "probe": position.probe,
            "opened_at_jst": iso_jst(position.opened_at),
        },
        "safety": {
            "kill_switch": state.kill_switch,
            "kill_switch_reason": state.kill_switch_reason,
            "owner_paused": state.owner_paused,
            "losing_streak": state.losing_streak,
            "losing_streak_until_jst": _jst_or_none(state.losing_streak_until),
            "flash_pause_until_jst": _jst_or_none(state.flash_pause_until),
            "daily_loss_pct": float(state.daily_loss_pct()),
            "daily_loss_limit_pct": float(ctx.config.risk.daily_loss_limit_pct),
        },
        "boredom_rule": {
            "hours_since_last_entry": round(idle, 1) if idle is not None else None,
            "threshold_hours": ctx.config.boredom.no_trade_hours,
            "probe_rule_suspended": state.monthly.probe_rule_suspended,
            "probe_pnl_month_jpy": float(jpy(state.monthly.probe_pnl_jpy)),
        },
        "cost": {
            "llm_spent_month_jpy": float(round(state.monthly.llm_cost_jpy, 1)),
            "llm_budget_jpy": float(ctx.config.cost.llm_budget_jpy),
            "used_pct": float(round(budget.used_pct, 1)),
            "ladder": str(budget.ladder),
            "infra_cost_jpy": float(ctx.config.cost.infra_cost_jpy),
        },
        "activity": {
            "full_debates_today": state.daily.full_debates,
            "daily_debate_limit": ctx.config.schedule.daily_full_debate_limit,
            "last_tick_jst": _jst_or_none(state.last_tick_at),
            "last_full_debate_jst": _jst_or_none(state.last_full_debate_at),
        },
    }


def _get_daily_report(ctx: AppContext, args: dict) -> dict:
    report = ctx.store.reports.get(args.get("date"))
    if report is None:
        return {"found": False,
                "message": "該当日のレポートがありません。"}
    return {
        "found": True,
        "date": report.jst_date,
        "headline": report.headline,
        "report": report.report_text,
        "equity_jpy": float(jpy(report.equity_jpy)),
        "realized_pnl_jpy": float(jpy(report.realized_pnl_jpy)),
        "llm_cost_month_jpy": float(round(report.llm_cost_month_jpy, 1)),
        "consensus_rate": report.consensus_rate,
        "hours_since_last_entry": report.hours_since_last_entry,
    }


def _get_trades(ctx: AppContext, args: dict) -> dict:
    days = int(args.get("days", 7))
    limit = int(args.get("limit", 20))
    include_probe = bool(args.get("include_probe", True))
    now = ctx.clock.now()
    rows = ctx.store.trades.list_between(now - timedelta(days=days), now)
    if not include_probe:
        rows = [t for t in rows if not t.probe]
    rows = sorted(rows, key=lambda t: t.entry_at, reverse=True)[:limit]

    closed = [t for t in rows if t.closed and t.net_pnl_jpy is not None]
    real = [t for t in closed if not t.probe]
    probes = [t for t in closed if t.probe]
    return {
        "window_days": days,
        "count": len(rows),
        "trades": [{
            "trade_id": t.trade_id,
            "probe": t.probe,
            "entry_jst": iso_jst(t.entry_at),
            "exit_jst": _jst_or_none(t.exit_at),
            "qty_btc": float(t.qty_btc),
            "entry_price": float(t.entry_price),
            "exit_price": float(t.exit_price) if t.exit_price else None,
            "stop_loss": float(t.stop_loss),
            "take_profit": float(t.take_profit),
            "exit_reason": t.exit_reason,
            "net_pnl_jpy": float(jpy(t.net_pnl_jpy)) if t.net_pnl_jpy is not None else None,
            "fee_jpy": float(jpy(t.fee_jpy)),
            "regime": t.regime,
            "cycle_id": t.cycle_id,
            "closed": t.closed,
        } for t in rows],
        "summary": {
            "strategy_trades": len(real),
            "strategy_net_pnl_jpy": float(jpy(sum((t.net_pnl_jpy for t in real),
                                                  Decimal(0)))),
            "probe_trades": len(probes),
            "probe_net_pnl_jpy": float(jpy(sum((t.net_pnl_jpy for t in probes),
                                               Decimal(0)))),
        },
    }


def _get_agent_log(ctx: AppContext, args: dict) -> dict:
    cycle_id = args.get("cycle_id")
    if not cycle_id and args.get("trade_id"):
        trade = ctx.store.trades.get(args["trade_id"])
        if trade is None:
            return {"found": False, "message": "該当トレードがありません。"}
        cycle_id = trade.cycle_id
    if not cycle_id:
        return {"found": False, "message": "trade_id か cycle_id を指定してください。"}

    calls = ctx.store.agent_calls.list_for_cycle(cycle_id)
    if not calls:
        return {"found": False, "cycle_id": cycle_id,
                "message": "このサイクルのエージェントログがありません。"}

    include_bodies = bool(args.get("include_bodies"))
    entries = []
    for call in calls:
        entry = {
            "agent": call.agent,
            "sequence": call.sequence,
            "at_jst": iso_jst(call.called_at),
            "model": call.model,
            "ok": call.ok,
            "retries": call.retries,
            "error": call.error,
            "tokens": {"input": call.input_tokens, "output": call.output_tokens,
                       "cache_read": call.cache_read_tokens,
                       "cache_write": call.cache_write_tokens},
            "cost_jpy": float(round(call.cost_jpy, 3)),
            # Spec 4.1 is auditable from here: a phase-1 strategist must show
            # only ["analyst"].
            "saw_agents": call.saw_agents,
        }
        if include_bodies and call.io_s3_key:
            entry["body"] = ctx.store.blobs.get_json(call.io_s3_key)
        entries.append(entry)

    return {
        "found": True,
        "cycle_id": cycle_id,
        "calls": entries,
        "total_cost_jpy": float(round(sum(c.cost_jpy for c in calls), 3)),
        "independent_proposals_verified": _proposals_were_blind(calls),
    }


def _get_lessons(ctx: AppContext, args: dict) -> dict:
    regime = args.get("regime")
    rows = ctx.store.lessons.list(
        regime=None if regime in (None, "all") else regime,
        limit=int(args.get("limit", 20)))
    return {
        "count": len(rows),
        "lessons": [{
            "lesson_id": row.lesson_id,
            "created_jst": iso_jst(row.created_at),
            "regime": row.regime_tag,
            "text": row.text,
            "evidence": row.evidence,
            "confidence": row.confidence,
            "trades_analysed": row.trades_analysed,
        } for row in rows],
    }


# -- operational ----------------------------------------------------------

def _pause_trading(ctx: AppContext, args: dict) -> dict:
    state = ctx.load_state()
    reason = args.get("reason") or "owner request via MCP"
    state.owner_paused = True
    state.owner_pause_reason = reason
    ctx.save_state(state)
    _audit(ctx, "pause_trading", reason)
    return {
        "ok": True,
        "owner_paused": True,
        "message": ("新規建てを停止しました。建玉の損切り・利確監視と5分tickは"
                    "継続します。"),
        "reason": reason,
    }


def _resume_trading(ctx: AppContext, args: dict) -> dict:
    state = ctx.load_state()
    reason = args.get("reason") or "owner request via MCP"
    was_killed = state.kill_switch
    state.kill_switch = False
    state.kill_switch_reason = None
    state.kill_switch_at = None
    state.owner_paused = False
    state.owner_pause_reason = None
    # A resume clears the operator-facing halts only. The losing-streak and
    # flash-move timers are market conditions, not owner decisions, and they
    # run out on their own.
    ctx.save_state(state)
    _audit(ctx, "resume_trading",
           f"{reason} (kill_switch_was={was_killed})")
    return {
        "ok": True,
        "kill_switch": False,
        "owner_paused": False,
        "kill_switch_was_engaged": was_killed,
        "message": ("取引を再開しました。連敗ブレーキと急変動停止は時間経過で"
                    "解除されるため、この操作では解除されません。"),
        "losing_streak": state.losing_streak,
        "losing_streak_until_jst": _jst_or_none(state.losing_streak_until),
    }


# -- helpers --------------------------------------------------------------

def _audit(ctx: AppContext, action: str, detail: str) -> None:
    ctx.store.audit.put(AuditEvent(
        event_id=uuid.uuid4().hex[:12], at=ctx.clock.now(), actor="mcp",
        action=action, detail=detail))


def _return_pct(ctx: AppContext, state) -> Decimal:
    initial = ctx.config.capital.initial_equity_jpy
    if initial <= 0:
        return Decimal(0)
    return (state.equity_jpy - initial) / initial * Decimal(100)


def _jst_or_none(value) -> str | None:
    return iso_jst(value) if value else None


def _proposals_were_blind(calls) -> bool:
    """Spec 14: the three phase-1 proposals must not have seen each other."""
    phase_one = [c for c in calls if c.agent.startswith("strategy:")]
    if not phase_one:
        return False
    return all(set(c.saw_agents) <= {"analyst"} for c in phase_one)


_HANDLERS = {
    "get_status": _get_status,
    "get_daily_report": _get_daily_report,
    "get_trades": _get_trades,
    "get_agent_log": _get_agent_log,
    "get_lessons": _get_lessons,
    "pause_trading": _pause_trading,
    "resume_trading": _resume_trading,
}
