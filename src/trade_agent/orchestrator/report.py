"""Owner-facing cycle report, assembled in Python.

This used to be an LLM's job (the commander agent). It is not, any more, and
the reason is worth stating: every ingredient is already a stored, structured
field by the time the report is written. The regime read, the proposal count,
the judge's rationale, the risk assessment, the exact prices and sizes — all of
it exists. Asking a model to restate them buys nothing and adds a way for the
report to disagree with what actually happened.

So Python does the arithmetic and the layout, and the agents' own sentences are
quoted verbatim. Every line traces to a field in `agent_calls` or `trades`,
which is what spec 14 asks for: the owner can audit the reasoning, not just
read a summary of it.
"""

from __future__ import annotations

from decimal import Decimal

from ..models.agent_io import AnalystOutput, RiskOutput
from ..models.state import SystemState
from ..models.trading import ExecutionPlan
from ..money import ZERO, dec, jpy


def compose_traded(*, analyst: AnalystOutput, plan: ExecutionPlan,
                   risk: RiskOutput | None, state: SystemState,
                   buy_count: int, proposal_count: int,
                   consensus: Decimal | None, protection_note: str = "") -> tuple[str, str]:
    """(headline, body) for a cycle that placed an order."""
    kind = "偵察発注(3日ルール)" if plan.probe else "発注"
    headline = (f"{kind}: {_btc(plan.qty_btc)} BTC @ {_yen(plan.entry)} 円 "
                f"({analyst.regime})")

    parts = [
        f"【{kind}】{_btc(plan.qty_btc)} BTC @ {_yen(plan.entry)} 円",
        "",
        _market_read(analyst),
        "",
        _consensus_line(buy_count, proposal_count, consensus),
        f"採択理由: {plan.thesis}",
        "",
        "執行計画",
        _plan_table(plan, state),
    ]
    if risk is not None:
        parts += ["", f"リスク査定: {risk.rationale}"]
        if risk.adjustments:
            parts.append("  調整: " + " / ".join(risk.adjustments))
    if analyst.risks:
        parts += ["", "この読みが外れる条件: " + " / ".join(analyst.risks)]
    if protection_note:
        parts += ["", f"保護: {protection_note}"]
    if plan.probe:
        parts += ["",
                  "注記: 退屈防止ルールによる偵察トレードです。統計的な優位性は"
                  "主張しておらず、戦略成績とは別に集計されます。"]

    safety = _safety_line(state)
    if safety:
        parts += ["", safety]
    return headline, "\n".join(parts)


def compose_no_trade(*, reason: str, analyst: AnalystOutput | None,
                     state: SystemState, buy_count: int, proposal_count: int,
                     consensus: Decimal | None) -> tuple[str, str]:
    """(headline, body) for a cycle that did not trade.

    A no-trade is a normal outcome (spec 4.1) and gets the same treatment as a
    trade: why it happened, on what evidence. "Why it did not trade" is as much
    information as "why it did".
    """
    headline = f"見送り: {_shorten(reason)}"
    parts = [f"【見送り】{reason}"]
    if analyst is not None:
        parts += ["", _market_read(analyst)]
        if proposal_count:
            parts += ["", _consensus_line(buy_count, proposal_count, consensus)]
        if analyst.risks:
            parts += ["", "見ていた下振れ材料: " + " / ".join(analyst.risks)]

    safety = _safety_line(state)
    if safety:
        parts += ["", safety]
    return headline, "\n".join(parts)


# -- pieces ---------------------------------------------------------------

def _market_read(analyst: AnalystOutput) -> str:
    indicators = "、".join(analyst.key_indicators) if analyst.key_indicators else "—"
    return (f"地合い: {analyst.regime}(確信度 {analyst.confidence:.2f})\n"
            f"{analyst.summary}\n"
            f"根拠にした指標: {indicators}")


def _consensus_line(buy_count: int, proposal_count: int,
                    consensus: Decimal | None) -> str:
    line = f"戦略案: {proposal_count}案中 {buy_count}案が買い"
    if consensus is not None:
        line += f"、合意度 {float(consensus):.2f}"
    return line


def _plan_table(plan: ExecutionPlan, state: SystemState) -> str:
    """The numbers, with the percentages the owner would otherwise compute."""
    entry = dec(plan.entry)
    stop_pct = (plan.stop_loss - entry) / entry * Decimal(100) if entry else ZERO
    take_pct = (plan.take_profit - entry) / entry * Decimal(100) if entry else ZERO
    reward = plan.take_profit - entry
    risk_distance = entry - plan.stop_loss
    rr = (reward / risk_distance) if risk_distance > 0 else None
    equity_pct = (plan.risk_jpy / state.equity_jpy * Decimal(100)
                  if state.equity_jpy > 0 else ZERO)

    rows = [
        f"  entry     {_yen(entry):>14} 円",
        f"  損切り     {_yen(plan.stop_loss):>14} 円  ({stop_pct:+.2f}%)",
        f"  利確       {_yen(plan.take_profit):>14} 円  ({take_pct:+.2f}%)",
        f"  数量       {_btc(plan.qty_btc):>14} BTC",
        f"  リスク額    {_yen(plan.risk_jpy):>14} 円  "
        f"(equity の {equity_pct:.2f}%)",
    ]
    if rr is not None:
        rows.append(f"  損益比      {float(rr):>14.2f}")
    return "\n".join(rows)


def _safety_line(state: SystemState) -> str:
    """Only shown when something is actually engaged — silence means normal."""
    flags = []
    if state.kill_switch:
        flags.append("キルスイッチ発動中")
    if state.owner_paused:
        flags.append("オーナーによる停止中")
    if state.losing_streak:
        flags.append(f"連敗 {state.losing_streak}")
    if state.monthly.probe_rule_suspended:
        flags.append("probeルール当月停止")
    if not flags:
        return ""
    return "安全装置: " + " / ".join(flags)


def _yen(value) -> str:
    return f"{int(jpy(value)):,}"


def _btc(value) -> str:
    return format(dec(value).normalize(), "f")


def _shorten(text: str, limit: int = 40) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"
