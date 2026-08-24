"""The deterministic guard (spec 5).

Everything an agent says that can be checked mechanically is checked here,
before any of it can influence an order. The guard uses no LLM and costs
nothing to run, which is the point: the expensive, fallible component is
wrapped in a cheap, infallible one.

A rejection carries the specific violations back to the agent, up to three
attempts (spec 5); after that the cycle is abandoned. "Skip this cycle" is
always available and always safe.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from ..config import Config
from ..errors import GuardRejection
from ..models.agent_io import (
    AnalystOutput,
    AuditorOutput,
    JudgeOutput,
    RiskOutput,
    StrategyOutput,
)
from ..models.market import MarketSnapshot
from ..money import ZERO, dec, deviation_pct, is_lot_multiple, jpy
from ..risk.rules import RiskEngine

# Indicator aliases as they appear in Japanese prose. The guard only compares a
# number when the agent states it as the current value ("RSIは58.1"), not when
# it is used as a threshold ("RSIが70を超えたら") — the latter is a plan, not a
# claim about the market.
INDICATOR_ALIASES: dict[str, str] = {
    "rsi": "rsi",
    "atr": "atr",
    "vwap": "vwap_24h",
    "現在価格": "last_price",
    "最終価格": "last_price",
    "終値": "last_price",
    "仲値": "mid_price",
    "最良買気配": "best_bid",
    "最良売気配": "best_ask",
    "24時間高値": "high_24h",
    "24時間安値": "low_24h",
    "sma": "sma_short",
    "ema": "ema_short",
    "equity": "equity_jpy",
    "総資産": "equity_jpy",
}

# name, optional spacing/particle, an assignment-ish marker, then the number.
QUOTE_PATTERN = re.compile(
    r"(?P<name>" + "|".join(sorted(map(re.escape, INDICATOR_ALIASES), key=len,
                                   reverse=True)) + r")"
    r"\s*(?:\((?:\d+)\))?\s*(?:は|＝|=|:|：)\s*"
    r"(?P<value>-?[0-9][0-9,]*(?:\.[0-9]+)?)",
    re.IGNORECASE)


def extract_quoted_numbers(text: str) -> list[tuple[str, Decimal]]:
    """(indicator field name, quoted value) for every stated current value."""
    out: list[tuple[str, Decimal]] = []
    for match in QUOTE_PATTERN.finditer(text or ""):
        field = INDICATOR_ALIASES[match.group("name").lower()]
        try:
            out.append((field, dec(match.group("value").replace(",", ""))))
        except (InvalidOperation, ValueError):
            continue
    return out


def check_quoted_indicators(text: str, snapshot: MarketSnapshot,
                            tolerance_pct: Decimal) -> list[str]:
    """Spec 5: a quoted indicator value must match the snapshot.

    A model that writes a plausible-looking number it was never given is
    fabricating, and a fabricated number is how a bad trade gets a convincing
    rationale. Values the snapshot does not carry are ignored — absence is not
    evidence of invention.
    """
    actual = snapshot.indicator_values()
    violations: list[str] = []
    for field, quoted in extract_quoted_numbers(text):
        truth = actual.get(field)
        if truth is None:
            continue
        if deviation_pct(quoted, truth) > tolerance_pct:
            violations.append(
                f"引用値が実値と一致しない: {field} は {truth} だが {quoted} と記載された")
    return violations


class DeterministicGuard:
    def __init__(self, config: Config, snapshot: MarketSnapshot):
        self.config = config
        self.snapshot = snapshot
        self.risk = RiskEngine(config)

    # -- per-agent validators ---------------------------------------------

    def validate_analyst(self, output: AnalystOutput) -> None:
        known = set(self.snapshot.indicator_values())
        unknown = [name for name in output.key_indicators if name not in known]
        violations = []
        if unknown:
            violations.append(
                "MarketSnapshotに存在しない指標名を挙げている: "
                + ", ".join(unknown)
                + "。使用可能な指標名: " + ", ".join(sorted(known)))
        violations += self._quotes(output.summary, *output.risks)
        _raise(violations, "analyst")

    def validate_strategy(self, output: StrategyOutput) -> None:
        violations: list[str] = []
        prices = (output.entry, output.take_profit, output.stop_loss)

        if output.action == "wait":
            if any(p is not None for p in prices):
                violations.append(
                    "action=wait のとき entry/take_profit/stop_loss は null にすること")
        else:
            if any(p is None for p in prices):
                violations.append(
                    "action=buy のとき entry/take_profit/stop_loss は必須")
            else:
                violations += self._price_geometry(
                    dec(output.entry), dec(output.stop_loss), dec(output.take_profit))
        violations += self._quotes(output.thesis, output.invalidation)
        _raise(violations, "strategy")

    def validate_judge(self, output: JudgeOutput, *, proposal_ids: list[str],
                       buy_count: int, consensus_min: int) -> None:
        violations: list[str] = []
        if output.decision == "adopt":
            if buy_count < consensus_min:
                violations.append(
                    f"buy案は{buy_count}件で、必要な{consensus_min}件に満たない。"
                    "no_trade を選ぶこと")
            if output.adopted_proposal_id not in proposal_ids:
                violations.append(
                    f"adopted_proposal_id が提案一覧にない: "
                    f"{output.adopted_proposal_id}. 候補: {', '.join(proposal_ids)}")
            if any(p is None for p in (output.entry, output.take_profit,
                                       output.stop_loss)):
                violations.append("adopt のとき entry/take_profit/stop_loss は必須")
            else:
                violations += self._price_geometry(
                    dec(output.entry), dec(output.stop_loss), dec(output.take_profit))
        else:
            if any(p is not None for p in (output.entry, output.take_profit,
                                           output.stop_loss)):
                violations.append("no_trade のとき価格フィールドは null にすること")
        violations += self._quotes(output.rationale)
        _raise(violations, "judge")

    def validate_risk(self, output: RiskOutput, *, expected_qty: Decimal,
                      expected_risk_jpy: Decimal, entry: Decimal,
                      stop_loss: Decimal) -> None:
        """The risk agent's numbers are advisory; Python's are authoritative.

        A mismatch is not corrected silently — it means the agent is reasoning
        about a different trade than the one that would be placed.
        """
        violations: list[str] = []
        if output.approved:
            if output.qty_btc is None:
                violations.append("approved=true のとき qty_btc は必須")
            elif dec(output.qty_btc) != expected_qty:
                violations.append(
                    f"qty_btc が資金管理ルールの算出値と異なる: "
                    f"算出値 {expected_qty} / 記載 {dec(output.qty_btc)}")
            if output.risk_jpy is None:
                violations.append("approved=true のとき risk_jpy は必須")
            elif jpy(output.risk_jpy) != jpy(expected_risk_jpy):
                violations.append(
                    f"risk_jpy が再計算値と一致しない: "
                    f"再計算 {jpy(expected_risk_jpy)} / 記載 {jpy(output.risk_jpy)}")
            if output.stop_loss is not None and dec(output.stop_loss) != dec(stop_loss):
                violations.append(
                    f"stop_loss が採択案と異なる: {dec(stop_loss)} / "
                    f"{dec(output.stop_loss)}。変更するなら adjustments に書くこと")
            if output.take_profit is not None:
                violations += self._price_geometry(
                    dec(entry), dec(stop_loss), dec(output.take_profit))
        elif not output.adjustments:
            violations.append("approved=false のとき adjustments を1つ以上書くこと")
        violations += self._quotes(output.rationale, *output.adjustments)
        _raise(violations, "risk")

    def validate_auditor(self, output: AuditorOutput) -> None:
        violations: list[str] = []
        if not output.ok and not output.violations:
            violations.append("ok=false のとき violations を1つ以上書くこと")
        if output.ok and output.violations:
            violations.append("ok=true のとき violations は空配列にすること")
        if output.ok and output.retry_target != "none":
            violations.append("ok=true のとき retry_target は none にすること")
        violations += self._quotes(output.notes, *output.violations)
        _raise(violations, "auditor")

    def validate_commander(self, output) -> None:
        _raise(self._quotes(output.headline, output.report_text), "commander")

    # -- shared checks -----------------------------------------------------

    def check_executable(self, *, entry: Decimal, stop_loss: Decimal,
                         take_profit: Decimal, qty_btc: Decimal,
                         jpy_available: Decimal, probe: bool = False) -> list[str]:
        """Final structural check before an order is built.

        This is the last place the spec-5 arithmetic runs, and it runs on the
        exact numbers the executor will use — not on the agent's version of
        them.
        """
        cfg = self.config
        violations = self._price_geometry(entry, stop_loss, take_profit)
        min_lot = cfg.exchange.min_order_btc

        if not is_lot_multiple(qty_btc, min_lot):
            violations.append(
                f"数量 {qty_btc} が最小注文数量 {min_lot} の整数倍でない")
        if qty_btc < min_lot:
            violations.append(f"数量 {qty_btc} が最小注文数量 {min_lot} 未満")

        cost = entry * qty_btc
        if cost > dec(jpy_available):
            violations.append(
                f"必要額 {jpy(cost)} JPY が利用可能残高 {jpy(jpy_available)} JPY を超える")

        risk = jpy((entry - stop_loss) * qty_btc)
        limit = self.risk.risk_limit_jpy(self.snapshot.account.equity_jpy, probe=probe)
        if risk > limit:
            violations.append(
                f"リスク額 {risk} JPY が1トレード上限 {limit} JPY を超える")
        return violations

    def _price_geometry(self, entry: Decimal, stop_loss: Decimal,
                        take_profit: Decimal) -> list[str]:
        violations: list[str] = []
        cfg = self.config
        if not (stop_loss < entry < take_profit):
            violations.append(
                f"ロングの価格関係が不正: stop_loss({stop_loss}) < entry({entry}) "
                f"< take_profit({take_profit}) を満たしていない")
            return violations

        reference = self.snapshot.last_price
        deviation = deviation_pct(entry, reference)
        if deviation > cfg.guard.entry_max_deviation_pct:
            violations.append(
                f"entry {entry} は現在価格 {reference} から {deviation:.2f}% 乖離しており、"
                f"上限 {cfg.guard.entry_max_deviation_pct}% を超える")

        # A target inside the round-trip fee is a loss even when it is hit.
        fee_pct = self.snapshot.constraints.round_trip_fee_pct
        gain_pct = (take_profit - entry) / entry * Decimal(100)
        if gain_pct <= fee_pct:
            violations.append(
                f"利確幅 {gain_pct:.3f}% が往復手数料 {fee_pct:.3f}% を上回っていない")
        return violations

    def _quotes(self, *texts: str) -> list[str]:
        violations: list[str] = []
        for text in texts:
            violations += check_quoted_indicators(
                text or "", self.snapshot, self.config.guard.indicator_tolerance_pct)
        return violations


def _raise(violations: list[str], target: str) -> None:
    if violations:
        raise GuardRejection(f"{target} rejected by the deterministic guard",
                             violations=violations, retry_target=target)
