"""Deterministic offline LLM.

Not a mock that returns canned strings — it derives its answers from the
snapshot with simple rules, so the orchestrator, the guard, the risk layer and
the executor can all be exercised end to end in CI and in `--dry-run` without
an API key or a yen of spend.

It is intentionally *not* a trading strategy. Its job is to produce
schema-valid, internally consistent output that the guard should accept, plus,
on demand, output the guard should reject.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal

from ..models.agent_io import (
    AnalystOutput,
    Critique,
    CritiqueOutput,
    JudgeOutput,
    Lesson,
    ReflectOutput,
    RiskOutput,
    ScoutOutput,
    StrategyOutput,
)
from ..money import dec
from ..roles import STRATEGISTS
from .base import LLMRequest, LLMResponse, TokenUsage


class StubLLMClient:
    """`bias` steers the strategists: "buy", "wait" or "mixed" (default)."""

    def __init__(self, *, bias: str = "mixed", cost_meter=None,
                 approve_risk: bool = True):
        self.bias = bias
        self.cost_meter = cost_meter
        self.approve_risk = approve_risk
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        parsed = self._answer(request)
        usage = TokenUsage(input_tokens=1200, output_tokens=180,
                           cache_read_tokens=3000 if request.cacheable else 0)
        cost = (self.cost_meter.cost_jpy(usage, model=request.model)
                if self.cost_meter else Decimal(0))
        return LLMResponse(parsed=parsed, usage=usage, model=request.model,
                           duration_ms=5, raw_text=parsed.model_dump_json(),
                           cost_jpy=cost)

    # -- answers ----------------------------------------------------------

    def _answer(self, request: LLMRequest):
        facts = _facts(request.shared_prefix)
        facts.update(_task_facts(request.task))
        kind = request.agent.split(":")[0]
        handler = getattr(self, f"_{kind}", None)
        if handler is None:
            raise NotImplementedError(f"stub has no answer for agent {request.agent}")
        return handler(request, facts)

    def _analyst(self, request, facts) -> AnalystOutput:
        rsi = facts.get("rsi")
        if rsi is None:
            regime, confidence = "range", 0.4
        elif rsi > 60:
            regime, confidence = "trend_up", 0.7
        elif rsi < 40:
            regime, confidence = "trend_down", 0.6
        else:
            regime, confidence = "range", 0.5
        return AnalystOutput(
            regime=regime, confidence=confidence,
            key_indicators=["rsi", "sma_short", "vwap_24h"],
            # Rounded the way a model would write it; the guard's tolerance
            # accepts it, which is itself worth exercising offline.
            summary=f"RSIは{rsi:.1f}で、レジームを {regime} と判断した。",
            risks=["ボラティリティ拡大", "出来高の細り"])

    def _strategy(self, request, facts) -> StrategyOutput:
        wants_buy = self._wants_buy(request.agent)
        last = dec(facts.get("last_price") or 0)
        if not wants_buy or last <= 0:
            return StrategyOutput(
                action="wait", entry=None, take_profit=None, stop_loss=None,
                confidence=0.55, thesis="優位性が確認できないため見送る。",
                invalidation="直近高値の明確なブレイク")
        entry = last * Decimal("0.999")
        return StrategyOutput(
            action="buy", entry=float(round(entry)),
            take_profit=float(round(entry * Decimal("1.012"))),
            stop_loss=float(round(entry * Decimal("0.992"))),
            confidence=0.62 + self._jitter(request.agent),
            thesis="押し目を拾い、直近レンジ上限を目標とする。",
            invalidation="安値割れで論拠は消える")

    def _critique(self, request, facts) -> CritiqueOutput:
        ids = [p["id"] for p in facts.get("_proposals", [])] or ["P1", "P2"]
        return CritiqueOutput(
            critiques=[Critique(proposal_id=i, weakness="損切り幅が浅い",
                                severity="medium") for i in ids],
            revised_confidence=0.6)

    def _judge(self, request, facts) -> JudgeOutput:
        buys = [p for p in facts.get("_proposals", []) if p.get("action") == "buy"]
        # The threshold comes from the task the orchestrator built, not from a
        # literal here. This was a hardcoded 2, which meant the fixture kept
        # enforcing the old 2-of-3 rule after consensus_min became 1 — the stub
        # would refuse trades the real system allows, and every cycle test
        # would have been asserting the wrong behaviour.
        required = (facts.get("_consensus_rule") or {}).get(
            "required_buy_proposals", 1)
        if len(buys) < required:
            return JudgeOutput(decision="no_trade", consensus=0.3,
                               adopted_proposal_id=None, entry=None,
                               take_profit=None, stop_loss=None,
                               rationale=f"買い提案が{required}件に届かなかった。")
        best = buys[0]
        return JudgeOutput(
            decision="adopt", consensus=0.72,
            adopted_proposal_id=best["id"], entry=best["entry"],
            take_profit=best["take_profit"], stop_loss=best["stop_loss"],
            rationale="複数案が同方向で、損益比も許容範囲にある。")

    def _risk(self, request, facts) -> RiskOutput:
        plan = facts.get("_adopted_plan") or {}
        return RiskOutput(
            approved=self.approve_risk,
            qty_btc=plan.get("qty_btc"),
            stop_loss=plan.get("stop_loss"),
            take_profit=plan.get("take_profit"),
            risk_jpy=plan.get("risk_jpy"),
            rationale="1トレードリスク上限の範囲内。" if self.approve_risk
                      else "リスクが上限を超える。",
            # The guard requires a rejection to say what would fix it.
            adjustments=[] if self.approve_risk else ["損切り幅を縮めること"])

    def _reflect(self, request, facts) -> ReflectOutput:
        return ReflectOutput(
            lessons=[Lesson(text="レンジ相場での順張りは勝率が低い",
                            regime_tag="range",
                            evidence="直近20トレードのレンジ勝率 35%",
                            confidence=0.6)],
            summary="集計ベースの所見。")

    def _scout(self, request, facts) -> ScoutOutput:
        return ScoutOutput(bias="neutral", worth_full_debate=False,
                           note="決め手に欠ける。")

    # -- helpers ----------------------------------------------------------

    def _wants_buy(self, agent: str) -> bool:
        if self.bias == "buy":
            return True
        if self.bias == "wait":
            return False
        # "mixed": the last strategist in the roster waits and the rest buy,
        # so the fixture keeps producing a split verdict whatever the roster
        # size. Named the contrarian explicitly until that agent was removed.
        return not agent.endswith(STRATEGISTS[-1].split(":")[-1])

    @staticmethod
    def _jitter(agent: str) -> float:
        digest = hashlib.sha256(agent.encode()).hexdigest()
        return int(digest[:2], 16) / 2550.0


def _facts(prefix: str) -> dict:
    """Pull the snapshot back out of the shared prefix.

    The stub reads the same JSON the real model would, which keeps the two on
    the same footing and catches prefix-assembly bugs.
    """
    start = prefix.find("{")
    end = prefix.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        data = json.loads(prefix[start:end + 1])
    except json.JSONDecodeError:
        return {}
    flat = dict(data.get("indicators", {}))
    flat["last_price"] = data.get("last_price")
    flat["mid_price"] = data.get("mid_price")
    return flat


def _task_facts(task: str) -> dict:
    """The orchestrator embeds one JSON object in every task body; the stub
    reads the fields it needs from there under `_`-prefixed keys."""
    start = task.find("{")
    end = task.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        payload = json.loads(task[start:end + 1])
    except json.JSONDecodeError:
        return {}
    return {f"_{k}": v for k, v in payload.items()}
