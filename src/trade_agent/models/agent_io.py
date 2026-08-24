"""Agent output schemas (spec 4).

Every agent returns JSON and nothing else. These models are both the parse
target and the JSON Schema sent to the API as a structured-output constraint,
so a malformed response is a transport-level failure rather than something the
guard has to untangle.

Numbers arrive as JSON floats and are converted to Decimal at the guard
boundary — the models never do arithmetic on them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Regime = Literal["trend_up", "trend_down", "range", "volatile"]
Action = Literal["buy", "wait"]


class _AgentModel(BaseModel):
    """Strict: unknown keys are a rejection, not something to shrug off."""

    model_config = ConfigDict(extra="forbid")


class AnalystOutput(_AgentModel):
    """A1 — qualitative read of the regime."""

    regime: Regime
    confidence: float = Field(ge=0, le=1)
    key_indicators: list[str] = Field(
        description="names of MarketSnapshot indicators that drove this read")
    summary: str = Field(description="two or three sentences, Japanese")
    risks: list[str] = Field(description="what would invalidate this read")


class StrategyOutput(_AgentModel):
    """A2a/A2b/A2c — one independent proposal.

    `entry`/`take_profit`/`stop_loss` are required when action is "buy" and
    must be null when it is "wait"; the guard enforces that pairing.
    """

    action: Action
    entry: float | None
    take_profit: float | None
    stop_loss: float | None
    confidence: float = Field(ge=0, le=1)
    thesis: str = Field(description="why this trade, in Japanese")
    invalidation: str = Field(description="the observation that would kill this idea")


class Critique(_AgentModel):
    proposal_id: str = Field(description="anonymised id of the proposal being critiqued")
    weakness: str
    severity: Literal["low", "medium", "high"]


class CritiqueOutput(_AgentModel):
    """A2 phase 2 — each strategist attacks the other two anonymised proposals."""

    critiques: list[Critique]
    revised_confidence: float = Field(
        ge=0, le=1, description="confidence in your own proposal after reading theirs")


class JudgeOutput(_AgentModel):
    """A3 — adoption or no_trade.

    The 2-of-3 consensus rule is enforced in Python (spec 4.1); the judge's job
    is to pick *which* proposal and to say why.
    """

    decision: Literal["adopt", "no_trade"]
    consensus: float = Field(ge=0, le=1)
    adopted_proposal_id: str | None
    entry: float | None
    take_profit: float | None
    stop_loss: float | None
    rationale: str


class RiskOutput(_AgentModel):
    """A4 — sizing and stop assessment.

    The numbers are advisory: Python recomputes qty and risk from the risk
    rules and treats a mismatch as a guard violation (spec 5).
    """

    approved: bool
    qty_btc: float | None
    stop_loss: float | None
    take_profit: float | None
    risk_jpy: float | None
    rationale: str
    adjustments: list[str]


class AuditorOutput(_AgentModel):
    """A5 — cross-agent logical consistency."""

    ok: bool
    violations: list[str]
    retry_target: Literal["analyst", "strategy", "judge", "risk", "none"]
    notes: str


class CommanderOutput(_AgentModel):
    """A6 — final GO/NO-GO plus the owner-facing text."""

    go: bool
    headline: str = Field(description="one line, Japanese")
    report_text: str = Field(description="owner-facing report, Japanese")


class Lesson(_AgentModel):
    text: str
    regime_tag: Regime | Literal["all"]
    evidence: str = Field(description="the aggregate statistic this rests on")
    confidence: float = Field(ge=0, le=1)


class ReflectOutput(_AgentModel):
    """A7 — lessons drawn from aggregate statistics, never a single trade."""

    lessons: list[Lesson]
    summary: str


class ScoutOutput(_AgentModel):
    """Optional scout mode (spec 9): one cheap call, one line of read."""

    bias: Literal["bullish", "bearish", "neutral"]
    worth_full_debate: bool
    note: str


AGENT_OUTPUT_MODELS = {
    "analyst": AnalystOutput,
    "strategy": StrategyOutput,
    "critique": CritiqueOutput,
    "judge": JudgeOutput,
    "risk": RiskOutput,
    "auditor": AuditorOutput,
    "commander": CommanderOutput,
    "reflect": ReflectOutput,
    "scout": ScoutOutput,
}
