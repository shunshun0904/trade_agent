"""The agents, wired to their inputs (spec 4).

Each function decides exactly what its agent is allowed to see. That narrowing
is the whole point of phase 1 in the debate protocol: a strategist that can see
another strategist's proposal is not an independent proposal, it is an echo
(spec 4.1). `saw_agents` records what was visible so the property is auditable
from the logs rather than taken on trust.

Roles and calls are different numbers, which is worth stating because they get
confused. Spec 4 defines nine roles; the inspector (A5) and the commander (A6)
are not implemented (see docs/OPEN-QUESTIONS.md A-5), leaving seven.

A cycle makes **seven or nine calls**, not one fixed number, because the three
strategists each speak twice — once to propose, once to critique the other two
— and because the last two agents are conditional:

    analyst            1
    strategy × 3       3   (phase 1, independent proposals)
    critique  × 3      3   (phase 2, same three agents, anonymised inputs)
                      ---
                       7   ← every cycle gets this far
    judge              1   ┐ only when at least `consensus_min` strategists
    risk               1   ┘ proposed a buy (2 of 3 normally, 1 during a probe)
                      ---
                       9

`Cycle._adjudicate` applies that consensus rule before calling the judge:
with nothing to adjudicate, there is nothing to pay a judge for. So a
no-consensus cycle — the common case, and a normal outcome under spec 4.1 —
costs seven calls, not nine.

The scout is a separate opt-in call in the screen function and is off by
default (`screening.scout_mode`); the reflector runs in its own function on its
own schedule. Neither is part of the count above.

Stored rows are a third number again: `AgentRunner.run` records every attempt,
so a call the guard rejects and retries leaves more than one row in
`agent_calls`. A row count is therefore attempts, not calls — filter on `ok`
to count calls, and group by `cycle_id` to count cycles.
"""

from __future__ import annotations

from typing import Callable

from ..models.agent_io import (
    AnalystOutput,
    CritiqueOutput,
    JudgeOutput,
    ReflectOutput,
    RiskOutput,
    ScoutOutput,
    StrategyOutput,
)
from .base import AgentRunner

STRATEGISTS = ["strategy:trend", "strategy:meanrev", "strategy:contrarian"]

Validator = Callable | None


def run_analyst(runner: AgentRunner, *, validator: Validator = None) -> AnalystOutput:
    return runner.run(
        "analyst", {}, AnalystOutput, saw_agents=[], validator=validator,
        instructions="MarketSnapshotのみを根拠に、現在の地合いを判定せよ。")


def run_strategy(runner: AgentRunner, agent: str, analyst: AnalystOutput, *,
                 boredom_probe: bool = False,
                 validator: Validator = None) -> StrategyOutput:
    """Phase 1 — independent proposal.

    The only other agent's output in scope is A1's regime read, which every
    strategist sees identically. No strategist ever sees another's proposal
    at this stage.
    """
    payload = {"market_read": analyst.model_dump()}
    instructions = "自分の立場から、買うか見送るかを1案だけ提示せよ。"
    if boredom_probe:
        payload["probe_mode"] = True
        instructions += (
            "\n注記: 今回は退屈防止ルール(72時間無取引)による偵察サイクルである。"
            "採用された場合の発注は最小ロット固定・損切りはentryから-0.7%以内に"
            "機械的に設定される。統計的優位を無理に主張する必要はない。"
            "根拠が薄いなら素直に wait と答えよ。")
    return runner.run(agent, payload, StrategyOutput, saw_agents=["analyst"],
                      validator=validator, instructions=instructions)


def run_critique(runner: AgentRunner, agent: str, others: list[dict], *,
                 validator: Validator = None) -> CritiqueOutput:
    """Phase 2 — critique of the other two, anonymised.

    `others` carries opaque ids only. Which strategist wrote which proposal is
    never revealed, so a critique cannot become an argument about roles.
    """
    return runner.run(
        agent.replace("strategy:", "critique:"),
        {"proposals": others}, CritiqueOutput,
        saw_agents=["strategy:anonymous"], validator=validator,
        instructions="匿名化された他の2案について、それぞれ最大の弱点を1つ指摘せよ。")


def run_judge(runner: AgentRunner, *, analyst: AnalystOutput, proposals: list[dict],
              critiques: list[dict], consensus_min: int, buy_count: int,
              validator: Validator = None) -> JudgeOutput:
    return runner.run(
        "judge",
        {
            "market_read": analyst.model_dump(),
            "proposals": proposals,
            "critiques": critiques,
            "consensus_rule": {
                "buy_proposals": buy_count,
                "required_buy_proposals": consensus_min,
                "total_proposals": len(proposals),
            },
        },
        JudgeOutput,
        saw_agents=["analyst", *STRATEGISTS, "critique"],
        validator=validator,
        instructions="3案と批判を統合し、採択案を1つ決めるか no_trade を選べ。")


def run_risk(runner: AgentRunner, *, plan: dict, account: dict, limits: dict,
             validator: Validator = None) -> RiskOutput:
    return runner.run(
        "risk",
        {"adopted_plan": plan, "account": account, "risk_limits": limits},
        RiskOutput, saw_agents=["judge"], validator=validator,
        instructions=("採択案のサイズ・損切り・利確を査定せよ。"
                      "数量とリスク額は算出済みである。作り直さず、妥当性を判断せよ。"))


def run_reflect(runner: AgentRunner, *, statistics: dict,
                validator: Validator = None) -> ReflectOutput:
    return runner.run(
        "reflect", statistics, ReflectOutput, saw_agents=[], validator=validator,
        instructions=("決済済みトレードの集計統計から教訓を抽出せよ。"
                      "個別トレードから断定してはならない。"))


def run_scout(runner: AgentRunner, *, validator: Validator = None) -> ScoutOutput:
    return runner.run(
        "scout", {}, ScoutOutput, saw_agents=[], validator=validator,
        instructions="市況を一言で評価し、フル議論に値するかだけを答えよ。")
