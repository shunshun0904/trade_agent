"""The agents, wired to their inputs (spec 4).

Each function decides exactly what its agent is allowed to see. That narrowing
is the whole point of phase 1 in the debate protocol: a strategist that can see
another strategist's proposal is not an independent proposal, it is an echo
(spec 4.1). `saw_agents` records what was visible so the property is auditable
from the logs rather than taken on trust.

Roles and calls are different numbers, which is worth stating because they get
confused. Spec 4 defines nine roles; five are not implemented. The inspector
(A5) and the commander (A6) were never built (docs/OPEN-QUESTIONS.md A-5); the
critique round, the judge (A3) and the risk reviewer (A4) were removed once the
roster came down to a single strategist.

A cycle makes **two calls**, and nothing about that is conditional any more:

    analyst            1   reads the regime from the snapshot
    strategy           1   proposes a buy, or waits
                      ---
                       2

The critique round, the judge (A3) and the risk reviewer (A4) were removed with
the second and third strategists. The first needs peers; the second needs
something to choose between; the third approved or vetoed a size Python had
already computed, and the guard rejected its output whenever its numbers
disagreed with Python's.

What replaced them is not nothing — it is the layer that was always underneath:

* `guards/deterministic.py` rejects bad price geometry, an entry too far from
  the market, a take-profit inside the round-trip fee, and any indicator value
  quoted in prose that does not match the snapshot;
* `risk/rules.py` sizes from the loss limit and refuses a plan whose stop is so
  wide that even the minimum lot would exceed it;
* `check_executable` runs those checks once more on the exact numbers the
  executor is about to send.

None of that guesses, and none of it can be argued with.

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
    ExitOutput,
    CritiqueOutput,
    JudgeOutput,
    ReflectOutput,
    RiskOutput,
    ScoutOutput,
    StrategyOutput,
)
from .base import AgentRunner
from ..roles import EXIT_AGENT, STRATEGISTS

# Re-exported from ..roles, which config also reads. Every count downstream
# derives from that one list — the prompts included, since a brief naming a
# number the roster disagrees with is a fact the model cannot check.

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
    instructions = ("market_read(A1の地合い判定)と MarketSnapshot の account / "
                    "constraints を踏まえ、買うか見送るかを1案だけ提示せよ。")
    if boredom_probe:
        payload["probe_mode"] = True
        instructions += (
            "\n注記: 今回は退屈防止ルール(72時間無取引)による偵察サイクルである。"
            "採用された場合の発注は最小ロット固定・損切りはentryから-0.7%以内に"
            "機械的に設定される。統計的優位を無理に主張する必要はない。"
            "根拠が薄いなら素直に wait と答えよ。")
    return runner.run(agent, payload, StrategyOutput, saw_agents=["analyst"],
                      validator=validator, instructions=instructions)


def run_exit(runner: AgentRunner, *, position: dict, allowed: dict,
             validator: Validator = None) -> ExitOutput:
    """The exit review on an open position.

    Sees the position and what the entry said would invalidate it, and nothing
    about opening anything — there is nothing to open while this runs. The
    `allowed` block states the bounds the guard will enforce anyway, so a
    rejection is a surprise rather than the normal path.
    """
    return runner.run(
        EXIT_AGENT, {"position": position, "allowed_actions": allowed},
        ExitOutput, saw_agents=[], validator=validator,
        instructions=("エントリー時の invalidation が実現したかを判定し、"
                      "hold か、締める操作を1つ選べ。"))


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
