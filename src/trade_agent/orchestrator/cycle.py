"""The decision cycle (spec 3 layer 2, spec 4.1).

    reconcile -> budget -> snapshot -> safety -> boredom
      -> A1 analyst
      -> A2a/A2b/A2c independent proposals   (mutually blind)
      -> A2 cross-critique                   (anonymised)
      -> consensus rule                      (Python, not the model)
      -> A3 judge -> sizing (Python) -> A4 risk
      -> execute

Three things are deliberately not the model's job. The consensus rule is
counted in Python, because a rule the judge could talk itself out of is not a
rule. Position size is computed in Python and the risk agent is checked against
it. And every safety gate is evaluated before the first token is spent, so a
halted system costs nothing to keep running.

The owner-facing report is assembled in Python too (see `report.py`), from the
structured output the agents have already produced. There is no separate agent
writing prose about a decision it did not make.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal

from ..agents.base import AgentRunner
from ..agents.roster import (
    STRATEGISTS,
    run_analyst,
    run_critique,
    run_judge,
    run_risk,
    run_strategy,
)
from ..errors import DuplicateOrder, ExchangeError, GuardRejection, LockNotAcquired
from ..guards.deterministic import DeterministicGuard
from ..models.agent_io import AnalystOutput, StrategyOutput
from ..models.market import MarketSnapshot
from ..models.state import CycleTrigger, Halt, HaltReason, SystemState
from ..models.trading import ExecutionPlan, TradeRecord
from ..money import ZERO, dec, jpy, quantize_price
from ..risk.boredom import (
    BoredomDecision,
    evaluate_boredom,
    mechanical_probe_plan,
    probe_stop_loss,
)
from ..storage.base import LOCK_DECIDE
from ..timeutil import iso
from .report import compose_no_trade, compose_traded

log = logging.getLogger(__name__)

CYCLE_LOCK_TTL_SECONDS = 86_400


@dataclass
class CycleOutcome:
    cycle_id: str
    trigger: CycleTrigger
    started_at: str
    traded: bool = False
    no_trade_reason: str = ""
    probe: bool = False
    halts: list[Halt] = field(default_factory=list)
    plan: ExecutionPlan | None = None
    report_text: str = ""
    headline: str = ""
    consensus: Decimal | None = None
    buy_count: int = 0
    consensus_min: int = 2
    regime: str | None = None
    llm_cost_jpy: Decimal = ZERO
    llm_calls: int = 0
    cache_hits: int = 0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.traded:
            return f"{self.cycle_id}: entry placed ({'probe' if self.probe else 'normal'})"
        return f"{self.cycle_id}: no trade — {self.no_trade_reason}"


def new_cycle_id(now, trigger: CycleTrigger) -> str:
    return f"cyc-{now.strftime('%Y%m%dT%H%M%SZ')}-{trigger}"


def entry_order_id(cycle_id: str) -> str:
    """Deterministic client order id.

    Re-running a cycle re-derives the same id, so the conditional create in the
    order table rejects the second attempt. This is the innermost of three
    defences against a double entry — the others are the per-cycle lock and the
    global decide lock (spec 8, spec 14).
    """
    return hashlib.sha256(f"{cycle_id}:entry".encode()).hexdigest()[:32]


class DecisionCycle:
    def __init__(self, ctx, *, trigger: CycleTrigger = CycleTrigger.MANUAL,
                 cycle_id: str | None = None):
        self.ctx = ctx
        self.config = ctx.config
        self.clock = ctx.clock
        self.trigger = trigger
        self.cycle_id = cycle_id or new_cycle_id(self.clock.now(), trigger)
        self.invocation_id = f"{self.cycle_id}:{uuid.uuid4().hex[:8]}"
        self.outcome = CycleOutcome(cycle_id=self.cycle_id, trigger=trigger,
                                    started_at=iso(self.clock.now()))
        # Kept so an abort after A1 can still say what the market looked like.
        self._analyst: AnalystOutput | None = None

    # -- entry point -------------------------------------------------------

    def run(self) -> CycleOutcome:
        if not self._acquire_locks():
            raise LockNotAcquired(
                f"another invocation is running (cycle {self.cycle_id})")
        try:
            return self._run_locked()
        finally:
            self.ctx.store.locks.release(LOCK_DECIDE, self.invocation_id)

    def _acquire_locks(self) -> bool:
        """Global decide lock, then a per-cycle lock that is never released.

        The per-cycle lock is what makes a *repeat* of the same cycle id a
        no-op, as opposed to the decide lock which only stops two cycles
        overlapping in time.
        """
        locks = self.ctx.store.locks
        now = self.clock.now()
        if not locks.acquire(LOCK_DECIDE, self.invocation_id,
                             self.config.execution.lock_lease_seconds, now):
            return False
        if not locks.acquire(f"cycle:{self.cycle_id}", self.invocation_id,
                             CYCLE_LOCK_TTL_SECONDS, now):
            locks.release(LOCK_DECIDE, self.invocation_id)
            log.warning("cycle %s has already run; refusing to repeat it",
                        self.cycle_id)
            return False
        return True

    # -- the cycle ---------------------------------------------------------

    def _run_locked(self) -> CycleOutcome:
        ctx, out = self.ctx, self.outcome
        now = self.clock.now()
        state = ctx.load_state()

        executor = ctx.executor(owner=self.invocation_id)
        out.notes.extend(executor.reconcile_pending())

        budget = ctx.budget_state(state)
        if not budget.llm_allowed:
            return self._abort(state, "monthly LLM budget exhausted; "
                                      "deterministic monitoring continues")

        try:
            snapshot = ctx.snapshot_builder().build(
                position=state.open_position, equity_override=None)
        except ExchangeError as exc:
            return self._abort(state, f"market data unavailable: {exc}")

        state.equity_jpy = snapshot.account.equity_jpy
        state.peak_equity_jpy = max(state.peak_equity_jpy, state.equity_jpy)

        if ctx.risk.should_kill(state):
            ctx.risk.engage_kill_switch(
                state, now, "drawdown from initial capital reached the limit")
            ctx.save_state(state)
            self._notify_kill(state)
            return self._abort(state, "kill switch engaged", save=False)

        if ctx.risk.check_flash_move(state, now, snapshot.indicators.change_15m_pct):
            out.notes.append("flash-move pause armed")

        halts = ctx.risk.evaluate_halts(
            state, now, change_15m_pct=snapshot.indicators.change_15m_pct,
            budget_stopped=not budget.llm_allowed)
        out.halts = halts

        boredom = evaluate_boredom(self.config, state, now, halts)
        out.probe = boredom.triggered
        out.consensus_min = boredom.consensus_min
        if boredom.triggered:
            out.notes.append(f"3-day rule active: {boredom.reason}")

        blocking = self._blocking_halts(halts, boredom)
        if blocking:
            return self._abort(state, f"halted: {blocking[0].detail or blocking[0].reason}")

        return self._debate(state, snapshot, boredom, executor)

    def _blocking_halts(self, halts: list[Halt],
                        boredom: BoredomDecision) -> list[Halt]:
        """The boredom rule can relax consensus, never a safety gate.

        It is allowed past exactly one thing: the daily debate cap, which is a
        cost control rather than a safety rule (spec 9).
        """
        blocking = [h for h in halts if h.reason is not HaltReason.DEBATE_LIMIT]
        if boredom.triggered:
            return blocking
        return blocking + [h for h in halts if h.reason is HaltReason.DEBATE_LIMIT]

    # -- the debate --------------------------------------------------------

    def _debate(self, state: SystemState, snapshot: MarketSnapshot,
                boredom: BoredomDecision, executor) -> CycleOutcome:
        ctx, out = self.ctx, self.outcome
        guard = DeterministicGuard(self.config, snapshot)
        runner = AgentRunner(llm=ctx.llm, config=self.config, store=ctx.store,
                             clock=self.clock, cycle_id=self.cycle_id,
                             router=ctx.router)
        runner.set_prefix(
            snapshot.to_prompt_json(),
            lessons=self._lessons_digest(),
            trade_digest=self._trade_digest(),
            state_digest=self._state_digest(state, boredom))

        try:
            analyst = run_analyst(runner, validator=guard.validate_analyst)
            self._analyst = analyst
            out.regime = analyst.regime
            proposals = self._collect_proposals(runner, guard, analyst,
                                                probe=boredom.triggered)
            self._collect_critiques(runner, guard, proposals)
            plan = self._adjudicate(runner, guard, state, snapshot, analyst,
                                    proposals, boredom)
        except GuardRejection as exc:
            self._finish_llm_accounting(state, runner)
            return self._abort(state,
                               f"agent output could not be validated: {exc}")

        if plan is None:
            self._finish_llm_accounting(state, runner)
            return self._abort(state, out.no_trade_reason or "no trade this cycle")

        approved = self._review(runner, guard, state, snapshot, plan, analyst)
        self._finish_llm_accounting(state, runner)
        if not approved:
            return self._abort(state, out.no_trade_reason or "final review said no")

        return self._execute(state, plan, executor)

    def _collect_proposals(self, runner: AgentRunner, guard: DeterministicGuard,
                           analyst: AnalystOutput, *, probe: bool) -> list[dict]:
        """Phase 1 — three proposals, none of which has seen the others.

        Sequential calls, but each request is built only from the snapshot and
        A1's read, so ordering carries no information between them. The
        `saw_agents` field on every logged call is what makes that auditable.
        """
        proposals: list[dict] = []
        for index, agent in enumerate(STRATEGISTS):
            output: StrategyOutput = run_strategy(
                runner, agent, analyst, boredom_probe=probe,
                validator=guard.validate_strategy)
            proposals.append({
                "id": f"P{index + 1}",
                "agent": agent,
                "action": output.action,
                "entry": output.entry,
                "take_profit": output.take_profit,
                "stop_loss": output.stop_loss,
                "confidence": output.confidence,
                "thesis": output.thesis,
                "invalidation": output.invalidation,
            })
        return proposals

    def _collect_critiques(self, runner: AgentRunner, guard: DeterministicGuard,
                           proposals: list[dict]) -> list[dict]:
        """Phase 2 — one round, each strategist sees the other two anonymised."""
        critiques: list[dict] = []
        for index, agent in enumerate(STRATEGISTS):
            others = [_anonymise(p) for i, p in enumerate(proposals) if i != index]
            output = run_critique(runner, agent, others)
            critiques.append({"by": f"C{index + 1}",
                              "revised_confidence": output.revised_confidence,
                              "critiques": [c.model_dump() for c in output.critiques]})
            proposals[index]["revised_confidence"] = output.revised_confidence
        return critiques

    def _adjudicate(self, runner: AgentRunner, guard: DeterministicGuard,
                    state: SystemState, snapshot: MarketSnapshot,
                    analyst: AnalystOutput, proposals: list[dict],
                    boredom: BoredomDecision) -> ExecutionPlan | None:
        """Consensus rule, judge, and Python-side sizing."""
        out = self.outcome
        buys = [p for p in proposals if p["action"] == "buy"]
        out.buy_count = len(buys)
        consensus_min = boredom.consensus_min

        if len(buys) < consensus_min:
            if boredom.triggered:
                return self._mechanical_probe(state, snapshot, analyst)
            out.no_trade_reason = (
                f"consensus not reached: {len(buys)}/{len(proposals)} buy proposals, "
                f"{consensus_min} required")
            return None

        critique_payload = [
            {"proposal_id": p["id"], "revised_confidence": p.get("revised_confidence")}
            for p in proposals]
        judge = run_judge(
            runner, analyst=analyst, proposals=[_anonymise(p) for p in proposals],
            critiques=critique_payload, consensus_min=consensus_min,
            buy_count=len(buys),
            validator=lambda o: guard.validate_judge(
                o, proposal_ids=[p["id"] for p in proposals],
                buy_count=len(buys), consensus_min=consensus_min))
        out.consensus = dec(judge.consensus)

        if judge.decision == "no_trade":
            if boredom.triggered:
                return self._mechanical_probe(state, snapshot, analyst)
            out.no_trade_reason = f"judge chose no_trade: {judge.rationale}"
            return None

        entry = quantize_price(dec(judge.entry), self.config.exchange.price_digits)
        stop = quantize_price(dec(judge.stop_loss), self.config.exchange.price_digits)
        take = quantize_price(dec(judge.take_profit), self.config.exchange.price_digits)

        if boredom.triggered:
            # Spec 7: a probe's stop is set mechanically, tighter than whatever
            # the debate produced, and its size is the minimum lot.
            stop = max(stop, probe_stop_loss(self.config, entry))

        return self._size(state, snapshot, entry, stop, take,
                          probe=boredom.triggered, regime=analyst.regime,
                          thesis=judge.rationale)

    def _mechanical_probe(self, state: SystemState, snapshot: MarketSnapshot,
                          analyst: AnalystOutput) -> ExecutionPlan | None:
        """Spec 7: no buy proposal, but the 72-hour clock still has to be reset."""
        out = self.outcome
        raw = mechanical_probe_plan(self.config, snapshot, analyst.regime)
        if raw is None:
            out.no_trade_reason = (
                "3-day rule fired but no VWAP was available for a mechanical probe")
            return None
        out.notes.append("no buy proposal; falling back to a mechanical probe")
        return self._size(state, snapshot, raw["entry"], raw["stop_loss"],
                          raw["take_profit"], probe=True, regime=analyst.regime,
                          thesis=raw["rationale"])

    def _size(self, state: SystemState, snapshot: MarketSnapshot, entry: Decimal,
              stop: Decimal, take: Decimal, *, probe: bool, regime: str | None,
              thesis: str) -> ExecutionPlan | None:
        out = self.outcome
        sizing = self.ctx.risk.position_size(
            equity=snapshot.account.equity_jpy, entry=entry, stop_loss=stop,
            jpy_available=snapshot.account.jpy_free, probe=probe)
        if not sizing.ok:
            out.no_trade_reason = f"sizing rejected the plan: {sizing.reason}"
            return None
        return ExecutionPlan(
            cycle_id=self.cycle_id, trade_id=f"trd-{self.cycle_id}", entry=entry,
            stop_loss=stop, take_profit=take, qty_btc=sizing.qty_btc,
            risk_jpy=sizing.risk_jpy, probe=probe, regime=regime, thesis=thesis,
            consensus=out.consensus,
            client_order_id=entry_order_id(self.cycle_id))

    def _review(self, runner: AgentRunner, guard: DeterministicGuard,
                state: SystemState, snapshot: MarketSnapshot,
                plan: ExecutionPlan, analyst: AnalystOutput) -> bool:
        """A4 sizing review, then the last structural check on the exact
        numbers that would go to the exchange.

        This is the end of the agent chain. What used to follow — an auditor
        looking for contradictions and a commander casting a final vote — was
        a model checking models, downstream of four gates that do not guess:
        the consensus count, the sizing calculation, the per-agent guard, and
        `check_executable` below. Removing them removed two ways for a cycle
        to be talked out of a decision Python had already settled.
        """
        out = self.outcome
        limits = {
            "per_trade_risk_jpy": float(self.ctx.risk.risk_limit_jpy(
                snapshot.account.equity_jpy, probe=plan.probe)),
            "computed_qty_btc": float(plan.qty_btc),
            "computed_risk_jpy": float(plan.risk_jpy),
            "min_order_btc": float(self.config.exchange.min_order_btc),
            "probe": plan.probe,
        }
        risk_out = run_risk(
            runner,
            plan={"entry": float(plan.entry), "stop_loss": float(plan.stop_loss),
                  "take_profit": float(plan.take_profit),
                  "qty_btc": float(plan.qty_btc), "risk_jpy": float(plan.risk_jpy),
                  "thesis": plan.thesis},
            account={"equity_jpy": float(snapshot.account.equity_jpy),
                     "jpy_free": float(snapshot.account.jpy_free)},
            limits=limits,
            validator=lambda o: guard.validate_risk(
                o, expected_qty=plan.qty_btc, expected_risk_jpy=plan.risk_jpy,
                entry=plan.entry, stop_loss=plan.stop_loss))
        if not risk_out.approved:
            out.no_trade_reason = f"risk management rejected the plan: {risk_out.rationale}"
            return False

        violations = guard.check_executable(
            entry=plan.entry, stop_loss=plan.stop_loss, take_profit=plan.take_profit,
            qty_btc=plan.qty_btc, jpy_available=snapshot.account.jpy_free,
            probe=plan.probe)
        if violations:
            out.no_trade_reason = "final structural check failed: " + "; ".join(violations)
            return False

        out.headline, out.report_text = compose_traded(
            analyst=analyst, plan=plan, risk=risk_out, state=state,
            buy_count=out.buy_count, proposal_count=len(STRATEGISTS),
            consensus=out.consensus,
            protection_note=self._protection_note())
        return True

    def _protection_note(self) -> str:
        """How this position will be protected, in one line for the report."""
        mode = self.config.execution.oco_mode
        if mode == "local":
            return "損切り・利確とも5分tickがローカル評価"
        return ("損切りは取引所側の stop 注文、利確は5分tickが監視"
                "(取引所が両脚を裏付ける場合は両方を取引所側に置く)")

    # -- execution ---------------------------------------------------------

    def _execute(self, state: SystemState, plan: ExecutionPlan,
                 executor) -> CycleOutcome:
        out = self.outcome
        now = self.clock.now()

        # The trade row is written before the order goes out so a later cold
        # `tick` can recover the stops for a fill it did not place.
        self.ctx.store.trades.put(TradeRecord(
            trade_id=plan.trade_id, cycle_id=self.cycle_id,
            pair=self.config.exchange.pair, probe=plan.probe, qty_btc=plan.qty_btc,
            entry_price=plan.entry, entry_order_id=entry_order_id(self.cycle_id),
            entry_at=now, stop_loss=plan.stop_loss, take_profit=plan.take_profit,
            regime=plan.regime, judge_output_id=self.cycle_id))

        try:
            result = executor.place_entry(plan)
        except DuplicateOrder:
            out.no_trade_reason = "this cycle has already placed its order"
            self.ctx.save_state(state)
            return out
        except ExchangeError as exc:
            out.no_trade_reason = f"order rejected by the exchange: {exc}"
            self.ctx.save_state(state)
            return out

        out.notes.extend(result.reconciled)
        if not result.placed:
            out.no_trade_reason = result.reason
        else:
            out.traded = True
            out.plan = plan

        state.last_full_debate_at = now
        state.daily.full_debates += 1
        if self.trigger is CycleTrigger.FLOOR:
            state.last_floor_run_at = now
        if self.trigger is CycleTrigger.FLASH_RECOVERY:
            state.flash_recovery_pending = False
        self.ctx.save_state(state)
        return out

    # -- prompt context ----------------------------------------------------

    def _lessons_digest(self) -> list[str]:
        rows = self.ctx.store.lessons.list(limit=self.config.snapshot.lessons_in_prompt)
        return [f"[{row.regime_tag}] {row.text}(根拠: {row.evidence})" for row in rows]

    def _trade_digest(self) -> str:
        """Aggregate recent performance. Never a per-trade narrative — the same
        discipline spec 9 imposes on the reflection agent."""
        trades = [t for t in self.ctx.store.trades.list_recent(50) if t.closed]
        real = [t for t in trades if not t.probe]
        if not real:
            return "決済済みトレードなし(統計的な判断材料は存在しない)。"
        wins = [t for t in real if (t.net_pnl_jpy or ZERO) > 0]
        total = sum((t.net_pnl_jpy or ZERO for t in real), ZERO)
        rr = [t.r_multiple() for t in real if t.r_multiple() is not None]
        avg_rr = (sum(rr, ZERO) / Decimal(len(rr))) if rr else None
        probes = [t for t in trades if t.probe]
        return (
            f"直近{len(real)}トレード(probe除外): 勝率 {len(wins)}/{len(real)}"
            f"、合計損益 {jpy(total)} 円"
            + (f"、平均RR {avg_rr:.2f}" if avg_rr is not None else "")
            + f"。probeトレード {len(probes)} 件は別集計。")

    def _state_digest(self, state: SystemState, boredom: BoredomDecision) -> str:
        idle = state.hours_since_last_entry(self.clock.now())
        idle_text = f"最終約定からの経過 {idle:.1f} 時間" if idle is not None else "約定履歴なし"
        parts = [
            f"equity {jpy(state.equity_jpy)} 円",
            f"連敗 {state.losing_streak}",
            f"当日実現損益 {jpy(state.daily.realized_pnl_jpy)} 円",
            idle_text,
        ]
        if boredom.triggered:
            parts.append(f"退屈防止ルール発動中(合意閾値 {boredom.consensus_min}/3)")
        return " / ".join(parts)

    # -- bookkeeping -------------------------------------------------------

    def _finish_llm_accounting(self, state: SystemState,
                               runner: AgentRunner) -> None:
        out = self.outcome
        out.llm_cost_jpy = runner.usage.cost_jpy
        out.llm_calls = runner.usage.calls
        out.cache_hits = runner.usage.cache_hits
        state.daily.llm_cost_jpy += runner.usage.cost_jpy
        state.monthly.llm_cost_jpy += runner.usage.cost_jpy
        if runner.usage.calls and not runner.usage.cache_hits:
            out.notes.append(
                "prompt cache did not engage on any call; the shared prefix is "
                "probably below the model's minimum cacheable size")

    def _abort(self, state: SystemState, reason: str, *,
               save: bool = True) -> CycleOutcome:
        out = self.outcome
        out.traded = False
        out.no_trade_reason = reason
        out.headline, out.report_text = compose_no_trade(
            reason=reason, analyst=self._analyst, state=state,
            buy_count=out.buy_count, proposal_count=len(STRATEGISTS)
            if self._analyst is not None else 0,
            consensus=out.consensus)
        if save:
            now = self.clock.now()
            if out.llm_calls:
                state.last_full_debate_at = now
                state.daily.full_debates += 1
            if self.trigger is CycleTrigger.FLOOR:
                state.last_floor_run_at = now
            self.ctx.save_state(state)
        log.info("cycle %s: no trade — %s", self.cycle_id, reason)
        return out

    def _notify_kill(self, state: SystemState) -> None:
        if self.ctx.notifier is None:
            return
        self.ctx.notifier.send(
            "キルスイッチ発動",
            f"equity {jpy(state.equity_jpy)} 円が初期資金からの許容下落幅に達しました。\n"
            "新規建てを完全に停止しました。再開には Claude から resume_trading を"
            "confirm=true で実行してください。")


def _anonymise(proposal: dict) -> dict:
    """Strip the author before a proposal is shown to anyone else (spec 4.1)."""
    return {k: v for k, v in proposal.items() if k != "agent"}
