"""Agent execution: one call, logged, guarded, retried.

Every LLM call in the system goes through :meth:`AgentRunner.run`, which is
what makes spec 10's audit requirement hold — there is no path that talks to a
model without leaving an `agent_calls` row and an S3 body behind.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, Type

from pydantic import BaseModel

from ..config import Config
from ..errors import GuardRejection, LLMError
from ..llm.base import LLMRequest, TokenUsage
from ..llm.registry import ModelRouter
from ..money import ZERO
from ..storage.base import AgentCallRecord
from ..timeutil import Clock, iso
from .prompts import ROLE_PROMPTS, constitution, rejection_note

log = logging.getLogger(__name__)


@dataclass
class CycleUsage:
    """Running total for one decision cycle."""

    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_jpy: Decimal = ZERO
    calls: int = 0
    cache_hits: int = 0

    def add(self, response) -> None:
        self.usage = self.usage + response.usage
        self.cost_jpy += response.cost_jpy
        self.calls += 1
        if response.usage.cache_read_tokens > 0:
            self.cache_hits += 1


class AgentRunner:
    """Holds the per-cycle shared prefix and runs individual agents against it."""

    def __init__(self, *, llm, config: Config, store, clock: Clock, cycle_id: str,
                 router: ModelRouter | None = None):
        self.llm = llm
        self.config = config
        self.store = store
        self.clock = clock
        self.cycle_id = cycle_id
        self.router = router or ModelRouter(config)
        self.usage = CycleUsage()
        self._prefix: str = ""
        self._sequence = 0

    # -- shared prefix -----------------------------------------------------

    def set_prefix(self, snapshot_json: str, *, lessons: list[str],
                   trade_digest: str, state_digest: str) -> str:
        """Assemble the cacheable block, once per cycle.

        Everything in here is identical across the cycle's agents; anything
        that differs per agent belongs in the role block or the task.
        """
        lesson_block = "\n".join(f"- {text}" for text in lessons) or "- (まだ教訓はない)"
        self._prefix = (
            f"{constitution(self.config)}\n"
            "# MarketSnapshot(確定値。この値以外を市場の事実として扱ってはならない)\n"
            f"{snapshot_json}\n\n"
            "# システム状態\n"
            f"{state_digest}\n\n"
            "# 直近の成績(集計値)\n"
            f"{trade_digest}\n\n"
            "# 教訓データベース(過去の集計分析から)\n"
            f"{lesson_block}\n"
        )
        self._warn_if_uncacheable()
        return self._prefix

    @property
    def prefix(self) -> str:
        return self._prefix

    def _warn_if_uncacheable(self) -> None:
        """A prefix under the model's minimum silently costs full price.

        claude-haiku-4-5 will not create a cache entry below 4096 tokens. The
        estimate below is deliberately crude — it only has to tell the owner
        which side of the line the prompt is on.
        """
        if not self.config.llm.use_prompt_cache:
            return
        estimate = _estimate_tokens(self._prefix)
        minimum = self.config.llm.cache_min_tokens
        if estimate < minimum:
            log.warning(
                "shared prefix is ~%d tokens, below the %d-token minimum for %s: "
                "prompt caching will not engage and every agent call pays full "
                "input price. Check cache_read_tokens in agent_calls.",
                estimate, minimum, self.config.llm.model)

    # -- calls -------------------------------------------------------------

    def run(self, agent: str, task_payload: dict, output_model: Type[BaseModel], *,
            saw_agents: list[str] | None = None,
            validator: Callable[[BaseModel], None] | None = None,
            role_override: str | None = None,
            instructions: str = "") -> BaseModel:
        """Call one agent, verify, retry on rejection.

        `validator` raises :class:`GuardRejection`; its violations are fed back
        to the model verbatim, up to `guard.max_retries` times (spec 5). After
        that the caller decides what a failed agent means — usually "skip this
        cycle".
        """
        role = role_override or ROLE_PROMPTS[agent]
        task = _render_task(task_payload, instructions)
        attempts = self.config.guard.max_retries
        last_error: Exception | None = None
        self._sequence += 1
        sequence = self._sequence

        for attempt in range(attempts):
            request = LLMRequest(
                agent=agent,
                shared_prefix=self._prefix,
                role_instruction=role,
                task=task,
                output_model=output_model,
                model=self.router.model_for(agent),
                max_tokens=self.config.llm.max_tokens,
                saw_agents=saw_agents or [],
            )
            try:
                response = self.llm.complete(request)
            except LLMError as exc:
                last_error = exc
                self._record(agent, sequence, None, attempt, ok=False,
                             error=str(exc), request=request)
                task = task + rejection_note([str(exc)])
                continue

            self.usage.add(response)
            try:
                if validator is not None:
                    validator(response.parsed)
            except GuardRejection as exc:
                last_error = exc
                self._record(agent, sequence, response, attempt, ok=False,
                             error="; ".join(exc.violations) or str(exc),
                             request=request)
                task = task + rejection_note(exc.violations or [str(exc)])
                continue

            self._record(agent, sequence, response, attempt, ok=True,
                         request=request)
            return response.parsed

        raise GuardRejection(
            f"{agent} failed {attempts} attempts: {last_error}",
            violations=getattr(last_error, "violations", []) or [str(last_error)],
            retry_target=agent)

    # -- logging -----------------------------------------------------------

    def _record(self, agent: str, sequence: int, response, attempt: int, *,
                ok: bool, request: LLMRequest, error: str | None = None) -> None:
        now = self.clock.now()
        key = None
        if self.store is not None:
            key = self._store_body(agent, sequence, attempt, request, response, now)
        record = AgentCallRecord(
            cycle_id=self.cycle_id,
            agent=agent,
            sequence=sequence,
            called_at=now,
            model=request.model,
            input_tokens=response.usage.input_tokens if response else 0,
            output_tokens=response.usage.output_tokens if response else 0,
            cache_read_tokens=response.usage.cache_read_tokens if response else 0,
            cache_write_tokens=response.usage.cache_write_tokens if response else 0,
            cost_jpy=response.cost_jpy if response else ZERO,
            retries=attempt,
            ok=ok,
            error=error,
            duration_ms=response.duration_ms if response else 0,
            io_s3_key=key,
            batch=bool(response.batch) if response else False,
            saw_agents=request.saw_agents,
        )
        if self.store is not None:
            self.store.agent_calls.put(record)

    def _store_body(self, agent: str, sequence: int, attempt: int,
                    request: LLMRequest, response, now) -> str | None:
        """Bodies go to S3; DynamoDB keeps the pointer (spec 10, 400KB limit)."""
        prefix = self.config.storage.agent_log_prefix
        key = (f"{prefix}{self.cycle_id}/{sequence:02d}-{agent.replace(':', '-')}"
               f"-{attempt}.json")
        payload = {
            "cycle_id": self.cycle_id,
            "agent": agent,
            "at": iso(now),
            "model": request.model,
            "saw_agents": request.saw_agents,
            "role_instruction": request.role_instruction,
            "task": request.task,
            # The shared prefix is identical for the whole cycle; storing it
            # once per call would multiply the log size for no information.
            "shared_prefix_sha": _sha(request.shared_prefix),
            "output": response.raw_text if response else None,
        }
        try:
            return self.store.blobs.put_json(key, payload)
        except Exception as exc:  # noqa: BLE001 - logging must not break trading
            log.warning("could not persist agent log %s: %s", key, exc)
            return None


def _render_task(payload: dict, instructions: str) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1,
                      default=str)
    header = f"{instructions}\n\n" if instructions else ""
    return f"{header}# 入力データ\n{body}"


def _estimate_tokens(text: str) -> int:
    """Rough token count for a mixed Japanese/English prompt.

    Japanese runs near one token per character while ASCII runs near four
    characters per token; counting them separately is far closer than a single
    ratio, and this only needs to be right to a factor well under two.
    """
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    other_chars = len(text) - ascii_chars
    return int(ascii_chars / 4) + other_chars


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()[:16]
