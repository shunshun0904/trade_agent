"""LLM transport contract.

An agent hands over: a shared prefix (identical for every agent in the cycle,
so the prompt cache can serve it), a role instruction, a task payload, and the
Pydantic model the answer must satisfy. It gets back a parsed model and the
token counts. Nothing else crosses this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Protocol, Type, TypeVar

from pydantic import BaseModel

from ..money import ZERO

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )

    @property
    def total(self) -> int:
        return (self.input_tokens + self.output_tokens
                + self.cache_read_tokens + self.cache_write_tokens)


@dataclass
class LLMRequest:
    """`shared_prefix` must be byte-identical across the agents of one cycle —
    that identity is the entire prompt-cache saving (spec 11)."""

    agent: str
    shared_prefix: str
    role_instruction: str
    task: str
    output_model: Type[BaseModel]
    model: str
    max_tokens: int = 2000
    cacheable: bool = True
    use_batch: bool = False
    # Names of other agents whose output appears in `task`. Recorded so the
    # blind-proposal requirement of spec 4.1 is provable from the logs.
    saw_agents: list[str] = field(default_factory=list)


@dataclass
class LLMResponse:
    parsed: BaseModel
    usage: TokenUsage
    model: str
    duration_ms: int = 0
    raw_text: str = ""
    cost_jpy: Decimal = ZERO
    batch: bool = False


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...
