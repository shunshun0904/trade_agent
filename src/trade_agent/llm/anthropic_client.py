"""Anthropic Messages API transport.

Prompt structure, and why it is this way (spec 11):

    system[0]  shared prefix   <- cache_control: ephemeral
    system[1]  role instruction
    messages   the agent's task

Caching is a prefix match, so the shared block has to come first and be
byte-identical for every agent in the cycle. It carries the MarketSnapshot and
the lessons digest — the two largest and most-repeated pieces of context.

One caveat worth knowing before reading the cost logs: claude-haiku-4-5 does
not create a cache entry below a 4096-token prefix. A short snapshot silently
costs full price rather than erroring, so `cache_min_tokens` in config drives a
warning and `cache_read_tokens` is recorded on every call to make the miss
visible instead of invisible.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Type

from pydantic import BaseModel, ValidationError

from ..config import LLMConfig
from ..errors import ConfigError, LLMError
from .base import LLMRequest, LLMResponse, TokenUsage

log = logging.getLogger(__name__)

JSON_ONLY_SUFFIX = (
    "\n\nReturn a single JSON object matching this schema and nothing else. "
    "No prose, no markdown fence.\n\nSchema:\n{schema}"
)


class AnthropicLLMClient:
    def __init__(self, api_key: str, config: LLMConfig, *, client=None,
                 cost_meter=None):
        self.config = config
        self.cost_meter = cost_meter
        self._client = client
        self._api_key = api_key
        self._structured = config.structured_output

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self._api_key,
                timeout=self.config.timeout_seconds,
                max_retries=self.config.max_api_retries,
            )
        return self._client

    # -- synchronous ------------------------------------------------------

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.monotonic()
        params = self._build_params(request)
        if self._structured:
            try:
                message = self.client.messages.parse(
                    output_format=request.output_model, **params)
                parsed = message.parsed_output
                raw = _first_text(message)
            except Exception as exc:  # noqa: BLE001 - inspected below
                if not _is_schema_unsupported(exc):
                    raise LLMError(f"{request.agent}: LLM call failed: {exc}") from exc
                log.warning("structured output unsupported for %s; falling back "
                            "to instructed JSON", self.config.model)
                self._structured = False
                return self.complete(request)
        else:
            message = self.client.messages.create(**self._with_schema_hint(
                params, request.output_model))
            raw = _first_text(message)
            parsed = _parse_json(raw, request.output_model, request.agent)

        usage = _usage_of(message)
        duration_ms = int((time.monotonic() - started) * 1000)
        # request.model, not config.model: the router may send an agent to a
        # different model, and both the price and the recorded model must be
        # the one that actually ran.
        cost = (self.cost_meter.cost_jpy(usage, model=request.model)
                if self.cost_meter else None)
        return LLMResponse(parsed=parsed, usage=usage, model=request.model,
                           duration_ms=duration_ms, raw_text=raw,
                           cost_jpy=cost if cost is not None else LLMResponse.cost_jpy,
                           batch=False)

    # -- batch (spec 11: A7 has no latency requirement) --------------------

    def submit_batch(self, requests: list[LLMRequest]) -> str:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        entries = []
        for index, request in enumerate(requests):
            if request.model != self.config.model:
                # The batch is submitted in one Lambda invocation and collected
                # in another, so nothing survives in memory to say which entry
                # ran on which model. Rather than bill the whole batch at the
                # default rate and understate the spend the hard stop reads,
                # refuse the case that would need the bookkeeping.
                raise ConfigError(
                    f"batch entry {request.agent} asks for {request.model!r} "
                    f"but the batch path can only bill {self.config.model!r}. "
                    "Send it through complete() instead.")
            params = self._with_schema_hint(self._build_params(request),
                                            request.output_model)
            entries.append(Request(
                custom_id=f"{request.agent}-{index}",
                params=MessageCreateParamsNonStreaming(**params)))
        batch = self.client.messages.batches.create(requests=entries)
        return batch.id

    def poll_batch(self, batch_id: str,
                   output_models: dict[str, Type[BaseModel]]
                   ) -> dict[str, LLMResponse] | None:
        """None while the batch is still running; a dict keyed by custom_id when
        it has ended."""
        batch = self.client.messages.batches.retrieve(batch_id)
        if batch.processing_status != "ended":
            return None
        out: dict[str, LLMResponse] = {}
        for result in self.client.messages.batches.results(batch_id):
            if result.result.type != "succeeded":
                log.warning("batch %s entry %s: %s", batch_id, result.custom_id,
                            result.result.type)
                continue
            message = result.result.message
            agent = result.custom_id.rsplit("-", 1)[0]
            model_cls = output_models.get(agent)
            if model_cls is None:
                continue
            raw = _first_text(message)
            usage = _usage_of(message)
            # Safe because submit_batch refuses any other model.
            model = self.config.model
            cost = (self.cost_meter.cost_jpy(usage, model=model, batch=True)
                    if self.cost_meter else None)
            out[result.custom_id] = LLMResponse(
                parsed=_parse_json(raw, model_cls, agent), usage=usage,
                model=model, raw_text=raw,
                cost_jpy=cost if cost is not None else LLMResponse.cost_jpy,
                batch=True)
        return out

    def cancel_batch(self, batch_id: str) -> None:
        self.client.messages.batches.cancel(batch_id)

    # -- prompt assembly --------------------------------------------------

    def _build_params(self, request: LLMRequest) -> dict:
        shared: dict = {"type": "text", "text": request.shared_prefix}
        if self.config.use_prompt_cache and request.cacheable:
            shared["cache_control"] = {"type": "ephemeral"}
        return {
            "model": request.model or self.config.model,
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "system": [shared, {"type": "text", "text": request.role_instruction}],
            "messages": [{"role": "user", "content": request.task}],
        }

    @staticmethod
    def _with_schema_hint(params: dict, output_model: Type[BaseModel]) -> dict:
        params = dict(params)
        schema = json.dumps(output_model.model_json_schema(), ensure_ascii=False)
        messages = list(params["messages"])
        last = dict(messages[-1])
        last["content"] = last["content"] + JSON_ONLY_SUFFIX.format(schema=schema)
        messages[-1] = last
        params["messages"] = messages
        return params


def _first_text(message) -> str:
    for block in message.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


def _usage_of(message) -> TokenUsage:
    usage = getattr(message, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


def _is_schema_unsupported(exc: Exception) -> bool:
    text = str(exc).lower()
    return ("output_format" in text or "output_config" in text
            or "json_schema" in text or "structured output" in text)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _parse_json(raw: str, model_cls: Type[BaseModel], agent: str) -> BaseModel:
    """Parse a model's JSON answer, tolerating a markdown fence.

    Anything looser than this is refused: a half-understood answer is worse
    than a rejected one, and the guard's retry loop exists for exactly this.
    """
    text = raw.strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise LLMError(f"{agent}: response contained no JSON object")
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"{agent}: response was not valid JSON: {exc}") from exc
    try:
        return model_cls.model_validate(payload)
    except ValidationError as exc:
        raise LLMError(f"{agent}: response did not match schema: {exc}") from exc
