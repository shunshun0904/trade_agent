"""Deterministic guard — the first gate, no LLM involved (spec 5)."""

from .deterministic import (  # noqa: F401
    DeterministicGuard,
    check_quoted_indicators,
    extract_quoted_numbers,
)
