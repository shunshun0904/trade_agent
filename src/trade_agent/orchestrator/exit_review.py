"""When to spend an LLM call on an open position (docs/OPEN-QUESTIONS.md D-1).

The mirror of `screening.evaluate_triggers`, for the other half of the trade.
Zero LLM calls, zero yen: it only decides whether a review is worth paying for.

The cadence is doing two jobs at once, and the second is the important one.
Cost is the obvious constraint. The other is that every review is a chance to
talk the position out of a plan that was made with more care than a five-minute
check can muster — so the cheapest protection against over-managing an exit is
simply to ask less often. `min_minutes` is a floor, not a schedule: a quiet
market falls through to `max_idle_minutes` and gets reviewed once every four
hours.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from ..config import Config
from ..models.market import MarketSnapshot
from ..models.state import SystemState
from ..money import ZERO, dec


@dataclass
class ReviewDecision:
    should_review: bool
    reason: str

    def __bool__(self) -> bool:
        return self.should_review


def should_review(config: Config, state: SystemState,
                  snapshot: MarketSnapshot | None, now: datetime, *,
                  cost_meter=None) -> ReviewDecision:
    """Decide whether to review the open position right now.

    Ordered cheapest-first, like the screener: a disabled feature or a spent
    budget never reads an indicator.
    """
    cfg = config.exit_review
    if not cfg.enabled:
        return ReviewDecision(False, "exit review is off")

    position = state.open_position
    if position is None:
        return ReviewDecision(False, "no open position")
    if position.exit_order_id:
        # An exit is already in flight. Re-deciding now could only fight it.
        return ReviewDecision(False, "an exit order is already working")

    if position.review_count >= cfg.max_reviews_per_position:
        return ReviewDecision(
            False,
            f"review limit reached ({position.review_count}/"
            f"{cfg.max_reviews_per_position}) for this position")

    if cost_meter is not None:
        allowance = cost_meter.daily_allowance_jpy(state.monthly.llm_cost_jpy, now)
        spent_today = dec(state.daily.llm_cost_jpy)
        if spent_today >= allowance:
            return ReviewDecision(
                False,
                f"today's LLM allowance is spent "
                f"({spent_today:.1f}/{allowance:.1f} JPY)")

    last_at = position.last_review_at or position.opened_at
    elapsed = now - last_at
    if elapsed < timedelta(minutes=cfg.min_minutes):
        return ReviewDecision(
            False,
            f"only {elapsed.total_seconds() / 60:.0f} minutes since the last "
            f"review (floor {cfg.min_minutes})")

    if elapsed >= timedelta(minutes=cfg.max_idle_minutes):
        return ReviewDecision(
            True, f"{elapsed.total_seconds() / 3600:.1f} hours without a review")

    moved = _price_move(position, snapshot)
    threshold = _atr_threshold(config, snapshot)
    if moved is None or threshold is None:
        return ReviewDecision(False, "no ATR available to size the move against")
    if moved >= threshold:
        return ReviewDecision(
            True, f"price moved {moved} against a {threshold} threshold "
                  f"({cfg.atr_multiple}x ATR)")

    return ReviewDecision(
        False, f"price moved {moved}, below the {threshold} threshold")


def _price_move(position, snapshot: MarketSnapshot | None) -> Decimal | None:
    """How far price has travelled since the last review, in yen.

    Measured from the last review rather than from entry: the question is
    whether anything has happened that the previous look did not already
    account for.
    """
    if snapshot is None:
        return None
    reference = position.last_review_price or position.entry_price
    return abs(dec(snapshot.last_price) - dec(reference))


def _atr_threshold(config: Config, snapshot: MarketSnapshot | None) -> Decimal | None:
    if snapshot is None or snapshot.indicators.atr is None:
        return None
    atr = dec(snapshot.indicators.atr)
    if atr <= ZERO:
        return None
    return atr * config.exit_review.atr_multiple
