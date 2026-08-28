"""Time handling.

Spec 17.3: everything internal is UTC; JST appears only at display time and in
the EventBridge schedules. A `Clock` is injected everywhere rather than calling
`datetime.now()` directly, so the 72-hour boredom rule and the daily counters
can be tested without waiting three days.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone

JST = timezone(timedelta(hours=9))
UTC = timezone.utc


class Clock:
    """Wall clock. Always returns timezone-aware UTC datetimes."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    def timestamp_ms(self) -> int:
        return int(self.now().timestamp() * 1000)


@dataclass
class FrozenClock(Clock):
    """Test clock. `advance()` moves it forward."""

    current: datetime = field(default_factory=lambda: datetime(2026, 1, 1, tzinfo=UTC))

    def now(self) -> datetime:
        return self.current

    def advance(self, **kwargs) -> datetime:
        self.current = self.current + timedelta(**kwargs)
        return self.current


def to_jst(dt: datetime) -> datetime:
    return _aware(dt).astimezone(JST)


def to_utc(dt: datetime) -> datetime:
    return _aware(dt).astimezone(UTC)


def _aware(dt: datetime) -> datetime:
    """Treat a naive datetime as UTC rather than as local time."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def iso(dt: datetime) -> str:
    return _aware(dt).astimezone(UTC).isoformat().replace("+00:00", "Z")


def iso_jst(dt: datetime) -> str:
    """Format in JST, keeping the +09:00 offset visible.

    `iso()` normalises to UTC, so `iso(to_jst(x))` would quietly hand back UTC
    under a JST-sounding field name. Owner-facing timestamps use this instead.
    """
    return to_jst(dt).isoformat()


def parse_iso(value: str) -> datetime:
    return _aware(datetime.fromisoformat(value.replace("Z", "+00:00")))


def from_ms(ms: int | float) -> datetime:
    return datetime.fromtimestamp(float(ms) / 1000.0, tz=UTC)


def to_ms(dt: datetime) -> int:
    return int(_aware(dt).timestamp() * 1000)


def jst_date(dt: datetime) -> date:
    """The JST calendar date a UTC instant falls on.

    Daily counters (loss limit, debate limit, equity curve) roll over at JST
    midnight because that is the day the owner lives in.
    """
    return to_jst(dt).date()


def jst_date_str(dt: datetime) -> str:
    return jst_date(dt).isoformat()


def jst_month_str(dt: datetime) -> str:
    """`YYYY-MM` in JST — the bucket monthly budget and probe losses roll up to."""
    return to_jst(dt).strftime("%Y-%m")


def jst_days_remaining_in_month(dt: datetime) -> int:
    """Days left in the JST month, counting today. Never below 1.

    The denominator when pacing the monthly LLM budget across days: today is
    included because today's spending is still ahead of us, and the floor of 1
    keeps the last day of the month from dividing by zero.
    """
    import calendar

    today = jst_date(dt)
    _, days_in_month = calendar.monthrange(today.year, today.month)
    return max(1, days_in_month - today.day + 1)


def parse_hhmm(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


def crossed_jst_time(previous: datetime | None, now: datetime, hhmm: str,
                     window_minutes: int) -> bool:
    """True when `now` is inside the `window_minutes` slot that starts at `hhmm` JST
    and no run has happened inside that same slot yet.

    The window exists because EventBridge fires on a cadence, not at an exact
    instant: a 30-minute screener cannot be relied on to wake at exactly 09:00.
    """
    target = parse_hhmm(hhmm)
    local = to_jst(now)
    slot_start = local.replace(hour=target.hour, minute=target.minute,
                               second=0, microsecond=0)
    if not (slot_start <= local < slot_start + timedelta(minutes=window_minutes)):
        return False
    return previous is None or to_jst(previous) < slot_start
