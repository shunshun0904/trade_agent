"""Technical indicators, computed exactly.

Written in plain Python over Decimals rather than pandas/TA-Lib. Two reasons:
a Lambda package stays small and cold starts stay short, and every value is
exact — the guard compares an agent's quoted figure against these numbers, so
a float rounding artefact would show up as a fabrication (spec 5).

Each function returns None when there is not enough history. "Unknown" is a
legitimate answer that the snapshot passes through to the agents; inventing a
value would be worse than admitting the gap.
"""

from __future__ import annotations

from decimal import Decimal

from ..models.market import Candle
from ..money import ZERO, dec

Num = Decimal | None


def sma(values: list[Decimal], period: int) -> Num:
    if period <= 0 or len(values) < period:
        return None
    window = values[-period:]
    return sum(window, ZERO) / Decimal(period)


def ema(values: list[Decimal], period: int) -> Num:
    """Seeded with the SMA of the first `period` values, then smoothed."""
    if period <= 0 or len(values) < period:
        return None
    k = Decimal(2) / Decimal(period + 1)
    current = sum(values[:period], ZERO) / Decimal(period)
    for value in values[period:]:
        current = (value - current) * k + current
    return current


def rsi(values: list[Decimal], period: int = 14) -> Num:
    """Wilder's RSI."""
    if period <= 0 or len(values) < period + 1:
        return None
    gains: list[Decimal] = []
    losses: list[Decimal] = []
    for prev, cur in zip(values, values[1:]):
        change = cur - prev
        gains.append(max(change, ZERO))
        losses.append(max(-change, ZERO))
    avg_gain = sum(gains[:period], ZERO) / Decimal(period)
    avg_loss = sum(losses[:period], ZERO) / Decimal(period)
    for gain, loss in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * Decimal(period - 1) + gain) / Decimal(period)
        avg_loss = (avg_loss * Decimal(period - 1) + loss) / Decimal(period)
    if avg_loss == 0:
        return Decimal(100) if avg_gain > 0 else Decimal(50)
    rs = avg_gain / avg_loss
    return Decimal(100) - (Decimal(100) / (Decimal(1) + rs))


def true_ranges(candles: list[Candle]) -> list[Decimal]:
    out: list[Decimal] = []
    for prev, cur in zip(candles, candles[1:]):
        out.append(max(cur.high - cur.low,
                       abs(cur.high - prev.close),
                       abs(cur.low - prev.close)))
    return out


def atr(candles: list[Candle], period: int = 14) -> Num:
    """Wilder's ATR."""
    ranges = true_ranges(candles)
    if len(ranges) < period or period <= 0:
        return None
    current = sum(ranges[:period], ZERO) / Decimal(period)
    for value in ranges[period:]:
        current = (current * Decimal(period - 1) + value) / Decimal(period)
    return current


def stdev(values: list[Decimal], period: int) -> Num:
    if period <= 1 or len(values) < period:
        return None
    window = values[-period:]
    mean = sum(window, ZERO) / Decimal(period)
    variance = sum(((v - mean) ** 2 for v in window), ZERO) / Decimal(period)
    return variance.sqrt()


def bollinger(values: list[Decimal], period: int = 20,
              width: Decimal = Decimal(2)) -> tuple[Num, Num, Num]:
    """Returns (upper, lower, band width as a percentage of the mid band)."""
    mid = sma(values, period)
    sd = stdev(values, period)
    if mid is None or sd is None:
        return None, None, None
    upper = mid + width * sd
    lower = mid - width * sd
    band_pct = (upper - lower) / mid * Decimal(100) if mid else None
    return upper, lower, band_pct


def vwap(candles: list[Candle]) -> Num:
    """Volume-weighted average of candle typical prices."""
    volume = sum((c.volume for c in candles), ZERO)
    if volume <= 0:
        return None
    total = sum((((c.high + c.low + c.close) / Decimal(3)) * c.volume
                 for c in candles), ZERO)
    return total / volume


def change_pct(values: list[Decimal], periods: int) -> Num:
    if len(values) < periods + 1:
        return None
    start = values[-(periods + 1)]
    if start == 0:
        return None
    return (values[-1] - start) / start * Decimal(100)


def volume_ratio(candles: list[Candle], lookback: int = 20) -> Num:
    """Latest candle's volume against the mean of the `lookback` before it."""
    if len(candles) < lookback + 1:
        return None
    prior = candles[-(lookback + 1):-1]
    mean = sum((c.volume for c in prior), ZERO) / Decimal(len(prior))
    if mean <= 0:
        return None
    return candles[-1].volume / mean


def realized_vol_pct(values: list[Decimal], period: int = 20) -> Num:
    """Standard deviation of simple returns over `period`, in percent.

    Simple rather than log returns: over 5-minute and 1-hour bars the
    difference is immaterial and simple returns stay exact in Decimal.
    """
    if len(values) < period + 1:
        return None
    returns: list[Decimal] = []
    for prev, cur in zip(values[-(period + 1):], values[-period:]):
        if prev == 0:
            return None
        returns.append((cur - prev) / prev)
    mean = sum(returns, ZERO) / Decimal(len(returns))
    variance = sum(((r - mean) ** 2 for r in returns), ZERO) / Decimal(len(returns))
    return variance.sqrt() * Decimal(100)


def highest(candles: list[Candle], count: int) -> Num:
    if not candles:
        return None
    return max(c.high for c in candles[-count:])


def lowest(candles: list[Candle], count: int) -> Num:
    if not candles:
        return None
    return min(c.low for c in candles[-count:])


def closes(candles: list[Candle]) -> list[Decimal]:
    return [dec(c.close) for c in candles]
