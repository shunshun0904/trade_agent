"""Decimal money and quantity helpers.

Every price, quantity and JPY amount in this system is a `Decimal`. Floats are
allowed only inside indicator maths, where the result is a statistic rather
than something the exchange will execute against.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, ROUND_UP

ZERO = Decimal("0")


def dec(value) -> Decimal:
    """Coerce to Decimal without going through binary float where avoidable."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(repr(value))
    return Decimal(str(value))


def quantize_price(price, digits: int) -> Decimal:
    """Round a price to the pair's tick size (bitbank `price_digits`)."""
    exp = Decimal(1).scaleb(-digits)
    return dec(price).quantize(exp, rounding=ROUND_HALF_UP)


def floor_to_lot(qty, lot) -> Decimal:
    """Largest multiple of `lot` that is <= qty. Never rounds an order up."""
    lot = dec(lot)
    if lot <= 0:
        raise ValueError("lot size must be positive")
    units = (dec(qty) / lot).to_integral_value(rounding=ROUND_DOWN)
    return (units * lot).normalize() + ZERO


def is_lot_multiple(qty, lot) -> bool:
    lot = dec(lot)
    if lot <= 0:
        return False
    return dec(qty) % lot == 0


def jpy(value) -> Decimal:
    """Round to whole yen. Spec 5 compares risk in 1-yen units."""
    return dec(value).quantize(Decimal("1"), rounding=ROUND_HALF_UP)


def jpy_ceil(value) -> Decimal:
    return dec(value).quantize(Decimal("1"), rounding=ROUND_UP)


def pct(part, whole) -> Decimal:
    """`part` as a percentage of `whole`; 0 when whole is 0."""
    whole = dec(whole)
    if whole == 0:
        return ZERO
    return dec(part) / whole * Decimal(100)


def apply_pct(value, percent) -> Decimal:
    """value * (1 + percent/100)."""
    return dec(value) * (Decimal(1) + dec(percent) / Decimal(100))


def deviation_pct(a, b) -> Decimal:
    """Absolute relative difference of `a` from `b`, in percent."""
    b = dec(b)
    if b == 0:
        return ZERO
    return abs((dec(a) - b) / b) * Decimal(100)


def fee_jpy(notional, rate) -> Decimal:
    """Fee in JPY for a notional amount. Negative for the maker rebate."""
    return dec(notional) * dec(rate)


def to_str(value: Decimal) -> str:
    """Exchange-safe string: fixed notation, no exponent."""
    d = dec(value)
    sign, digits, exponent = d.as_tuple()
    if isinstance(exponent, int) and exponent > 0:
        d = d.quantize(Decimal(1))
    return format(d, "f")
