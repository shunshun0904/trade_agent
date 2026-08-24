"""Indicators, checked against worked examples rather than themselves."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from trade_agent.data.indicators import (
    atr,
    bollinger,
    change_pct,
    ema,
    realized_vol_pct,
    rsi,
    sma,
    volume_ratio,
    vwap,
)
from trade_agent.models.market import Candle

E = Decimal

# Wilder's own worked example (the series used in New Concepts in Technical
# Trading Systems and reproduced by every reference implementation).
WILDER = [E(x) for x in
          "44.34 44.09 44.15 43.61 44.33 44.83 45.10 45.42 45.84 46.08 45.89 "
          "46.03 45.61 46.28 46.28 46.00 46.03 46.41 46.22 45.64".split()]


def test_rsi_matches_the_reference_series():
    assert round(rsi(WILDER[:15], 14), 2) == E("70.46")
    assert round(rsi(WILDER[:16], 14), 2) == E("66.25")
    assert round(rsi(WILDER[:17], 14), 2) == E("66.48")


def test_rsi_is_none_without_enough_history():
    assert rsi(WILDER[:5], 14) is None


def test_rsi_saturates_cleanly():
    rising = [E(100) + E(i) for i in range(20)]
    assert rsi(rising, 14) == E(100)


def test_sma_and_ema():
    assert sma([E(1), E(2), E(3)], 3) == E(2)
    assert sma([E(1), E(2)], 3) is None
    # EMA seeds on the SMA, so with a constant series it stays constant.
    assert ema([E(5)] * 10, 5) == E(5)


def test_bollinger_bands_straddle_the_mean():
    values = [E(100), E(102), E(98), E(101), E(99)] * 4
    upper, lower, width = bollinger(values, 20)
    mean = sma(values, 20)
    assert lower < mean < upper
    assert width > 0


def test_atr_of_a_constant_range():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = [Candle(open=E(100), high=E(105), low=E(95), close=E(100),
                      volume=E(1), opened_at=base + timedelta(hours=i))
               for i in range(30)]
    assert atr(candles, 14) == E(10)


def test_vwap_weights_by_volume():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = [
        Candle(open=E(100), high=E(100), low=E(100), close=E(100), volume=E(1),
               opened_at=base),
        Candle(open=E(200), high=E(200), low=E(200), close=E(200), volume=E(3),
               opened_at=base + timedelta(hours=1)),
    ]
    assert vwap(candles) == E(175)  # (100*1 + 200*3) / 4


def test_vwap_is_none_without_volume():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert vwap([Candle(open=E(1), high=E(1), low=E(1), close=E(1), volume=E(0),
                        opened_at=base)]) is None


def test_change_pct():
    assert change_pct([E(100), E(110)], 1) == E(10)
    assert change_pct([E(100)], 5) is None


def test_volume_ratio_flags_a_spike():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = [Candle(open=E(1), high=E(1), low=E(1), close=E(1), volume=E(2),
                      opened_at=base + timedelta(hours=i)) for i in range(21)]
    candles.append(Candle(open=E(1), high=E(1), low=E(1), close=E(1),
                          volume=E(10), opened_at=base + timedelta(hours=21)))
    assert volume_ratio(candles, 20) == E(5)


def test_realized_vol_is_zero_for_a_flat_series():
    assert realized_vol_pct([E(100)] * 30, 20) == E(0)


def test_the_synthetic_market_is_reproducible():
    """`--local` must replay identically, or a reported cycle cannot be
    re-examined."""
    from datetime import date

    from trade_agent.exchange.synthetic import SyntheticMarket
    from trade_agent.timeutil import UTC

    now = datetime(2026, 3, 2, 12, tzinfo=UTC)
    a = SyntheticMarket(now=lambda: now)
    b = SyntheticMarket(now=lambda: now)
    assert a.get_candles("1hour", date(2026, 3, 2)) == \
        b.get_candles("1hour", date(2026, 3, 2))
    assert a.get_ticker()["last"] == b.get_ticker()["last"]


def test_the_synthetic_market_stays_ordered_and_bounded():
    from datetime import date

    from trade_agent.exchange.synthetic import SyntheticMarket
    from trade_agent.timeutil import UTC

    now = datetime(2026, 3, 2, 23, 59, tzinfo=UTC)
    candles = SyntheticMarket(now=lambda: now).get_candles("1hour",
                                                           date(2026, 3, 2))
    assert len(candles) == 24
    assert candles == sorted(candles, key=lambda c: c.opened_at)
    for candle in candles:
        assert candle.low <= candle.open <= candle.high
        assert candle.low <= candle.close <= candle.high
        assert candle.volume > 0
