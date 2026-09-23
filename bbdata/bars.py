"""約定履歴から時間足と約定集計を作る。

足の定義: 左閉右開 [t, t+freq)、index は足の開始時刻（UTC）。

列:
    open, high, low, close   約定価格の OHLC（約定なしの足は直前 close で埋める）
    volume, notional         出来高（base 通貨）と売買代金（quote 通貨）
    vwap                     notional / volume（約定なしの足は close）
    n_trades, n_buy, n_sell  約定件数と side 別件数
    buy_volume, sell_volume  side 別出来高
    volume_imbalance         (buy_volume - sell_volume) / volume（約定なしは 0）
    max_buy_price            side=buy の約定の最高値（約定なしは NaN）
    min_sell_price           side=sell の約定の最安値（約定なしは NaN）
    large_volume             数量が large_trade_amount 以上の約定の出来高（指定時のみ）
    is_empty                 約定がなかった足

max_buy_price / min_sell_price は、次の段階（約定考慮トリプルバリア）で
指値が約定したかどうかを近似判定するための列。side がテイカー側を表す、
という前提に立っている（公式ドキュメントには side の意味の明記がない）。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .download import load_transactions, to_utc

log = logging.getLogger(__name__)

SUM_COLUMNS = ["volume", "notional", "n_trades", "n_buy", "buy_volume", "sell_volume"]


def _check_freq(freq: str) -> pd.Timedelta:
    step = pd.Timedelta(freq)
    if step <= pd.Timedelta(0) or pd.Timedelta(days=1) % step != pd.Timedelta(0):
        raise ValueError(f"freq は1日を割り切る長さにしてください: {freq}")
    return step


def aggregate_trades(
    trades: pd.DataFrame,
    freq: str,
    grid_start,
    grid_end,
    large_trade_amount: float | None = None,
) -> pd.DataFrame:
    """[grid_start, grid_end) の格子に約定を集計する（欠損の穴埋めはしない）。

    trades は load_transactions の戻り値（ts, side, price, amount を使う）。
    行は ts 昇順であること（open / close の決定に使う）。
    """
    grid = pd.date_range(to_utc(grid_start), to_utc(grid_end), freq=freq, inclusive="left")

    side = trades["side"].astype("object")
    valid = side.isin(["buy", "sell"])
    if not valid.all():
        log.warning("side が buy/sell 以外の約定を %d 件除外", int((~valid).sum()))
        trades, side = trades[valid], side[valid]

    is_buy = side.eq("buy").to_numpy(dtype=bool)
    price = trades["price"].to_numpy(dtype="float64")
    amount = trades["amount"].to_numpy(dtype="float64")
    work = pd.DataFrame(
        {
            "bucket": trades["ts"].dt.floor(freq).reset_index(drop=True),
            "price": price,
            "amount": amount,
            "notional": price * amount,
            "is_buy": is_buy.astype("int64"),
            "buy_amount": np.where(is_buy, amount, 0.0),
            "sell_amount": np.where(is_buy, 0.0, amount),
            "buy_price": np.where(is_buy, price, np.nan),
            "sell_price": np.where(is_buy, np.nan, price),
        }
    )
    if large_trade_amount is not None:
        work["large_amount"] = np.where(amount >= large_trade_amount, amount, 0.0)

    g = work.groupby("bucket", sort=True)
    bars = pd.DataFrame(
        {
            "open": g["price"].first(),
            "high": g["price"].max(),
            "low": g["price"].min(),
            "close": g["price"].last(),
            "volume": g["amount"].sum(),
            "notional": g["notional"].sum(),
            "n_trades": g["price"].size(),
            "n_buy": g["is_buy"].sum(),
            "buy_volume": g["buy_amount"].sum(),
            "sell_volume": g["sell_amount"].sum(),
            "max_buy_price": g["buy_price"].max(),
            "min_sell_price": g["sell_price"].min(),
        }
    )
    if large_trade_amount is not None:
        bars["large_volume"] = g["large_amount"].sum()

    bars = bars.reindex(grid)
    fill_zero = SUM_COLUMNS + (["large_volume"] if large_trade_amount is not None else [])
    bars[fill_zero] = bars[fill_zero].fillna(0.0)
    bars[["n_trades", "n_buy"]] = bars[["n_trades", "n_buy"]].astype("int64")
    return bars


def finalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """約定なしの足を埋め、派生列を加える。"""
    out = bars.copy()
    empty = out["n_trades"].eq(0)
    out["close"] = out["close"].ffill()  # 過去方向のみの穴埋め（先読みなし）
    for col in ("open", "high", "low"):
        out[col] = out[col].where(~empty, out["close"])
    has_vol = out["volume"] > 0
    out["vwap"] = (out["notional"] / out["volume"].where(has_vol)).fillna(out["close"])
    out["volume_imbalance"] = (
        (out["buy_volume"] - out["sell_volume"]) / out["volume"].where(has_vol)
    ).fillna(0.0)
    out["n_sell"] = out["n_trades"] - out["n_buy"]
    out["is_empty"] = empty
    out.index.name = "ts"

    n_lead = int(out["close"].isna().sum())
    if n_lead:
        log.warning("期間先頭の %d 本は直前の約定がなく close が NaN です", n_lead)
    return out


def build_bars(
    root: str | Path,
    pair: str,
    start,
    end,
    freq: str = "15min",
    large_trade_amount: float | None = None,
    chunk_days: int = 7,
) -> pd.DataFrame:
    """保存済みの約定から [start, end)（UTC）の足を作る。

    メモリを抑えるため chunk_days 日ずつ読み込んで集計する。
    large_trade_amount は固定値で指定する（全期間の分位点などで決めると先読みになる）。
    """
    start, end = to_utc(start), to_utc(end)
    _check_freq(freq)
    if start != start.floor(freq) or end != end.floor(freq):
        raise ValueError("start / end は freq の境界に揃えてください")

    chunks = []
    cur = start
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=chunk_days), end)
        trades = load_transactions(root, pair, cur, nxt)
        chunks.append(aggregate_trades(trades, freq, cur, nxt, large_trade_amount))
        cur = nxt
    return finalize_bars(pd.concat(chunks))


def bars_path(root: str | Path, pair: str, freq: str) -> Path:
    return Path(root) / "bars" / pair / f"{freq}.parquet"


def save_bars(bars: pd.DataFrame, root: str | Path, pair: str, freq: str) -> Path:
    path = bars_path(root, pair, freq)
    path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(path)
    return path
