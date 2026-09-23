"""Phase 3: 約定を考慮したトリプルバリア。

イベントごとに post_only 買い指値を出した場合の約定と決済を、約定単位のデータで再現する。
約定判定（fill_time）と決済判定（first_exit）は本番（Phase 7）でも同じ関数を使う前提で、
配列を受け取る純粋関数として書いている。

前提と制約（SPEC.md §5 Phase 3）:
- 発注数量の全量が P_e で約定すると仮定する（発注量が出来高に比べて十分小さい場合のみ成立）。
- 約定の検知から利確指値を出すまでの遅延は無視する。
- 約定の side はテイカー側の方向を表すと仮定する（V2 で検証）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

LABEL_COLUMNS = [
    "event_id", "filled", "t_f", "P_e", "U", "L", "t_v", "exit_type", "t_x", "P_x",
    "f_e", "f_x", "ret_net", "y",
]


@dataclass(frozen=True)
class TradeTape:
    """時刻順に並んだ約定。ts は UnixTime ミリ秒。"""

    ts: np.ndarray        # int64
    is_buy: np.ndarray    # bool（side == "buy"）
    price: np.ndarray     # float64
    amount: np.ndarray | None = None  # float64（価格帯別出来高に使う。約定・決済の判定には使わない）

    @classmethod
    def from_frame(cls, trades: pd.DataFrame) -> "TradeTape":
        """load_transactions の戻り値（executed_at, side, price）から作る。"""
        t = trades.sort_values(["executed_at", "transaction_id"]) if "transaction_id" in trades else trades
        return cls(
            ts=t["executed_at"].to_numpy(dtype="int64"),
            is_buy=t["side"].astype("object").eq("buy").to_numpy(dtype=bool),
            price=t["price"].to_numpy(dtype="float64"),
            amount=t["amount"].to_numpy(dtype="float64") if "amount" in t else None,
        )

    @classmethod
    def load(cls, root, pair: str, start, end, chunk_days: int = 7) -> "TradeTape":
        """保存済みの約定を chunk_days 日ずつ読み、配列だけを残す（メモリ節約）。"""
        from bbdata.download import load_transactions, to_utc

        start, end = to_utc(start), to_utc(end)
        parts = []
        cur = start
        while cur < end:
            nxt = min(cur + pd.Timedelta(days=chunk_days), end)
            parts.append(cls.from_frame(load_transactions(root, pair, cur, nxt)))
            cur = nxt
        return cls(
            ts=np.concatenate([p.ts for p in parts]) if parts else np.array([], dtype="int64"),
            is_buy=np.concatenate([p.is_buy for p in parts]) if parts else np.array([], dtype=bool),
            price=np.concatenate([p.price for p in parts]) if parts else np.array([], dtype="float64"),
            amount=np.concatenate([p.amount for p in parts]) if parts else np.array([], dtype="float64"),
        )

    def __len__(self) -> int:
        return len(self.ts)


@dataclass(frozen=True)
class BarrierParams:
    delta: float = 0.0          # エントリー価格の close からの下げ幅（比率）
    t_fill_min: int = 15        # エントリー指値の待機時間（分）
    k_up: float = 2.0
    k_dn: float = 2.0
    n_v: int = 8                # 時間切れまでの本数
    bar_minutes: int = 15
    s_slip: float = 0.0005      # 成行決済の滑り（比率）
    fill_rule: str = "strict"   # strict / touch
    maker_fee: float = -0.0002  # メイカー手数料率（報酬は負）。V6 で符号を確定させる
    taker_fee: float = 0.0012   # テイカー手数料率


# ------------------------------------------------------------------ 価格の丸め

def _round_to_tick(x: float, tick: float, up: bool) -> float:
    # 浮動小数の誤差で1呼値ずれないよう、比を丸めてから切り捨て・切り上げる
    q = round(x / tick, 9)
    return (math.ceil(q) if up else math.floor(q)) * tick


def floor_price(x: float, tick: float) -> float:
    return _round_to_tick(x, tick, up=False)


def ceil_price(x: float, tick: float) -> float:
    return _round_to_tick(x, tick, up=True)


def entry_price(close_t0: float, delta: float, tick: float) -> float:
    """P_e = close(t0) × (1 − δ) を呼値に合わせて切り捨てる。"""
    return floor_price(close_t0 * (1.0 - delta), tick)


def barriers(p_e: float, sigma0: float, k_up: float, k_dn: float, tick: float) -> tuple[float, float]:
    """(U, L)。U は呼値に合わせて切り上げる。L は判定の閾値なので丸めない。"""
    return ceil_price(p_e * (1.0 + k_up * sigma0), tick), p_e * (1.0 - k_dn * sigma0)


# ------------------------------------------------------------------ 約定・決済の判定

def fill_time(tape: TradeTape, p_e: float, t_start: int, t_end: int, rule: str = "strict") -> tuple[int, int] | None:
    """[t_start, t_end) で買い指値 p_e が約定した時刻と、その約定のインデックス。

    strict: side == sell かつ price < p_e（同値はキュー位置が分からないので約定とみなさない）
    touch:  side == sell かつ price <= p_e
    """
    i0 = int(np.searchsorted(tape.ts, t_start, side="left"))
    i1 = int(np.searchsorted(tape.ts, t_end, side="left"))
    if i1 <= i0:
        return None
    px = tape.price[i0:i1]
    sell = ~tape.is_buy[i0:i1]
    if rule == "strict":
        hit = sell & (px < p_e)
    elif rule == "touch":
        hit = sell & (px <= p_e)
    else:
        raise ValueError(f"未知の fill_rule: {rule}")
    j = np.flatnonzero(hit)
    if len(j) == 0:
        return None
    k = i0 + int(j[0])
    return int(tape.ts[k]), k


def first_exit(
    tape: TradeTape, start_idx: int, t_f: int, u: float, l: float, t_v: int, s_slip: float
) -> tuple[str, int, float] | None:
    """t_f より後の約定を時刻順に調べ、最初に該当した決済条件を返す。

    戻り値は (exit_type, t_x, P_x)。データが t_v に届かず判定できない場合は None。
    - tp:   side == buy かつ price > U。決済価格は U（メイカー）
    - sl:   price <= L（side は問わない）。決済価格はその約定価格 × (1 − s_slip)
    - time: t_v に達した。決済価格は t_v 直前の約定価格 × (1 − s_slip)
    """
    i0 = int(np.searchsorted(tape.ts, t_f, side="right", sorter=None))
    i0 = max(i0, start_idx + 1)
    i1 = int(np.searchsorted(tape.ts, t_v, side="left"))
    if i1 > i0:
        px = tape.price[i0:i1]
        tp = tape.is_buy[i0:i1] & (px > u)
        sl = px <= l
        j = np.flatnonzero(tp | sl)
        if len(j):
            k = i0 + int(j[0])
            if sl[j[0]]:
                return "sl", int(tape.ts[k]), float(tape.price[k]) * (1.0 - s_slip)
            return "tp", int(tape.ts[k]), float(u)
    if len(tape) == 0 or tape.ts[-1] < t_v:
        return None  # t_v までのデータがない
    last = i1 - 1  # t_v より前の最後の約定（約定そのものを含む）
    return "time", int(t_v), float(tape.price[last]) * (1.0 - s_slip)


def net_return(p_e: float, p_x: float, f_e: float, f_x: float) -> float:
    """ret_net = (P_x (1 − f_x) − P_e (1 + f_e)) / (P_e (1 + f_e))"""
    cost = p_e * (1.0 + f_e)
    return (p_x * (1.0 - f_x) - cost) / cost


# ------------------------------------------------------------------ イベント単位

def label_event(
    tape: TradeTape, t0_ms: int, close_t0: float, sigma0: float, prm: BarrierParams, tick: float
) -> dict:
    p_e = entry_price(close_t0, prm.delta, tick)
    out: dict = {"P_e": p_e, "filled": False}
    f = fill_time(tape, p_e, t0_ms, t0_ms + prm.t_fill_min * 60_000, prm.fill_rule)
    if f is None:
        return out
    t_f, idx = f
    u, l = barriers(p_e, sigma0, prm.k_up, prm.k_dn, tick)
    t_v = t_f + prm.n_v * prm.bar_minutes * 60_000
    out.update(filled=True, t_f=t_f, U=u, L=l, t_v=t_v)
    ex = first_exit(tape, idx, t_f, u, l, t_v, prm.s_slip)
    if ex is None:
        return out
    exit_type, t_x, p_x = ex
    f_e = prm.maker_fee
    f_x = prm.maker_fee if exit_type == "tp" else prm.taker_fee
    r = net_return(p_e, p_x, f_e, f_x)
    out.update(exit_type=exit_type, t_x=t_x, P_x=p_x, f_e=f_e, f_x=f_x, ret_net=r, y=int(r > 0))
    return out


def label_events(
    events: pd.DataFrame, bars: pd.DataFrame, tape: TradeTape, prm: BarrierParams, tick: float, sigma: pd.Series
) -> pd.DataFrame:
    """labels テーブル。時刻列は UTC の Timestamp に戻す。

    close(t0) と σ_0 は、イベント足（終了時刻が t0 の足）の値を使う。
    """
    step = bars.index[1] - bars.index[0]
    rows = []
    for ev in events.itertuples(index=False):
        bar_start = ev.t0 - step
        close_t0 = float(bars.at[bar_start, "close"])
        sigma0 = float(sigma.at[bar_start])
        t0_ms = int(ev.t0.value // 1_000_000)
        rec = {"event_id": ev.event_id, **label_event(tape, t0_ms, close_t0, sigma0, prm, tick)}
        rows.append(rec)
    lab = pd.DataFrame(rows).reindex(columns=LABEL_COLUMNS)
    for c in ("t_f", "t_v", "t_x"):
        lab[c] = pd.to_datetime(lab[c], unit="ms", utc=True)
    lab["filled"] = lab["filled"].astype(bool)
    return lab
