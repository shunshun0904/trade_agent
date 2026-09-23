"""深い指値で買い、短く保有し、売りも指値にする決済（案 B の続き、2026-09-23 オーナー指示）。

約定と決済は約定単位のデータで再現する（判定は labeling.fill_time / first_exit を使う）。

1. エントリー: close(t0) × (1 − delta_sigma × σ_0) を呼値に切り捨てた post_only 買い指値。
   [t0, t0 + T_fill) の約定で fill_time と同じ規則で約定を判定する。
2. 保有中 [t_f, t_v): 利確は U = P_e (1 + k_up σ_0) の post_only 売り指値（メイカー）、
   損切りは L = P_e (1 − k_dn σ_0) 以下の約定で成行（テイカー + 滑り。D5 のまま、オーナー決定）。
   t_v = t_f + n_v 本。
3. 時間切れ: t_v に、直前の約定価格 + 1 呼値の post_only 売り指値を出す（時間切れの決済を指値にする、D5 の変更）。
   [t_v, t_v + grace) に、その価格を上回る買いの約定があれば約定（メイカー）。その前に L 以下の約定が
   あれば成行の損切り。どちらもなければ t_v + grace で成行（テイカー + 滑り）。

exit_type: tp / sl / time_maker / time_taker
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .labeling import LABEL_COLUMNS, TradeTape, barriers, fill_time, first_exit, floor_price, net_return


@dataclass(frozen=True)
class MakerExitParams:
    delta_sigma: float = 2.0     # エントリー指値を close から何 σ 下に置くか
    t_fill_min: int = 15
    k_up: float = 1.0
    k_dn: float = 4.0
    n_v: int = 4                 # 時間切れまでの本数
    grace_min: int = 15          # 時間切れの売り指値を待つ時間（分）
    bar_minutes: int = 15
    s_slip: float = 0.0005
    fill_rule: str = "strict"
    maker_fee: float = 0.0
    taker_fee: float = 0.001


def label_event_maker(tape: TradeTape, t0_ms: int, close_t0: float, sigma0: float, prm: MakerExitParams,
                      tick: float) -> dict:
    p_e = floor_price(close_t0 * (1.0 - prm.delta_sigma * sigma0), tick)
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
    if exit_type == "time":
        # t_v 直前の約定価格 + 1 呼値に post_only 売り指値。t_v 以降の約定で判定する
        last = int(np.searchsorted(tape.ts, t_v, side="left")) - 1
        x = float(tape.price[last]) + tick
        t_end = t_v + prm.grace_min * 60_000
        ex2 = first_exit(tape, last, t_v - 1, x, l, t_end, prm.s_slip)
        if ex2 is None:
            return out
        e2, t_x, p_x = ex2
        exit_type = {"tp": "time_maker", "sl": "sl", "time": "time_taker"}[e2]
    f_x = prm.maker_fee if exit_type in ("tp", "time_maker") else prm.taker_fee
    r = net_return(p_e, p_x, prm.maker_fee, f_x)
    out.update(exit_type=exit_type, t_x=t_x, P_x=p_x, f_e=prm.maker_fee, f_x=f_x, ret_net=r, y=int(r > 0))
    return out


def label_events_maker(events: pd.DataFrame, bars: pd.DataFrame, tape: TradeTape, prm: MakerExitParams,
                       tick: float, sigma: pd.Series) -> pd.DataFrame:
    """labeling.label_events と同じ形式の labels テーブル。close(t0) と σ_0 はイベント足の値。"""
    step = bars.index[1] - bars.index[0]
    rows = []
    for ev in events.itertuples(index=False):
        bs = ev.t0 - step
        rows.append({"event_id": ev.event_id, **label_event_maker(
            tape, int(ev.t0.value // 1_000_000), float(bars.at[bs, "close"]), float(sigma.at[bs]), prm, tick)})
    lab = pd.DataFrame(rows).reindex(columns=LABEL_COLUMNS)
    for c in ("t_f", "t_v", "t_x"):
        lab[c] = pd.to_datetime(lab[c], unit="ms", utc=True)
    lab["filled"] = lab["filled"].astype(bool)
    return lab
