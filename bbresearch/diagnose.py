"""損失の要因分解（開発期間のみ）。【確定】事項（D4 エントリー、D5 決済、D6 一次シグナル）の見直し材料。

    python -m bbresearch diagnose --config configs/research.yaml --sets configs/diagnose.yaml --out reports/diagnose

パラメータの組ごとに、一次シグナルのイベントについて次を比べる（すべて開発期間のイベントのみ）。

1. 事後の値動き: close(t0) から h 本後までの対数リターン（h = 1, 4, 8, 16）。全イベント・約定・未約定別。
   約定したイベントのほうが悪ければ、post_only 指値の約定が逆選択になっている（D4）。
2. 取引ごとの損益の分解（同じ約定・決済のタイミングで、費用の扱いだけを変える）:
   - base:        現行（エントリー メイカー、利確 メイカー、損切り・時間切れ 成行 + 滑り）
   - gross:       手数料・滑りなし（シグナルとバリアだけの損益）
   - maker_stop:  損切りを L ちょうどの指値（メイカー）で約定したとみなす（L を飛び越えた下落でも L で
                  約定する楽観的な仮定。時間切れは成行のまま）（D5）
3. 成行エントリー: t0 直後の最初の約定価格 ×（1 + 滑り）でテイカーとして必ず買い、同じバリアで決済する（D4）。
"""
from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bbdata.download import to_utc

from .labeling import barriers, first_exit, label_events, net_return
from .pipeline import barrier_params, load_market
from .search import apply_combo
from .signals import cusum_events, ewm_sigma

log = logging.getLogger(__name__)
HORIZONS = (1, 4, 8, 16)


def _stats(x) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    sd = x.std(ddof=1) if len(x) > 1 else np.nan
    return {"n": int(len(x)), "mean": float(x.mean()), "t": float(x.mean() / sd * np.sqrt(len(x))) if sd > 0 else None,
            "win": float((x > 0).mean())}


def diagnose_set(cfg: dict, market) -> dict:
    s = cfg["signal"]
    start, end = to_utc(cfg["data"]["start"]), to_utc(cfg["data"]["end"])
    holdout_start = end - pd.Timedelta(days=int(cfg["split"]["holdout_days"]))
    bars, tape, tick = market.bars, market.tape, market.tick
    prm = barrier_params(cfg, market.spec)
    sigma = ewm_sigma(bars, s["sigma_span"])
    events = cusum_events(bars, cfg["data"]["pair"], s["sigma_span"], s["k_h"])
    events = events[events["t0"] < holdout_start].reset_index(drop=True)
    lab = label_events(events, bars, tape, prm, tick, sigma)
    ev = events.merge(lab, on="event_id")
    step = bars.index[1] - bars.index[0]
    close = bars["close"]
    pos = close.index.get_indexer(ev["t0"] - step)  # イベント足の位置
    c = close.to_numpy()
    out: dict = {"n_events": int(len(ev)), "fees": {"maker": prm.maker_fee, "taker": prm.taker_fee},
                 "s_slip": prm.s_slip, "signals": {}}
    for sig in ("dip", "breakout"):
        m = (ev["signal_type"] == sig).to_numpy()
        e = ev[m].reset_index(drop=True)
        p = pos[m]
        res: dict = {"n_events": int(len(e)), "fill_rate": float(e["filled"].mean()) if len(e) else None}

        # 1. 事後の値動き
        fwd = {}
        for h in HORIZONS:
            j = p + h
            ok = (p >= 0) & (j < len(c))
            r = np.full(len(e), np.nan)
            r[ok] = np.log(c[j[ok]] / c[p[ok]])
            fwd[h] = {"all": _stats(r), "filled": _stats(r[e["filled"].to_numpy()]),
                      "unfilled": _stats(r[~e["filled"].to_numpy()])}
        res["forward"] = fwd

        # 2. 損益の分解（約定して決済まで判定できた取引）
        f = e[e["filled"] & e["exit_type"].notna()]
        raw_px = np.where(f["exit_type"] == "tp", f["P_x"], f["P_x"] / (1 - prm.s_slip))
        gross = raw_px / f["P_e"] - 1
        maker_stop_px = np.where(f["exit_type"] == "sl", f["L"], f["P_x"])
        maker_stop_fx = np.where(f["exit_type"] == "time", prm.taker_fee, prm.maker_fee)
        ms = [net_return(pe, px, prm.maker_fee, fx) for pe, px, fx in zip(f["P_e"], maker_stop_px, maker_stop_fx)]
        res["pnl"] = {"base": _stats(f["ret_net"]), "gross": _stats(gross), "maker_stop": _stats(ms)}
        res["by_exit"] = {
            k: {"n": int(len(g)), "base_mean": float(g["ret_net"].mean()),
                "gross_mean": float((np.where(g["exit_type"] == "tp", g["P_x"], g["P_x"] / (1 - prm.s_slip))
                                     / g["P_e"] - 1).mean())}
            for k, g in f.groupby("exit_type")
        }
        # sl の約定価格が L をどれだけ下回ったか（飛び越えの大きさ）
        sl = f[f["exit_type"] == "sl"]
        if len(sl):
            gap = (sl["P_x"] / (1 - prm.s_slip)) / sl["L"] - 1
            res["sl_gap"] = {"mean": float(gap.mean()), "p10": float(gap.quantile(0.1))}

        # 3. 成行エントリー
        mk = []
        for ev_row, sig0, pi in zip(e.itertuples(index=False), sigma.to_numpy()[p], p):
            t0_ms = int(ev_row.t0.value // 1_000_000)
            k = int(np.searchsorted(tape.ts, t0_ms, side="left"))
            if k >= len(tape) or not np.isfinite(sig0):
                continue
            p_e = float(tape.price[k]) * (1 + prm.s_slip)
            u, lo = barriers(p_e, sig0, prm.k_up, prm.k_dn, tick)
            t_v = int(tape.ts[k]) + prm.n_v * prm.bar_minutes * 60_000
            ex = first_exit(tape, k, int(tape.ts[k]), u, lo, t_v, prm.s_slip)
            if ex is None:
                continue
            et, _, p_x = ex
            mk.append(net_return(p_e, p_x, prm.taker_fee, prm.maker_fee if et == "tp" else prm.taker_fee))
        res["market_entry"] = _stats(mk)
        out["signals"][sig] = res
    return out


def run_diagnose(cfg: dict, sets: dict, market=None) -> dict:
    market = market or load_market(cfg)
    base = copy.deepcopy(cfg)
    base["pair_spec"] = market.spec
    results = {}
    for name, combo in sets["sets"].items():
        log.info("パラメータの組 %s", name)
        results[name] = {"combo": combo, **diagnose_set(apply_combo(base, combo), market)}
    return {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sets": results}


def run_horizon(cfg: dict, sets: dict, market=None) -> dict:
    market = market or load_market(cfg)
    base = copy.deepcopy(cfg)
    base["pair_spec"] = market.spec
    horizons = sets.get("horizons", [1, 4, 16, 32, 96, 192, 384, 672])
    results = {name: {"combo": combo, **horizon_set(apply_combo(base, combo), market, horizons)}
               for name, combo in sets["sets"].items()}
    return {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sets": results}


def _p(x, pct=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x * 100:+.3f}%" if pct else f"{x:.2f}"


def render(rep: dict) -> str:
    md = [f"# 損失の要因分解（{rep['run_at']}、開発期間のみ）\n"]
    for name, r in rep["sets"].items():
        md.append(f"## {name}: `{r['combo']}`\n")
        md.append(f"手数料率 メイカー {r['fees']['maker']}、テイカー {r['fees']['taker']}、滑り {r['s_slip']}\n")
        for sig, v in r["signals"].items():
            md.append(f"### {sig}（イベント {v['n_events']}、約定率 {_p(v['fill_rate'])[1:]}）\n")
            md.append("事後の対数リターン（close(t0) から h 本後。平均 / t 値）\n")
            md.append("| h | 全イベント | 約定 | 未約定 |\n|---|---|---|---|")
            for h, f in v["forward"].items():
                md.append(f"| {h} | {_p(f['all'].get('mean'))} / {_p(f['all'].get('t'), False)} | "
                          f"{_p(f['filled'].get('mean'))} / {_p(f['filled'].get('t'), False)} | "
                          f"{_p(f['unfilled'].get('mean'))} / {_p(f['unfilled'].get('t'), False)} |")
            md.append("\n取引ごとの損益（平均 / t 値 / 勝率）\n")
            md.append("| 方式 | n | 平均 | t 値 | 勝率 |\n|---|---|---|---|---|")
            for k, s in {**v["pnl"], "market_entry": v["market_entry"]}.items():
                md.append(f"| {k} | {s.get('n', 0)} | {_p(s.get('mean'))} | {_p(s.get('t'), False)} | {_p(s.get('win'))[1:]} |")
            md.append("\n決済種別ごと（現行 / 手数料・滑りなし）: " + ", ".join(
                f"{k} n={x['n']} {_p(x['base_mean'])} / {_p(x['gross_mean'])}" for k, x in v["by_exit"].items()))
            if "sl_gap" in v:
                md.append(f"\n損切りの約定価格の L からの乖離: 平均 {_p(v['sl_gap']['mean'])}、下位 10% {_p(v['sl_gap']['p10'])}")
            md.append("")
    return "\n".join(md) + "\n"


def main(args) -> None:
    import yaml

    from .pipeline import load_config

    cfg = load_config(args.config)
    sets = yaml.safe_load(Path(args.sets).read_text())
    market = load_market(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = run_diagnose(cfg, sets, market)
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    md = render(rep)
    (out / "report.md").write_text(md)
    hz = run_horizon(cfg, sets, market)
    (out / "horizon.json").write_text(json.dumps(hz, ensure_ascii=False, indent=2, default=str))
    md_h = render_horizon(hz)
    (out / "horizon.md").write_text(md_h)
    print(md + "\n" + md_h)


# ---------------------------------------------------------------- ホライズン（案 A の測定）

def _non_overlapping(t: np.ndarray, gap: int) -> np.ndarray:
    """時刻（足の番号）t の昇順の配列から、間隔が gap 以上になるよう前から選んだ位置。"""
    keep, last = [], -np.inf
    for i, v in enumerate(t):
        if v - last >= gap:
            keep.append(i)
            last = v
    return np.array(keep, dtype=int)


def horizon_set(cfg: dict, market, horizons) -> dict:
    """イベント後 h 本の値動きと、費用込みの損益（開発期間のみ）。

    - event:     close(t0) から h 本後の close までの対数リターン（全イベントの平均）
    - baseline:  開発期間のすべての足を起点にした同じ量（相場全体の上昇・下落の影響）
    - excess:    event − baseline
    - t_nonoverlap: 測定期間が重ならないイベントだけを選んだ t 値（重なると t 値が過大になるため）
    - market_rt: t0 直後の約定で成行買い、t0 + h 本の直後の約定で成行売り（テイカー手数料 + 滑りを往復）
    - maker_in:  post_only 買い指値（ラベルと同じ約定判定）で約定した取引を、約定から h 本後に成行売り
    """
    s = cfg["signal"]
    start, end = to_utc(cfg["data"]["start"]), to_utc(cfg["data"]["end"])
    holdout_start = end - pd.Timedelta(days=int(cfg["split"]["holdout_days"]))
    bars, tape = market.bars, market.tape
    prm = barrier_params(cfg, market.spec)
    events = cusum_events(bars, cfg["data"]["pair"], s["sigma_span"], s["k_h"])
    step = bars.index[1] - bars.index[0]
    c = bars["close"].to_numpy()
    lc = np.log(c)
    dev_bars = np.flatnonzero((bars.index >= start) & (bars.index + step <= holdout_start))
    fee_t, slip = prm.taker_fee, prm.s_slip
    step_ms = int(step / pd.Timedelta(milliseconds=1))
    out: dict = {"horizons": list(horizons), "signals": {}}
    for sig in ("dip", "breakout"):
        e = events[(events["signal_type"] == sig) & (events["t0"] < holdout_start)].reset_index(drop=True)
        p = bars.index.get_indexer(e["t0"] - step)
        t0_ms = (e["t0"].astype("int64") // 1_000_000).to_numpy()
        k_in = np.searchsorted(tape.ts, t0_ms, side="left")
        from .labeling import entry_price, fill_time

        fills = []
        for tm, cl in zip(t0_ms, c[p]):
            pe = entry_price(float(cl), prm.delta, market.tick)
            f = fill_time(tape, pe, int(tm), int(tm) + prm.t_fill_min * 60_000, prm.fill_rule)
            fills.append((f[0], pe) if f else (None, pe))
        rows = {}
        for h in horizons:
            ok = (p >= 0) & (p + h < len(c)) & (e["t0"] + h * step <= holdout_start).to_numpy()
            ev_r = lc[p[ok] + h] - lc[p[ok]]
            base_idx = dev_bars[dev_bars + h < len(c)]
            base = lc[base_idx + h] - lc[base_idx]
            sel = _non_overlapping(p[ok], h)
            sub = ev_r[sel] - base.mean()
            t_no = float(sub.mean() / sub.std(ddof=1) * np.sqrt(len(sub))) if len(sub) > 2 and sub.std(ddof=1) > 0 else None
            # 成行の往復
            k_out = np.searchsorted(tape.ts, t0_ms[ok] + h * step_ms, side="left")
            k_i = k_in[ok]
            good = (k_out < len(tape)) & (k_i < len(tape))
            p_in = tape.price[k_i[good]] * (1 + slip)
            p_out = tape.price[k_out[good]] * (1 - slip)
            mrt = (p_out * (1 - fee_t)) / (p_in * (1 + fee_t)) - 1
            # 指値で買い、約定から h 本後に成行売り
            mk = []
            for (tf, pe), okk in zip(fills, ok):
                if tf is None or not okk:
                    continue
                ko = int(np.searchsorted(tape.ts, tf + h * step_ms, side="left"))
                if ko >= len(tape):
                    continue
                px = tape.price[ko] * (1 - slip)
                mk.append(net_return(pe, px, prm.maker_fee, fee_t))
            rows[h] = {
                "event": _stats(ev_r), "baseline_mean": float(base.mean()), "excess_mean": float(ev_r.mean() - base.mean()),
                "n_nonoverlap": int(len(sel)), "t_nonoverlap": t_no,
                "market_rt": _stats(mrt), "maker_in": _stats(mk),
            }
        out["signals"][sig] = rows
    return out


def render_horizon(rep: dict) -> str:
    md = [f"# ホライズン別の優位と費用（{rep['run_at']}、開発期間のみ）\n",
          "event: close(t0) から h 本後までの対数リターン。baseline: 開発期間のすべての足を起点にした同じ量。"
          "excess = event − baseline。t（重なりなし）は測定期間が重ならないイベントだけで計算した excess の t 値。"
          "成行往復・指値買いは手数料と滑りを含む1取引あたりの損益。\n"]
    for name, r in rep["sets"].items():
        md.append(f"## {name}: `{r['combo']}`\n")
        for sig, rows in r["signals"].items():
            md.append(f"### {sig}\n")
            md.append("| h（本） | 時間 | event | baseline | excess | t（重なりなし, n） | 成行往復 | 指値買い→成行売り |\n|---|---|---|---|---|---|---|---|")
            for h, v in rows.items():
                hrs = int(h) * 15 / 60
                md.append(f"| {h} | {hrs:g}h | {_p(v['event'].get('mean'))} | {_p(v['baseline_mean'])} | {_p(v['excess_mean'])} | "
                          f"{_p(v['t_nonoverlap'], False)}（{v['n_nonoverlap']}） | {_p(v['market_rt'].get('mean'))} | "
                          f"{_p(v['maker_in'].get('mean'))} |")
            md.append("")
    return "\n".join(md) + "\n"
