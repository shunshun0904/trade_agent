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


def run_diagnose(cfg: dict, sets: dict) -> dict:
    market = load_market(cfg)
    base = copy.deepcopy(cfg)
    base["pair_spec"] = market.spec
    results = {}
    for name, combo in sets["sets"].items():
        log.info("パラメータの組 %s", name)
        results[name] = {"combo": combo, **diagnose_set(apply_combo(base, combo), market)}
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

    rep = run_diagnose(load_config(args.config), yaml.safe_load(Path(args.sets).read_text()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    md = render(rep)
    (out / "report.md").write_text(md)
    print(md)
