"""スイングの研究の H1（時系列モメンタム 24 通り）の前向きのドライラン（発注しない）。2026-09-28 オーナー決定。

    python scripts/swing_forward.py

評価（configs/swing.yaml、2026-09-28 承認）と同じコード（bbresearch/swing.py）と数値で、configs/swing_forward.yaml の
forward_start 以降の公式 4 時間足だけで成績を出す。判断は 4 時間足の終値だけで決まるので、足がそろってから計算すれば
4 時間ごとに動かし続けた場合と同じ結果になる（実際の約定のずれは含まない）。評価に使ったファイルの SHA-256 が
変わっていたら実行しない。判定は judge_at に、24 通りの超過リターン（アルファ × 日数）の平均が正かどうか。
Actions（swing_forward.yml）で毎月 1 日に実行し、reports/swing_forward/ をコミットする。
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import scripts.swing as sw
from bbdata.download import to_utc
from bbresearch.swing import (BARS_PER_DAY, bar_sigma, configs, daily, decisions, eligible, hac_alpha, panel,
                              risk_parity, simulate, stagger)

OUT = Path("reports/swing_forward")
MIN_DAYS_FOR_ALPHA = 10  # これより短い間はアルファの回帰をしない


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_frozen(fwd: dict) -> None:
    changed = sorted(p for p, h in fwd["frozen"].items() if sha256(p) != h)
    if changed:
        sys.exit(f"評価に使ったファイルが変わっている: {changed}。前向きの検証は同じコードと数値で行う。")


def forward(cfg: dict, fwd: dict, now: datetime) -> dict:
    """forward_start から min(now, judge_at) までの足で、H1 の 24 通りと常時保有（同じ H）を比べる。"""
    check_frozen(fwd)
    api = sw.PublicClient(min_interval=0.5)
    pairs = [p for p in sw.jpy_pairs() if p["name"] not in set(cfg["data"]["exclude"])]
    cost = {p["name"]: float(p["taker_fee_rate_quote"]) + cfg["cost"]["slippage"] for p in pairs}
    end = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    candles = sw.load_4h(api, [p["name"] for p in pairs], cfg["data"]["start"], end)
    close, vol = panel(candles, cfg["data"]["start"], end)
    done = close.index + pd.Timedelta(hours=4) <= pd.Timestamp(now)  # 形成中の足は使わない
    close, vol = close[done], vol[done]
    cost = {n: cost[n] for n in close.columns}
    elig = eligible(close, vol, cfg["min_bars"], cfg["turnover_days"], cfg["min_turnover_jpy"])
    base = risk_parity(bar_sigma(close, cfg["sigma_days"]), elig)
    t0, t1 = to_utc(fwd["forward_start"]), to_utc(fwd["judge_at"])
    win = (close.index >= t0) & (close.index < t1)
    fam = [c for c in configs(cfg) if c["family"] == fwd["family"]]

    held, on = {}, {}
    for c in fam:
        dec = decisions(c, close, elig, base, cfg)
        held[c["key"]] = stagger(dec, c["hold_days"] * BARS_PER_DAY)
        on[c["key"]] = dec.iloc[-1] > 0
    last = close.index[-1]
    avg_w = pd.concat([h.iloc[-1] for h in held.values()], axis=1).mean(axis=1)
    share_on = pd.concat(list(on.values()), axis=1).mean(axis=1)
    holdings = [{"pair": p, "eligible": bool(elig.loc[last, p]), "base": float(base.loc[last, p]),
                 "share_on": float(share_on[p]), "weight": float(avg_w[p])}
                for p in close.columns if base.loc[last, p] > 0 or avg_w[p] > 0]
    out = {"run_at": now.isoformat(), "forward_start": str(t0), "judge_at": str(t1), "last_bar": str(last),
           "bars": int(win.sum()), "holdings": sorted(holdings, key=lambda h: -h["weight"]), "results": {}}
    if not win.any():
        return out

    bench = {h: simulate(stagger(base, h * BARS_PER_DAY), close, cost)[win] for h in cfg["hold_days"]}
    for c in fam:
        sim = simulate(held[c["key"]], close, cost)[win]
        b = bench[c["hold_days"]]
        y, x = daily(sim["net"]), daily(b["net"])
        r = {"return": float((1 + sim["net"]).prod() - 1), "bench_return": float((1 + b["net"]).prod() - 1),
             "days": int(len(y)), "exposure": float(sim["exposure"].mean())}
        if len(y) >= MIN_DAYS_FOR_ALPHA:
            reg = hac_alpha(y, x)
            act = reg.pop("active")
            r |= {"alpha_ann": reg["alpha"] * 365, "beta": reg["beta"], "t_alpha": reg["t_alpha"],
                  "active_sum": float(act.sum())}
        out["results"][c["key"]] = r
    res = [r for r in out["results"].values() if "active_sum" in r]
    days = max(r["days"] for r in out["results"].values())
    out["summary"] = {"days": days,
                      "mean_alpha_ann": float(np.mean([r["alpha_ann"] for r in res])) if res else None,
                      "mean_active_sum": float(np.mean([r["active_sum"] for r in res])) if res else None,
                      "n_active_pos": int(sum(r["active_sum"] > 0 for r in res)), "n": len(fam)}
    if pd.Timestamp(now) >= t1 and res:
        out["verdict"] = "前向きでも正" if out["summary"]["mean_active_sum"] > 0 else "前向きでは正にならなかった"
    return out


def pct(x, nd=1) -> str:
    return "-" if x is None else f"{x * 100:.{nd}f}%"


def num(x, nd=2) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def write(fwd: dict, m: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(m, indent=1, default=str))
    s = m.get("summary")
    with (OUT / "history.jsonl").open("a") as fh:
        fh.write(json.dumps({"run_at": m["run_at"], "last_bar": m["last_bar"], "bars": m["bars"]}
                            | (s or {}) | ({"verdict": m["verdict"]} if "verdict" in m else {})) + "\n")
    state = m.get("verdict") or f"途中経過（判定は {m['judge_at'][:10]}）"
    md = ["# H1（時系列モメンタム 24 通り）の前向きのドライラン（発注しない）\n",
          f"- 期間: {m['forward_start'][:16]} 〜 最後の足 {m['last_bar'][:16]}（UTC、4 時間足 {m['bars']} 本）。"
          f"判定は {m['judge_at'][:10]} に、24 通りの超過リターンの平均が正かどうか（configs/swing_forward.yaml）",
          f"- 状態: {state}",
          "- 同じコード・同じ数値: bbresearch/swing.py と configs/swing.yaml の SHA-256 が評価のときと一致"]
    if s:
        md.append(f"- 24 通りの平均: アルファ 年率 {pct(s['mean_alpha_ann'])}（評価では {pct(fwd['backtest_mean_alpha_ann'])}）、"
                  f"超過リターンの合計 {pct(s['mean_active_sum'], 2)}、超過リターンが正の組み合わせ {s['n_active_pos']}/{s['n']}"
                  f"（{s['days']} 日。{MIN_DAYS_FOR_ALPHA} 日未満はアルファを出さない）")
    else:
        md.append("- 前向きの期間の足がまだない。")
    md += ["", f"## 今の保有（最後の足 {m['last_bar'][:16]} の終値で決めた重み。24 通りの平均）\n",
           "| ペア | 対象 | リスク均等の枠 | 信号が出ている組み合わせ | 平均の重み |", "|---|---|---|---|---|"]
    md += [f"| {h['pair']} | {'○' if h['eligible'] else '×'} | {pct(h['base'])} | {pct(h['share_on'], 0)} | {pct(h['weight'])} |"
           for h in m["holdings"]]
    md.append(f"| 合計 | | | | {pct(sum(h['weight'] for h in m['holdings']))} |")
    if m["results"]:
        md += ["", "## 組み合わせごと（費用込み）\n",
               "| 組み合わせ | リターン | 常時保有 | 保有率 | アルファ（年率） | t | 超過リターンの合計 |", "|---|---|---|---|---|---|---|"]
        for k, r in m["results"].items():
            md.append(f"| {k} | {pct(r['return'], 2)} | {pct(r['bench_return'], 2)} | {pct(r['exposure'], 0)} | "
                      f"{pct(r.get('alpha_ann'))} | {num(r.get('t_alpha'))} | "
                      f"{pct(r.get('active_sum'), 2)} |")
    md += ["", "## 注意\n",
           "- 約定は足の終値に片側の費用（テイカー手数料 + 0.1%）を足したものとみなす。実際の約定のずれは含まない。",
           "- 12 か月では、効きが評価どおり（アクティブのシャープ 0.8 前後）でも超過リターンが正になる確率は約 79%、"
           "効きがなくても 50%。判定は弱い証拠にしかならない。途中の報告は判定に使わない。"]
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main() -> None:
    with open("configs/swing.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open("configs/swing_forward.yaml", encoding="utf-8") as f:
        fwd = yaml.safe_load(f)
    now = datetime.now(timezone.utc)
    write(fwd, forward(cfg, fwd, now))


if __name__ == "__main__":
    main()
