"""スイング（4 時間足で判断し 1〜3 日保有）の方向の研究の実行（公開 API のみ、認証不要）。2026-09-28 オーナー決定。

    python scripts/swing.py --mode check   # データの確認（足の境界、ペアごとの期間・抜け・出来高、今のスプレッド）。損益は出さない
    python scripts/swing.py --mode eval    # 事前登録した 45 通りを全期間で 1 回評価（configs/swing.yaml の owner_approved が必要）

公式 4 時間足（/candlestick/4hour/{YYYY}）をペア × 年ごとに 1 リクエストで取り、.cache/swing にためる。
結果は reports/swing/（check.md・check.json、report.md・metrics.json）。評価の試行は reports/experiments.jsonl に追記する。
設計は bbresearch/swing.py の冒頭。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bbdata.client import BitbankAPIError, PublicClient
from bbdata.download import download_candles, load_candles
from bbresearch.swing import (BARS_PER_DAY, BARS_PER_YEAR, bar_sigma, crash_events, daily, deflate, eligible, evaluate,
                              event_study, judge, panel, risk_parity, simulate, stats)

PAIRS_BASE_URL = "https://api.bitbank.cc/v1"
CACHE = Path(".cache/swing")
OUT = Path("reports/swing")
CHECK_DAYS = ("2026-09-14", "2026-09-21")  # 4 時間足の境界を 1 時間足と照合する期間（UTC、終わりは含まない）


def jpy_pairs() -> list[dict]:
    """JPY ペアの一覧と手数料率。/spot/pairs は public.bitbank.cc ではなく api.bitbank.cc/v1 にある（scripts/activity.py と同じ）。"""
    api = PublicClient(base_url=PAIRS_BASE_URL, min_interval=0.5)
    return [p for p in api.get("/spot/pairs")["pairs"] if p["name"].endswith("_jpy")]


def load_4h(api: PublicClient, names: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    out = {}
    for p in names:
        download_candles(api, p, "4hour", s, e, CACHE)
        out[p] = load_candles(CACHE, p, "4hour", start, end)
    return out


def pct(x, nd=1) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{nd}f}%"


def num(x, nd=2) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


# ------------------------------------------------------------------ データの確認

def to_4h(h1: pd.DataFrame, offset_hours: int) -> pd.DataFrame:
    """1 時間足を、offset_hours 時間ずらした 4 時間の区切りでまとめる（4 本そろった区切りだけ）。"""
    key = (h1.index - pd.Timedelta(hours=offset_hours)).floor("4h") + pd.Timedelta(hours=offset_hours)
    g = h1.groupby(key)
    agg = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return agg[g.size() == 4]


def check_boundary(api: PublicClient) -> list[dict]:
    """公式 4 時間足の時刻 T の足が、1 時間足の [T + o, T + o + 4 時間) をまとめたものかを o = -3〜3 時間で照合する。"""
    s, e = (date.fromisoformat(d) for d in CHECK_DAYS)
    download_candles(api, "btc_jpy", "1hour", s, e, CACHE)
    h1 = load_candles(CACHE, "btc_jpy", "1hour", CHECK_DAYS[0], CHECK_DAYS[1])
    h4 = load_candles(CACHE, "btc_jpy", "4hour", CHECK_DAYS[0], CHECK_DAYS[1])
    rows = []
    for off in range(-3, 4):
        a = to_4h(h1, off % 4)
        a.index = a.index - pd.Timedelta(hours=off)
        common = a.index.intersection(h4.index)
        if len(common) == 0:
            rows.append({"offset_hours": off, "n": 0, "close_match": None, "ohlc_match": None})
            continue
        x, y = a.loc[common], h4.loc[common]
        close_ok = np.isclose(x["close"], y["close"])
        ohlc_ok = np.all([np.isclose(x[c], y[c]) for c in ("open", "high", "low", "close")], axis=0)
        vol_rel = float(np.median(np.abs(x["volume"] / y["volume"] - 1)))
        rows.append({"offset_hours": off, "n": int(len(common)), "close_match": float(close_ok.mean()),
                     "ohlc_match": float(ohlc_ok.mean()), "volume_rel_err_median": vol_rel})
    return rows


def spreads(api: PublicClient, names: list[str], rounds: int = 5) -> dict[str, float | None]:
    """板の最良気配から半スプレッド（(売り - 買い) / (売り + 買い)）を rounds 回測った中央値。"""
    obs: dict[str, list[float]] = {p: [] for p in names}
    for _ in range(rounds):
        for p in names:
            try:
                d = api.depth(p)
                ask, bid = float(d["asks"][0][0]), float(d["bids"][0][0])
                obs[p].append((ask - bid) / (ask + bid))
            except (BitbankAPIError, KeyError, IndexError, ValueError):
                continue
    return {p: (float(np.median(v)) if v else None) for p, v in obs.items()}


def main_check(cfg: dict) -> None:
    api = PublicClient(min_interval=0.5)
    pairs = jpy_pairs()
    names = [p["name"] for p in pairs]
    candles = load_4h(api, names, cfg["data"]["start"], cfg["data"]["end"])
    boundary = check_boundary(api)
    half = spreads(api, names)
    close, vol = panel(candles, cfg["data"]["start"], cfg["data"]["end"])
    elig = eligible(close, vol, cfg["min_bars"], cfg["turnover_days"], cfg["min_turnover_jpy"])
    years = sorted(set(close.index.year))
    per = []
    for p in pairs:
        n = p["name"]
        df = candles.get(n, pd.DataFrame())
        row = {"pair": n, "taker_fee": float(p["taker_fee_rate_quote"]), "half_spread": half.get(n),
               "is_enabled": p.get("is_enabled")}
        if df.empty or n not in close:
            per.append(row | {"first": None})
            continue
        grid = close[n].dropna().index
        vd = vol[n].resample("D").sum()
        row |= {"first": str(df.index.min()), "last": str(df.index.max()), "bars": int(len(df)),
                "missing": int(len(grid.difference(df.index))), "zero_volume_share": float((df["volume"] == 0).mean()),
                "median_daily_jpy": {int(y): float(g[g > 0].median()) if (g > 0).any() else 0.0
                                     for y, g in vd.groupby(vd.index.year)},
                "eligible_share": {int(y): float(g.mean()) for y, g in elig[n].groupby(elig.index.year)}}
        per.append(row)
    count = elig.sum(axis=1)
    by_year = {int(y): {"mean": float(g.mean()), "min": int(g.min()), "max": int(g.max())}
               for y, g in count.groupby(count.index.year)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check.json").write_text(json.dumps({"boundary": boundary, "pairs": per, "eligible_count": by_year,
                                                "run_at": datetime.now(timezone.utc).isoformat()}, indent=1))
    md = [f"# スイングの研究: データの確認（{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC）\n",
          "損益は計算していない（事前登録の前に結果を見ないため）。\n",
          f"## 4 時間足の境界（btc_jpy、{CHECK_DAYS[0]} 〜 {CHECK_DAYS[1]}）\n",
          "公式の時刻 T の 4 時間足と、1 時間足の [T + o, T + o + 4 時間) をまとめたものを照合した。o = 0 だけが一致すれば、"
          "時刻は足の開始で区切りは UTC の 0 時起点。\n",
          "| o（時間） | 照合した本数 | 終値の一致率 | OHLC の一致率 | 出来高の相対誤差（中央値） |", "|---|---|---|---|---|"]
    md += [f"| {b['offset_hours']} | {b['n']} | {num(b['close_match'], 3)} | {num(b['ohlc_match'], 3)} | "
           f"{num(b.get('volume_rel_err_median'), 4)} |" for b in boundary]
    md += ["", f"## 対象ペアの数（時点ごと。履歴 {cfg['min_bars']} 本以上、直近 {cfg['turnover_days']} 日の 24 時間出来高の中央値 "
           f"{cfg['min_turnover_jpy'] / 1e6:,.0f} 百万円以上）\n",
           "| 年 | 平均 | 最小 | 最大 |", "|---|---|---|---|"]
    md += [f"| {y} | {v['mean']:.1f} | {v['min']} | {v['max']} |" for y, v in by_year.items()]
    md += ["", "## ペアごと\n",
           "| ペア | 最初の足（UTC） | 本数 | 抜け | 出来高 0 の足 | テイカー手数料 | 今の半スプレッド | "
           + " | ".join(f"{y} 出来高 / 対象" for y in years) + " |",
           "|---|---|---|---|---|---|---|" + "---|" * len(years)]
    for r in sorted(per, key=lambda r: r["first"] or "9999"):
        if r.get("first") is None:
            md.append(f"| {r['pair']} | データなし | | | | {pct(r['taker_fee'], 2)} | {pct(r['half_spread'], 3)} |"
                      + " |" * len(years))
            continue
        cells = [f"{r['median_daily_jpy'].get(y, 0) / 1e6:,.0f} / {pct(r['eligible_share'].get(y), 0)}"
                 if y in r["median_daily_jpy"] else "" for y in years]
        md.append(f"| {r['pair']} | {r['first'][:16]} | {r['bars']:,} | {r['missing']:,} | {pct(r['zero_volume_share'])} | "
                  f"{pct(r['taker_fee'], 2)} | {pct(r['half_spread'], 3)} | " + " | ".join(cells) + " |")
    md += ["", "出来高は UTC の 1 日あたり出来高（JPY、百万円）のその年の中央値（出来高 0 の日を除く）。対象はその年の足のうち"
           "対象の条件を満たした割合。半スプレッドは板の最良気配から 5 回測った中央値。"]
    (OUT / "check.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# ------------------------------------------------------------------ 評価（事前登録どおりに 1 回）

def cfg_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def main_eval(cfg: dict) -> None:
    if not cfg.get("owner_approved"):
        sys.exit("configs/swing.yaml の owner_approved が空。オーナーの承認の前は評価しない。")
    api = PublicClient(min_interval=0.5)
    pairs = [p for p in jpy_pairs() if p["name"] not in set(cfg["data"]["exclude"])]
    names = [p["name"] for p in pairs]
    fee = {p["name"]: float(p["taker_fee_rate_quote"]) for p in pairs}
    cost = {n: fee[n] + cfg["cost"]["slippage"] for n in names}
    candles = load_4h(api, names, cfg["data"]["start"], cfg["data"]["end"])
    close, vol = panel(candles, cfg["data"]["start"], cfg["data"]["end"])
    cost = {n: cost[n] for n in close.columns}
    res = evaluate(close, vol, cost, cfg)
    results, start = res["results"], res["start"]
    dsr = deflate(results)
    j = cfg["judge"]
    verdicts = [judge(results, dsr, fam, j["min_share_alpha_pos"], j["min_dsr"], j["min_year_share"], j["full_years"])
                for fam in ("tsmom", "xsmom", "rebound")]
    stress = evaluate(close, vol, cost, cfg, cost_mult=cfg["cost"]["stress_mult"],
                      only={v["best"] for v in verdicts})["results"]

    # 記述用の比較: BTC の保有、等分の常時保有
    win = close.index >= start
    elig = eligible(close, vol, cfg["min_bars"], cfg["turnover_days"], cfg["min_turnover_jpy"])
    btc = simulate(pd.DataFrame({"btc_jpy": 1.0}, index=close.index).where(close[["btc_jpy"]].notna(), 0.0),
                   close, cost)[win]
    ew = elig.astype(float).div(elig.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    ew_sim = simulate(ew, close, cost)[win]
    extra = {"BTC を保有": stats(btc["net"]), "等分の常時保有（毎足）": stats(ew_sim["net"])}

    # 急落後の反発の記述統計
    base_elig = elig
    events = []
    for bars in cfg["rebound"]["window_bars"]:
        for k in cfg["rebound"]["k_sigma"]:
            ev = crash_events(close, bars, k, cfg["sigma_days"])
            for hold in cfg["hold_days"]:
                events.append({"window_bars": bars, "k_sigma": k, "hold_days": hold}
                              | event_study(close[win], ev[win], base_elig[win], hold * BARS_PER_DAY))

    th = cfg_hash(cfg)
    run_at = datetime.now(timezone.utc).isoformat()
    log = Path("reports/experiments.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as fh:
        for k, r in results.items():
            fh.write(json.dumps({"run_at": run_at, "stage": "swing", "trial_hash": th,
                                 "code_version": os.environ.get("GITHUB_SHA"), "key": f"swing/{k}",
                                 "alpha_ann": r["alpha_ann"], "t_alpha": r["t_alpha"],
                                 "active_sr_ann": (r["active_sr_daily"] or 0) * math.sqrt(365),
                                 "dsr": dsr[k], "sharpe": r["sharpe"]}) + "\n")

    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {"run_at": run_at, "trial_hash": th, "owner_approved": cfg["owner_approved"], "start": str(start),
               "end": str(res["end"]), "cost": cost, "results": results, "dsr": dsr, "verdicts": verdicts,
               "stress": stress, "benchmarks": res["benchmarks"], "extra": extra, "events": events}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    write_report(cfg, metrics, res["eligible_count"])


def write_report(cfg: dict, m: dict, count: pd.Series) -> None:
    res, dsr = m["results"], m["dsr"]
    fam_ja = {"tsmom": "H1 時系列モメンタム", "xsmom": "H2 横断モメンタム", "rebound": "H3 急落後の反発"}
    j = cfg["judge"]
    fees = sorted({round(v - cfg["cost"]["slippage"], 6) for v in m["cost"].values()})
    md = [f"# スイングの方向の研究（4 時間足・1〜3 日保有、事前登録した {len(res)} 通りを全期間で 1 回）\n",
          f"- 期間: {m['start'][:16]} 〜 {m['end'][:16]}（UTC、4 時間足）。設定のハッシュ {m['trial_hash']}"
          f"（configs/swing.yaml、オーナー承認 {m['owner_approved']}）",
          f"- 対象: 時点ごとに、履歴 {cfg['min_bars']} 本以上かつ直近 {cfg['turnover_days']} 日の 24 時間出来高の中央値が "
          f"{cfg['min_turnover_jpy'] / 1e6:,.0f} 百万円以上の JPY ペア。数の年平均: "
          + "、".join(f"{y} {g.mean():.1f}" for y, g in count.groupby(count.index.year)),
          f"- 費用: 片側 = テイカー手数料（{' / '.join(pct(f, 2) for f in fees)}）+ 滑り {pct(cfg['cost']['slippage'], 2)}",
          "- 比較の基準: 同じ配分（リスク均等）を常に保有し、同じ H でずらしたもの。アルファは日次リターンの回帰"
          "（Newey-West 5 次）の切片を年率にしたもの。アクティブ = 戦略 − β × 基準。",
          "",
          f"## 判定（事前登録: アルファが正の割合 ≥ {j['min_share_alpha_pos']:.2f}、最良の DSR ≥ {j['min_dsr']}、"
          f"最良の年ごとのアクティブが正の年 ≥ {j['min_year_share']:.0%}（{j['full_years'][0]}〜{j['full_years'][-1]}））\n",
          "| 仮説 | 組み合わせ | アルファが正 | 最良（アクティブのシャープ） | 最良の DSR | 正の年 | 判定 |",
          "|---|---|---|---|---|---|---|"]
    for v in m["verdicts"]:
        md.append(f"| {fam_ja[v['family']]} | {v['n']} | {v['share_alpha_pos']:.0%} | {v['best']} | {num(v['best_dsr'], 3)} | "
                  f"{v['best_year_share']:.0%} | {'支持' if v['supported'] else '支持しない'} |")
    md += ["", "## すべての組み合わせ（費用込み）\n",
           "| 組み合わせ | 年率リターン | 年率ボラ | シャープ | 最大DD | 保有率 | 回転/年 | 費用/年 | アルファ（年率） | β | t | "
           "アクティブのシャープ（年率） | DSR |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        md.append(f"| {k} | {pct(r['ann_return'])} | {pct(r['ann_vol'])} | {num(r['sharpe'])} | {pct(r['max_drawdown'])} | "
                  f"{pct(r['exposure'], 0)} | {num(r['turnover_per_year'], 0)} | {pct(r['cost_per_year'])} | "
                  f"{pct(r['alpha_ann'])} | {num(r['beta'])} | {num(r['t_alpha'])} | "
                  f"{num((r['active_sr_daily'] or 0) * math.sqrt(365))} | {num(dsr[k], 3)} |")
    md += ["", "## 比較（費用込み）\n", "| 基準 | 年率リターン | 年率ボラ | シャープ | 最大DD |", "|---|---|---|---|---|"]
    for name, s in list(m["benchmarks"].items()) + list(m["extra"].items()):
        md.append(f"| {name.replace('always/', '常時保有（リスク均等）').replace('H', ' H') if name.startswith('always') else name} | "
                  f"{pct(s['ann_return'])} | {pct(s['ann_vol'])} | {num(s['sharpe'])} | {pct(s['max_drawdown'])} |")
    md += ["", "## 仮説ごとの最良の組み合わせの年ごと\n"]
    for v in m["verdicts"]:
        r = res[v["best"]]
        b = m["benchmarks"][f"always/H{r['hold_days']}d"]["yearly_return"]
        md += [f"### {fam_ja[v['family']]}: {v['best']}\n", "| 年 | 戦略 | 常時保有 | アクティブ（日次の合計） |",
               "|---|---|---|---|"]
        md += [f"| {y} | {pct(r['yearly_return'].get(y))} | {pct(b.get(y))} | {pct(a)} |"
               for y, a in r["yearly_active"].items()]
        md.append("")
    md += [f"## 費用を {cfg['cost']['stress_mult']} 倍にした感度（最良の組み合わせ。試行には数えない）\n",
           "| 組み合わせ | 年率リターン | シャープ | アルファ（年率） | t |", "|---|---|---|---|---|"]
    md += [f"| {k} | {pct(r['ann_return'])} | {num(r['sharpe'])} | {pct(r['alpha_ann'])} | {num(r['t_alpha'])} |"
           for k, r in m["stress"].items()]
    md += ["", "## 急落後の反発の記述統計（判定には使わない。急落が続く足は最初の足だけを数える）\n",
           "| 窓（本） | k | 保有 | 局面の数 | その後の平均（対数） | t | 全時点の平均 |", "|---|---|---|---|---|---|---|"]
    md += [f"| {e['window_bars']} | {e['k_sigma']} | {e['hold_days']} 日 | {e['episodes']} | {pct(e['mean_fwd'], 2)} | "
           f"{num(e['t'])} | {pct(e['uncond_mean_fwd'], 2)} |" for e in m["events"]]
    md += ["", "## 注意\n",
           "- 今上場している JPY ペアだけを使っている（上場廃止になったペアは入らない。生存バイアス）。",
           "- 手数料は今の値を過去にも適用し、滑りは一律。足の間は重みを一定とみなし、値動きによるずれの調整の費用は入れていない。",
           "- 2024 年以降のデータは過去の研究（15 分〜4 時間の時間軸）で何度も使っている。今回は数値を事前に固定し、1 回だけ評価した。",
           "- 支持された仮説があっても、次は前向きのドライランで確かめる（実際の発注はしない）。"]
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["check", "eval"], required=True)
    ap.add_argument("--config", default="configs/swing.yaml")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    t0 = time.monotonic()
    (main_check if args.mode == "check" else main_eval)(cfg)
    print(f"所要 {time.monotonic() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
