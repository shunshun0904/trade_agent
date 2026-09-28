"""評価の土台（候補 1）: ボラだけの基準（HAR 型・GARCH-t 型）に対する、分位点回帰フォレストの幅と向きの上乗せ。2026-09-28 オーナー決定。

    python scripts/dist_base.py --mode check              # データの確認と学習期間での当てはめ（評価期間の成績は出さない）
    python scripts/dist_base.py --mode null-prepare       # 帰無の監査に使う実データの 1 分の終値をファイルにする（並列のジョブの前に 1 回）
    python scripts/dist_base.py --mode null --shard 0     # 帰無の監査（期待リターン 0 の合成データ）。null_audit.shards 分の 1 を実行
    python scripts/dist_base.py --mode null-summary       # 帰無の監査の集計（reports/dist_base/null.md・null.json）
    python scripts/dist_base.py --mode eval               # 事前登録どおりに 1 回評価（configs/dist_base.yaml の owner_approved が必要）

データ: 公式 1 時間足（scripts/dist_forecast.py と同じキャッシュ .cache/candles）と、保存済みの約定（data/raw、Actions のキャッシュ）
から作る 1 分の終値。新しい種類の API アクセスはない。設計は bbresearch/distbase.py と bbresearch/nullsim.py の冒頭。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bbresearch import nullsim
from bbresearch.distbase import (LABELS, MODELS, fit_garch_t, har_features, har_target, hourly_rv, losses,
                                 minute_close_from_raw, pit_subseries, rank_percentile, run_pipeline, summary_row, year_means)
from bbresearch.fcompare import compare, holm

OUT = Path("reports/dist_base")
COMP_METRICS = ("brier_0", "brier_1", "brier_2", "logloss_0", "pinball50", "tw_up", "tw_dn", "neg_payoff", "crps", "pinball")
METRIC_JA = {"brier_0": "ブライア P(>0)", "brier_1": "ブライア P(>+費用)", "brier_2": "ブライア P(>−費用)",
             "logloss_0": "対数損失 P(>0)", "pinball50": "50% のピンボール", "tw_up": "上側の twCRPS", "tw_dn": "下側の twCRPS",
             "neg_payoff": "−損益（θ = 費用）", "crps": "CRPS", "pinball": "ピンボールの平均"}


# ------------------------------------------------------------------ データ

def load_hourly(cfg: dict) -> pd.DataFrame:
    from bbdata.client import PublicClient
    from scripts.dist_forecast import fetch_hourly

    end = (pd.Timestamp(cfg["end"], tz="UTC").to_pydatetime() if cfg.get("end") else
           datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0))   # 当日分（形成中）は取らない
    return fetch_hourly(PublicClient(min_interval=0.3), cfg["pair"], cfg["start"], end)


def load_minutes(cfg: dict, h1: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return minute_close_from_raw(cfg["trades_root"], cfg["pair"], cfg["start"], h1.index[-1] + pd.Timedelta(hours=1))


def cfg_hash(cfg: dict) -> str:
    c = {k: v for k, v in cfg.items() if k != "owner_approved"}
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]


CODE_FILES = ("bbresearch/distbase.py", "bbresearch/fcompare.py", "bbresearch/nullsim.py", "bbresearch/qrf.py",
              "bbresearch/indicators.py", "bbresearch/ta.py", "scripts/dist_base.py")


def code_hash() -> str:
    """帰無の監査と評価が同じ手順（コード）で走ったことを確かめるためのハッシュ。"""
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update((root / f).read_bytes())
    return h.hexdigest()[:12]


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (pd.Timestamp, datetime)):
        return str(x)
    return x


def f6(x) -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.6f}"


def num(x, nd=2) -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def pct(x, nd=1) -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x * 100 + 0.0:.{nd}f}%"


def pv(x) -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else (f"{x:.3f}" if x >= 0.001 else "<0.001")


# ------------------------------------------------------------------ 集計（実データと合成データで同じ）

def summarize(out: dict, cfg: dict, full: bool) -> dict:
    t = cfg["tests"]
    kw = dict(nw_mult=t["nw_mult"], ewc_mult=t["ewc_mult"], fixedb_sims=t["fixedb_sims"], seed=t["seed"],
              acf_lags=tuple(t["acf_lags"]), hill_share=t["hill_share"])
    kw_full = kw | {"bootstrap_reps": t["bootstrap_reps"] if full else 0, "adf": full}
    S = {"garch": out["garch"], "horizons": {}}
    for h, H in out["horizons"].items():
        y, idx = H["y"], H["index"]
        rows, L = {}, {}
        for m in MODELS:
            r = summary_row(H["models"][m], y, cfg)
            spread = r.pop("spread_series", None)
            r["spread"] = compare(spread, **kw) if spread is not None else None
            payoff = -losses(H["models"][m], y, cfg)["neg_payoff"]
            r["payoff_test"] = compare(payoff, **kw) if np.any(payoff != 0) else None
            rows[m] = r
            L[m] = losses(H["models"][m], y, cfg)
        comps = {}
        for m in MODELS:                                   # 幅: 各モデル − HAR 型
            if m != "har":
                for met in ("crps", "pinball"):
                    comps[f"{m}|har|{met}"] = compare(L[m][met] - L["har"][met], **kw_full)
        pairs = [tuple(p) for p in cfg["judge"]["comparisons"]] + [("forest", "har")]
        for a, b in pairs:                                 # 向き（主指標と副指標）
            for met in COMP_METRICS:
                key = f"{a}|{b}|{met}"
                if key not in comps:
                    comps[key] = compare(L[a][met] - L[b][met], **kw_full)
        S["horizons"][h] = {
            "n_test": int(len(y)), "n_train": H["n_train"], "test_start": str(idx[0]), "test_end": str(idx[-1]),
            "train_start": H["train_start"], "train_end": H["train_end"], "har_coef": H["har_coef"], "har_r2": H["har_r2"],
            "sm_trees": H["sm_trees"],
            "rows": rows, "comps": comps,
            "pit": {m: pit_subseries(H["models"][m]["pit"], int(h)) for m in MODELS},
            "crps_by_year": {m: year_means(idx, L[m]["crps"]) for m in MODELS},
            "brier0_diff_by_year": {f"{a}|{b}": year_means(idx, L[a]["brier_0"] - L[b]["brier_0"]) for a, b in pairs}}
        if full:
            S["horizons"][h]["mcs"] = {met: mcs(pd.DataFrame({m: L[m][met] for m in MODELS}), t) for met in ("crps", "brier_0")}
    return S


def mcs(losses_df: pd.DataFrame, t: dict) -> dict:
    """モデル信頼集合（Hansen ほか 2011、arch）。損失が全く同じモデル（例: P(>0) が一定の無条件・HAR 型・GARCH-t 型）は
    1 つにまとめて計算し（差の分散が 0 で計算できないため）、同じ結果を付ける。"""
    from arch.bootstrap import MCS

    reps, same = [], {}
    for c in losses_df.columns:
        k = next((r for r in reps if np.array_equal(losses_df[c].to_numpy(), losses_df[r].to_numpy())), None)
        if k is None:
            reps.append(c)
        else:
            same.setdefault(k, []).append(c)
    m = MCS(losses_df[reps], size=t["mcs_size"], reps=t["mcs_reps"], block_size=t["mcs_block"], method="R", seed=t["seed"])
    m.compute()
    pv_ = {k: float(v) for k, v in m.pvalues["Pvalue"].items()}
    pv_ |= {c: pv_[k] for k, cs in same.items() for c in cs}
    included = [c for c in losses_df.columns if c in m.included or any(c in same.get(k, []) for k in m.included)]
    return {"included": included, "pvalues": pv_, "same": same}


def slim(S: dict) -> dict:
    """帰無の監査で残す値（平均と検定の統計量）。"""
    keep_rows = ("crps", "pinball", "brier_0", "brier_1", "brier_2", "logloss_0", "pinball50", "tw_up", "tw_dn", "payoff",
                 "auc_0", "auc_1", "auc_2", "interval90", "buy_share")
    out = {"garch": S["garch"], "horizons": {}}
    for h, H in S["horizons"].items():
        out["horizons"][h] = {
            "sm_trees": H["sm_trees"],
            "rows": {m: {k: r[k] for k in keep_rows} | {"spread": (r["spread"] or {}).get("mean"),
                                                         "t_spread": (r["spread"] or {}).get("t_fb")} for m, r in H["rows"].items()},
            "comps": {k: {kk: c[kk] for kk in ("mean", "t_fb", "p_fb", "t_ewc", "p_ewc")} for k, c in H["comps"].items()}}
    return out


def judge(S: dict, null: dict | None, cfg: dict) -> dict:
    J = cfg["judge"]
    prim = J["primary"]
    fam = [(h, a, b) for h in S["horizons"] for a, b in J["comparisons"]]
    keys = [f"{a}|{b}|{prim}" for _, a, b in fam]
    cs = [S["horizons"][h]["comps"][k] for (h, _, _), k in zip(fam, keys)]
    adj_fb, adj_ewc = holm([c["p_fb"] for c in cs]), holm([c["p_ewc"] for c in cs])
    rows = []
    for (h, a, b), k, c, pf, pe in zip(fam, keys, cs, adj_fb, adj_ewc):
        imp = -c["mean"]
        nreps = (null or {}).get("reps", [])
        nl = [-rep["summary"]["horizons"][str(h)]["comps"][k]["mean"] for rep in nreps]
        fp = sum(significant(rep["summary"]["horizons"][str(h)]["comps"][k], better=True) for rep in nreps)
        p_null = rank_percentile(imp, nl) if nl else float("nan")
        ok = bool(c["mean"] < 0 and pf < J["alpha"] and pe < J["alpha"] and np.isfinite(p_null) and p_null < J["null_alpha"])
        rows.append({"h": int(h), "model": a, "reference": b, "diff": c["mean"], "t_fb": c["t_fb"], "p_fb": c["p_fb"],
                     "p_fb_holm": pf, "t_ewc": c["t_ewc"], "p_ewc": c["p_ewc"], "p_ewc_holm": pe, "p_null": p_null,
                     "n_null": len(nl), "null_false_pos": fp, "increment": ok})
    wa, wb = J["width_pair"]
    width = []
    for h, H in S["horizons"].items():
        c = H["comps"][f"{wa}|{wb}|crps"]
        verdict = ("差があるとは言えない" if not (c["p_fb"] < 0.05) else
                   (f"{LABELS[wa]}が良い" if c["mean"] < 0 else f"{LABELS[wb]}が良い"))
        width.append({"h": int(h), "diff": c["mean"], "rel": c["mean"] / H["rows"][wb]["crps"], "t_fb": c["t_fb"],
                      "p_fb": c["p_fb"], "p_ewc": c["p_ewc"], "verdict": verdict})
    return {"direction": rows, "any_increment": any(r["increment"] for r in rows), "width": width}


# ------------------------------------------------------------------ check

def main_check(cfg: dict) -> None:
    h1 = load_hourly(cfg)
    split = pd.Timestamp(cfg["split"], tz="UTC")
    full = pd.date_range(h1.index[0], h1.index[-1], freq="h")
    mc, cnt = load_minutes(cfg, h1)
    has = cnt > 0
    years = []
    for yr, g in has.groupby(has.index.year):
        runs = (~g).astype(int).groupby(g.cumsum()).sum()
        day_trades = cnt[cnt.index.year == yr].groupby(cnt[cnt.index.year == yr].index.floor("D")).sum()
        years.append({"year": int(yr), "minutes": int(len(g)), "share_with_trades": float(g.mean()),
                      "days_without_trades": int((day_trades == 0).sum()), "longest_gap_min": int(runs.max())})
    close_tape = mc.ffill().groupby(mc.index.floor("h")).last().reindex(h1.index)
    rel = (close_tape / h1["close"] - 1).abs().dropna()
    rv = hourly_rv(mc, h1.index)
    r = np.log(h1["close"]).diff()
    rv_ratio = [{"year": int(yr), "rv_over_r2": float(rv[rv.index.year == yr].mean() / (r[r.index.year == yr] ** 2).mean()),
                 "zero_rv_hours": int((rv[rv.index.year == yr] <= 0).sum())} for yr in sorted(set(h1.index.year))]
    # 学習期間だけでの当てはめ（評価期間の成績は出さない）
    hc = cfg["har"]
    Xh = har_features(rv, tuple(hc["windows"]), float(hc["rv_floor"]), bool(hc["calendar"]))
    har_fit = {}
    for h in cfg["horizons"]:
        yv = har_target(rv, int(h), float(hc["rv_floor"]))
        d = Xh.join(yv.rename("yv")).dropna()
        d = d[d.index < split - pd.Timedelta(hours=int(h))]
        A = np.column_stack([np.ones(len(d)), d[Xh.columns].to_numpy()])
        coef, *_ = np.linalg.lstsq(A, d["yv"].to_numpy(), rcond=None)
        res = d["yv"].to_numpy() - A @ coef
        har_fit[int(h)] = {"n": int(len(d)), "r2": float(1 - res.var() / d["yv"].var()),
                           "coef": {k: float(v) for k, v in zip(["const"] + list(Xh.columns[:len(hc["windows"])]), coef)}}
    r_train = r[r.index < split].dropna().to_numpy()
    g = fit_garch_t(r_train)
    cal = nullsim.calibrate(h1, split, g, seed=int(cfg["null_audit"]["seed0"]))
    real_stats = nullsim.path_stats(h1[h1.index < split])
    syn_stats = {}
    for kind in cfg["null_audit"]["models"]:
        h1s, _ = nullsim.simulate(h1.index, kind, cal, int(cfg["null_audit"]["seed0"]), real=(h1, mc))
        syn_stats[kind] = nullsim.path_stats(h1s[h1s.index < split])
    chk = {"hourly": {"n": int(len(h1)), "first": str(h1.index[0]), "last": str(h1.index[-1]), "missing": int(len(full) - len(h1))},
           "minutes": years, "close_tape_vs_official": {"n": int(len(rel)), "median": float(rel.median()),
                                                        "p99": float(rel.quantile(0.99)), "share_below_0.1pct": float((rel < 0.001).mean())},
           "rv": rv_ratio, "har_train": har_fit, "garch_train": g, "null_calibration": cal,
           "path_stats": {"real_train": real_stats} | syn_stats, "config_hash": cfg_hash(cfg)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check.json").write_text(json.dumps(jsonable(chk), ensure_ascii=False, indent=1))
    md = [f"# 評価の土台（候補 1）: データの確認（{cfg['pair']}、評価期間の成績は出さない）\n",
          f"- 1 時間足: {len(h1):,} 本（{h1.index[0]} 〜 {h1.index[-1]}）、欠けた足 {len(full) - len(h1)}",
          f"- 約定から作った 1 時間の終値と公式 1 時間足の終値の差: 中央値 {pct(chk['close_tape_vs_official']['median'], 4)}、"
          f"99% 点 {pct(chk['close_tape_vs_official']['p99'], 3)}、0.1% 未満の割合 {pct(chk['close_tape_vs_official']['share_below_0.1pct'])}",
          f"- 設定のハッシュ {cfg_hash(cfg)}（owner_approved を除く）\n",
          "## 1 分ごとの約定の有無（年ごと）\n", "| 年 | 分の数 | 約定のある分の割合 | 約定のない日 | 約定のない最長の連続（分） |", "|---|---|---|---|---|"]
    md += [f"| {y['year']} | {y['minutes']:,} | {pct(y['share_with_trades'])} | {y['days_without_trades']} | {y['longest_gap_min']:,} |" for y in years]
    md += ["", "## 実現分散（1 分）と 1 時間リターンの二乗の比（1 に近いほど、1 分の値動きの雑音が小さい）\n",
           "| 年 | 平均 RV ÷ 平均 r² | RV が 0 の時間 |", "|---|---|---|"]
    md += [f"| {x['year']} | {num(x['rv_over_r2'], 3)} | {x['zero_rv_hours']} |" for x in rv_ratio]
    md += ["", f"## 学習期間（{cfg['split']} より前）での当てはめ\n", "| ホライズン | HAR の行数 | R² | 係数（1・4・24・168 本） |", "|---|---|---|---|"]
    for h, fh in har_fit.items():
        md.append(f"| {h} 時間 | {fh['n']:,} | {num(fh['r2'], 3)} | " + "、".join(f"{v:.3f}" for k, v in fh["coef"].items() if k != "const") + " |")
    md += ["", f"- GARCH(1,1)-t（1 時間）: μ {g['mu']:.2e}、ω {g['omega']:.2e}、α {g['alpha']:.4f}、β {g['beta']:.4f}"
               f"（α + β = {g['alpha'] + g['beta']:.4f}）、ν {g['nu']:.2f}",
           f"- 合成データの GARCH: α {cal['garch']['alpha']:.4f}、β {cal['garch']['beta']:.4f}（α + β が {nullsim.MAX_PERSISTENCE} を超えるときは"
           f"比を保って縮める）、ω {cal['garch']['omega']:.2e}（無条件分散を学習期間の分散 {pct(cal['train_sd'], 3)}² に合わせる）、ν {cal['garch']['nu']:.2f}",
           f"- 2 状態のボラ: σ {pct(cal['ms2']['sigma'][0], 3)} / {pct(cal['ms2']['sigma'][1], 3)}、とどまる確率 "
           f"{cal['ms2']['P'][0][0]:.4f} / {cal['ms2']['P'][1][1]:.4f}",
           f"- 出来高: log(出来高) = {cal['volume']['a']:.3f} + {cal['volume']['b']:.3f} × log(値幅) + u、u の AR(1) 係数 "
           f"{cal['volume']['phi']:.3f}、ショックの標準偏差 {cal['volume']['sd']:.3f}",
           "", "## 合成データ（期待リターン 0）と実データの学習期間の比較\n",
           "| 系列 | 1 時間の標準偏差 | 尖度 | r² の自己相関 1 / 24 / 168 | log 出来高と log 値幅の相関 |", "|---|---|---|---|---|"]
    for name, s in chk["path_stats"].items():
        md.append(f"| {({'real_train': '実データ（学習期間）'}).get(name, name)} | {pct(s['sd'], 3)} | {num(s['kurt'], 1)} | "
                  f"{num(s['acf_r2_1'], 3)} / {num(s['acf_r2_24'], 3)} / {num(s['acf_r2_168'], 3)} | {num(s['corr_logvol_logrange'], 3)} |")
    (OUT / "check.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# ------------------------------------------------------------------ 帰無の監査

MINUTES_FILE = Path(".cache/dist_base_minutes.npz")


def save_minutes(path: Path, mc: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, start=np.int64(mc.index[0].value), values=mc.to_numpy())


def read_minutes(path: Path) -> pd.Series:
    d = np.load(path)
    idx = pd.date_range(pd.Timestamp(int(d["start"]), tz="UTC"), periods=len(d["values"]), freq="1min")
    return pd.Series(d["values"], index=idx)


def main_null_prepare(cfg: dict) -> None:
    """帰無の監査の並列のジョブが使う 1 分の終値（符号の入れ替えに要る実データ）を 1 回だけ作ってファイルにする。"""
    h1 = load_hourly(cfg)
    mc, _ = load_minutes(cfg, h1)
    save_minutes(MINUTES_FILE, mc)
    print(f"1 分の終値 {len(mc):,} 本を {MINUTES_FILE} に保存")


def main_null(cfg: dict, shard: int) -> None:
    h1 = load_hourly(cfg)
    mc = read_minutes(MINUTES_FILE) if MINUTES_FILE.exists() else load_minutes(cfg, h1)[0]
    split = pd.Timestamp(cfg["split"], tz="UTC")
    r_train = np.log(h1["close"]).diff()[lambda s: s.index < split].dropna().to_numpy()
    cal = nullsim.calibrate(h1, split, fit_garch_t(r_train), seed=int(cfg["null_audit"]["seed0"]))
    N = cfg["null_audit"]
    total = len(N["models"]) * int(N["reps"])
    mine = [i for i in range(total) if i % int(N["shards"]) == shard]
    reps = []
    for i in mine:
        kind, seed = N["models"][i // int(N["reps"])], int(N["seed0"]) + i
        t0 = time.monotonic()
        h1s, mcs_ = nullsim.simulate(h1.index, kind, cal, seed, real=(h1, mc))
        out = run_pipeline(h1s, mcs_, cfg)
        reps.append({"i": i, "kind": kind, "seed": seed, "summary": slim(summarize(out, cfg, full=False))})
        print(f"帰無 {i}（{kind}、種 {seed}）: {time.monotonic() - t0:.0f} 秒")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"null_shard_{shard}.json").write_text(json.dumps(jsonable({"calibration": cal, "reps": reps}), ensure_ascii=False))


def significant(c: dict, better: bool, alpha: float = 0.05) -> bool:
    """損失差 d = モデル − 参照 が、良い側（d < 0）または悪い側で、fixed-b と EWC の両方で p < alpha か。"""
    if c.get("p_fb") is None or c.get("p_ewc") is None or not (np.isfinite(c["p_fb"]) and np.isfinite(c["p_ewc"])):
        return False
    return (c["mean"] < 0) == better and c["p_fb"] < alpha and c["p_ewc"] < alpha


def main_null_summary(cfg: dict, shard_dir: Path) -> None:
    reps, cal = [], None
    for f in sorted(shard_dir.glob("null_shard_*.json")):
        d = json.loads(f.read_text())
        reps += d["reps"]
        cal = cal or d["calibration"]
    reps.sort(key=lambda r: r["i"])
    expected = len(cfg["null_audit"]["models"]) * int(cfg["null_audit"]["reps"])
    null = {"config_hash": cfg_hash(cfg), "code_hash": code_hash(), "calibration": cal, "n_reps": len(reps), "expected": expected,
            "reps": reps}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "null.json").write_text(json.dumps(null, ensure_ascii=False))
    md = [f"# 評価の土台（候補 1）: 帰無の監査（期待リターン 0 の合成データ、{len(reps)} / {expected} 回）\n",
          "合成データには向きの情報がないので、「向きのモデル − 参照」の改善は 0 以下に集まるはず（向きのモデルは位置の雑音の分だけ"
          "悪くなる）。0 より大きい側に偏るなら、手順に先読みなどの問題がある。\n",
          "- 良い側で有意: 改善が正で、fixed-b と EWC の両方で p < 0.05 だった回の割合（判定の偽陽性。ホルム補正の前）。",
          "- 悪い側で有意: 向きのモデル（幅の行はそのモデル）が有意に悪かった回の割合。帰無のもとでも損失の期待値は等しくないので、"
          "検定の大きさ（名目 5%）とは比べられない。",
          f"- 合成データは {'、'.join(cfg['null_audit']['models'])} を {cfg['null_audit']['reps']} 回ずつ。符号の入れ替えは実データの大きさを"
          "そのまま使うので、幅の比較は評価期間の実データの幅の比較に近い（幅の判定の基準は、この監査の前に固定した）。\n",
          f"設定のハッシュ {cfg_hash(cfg)}（owner_approved を除く）、コードのハッシュ {code_hash()}\n"]
    comps = [f"{a}|{b}|{m}" for a, b in cfg["judge"]["comparisons"] for m in COMP_METRICS] + \
            [f"{m}|har|{met}" for m in MODELS if m != "har" for met in ("crps",)]
    for h in cfg["horizons"]:
        md += [f"## {h} 時間\n", "| 比較 | 指標 | 改善の平均（×10⁻⁴） | 標準偏差 | 5% 点 | 95% 点 | 最大 | 良い側で有意 | 悪い側で有意 |",
               "|---|---|---|---|---|---|---|---|---|"]
        for k in comps:
            vals = [v for v in (r["summary"]["horizons"][str(h)]["comps"].get(k) for r in reps) if v]
            if not vals:
                continue
            imp = np.array([-v["mean"] for v in vals]) * 1e4
            a, b, m = k.split("|")
            md.append(f"| {LABELS[a]} − {LABELS[b]} | {METRIC_JA[m]} | {imp.mean():+.3f} | {imp.std(ddof=1) if len(imp) > 1 else 0:.3f} | "
                      f"{np.quantile(imp, 0.05):+.3f} | {np.quantile(imp, 0.95):+.3f} | {imp.max():+.3f} | "
                      f"{pct(np.mean([significant(v, better=True) for v in vals]), 0)} | {pct(np.mean([significant(v, better=False) for v in vals]), 0)} |")
        md += ["", f"### {h} 時間: 合成データの種類ごと（主指標 {METRIC_JA[cfg['judge']['primary']]}）\n",
               "| 種類 | 比較 | 改善の平均（×10⁻⁴） | 最大 | 良い側で有意 | フォレストの AUC P(>0) の平均 | フォレスト − HAR 型の CRPS（相対。正ならフォレストが悪い） |",
               "|---|---|---|---|---|---|---|"]
        for kind in cfg["null_audit"]["models"]:
            rs = [r["summary"]["horizons"][str(h)] for r in reps if r["kind"] == kind]
            if not rs:
                continue
            width = np.mean([H["comps"]["forest|har|crps"]["mean"] / H["rows"]["har"]["crps"] for H in rs])
            auc = np.mean([H["rows"]["forest"]["auc_0"] for H in rs])
            for a, b in cfg["judge"]["comparisons"]:
                vals = [H["comps"][f"{a}|{b}|{cfg['judge']['primary']}"] for H in rs]
                imp = np.array([-v["mean"] for v in vals]) * 1e4
                md.append(f"| {kind} | {LABELS[a]} − {LABELS[b]} | {imp.mean():+.3f} | {imp.max():+.3f} | "
                          f"{sum(significant(v, better=True) for v in vals)} / {len(vals)} | {auc:.3f} | {pct(width, 2)} |")
        md.append("")
    fam = [(str(h), a, b) for h in cfg["horizons"] for a, b in cfg["judge"]["comparisons"]]
    hits = 0
    for r in reps:
        cs = [r["summary"]["horizons"][h]["comps"][f"{a}|{b}|{cfg['judge']['primary']}"] for h, a, b in fam]
        pf, pe = holm([c["p_fb"] for c in cs]), holm([c["p_ewc"] for c in cs])
        hits += any(c["mean"] < 0 and x < cfg["judge"]["alpha"] and y < cfg["judge"]["alpha"] for c, x, y in zip(cs, pf, pe))
    md.append(f"4 つの主比較をホルム法でまとめた判定（帰無の p を除く）で「上乗せあり」になった回: {hits} / {len(reps)}\n")
    (OUT / "null.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# ------------------------------------------------------------------ 評価（1 回だけ）

def main_eval(cfg: dict) -> None:
    if not cfg.get("owner_approved"):
        sys.exit("configs/dist_base.yaml の owner_approved が空。オーナーの承認の前は評価しない。")
    null_path = OUT / "null.json"
    null = json.loads(null_path.read_text()) if null_path.exists() else None
    if (not null or null.get("config_hash") != cfg_hash(cfg) or null.get("code_hash") != code_hash()
            or null.get("n_reps", 0) < null.get("expected", 1)):
        sys.exit("帰無の監査（reports/dist_base/null.json）がないか、設定かコードのハッシュが今と違うか、回数が足りない。"
                 "監査をやり直してから評価する。")
    h1 = load_hourly(cfg)
    mc, _ = load_minutes(cfg, h1)
    out = run_pipeline(h1, mc, cfg)
    S = summarize(out, cfg, full=True)
    J = judge(S, null, cfg)
    th = cfg_hash(cfg)
    run_at = datetime.now(timezone.utc).isoformat()
    log = Path("reports/experiments.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as fh:
        for h, H in S["horizons"].items():
            for m in MODELS:
                r = H["rows"][m]
                fh.write(json.dumps(jsonable({"run_at": run_at, "stage": "dist_base", "trial_hash": th,
                                              "code_version": os.environ.get("GITHUB_SHA"), "key": f"dist_base/h{h}/{m}",
                                              "crps": r["crps"], "pinball": r["pinball"], "brier_0": r["brier_0"],
                                              "auc_0": r["auc_0"], "payoff": r["payoff"]})) + "\n")
    metrics = {"run_at": run_at, "trial_hash": th, "code_hash": code_hash(), "owner_approved": cfg["owner_approved"], "judge": J,
               "null_reps": (null or {}).get("n_reps", 0), "summary": S}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(jsonable(metrics), ensure_ascii=False, indent=1))
    write_report(cfg, metrics)


def write_report(cfg: dict, m: dict) -> None:
    S, J = m["summary"], m["judge"]
    md = [f"# 評価の土台（候補 1）: ボラだけの基準に対する幅と向き（{cfg['pair']}、{'・'.join(str(h) for h in cfg['horizons'])} 時間）\n",
          f"- 設定のハッシュ {m['trial_hash']}（configs/dist_base.yaml、オーナー承認 {m['owner_approved']}）。学習は {cfg['split']} より前で 1 回だけ。"
          f"評価期間（{cfg['split']} 以降）はこれまでの分布推定で 5 回使っているので、結果は診断として扱う（売買の判断には使わない）",
          "- 分布: " + "、".join(LABELS[x] for x in MODELS) + "。説明は bbresearch/distbase.py の冒頭",
          f"- 帰無の監査: 期待リターン 0 の合成データ {m['null_reps']} 回（reports/dist_base/null.md）",
          f"- 検定: 損失差の平均が 0。fixed-b（Bartlett、M = ⌈{cfg['tests']['nw_mult']}√P⌉）、EWC（B = ⌊{cfg['tests']['ewc_mult']}P^(2/3)⌋）、"
          "定常ブートストラップ。p はすべて両側\n",
          "## 判定（事前登録）\n",
          f"向きの上乗せ: 主指標 {METRIC_JA[cfg['judge']['primary']]} の差（向きのモデル − 参照）が負で、4 検定のホルム補正後に fixed-b と EWC の"
          f"両方で p < {cfg['judge']['alpha']}、かつ帰無の監査の p < {cfg['judge']['null_alpha']}。\n",
          "| ホライズン | 向きのモデル | 参照 | 差（×10⁻⁴） | fixed-b t | p（ホルム） | EWC t | p（ホルム） | 帰無の p | 帰無で良い側に有意 | 上乗せ |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in J["direction"]:
        md.append(f"| {r['h']} 時間 | {LABELS[r['model']]} | {LABELS[r['reference']]} | {r['diff'] * 1e4:+.3f} | {num(r['t_fb'])} | "
                  f"{pv(r['p_fb_holm'])} | {num(r['t_ewc'])} | {pv(r['p_ewc_holm'])} | {pv(r['p_null'])}（{r['n_null']} 回） | "
                  f"{r['null_false_pos']} / {r['n_null']} | {'あり' if r['increment'] else 'なし'} |")
    md += ["", "幅（CRPS、" + f"{LABELS[cfg['judge']['width_pair'][0]]} − {LABELS[cfg['judge']['width_pair'][1]]}、fixed-b の両側 5%）:\n",
           "| ホライズン | 差 | 相対 | fixed-b t | p | EWC の p | 判定 |", "|---|---|---|---|---|---|---|"]
    md += [f"| {w['h']} 時間 | {w['diff']:+.2e} | {pct(w['rel'], 2)} | {num(w['t_fb'])} | {pv(w['p_fb'])} | {pv(w['p_ewc'])} | {w['verdict']} |"
           for w in J["width"]]
    md += ["", f"結論: 向きの上乗せは{'ある（前向きの期間で確かめる）' if J['any_increment'] else '確認できない'}。\n"]
    for h, H in S["horizons"].items():
        R, C = H["rows"], H["comps"]
        md += [f"## {h} 時間先\n",
               f"- 学習 {H['n_train']:,} 行（{H['train_start'][:10]} 〜 {H['train_end'][:10]}）、評価 {H['n_test']:,} 行（{H['test_start'][:10]} 〜 {H['test_end'][:10]}）",
               f"- HAR 型の学習期間の R² {num(H['har_r2'], 3)}。符号の分類器の木の数（学習期間の末尾で早期停止）{H['sm_trees']}\n",
               "### 分布の精度（幅。小さいほど良い。HAR 型との差の検定）\n",
               "| 分布 | ピンボールの平均 | CRPS | 90% 区間の的中率 | 90% 区間の幅の中央値 | CRPS の差（対 HAR） | fixed-b t | p | EWC の p | ブートストラップの p |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for mm in MODELS:
            r = R[mm]
            c = C.get(f"{mm}|har|crps")
            md.append(f"| {LABELS[mm]} | {f6(r['pinball'])} | {f6(r['crps'])} | {pct(r['interval90'])} | {pct(r['width90_median'], 2)} | "
                      + (f"{pct(c['mean'] / R['har']['crps'], 2)} | {num(c['t_fb'])} | {pv(c['p_fb'])} | {pv(c['p_ewc'])} | {pv(c.get('p_sb'))} |"
                         if c else "- | - | - | - | - |"))
        md += ["", "### 向き（P(収益率 > c) の精度。小さいほど良い。AUC は 0.5 が情報なし）\n",
               "| 分布 | ブライア c=0 | c=+費用 | c=−費用 | 対数損失 c=0 | AUC c=0 | AUC +費用 | AUC −費用 | 50% のピンボール | MCB | DSC |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for mm in MODELS:
            r = R[mm]
            md.append(f"| {LABELS[mm]} | {r['brier_0']:.5f} | {r['brier_1']:.5f} | {r['brier_2']:.5f} | {r['logloss_0']:.5f} | {num(r['auc_0'], 3)} | "
                      f"{num(r['auc_1'], 3)} | {num(r['auc_2'], 3)} | {f6(r['pinball50'])} | {r['corp_0']['mcb']:.5f} | {r['corp_0']['dsc']:.5f} |")
        md += ["", "### 向きの比較（差 = 向きのモデル − 参照。負なら向きのモデルが良い。副指標は判定に使わない）\n",
               "| 比較 | 指標 | 差 | fixed-b t | p | EWC t | p | ブートストラップの p |", "|---|---|---|---|---|---|---|---|"]
        pairs = [tuple(p) for p in cfg["judge"]["comparisons"]] + [("forest", "har")]
        for a, b in pairs:
            for met in COMP_METRICS:
                c = C[f"{a}|{b}|{met}"]
                md.append(f"| {LABELS[a]} − {LABELS[b]} | {METRIC_JA[met]} | {c['mean']:+.2e} | {num(c['t_fb'])} | {pv(c['p_fb'])} | "
                          f"{num(c['t_ewc'])} | {pv(c['p_ewc'])} | {pv(c.get('p_sb'))} |")
        md += ["", f"### 費用の水準の採点（予測平均が {pct(cfg['elementary_theta'], 1)} を超えたら買い、{h} 本後に売る。判断 1 回あたりの費用控除後の損益）\n",
               "| 分布 | 買う割合 | 1 回あたりの損益 | t（fixed-b） | " + " | ".join(f"θ={pct(float(x), 1)}" for x in cfg["murphy_thetas"]) + " |",
               "|---|---|---|---|" + "---|" * len(cfg["murphy_thetas"])]
        for mm in MODELS:
            r = R[mm]
            md.append(f"| {LABELS[mm]} | {pct(r['buy_share'], 2)} | {pct(r['payoff'], 4)} | {num((r['payoff_test'] or {}).get('t_fb'))} | "
                      + " | ".join(pct(r["murphy"][str(float(x))], 4) for x in cfg["murphy_thetas"]) + " |")
        md += ["", f"### 予測平均の上位 {cfg['top_share']:.0%} と下位 {cfg['top_share']:.0%} の実現リターンの差\n",
               "| 分布 | 差 | fixed-b t | p |", "|---|---|---|---|"]
        for mm in MODELS:
            sp = R[mm]["spread"]
            md.append(f"| {LABELS[mm]} | " + (f"{pct(sp['mean'], 3)} | {num(sp['t_fb'])} | {pv(sp['p_fb'])} |" if sp else "- | - | - |"))
        md += ["", f"### 裾（閾値 ±{pct(cfg['cost'], 1)} で重みを付けた CRPS。小さいほど良い）\n", "| 分布 | 上側 | 下側 |", "|---|---|---|"]
        md += [f"| {LABELS[mm]} | {R[mm]['tw_up']:.3e} | {R[mm]['tw_dn']:.3e} |" for mm in MODELS]
        md += ["", f"### 較正（分位点を下回った割合と、PIT の {h} 本の部分系列（t mod {h}）ごとの一様性の検定）\n",
               "| 分布 | " + " | ".join(f"{float(q):.0%}" for q in cfg["quantiles"]) + " | PIT の最小の p | ボンフェローニ |",
               "|---|" + "---|" * (len(cfg["quantiles"]) + 2)]
        for mm in MODELS:
            cov = R[mm]["coverage"]
            md.append(f"| {LABELS[mm]} | " + " | ".join(pct(cov[str(float(q))]) for q in cfg["quantiles"]) +
                      f" | {pv(H['pit'][mm]['min_p'])} | {pv(H['pit'][mm]['bonferroni'])} |")
        yrs = sorted({y for d in H["crps_by_year"].values() for y in d})
        md += ["", "### 年ごとの CRPS\n", "| 分布 | " + " | ".join(str(y) for y in yrs) + " |", "|---|" + "---|" * len(yrs)]
        md += [f"| {LABELS[mm]} | " + " | ".join(f6(H['crps_by_year'][mm].get(y, H['crps_by_year'][mm].get(str(y)))) for y in yrs) + " |" for mm in MODELS]
        md += ["", "### 年ごとのブライア c=0 の差（×10⁻⁴）\n", "| 比較 | " + " | ".join(str(y) for y in yrs) + " |", "|---|" + "---|" * len(yrs)]
        for k, d in H["brier0_diff_by_year"].items():
            a, b = k.split("|")
            md.append(f"| {LABELS[a]} − {LABELS[b]} | " + " | ".join(f"{(d.get(y, d.get(str(y))) or 0) * 1e4:+.2f}" for y in yrs) + " |")
        md += ["", "### 損失差の性質（自己相関、裾の指数、単位根検定）\n", "| 比較 | 指標 | 自己相関 " + " / ".join(str(k) for k in cfg["tests"]["acf_lags"]) +
               " | 裾の指数（Hill） | ADF の p | ブートストラップのブロック長 |", "|---|---|---|---|---|---|"]
        for key in [f"{a}|{b}|{cfg['judge']['primary']}" for a, b in cfg["judge"]["comparisons"]] + [f"{cfg['judge']['width_pair'][0]}|har|crps"]:
            c = C[key]
            a, b, met = key.split("|")
            md.append(f"| {LABELS[a]} − {LABELS[b]} | {METRIC_JA[met]} | " + " / ".join(num(v, 3) for v in c["acf"].values()) +
                      f" | {num(c['hill'], 2)} | {pv(c.get('adf_p'))} | {num(c.get('sb_block'), 1)} |")
        if "mcs" in H:
            md += ["", f"### モデル信頼集合（{cfg['tests']['mcs_size']:.0%}、ブロック {cfg['tests']['mcs_block']} 本）\n"]
            for met, r in H["mcs"].items():
                md.append(f"- {METRIC_JA[met]}: " + "、".join(f"{LABELS[k]}（p {pv(r['pvalues'].get(k))}）" for k in r["included"]))
        md.append("")
    md += ["## 注意\n",
           f"- 評価期間（{cfg['split']} 以降）は分布推定で何度も使っている。この結果は診断で、売買の判断には使わない。向きの上乗せが見えたら、"
           "前向きの期間で確かめる（2026-09-28 オーナー決定）。",
           "- HAR 型の 1 分の実現分散は、保存済みの約定の最後の価格から作った（約定のない分は直前の値）。",
           "- 検定は損失差の平均についてのもの。予測期間が重なるので、行は独立ではない。"]
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["check", "null-prepare", "null", "null-summary", "eval"], required=True)
    ap.add_argument("--config", default="configs/dist_base.yaml")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shard-dir", default=str(OUT))
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    t0 = time.monotonic()
    if args.mode == "check":
        main_check(cfg)
    elif args.mode == "null-prepare":
        main_null_prepare(cfg)
    elif args.mode == "null":
        main_null(cfg, args.shard)
    elif args.mode == "null-summary":
        main_null_summary(cfg, Path(args.shard_dir))
    else:
        main_eval(cfg)
    print(f"所要 {time.monotonic() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
