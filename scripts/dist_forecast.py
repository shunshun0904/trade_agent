"""1 時間足のテクニカル指標 → 4 時間後までの収益率の条件付き分布（分位点回帰フォレスト）。2026-09-27 オーナー指示。

    python scripts/dist_forecast.py [--config configs/dist.yaml] [--out reports/dist]

1. 公式 1 時間足（/candlestick/1hour/{YYYYMMDD}、公開 API）を日付ごとに取り、.cache/candles にためる。
2. bbresearch.indicators で指標を作り、split より前で学習、以後で 1 回評価する。
3. 分布の精度を、無条件の経験分布・直近 n 本の経験分布と比べる（ピンボール損失、CRPS、較正、90% 区間）。
4. 最後に全期間で学習し直し、最新の足での分布（今買った場合）を出す。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bbdata.client import BitbankAPIError, PublicClient
from bbresearch.indicators import GROUPS, forward_return, indicator_table
from bbresearch.qrf import (QuantileForest, crps_weighted, evaluate_quantiles, prob_exceed, reliability,
                            rolling_empirical)

CACHE = Path(".cache/candles")


def fetch_hourly(api: PublicClient, pair: str, start: str, end_exclusive: datetime) -> pd.DataFrame:
    """日付ごとの 1 時間足をキャッシュしながら集める。当日分（形成中）は取らない。"""
    d = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    rows, n_req, n_miss = [], 0, 0
    while d < end_exclusive:
        day = d.strftime("%Y%m%d")
        f = CACHE / pair / "1hour" / f"{day}.json"
        if f.exists():
            data = json.loads(f.read_text())
        else:
            try:
                data = api.candlestick(pair, "1hour", day)
            except BitbankAPIError:
                data = []
                n_miss += 1
            n_req += 1
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(data))
        rows += data
        d += timedelta(days=1)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume", "ts"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df.index = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    df = df.drop(columns="ts")
    df = df[~df.index.duplicated()].sort_index()
    print(f"1 時間足: {len(df):,} 本（{df.index[0]} 〜 {df.index[-1]}）、新規リクエスト {n_req}、データなしの日 {n_miss}")
    return df


def crps_uniform(y_sorted: np.ndarray, obs: np.ndarray, chunk: int = 2000) -> np.ndarray:
    out = np.empty(len(obs))
    w = np.full((1, len(y_sorted)), 1.0 / len(y_sorted))
    for s in range(0, len(obs), chunk):
        o = obs[s:s + chunk]
        out[s:s + chunk] = crps_weighted(y_sorted, np.repeat(w, len(o), axis=0), o)
    return out


def crps_rolling(y_train_tail: np.ndarray, y_test: np.ndarray, n: int, horizon: int) -> np.ndarray:
    y_all = np.concatenate([y_train_tail, y_test])
    start = len(y_train_tail)
    out = np.full(len(y_test), np.nan)
    for i in range(len(y_test)):
        end = start + i - horizon
        win = np.sort(y_all[max(0, end - n):end])
        if len(win) >= 20:
            out[i] = crps_weighted(win, np.full((1, len(win)), 1.0 / len(win)), y_test[i:i + 1])[0]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dist.yaml")
    ap.add_argument("--out", default="reports/dist")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    pair, horizon, cost = cfg["pair"], int(cfg["horizon"]), float(cfg["cost"])
    qs = tuple(float(q) for q in cfg["quantiles"])
    fp = cfg["forest"]

    api = PublicClient(min_interval=0.3)
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    df = fetch_hourly(api, pair, cfg["start"], today)
    full_idx = pd.date_range(df.index[0], df.index[-1], freq="h")
    gaps = len(full_idx) - len(df)
    feats = indicator_table(df)
    y = forward_return(df["close"], horizon)
    data = feats.join(y.rename("y")).dropna()
    split = pd.Timestamp(cfg["split"], tz="UTC")
    train = data[data.index < split - pd.Timedelta(hours=horizon)]   # 目的変数が評価期間にかからないように
    test = data[data.index >= split]
    cols = list(feats.columns)
    Xtr, ytr, Xte, yte = train[cols].to_numpy(), train["y"].to_numpy(), test[cols].to_numpy(), test["y"].to_numpy()
    print(f"学習 {len(train):,} 行（{train.index[0].date()} 〜 {train.index[-1].date()}）、評価 {len(test):,} 行、特徴量 {len(cols)}")

    # ---- 葉の最小サンプル数を学習期間の末尾 20% で選ぶ
    n_val = len(train) // 5
    val_scores = {}
    for leaf in fp["min_samples_leaf_grid"]:
        qf = QuantileForest(fp["n_estimators"] // 2, leaf, fp["max_features"], fp["max_samples"]).fit(
            Xtr[:-n_val - horizon], ytr[:-n_val - horizon])
        val_scores[leaf] = evaluate_quantiles(ytr[-n_val:], qf.predict_quantiles(Xtr[-n_val:], qs), qs)["pinball_mean"]
    best_leaf = min(val_scores, key=val_scores.get)
    print("検証（ピンボール損失の平均）: " + "、".join(f"葉 {k}: {v:.6f}" for k, v in val_scores.items()) + f" → 葉 {best_leaf}")

    # ---- 学習と評価
    qf = QuantileForest(fp["n_estimators"], best_leaf, fp["max_features"], fp["max_samples"]).fit(Xtr, ytr)
    q_model = qf.predict_quantiles(Xte, qs)
    crps_model = np.concatenate([crps_weighted(ys, w, yte[s:s + len(w)]) for s, (ys, w) in
                                 zip(range(0, len(Xte), 512), qf.predict_distribution(Xte, 512))])
    p_model = np.concatenate([prob_exceed(ys, w, cost) for ys, w in qf.predict_distribution(Xte, 512)])
    results = {"forest": evaluate_quantiles(yte, q_model, qs)}
    results["forest"]["crps"] = float(np.nanmean(crps_model))
    ytr_sorted = np.sort(ytr)
    q_unc = np.tile(np.quantile(ytr, qs), (len(yte), 1))
    results["unconditional"] = evaluate_quantiles(yte, q_unc, qs)
    results["unconditional"]["crps"] = float(np.mean(crps_uniform(ytr_sorted, yte)))
    p_unc = float(np.mean(ytr > cost))
    tail = ytr[-(max(cfg["baselines"]["rolling_n"]) + horizon + 10):]
    rolling_q = {}
    for n in cfg["baselines"]["rolling_n"]:
        q_r = rolling_empirical(tail, yte, n, horizon, qs)
        rolling_q[n] = q_r
        results[f"rolling{n}"] = evaluate_quantiles(yte, q_r, qs)
        results[f"rolling{n}"]["crps"] = float(np.nanmean(crps_rolling(tail, yte, n, horizon)))
    # 特徴量の群ごと（どの群が効くか。葉の最小は全体で選んだ値を使う）
    subset_rows = []
    for name, prefixes in (cfg.get("subsets") or {}).items():
        use = [i for i, col in enumerate(cols) if prefixes is None or any(col.startswith(pfx) for pfx in prefixes)]
        if not use or len(use) == len(cols):
            continue
        qf_s = QuantileForest(fp["n_estimators"], best_leaf, fp["max_features"], fp["max_samples"]).fit(Xtr[:, use], ytr)
        ev = evaluate_quantiles(yte, qf_s.predict_quantiles(Xte[:, use], qs), qs)
        subset_rows.append({"name": name, "n_features": len(use), "pinball_mean": ev["pinball_mean"], "interval90": ev["interval90"]})
        print(f"群 {name}: {len(use)} 個、ピンボール {ev['pinball_mean']:.6f}")
    # 年ごとのピンボール損失（フォレスト vs 直近 720 本）
    years = test.index.year
    by_year = []
    for yr in sorted(set(years)):
        m = years == yr
        a = evaluate_quantiles(yte[m], q_model[m], qs)["pinball_mean"]
        b = evaluate_quantiles(yte[m], rolling_q[cfg["baselines"]["rolling_n"][-1]][m], qs)["pinball_mean"]
        c = evaluate_quantiles(yte[m], q_unc[m], qs)["pinball_mean"]
        by_year.append({"year": int(yr), "n": int(m.sum()), "forest": a, "rolling": b, "unconditional": c})
    # P(r > cost) の信頼性
    hit = (yte > cost).astype(float)
    rel = reliability(p_model, hit)
    # 特徴量の重要度（不純度ベース）と群ごとの和
    imp = pd.Series(qf.rf.feature_importances_, index=cols).sort_values(ascending=False)
    group = imp.groupby(imp.index.str[0]).sum()

    # ---- 全期間で学習し直して「今」の分布
    qf_all = QuantileForest(fp["n_estimators"], best_leaf, fp["max_features"], fp["max_samples"]).fit(
        data[cols].to_numpy(), data["y"].to_numpy())
    last_feats = feats.dropna().iloc[[-1]]
    q_now = qf_all.predict_quantiles(last_feats.to_numpy(), qs)[0]
    p_now = float(next(prob_exceed(ys, w, cost) for ys, w in qf_all.predict_distribution(last_feats.to_numpy()))[0])
    p_now_up = float(next(prob_exceed(ys, w, 0.0) for ys, w in qf_all.predict_distribution(last_feats.to_numpy()))[0])
    q_now_unc = np.quantile(data["y"].to_numpy(), qs)

    # ---- 報告
    pct = lambda v: f"{v * 100:+.2f}%"  # noqa: E731
    md = [f"# 1 時間足の指標 → {horizon} 時間後までの収益率の分布（{pair}、分位点回帰フォレスト）\n",
          f"- データ: 公式 1 時間足 {df.index[0].date()} 〜 {df.index[-1].date()}（{len(df):,} 本、欠けた足 {gaps}）。"
          f"学習 {train.index[0].date()} 〜 {train.index[-1].date()}（{len(train):,} 行）、評価 {test.index[0].date()} 〜 {test.index[-1].date()}（{len(test):,} 行）",
          f"- 目的変数: 足 t の終値で買い {horizon} 本後の終値で売った対数収益率（費用は引いていない）。費用 {cost:.2%} を超える確率も出す",
          f"- 特徴量 {len(cols)} 個（" + "、".join(f"{GROUPS[g]} {sum(c.startswith(g + '_') for c in cols)}" for g in GROUPS) + "）。すべて足 t までのデータで計算（`tests/test_indicators.py`）。一覧は metrics.json の features",
          f"- フォレスト: 木 {fp['n_estimators']}、葉の最小 {best_leaf}（検証で選択: " + "、".join(f"{k}: {v:.6f}" for k, v in val_scores.items()) + f"）、特徴量の割合 {fp['max_features']}、標本 {fp['max_samples']}",
          "- 比較: 無条件 = 学習期間全体の経験分布。直近 n = その時点で確定している直近 n 本の収益率の経験分布（ボラの変化を追う基準）\n",
          "## 分布の精度（評価期間、値が小さいほど良い。CRPS と損失の単位は対数収益率）\n",
          "| 分布 | ピンボール損失の平均 | CRPS | 90% 区間の的中率（目標 90%） | 90% 区間の幅の中央値 | 直近 720 本に対する改善 |", "|---|---|---|---|---|---|"]
    base = results[f"rolling{cfg['baselines']['rolling_n'][-1]}"]["pinball_mean"]
    labels = {"forest": "フォレスト（指標で条件付け）", "unconditional": "無条件"}
    for k, r in results.items():
        name = labels.get(k, f"直近 {k[7:]} 本")
        md.append(f"| {name} | {r['pinball_mean']:.6f} | {r['crps']:.6f} | {r['interval90']:.1%} | {r['width90_median'] * 100:.2f}% | {1 - r['pinball_mean'] / base:+.1%} |")
    md.append("\n## 較正（予測した分位点を実際の収益率が下回った割合。目標は分位点そのもの）\n")
    md.append("| 分布 | " + " | ".join(f"{q:.0%}" for q in qs) + " |")
    md.append("|---|" + "---|" * len(qs))
    for k, r in results.items():
        md.append(f"| {labels.get(k, '直近 ' + k[7:] + ' 本')} | " + " | ".join(f"{r['coverage'][q]:.1%}" for q in qs) + " |")
    if subset_rows:
        md.append("\n## 特徴量の群ごとの精度（その群だけで学習。値が小さいほど良い）\n")
        md.append("| 群 | 特徴量の数 | ピンボール損失の平均 | 90% 区間の的中率 | 直近 720 本に対する改善 |\n|---|---|---|---|---|")
        md.append(f"| 全部 | {len(cols)} | {results['forest']['pinball_mean']:.6f} | {results['forest']['interval90']:.1%} | {1 - results['forest']['pinball_mean'] / base:+.1%} |")
        for r in subset_rows:
            md.append(f"| {r['name']} | {r['n_features']} | {r['pinball_mean']:.6f} | {r['interval90']:.1%} | {1 - r['pinball_mean'] / base:+.1%} |")
    md.append("\n## 年ごとのピンボール損失の平均\n")
    md.append("| 年 | 行数 | フォレスト | 直近 720 本 | 無条件 | 改善（対 720 本） |\n|---|---|---|---|---|---|")
    for r in by_year:
        md.append(f"| {r['year']} | {r['n']:,} | {r['forest']:.6f} | {r['rolling']:.6f} | {r['unconditional']:.6f} | {1 - r['forest'] / r['rolling']:+.1%} |")
    md.append(f"\n## 費用 {cost:.2%} を超える確率の信頼性（評価期間。無条件の確率は {p_unc:.1%}）\n")
    md.append("| 予測確率の区間 | 件数 | 予測の平均 | 実際に超えた割合 |\n|---|---|---|---|")
    for _, r in rel.iterrows():
        md.append(f"| {r['bin']} | {r['n']:,} | {r['p_mean']:.1%} | {r['hit_rate']:.1%} |" if r["n"] else f"| {r['bin']} | 0 | - | - |")
    md.append("\n## 特徴量の重要度（不純度ベース、上位 20）\n")
    md.append("| 特徴量 | 重要度 |\n|---|---|")
    for k, v in imp.head(20).items():
        md.append(f"| {k} | {v:.3f} |")
    md.append("\n群ごとの和: " + "、".join(f"{GROUPS.get(g, g)} {v:.2f}" for g, v in group.items()))
    md.append(f"\n## 今の推定（全期間で学習し直し、最新の足 {last_feats.index[0]} の終値で買った場合）\n")
    md.append("| | " + " | ".join(f"{q:.0%}" for q in qs) + f" | P(収益率 > 0) | P(収益率 > {cost:.2%}) |")
    md.append("|---|" + "---|" * (len(qs) + 2))
    md.append("| 条件付き | " + " | ".join(pct(v) for v in q_now) + f" | {p_now_up:.1%} | {p_now:.1%} |")
    md.append("| 無条件（全期間） | " + " | ".join(pct(v) for v in q_now_unc) + f" | {float(np.mean(data['y'] > 0)):.1%} | {float(np.mean(data['y'] > cost)):.1%} |")
    md.append("\n注意: 評価は 2024 年以降の 1 回だけ。年ごとの改善が安定しているかを見る。分布の改善が費用を超える売買につながるかは別の検証（売買規則）が要る。")
    text = "\n".join(md) + "\n"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(text)
    (out / "metrics.json").write_text(json.dumps({
        "results": {k: {kk: ({str(q): v for q, v in vv.items()} if isinstance(vv, dict) else vv) for kk, vv in r.items()} for k, r in results.items()},
        "by_year": by_year, "best_leaf": best_leaf, "val_scores": val_scores, "importance": imp.round(5).to_dict(),
        "subsets": subset_rows, "features": cols,
        "reliability": rel.to_dict("records"),
        "now": {"as_of": str(last_feats.index[0]), "quantiles": dict(zip([str(q) for q in qs], q_now.tolist())),
                "p_up": p_now_up, "p_exceed_cost": p_now}}, ensure_ascii=False, indent=1))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
