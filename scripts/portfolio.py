"""ボラティリティを抑えるポートフォリオの前向き検証（公式日足、認証不要）。

    python scripts/portfolio.py

候補は直近の約定数が多い JPY ペア（reports/activity/report.md の上位）。年ごとに 1 リクエスト（1day）。
結果は reports/portfolio/report.md と weights.json。設計は bbresearch/portfolio.py の冒頭。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bbdata.client import BitbankAPIError, PublicClient
from bbresearch.portfolio import best_combination, shrunk_cov, shrunk_mean, summary, walk_forward

POOL = ["btc_jpy", "xrp_jpy", "eth_jpy", "doge_jpy", "sol_jpy", "xlm_jpy", "link_jpy", "ada_jpy", "bcc_jpy",
        "ltc_jpy", "avax_jpy", "trx_jpy", "bnb_jpy", "dot_jpy", "pol_jpy"]
START, K_MAX, W_MAX, COST = "2021-01-01", 5, 0.4, 0.0015
TARGETS = (None, 1.0, 1.5)


def fetch_daily(api: PublicClient, pair: str, years: range) -> pd.Series:
    rows = []
    for y in years:
        try:
            rows += api.candlestick(pair, "1day", str(y))
        except BitbankAPIError:
            continue
    if not rows:
        return pd.Series(dtype=float, name=pair)
    df = pd.DataFrame(rows, columns=["o", "h", "l", "c", "v", "ts"])
    s = pd.Series(df["c"].astype(float).to_numpy(), index=pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True),
                  name=pair)
    return s[~s.index.duplicated()].sort_index()


def main() -> None:
    api = PublicClient(min_interval=0.5)
    years = range(2019, datetime.now(timezone.utc).year + 1)
    close = pd.concat([fetch_daily(api, p, years) for p in POOL], axis=1)
    close = close[close.index < pd.Timestamp.now(tz="UTC").normalize()]  # 当日（形成中）は除く
    res = walk_forward(close, START, K_MAX, W_MAX, TARGETS, cost=COST)
    stats = {s: summary(res.daily[s]) for s in res.daily.columns}
    # 直近 365 日で推定した「今」の重み
    ret = np.log(close).diff()
    est = ret[ret.index >= close.index[-1] - pd.Timedelta(days=365)]
    avail = [c for c in close.columns if est[c].notna().sum() >= 330]
    R = est[avail].dropna().to_numpy()
    cov, mu = shrunk_cov(R), shrunk_mean(R)
    now = {}
    for m in TARGETS:
        b = best_combination(cov, mu, avail, K_MAX, None if m is None else float(mu.mean()) * m, W_MAX)
        key = "minvar" if m is None else f"minvar_x{m}"
        now[key] = None if b is None else {"weights": dict(zip(b["names"], [round(float(x), 4) for x in b["weights"]])),
                                          "ann_vol": float(np.sqrt(b["var"] * 365))}
    label = {"minvar": "最小分散（目標なし）", "minvar_x1.0": "最小分散（目標 = 均等配分の期待リターン）",
             "minvar_x1.5": "最小分散（目標 = 均等配分 × 1.5）", "equal": "均等配分（候補全体）", "btc_jpy": "BTC 単独"}
    md = [f"# ボラティリティを抑えるポートフォリオ（{START} 〜 {close.index[-1].date()}、毎月再計算、日足）\n",
          f"- 候補: {', '.join(POOL)}（各月、直前 365 日の 9 割以上のデータがある銘柄だけ）",
          f"- 制約: {K_MAX} 銘柄以内、1 銘柄 {W_MAX:.0%} まで、ロングのみ、全額投資。売買費用は回転率 × {COST:.2%}",
          "- 共分散は Ledoit-Wolf の縮小推定、期待リターンは標本平均を横断平均へ 50% 縮めたもの。目標は均等配分の期待リターンの倍率\n",
          "## 前向き検証（各月の重みは、その月より前のデータだけで決めている）\n",
          "| 戦略 | 年率ボラ | 年率リターン | シャープ | 最大ドローダウン | 最悪の日 | 累積 |", "|---|---|---|---|---|---|---|"]
    for s, st in stats.items():
        md.append(f"| {label.get(s, s)} | {st['ann_vol']:.1%} | {st['ann_return']:.1%} | {st['sharpe']:.2f} | "
                  f"{st['max_drawdown']:.1%} | {st['worst_day']:.1%} | {st['total']:+.0%} |")
    # 年ごとのボラ
    md.append("\n## 年ごとの年率ボラ\n")
    md.append("| 年 | " + " | ".join(label.get(s, s) for s in stats) + " |")
    md.append("|---|" + "---|" * len(stats))
    for y, g in res.daily.groupby(res.daily.index.year):
        md.append(f"| {y} | " + " | ".join(f"{g[s].std(ddof=1) * np.sqrt(365):.1%}" for s in stats) + " |")
    # 選ばれた回数
    md.append("\n## 選ばれた回数（最小分散、目標 = 均等配分の期待リターン）\n")
    cnt: dict[str, int] = {}
    for w in res.weights["minvar_x1.0"]:
        for k in w["weights"]:
            cnt[k] = cnt.get(k, 0) + 1
    n_m = len(res.weights["minvar_x1.0"])
    md.append("| 銘柄 | 回数 / 月数 |\n|---|---|")
    for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
        md.append(f"| {k} | {v} / {n_m} |")
    md.append("\n## 直近 365 日で推定した今の重み（参考。次の月初まで有効な想定）\n")
    md.append("| 戦略 | 重み | 推定の年率ボラ |\n|---|---|---|")
    for k, v in now.items():
        if v is None:
            md.append(f"| {label[k]} | 目標に届く組み合わせなし | - |")
        else:
            md.append(f"| {label[k]} | " + "、".join(f"{p} {w:.0%}" for p, w in v["weights"].items()) + f" | {v['ann_vol']:.1%} |")
    md.append("\n注意: 期待リターンの推定は誤差が大きい。目標付きの結果は目標なしより過去に合わせた度合いが強い。"
              "候補は 2026-09-25 の約定数で選んでおり、上場の遅い銘柄は途中から入る。")
    text = "\n".join(md) + "\n"
    out = Path("reports/portfolio")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(text)
    (out / "weights.json").write_text(json.dumps({"walk_forward": res.weights, "now": now, "stats": stats},
                                                 ensure_ascii=False, indent=1))
    print(text)


if __name__ == "__main__":
    main()
