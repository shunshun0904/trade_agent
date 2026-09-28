"""候補 2（24 時間の新しい情報源）の研究。2026-09-28 オーナー決定。

    python scripts/ext_study.py --mode check    # データの確認（期間・抜け・書式、事象の数）。先の収益率は一切出さない

データは bbresearch/extdata.py（Binance のアーカイブ、BitMEX、Deribit、Dukascopy、FRED）と bitbank の公式 1 時間足
（scripts/dist_forecast.py と同じキャッシュ）。事前登録の数値はこの確認の後に configs/ext_study.yaml に固定し、
オーナーの承認後に 1 回だけ評価する（評価の部分は承認の前に加える）。
信号はすべて、値が分かる時刻（資金調達率は決済の時刻、1 時間足の値は足が閉じる時刻）を index にする。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bbresearch import extdata as xd

OUT = Path("reports/ext_study")
SETTLE_HOURS = (0, 8, 16)           # Binance の資金調達の決済の時刻（UTC）。Deribit の記録もこの時刻の分を使う


def pct(x, nd=2) -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x * 100 + 0.0:.{nd}f}%"


def load_all(cfg: dict, fetcher: xd.Fetcher, hourly=None) -> dict:
    """評価に使うすべての系列。hourly は bitbank の 1 時間足を作る関数（テストで差し替える）。"""
    s, e = cfg["start"], cfg["end"]
    sym = cfg["binance_symbol"]
    if hourly is None:
        from bbdata.client import PublicClient
        from scripts.dist_forecast import fetch_hourly

        def hourly(start, end):
            return fetch_hourly(PublicClient(min_interval=0.3), cfg["pair"], start, pd.Timestamp(end, tz="UTC").to_pydatetime())
    d = {"bitbank": hourly(s, e)}
    d["binance_spot"] = xd.binance_klines(fetcher, "spot", "klines", sym, "1h", s, e)
    d["binance_perp"] = xd.binance_klines(fetcher, "futures/um", "klines", sym, "1h", s, e)
    d["binance_premium"] = xd.binance_klines(fetcher, "futures/um", "premiumIndexKlines", sym, "1h", s, e)
    d["funding_binance"] = xd.binance_funding(fetcher, sym, s, e)
    d["funding_bitmex"] = xd.bitmex_funding(fetcher, s, e)
    d["funding_deribit"] = xd.deribit_funding(fetcher, s, e)
    d["metrics"] = xd.binance_metrics(fetcher, sym, cfg.get("metrics_start", s), e)
    d["usdjpy"] = xd.dukascopy_hourly(fetcher, s, e)
    d["fred"] = xd.fred_series(fetcher)
    return d


def coverage(name: str, x, freq: str | None, start: str, end: str, offset: str = "0h") -> dict:
    """件数・最初と最後・（間隔が決まっている系列は）期待する時刻に対する抜けと余分。"""
    n = len(x)
    row = {"name": name, "n": n, "first": str(x.index[0]) if n else "-", "last": str(x.index[-1]) if n else "-"}
    if freq and n:
        full = pd.date_range(pd.Timestamp(start, tz="UTC") + pd.Timedelta(offset), pd.Timestamp(end, tz="UTC"), freq=freq,
                             inclusive="left")
        row["expected"] = len(full)
        row["missing"] = int(len(full.difference(x.index)))
        row["extra"] = int(len(x.index.difference(full)))
    return row


def raw_signals(d: dict) -> dict[str, pd.Series]:
    """信号の元の値（順位にする前）。index は値が分かる時刻。"""
    out = {"funding_binance": d["funding_binance"], "funding_bitmex": d["funding_bitmex"]}
    de = d["funding_deribit"]["interest_8h"] if len(d["funding_deribit"]) else pd.Series(dtype=float)
    out["funding_deribit"] = de[de.index.hour.isin(SETTLE_HOURS) & (de.index.minute == 0)].rename("funding_deribit")
    prem = xd.jpy_premium(d["bitbank"]["close"], d["binance_spot"]["close"], d["usdjpy"]["close"]).dropna()
    out["jpy_premium"] = xd.available(prem)
    out["carry"] = xd.available(d["binance_premium"]["close"].rename("carry"))
    m = d["metrics"]
    if len(m):
        oi = m["sum_open_interest"].resample("1h").last()          # [t, t + 1h) の最後の記録。t + 1h に分かる
        out["oi_change"] = xd.available(np.log(oi).diff(24).dropna().rename("oi_change"))
    return out


def signals(d: dict, cfg: dict) -> dict[str, pd.Series]:
    """判断の時刻に分かる値だけで作る信号: 直前 window の値の中での順位（自分自身は含めない）。先の収益率は使わない。"""
    w, days = cfg["window"], cfg.get("min_days", 90)
    out = {}
    for name, x in raw_signals(d).items():
        per_day = 3 if name.startswith("funding_") else 24
        out[name] = xd.trailing_rank(x.dropna(), w, min_periods=days * per_day)
    return out


def episodes(t: pd.DatetimeIndex, gap: str = "24h") -> int:
    """事象の時刻を、前の事象から gap より離れたものだけ数える（24 時間先を見る評価では、続けて起きる事象は独立でない）。"""
    if len(t) == 0:
        return 0
    return int(1 + ((t[1:] - t[:-1]) > pd.Timedelta(gap)).sum())


def main_check(cfg: dict, fetcher: xd.Fetcher | None = None, hourly=None) -> dict:
    f = fetcher or xd.Fetcher(cfg.get("cache", ".cache/ext"), strict=False)     # 取れなかったファイルは報告に出して続ける
    t0 = time.monotonic()
    d = load_all(cfg, f, hourly)
    s, e = cfg["start"], cfg["end"]
    ms = cfg.get("metrics_start", s)
    cov = [coverage("bitbank BTC/JPY 1 時間足", d["bitbank"], "1h", s, e),
           coverage("Binance 現物 BTCUSDT 1 時間足", d["binance_spot"], "1h", s, e),
           coverage("Binance 永久先物 1 時間足", d["binance_perp"], "1h", s, e),
           coverage("Binance プレミアム指数 1 時間足", d["binance_premium"], "1h", s, e),
           coverage("Binance 資金調達率（0・8・16 時）", d["funding_binance"], "8h", s, e),
           coverage("BitMEX 資金調達率（4・12・20 時）", d["funding_bitmex"], "8h", s, e, offset="4h"),
           coverage("Deribit 資金調達率（1 時間ごと）", d["funding_deribit"], "1h", s, e),
           coverage(f"Binance 建玉など（5 分、{ms} から探す）", d["metrics"], "5min", ms, e),
           coverage("Dukascopy ドル円 1 時間足（中値）", d["usdjpy"], "1h", s, e),
           coverage("FRED ドル円（日次）", d["fred"][d["fred"].index >= pd.Timestamp(s, tz="UTC")], None, s, e)]
    samples = {k: (v.head(2).reset_index().astype(str).values.tolist() if hasattr(v, "head") else []) for k, v in d.items()}
    fb = d["funding_binance"]
    gaps = pd.Series((fb.index[1:] - fb.index[:-1]) / pd.Timedelta(hours=1)).round(2).value_counts().sort_index() if len(fb) > 1 else pd.Series(dtype=int)
    funding_gaps = {f"{k:g}h": int(v) for k, v in gaps.items()}

    sig = signals(d, cfg)
    qs = cfg["check_thresholds"]
    counts = []
    for name, r in sig.items():
        r = r.dropna()
        for yr, g in r.groupby(r.index.year):
            row = {"signal": name, "year": int(yr), "n": int(len(g))}
            for q in qs:
                for side, mask in (("low", g <= q), ("high", g >= 1 - q)):
                    row[f"{side}{q}"] = int(mask.sum())
                    row[f"{side}{q}_ep"] = episodes(g.index[mask.to_numpy()])
            counts.append(row)
    first_signal = {k: (str(v.dropna().index[0]) if v.notna().any() else "-") for k, v in sig.items()}

    # 信号どうしの順位相関（Binance の決済の時刻に、その時刻に分かっている最新の値をそろえる）
    at = sig["funding_binance"].dropna().index
    aligned = pd.DataFrame({k: v.dropna().reindex(at, method="ffill", tolerance=pd.Timedelta("8h")) for k, v in sig.items()})
    sig_corr = aligned.corr(method="spearman", min_periods=100).round(3)

    # 取引所の間の資金調達率の相関（日ごとの平均）
    daily = pd.DataFrame({"binance": d["funding_binance"].resample("1D").mean(),
                          "bitmex": d["funding_bitmex"].resample("1D").mean(),
                          "deribit": (d["funding_deribit"]["interest_8h"].resample("1D").mean()
                                      if len(d["funding_deribit"]) else pd.Series(dtype=float))}).dropna()
    corr = daily.corr().round(3).to_dict() if len(daily) > 2 else {}

    # Deribit の interest_8h と interest_1h の関係（8 倍か、直前 8 時間の和か）
    der = {}
    if len(d["funding_deribit"]) > 8:
        i8, i1 = d["funding_deribit"]["interest_8h"], d["funding_deribit"]["interest_1h"]
        full = i1.reindex(pd.date_range(i1.index[0], i1.index[-1], freq="1h"))
        sum8 = full.rolling(8, min_periods=8).sum().reindex(i1.index)
        ok = i8.abs() > 1e-8
        der = {"corr_8x1h": float(np.corrcoef(i8, 8 * i1)[0, 1]),
               "corr_sum8_1h": float(pd.concat([i8, sum8], axis=1).dropna().corr().iloc[0, 1]),
               "median_ratio_8h_to_1h": float((i8[ok] / i1[ok]).replace([np.inf, -np.inf], np.nan).median()),
               "median_abs_diff_8x1h": float((i8 - 8 * i1).abs().median()),
               "median_abs_diff_sum8": float((i8 - sum8).abs().median())}

    # ドル円の 2 つの出どころの照合: FRED（ニューヨーク正午）と、その時刻に閉じる Dukascopy の足の終値
    fx = d["usdjpy"]["close"]
    fred = d["fred"][d["fred"].index >= pd.Timestamp(s, tz="UTC")]
    noon = (fred.index.tz_localize(None).normalize() + pd.Timedelta(hours=12)).tz_localize("America/New_York").tz_convert("UTC")
    fx_at = pd.Series(fx.reindex(noon - pd.Timedelta(hours=1)).to_numpy(), index=fred.index)
    fxcmp = pd.DataFrame({"dukascopy": fx_at, "fred": fred}).dropna()
    fx_diff = (fxcmp["dukascopy"] / fxcmp["fred"] - 1).abs()
    fx_row = {"n": int(len(fxcmp)), "median_abs_diff": float(fx_diff.median()) if len(fx_diff) else None,
              "p99_abs_diff": float(fx_diff.quantile(0.99)) if len(fx_diff) else None}

    raw = raw_signals(d)
    prem = raw["jpy_premium"]
    prem_q = {int(y): g.quantile([0.05, 0.5, 0.95]).round(5).tolist() for y, g in prem.groupby(prem.index.year)}
    fund_q = {int(y): g.quantile([0.05, 0.1, 0.5, 0.9, 0.95]).round(6).tolist() for y, g in fb.groupby(fb.index.year)}

    # 事象の直前 24 時間の bitbank の値動き（過去の値動きだけ）: 2σ を超える下げ・上げの後だった割合
    c = d["bitbank"]["close"]
    lc = np.log(c)
    sd24 = lc.diff().rolling(24 * 30, min_periods=24 * 20).std() * math.sqrt(24)
    z24 = xd.available(lc.diff(24) / sd24)                       # 足が閉じる時刻に分かる
    q_main = max(qs)
    prior = []
    for name, r in sig.items():
        r = r.dropna()
        z = z24.reindex(r.index)
        for side, mask in (("all", pd.Series(True, index=r.index)), ("low", r <= q_main), ("high", r >= 1 - q_main)):
            zz = z[mask.to_numpy()].dropna()
            prior.append({"signal": name, "side": side, "n": int(len(zz)),
                          "after_drop": float((zz <= -2).mean()) if len(zz) else None,
                          "after_rise": float((zz >= 2).mean()) if len(zz) else None})

    chk = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "requests": f.n_requests,
           "seconds": round(time.monotonic() - t0), "config": {k: cfg.get(k) for k in ("start", "end", "window", "min_days", "check_thresholds",
                                                                                      "metrics_start", "binance_symbol", "pair")},
           "coverage": cov, "funding_binance_gaps": funding_gaps, "first_signal": first_signal, "samples": samples,
           "event_counts": counts, "signal_corr": sig_corr.to_dict(), "funding_corr_daily": corr, "deribit_interest": der,
           "fx_dukascopy_vs_fred": fx_row, "premium_quantiles_by_year": prem_q, "funding_quantiles_by_year": fund_q,
           "prior_move": prior, "failures": f.failures}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check.json").write_text(json.dumps(chk, ensure_ascii=False, indent=1, default=str))

    md = ["# 候補 2: データの確認（期間・抜け・書式、事象の数。先の収益率は出さない）\n",
          f"- 期間 {s} 〜 {e}（UTC、終わりは含まない）。新しい要求 {f.n_requests:,} 件、所要 {chk['seconds']} 秒",
          f"- 信号はすべて、値が分かる時刻（資金調達率は決済の時刻、1 時間足の値は足が閉じる時刻）での、直前 {cfg['window']} の値の中の順位"
          f"（自分自身は含めない）。直前の履歴が {cfg.get('min_days', 90)} 日分に満たない間は出さない",
          "- 信号: funding_*（各取引所の資金調達率。Deribit は 0・8・16 時の記録の interest_8h）、jpy_premium（log bitbank − log(Binance 現物 × ドル円)）、"
          "carry（Binance のプレミアム指数の 1 時間足の終値）、oi_change（Binance の建玉の 24 時間の対数変化）\n",
          "## データ\n", "| 系列 | 件数 | 最初 | 最後 | 期待する件数 | 抜け | 余分 |", "|---|---|---|---|---|---|---|"]
    fails = [f"- 取れなかったファイル {len(f.failures)} 件" + ("" if not f.failures else "（最初の 20 件）:")]
    fails += [f"  - {x['url']}: {x['status'][:120]}" for x in f.failures[:20]]
    md += [f"| {r['name']} | {r['n']:,} | {r['first'][:16]} | {r['last'][:16]} | {r.get('expected', '-')} | {r.get('missing', '-')} | {r.get('extra', '-')} |"
           for r in cov]
    md += [""] + fails
    md += ["", "- Binance の資金調達の間隔（時間: 回数）: " + "、".join(f"{k}: {v:,}" for k, v in funding_gaps.items()),
           "- 信号が出始める時刻: " + "、".join(f"{k} {v[:16]}" for k, v in first_signal.items()),
           "", "## 事象の数（年ごと。下位 q = 順位が q 以下、上位 q = 1 − q 以上。括弧内は前の事象から 24 時間より離れたものだけ数えた数）\n",
           "| 信号 | 年 | 件数 | " + " | ".join(f"下位 {q:.0%} | 上位 {q:.0%}" for q in qs) + " |",
           "|---|---|---|" + "---|---|" * len(qs)]
    md += [f"| {r['signal']} | {r['year']} | {r['n']:,} | " +
           " | ".join(f"{r[f'low{q}']:,}（{r[f'low{q}_ep']}） | {r[f'high{q}']:,}（{r[f'high{q}_ep']}）" for q in qs) + " |" for r in counts]
    md += ["", "## 信号どうしの順位相関（Binance の決済の時刻に、その時刻に分かっている最新の値をそろえる）\n",
           "| | " + " | ".join(sig_corr.columns) + " |", "|---|" + "---|" * len(sig_corr.columns)]
    md += [f"| {i} | " + " | ".join("-" if not np.isfinite(v) else f"{v:.2f}" for v in row) + " |" for i, row in sig_corr.iterrows()]
    md += ["", f"## 事象の直前 24 時間の値動き（bitbank、直前 30 日の 1 時間の標準偏差 × √24 で割った値。q = {q_main:.0%}）\n",
           "| 信号 | 側 | 件数 | 2σ を超える下げの後 | 2σ を超える上げの後 |", "|---|---|---|---|---|"]
    md += [f"| {r['signal']} | {r['side']} | {r['n']:,} | {pct(r['after_drop'], 1)} | {pct(r['after_rise'], 1)} |" for r in prior]
    md += ["", "## 照合\n",
           "- 資金調達率の日ごとの平均の相関: " +
           ("、".join(f"{a}–{b} {corr[a][b]:.3f}" for a, b in (("binance", "bitmex"), ("binance", "deribit"), ("bitmex", "deribit")))
            if corr else "-"),
           "- Deribit の interest_8h と interest_1h: " +
           (f"8 × interest_1h との相関 {der['corr_8x1h']:.3f}（差の中央値 {der['median_abs_diff_8x1h']:.2e}）、"
            f"直前 8 時間の interest_1h の和との相関 {der['corr_sum8_1h']:.3f}（差の中央値 {der['median_abs_diff_sum8']:.2e}）、"
            f"比の中央値 {der['median_ratio_8h_to_1h']:.2f}" if der else "-"),
           f"- ドル円: FRED（ニューヨーク正午）と、その時刻に閉じる Dukascopy の足の終値の差の中央値 {pct(fx_row['median_abs_diff'], 3)}、"
           f"99% 点 {pct(fx_row['p99_abs_diff'], 3)}（{fx_row['n']:,} 日）",
           "- JPY の内外価格差（log、年ごとの 5%・50%・95% 点）: " + "、".join(f"{y}: {', '.join(pct(v, 3) for v in q)}" for y, q in prem_q.items()),
           "- Binance の資金調達率（年ごとの 5%・10%・50%・90%・95% 点）: " +
           "、".join(f"{y}: {', '.join(pct(v, 4) for v in q)}" for y, q in fund_q.items()),
           "", "## 書式の確認（各系列の最初の 2 行）\n"]
    md += [f"- {k}: {rows}" for k, rows in samples.items()]
    (OUT / "check.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return chk


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["check"], required=True)
    ap.add_argument("--config", default="configs/ext_study.yaml")
    args = ap.parse_args(argv)
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    main_check(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
