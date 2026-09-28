"""候補 2（24 時間の新しい情報源）の研究。2026-09-28 オーナー決定。

    python scripts/ext_study.py --mode check    # データの確認（期間・抜け・書式、事象の数）。先の収益率は一切出さない
    python scripts/ext_study.py --mode eval     # 事前登録（configs/ext_study.yaml の eval）どおりに 1 回評価。owner_approved が必要
    python scripts/ext_study.py --mode forward  # 支持した仮説（H2a）の前向きのドライラン。確定した月まで同じ計算をする（発注しない）

データは bbresearch/extdata.py（Binance のアーカイブ、BitMEX、Deribit、Dukascopy、FRED）と bitbank の公式 1 時間足
（scripts/dist_forecast.py と同じキャッシュ）、HAR 型に使う bitbank の約定（data/raw）。事前登録の数値は configs/ext_study.yaml の eval に
固定してあり、オーナーの承認後に 1 回だけ評価する。評価の手順は bbresearch/eventstudy.py の冒頭。
信号はすべて、値が分かる時刻（資金調達率は決済の時刻、1 時間足の値は足が閉じる時刻）を index にする。
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

from bbresearch import eventstudy as es
from bbresearch import extdata as xd
from bbresearch.distbase import hourly_rv, minute_close_from_raw
from bbresearch.fcompare import holm

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

    mins = None
    if (Path(cfg.get("trades_root", "data")) / "raw" / "transactions" / cfg["pair"]).exists():
        mins = minutes_summary(load_minutes(cfg), d["bitbank"])
    chk = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "requests": f.n_requests,
           "seconds": round(time.monotonic() - t0), "config": {k: cfg.get(k) for k in ("start", "end", "window", "min_days", "check_thresholds",
                                                                                      "metrics_start", "binance_symbol", "pair")},
           "coverage": cov, "funding_binance_gaps": funding_gaps, "first_signal": first_signal, "samples": samples,
           "event_counts": counts, "signal_corr": sig_corr.to_dict(), "funding_corr_daily": corr, "deribit_interest": der,
           "fx_dukascopy_vs_fred": fx_row, "premium_quantiles_by_year": prem_q, "funding_quantiles_by_year": fund_q,
           "prior_move": prior, "failures": f.failures, "minutes": mins}
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
           ""]
    if mins:
        md += ["## bitbank の約定から作る 1 分の終値（HAR 型の実現分散に使う）\n",
               "| 年 | 分 | 約定のある分 | 約定のない日 | 約定のない最長の連続（分） | 1 時間の終値が公式足と一致 |", "|---|---|---|---|---|---|"]
        md += [f"| {r['year']} | {r['minutes']:,} | {pct(r['share_with_trades'], 1)} | {r['days_without_trades']} | {r['longest_gap_min']:,} | "
               f"{pct(mins['hourly_close_match'].get(r['year']), 2)} |" for r in mins["by_year"]]
        md.append("")
    else:
        md += ["- bitbank の約定（HAR 型に使う）は、このマシンに保存されていないので確認していない\n"]
    md += ["## 書式の確認（各系列の最初の 2 行）\n"]
    md += [f"- {k}: {rows}" for k, rows in samples.items()]
    (OUT / "check.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return chk


# ------------------------------------------------------------------ 約定から作る 1 分の終値（HAR 型に使う）の確認

def load_minutes(cfg: dict) -> pd.Series:
    mc, _ = minute_close_from_raw(cfg["trades_root"], cfg["pair"], cfg["start"], cfg["end"])
    return mc


def minutes_summary(minute_close: pd.Series, hourly: pd.DataFrame) -> dict:
    """年ごとの約定のある分の割合、約定のない日、約定のない最長の連続（分）と、約定から作った 1 時間の終値と公式 1 時間足の一致。"""
    has = minute_close.notna()
    rows = []
    for yr, g in has.groupby(has.index.year):
        runs = (~g).astype(int).groupby(g.cumsum()).sum()
        day = g.groupby(g.index.floor("D")).any()
        rows.append({"year": int(yr), "minutes": int(len(g)), "share_with_trades": float(g.mean()),
                     "days_without_trades": int((~day).sum()), "longest_gap_min": int(runs.max()) if len(runs) else 0})
    ff = minute_close.ffill()
    hc = ff.groupby(ff.index.floor("h")).last()
    common = hourly.index.intersection(hc.index)
    eq = (hc.reindex(common) - hourly["close"].reindex(common)).abs() < 0.5
    match = {int(y): float(g.mean()) for y, g in eq.groupby(eq.index.year)}
    return {"by_year": rows, "hourly_close_match": match}


# ------------------------------------------------------------------ 評価（1 回だけ）

CODE_FILES = ("bbresearch/eventstudy.py", "bbresearch/extdata.py", "bbresearch/distbase.py", "bbresearch/fcompare.py",
              "scripts/ext_study.py")


def cfg_hash(cfg: dict) -> str:
    c = {k: v for k, v in cfg.items() if k != "owner_approved"}
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]


def code_hash() -> str:
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
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (pd.Timestamp, datetime)):
        return str(x)
    return x


def evaluate(d: dict, minute_close: pd.Series, cfg: dict) -> dict:
    """事前登録どおりの計算。主な検定 6 つ、他の取引所の向き、副指標（判定には使わない）。"""
    E = cfg["eval"]
    H = f"{E['horizon_hours']}h"
    times = es.grid_times(cfg["start"], cfg["end"], E["grid_hours"], H)
    t_ns = times.as_unit("ns").asi8
    gap_ns = int(pd.Timedelta(hours=E["gap_hours"]).value)
    n_bins = len(E["zbins"]) + 1
    min_shift = int(E["min_shift_days"] * len(E["grid_hours"]))
    close_known = xd.available(d["bitbank"]["close"])
    y = es.forward_log_return(close_known, times, H)
    zb = es.zbins(es.prior_z(close_known, times, H, E["z_vol_hours"], E["z_min_hours"]), E["zbins"])
    hw = E["har"]
    rv = hourly_rv(minute_close, d["bitbank"].index)
    har = es.har_walkforward(rv, np.log(close_known), times, tuple(hw["windows"]), hw["calendar"], hw["rv_floor"],
                             E["horizon_hours"], hw["min_train_days"], tuple(hw["quantiles"]))
    tails = {q: np.where(np.isfinite(har["q"][q]) & np.isfinite(y), (y < har["q"][q]).astype(float), np.nan)
             for q in hw["quantiles"]}
    y_std = y / har["sigma"]
    sig = signals(d, cfg)
    ranks = {k: es.value_at(v, times, f"{E['staleness_hours']['funding' if k.startswith('funding_') else 'hourly']}h")
             for k, v in sig.items()}

    def run(signal: str, side: str, yv: np.ndarray, direction: int) -> dict:
        r = ranks[signal]
        obs = es.event_study(r, yv, zb, t_ns, E["q"], side, gap_ns, n_bins)
        ep = obs.pop("episodes") & np.isfinite(yv)
        null = es.shift_null(r, yv, zb, t_ns, E["q"], side, gap_ns, n_bins, E["null_reps"], min_shift, E["seed"])
        by_year = {int(yr): {"n": int(len(g)), "mean": float(np.mean(g))}
                   for yr, g in pd.Series(yv[ep], index=times[ep]).groupby(times[ep].year)}
        fin = null[np.isfinite(null)]
        return obs | {"signal": signal, "side": side, "direction": direction, "p": es.p_one_sided(obs["effect"], null, direction),
                      "null_q": np.quantile(fin, [0.025, 0.5, 0.975]).tolist() if len(fin) else [],
                      "n_null": int(len(fin)), "by_year": by_year, "first": str(times[ep][0]) if ep.any() else "-",
                      "last": str(times[ep][-1]) if ep.any() else "-"}

    tq = E["tail_quantile"]
    primary = []
    for P in E["primary"]:
        yv = y if P["kind"] == "mean" else tails[tq]
        primary.append(run(P["signal"], P["side"], yv, P["direction"]) | {"id": P["id"], "kind": P["kind"]})
    raw_p = [r["p"] if np.isfinite(r["p"]) else 1.0 for r in primary]      # 事象がなく p が出ないものは 1 として補正の数に入れる
    for r, ph in zip(primary, holm(raw_p)):
        r["p_holm"] = ph if np.isfinite(r["p"]) else float("nan")
    venues = {(v, s_): run(v, s_, y, 1 if s_ == "low" else -1) for v in E["venue_check"] for s_ in ("low", "high")}
    for r in primary:
        r["significant"] = bool(np.isfinite(r["p_holm"]) and r["p_holm"] < E["alpha"])
        if r["kind"] == "mean":
            r["economic"] = bool(r["ev_mean"] > E["cost"]) if r["direction"] > 0 else bool(r["ev_mean"] < -E["cost"])
            r["ratio"] = None
        else:
            r["ratio"] = (r["ev_mean"] / r["ctrl_mean"]) if r["ctrl_mean"] > 0 else (math.inf if r["ev_mean"] > 0 else math.nan)
            r["economic"] = bool(r["ratio"] >= E["tail_ratio"]) if r["ratio"] is not None and not math.isnan(r["ratio"]) else False
        if r["signal"] == "funding_binance":
            vs = [venues[(v, r["side"])] for v in E["venue_check"]]
            r["venues"] = {v["signal"]: v["effect"] for v in vs}
            r["venue_ok"] = bool(all(np.isfinite(v["effect"]) and np.sign(v["effect"]) == r["direction"] for v in vs))
        else:
            r["venues"], r["venue_ok"] = {}, True
        r["supported"] = r["significant"] and r["economic"] and r["venue_ok"]
    secondary = {"venues": [v for v in venues.values()],
                 "standardized": [run(P["signal"], P["side"], y_std, P["direction"]) | {"id": P["id"]}
                                  for P in E["primary"] if P["kind"] == "mean"],
                 "tail_1pct": [run(P["signal"], P["side"], tails[min(hw["quantiles"])], P["direction"]) | {"id": P["id"]}
                               for P in E["primary"] if P["kind"] == "tail"]}
    okq = np.isfinite(tails[tq])
    diag = {"n_grid": int(len(times)), "first_grid": str(times[0]), "last_grid": str(times[-1]),
            "y_mean": float(np.nanmean(y)), "y_sd": float(np.nanstd(y)), "har_refits": len(har["fits"]),
            "har_r2_median": float(np.median([f["r2"] for f in har["fits"]])) if har["fits"] else None,
            "har_first_fit": har["fits"][0] if har["fits"] else None, "har_last_fit": har["fits"][-1] if har["fits"] else None,
            "tail_rate_all": float(np.mean(tails[tq][okq])) if okq.any() else None, "n_tail_defined": int(okq.sum()),
            "rank_defined": {k: int(np.isfinite(v).sum()) for k, v in ranks.items()}}
    return {"primary": primary, "secondary": secondary, "diag": diag}


def main_eval(cfg: dict, fetcher: xd.Fetcher | None = None, hourly=None, minute_close: pd.Series | None = None) -> dict:
    if not cfg.get("owner_approved"):
        sys.exit("configs/ext_study.yaml の owner_approved が空。オーナーの承認の前は評価しない。")
    f = fetcher or xd.Fetcher(cfg.get("cache", ".cache/ext"), strict=True)      # 評価では、取れないファイルがあれば止める
    d = load_all(cfg, f, hourly)
    mc = load_minutes(cfg) if minute_close is None else minute_close
    res = evaluate(d, mc, cfg)
    th, run_at = cfg_hash(cfg), datetime.now(timezone.utc).isoformat()
    log = Path("reports/experiments.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as fh:
        for r in res["primary"]:
            fh.write(json.dumps(jsonable({"run_at": run_at, "stage": "ext_study", "trial_hash": th,
                                          "code_version": os.environ.get("GITHUB_SHA"), "key": f"ext_study/{r['id']}",
                                          "signal": r["signal"], "side": r["side"], "effect": r["effect"], "ev_mean": r["ev_mean"],
                                          "n_ev": r["n_ev"], "p": r["p"], "p_holm": r["p_holm"], "supported": r["supported"]})) + "\n")
    metrics = {"run_at": run_at, "trial_hash": th, "code_hash": code_hash(), "owner_approved": cfg["owner_approved"],
               "requests": f.n_requests} | res
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(jsonable(metrics), ensure_ascii=False, indent=1))
    write_report(cfg, metrics)
    return metrics


LABEL = {"funding_binance": "資金調達率（Binance）", "funding_bitmex": "資金調達率（BitMEX）", "funding_deribit": "資金調達率（Deribit）",
         "jpy_premium": "JPY の内外価格差", "carry": "キャリー（プレミアム指数）", "oi_change": "建玉の 24 時間の変化"}
SIDE = {"low": "下位", "high": "上位"}


def pv(x) -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else (f"{x:.3f}" if x >= 0.001 else "<0.001")


def sg(x, nd=3) -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.{nd}f}"


def write_report(cfg: dict, m: dict) -> None:
    E, D = cfg["eval"], m["diag"]
    sup = [r["id"] for r in m["primary"] if r["supported"]]
    md = [f"# 候補 2（24 時間の新しい情報源）: 事前登録どおりの評価（{cfg['pair']}、24 時間先）\n",
          f"結論: 主な検定 6 つのうち支持は {len(sup)} 個" + (f"（{'、'.join(sup)}。{E['forward_months']} か月の前向きのドライランで確かめる）" if sup else
                                                       "（前向きの確認はしない）") + "。\n",
          f"- 設定のハッシュ {m['trial_hash']}（configs/ext_study.yaml、オーナー承認 {m['owner_approved']}）、"
          f"コードのハッシュ {m['code_hash']}",
          f"- 判断の時刻 {len(E['grid_hours'])} 回/日（{'・'.join(str(h) for h in E['grid_hours'])} 時 UTC）、{D['first_grid'][:16]} 〜 {D['last_grid'][:16]}"
          f"（{D['n_grid']:,} 回）。24 時間の対数収益率の平均 {pct(D['y_mean'], 3)}、標準偏差 {pct(D['y_sd'], 2)}",
          f"- 事象: 直前 {cfg['window']} の中の順位が下位・上位 {E['q']:.0%}。数えた事象から {E['gap_hours']} 時間の間の事象は数えない。"
          f"統制: 前後 {E['gap_hours']} 時間に同じ側の事象がない時刻を、直前 24 時間の値動き z の区分（{E['zbins']}）でそろえる",
          f"- p: 信号の列を {E['min_shift_days']} 日以上ずらす循環シフト {E['null_reps']} 回（片側）。主な検定 6 つを Holm 法で補正し {E['alpha']:.0%} 未満",
          f"- 支持の条件: 上げ下げ（H1・H2）は有意 かつ 事象の後の平均が {pct(E['cost'], 1)} を超える（上位側は −{pct(E['cost'], 1)} を下回る）。"
          f"H1 は BitMEX・Deribit でも統制との差が同じ向き。下側の裾（H3）は有意 かつ HAR 型の {E['tail_quantile']:.0%} 点を下回る割合が統制の "
          f"{E['tail_ratio']:g} 倍以上\n",
          "## 判定（事前登録）\n",
          "| 仮説 | 信号 | 側 | 事象（数えた回数 / 全体） | 事象の後 | 統制（そろえた） | 差 | p | p（Holm） | 費用・倍率 | 他の取引所 | 判定 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in m["primary"]:
        if r["kind"] == "mean":
            a, b, c = pct(r["ev_mean"], 2), pct(r["ctrl_mean"], 2), pct(r["effect"], 2)
            econ = ("満たす" if r["economic"] else "満たさない")
        else:
            a, b, c = pct(r["ev_mean"], 1), pct(r["ctrl_mean"], 1), pct(r["effect"], 1)
            econ = f"{r['ratio']:.2f} 倍（{'満たす' if r['economic'] else '満たさない'}）" if r["ratio"] is not None and math.isfinite(r["ratio"]) else "-"
        ven = "、".join(f"{k.split('_')[1]} {pct(v, 2)}" for k, v in r["venues"].items()) + ("（同じ向き）" if r["venue_ok"] else "（向きが違う）") \
            if r["venues"] else "-"
        md.append(f"| {r['id']} | {LABEL[r['signal']]} | {SIDE[r['side']]} {E['q']:.0%} | {r['n_ev']:,} / {r['n_events']:,} | {a} | {b} | {c} | {pv(r['p'])} | "
                  f"{pv(r['p_holm'])} | {econ} | {ven} | {'支持' if r['supported'] else '支持しない'} |")
    md += ["", "H1・H2 の「事象の後」「統制」は 24 時間の対数収益率の平均、H3 は HAR 型の 5% 点を下回った割合。\n",
           "## 副指標（判定には使わない）\n", "### 他の取引所の資金調達率（統制との差、片側 p は Holm 補正なし）\n",
           "| 信号 | 側 | 事象（数えた回数） | 事象の後 | 統制 | 差 | p |", "|---|---|---|---|---|---|---|"]
    md += [f"| {LABEL[v['signal']]} | {SIDE[v['side']]} {E['q']:.0%} | {v['n_ev']:,} | {pct(v['ev_mean'], 2)} | {pct(v['ctrl_mean'], 2)} | {pct(v['effect'], 2)} | {pv(v['p'])} |"
           for v in m["secondary"]["venues"]]
    md += ["", "### HAR 型の σ̂ で割った 24 時間の収益率（H1・H2）\n", "| 仮説 | 事象の後 | 統制 | 差 | p |", "|---|---|---|---|---|"]
    md += [f"| {v['id']} | {sg(v['ev_mean'])} | {sg(v['ctrl_mean'])} | {sg(v['effect'])} | {pv(v['p'])} |" for v in m["secondary"]["standardized"]]
    md += ["", "### HAR 型の 1% 点を下回った割合（H3）\n", "| 仮説 | 事象の後 | 統制 | 差 | p |", "|---|---|---|---|---|"]
    md += [f"| {v['id']} | {pct(v['ev_mean'], 2)} | {pct(v['ctrl_mean'], 2)} | {pct(v['effect'], 2)} | {pv(v['p'])} |" for v in m["secondary"]["tail_1pct"]]
    md += ["", "### 年ごとの事象の後の平均（数えた事象）\n", "| 仮説 | " + " | ".join(str(y) for y in range(2020, 2027)) + " |",
           "|---|" + "---|" * 7]
    for r in m["primary"]:
        by = r["by_year"]
        fmt = (lambda v: pct(v, 2)) if r["kind"] == "mean" else (lambda v: pct(v, 1))
        md.append(f"| {r['id']} | " + " | ".join(f"{fmt(by[y]['mean'])}（{by[y]['n']}）" if y in by else "-" for y in range(2020, 2027)) + " |")
    md += ["", "## HAR 型（下側の裾の物差し）\n",
           f"- 当てはめ直し {D['har_refits']} 回（毎月初め、その時点までに結果が出そろった行だけ）。R² の中央値 "
           f"{D['har_r2_median']:.3f}" if D["har_r2_median"] is not None else "- 当てはめなし",
           f"- 全体で 5% 点を下回った割合 {pct(D['tail_rate_all'], 2)}（{D['n_tail_defined']:,} 回。5% に近いほど較正が良い）",
           "- 判断の時刻に信号がそろっていた回数: " + "、".join(f"{k} {v:,}" for k, v in D["rank_defined"].items())]
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# ------------------------------------------------------------------ 前向きのドライラン（支持した仮説だけ、発注しない）

def first_taker_price(trades: pd.DataFrame, times: pd.DatetimeIndex, side: str, window_min: float) -> np.ndarray:
    """各時刻 T から window_min 分以内の、最初の side（テイカー側。buy は売り気配で買った約定）の約定の価格。なければ NaN。"""
    t = trades[trades["side"] == side]
    ms_ = t["executed_at"].to_numpy("int64")
    px = t["price"].to_numpy(float)
    q = times.as_unit("ms").asi8
    out = np.full(len(q), np.nan)
    if len(ms_) == 0:
        return out
    pos = np.searchsorted(ms_, q, side="left")
    ok = pos < len(ms_)
    ok[ok] = ms_[pos[ok]] - q[ok] <= window_min * 60_000
    out[ok] = px[pos[ok]]
    return out


def read_quotes(path: str | Path, max_delay_min: float) -> pd.DataFrame:
    """記録した気配（1 行 1 回）。同じ判断の時刻が複数あれば最初の記録。遅れが上限を超えたものは除く。"""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["sell", "buy", "delay_min"], index=pd.DatetimeIndex([], tz="UTC"))
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    q = pd.DataFrame(rows)
    q.index = pd.DatetimeIndex(pd.to_datetime(q["slot"], utc=True))
    q = q[~q.index.duplicated()].sort_index()
    return q[q["delay_min"] <= max_delay_min][["sell", "buy", "delay_min"]]


def forward_window(cfg: dict, today: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """前向きの期間の始まり、今回の終わり（確定した月の終わり、含まない）、予定の終わり。"""
    F = cfg["forward"]
    start = pd.Timestamp(F["start"], tz="UTC")
    planned = start + pd.DateOffset(months=F["months"])
    return start, min(today.normalize().replace(day=1), planned), planned


def main_forward(cfg: dict, fetcher: xd.Fetcher | None = None, hourly=None, trades: pd.DataFrame | None = None,
                 today: pd.Timestamp | None = None) -> dict:
    if not cfg.get("owner_approved"):
        sys.exit("configs/ext_study.yaml の owner_approved が空。事前登録の承認の前は前向きの確認もしない。")
    E, F = cfg["eval"], cfg["forward"]
    today = today if today is not None else pd.Timestamp.now(tz="UTC")
    f_start, f_end, planned = forward_window(cfg, today)
    out = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "start": str(f_start), "end": str(f_end),
           "planned_end": str(planned), "complete": bool(f_end >= planned), "hypotheses": []}
    if f_end <= f_start:
        out["note"] = "確定した月がまだない"
    else:
        c2 = cfg | {"end": str(f_end.date())}
        f = fetcher or xd.Fetcher(cfg.get("cache", ".cache/ext"), strict=True)
        d = load_all(c2, f, hourly)
        H = f"{E['horizon_hours']}h"
        times = es.grid_times(f_start, f_end, E["grid_hours"], H)
        t_ns = times.as_unit("ns").asi8
        gap_ns = int(pd.Timedelta(hours=E["gap_hours"]).value)
        n_bins = len(E["zbins"]) + 1
        close_known = xd.available(d["bitbank"]["close"])
        y = es.forward_log_return(close_known, times, H)
        zb = es.zbins(es.prior_z(close_known, times, H, E["z_vol_hours"], E["z_min_hours"]), E["zbins"])
        sig = signals(d, c2)
        if trades is None:
            from bbdata.download import load_transactions
            trades = load_transactions(cfg["trades_root"], cfg["pair"], f_start, f_end + pd.Timedelta(days=1))
        quotes = read_quotes(F["quotes"], F["max_quote_delay_min"])
        for hid in F["hypotheses"]:
            P = next(p for p in E["primary"] if p["id"] == hid)
            tol = f"{E['staleness_hours']['funding' if P['signal'].startswith('funding_') else 'hourly']}h"
            r = es.value_at(sig[P["signal"]], times, tol)
            obs = es.event_study(r, y, zb, t_ns, E["q"], P["side"], gap_ns, n_bins)
            ep = obs.pop("episodes") & np.isfinite(y)
            min_shift = min(int(E["min_shift_days"] * len(E["grid_hours"])), max(1, len(times) // 4))
            null = es.shift_null(r, y, zb, t_ns, E["q"], P["side"], gap_ns, n_bins, E["null_reps"], min_shift, E["seed"])
            te = times[ep]
            buy = first_taker_price(trades, te, "buy", F["entry_window_min"])
            sell = first_taker_price(trades, te + pd.Timedelta(H), "sell", F["entry_window_min"])
            r_exec = np.log(sell / buy)
            ask = quotes["sell"].reindex(te).to_numpy(float)
            bid = quotes["buy"].reindex(te + pd.Timedelta(H)).to_numpy(float)
            r_quote = np.log(bid / ask)
            ok_e, ok_q = np.isfinite(r_exec), np.isfinite(r_quote)
            passed = bool(np.isfinite(obs["effect"]) and obs["effect"] > 0 and obs["ev_mean"] > E["cost"])
            out["hypotheses"].append(obs | {
                "id": hid, "signal": P["signal"], "side": P["side"], "p_info": es.p_one_sided(obs["effect"], null, P["direction"]),
                "criterion_met": passed,
                "exec": {"n": int(ok_e.sum()), "mean_close": float(np.mean(y[ep][ok_e])) if ok_e.any() else None,
                         "mean_exec": float(np.mean(r_exec[ok_e])) if ok_e.any() else None,
                         "mean_exec_net": float(np.mean(r_exec[ok_e]) - F["fee_round_trip"]) if ok_e.any() else None},
                "quotes": {"n": int(ok_q.sum()), "mean_close": float(np.mean(y[ep][ok_q])) if ok_q.any() else None,
                           "mean_quote": float(np.mean(r_quote[ok_q])) if ok_q.any() else None,
                           "mean_quote_net": float(np.mean(r_quote[ok_q]) - F["fee_round_trip"]) if ok_q.any() else None,
                           "n_recorded": int(len(quotes[(quotes.index >= f_start) & (quotes.index < f_end)]))},
                "events": [{"t": str(t), "y": float(v), "exec": None if not np.isfinite(a) else float(a),
                            "quote": None if not np.isfinite(b) else float(b)} for t, v, a, b in zip(te, y[ep], r_exec, r_quote)]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "forward.json").write_text(json.dumps(jsonable(out), ensure_ascii=False, indent=1))
    write_forward_report(cfg, out)
    return out


def write_forward_report(cfg: dict, o: dict) -> None:
    E, F = cfg["eval"], cfg["forward"]
    md = ["# 候補 2: 支持した仮説の前向きのドライラン（発注しない）\n",
          f"- 期間 {o['start'][:10]} 〜 {o['end'][:10]}（終わりは含まない。予定は {o['planned_end'][:10]} まで、"
          f"{'そろった。判定する' if o['complete'] else '途中経過（判定は予定の期間がそろってから）'}）。更新 {o['run_at']}",
          f"- 計算は事前登録の評価と同じ（configs/ext_study.yaml の eval）。判定の基準: 統制との差が正 かつ 事象の後の平均が {pct(E['cost'], 1)} を超える"
          "（回数が少ないので有意は求めない。p は参考）",
          f"- 実際の価格: 約定から測る値は、判断の時刻から {F['entry_window_min']} 分以内の最初の買い（テイカー）の約定で買い、24 時間後から "
          f"{F['entry_window_min']} 分以内の最初の売りの約定で売ったときの対数収益率。気配から測る値は、0・8・16 時 UTC に記録した売り気配で買い、"
          f"24 時間後の買い気配で売ったとき（記録の遅れが {F['max_quote_delay_min']} 分以内のものだけ）。手数料 {pct(F['fee_round_trip'], 1)}（往復）を引いた値も出す\n"]
    if not o["hypotheses"]:
        md.append(f"{o.get('note', '')}。\n")
    for h in o["hypotheses"]:
        verdict = ("基準を満たす" if h["criterion_met"] else "基準を満たさない") + ("" if o["complete"] else "（途中経過）")
        ex, qu = h["exec"], h["quotes"]
        md += [f"## {h['id']}（{LABEL[h['signal']]}、{SIDE[h['side']]} {E['q']:.0%}）: {verdict}\n",
               "| 事象（数えた回数 / 全体） | 事象の後（終値） | 統制（そろえた） | 差 | p（参考） |", "|---|---|---|---|---|",
               f"| {h['n_ev']} / {h['n_events']} | {pct(h['ev_mean'], 2)} | {pct(h['ctrl_mean'], 2)} | {pct(h['effect'], 2)} | {pv(h['p_info'])} |", "",
               "| 実際の価格の測り方 | 回数 | 同じ回の終値での平均 | 実際の価格での平均 | 手数料を引いた後 |", "|---|---|---|---|---|",
               f"| 約定（{F['entry_window_min']} 分以内の最初の約定） | {ex['n']} | {pct(ex['mean_close'], 2)} | {pct(ex['mean_exec'], 2)} | {pct(ex['mean_exec_net'], 2)} |",
               f"| 記録した気配（記録 {qu['n_recorded']} 回） | {qu['n']} | {pct(qu['mean_close'], 2)} | {pct(qu['mean_quote'], 2)} | {pct(qu['mean_quote_net'], 2)} |", "",
               "| 判断の時刻 | 終値で測った 24 時間 | 約定で測った 24 時間 | 気配で測った 24 時間 |", "|---|---|---|---|"]
        md += [f"| {e['t'][:16]} | {pct(e['y'], 2)} | {pct(e['exec'], 2)} | {pct(e['quote'], 2)} |" for e in h["events"]]
        md.append("")
    (OUT / "forward.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["check", "eval", "forward"], required=True)
    ap.add_argument("--config", default="configs/ext_study.yaml")
    args = ap.parse_args(argv)
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if args.mode == "check":
        main_check(cfg)
    elif args.mode == "eval":
        main_eval(cfg)
    else:
        main_forward(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
