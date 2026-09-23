"""Phase 0: 実データ検証（SPEC.md §5 の V1〜V5 と、ペア仕様の記録）。

公開 API だけを使う（認証不要）。リクエスト数は全体で数百件に抑えている。
結果は Markdown と JSON で --out に書き出し、標準出力にも Markdown を出す。

    PYTHONPATH=. python scripts/phase0.py --out reports/phase0
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bbdata.bars import build_bars
from bbdata.client import BitbankAPIError, PublicClient
from bbdata.download import download_candles, download_transactions, load_candles, trades_path
from bbdata.validate import compare_with_official

log = logging.getLogger("phase0")

PRIVATE_BASE_URL = "https://api.bitbank.cc/v1"


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


# ------------------------------------------------------------------ ペア仕様

def check_spot(pair: str) -> dict:
    """/spot/pairs と /spot/status（どちらも認証不要）から対象ペアの仕様を記録する。"""
    api = PublicClient(base_url=PRIVATE_BASE_URL, min_interval=0.5)
    pairs = api.get("/spot/pairs")["pairs"]
    spec = next((p for p in pairs if p.get("name") == pair), None)
    status = api.get("/spot/status")
    st = next((s for s in status.get("statuses", []) if s.get("pair") == pair), None)
    return {"pair_spec": spec, "spot_status": st}


# ------------------------------------------------------------------------ V1

def v1_candles(client: PublicClient, pair: str, root: Path, days: int) -> dict:
    end = _utc_today() - timedelta(days=1)  # 前日までの確定した日だけを使う
    start = end - timedelta(days=days)
    download_transactions(client, pair, start, end, root)
    download_candles(client, pair, "15min", start, end, root)
    bars = build_bars(root, pair, start, end, "15min")
    candles = load_candles(root, pair, "15min", start, end)
    table = compare_with_official(bars, candles, max_shift=1)
    best = table["ohlc_match_rate"].idxmax() if "ohlc_match_rate" in table else None
    return {
        "start": str(start),
        "end": str(end),
        "n_bars": len(bars),
        "n_empty_bars": int(bars["is_empty"].sum()),
        "n_official": len(candles),
        "table": table.reset_index().to_dict(orient="records"),
        "best_shift_bars": None if best is None else int(best),
    }


# ------------------------------------------------------------------------ V5

def v5_day_boundary(pair: str, root: Path, start: date, end: date) -> dict:
    """各日ファイルの executed_at の最小・最大から、日付境界のタイムゾーンを推定する。"""
    rows = []
    d = start
    while d < end:
        p = trades_path(root, pair, d)
        if p.exists():
            df = pd.read_parquet(p)
            if not df.empty:
                day0 = pd.Timestamp(d, tz="UTC")
                lo = pd.to_datetime(df["executed_at"].min(), unit="ms", utc=True)
                hi = pd.to_datetime(df["executed_at"].max(), unit="ms", utc=True)
                rows.append(
                    {
                        "date": str(d),
                        "n": len(df),
                        "first_utc": lo.isoformat(),
                        "last_utc": hi.isoformat(),
                        "first_minus_day0_h": round((lo - day0) / pd.Timedelta(hours=1), 3),
                        "day0_plus_24h_minus_last_h": round((day0 + pd.Timedelta(days=1) - hi) / pd.Timedelta(hours=1), 3),
                    }
                )
        d += timedelta(days=1)
    offs = [r["first_minus_day0_h"] for r in rows]
    guess = None
    if offs:
        med = float(np.median(offs))
        # 最初の約定は境界の直後に来るはず。境界が UTC なら 0 付近、JST なら -9 付近
        guess = "UTC" if -0.5 < med < 1.0 else ("JST(UTC+9)" if -9.5 < med < -8.0 else f"不明(median={med}h)")
    return {"days": rows, "guess": guess}


# ------------------------------------------------------------------------ V2

def v2_side(client: PublicClient, pair: str, minutes: float, interval: float) -> dict:
    """板と最新約定を交互に取得し、約定の side と直前の気配の関係を調べる。"""
    snaps: list[tuple[int, float, float]] = []  # (timestamp_ms, best_bid, best_ask)
    trades: dict[int, dict] = {}
    depth_keys = None
    deadline = time.monotonic() + minutes * 60
    while time.monotonic() < deadline:
        try:
            dep = client.depth(pair)
            depth_keys = depth_keys or sorted(dep)
            ts = int(dep.get("timestamp") or 0)
            if dep.get("bids") and dep.get("asks") and ts:
                snaps.append((ts, float(dep["bids"][0][0]), float(dep["asks"][0][0])))
            for t in client.transactions(pair):
                trades[int(t["transaction_id"])] = t
        except (BitbankAPIError, RuntimeError) as exc:
            log.warning("V2 取得失敗: %s", exc)
        time.sleep(interval)

    snaps.sort()
    snap_ts = np.array([s[0] for s in snaps], dtype="int64")
    out_rows = []
    for t in trades.values():
        ms = int(t["executed_at"])
        i = int(np.searchsorted(snap_ts, ms, side="right")) - 1
        if i < 0:
            continue
        s_ts, bid, ask = snaps[i]
        out_rows.append(
            {"side": t["side"], "price": float(t["price"]), "age_ms": ms - s_ts, "bid": bid, "ask": ask}
        )
    df = pd.DataFrame(out_rows)
    res: dict = {"n_snapshots": len(snaps), "n_trades": len(trades), "depth_keys": depth_keys, "by_age": []}
    if df.empty:
        return res
    mid = (df["bid"] + df["ask"]) / 2
    for max_age in (500, 1000, 2000, 5000):
        m = df["age_ms"] <= max_age
        b = m & df["side"].eq("buy")
        s = m & df["side"].eq("sell")
        res["by_age"].append(
            {
                "max_age_ms": max_age,
                "n_buy": int(b.sum()),
                "buy_ge_prev_ask": float((df.loc[b, "price"] >= df.loc[b, "ask"]).mean()) if b.any() else None,
                "buy_gt_prev_mid": float((df.loc[b, "price"] > mid[b]).mean()) if b.any() else None,
                "n_sell": int(s.sum()),
                "sell_le_prev_bid": float((df.loc[s, "price"] <= df.loc[s, "bid"]).mean()) if s.any() else None,
                "sell_lt_prev_mid": float((df.loc[s, "price"] < mid[s]).mean()) if s.any() else None,
            }
        )
    return res


# ------------------------------------------------------------------------ V3

def _has_trades(client: PublicClient, pair: str, d: date) -> bool:
    try:
        return len(client.transactions(pair, f"{d:%Y%m%d}")) > 0
    except BitbankAPIError as exc:
        log.info("V3 %s: %s", d, exc)
        return False


def v3_oldest(client: PublicClient, pair: str, lo: date) -> dict:
    """取得できる最古の日付を二分探索する（取得可能な日付が連続している前提）。"""
    probes = []
    hi = _utc_today() - timedelta(days=3)
    if not _has_trades(client, pair, hi):
        return {"oldest": None, "probes": [(str(hi), False)]}
    if _has_trades(client, pair, lo):
        return {"oldest": str(lo), "probes": [(str(lo), True)], "note": "探索下限でも取得できた"}
    # 不変条件: lo は取得不可、hi は取得可
    while (hi - lo).days > 1:
        mid = lo + timedelta(days=(hi - lo).days // 2)
        ok = _has_trades(client, pair, mid)
        probes.append((str(mid), ok))
        if ok:
            hi = mid
        else:
            lo = mid
    # 連続性の前提を少しだけ確かめる: 最古日の前後数日をサンプル
    around = [(str(hi + timedelta(days=k)), _has_trades(client, pair, hi + timedelta(days=k))) for k in (1, 7, 30)]
    return {"oldest": str(hi), "probes": probes, "after_oldest_check": around}


# ------------------------------------------------------------------ report

def _fmt_table(rows: list[dict]) -> str:
    if not rows:
        return "(なし)\n"
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join("" if r.get(c) is None else f"{r.get(c):.6g}" if isinstance(r.get(c), float) else str(r.get(c)) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def render(res: dict) -> str:
    md = [f"# Phase 0 検証結果（{res['run_at']}）\n", f"pair: `{res['pair']}`\n"]
    md.append("## ペア仕様（/spot/pairs, /spot/status）\n")
    md.append("```json\n" + json.dumps(res.get("spot"), ensure_ascii=False, indent=2) + "\n```\n")
    if "v1" in res:
        v1 = res["v1"]
        md.append(f"## V1 公式ロウソク足との一致（{v1['start']} 〜 {v1['end']}、15min）\n")
        md.append(f"自前の足 {v1['n_bars']} 本（約定なし {v1['n_empty_bars']} 本）、公式 {v1['n_official']} 本。"
                  f" ohlc_match_rate 最大の shift_bars = **{v1['best_shift_bars']}**\n\n")
        md.append(_fmt_table(v1["table"]))
    if "v5" in res:
        md.append(f"## V5 日付境界\n\n推定: **{res['v5']['guess']}**\n\n")
        md.append(_fmt_table(res["v5"]["days"]))
    if "v2" in res:
        v2 = res["v2"]
        md.append(f"## V2 約定の side\n\n板 {v2['n_snapshots']} 回、約定 {v2['n_trades']} 件。depth のキー: {v2['depth_keys']}\n\n")
        md.append("age_ms は約定時刻と直前の板の timestamp の差。\n\n")
        md.append(_fmt_table(v2["by_age"]))
    if "v3" in res:
        md.append(f"## V3 最古の約定日\n\n最古: **{res['v3']['oldest']}**\n\n```\n{json.dumps(res['v3'], ensure_ascii=False)}\n```\n")
    md.append("## V4 Public REST の応答ステータス\n\n")
    md.append("```\n" + json.dumps(res["status_counts"]) + "\n```\n")
    return "\n".join(md)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="btc_jpy")
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="reports/phase0")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--v2-minutes", type=float, default=5.0)
    ap.add_argument("--v2-interval", type=float, default=1.0)
    ap.add_argument("--v3-lower", default="2014-01-01")
    ap.add_argument("--only", nargs="*", default=["spot", "v1", "v5", "v2", "v3"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    root = Path(args.root)
    client = PublicClient(min_interval=0.3)
    res: dict = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "pair": args.pair}

    def step(name, fn):
        if name not in args.only:
            return
        try:
            res[name] = fn()
        except Exception as exc:  # 1項目の失敗で他の検証を止めない
            log.exception("%s 失敗", name)
            res[name + "_error"] = repr(exc)

    step("spot", lambda: check_spot(args.pair))
    step("v1", lambda: v1_candles(client, args.pair, root, args.days))
    if "v1" in res:
        step("v5", lambda: v5_day_boundary(args.pair, root, date.fromisoformat(res["v1"]["start"]), date.fromisoformat(res["v1"]["end"])))
    v2_client = PublicClient(min_interval=0.0)
    step("v2", lambda: v2_side(v2_client, args.pair, args.v2_minutes, args.v2_interval))
    step("v3", lambda: v3_oldest(client, args.pair, date.fromisoformat(args.v3_lower)))
    counts = client.status_counts + v2_client.status_counts
    res["status_counts"] = {str(k): v for k, v in counts.items()}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase0.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    md = render(res)
    (out / "phase0.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
