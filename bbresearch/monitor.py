"""ポートフォリオのモニタリング（2026-09-27 オーナー指示）。

- 記録: docs/monitor/daily.jsonl（1 日 1 行。評価額、保有の内訳、目標の重み、BTC 価格）と
  docs/monitor/rebalances.jsonl（リバランスの計画。scripts/rebalance.py が追記）。
- 画面: docs/monitor/index.html を記録から生成する（データを埋め込んだ 1 ファイル。GitHub Pages か手元で開く）。
  オーナー決定: 少額なので金額は伏せない（そのうちリポジトリを非公開にする）。
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

DAYS_PER_YEAR = 365


def load_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def stats(daily: list[dict]) -> dict:
    """日次の記録から、指数（開始 = 100）、実現ボラ、ドローダウンを出す。同じ日が複数あれば最後を使う。"""
    by_day: dict[str, dict] = {}
    for r in daily:
        by_day[r["date"]] = r
    rows = [by_day[d] for d in sorted(by_day)]
    if not rows:
        return {"rows": [], "index": [], "btc_index": [], "vol_30d": None, "drawdown": None, "max_drawdown": None}
    v0, b0 = rows[0]["total_jpy"], rows[0]["btc_price"]
    index = [r["total_jpy"] / v0 * 100 if v0 else None for r in rows]
    btc = [r["btc_price"] / b0 * 100 if b0 else None for r in rows]
    rets = [math.log(rows[i]["total_jpy"] / rows[i - 1]["total_jpy"]) for i in range(1, len(rows))
            if rows[i - 1]["total_jpy"] > 0 and rows[i]["total_jpy"] > 0]
    last = rets[-30:]
    vol = None
    if len(last) >= 5:
        m = sum(last) / len(last)
        vol = math.sqrt(sum((x - m) ** 2 for x in last) / (len(last) - 1) * DAYS_PER_YEAR)
    peak, dd, mdd = -1.0, 0.0, 0.0
    for x in index:
        peak = max(peak, x)
        dd = x / peak - 1
        mdd = min(mdd, dd)
    return {"rows": rows, "index": index, "btc_index": btc, "vol_30d": vol, "drawdown": dd if index else None,
            "max_drawdown": mdd}


def _svg_lines(series: list[tuple[str, list[float], str]], dates: list[str], w: int = 760, h: int = 220) -> str:
    """複数の折れ線（同じ x）。series は (名前, 値, 色)。"""
    vals = [v for _, s, _ in series for v in s if v is not None]
    if len(dates) < 2 or not vals:
        return '<p class="note">記録が 2 日分そろうと表示します。</p>'
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad
    L, R, T, B = 44, 12, 10, 24
    x = lambda i: L + i / (len(dates) - 1) * (w - L - R)  # noqa: E731
    y = lambda v: T + (1 - (v - lo) / (hi - lo)) * (h - T - B)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    step = (hi - lo) / 4
    for k in range(5):
        v = lo + step * k
        out.append(f'<line x1="{L}" x2="{w - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#222a3a"/>'
                   f'<text x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.0f}</text>')
    n_lab = min(6, len(dates))
    for j in range(n_lab):
        i = round(j * (len(dates) - 1) / max(1, n_lab - 1))
        out.append(f'<text x="{x(i):.1f}" y="{h - 6}" text-anchor="middle">{dates[i][5:]}</text>')
    for name, s, color in series:
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(s) if v is not None)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
    out.append("</svg>")
    legend = " ".join(f'<span><i style="background:{c}"></i>{html.escape(n)}</span>' for n, _, c in series)
    return "".join(out) + f'<div class="legend">{legend}</div>'


def render(daily: list[dict], rebalances: list[dict], target_vol: float = 0.30) -> str:
    st = stats(daily)
    rows = st["rows"]
    last = rows[-1] if rows else None
    pct = lambda v: "-" if v is None else f"{v * 100:+.1f}%"  # noqa: E731
    yen = lambda v: "-" if v is None else f"{v:,.0f} 円"  # noqa: E731
    head = ['<title>Portfolio Monitor</title>',
            '<style>:root{color-scheme:dark;--bg:#0b0e14;--panel:#111621;--ink:#e6ebf2;--ink2:#9aa4b5;--line:#222a3a;'
            '--vp:#3b8fe8;--poc:#f5b32b;--up:#22c58a;--down:#f2524f}'
            'body{margin:0;background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans JP","Hiragino Sans",system-ui,sans-serif;font-size:13px}'
            '.app{max-width:1100px;margin-inline:auto;padding-block:14px 30px;padding-inline:16px;display:grid;gap:12px}'
            'h1{font-size:16px;margin:0}h2{font-size:13px;margin:0 0 8px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:12px 14px}'
            '.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}.tile{border-top:2px solid var(--line);padding-top:4px}'
            '.tile .k{font-size:11px;color:var(--ink2)}.tile .v{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:18px;font-variant-numeric:tabular-nums}'
            'table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th{text-align:left;color:var(--ink2);font-weight:500;font-size:11px;border-bottom:1px solid var(--line);padding:3px 6px}'
            'td{padding:4px 6px;border-bottom:1px solid #1a2130}td.n,th.n{text-align:right;font-family:"IBM Plex Mono",ui-monospace,monospace}'
            'svg text{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10px;fill:var(--ink2)}.legend{display:flex;gap:14px;font-size:11px;color:var(--ink2)}'
            '.legend i{display:inline-block;width:12px;height:3px;margin-right:5px;vertical-align:middle}.note{color:var(--ink2);font-size:12px}'
            '.pos{color:var(--up)}.neg{color:var(--down)}.wrap{overflow-x:auto}</style>']
    body = ['<div class="app">', '<h1>ポートフォリオ・モニター</h1>']
    if not last:
        body.append('<div class="panel"><p class="note">まだ記録がありません。monitor.yml が 1 日 1 回、残高を記録します。</p></div>')
        return "\n".join(head + body + ["</div>"])
    total0 = rows[0]["total_jpy"]
    since = last["total_jpy"] / total0 - 1 if total0 else None
    vol_cls = "" if st["vol_30d"] is None else ("neg" if st["vol_30d"] > target_vol * 1.2 else "pos")
    vol_txt = "-" if st["vol_30d"] is None else f"{st['vol_30d']:.1%}"
    body.append('<div class="panel"><div class="tiles">'
                f'<div class="tile"><div class="k">評価額（{html.escape(last["date"])}）</div><div class="v">{yen(last["total_jpy"])}</div></div>'
                f'<div class="tile"><div class="k">開始（{html.escape(rows[0]["date"])}）から</div><div class="v {"pos" if (since or 0) >= 0 else "neg"}">{pct(since)}</div></div>'
                f'<div class="tile"><div class="k">実現ボラ（直近 30 日、年率）/ 目標 {target_vol:.0%}</div><div class="v {vol_cls}">{vol_txt}</div></div>'
                f'<div class="tile"><div class="k">ドローダウン（現在 / 最大）</div><div class="v">{pct(st["drawdown"])} / {pct(st["max_drawdown"])}</div></div>'
                f'<div class="tile"><div class="k">JPY の割合 / 目標</div><div class="v">{last["cash_frac"] * 100:.0f}% / {last["target_cash"] * 100:.0f}%</div></div>'
                '</div></div>')
    dates = [r["date"] for r in rows]
    body.append('<div class="panel"><h2>資産の推移（開始 = 100）</h2>'
                + _svg_lines([("ポートフォリオ", st["index"], "#3b8fe8"), ("BTC 単独", st["btc_index"], "#f5b32b")], dates) + '</div>')
    body.append('<div class="panel"><h2>JPY の割合の推移</h2>'
                + _svg_lines([("JPY の割合（%）", [r["cash_frac"] * 100 for r in rows], "#8f7ff0"),
                              ("目標（%）", [r["target_cash"] * 100 for r in rows], "#66718a")], dates) + '</div>')
    # 保有 vs 目標
    pairs = sorted(set(last["fractions"]) | set(last["target_weights"]))
    trs = []
    for p in pairs:
        cur, tgt = last["fractions"].get(p, 0.0), last["target_weights"].get(p, 0.0)
        if cur < 0.001 and tgt < 0.001:
            continue
        d = tgt - cur
        trs.append(f'<tr><td>{html.escape(p)}</td><td class="n">{yen(last["values_jpy"].get(p))}</td><td class="n">{cur:.1%}</td>'
                   f'<td class="n">{tgt:.1%}</td><td class="n {"pos" if d >= 0 else "neg"}">{d * 100:+.1f} pt</td></tr>')
    trs.append(f'<tr><td>JPY</td><td class="n">{yen(last["jpy"])}</td><td class="n">{last["cash_frac"]:.1%}</td>'
               f'<td class="n">{last["target_cash"]:.1%}</td><td class="n">{(last["target_cash"] - last["cash_frac"]) * 100:+.1f} pt</td></tr>')
    body.append('<div class="panel"><h2>現在の保有と目標の重み</h2><div class="wrap"><table><thead><tr><th>銘柄</th><th class="n">評価額</th>'
                '<th class="n">現在</th><th class="n">目標</th><th class="n">差</th></tr></thead><tbody>' + "".join(trs) + '</tbody></table></div>'
                f'<p class="note">目標は {html.escape(last["date"])} までのデータで推定した重み（推定ボラ {last.get("est_vol", 0) * 100:.1f}%'
                + {None: "", True: "、BTC は移動平均の上", False: "、BTC は移動平均の下、暗号資産を縮める"}[last.get("trend")]
                + "）。差が総額の 1% 以上のものが売買の対象。</p></div>")
    # リバランスの履歴
    hist = []
    for r in reversed(rebalances[-24:]):
        tw = "、".join(f"{p} {w:.0%}" for p, w in r["target_weights"].items()) + f"、JPY {r['target_cash']:.0%}"
        od = "、".join(f"{o['pair']} {o['side']} {o['frac']:.1%}" for o in r.get("orders", [])) or "なし"
        kind = "月初" if r.get("kind", "monthly") == "monthly" else "週次"
        hist.append(f'<tr><td>{html.escape(r["date"])}</td><td>{kind}</td><td>{html.escape(tw)}</td><td>{html.escape(od)}</td>'
                    f'<td>{"ドライラン" if r.get("dry_run", True) else "発注"}</td></tr>')
    body.append('<div class="panel"><h2>リバランスの履歴</h2><div class="wrap"><table><thead><tr><th>日付</th><th>方式</th><th>目標</th><th>計画した売買（総額比）</th><th>種別</th></tr></thead>'
                '<tbody>' + ("".join(hist) or '<tr><td colspan="5" class="note">まだありません</td></tr>') + '</tbody></table></div></div>')
    body.append("</div>")
    return "\n".join(head + body)
