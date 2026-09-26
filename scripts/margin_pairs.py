"""bitbank の信用取引の対応ペアと条件を公開 API（/spot/pairs、認証不要）から一覧にする。2026-09-27 オーナー指示。

    python scripts/margin_pairs.py

判定: margin_long_interest / margin_short_interest が入っていて（None でない）、stop_margin_long_order / stop_margin_short_order が
false のペアを「信用取引できる」とみなす。結果は reports/margin/report.md。
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from bbdata.client import PublicClient

FIELDS = ["margin_open_maker_fee_rate_quote", "margin_open_taker_fee_rate_quote", "margin_close_maker_fee_rate_quote",
          "margin_close_taker_fee_rate_quote", "margin_long_interest", "margin_short_interest",
          "margin_current_individual_ratio", "margin_current_individual_until", "margin_next_individual_ratio",
          "margin_next_individual_until", "stop_margin_long_order", "stop_margin_short_order", "is_enabled"]


def pct(v) -> str:
    if v is None or v == "":
        return "-"
    try:
        return f"{float(v) * 100:.3f}%"
    except (TypeError, ValueError):
        return str(v)


def main() -> None:
    api = PublicClient(base_url="https://api.bitbank.cc/v1", min_interval=0.5)
    pairs = api.get("/spot/pairs")["pairs"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows, enabled = [], []
    for p in pairs:
        li, si = p.get("margin_long_interest"), p.get("margin_short_interest")
        can = li not in (None, "") and si not in (None, "") and not p.get("stop_margin_long_order") and not p.get("stop_margin_short_order")
        if can:
            enabled.append(p["name"])
        rows.append(p)
    md = [f"# bitbank の信用取引の対応ペア（{now}、/spot/pairs から）\n",
          f"- 信用取引できると判定したペア（{len(enabled)}）: " + ("、".join(enabled) if enabled else "なし"),
          "- 判定: 建玉金利（long / short）が入っていて、信用の新規建て停止フラグが両方 false のもの",
          "- 金利は 1 日あたり。委託保証金率（個人）は現在の値と、次に予定されている値\n",
          "| ペア | 現物 | 信用 | 建玉金利 long/日 | short/日 | 新規 maker | 新規 taker | 返済 maker | 返済 taker | 保証金率（個人）現在 | まで | 次 | から | 停止 long | 停止 short |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(rows, key=lambda r: (r["name"] not in enabled, r["name"])):
        md.append("| " + " | ".join([
            p["name"], "○" if p.get("is_enabled") else "×", "○" if p["name"] in enabled else "-",
            pct(p.get("margin_long_interest")), pct(p.get("margin_short_interest")),
            pct(p.get("margin_open_maker_fee_rate_quote")), pct(p.get("margin_open_taker_fee_rate_quote")),
            pct(p.get("margin_close_maker_fee_rate_quote")), pct(p.get("margin_close_taker_fee_rate_quote")),
            pct(p.get("margin_current_individual_ratio")), str(p.get("margin_current_individual_until") or "-"),
            pct(p.get("margin_next_individual_ratio")), str(p.get("margin_next_individual_until") or "-"),
            str(p.get("stop_margin_long_order")), str(p.get("stop_margin_short_order"))]) + " |")
    text = "\n".join(md) + "\n"
    out = Path("reports/margin")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
