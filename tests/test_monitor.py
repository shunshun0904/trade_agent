import json
import math

from bbresearch.monitor import append_jsonl, load_jsonl, render, stats


def _row(date, total, btc, cash=0.1, tcash=0.08):
    return {"date": date, "total_jpy": total, "jpy": total * cash, "values_jpy": {"btc_jpy": total * 0.5},
            "fractions": {"btc_jpy": 0.5, "trx_jpy": 0.4}, "cash_frac": cash,
            "target_weights": {"btc_jpy": 0.42, "trx_jpy": 0.37, "bnb_jpy": 0.13}, "target_cash": tcash,
            "est_vol": 0.3, "btc_price": btc}


def test_stats_index_vol_and_drawdown():
    rows = [_row("2026-10-01", 100_000, 10_000_000), _row("2026-10-02", 110_000, 10_500_000),
            _row("2026-10-03", 99_000, 10_000_000), _row("2026-10-03", 104_500, 10_100_000),  # 同じ日は最後を使う
            _row("2026-10-04", 104_500, 10_100_000), _row("2026-10-05", 108_000, 10_300_000),
            _row("2026-10-06", 107_000, 10_200_000)]
    st = stats(rows)
    assert len(st["rows"]) == 6 and st["index"][0] == 100 and abs(st["index"][1] - 110) < 1e-9
    assert abs(st["btc_index"][-1] - 102) < 1e-9
    assert abs(st["max_drawdown"] - (104_500 / 110_000 - 1)) < 1e-9
    assert abs(st["drawdown"] - (107_000 / 110_000 - 1)) < 1e-9
    rets = [math.log(b / a) for a, b in zip([100_000, 110_000, 104_500, 104_500, 108_000], [110_000, 104_500, 104_500, 108_000, 107_000])]
    m = sum(rets) / 5
    assert abs(st["vol_30d"] - math.sqrt(sum((x - m) ** 2 for x in rets) / 4 * 365)) < 1e-9


def test_render_contains_holdings_targets_and_history():
    rows = [_row("2026-10-01", 100_000, 10_000_000), _row("2026-10-02", 103_000, 10_100_000)]
    reb = [{"date": "2026-10-01", "target_weights": {"btc_jpy": 0.42}, "target_cash": 0.08,
            "orders": [{"pair": "btc_jpy", "side": "buy", "frac": 0.2}], "dry_run": True}]
    page = render(rows, reb, 0.30)
    assert "<title>Portfolio Monitor</title>" in page and "103,000 円" in page and "+3.0%" in page
    assert "btc_jpy buy 20.0%" in page and "ドライラン" in page
    assert "<polyline" in page and "JPY の割合" in page
    assert "記録が 2 日分" not in page
    empty = render([], [], 0.30)
    assert "まだ記録がありません" in empty


def test_jsonl_roundtrip(tmp_path):
    p = tmp_path / "d.jsonl"
    assert load_jsonl(p) == []
    append_jsonl(p, {"a": 1, "s": "日本語"})
    append_jsonl(p, {"a": 2})
    assert load_jsonl(p) == [{"a": 1, "s": "日本語"}, {"a": 2}]
    assert "日本語" in p.read_text()
    assert json.loads(p.read_text().splitlines()[0])["s"] == "日本語"
