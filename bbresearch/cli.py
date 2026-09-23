"""python -m bbresearch run    --config configs/research.yaml --out reports/research
python -m bbresearch search --config configs/research.yaml --grid configs/search.yaml --out reports/search"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .pipeline import load_config, render, run


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="bbresearch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="研究パイプラインを実行してレポートを書き出す")
    r.add_argument("--config", default="configs/research.yaml")
    r.add_argument("--out", default="reports/research")
    r.add_argument("-v", "--verbose", action="store_true")
    sr = sub.add_parser("search", help="SPEC §7 の範囲でパラメータを探索する（開発期間のみ）")
    sr.add_argument("--config", default="configs/research.yaml")
    sr.add_argument("--grid", default="configs/search.yaml")
    sr.add_argument("--out", default="reports/search")
    sr.add_argument("--workers", type=int, default=None)
    sr.add_argument("-v", "--verbose", action="store_true")
    dg = sub.add_parser("diagnose", help="損失の要因分解（開発期間のみ）")
    dg.add_argument("--config", default="configs/research.yaml")
    dg.add_argument("--sets", default="configs/diagnose.yaml")
    dg.add_argument("--out", default="reports/diagnose")
    dg.add_argument("-v", "--verbose", action="store_true")
    ms = sub.add_parser("maker-search", help="深い指値・短い保有・売りも指値の決済条件を探索し、確認期間で1回評価する")
    ms.add_argument("--config", default="configs/research.yaml")
    ms.add_argument("--grid", default="configs/maker_search.yaml")
    ms.add_argument("--out", default="reports/maker_search")
    ms.add_argument("--workers", type=int, default=None)
    ms.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "maker-search":
        from .maker_search import main as maker_main

        maker_main(args)
        return
    if args.cmd == "diagnose":
        from .diagnose import main as diagnose_main

        diagnose_main(args)
        return
    if args.cmd == "search":
        from .search import render as render_search
        from .search import run_search

        res = run_search(load_config(args.config), load_config(args.grid), out, workers=args.workers)
        (out / "report.json").write_text(json.dumps(res["report"], ensure_ascii=False, indent=2, default=str))
        md = render_search(res)
        (out / "report.md").write_text(md)
        print(md)
        return
    rep = run(load_config(args.config), out)
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    md = render(rep)
    (out / "report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
