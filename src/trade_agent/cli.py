"""Command line entry point.

Everything the Lambdas do can be run locally through here. `--local` swaps in
the in-memory store and the offline LLM so a full decision cycle can be
exercised with no AWS account, no API key and no money at risk — which is how
the acceptance checks in spec 14 are meant to be reproduced by hand.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from decimal import Decimal
from pathlib import Path

from .config import load_config
from .models.state import CycleTrigger
from .orchestrator.context import build_context
from .timeutil import Clock


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    if args.command is None:
        parser.print_help()
        return 1
    return args.func(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trade-agent",
                                     description="BTC/JPY multi-agent trading system")
    parser.add_argument("--config", help="path to a config YAML")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--local", action="store_true",
                        help="fully offline: in-memory storage, the LLM stub "
                             "and a synthetic market. No network, no API key.")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("verify-pair",
                       help="re-check the exchange constants against bitbank")
    p.set_defaults(func=cmd_verify_pair)

    p = sub.add_parser("snapshot", help="build and print a MarketSnapshot")
    p.add_argument("--prompt", action="store_true",
                   help="print the exact JSON the agents receive")
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("status", help="print the owner-facing status payload")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("tick", help="run the 5-minute monitoring pass once")
    p.set_defaults(func=cmd_tick)

    p = sub.add_parser("screen", help="run the 30-minute screener once")
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("decide", help="run one full decision cycle")
    p.add_argument("--trigger", default="manual",
                   choices=[t.value for t in CycleTrigger])
    p.add_argument("--cycle-id")
    p.set_defaults(func=cmd_decide)

    p = sub.add_parser("reflect", help="run the post-settlement analysis")
    p.set_defaults(func=cmd_reflect)

    p = sub.add_parser("backfill", help="download historical candles")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--candle-type", default="1hour")
    p.add_argument("--out", default="data/candles")
    p.set_defaults(func=cmd_backfill)

    p = sub.add_parser("backtest", help="replay the deterministic layers")
    p.add_argument("--candles", default="data/candles")
    p.add_argument("--candle-type", default="1hour")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("mcp", help="call an MCP tool locally")
    p.add_argument("tool")
    p.add_argument("--args", default="{}", help="JSON object of arguments")
    p.set_defaults(func=cmd_mcp)
    return parser


# -- commands -------------------------------------------------------------

def cmd_verify_pair(args) -> int:
    """Spec 2 asks for the pair constants to be re-checked against bitbank.

    `GET /v1/spot/pairs` is unauthenticated and carries the authoritative
    minimum order size, tick size and fee rates, so this is a live diff rather
    than a note in a document that goes stale.
    """
    ctx = _context(args, needs_trading_credentials=False)
    live = ctx.exchange.get_pair_settings()
    cfg = ctx.config.exchange

    rows = [
        ("min_order_btc", cfg.min_order_btc, live.min_order_btc),
        ("price_digits", cfg.price_digits, live.price_digits),
        ("amount_digits", cfg.amount_digits, live.amount_digits),
        ("maker_fee_rate", cfg.maker_fee_rate, live.maker_fee_rate),
        ("taker_fee_rate", cfg.taker_fee_rate, live.taker_fee_rate),
    ]
    drift = [name for name, configured, actual in rows if str(configured) != str(actual)]
    print(f"pair: {live.name}  enabled={live.is_enabled}  "
          f"orders_suspended={live.stop_order_and_cancel}")
    print(f"{'setting':<16} {'config':>16} {'exchange':>16}  status")
    for name, configured, actual in rows:
        status = "OK" if str(configured) == str(actual) else "DRIFT"
        print(f"{name:<16} {str(configured):>16} {str(actual):>16}  {status}")
    if drift:
        print(f"\n{len(drift)} setting(s) differ from config/default.yaml: "
              f"{', '.join(drift)}")
        print("Update the config before trading; these values size every order.")
        return 2
    print("\nAll exchange constants match the configuration.")
    return 0


def cmd_snapshot(args) -> int:
    ctx = _context(args, needs_trading_credentials=False)
    snapshot = ctx.snapshot_builder().build()
    if args.prompt:
        print(snapshot.to_prompt_json())
    else:
        print(json.dumps(snapshot.to_prompt_dict(), ensure_ascii=False, indent=2))
    if snapshot.data_quality:
        print("\ndata quality warnings:", file=sys.stderr)
        for note in snapshot.data_quality:
            print(f"  - {note}", file=sys.stderr)
    return 0


def cmd_status(args) -> int:
    from .mcp.tools import call_tool

    ctx = _context(args, needs_trading_credentials=False)
    print(json.dumps(call_tool(ctx, "get_status", {}), ensure_ascii=False, indent=2))
    return 0


def cmd_tick(args) -> int:
    from .handlers.tick import run

    print(_dump(run(_context(args))))
    return 0


def cmd_screen(args) -> int:
    from .handlers.screen import run

    print(_dump(run(_context(args))))
    return 0


def cmd_decide(args) -> int:
    from .handlers.decide import run

    result = run(_context(args),
                 {"trigger": args.trigger, "cycle_id": args.cycle_id})
    print(_dump(result))
    return 0


def cmd_reflect(args) -> int:
    from .handlers.reflect import run

    print(_dump(run(_context(args), {})))
    return 0


def cmd_backfill(args) -> int:
    from .data.backfill import backfill

    ctx = _context(args, needs_trading_credentials=False)
    candles = backfill(ctx.exchange, candle_type=args.candle_type,
                       days=args.days, out_dir=args.out)
    if not candles:
        print("no candles downloaded", file=sys.stderr)
        return 1
    print(f"{len(candles)} {args.candle_type} candles "
          f"({candles[0].opened_at.date()} .. {candles[-1].opened_at.date()}) "
          f"-> {args.out}")
    return 0


def cmd_backtest(args) -> int:
    from .backtest import run_backtest
    from .data.backfill import load_cached

    config = load_config(args.config) if args.config else load_config()
    candles = load_cached(args.candles, args.candle_type)
    if len(candles) < 100:
        print(f"only {len(candles)} candles in {args.candles}; "
              "run `trade-agent backfill` first", file=sys.stderr)
        return 1
    result = run_backtest(candles, config)
    print(json.dumps(result.summary(), ensure_ascii=False, indent=2))
    print("\nNote: this replays the deterministic layers only — screener, "
          "sizing, fees and stop mechanics. It says nothing about how the "
          "agents would have decided.", file=sys.stderr)
    return 0


def cmd_mcp(args) -> int:
    from .mcp.tools import call_tool

    ctx = _context(args, needs_trading_credentials=False)
    result = call_tool(ctx, args.tool, json.loads(args.args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


# -- helpers --------------------------------------------------------------

def _context(args, *, needs_trading_credentials: bool = True):
    config = load_config(args.config) if args.config else load_config()
    if args.local:
        config = config.model_copy(deep=True)
        config.storage.backend = "memory"
        config.llm.provider = "stub"
        config.system.paper_trading = True
    return build_context(config=config, clock=Clock(), owner="cli",
                         needs_trading_credentials=needs_trading_credentials,
                         offline=bool(args.local))


def _dump(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=_default)


def _default(value):
    if isinstance(value, Decimal):
        return float(value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
