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

    p = sub.add_parser("preflight",
                       help="check everything a deployment needs before it trades")
    p.set_defaults(func=cmd_preflight)

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

def pair_drift(config, live) -> list[tuple[str, str, str]]:
    """(setting, configured, live) for every pair constant spec 2 pins down."""
    cfg = config.exchange
    return [
        ("min_order_btc", str(cfg.min_order_btc), str(live.min_order_btc)),
        ("price_digits", str(cfg.price_digits), str(live.price_digits)),
        ("amount_digits", str(cfg.amount_digits), str(live.amount_digits)),
        ("maker_fee_rate", str(cfg.maker_fee_rate), str(live.maker_fee_rate)),
        ("taker_fee_rate", str(cfg.taker_fee_rate), str(live.taker_fee_rate)),
    ]


def _report_pair(config, live) -> int:
    rows = pair_drift(config, live)
    drift = [name for name, configured, actual in rows if configured != actual]
    print(f"pair: {live.name}  enabled={live.is_enabled}  "
          f"orders_suspended={live.stop_order_and_cancel}")
    print(f"{'setting':<16} {'config':>16} {'exchange':>16}  status")
    for name, configured, actual in rows:
        print(f"{name:<16} {configured:>16} {actual:>16}  "
              f"{'OK' if configured == actual else 'DRIFT'}")
    if drift:
        print(f"\n{len(drift)} setting(s) differ from config/default.yaml: "
              f"{', '.join(drift)}")
        print("Update the config before trading; these values size every order.")
        return 2
    print("\nAll exchange constants match the configuration.")
    return 0


def cmd_verify_pair(args) -> int:
    """Spec 2 asks for the pair constants to be re-checked against bitbank.

    `GET /v1/spot/pairs` is unauthenticated and carries the authoritative
    minimum order size, tick size and fee rates, so this is a live diff rather
    than a note in a document that goes stale.
    """
    ctx = _context(args, needs_trading_credentials=False)
    return _report_pair(ctx.config, ctx.exchange.get_pair_settings())


def cmd_preflight(args) -> int:
    """Everything a fresh deployment needs to be right, checked in one go.

    Run this straight after deploying and again before each phase change. It
    reaches the real exchange with the real credentials, so a wrong key or a
    drifted constant surfaces here rather than at the first order.
    """
    from .exchange.bitbank import BitbankClient
    from .storage.secrets import default_provider

    config = load_config(args.config) if args.config else load_config()
    secrets = default_provider()
    failures: list[str] = []

    print("=" * 62)
    print(f"trade-agent preflight  env={config.system.environment}  "
          f"phase={config.system.phase}  paper={config.system.paper_trading}")
    print("=" * 62)

    def public_client(credentials=None) -> BitbankClient:
        return BitbankClient(
            public_base_url=config.exchange.public_base_url,
            private_base_url=config.exchange.private_base_url,
            pair=config.exchange.pair, credentials=credentials,
            timeout=config.exchange.http_timeout_seconds,
            max_retries=1)

    # 1. Public market data.
    print("\n[1/4] public market data")
    try:
        ticker = public_client().get_ticker()
        print(f"      OK  last={ticker['last']} JPY  "
              f"bid={ticker['buy']} ask={ticker['sell']}")
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        print(f"      FAIL  {exc}")
        failures.append("public market data unreachable")

    # 2. Exchange constants (spec 2).
    print("\n[2/4] exchange constants vs config")
    try:
        if _report_pair(config, public_client().get_pair_settings()) != 0:
            failures.append("exchange constants differ from config")
    except Exception as exc:  # noqa: BLE001
        print(f"      FAIL  {exc}")
        failures.append("could not read the pair settings")

    # 3. Private credentials.
    print("\n[3/4] bitbank private API credentials")
    key = secrets.get_optional(config.secrets.ssm_bitbank_api_key)
    secret = secrets.get_optional(config.secrets.ssm_bitbank_api_secret)
    if not key or not secret:
        print("      FAIL  credentials not readable from SSM")
        failures.append("bitbank credentials missing")
    else:
        try:
            balances = public_client((key, secret)).get_balances()
            jpy = balances.get(config.exchange.quote_asset)
            btc = balances.get(config.exchange.base_asset)
            print(f"      OK  key works. JPY free={jpy.free if jpy else 0} "
                  f"BTC free={btc.free if btc else 0}")
            if jpy is not None and jpy.free < config.capital.initial_equity_jpy:
                print(f"      NOTE  JPY balance is below the configured initial "
                      f"capital ({config.capital.initial_equity_jpy})")
        except Exception as exc:  # noqa: BLE001
            print(f"      FAIL  {exc}")
            failures.append("bitbank credentials rejected")

    # 4. Anthropic key.
    print("\n[4/4] Anthropic API key")
    if secrets.get_optional(config.secrets.ssm_anthropic_api_key):
        print("      OK  present in SSM")
    else:
        print("      FAIL  not readable from SSM")
        failures.append("Anthropic API key missing")

    print("\n" + "-" * 62)
    if failures:
        print(f"{len(failures)} problem(s) found:")
        for item in failures:
            print(f"  - {item}")
        return 1
    if config.system.paper_trading:
        print("All checks passed. Paper trading is ON: no live order can be "
              "placed until it is switched off.")
    else:
        print("All checks passed. LIVE TRADING IS ENABLED.")
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
