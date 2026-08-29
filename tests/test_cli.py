"""The CLI's read-only subcommands, checked at their call sites.

`status` and `mcp` are the two commands an owner runs from a shell that is not
a Lambda: no bitbank keys, no `TA_STORAGE__S3_BUCKET`. Both must therefore
build a context with no exchange, because building one under paper trading
reads the simulated account from S3 and boto3 rejects the empty bucket name
with `Invalid bucket name ""`.
"""

from __future__ import annotations

import pytest

from trade_agent import cli


class _Args:
    """The attributes `_context` reads off an argparse namespace."""

    config = None
    local = False


def _spy_on_build_context(monkeypatch) -> dict:
    seen: dict = {}

    def spy(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here; only the arguments matter")

    monkeypatch.setattr(cli, "build_context", spy)
    return seen


@pytest.mark.parametrize("command", ["status", "mcp"])
def test_the_read_only_commands_ask_for_no_exchange(monkeypatch, command):
    """The option existing is no use if the command stops passing it."""
    seen = _spy_on_build_context(monkeypatch)
    args = _Args()
    args.tool = "get_status"
    args.args = "{}"

    with pytest.raises(RuntimeError):
        getattr(cli, f"cmd_{command}")(args)

    assert seen.get("needs_exchange") is False
    assert seen.get("needs_trading_credentials") is False


def test_the_trading_commands_still_get_an_exchange(monkeypatch):
    """The default has to stay True: `tick` reads the ticker on every pass."""
    seen = _spy_on_build_context(monkeypatch)

    with pytest.raises(RuntimeError):
        cli._context(_Args())

    assert seen.get("needs_exchange") is True


def test_mcp_reaches_a_tool_end_to_end(capsys):
    """Smoke test of the wiring through the real context builder. `--local`
    never reaches S3, so this passes with or without the fix — it guards the
    rest of the path, not the bucket."""
    args = _Args()
    args.local = True
    args.tool = "get_status"
    args.args = "{}"

    assert cli.cmd_mcp(args) == 0
    assert "equity_jpy" in capsys.readouterr().out


def test_building_an_exchange_is_what_reads_the_bucket():
    """The chain the fix breaks, pinned so it cannot quietly move.

    Under paper trading `_build_exchange` loads the simulated account from
    blob storage during __init__. With `TA_STORAGE__S3_BUCKET` unset — a
    CloudShell session — that is boto3's `Invalid bucket name ""`. Nothing
    else in a context build touches blobs, which is why skipping the exchange
    is enough to make the read-only commands work there.
    """
    from trade_agent.orchestrator import context as ctx_mod
    from trade_agent.storage.memory import MemoryStore

    store = MemoryStore()

    def unset_bucket(*args, **kwargs):
        raise ValueError('Invalid bucket name ""')

    store.blobs.get_json = unset_bucket

    with pytest.raises(ValueError, match="Invalid bucket name"):
        ctx_mod.build_context(owner="cli", needs_trading_credentials=False,
                              store=store, secrets=_NoSecrets(),
                              notifier=object())

    built = ctx_mod.build_context(owner="cli", needs_trading_credentials=False,
                                  needs_exchange=False, store=store,
                                  secrets=_NoSecrets(), notifier=object())
    assert not built.exchange


class _NoSecrets:
    def get(self, *args, **kwargs):
        raise AssertionError("a read-only context must not ask for secrets")
