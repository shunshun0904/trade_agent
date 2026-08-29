"""Checks on scripts/tool.sh — read-only MCP tools from a shell.

Two things in it are copies of facts that live in the code: the tool names, and
the environment variables the Lambdas get from template.yaml. Copies go stale
silently, and a stale copy here is not a crash — it is a script that reads the
wrong tables and reports an empty system as if that were the answer.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "tool.sh"
SCRIPT = SCRIPT_PATH.read_text(encoding="utf-8")


def _listed(name: str) -> set[str]:
    match = re.search(rf'^{name}="([^"]*)"', SCRIPT, re.MULTILINE)
    assert match, f"no {name} list found in tool.sh"
    return set(match.group(1).split())


def test_it_offers_exactly_the_read_only_tools():
    from trade_agent.mcp.tools import READ_ONLY

    assert _listed("READERS") == set(READ_ONLY)


def test_it_refuses_exactly_the_writing_tools():
    """Named individually rather than inferred, so a new writing tool that
    nobody adds here is caught now rather than the first time someone pauses
    live trading by typing a word into a shell."""
    from trade_agent.mcp.tools import _HANDLERS, READ_ONLY

    assert _listed("WRITERS") == set(_HANDLERS) - set(READ_ONLY)


def test_it_passes_the_table_prefix_the_stack_uses():
    """The bug this script exists to avoid: without it the CLI reads
    `trade-agent-agent-calls` while the deployed tables are
    `trade-agent-prod-agent-calls`, and finds a system that has never run."""
    template = (ROOT / "template.yaml").read_text(encoding="utf-8")
    for variable in ("TA_STORAGE__TABLE_PREFIX", "TA_STORAGE__S3_BUCKET"):
        assert f"{variable}:" in template, f"{variable} left template.yaml"
        assert f"export {variable}=" in SCRIPT, (
            f"tool.sh must pass {variable} through to the CLI")

    prefix = re.search(r'export TA_STORAGE__TABLE_PREFIX="([^"]*)"', SCRIPT)
    assert prefix and prefix.group(1) == "trade-agent-${ENVIRONMENT}", (
        "the prefix must be built from the environment, not hardcoded")


def test_a_writing_tool_is_refused_before_anything_is_read():
    """Refused by name, without needing AWS credentials or a deployed stack —
    so the refusal cannot be bypassed by running it somewhere unconfigured."""
    result = subprocess.run(["bash", str(SCRIPT_PATH), "pause_trading"],
                            capture_output=True, text=True, timeout=30,
                            env={"PATH": "/usr/bin:/bin"})
    assert result.returncode == 2
    assert "only reads" in result.stderr
    assert result.stdout == ""


def test_no_argument_prints_the_tools_it_can_run():
    result = subprocess.run(["bash", str(SCRIPT_PATH)],
                            capture_output=True, text=True, timeout=30,
                            env={"PATH": "/usr/bin:/bin"})
    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert "get_cycles" in result.stderr


def test_it_cannot_send_mail():
    """Without this the notifier warns "emergency email is inert" on every
    run. That is true of the shell and false of the deployment — the Lambdas
    get the addresses from template.yaml — so it reads as a broken alerting
    path that is not broken."""
    assert "export TA_DISABLE_EMAIL=1" in SCRIPT

    template = (ROOT / "template.yaml").read_text(encoding="utf-8")
    assert "TA_NOTIFY__TO_ADDRESS:" in template, (
        "if the Lambdas stop being given an address, the warning is real "
        "and this script should not be suppressing it")
