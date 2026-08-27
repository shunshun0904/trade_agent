"""Checks on scripts/status.sh — the tool that answers "is it running?".

A monitoring script that is wrong is worse than none: it either hides a real
outage or, as happened here, reports a working system as broken. Both of the
things checked below were live defects.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "scripts" / "status.sh").read_text(encoding="utf-8")


def test_the_state_key_matches_the_one_the_code_writes():
    """It looked up pk="system"; the code writes pk="state". The script
    reported "no tick has ever finished its work" against a table that had
    exactly that row."""
    from trade_agent.storage.dynamo import STATE_KEY

    keys = re.findall(r'"pk":\{"S":"([^"]+)"\}', SCRIPT)
    assert keys, "no DynamoDB key literal found in status.sh"
    assert set(keys) == {STATE_KEY}, (
        f"status.sh looks up {set(keys)}, but the code writes {STATE_KEY!r}")


def test_the_measurement_window_starts_at_the_deploy():
    """Metrics from code that is no longer deployed say nothing about the code
    that is. Without this, a deploy that fixes a crash reads as broken for the
    next hour, because the window still holds the failures it fixed."""
    assert "LastUpdatedTime" in SCRIPT
    assert "DEPLOY_EPOCH > WINDOW_EPOCH" in SCRIPT, (
        "status.sh must clamp the metric window to the last deploy")


def test_a_failing_side_function_does_not_read_as_not_running():
    """The tick is the system. A broken mcp function is worth reporting and is
    not the difference between trading and not trading."""
    verdict = SCRIPT[SCRIPT.index("# The tick is the system."):]
    running = verdict.index("RUNNING — the tick is firing")
    not_running = verdict.index("NOT RUNNING — invocations are failing")
    assert running < not_running, (
        "VERDICT_TRADING must be tested before VERDICT_BROKEN, or a working "
        "tick alongside any other failing function reports NOT RUNNING")


def test_the_script_is_valid_bash():
    subprocess.run(["bash", "-n", str(ROOT / "scripts" / "status.sh")],
                   check=True, capture_output=True)


# Read-only AWS verbs. Anything else has to be added here deliberately, which
# is the point: this script is run to diagnose a system that may be
# mid-incident, and it must not be able to change what it is describing.
READ_ONLY_VERBS = {
    "describe-stacks", "describe-stack-events", "describe-alarms",
    "get-metric-statistics", "get-item", "scan", "query",
    "filter-log-events", "describe-log-groups", "get-parameter",
    "get-caller-identity",
}


def test_the_script_only_reads():
    """Checked against the aws sub-commands it actually runs, not against words
    appearing anywhere — "last deployed" and "invocation" are prose."""
    calls = set(re.findall(r"aws\s+([a-z0-9-]+)\s+([a-z0-9-]+)", SCRIPT))
    verbs = {verb for _service, verb in calls}
    mutating = sorted(verbs - READ_ONLY_VERBS)
    assert not mutating, (
        f"status.sh must only read, but calls: {mutating}")
    assert verbs, "no aws calls found — the regex is wrong"
