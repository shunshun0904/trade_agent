"""Checks on scripts/stop.sh — the switch that stops the whole system.

Two things here are worth a test rather than a careful read. `update-schedule`
replaces a schedule wholesale, so a field dropped while flipping State is
silently reset — a timezone, a retry policy, or the target itself. And the
script has to refuse when a position is open, because disabling the schedules
stops the tick, and the tick is what evaluates the take profit.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "stop.sh"
SCRIPT = SCRIPT_PATH.read_text(encoding="utf-8")


def _code() -> str:
    """The lines that run, without the comments.

    The header explains which API not to use and why, so a naive substring
    search over the whole file finds the thing it is checking against.
    """
    return "\n".join(line for line in SCRIPT.splitlines()
                     if not line.lstrip().startswith("#"))


def _rebuild_source() -> str:
    """The Python the script writes out to rewrite a schedule definition."""
    lines = SCRIPT.splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.startswith('cat > "$REBUILD"'))
    end = next(i for i, line in enumerate(lines[start:], start)
               if line == "PYEOF")
    return "\n".join(lines[start + 1:end])


def _rebuild(schedule: dict, state: str, tmp_path: Path) -> dict:
    script = tmp_path / "rebuild.py"
    script.write_text(_rebuild_source(), encoding="utf-8")
    result = subprocess.run(
        ["python3", str(script), state], input=json.dumps(schedule),
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _schedule(**overrides) -> dict:
    base = {
        "Arn": "arn:aws:scheduler:ap-northeast-1:1234:schedule/default/s",
        "Name": "trade-agent-prod-TickFunctionFiveMinutes-AbC123",
        "GroupName": "default",
        "State": "ENABLED",
        "ScheduleExpression": "rate(5 minutes)",
        "ScheduleExpressionTimezone": "Asia/Tokyo",
        "FlexibleTimeWindow": {"Mode": "OFF"},
        "Target": {
            "Arn": "arn:aws:lambda:ap-northeast-1:1234:function:tick",
            "RoleArn": "arn:aws:iam::1234:role/x",
            "RetryPolicy": {"MaximumRetryAttempts": 0},
        },
        "CreationDate": "2026-08-24T00:00:00+00:00",
    }
    base.update(overrides)
    return base


def test_only_the_state_changes(tmp_path):
    """Everything else has to be carried through. A dropped
    ScheduleExpressionTimezone turns a 09:00 JST floor into 09:00 UTC on the
    way back up, and nothing would report that."""
    before = _schedule()
    after = _rebuild(before, "DISABLED", tmp_path)

    assert after["State"] == "DISABLED"
    for field in ("Name", "ScheduleExpression", "ScheduleExpressionTimezone",
                  "FlexibleTimeWindow", "Target", "GroupName"):
        assert after[field] == before[field], f"{field} was not carried through"


def test_the_retry_policy_survives(tmp_path):
    """MaximumRetryAttempts is 0 on purpose (template.yaml): a retried tick is
    a second one racing the first. The default is not 0."""
    after = _rebuild(_schedule(), "DISABLED", tmp_path)
    assert after["Target"]["RetryPolicy"]["MaximumRetryAttempts"] == 0


def test_read_only_fields_are_dropped(tmp_path):
    """update-schedule rejects Arn and CreationDate."""
    after = _rebuild(_schedule(), "DISABLED", tmp_path)
    assert "Arn" not in after
    assert "CreationDate" not in after


def test_a_description_with_an_apostrophe_survives(tmp_path):
    """The definition is API JSON and reaches the rebuild on stdin. An earlier
    version pasted it into the program's source, where this input would have
    ended the string literal."""
    after = _rebuild(_schedule(Description="owner's schedule; don't break it"),
                     "DISABLED", tmp_path)
    assert after["Description"] == "owner's schedule; don't break it"


def test_it_round_trips(tmp_path):
    """Disable then enable must return the definition it started from."""
    before = _schedule()
    disabled = _rebuild(before, "DISABLED", tmp_path)
    enabled = _rebuild({**before, **disabled}, "ENABLED", tmp_path)

    assert enabled["State"] == "ENABLED"
    assert enabled == {**disabled, "State": "ENABLED"}


# -- the guard rails ------------------------------------------------------

def test_it_reads_the_position_from_the_key_the_code_writes():
    from trade_agent.storage.dynamo import STATE_KEY

    keys = set(__import__("re").findall(r'"pk":\{"S":"([^"]+)"\}', SCRIPT))
    assert keys == {STATE_KEY}, (
        f"stop.sh looks up {keys}, but the code writes {STATE_KEY!r}")


def test_an_open_position_blocks_it_without_force():
    """The tick evaluates the take profit. Turning it off under an open
    position is a decision, not a default."""
    assert "refusing. Re-run with --force" in SCRIPT
    assert "$FORCE -ne 1" in SCRIPT


def test_it_asks_cloudformation_which_schedules_exist():
    """This was a live defect. The first version filtered `list-schedules` on
    names containing the stack name, and CloudFormation's generated physical
    names for AWS::Scheduler::Schedule do not carry it — so the script found
    nothing and reported a stopped system that was running happily. The stack
    is the authority on what it created; a naming convention is not."""
    code = _code()
    assert "aws cloudformation list-stack-resources" in code
    assert "AWS::Scheduler::Schedule" in code
    assert "contains(Name" not in code, (
        "back to matching schedule names, which is what failed")


def test_the_schedule_group_is_carried_through():
    """A schedule outside the default group is addressed as `group|name` in
    the physical id, and every scheduler call needs the two apart."""
    code = _code()
    # get-schedule and the state read take it as a flag; update-schedule gets
    # it inside --cli-input-json, which is why the rebuild keeps GroupName.
    assert code.count("--group-name") >= 2
    assert '"GroupName"' in _rebuild_source()
    assert "entry%%|*" in code and "entry##*|" in code


def test_it_uses_the_scheduler_api_not_the_events_api():
    """SAM's ScheduleV2 creates EventBridge Scheduler schedules. `aws events
    list-rules` finds nothing for them, which reads as "already stopped".

    Checked against the lines that run, not the whole file: the header explains
    the distinction and naming the wrong API there is the point.
    """
    code = _code()
    assert "aws scheduler get-schedule" in code
    assert "aws events " not in code


def test_it_verifies_the_state_it_asked_for():
    """An accepted update is not a stopped system; the summary may only claim
    what a re-read confirms."""
    assert 'confirmed="$(schedule_state "$name" "$group")"' in SCRIPT
    assert "update accepted but state reads" in SCRIPT


def test_finding_no_schedules_is_a_failure():
    """Silence here would look exactly like success."""
    assert "lists no AWS::Scheduler::Schedule" in SCRIPT


@pytest.mark.parametrize("flag", ["--nope", "-x"])
def test_an_unknown_flag_is_refused(flag):
    result = subprocess.run(["bash", str(SCRIPT_PATH), flag],
                            capture_output=True, text=True, timeout=30,
                            env={"PATH": "/usr/bin:/bin"})
    assert result.returncode == 2
    assert "unknown argument" in result.stderr
