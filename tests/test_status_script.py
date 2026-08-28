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


# -- section 4: the state row -------------------------------------------------
#
# Two defects lived here. The row was read with a grep over the raw get-item
# response, which is the low-level typed form — {"equity_jpy":{"N":"10000"}} —
# so the grep returned the type descriptor and printed a bare `"equity_jpy": {`.
# And the arithmetic that replaced it was written as
#
#     printf "%.1f", budget > 0 ? spent / budget * 100 : 0
#
# where awk reads `> 0` as output redirection, writes the number to a file
# named "0", and leaves the percentage blank — which silently disabled the
# budget ladder as well.

def _summary_awk() -> str:
    """The awk program that formats section 4, lifted out of the script."""
    # Anchored on the mapfile that runs it, not on whichever -v happens to be
    # last — that anchor broke the moment the variables changed.
    match = re.search(r"mapfile -t SUMMARY < <\(awk.*?'(.*?)'\)", SCRIPT, re.S)
    assert match, "could not find the section 4 awk program in status.sh"
    return match.group(1)


def _run_summary(*, equity="10000", spent="0", total="3000", infra="100",
                 debates="0", today="0", days_left="15",
                 mult="2") -> list[str]:
    result = subprocess.run(
        ["awk", "-v", f"equity={equity}", "-v", f"spent={spent}",
         "-v", f"total={total}", "-v", f"infra={infra}",
         "-v", f"debates={debates}", "-v", f"today={today}",
         "-v", f"days_left={days_left}", "-v", f"mult={mult}",
         _summary_awk()],
        capture_output=True, text=True, check=True, cwd="/tmp")
    return result.stdout.splitlines()


def test_the_state_row_is_read_with_query_not_grep():
    """--query unwraps DynamoDB's type descriptors; grep returns them."""
    section = SCRIPT[SCRIPT.index("STATE_ROW="):SCRIPT.index("TRADES=")]
    assert "--query" in section
    assert "grep" not in section, (
        "the state row must be read with --query — grepping the raw response "
        "returns the type descriptor, not the value")


def test_the_queried_fields_exist_on_the_state_model():
    """The --query paths name real fields, so a rename cannot leave the script
    quietly reporting None."""
    from trade_agent.models.state import DailyCounters, MonthlyCounters, SystemState

    query = re.search(r"--query 'Item\.\[(.*?)\]'", SCRIPT, re.S).group(1)
    paths = [p.strip() for p in query.replace("\n", " ").split(",")]

    owners = {"": SystemState, "monthly": MonthlyCounters, "daily": DailyCounters}
    for path in paths:
        parts = path.split(".")
        if len(parts) == 2:              # equity_jpy.N
            prefix, field = "", parts[0]
        else:                            # monthly.M.llm_cost_jpy.N
            prefix, field = parts[0], parts[2]
        assert field in owners[prefix].model_fields, (
            f"status.sh queries {path!r}, but {owners[prefix].__name__} has no "
            f"field {field!r}")


def test_the_percentage_is_actually_computed():
    """Blank here is the awk redirection bug: the ladder never fires."""
    lines = _run_summary(spent="12.3456")
    assert "(0.4%)" in lines[1], lines


def test_the_budget_ladder_has_two_rungs():
    """The 80% "degraded" rung is gone — it cut the day to one debate, a count
    standing in for money. Pacing does that job continuously now."""
    assert "normal" in _run_summary(spent="100")[1]
    assert "normal" in _run_summary(spent="2400")[1]        # 82.8%, still trading
    assert "STOPPED" in _run_summary(spent="2950")[1]       # 101.7%
    assert _run_summary(spent="100")[3] == "ok"
    assert _run_summary(spent="2950")[3] == "alert"


def test_todays_allowance_is_the_remainder_spread_over_the_days_left():
    """(2900 - 900) / 10 * 2 = 400."""
    line = _run_summary(spent="900", today="12.5", days_left="10")[2]
    assert "12.50 / 400.00 JPY" in line, line


def test_an_exhausted_budget_leaves_no_allowance():
    assert "/ 0.00 JPY" in _run_summary(spent="2900", days_left="10")[2]


def test_a_large_debate_count_is_reported_but_does_not_cap_anything():
    """The change this was made for: a busy day is fine if it is paid for."""
    line = _run_summary(spent="100", today="5", debates="40", days_left="10")[2]
    assert "40 debate(s)" in line
    assert _run_summary(spent="100", today="5", debates="40")[3] == "ok"


def test_money_is_grouped_without_relying_on_a_locale():
    """printf's `'` flag is a no-op outside a grouping locale and printed a
    bare 10000 in CloudShell."""
    assert "1,234,567 JPY" in _run_summary(equity="1234567")[0]
    assert "10,000 JPY" in _run_summary(equity="10000")[0]


def test_the_budget_comes_from_config_not_a_repeated_constant():
    """A monitoring script that disagrees with the system it monitors is worse
    than none, so the budget is derived from the same file the code reads."""
    import yaml

    config = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    budget = config["cost"]["total_budget_jpy"] - config["cost"]["infra_cost_jpy"]
    assert f"{budget:,} JPY" in _run_summary()[1]
    assert "config_number total_budget_jpy" in SCRIPT
    assert "config_number daily_allowance_multiplier" in SCRIPT
