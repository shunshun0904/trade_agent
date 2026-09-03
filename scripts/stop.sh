#!/usr/bin/env bash
#
# Stop the system by disabling its schedules, and start it again.
#
#   bash scripts/stop.sh              # disable every schedule
#   bash scripts/stop.sh --resume     # enable them again
#   bash scripts/stop.sh --status     # just show what is enabled
#
# This is the heaviest of the stop options and the only one that reaches the
# tick. Read this before using it with a position open:
#
#   * The stop loss survives. It is an order resting on bitbank, so it fires
#     whether or not anything of ours is running.
#   * The take profit does NOT. It is evaluated locally by the 5-minute tick
#     (position_manager._maybe_exit), and that tick is exactly what this turns
#     off. A position left here can run through its target and back.
#   * Nothing reconciles fills, updates equity, or watches the kill switch
#     while the schedules are off.
#
# So this refuses to run while a position is open unless --force is given. To
# stop trading with a position on the books, use pause_trading instead: it
# blocks new entries and leaves the monitoring alone.
#
# Schedules are created by SAM's ScheduleV2 events, which are EventBridge
# Scheduler (`aws scheduler`) rather than EventBridge Rules (`aws events`) —
# `aws events list-rules` will show nothing and is not evidence of anything.
#
set -uo pipefail

REGION="${TA_REGION:-ap-northeast-1}"
ENVIRONMENT="${TA_ENVIRONMENT:-prod}"
STACK_NAME="trade-agent-${ENVIRONMENT}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'
YELLOW=$'\033[33m'; RESET=$'\033[0m'
ok()   { printf '    %s✓%s %s\n' "$GREEN" "$RESET" "$*"; }
bad()  { printf '    %s✗%s %s\n' "$RED" "$RESET" "$*"; }
warn() { printf '    %s!%s %s\n' "$YELLOW" "$RESET" "$*"; }
note() { printf '    %s%s%s\n' "$DIM" "$*" "$RESET"; }
head_() { printf '\n%s%s%s\n' "$BOLD" "$*" "$RESET"; }

ACTION="disable"
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --resume|--start)  ACTION="enable" ;;
        --status)          ACTION="status" ;;
        --force)           FORCE=1 ;;
        -h|--help)         sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

command -v aws >/dev/null || { echo "run this in AWS CloudShell" >&2; exit 2; }

# -- what are the schedules? ----------------------------------------------
#
# Named by CloudFormation from the stack and the function, so the stack name is
# the only reliable filter. A name match is checked rather than assumed: if the
# naming convention ever changes, this finds nothing and says so instead of
# silently leaving a running system behind.

mapfile -t SCHEDULES < <(aws scheduler list-schedules --region "$REGION" \
    --query "Schedules[?contains(Name, '${STACK_NAME}')].Name" \
    --output text 2>/dev/null | tr '\t' '\n' | grep -v '^$')

if [[ ${#SCHEDULES[@]} -eq 0 ]]; then
    head_ "Schedules"
    bad "no schedule found whose name contains '${STACK_NAME}' in ${REGION}"
    note "Is the stack deployed, and is TA_ENVIRONMENT right?"
    note "Check by hand: aws scheduler list-schedules --region ${REGION}"
    exit 1
fi

schedule_state() {  # schedule_state <name>
    aws scheduler get-schedule --name "$1" --region "$REGION" \
        --query 'State' --output text 2>/dev/null || echo "UNKNOWN"
}

head_ "Schedules (${STACK_NAME}, ${REGION})"
ENABLED_COUNT=0
for name in "${SCHEDULES[@]}"; do
    state="$(schedule_state "$name")"
    if [[ "$state" == "ENABLED" ]]; then
        ENABLED_COUNT=$(( ENABLED_COUNT + 1 ))
        ok "${name}  ${state}"
    else
        note "${name}  ${state}"
    fi
done

if [[ "$ACTION" == "status" ]]; then
    head_ "Verdict"
    if [[ $ENABLED_COUNT -gt 0 ]]; then
        ok "${ENABLED_COUNT} schedule(s) enabled — the system is running"
    else
        warn "every schedule is disabled — the system is stopped"
    fi
    exit 0
fi

# -- the position check ---------------------------------------------------
#
# Only matters when disabling. Turning the schedules back on with a position
# open is a return to normal, not a risk.

if [[ "$ACTION" == "disable" ]]; then
    head_ "Open position"
    POSITION="$(aws dynamodb get-item --table-name "${STACK_NAME}-state" \
        --key '{"pk":{"S":"state"}}' --region "$REGION" --output text \
        --query 'Item.open_position.M.trade_id.S' 2>/dev/null || true)"

    if [[ -z "$POSITION" || "$POSITION" == "None" ]]; then
        ok "no position is open; the tick has nothing to protect"
    else
        bad "a position is open: ${POSITION}"
        echo
        warn "Disabling the schedules stops the 5-minute tick, and the tick is"
        warn "what evaluates the take profit. The exchange-side stop loss keeps"
        warn "working, so the downside stays capped — but the position can run"
        warn "through its target and back with nothing to close it."
        echo
        note "To stop trading and keep this position monitored, use instead:"
        note "  pause_trading  (blocks new entries, leaves the tick running)"
        echo
        if [[ $FORCE -ne 1 ]]; then
            bad "refusing. Re-run with --force if you meant it."
            exit 1
        fi
        warn "--force given; continuing with a position open"
    fi
fi

# -- apply ----------------------------------------------------------------
#
# update-schedule replaces the whole schedule, so every field has to be read
# back and passed through. Omitting one silently resets it to a default — the
# target, the cron expression, the timezone. The rebuild happens in Python
# reading the definition on stdin: the definition is API JSON, and pasting it
# into a program's source is a quoting accident waiting for the first
# description with an apostrophe in it.

REBUILD="${ROOT}/scripts/.stop_rebuild.py"
cat > "$REBUILD" <<'PYEOF'
"""Copy a schedule definition, changing only State."""
import json
import sys

state = sys.argv[1]
schedule = json.load(sys.stdin)
out = {
    "Name": schedule["Name"],
    "ScheduleExpression": schedule["ScheduleExpression"],
    "Target": schedule["Target"],
    "FlexibleTimeWindow": schedule["FlexibleTimeWindow"],
    "State": state,
}
for key in ("GroupName", "ScheduleExpressionTimezone", "Description",
            "StartDate", "EndDate", "KmsKeyArn"):
    if schedule.get(key):
        out[key] = schedule[key]
json.dump(out, sys.stdout)
PYEOF
trap 'rm -f "$REBUILD"' EXIT

TARGET_STATE=$([[ "$ACTION" == "enable" ]] && echo ENABLED || echo DISABLED)

head_ "Setting every schedule to ${TARGET_STATE}"
FAILED=0
CHANGED=0
for name in "${SCHEDULES[@]}"; do
    current="$(schedule_state "$name")"
    if [[ "$current" == "$TARGET_STATE" ]]; then
        note "${name}: already ${TARGET_STATE}"
        continue
    fi

    existing="$(aws scheduler get-schedule --name "$name" --region "$REGION" \
        --output json 2>/dev/null)"
    if [[ -z "$existing" ]]; then
        bad "${name}: could not read the current definition"
        FAILED=$(( FAILED + 1 ))
        continue
    fi

    args="$(printf '%s' "$existing" | python3 "$REBUILD" "$TARGET_STATE" 2>/dev/null)"
    if [[ -z "$args" ]]; then
        bad "${name}: could not build the update from its definition"
        FAILED=$(( FAILED + 1 ))
        continue
    fi

    if aws scheduler update-schedule --region "$REGION" \
            --cli-input-json "$args" >/dev/null 2>&1; then
        confirmed="$(schedule_state "$name")"
        if [[ "$confirmed" == "$TARGET_STATE" ]]; then
            ok "${name}: ${current} -> ${confirmed}"
            CHANGED=$(( CHANGED + 1 ))
        else
            # Accepted but not in the state we asked for: report it rather than
            # counting a success, so the summary cannot claim a stopped system
            # that is still running.
            bad "${name}: update accepted but state reads ${confirmed}"
            FAILED=$(( FAILED + 1 ))
        fi
    else
        bad "${name}: update-schedule failed"
        FAILED=$(( FAILED + 1 ))
    fi
done

head_ "Result"
if [[ $FAILED -gt 0 ]]; then
    bad "${FAILED} schedule(s) did not change — the system is in a mixed state"
    note "Re-run this script, or set the rest by hand in the console."
    exit 1
fi

if [[ "$ACTION" == "disable" ]]; then
    ok "${CHANGED} schedule(s) disabled. Nothing is scheduled to run."
    echo
    note "Still true while stopped:"
    note "  - the exchange-side stop loss stays live on bitbank"
    note "  - DynamoDB, S3 and the MCP endpoint are untouched and still readable"
    note "  - no LLM call can happen, so the budget stops moving"
    echo
    note "Start again with: bash scripts/stop.sh --resume"
else
    ok "${CHANGED} schedule(s) enabled. The system is running again."
    note "The first tick lands within 5 minutes; the first screen within 30."
fi
