#!/usr/bin/env bash
#
# Is it actually running?
#
#   bash scripts/status.sh
#
# "The stack deployed" and "the system is trading" are different claims. A
# scheduled Lambda that throws on every invocation looks perfectly healthy in
# CloudFormation: the stack is CREATE_COMPLETE, the schedules exist, and
# nothing turns red except a metric nobody is looking at. This answers the
# second question from evidence rather than from the first.
#
# Read-only. It changes nothing.
#
set -uo pipefail

REGION="${TA_REGION:-ap-northeast-1}"
ENVIRONMENT="${TA_ENVIRONMENT:-prod}"
STACK_NAME="trade-agent-${ENVIRONMENT}"
WINDOW_MIN="${TA_WINDOW_MIN:-60}"

BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'
YELLOW=$'\033[33m'; RESET=$'\033[0m'
ok()   { printf '    %s✓%s %s\n' "$GREEN" "$RESET" "$*"; }
bad()  { printf '    %s✗%s %s\n' "$RED" "$RESET" "$*"; }
warn() { printf '    %s!%s %s\n' "$YELLOW" "$RESET" "$*"; }
note() { printf '    %s%s%s\n' "$DIM" "$*" "$RESET"; }
head_() { printf '\n%s%s%s\n' "$BOLD" "$*" "$RESET"; }

command -v aws >/dev/null || { echo "run this in AWS CloudShell"; exit 2; }

END="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START="$(date -u -d "-${WINDOW_MIN} min" +%Y-%m-%dT%H:%M:%SZ)"
VERDICT_TRADING=0   # a tick ran and did not error
VERDICT_BROKEN=0    # something invoked and threw
VERDICT_PENDING=0   # too soon after a deploy to have data
SINCE_DEPLOY=0

metric_sum() {  # metric_sum <metric> <function>
    local value
    value="$(aws cloudwatch get-metric-statistics --namespace AWS/Lambda \
        --metric-name "$1" --dimensions "Name=FunctionName,Value=$2" \
        --start-time "$START" --end-time "$END" --period $(( WINDOW_MIN * 60 )) \
        --statistics Sum --region "$REGION" \
        --query 'Datapoints[0].Sum' --output text 2>/dev/null)"
    [[ -z "$value" || "$value" == "None" ]] && value=0
    printf '%.0f' "$value"
}

# ---------------------------------------------------------------- the stack

head_ "1. Stack"

STATUS="$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" \
    --region "$REGION" --query 'Stacks[0].StackStatus' --output text 2>/dev/null)"
if [[ -z "$STATUS" || "$STATUS" == "None" ]]; then
    bad "${STACK_NAME} does not exist in ${REGION}. Nothing is deployed."
    exit 1
fi
case "$STATUS" in
    CREATE_COMPLETE|UPDATE_COMPLETE|UPDATE_ROLLBACK_COMPLETE) ok "$STATUS" ;;
    *) bad "$STATUS — the stack is not in a good state" ;;
esac

PARAMS="$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" \
    --region "$REGION" --query 'Stacks[0].Parameters' --output text 2>/dev/null)"
PAPER="$(printf '%s\n' "$PARAMS" | awk '$1=="PaperTrading"{print $2}')"
PHASE="$(printf '%s\n' "$PARAMS" | awk '$1=="Phase"{print $2}')"
if [[ "$PAPER" == "true" ]]; then
    ok "paper trading — no real order can be placed (Phase ${PHASE:-?})"
else
    warn "LIVE TRADING (PaperTrading=false, Phase ${PHASE:-?}) — real money"
fi

LAST_DEPLOY="$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" \
    --region "$REGION" --query 'Stacks[0].LastUpdatedTime' --output text 2>/dev/null)"
note "last deployed: ${LAST_DEPLOY:-unknown}"

# Metrics from code that is no longer deployed are not evidence about the code
# that is. Without this, a deploy that *fixes* a crash reads as broken for the
# next hour: the window still holds every failure from before it.
if [[ -n "$LAST_DEPLOY" && "$LAST_DEPLOY" != "None" ]]; then
    DEPLOY_EPOCH="$(date -u -d "$LAST_DEPLOY" +%s 2>/dev/null || echo 0)"
    WINDOW_EPOCH="$(date -u -d "-${WINDOW_MIN} min" +%s)"
    if (( DEPLOY_EPOCH > WINDOW_EPOCH )); then
        START="$(date -u -d "@${DEPLOY_EPOCH}" +%Y-%m-%dT%H:%M:%SZ)"
        SINCE_DEPLOY=1
        MINUTES_LIVE=$(( ($(date -u +%s) - DEPLOY_EPOCH) / 60 ))
        note "measuring from the deploy, not the last ${WINDOW_MIN} min"
        note "(${MINUTES_LIVE} min of data; earlier failures were other code)"
    fi
fi

# ------------------------------------------------------------- the functions

if (( SINCE_DEPLOY )); then
    head_ "2. Functions  (since the deploy)"
else
    head_ "2. Functions  (last ${WINDOW_MIN} min)"
fi

for fn in tick screen decide reflect mcp; do
    name="trade-agent-${ENVIRONMENT}-${fn}"
    inv="$(metric_sum Invocations "$name")"
    err="$(metric_sum Errors "$name")"

    if (( inv == 0 )); then
        if [[ "$fn" == "tick" && ${MINUTES_LIVE:-999} -lt 6 ]]; then
            warn "$(printf '%-8s' "$fn") no data yet — the 5-minute tick has not come round"
            VERDICT_PENDING=1
        elif [[ "$fn" == "tick" ]]; then
            bad "$(printf '%-8s' "$fn") not invoked at all — it should run every 5 min"
        else
            note "$(printf '%-8s' "$fn") not invoked (may be normal for this window)"
        fi
    elif (( err > 0 && err >= inv )); then
        bad "$(printf '%-8s' "$fn") ${inv} invocation(s), ALL ${err} failed"
        VERDICT_BROKEN=1
    elif (( err > 0 )); then
        warn "$(printf '%-8s' "$fn") ${inv} invocation(s), ${err} failed"
        [[ "$fn" == "tick" ]] && VERDICT_TRADING=1
    else
        ok "$(printf '%-8s' "$fn") ${inv} invocation(s), no errors"
        [[ "$fn" == "tick" ]] && VERDICT_TRADING=1
    fi
done

# ------------------------------------------------------------------- alarms

head_ "3. Dead-man's switch"

for alarm in tick-heartbeat tick-errors decide-errors; do
    state="$(aws cloudwatch describe-alarms \
        --alarm-names "trade-agent-${ENVIRONMENT}-${alarm}" --region "$REGION" \
        --query 'MetricAlarms[0].StateValue' --output text 2>/dev/null)"
    case "$state" in
        OK)                ok   "$(printf '%-16s' "$alarm") OK" ;;
        ALARM)             bad  "$(printf '%-16s' "$alarm") ALARM" ;;
        INSUFFICIENT_DATA) warn "$(printf '%-16s' "$alarm") INSUFFICIENT_DATA (normal for ~15 min after a deploy)" ;;
        *)                 warn "$(printf '%-16s' "$alarm") ${state:-not found}" ;;
    esac
done

# --------------------------------------------------------------------- work

head_ "4. Has it actually done anything?"

# "state", matching STATE_KEY in storage/dynamo.py. It was "system" here, which
# is nothing, so this reported "no tick has ever finished its work" against a
# table that had one. tests/test_status_script.py holds the two together now.
STATE="$(aws dynamodb get-item --table-name "trade-agent-${ENVIRONMENT}-state" \
    --key '{"pk":{"S":"state"}}' --region "$REGION" \
    --query 'Item' --output json 2>/dev/null)"
if [[ -n "$STATE" && "$STATE" != "null" ]]; then
    equity="$(printf '%s' "$STATE" | grep -o '"equity_jpy":[^,}]*' | head -1)"
    ok "system state row exists — a tick has completed at least once"
    [[ -n "$equity" ]] && note "  ${equity}"
else
    bad "no system state row — no tick has ever finished its work"
fi

TRADES="$(aws dynamodb scan --table-name "trade-agent-${ENVIRONMENT}-trades" \
    --select COUNT --region "$REGION" --query 'Count' --output text 2>/dev/null)"
note "trades recorded: ${TRADES:-unknown}"

CALLS="$(aws dynamodb scan --table-name "trade-agent-${ENVIRONMENT}-agent-calls" \
    --select COUNT --region "$REGION" --query 'Count' --output text 2>/dev/null)"
note "LLM agent calls: ${CALLS:-unknown}"

# ------------------------------------------------------------------- errors

if (( VERDICT_BROKEN )) || ! (( VERDICT_TRADING )); then
    head_ "5. What the tick logged"
    since=$(( ($(date +%s) - WINDOW_MIN * 60) * 1000 ))
    events="$(aws logs filter-log-events \
        --log-group-name "/aws/lambda/trade-agent-${ENVIRONMENT}-tick" \
        --region "$REGION" --start-time "$since" \
        --query 'events[].message' --output text 2>/dev/null)"
    if [[ -z "$events" ]]; then
        note "nothing logged. The function is not being invoked at all —"
        note "check the EventBridge schedule, not the code."
    else
        printf '%s\n' "$events" | tr '\t' '\n' \
            | grep -v -E '^(START|END|REPORT|INIT_START|XRAY)' \
            | tail -n 30 | sed 's/^/      /'
    fi
fi

# ------------------------------------------------------------------ verdict

printf '\n%s%s%s\n' "$BOLD" "$(printf '=%.0s' {1..58})" "$RESET"
# The tick is the system. screen/decide/reflect failing is a real problem and
# is reported above, but it is not the difference between running and not.
if (( VERDICT_TRADING )); then
    printf '%s  RUNNING — the tick is firing and completing%s\n' "$BOLD$GREEN" "$RESET"
    if (( VERDICT_BROKEN )); then
        printf '%s  (but another function is failing — see section 2)%s\n' "$YELLOW" "$RESET"
    fi
elif (( ${VERDICT_PENDING:-0} )); then
    printf '%s  TOO EARLY TO SAY — wait for the next 5-minute tick%s\n' "$BOLD$YELLOW" "$RESET"
elif (( VERDICT_BROKEN )); then
    printf '%s  NOT RUNNING — invocations are failing%s\n' "$BOLD$RED" "$RESET"
else
    printf '%s  NOT RUNNING — the tick has not fired in this window%s\n' "$BOLD$RED" "$RESET"
fi
printf '%s%s%s\n\n' "$BOLD" "$(printf '=%.0s' {1..58})" "$RESET"
