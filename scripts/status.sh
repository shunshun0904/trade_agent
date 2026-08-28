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
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

# Budget figures come from config/default.yaml rather than being repeated here.
# A monitoring script that disagrees with the system it monitors is worse than
# no monitoring script, and these values decide the Phase 2 go/no-go (spec 14).
config_number() {  # config_number <key> <fallback>
    local value
    value="$(awk -v key="$1:" '$1 == key {print $2; exit}' \
        "${ROOT}/config/default.yaml" 2>/dev/null)"
    if [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        printf '%s' "$value"
    else
        printf '%s' "$2"
    fi
}

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
# table that had one.
#
# The values are read with --query, not grepped out of the raw response.
# get-item returns the low-level form, where every value is wrapped in a type
# descriptor — {"equity_jpy":{"N":"10000"}} — so a grep for the field returns
# the descriptor instead of the number, and printed a bare `"equity_jpy": {`.
# --query unwraps it. The paths track models/state.py (SystemState.equity_jpy,
# MonthlyCounters.llm_cost_jpy, DailyCounters.full_debates); tests/
# test_status_script.py holds the two together.
STATE_ROW="$(aws dynamodb get-item --table-name "trade-agent-${ENVIRONMENT}-state" \
    --key '{"pk":{"S":"state"}}' --region "$REGION" --output text \
    --query 'Item.[equity_jpy.N, monthly.M.llm_cost_jpy.N, daily.M.full_debates.N, kill_switch.BOOL, daily.M.llm_cost_jpy.N]' \
    2>/dev/null || true)"

if [[ -n "$STATE_ROW" && "$STATE_ROW" != "None" ]]; then
    IFS=$'\t' read -r EQUITY LLM_MONTH DEBATES_TODAY KILLED LLM_TODAY <<< "$STATE_ROW"
    ok "system state row exists — a tick has completed at least once"

    [[ "$EQUITY" == "None" || -z "$EQUITY" ]] && EQUITY=0
    [[ "$LLM_MONTH" == "None" || -z "$LLM_MONTH" ]] && LLM_MONTH=0
    [[ "$DEBATES_TODAY" == "None" || -z "$DEBATES_TODAY" ]] && DEBATES_TODAY=0
    [[ "$LLM_TODAY" == "None" || -z "$LLM_TODAY" ]] && LLM_TODAY=0

    # Days left in the JST month, counting today — the same denominator the
    # system paces against (timeutil.jst_days_remaining_in_month).
    JST_TODAY="$(date -u -d '+9 hours' +%Y-%m-%d)"
    DAYS_IN_MONTH="$(date -u -d "${JST_TODAY%-*}-01 +1 month -1 day" +%-d)"
    DAYS_LEFT=$(( DAYS_IN_MONTH - $(date -u -d "$JST_TODAY" +%-d) + 1 ))
    (( DAYS_LEFT < 1 )) && DAYS_LEFT=1

    # One awk for the arithmetic and the formatting, so the money is rendered
    # in exactly one place. Two traps are deliberately avoided here:
    #
    #   * every `?:` is parenthesised. In `printf "%.1f", budget > 0 ? a : b`
    #     awk reads `> 0` as output redirection and writes the number to a file
    #     called "0", leaving the field blank — which is how the percentage and
    #     the whole budget ladder silently did nothing.
    #   * thousands separators are built by hand rather than with printf's `'`
    #     flag, which is a no-op outside a grouping locale and produced bare
    #     "10000" in CloudShell.
    mapfile -t SUMMARY < <(awk \
        -v equity="$EQUITY" -v spent="$LLM_MONTH" \
        -v total="$(config_number total_budget_jpy 3000)" \
        -v infra="$(config_number infra_cost_jpy 100)" \
        -v today="$LLM_TODAY" -v days_left="$DAYS_LEFT" \
        -v debates="$DEBATES_TODAY" \
        -v mult="$(config_number daily_allowance_multiplier 2)" '
        function commas(n,   s, out, i, len, sign) {
            sign = (n < 0 ? "-" : ""); n = (n < 0 ? -n : n)
            s = sprintf("%.0f", n); len = length(s); out = ""
            for (i = 0; i < len; i++) {
                if (i > 0 && i % 3 == 0) out = "," out
                out = substr(s, len - i, 1) out
            }
            return sign out
        }
        BEGIN {
            budget = total - infra
            pct = (budget > 0 ? spent / budget * 100 : 0)
            # Two rungs. There is no "degraded" middle any more: spending is
            # paced daily rather than cut to a fixed debate count at 80%.
            ladder = (pct >= 100 ? "STOPPED — no LLM calls this month" : "normal")

            # What today may still spend: the remaining monthly budget spread
            # over the days left, times the slack multiplier. How many debates
            # that buys is an outcome, not a target; the count is information.
            allowance = (budget - spent) / days_left * mult
            if (allowance < 0) allowance = 0
            printf "  equity              %s JPY\n", commas(equity)
            printf "  LLM cost, month     %.2f / %s JPY  (%.1f%%)  %s\n", \
                   spent, commas(budget), pct, ladder
            printf "  today               %.2f / %.2f JPY  (%s debate(s), %s day(s) left)\n", \
                   today, allowance, debates, days_left
            printf "%s\n", (ladder == "normal" ? "ok" : "alert")
        }')

    note "${SUMMARY[0]}"
    # The number that decides whether the budget holds, and with it whether
    # Phase 2 is affordable (spec 14). Counting agent calls does not answer it.
    if [[ "${SUMMARY[3]}" == "ok" ]]; then note "${SUMMARY[1]}"; else warn "${SUMMARY[1]# }"; fi
    note "${SUMMARY[2]}"
    [[ "$KILLED" == "True" ]] && bad "  kill switch is ENGAGED"
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
