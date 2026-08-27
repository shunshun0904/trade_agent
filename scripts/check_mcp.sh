#!/usr/bin/env bash
#
# Verify the MCP endpoint and print the URL to register in claude.ai.
#
#   bash scripts/check_mcp.sh
#
# This exists because the equivalent curl is a quoting trap. The JSON body has
# to survive the shell intact, and a shell that does not honour single quotes —
# Windows cmd, and PowerShell by its own rules — hands the body to curl as a
# second URL instead. curl then reads `"jsonrpc":"2.0"` as host-and-port and
# says:
#
#   curl: (3) URL rejected: Port number was not a decimal number between 0 and 65535
#
# which points at a port, and has nothing to do with ports. Here the body is
# written to a file and passed as `-d @file`, so no quoting survives to be got
# wrong. Run it in AWS CloudShell, where the AWS credentials already work.
#
set -euo pipefail

REGION="${TA_REGION:-ap-northeast-1}"
ENVIRONMENT="${TA_ENVIRONMENT:-prod}"
STACK_NAME="trade-agent-${ENVIRONMENT}"

BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'
YELLOW=$'\033[33m'; RESET=$'\033[0m'
ok()   { printf '    %s✓%s %s\n' "$GREEN" "$RESET" "$*"; }
warn() { printf '    %s!%s %s\n' "$YELLOW" "$RESET" "$*"; }
die()  { printf '\n    %s✗ %s%s\n\n' "$RED" "$*" "$RESET" >&2; exit 1; }
note() { printf '    %s%s%s\n' "$DIM" "$*" "$RESET"; }

command -v aws >/dev/null || die "the AWS CLI is not on PATH. Run this in AWS CloudShell."

BASE_URL="$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='McpEndpoint'].OutputValue" \
    --output text 2>/dev/null || true)"
[[ -n "$BASE_URL" && "$BASE_URL" != "None" ]] \
    || die "no McpEndpoint output on ${STACK_NAME}. Has it been deployed?"

TOKEN="$(aws ssm get-parameter --with-decryption \
    --name /trade-agent/mcp/bearer-token --region "$REGION" \
    --query Parameter.Value --output text 2>/dev/null || true)"
[[ -n "$TOKEN" && "$TOKEN" != "None" ]] \
    || die "no token at /trade-agent/mcp/bearer-token"

BODY="$(mktemp)"
trap 'rm -f "$BODY"' EXIT
printf '%s' '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' > "$BODY"

CONNECTOR_URL="${BASE_URL%/}/mcp/${TOKEN}"

probe() {  # probe <label> <url> [header...]
    local label="$1" url="$2"; shift 2
    local body status
    body="$(curl -sS -o - -w '\n%{http_code}' -X POST "$url" \
        -H 'Content-Type: application/json' "$@" -d @"$BODY" 2>&1)" || {
        warn "${label}: curl failed — ${body}"
        return 1
    }
    status="$(printf '%s' "$body" | tail -n1)"
    body="$(printf '%s' "$body" | sed '$d')"

    if [[ "$status" == "200" ]] && printf '%s' "$body" | grep -q '"tools"'; then
        local count
        count="$(printf '%s' "$body" | grep -o '"name"' | wc -l | tr -d ' ')"
        ok "${label}: HTTP 200, ${count} tool(s)"
        return 0
    fi
    warn "${label}: HTTP ${status}"
    note "  ${body:0:200}"
    return 1
}

printf '\n%sMCP endpoint check%s  (%s)\n\n' "$BOLD" "$RESET" "$STACK_NAME"

# The path form is the one claude.ai will use, so it is the one that matters.
PATH_OK=0
probe "token in the path  (claude.ai)" "$CONNECTOR_URL" && PATH_OK=1

probe "Authorization header (curl / Claude Code)" "${BASE_URL%/}/" \
    -H "Authorization: Bearer ${TOKEN}" || true

# A request with no credential must be refused; a 200 here would mean the
# endpoint is open to anyone who finds the URL.
UNAUTH="$(curl -sS -o /dev/null -w '%{http_code}' -X POST "${BASE_URL%/}/" \
    -H 'Content-Type: application/json' -d @"$BODY" 2>/dev/null || echo 000)"
if [[ "$UNAUTH" == "401" ]]; then
    ok "no token: HTTP 401 (correctly refused)"
else
    warn "no token: HTTP ${UNAUTH} — expected 401. Investigate before registering."
fi

printf '\n'
if (( PATH_OK )); then
    printf '  %sRegister this in claude.ai%s  (Authentication: %sNone%s)\n\n' \
        "$BOLD" "$RESET" "$BOLD" "$RESET"
    printf '    %s\n\n' "$CONNECTOR_URL"
    printf '  %sThis whole URL is the credential — it carries the token.%s\n\n' \
        "$DIM" "$RESET"
else
    die "the endpoint did not answer on the path form; do not register it yet"
fi
