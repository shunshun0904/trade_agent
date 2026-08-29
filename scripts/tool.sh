#!/usr/bin/env bash
#
# Run one read-only MCP tool against the deployed system, from a shell.
#
#   bash scripts/tool.sh get_cycles
#   bash scripts/tool.sh get_trades '{"limit": 5}'
#   bash scripts/tool.sh get_daily_report '{"date": "2026-08-28"}'
#
# The same read-only tools the claude.ai connector serves, without the connector.
# Useful while the connector is not registered, and for anything easier to read
# as JSON than as prose.
#
# The CLI on its own is not enough here. Table names and the bucket reach the
# Lambdas as environment variables from template.yaml; a CloudShell session has
# none of them, so `trade-agent mcp` would read `trade-agent-agent-calls`
# instead of `trade-agent-prod-agent-calls` and report an empty system. This
# reads both from the deployed stack and passes them through.
#
# Read-only: the two writing tools (pause_trading, resume_trading) are refused
# here on purpose. Pausing live trading from a half-typed shell command is not
# something a convenience script should make easy — use the connector, which
# asks for confirm=true.
#
set -uo pipefail

REGION="${TA_REGION:-ap-northeast-1}"
ENVIRONMENT="${TA_ENVIRONMENT:-prod}"
STACK_NAME="trade-agent-${ENVIRONMENT}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TOOL="${1:-}"
ARGS="${2:-}"
[[ -z "$ARGS" ]] && ARGS='{}'

WRITERS=" pause_trading resume_trading "
READERS="get_status get_trades get_daily_report get_lessons get_agent_log get_cycles"

if [[ -z "$TOOL" ]]; then
    echo "usage: bash scripts/tool.sh <tool> [json-args]" >&2
    echo "tools: ${READERS}" >&2
    exit 2
fi
if [[ "$WRITERS" == *" $TOOL "* ]]; then
    echo "$TOOL changes the system's state; this script only reads." >&2
    echo "Use the claude.ai connector, which requires confirm=true." >&2
    exit 2
fi

command -v aws >/dev/null || { echo "run this in AWS CloudShell" >&2; exit 2; }

BUCKET="$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" \
    --region "$REGION" --output text \
    --query "Stacks[0].Outputs[?OutputKey=='DataBucketName'].OutputValue" \
    2>/dev/null)"
if [[ -z "$BUCKET" || "$BUCKET" == "None" ]]; then
    echo "could not read stack ${STACK_NAME} in ${REGION}." >&2
    echo "Is it deployed, and is TA_ENVIRONMENT right?" >&2
    exit 1
fi

# The store reads AWS_REGION, which Lambda sets for itself and a shell
# does not. AWS_DEFAULT_REGION covers the aws CLI and boto3's own default.
# A read-only script must not be able to send mail, and without this the
# notifier warns "SES addresses are not configured; emergency email is inert"
# on every run — which is a statement about this shell, not about the
# deployment. The Lambdas get the addresses from template.yaml and do alert.
export TA_DISABLE_EMAIL=1

export AWS_REGION="$REGION"
export AWS_DEFAULT_REGION="$REGION"
export TA_STORAGE__TABLE_PREFIX="trade-agent-${ENVIRONMENT}"
export TA_STORAGE__S3_BUCKET="$BUCKET"
export PYTHONPATH="${ROOT}/src"

exec python3 -m trade_agent.cli mcp "$TOOL" --args "$ARGS"
