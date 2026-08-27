#!/usr/bin/env bash
#
# One-shot interactive deploy, built for AWS CloudShell.
#
#   curl -fsSL <raw url>/scripts/deploy.sh | bash      # or, having cloned:
#   bash scripts/deploy.sh
#
# It is safe to re-run: every step checks what already exists and asks before
# changing it. Secrets are read with a hidden prompt, written straight to SSM
# Parameter Store, and never echoed, logged, or written to disk.
#
# What it does, in order:
#   0. preflight    region, credentials, SAM CLI (installs it if missing)
#   1. source       clone or update the repository
#   2. secrets      four SSM SecureString parameters
#   3. email        SES identity verification for the alert addresses
#   4. deploy       sam build && sam deploy
#   5. verify       exchange constants, credentials, heartbeat
#
set -euo pipefail

REPO_URL="${TA_REPO_URL:-https://github.com/shunshun0904/trade_agent.git}"
BRANCH="${TA_BRANCH:-claude/trade-agent-spec-mtwldk}"
REGION="${TA_REGION:-ap-northeast-1}"
ENVIRONMENT="${TA_ENVIRONMENT:-prod}"
STACK_NAME="trade-agent-${ENVIRONMENT}"
WORKDIR="${TA_WORKDIR:-$HOME/trade_agent}"

BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'
YELLOW=$'\033[33m'; BLUE=$'\033[34m'; RESET=$'\033[0m'

step()  { printf '\n%s==> %s%s\n' "$BOLD$BLUE" "$*" "$RESET"; }
ok()    { printf '    %s✓%s %s\n' "$GREEN" "$RESET" "$*"; }
warn()  { printf '    %s!%s %s\n' "$YELLOW" "$RESET" "$*"; }
die()   { printf '\n    %s✗ %s%s\n\n' "$RED" "$*" "$RESET" >&2; exit 1; }
note()  { printf '    %s%s%s\n' "$DIM" "$*" "$RESET"; }

ask() {  # ask <prompt> <default> -> echoes the answer
    local prompt="$1" default="${2:-}" reply
    if [[ -n "$default" ]]; then
        read -r -p "    ${prompt} [${default}]: " reply </dev/tty
        echo "${reply:-$default}"
    else
        read -r -p "    ${prompt}: " reply </dev/tty
        echo "$reply"
    fi
}

ask_secret() {  # ask_secret <prompt> -> echoes the answer, never displayed
    local prompt="$1" reply
    read -r -s -p "    ${prompt}: " reply </dev/tty
    echo >/dev/tty
    echo "$reply"
}

confirm() {  # confirm <question> ; returns 0 for yes
    local reply
    read -r -p "    ${1} [y/N]: " reply </dev/tty
    [[ "$reply" =~ ^[Yy]$ ]]
}

# ------------------------------------------------------ CloudFormation state

stack_status() {  # echoes the stack's status, or nothing if it does not exist
    aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" --region "$REGION" \
        --query 'Stacks[0].StackStatus' --output text 2>/dev/null || true
}

# The reason a stack rolled back is one line in a long event list, and by the
# time `sam deploy` reports failure it has usually scrolled away. Every other
# resource reports "Resource creation cancelled", which is noise: the useful
# events are the ones with a real reason.
failure_reasons() {
    aws cloudformation describe-stack-events \
        --stack-name "$STACK_NAME" --region "$REGION" \
        --query "StackEvents[?ResourceStatus=='CREATE_FAILED'
                   || ResourceStatus=='UPDATE_FAILED'
                   || ResourceStatus=='DELETE_FAILED'].
                 [LogicalResourceId,ResourceType,ResourceStatusReason]" \
        --output text 2>/dev/null \
        | grep -v 'Resource creation cancelled' || true
}

show_failure_reasons() {
    local reasons
    reasons="$(failure_reasons)"
    if [[ -z "$reasons" ]]; then
        warn "CloudFormation reported no specific resource failure"
        return
    fi
    printf '\n    %sWhy CloudFormation rolled back:%s\n\n' "$BOLD" "$RESET"
    # Oldest last in the API response; the first failure is the real cause and
    # the rest are usually consequences, so show it at the top.
    printf '%s\n' "$reasons" | tac | while IFS=$'\t' read -r id type reason; do
        printf '    %s%s%s  (%s)\n' "$BOLD" "$id" "$RESET" "$type"
        printf '      %s\n\n' "$reason"
    done
}

# A stack whose *first* create failed is left as an empty ROLLBACK_COMPLETE
# shell. CloudFormation will not update it — it can only be deleted and
# recreated — so a retry fails with a confusing "cannot be updated" until it
# is cleared. Do that here rather than making the owner discover it.
clear_rolled_back_stack() {
    local status="$1"
    warn "the stack is in ${status} — a previous deploy failed"
    show_failure_reasons

    note "CloudFormation cannot update a stack in this state; it has to be"
    note "deleted first. The stack holds no data yet (the rollback already"
    note "removed every resource it created)."
    if ! confirm "delete the failed stack and retry?"; then
        die "leaving ${STACK_NAME} in ${status}. Fix the cause above, then re-run."
    fi

    aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$REGION" \
        || die "could not delete ${STACK_NAME}"
    note "waiting for the delete to finish"
    aws cloudformation wait stack-delete-complete \
        --stack-name "$STACK_NAME" --region "$REGION" 2>/dev/null \
        || die "delete did not complete. Check the stack in the CloudFormation console."
    ok "failed stack removed"
}

# CloudFormation refuses to create a resource that already exists outside the
# stack. For this template that means the Lambda log groups: Lambda creates
# them by itself the first time a function is invoked, and once a failed deploy
# has left one behind, every later deploy fails on it identically. They are
# orphans holding only the failed attempt's logs, so offer to clear them.
clear_orphaned_log_groups() {
    local groups
    groups="$(aws logs describe-log-groups \
        --log-group-name-prefix "/aws/lambda/trade-agent-${ENVIRONMENT}-" \
        --region "$REGION" --query 'logGroups[].logGroupName' --output text 2>/dev/null || true)"
    [[ -n "$groups" && "$groups" != "None" ]] || return 0

    warn "log groups left over from the failed attempt:"
    for group in $groups; do note "  $group"; done
    note "CloudFormation cannot create these while they exist, and it does not"
    note "own them, so the next deploy would fail on them too."
    if ! confirm "delete them? (they contain only the failed attempt's logs)"; then
        warn "keeping them; the deploy may fail with 'already exists'"
        return 0
    fi
    for group in $groups; do
        aws logs delete-log-group --log-group-name "$group" --region "$REGION" 2>/dev/null \
            || warn "could not delete $group"
    done
    ok "orphaned log groups removed"
}

# --------------------------------------------------------------- 0. preflight

step "0/5  Environment"

command -v aws >/dev/null || die "the AWS CLI is not on PATH. Run this in AWS CloudShell."

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)" \
    || die "no usable AWS credentials. In CloudShell this should already work."
CALLER="$(aws sts get-caller-identity --query Arn --output text)"
ok "account ${ACCOUNT_ID}"
note "identity: ${CALLER}"
ok "deploying into ${REGION} (stack: ${STACK_NAME})"

note "python $(python3 -V 2>&1 | awk '{print $2}') (any version works — the"
note "Lambda package is built for its own runtime, not this one)"

if ! command -v sam >/dev/null; then
    warn "SAM CLI not found; installing it into ~/.local (a few minutes)"
    python3 -m pip install --quiet --user --upgrade aws-sam-cli \
        || die "could not install the SAM CLI"
    export PATH="$HOME/.local/bin:$PATH"
    command -v sam >/dev/null || die "SAM CLI installed but not on PATH"
fi
ok "sam $(sam --version 2>&1 | awk '{print $NF}')"

# CloudShell's home is small and a stale build directory is the usual cause of
# a mid-deploy disk error.
AVAIL_MB="$(df -Pm "$HOME" | awk 'NR==2 {print $4}')"
if (( AVAIL_MB < 400 )); then
    warn "only ${AVAIL_MB}MB free in \$HOME; the build needs roughly 300MB"
    if confirm "Delete previous build artifacts and continue?"; then
        rm -rf "${WORKDIR}/.aws-sam"
    fi
fi

# ------------------------------------------------------------------ 1. source

step "1/5  Source"

if [[ -d "${WORKDIR}/.git" ]]; then
    git -C "$WORKDIR" fetch --quiet origin "$BRANCH"
    git -C "$WORKDIR" checkout --quiet "$BRANCH"
    git -C "$WORKDIR" reset --hard --quiet "origin/${BRANCH}"
    ok "updated ${WORKDIR} to origin/${BRANCH}"
else
    git clone --quiet --branch "$BRANCH" "$REPO_URL" "$WORKDIR" \
        || die "clone failed. For a private repo, set up git credentials first."
    ok "cloned into ${WORKDIR}"
fi
cd "$WORKDIR"
note "$(git log --oneline -1)"

# ----------------------------------------------------------------- 2. secrets

step "2/5  Secrets  (SSM Parameter Store, SecureString)"
note "Nothing typed here is echoed, logged, or written to disk."

put_secret() {  # put_secret <name> <prompt> [generated]
    local name="$1" prompt="$2" generated="${3:-}" value
    if aws ssm get-parameter --name "$name" --region "$REGION" >/dev/null 2>&1; then
        if ! confirm "${name} already exists. Replace it?"; then
            ok "${name} kept"
            return
        fi
    fi
    if [[ -n "$generated" ]]; then
        value="$generated"
    else
        value="$(ask_secret "$prompt")"
        [[ -n "$value" ]] || die "${name}: empty value"
    fi
    aws ssm put-parameter --name "$name" --value "$value" \
        --type SecureString --overwrite --region "$REGION" >/dev/null
    ok "${name} stored"
}

printf '\n'
warn "Use a bitbank API key with 参照 + 取引 only. NEVER grant 出金 (withdrawal)."
printf '\n'
put_secret /trade-agent/bitbank/api-key    "bitbank API key"
put_secret /trade-agent/bitbank/api-secret "bitbank API secret"
put_secret /trade-agent/anthropic/api-key  "Anthropic API key (sk-ant-...)"

MCP_TOKEN=""
if aws ssm get-parameter --name /trade-agent/mcp/bearer-token --region "$REGION" \
        >/dev/null 2>&1; then
    ok "/trade-agent/mcp/bearer-token kept"
else
    MCP_TOKEN="$(openssl rand -base64 32)"
    put_secret /trade-agent/mcp/bearer-token "" "$MCP_TOKEN"
fi

# ------------------------------------------------------------------- 3. email

step "3/5  Alert email  (SES)"
note "Emergency alerts are the only channel that reaches you when you are not"
note "looking at the chat, so the addresses have to be verified."

OWNER_EMAIL="$(ask 'Address to receive alerts')"
[[ "$OWNER_EMAIL" == *@* ]] || die "that does not look like an email address"
SENDER_EMAIL="$(ask 'Address to send from' "$OWNER_EMAIL")"

verify_identity() {
    local address="$1" status
    status="$(aws ses get-identity-verification-attributes \
        --identities "$address" --region "$REGION" \
        --query "VerificationAttributes.\"${address}\".VerificationStatus" \
        --output text 2>/dev/null || echo "None")"
    if [[ "$status" == "Success" ]]; then
        ok "${address} already verified"
        return 0
    fi
    aws ses verify-email-identity --email-address "$address" --region "$REGION"
    warn "verification email sent to ${address} — click the link in it"
    return 1
}

PENDING=0
verify_identity "$OWNER_EMAIL" || PENDING=1
if [[ "$SENDER_EMAIL" != "$OWNER_EMAIL" ]]; then
    verify_identity "$SENDER_EMAIL" || PENDING=1
fi

if (( PENDING )); then
    printf '\n'
    note "Waiting for verification. The deploy will proceed either way — an"
    note "unverified address only means alerts stay silent until you click."
    for _ in $(seq 1 30); do
        sleep 10
        status="$(aws ses get-identity-verification-attributes \
            --identities "$SENDER_EMAIL" --region "$REGION" \
            --query "VerificationAttributes.\"${SENDER_EMAIL}\".VerificationStatus" \
            --output text 2>/dev/null || echo None)"
        if [[ "$status" == "Success" ]]; then
            ok "verified"
            PENDING=0
            break
        fi
        printf '.'
    done
    printf '\n'
    if (( PENDING )); then
        warn "still unverified; you can click the link later"
    fi
fi

# ------------------------------------------------------------------ 4. deploy

step "4/5  Deploy"
printf '\n'
note "Phase 1 (paper trading) is the only supported first deploy: the executor"
note "physically cannot reach bitbank's order API while it is on."
printf '\n'

# Clear the wreckage of a previous failed deploy before building, so the build
# time is not spent on a deploy that cannot start.
STACK_STATE="$(stack_status)"
case "$STACK_STATE" in
    ROLLBACK_COMPLETE|ROLLBACK_FAILED|CREATE_FAILED)
        clear_rolled_back_stack "$STACK_STATE"
        clear_orphaned_log_groups
        ;;
    UPDATE_ROLLBACK_FAILED)
        show_failure_reasons
        die "the stack needs manual recovery (continue-update-rollback) in the CloudFormation console."
        ;;
    *_IN_PROGRESS)
        die "another deploy of ${STACK_NAME} is in progress (${STACK_STATE}). Wait for it to finish."
        ;;
    "")
        note "no existing stack; this is a first deploy"
        ;;
    *)
        ok "existing stack is ${STACK_STATE}; applying the difference"
        ;;
esac

sam build || die "sam build failed"
ok "build complete"

# Check the packages before uploading them. A wrong ABI, a missing config or a
# missing handler all deploy cleanly and then crash every function on import,
# minutes later, with CloudFormation showing success.
CHECKED=0
for fn in TickFunction ScreenFunction DecideFunction ReflectFunction McpFunction; do
    BUILT="${WORKDIR}/.aws-sam/build/${fn}"
    # SAM builds functions that share a build definition once and points the
    # rest at the same artifact, so some of these directories may not exist.
    if [[ -d "$BUILT" ]]; then
        printf '  %-16s ' "$fn"
        python3 "${WORKDIR}/scripts/verify_artifact.py" "$BUILT" \
            || die "refusing to deploy a package that will not start"
        CHECKED=$(( CHECKED + 1 ))
    fi
done
if (( CHECKED == 0 )); then
    die "sam build produced no function artifacts under .aws-sam/build"
fi

sam deploy \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --capabilities CAPABILITY_IAM \
    --resolve-s3 \
    --no-confirm-changeset \
    --no-fail-on-empty-changeset \
    --parameter-overrides \
        "Environment=${ENVIRONMENT}" \
        "OwnerEmail=${OWNER_EMAIL}" \
        "SenderEmail=${SENDER_EMAIL}" \
        "PaperTrading=true" \
        "Phase=1" \
    || {
        show_failure_reasons
        note "Nothing was left half-built: CloudFormation rolled the stack back."
        note "Fix the cause above and re-run this script — it will clear the"
        note "failed stack for you."
        die "sam deploy failed"
    }
ok "stack deployed"

MCP_URL="$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" \
    --region "$REGION" --query \
    "Stacks[0].Outputs[?OutputKey=='McpEndpoint'].OutputValue" --output text)"

# ------------------------------------------------------------------ 5. verify

step "5/5  Verify"

python3 -m pip install --quiet --user -r requirements.txt 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"

set +e
PYTHONPATH=src TA_ENV="$ENVIRONMENT" AWS_REGION="$REGION" \
    python3 -m trade_agent.cli preflight
PREFLIGHT=$?
set -e

printf '\n'
note "The first 5-minute tick can take a few minutes to fire. Until it does,"
note "the heartbeat alarm sits in INSUFFICIENT_DATA, which is expected."

# ------------------------------------------------------------------ summary

printf '\n%s%s%s\n' "$BOLD" "$(printf '=%.0s' {1..62})" "$RESET"
printf '%sDeployment complete%s\n' "$BOLD$GREEN" "$RESET"
printf '%s%s%s\n\n' "$BOLD" "$(printf '=%.0s' {1..62})" "$RESET"

printf '  Stack        %s (%s)\n' "$STACK_NAME" "$REGION"
printf '  Mode         paper trading — no live order can be placed\n'
printf '  MCP endpoint %s\n' "${MCP_URL:-<not found>}"
if [[ -n "$MCP_TOKEN" ]]; then
    printf '  Bearer token %s\n' "$MCP_TOKEN"
    printf '               %sshown once; it is stored in SSM at%s\n' "$DIM" "$RESET"
    printf '               %s/trade-agent/mcp/bearer-token%s\n' "$DIM" "$RESET"
else
    printf '  Bearer token %skept from the existing SSM parameter%s\n' "$DIM" "$RESET"
    printf '               aws ssm get-parameter --with-decryption \\\n'
    printf '                 --name /trade-agent/mcp/bearer-token --region %s\n' "$REGION"
fi

printf '\n  %sNext%s\n' "$BOLD" "$RESET"
printf '   1. Register the MCP endpoint in claude.ai as a custom connector,\n'
printf '      using the bearer token above.\n'
printf '   2. Check the heartbeat alarm in ~15 minutes:\n'
printf '      aws cloudwatch describe-alarms --alarm-names %s-tick-heartbeat \\\n' "$STACK_NAME"
printf '        --region %s --query "MetricAlarms[0].StateValue" --output text\n' "$REGION"
printf '   3. Let it paper-trade for a month, then review docs/DEPLOY.md\n'
printf '      before considering Phase 2.\n'

if (( PREFLIGHT != 0 )); then
    printf '\n  %s! preflight reported problems — see section 5 above.%s\n' "$YELLOW" "$RESET"
    printf '    The stack is deployed but will not trade correctly until they\n'
    printf '    are fixed. Re-run: PYTHONPATH=src python3 -m trade_agent.cli preflight\n'
fi
printf '\n'
