#!/usr/bin/env bash
# TPO・価格帯別出来高ダッシュボードを AWS にデプロイする（AWS CloudShell で実行）。
#
#   curl -fsSL https://raw.githubusercontent.com/shunshun0904/trade_agent/claude/bitbank-trade-agent-7ers55/dashboard/deploy.sh | bash
#
# 何度実行してもよい。自宅の IP アドレスが変わったら、もう一度実行して新しいアドレスを入れる。
# 作るもの: API Gateway（指定した IP からだけ受け付ける）、Lambda、非公開の S3 バケット（約定の一時保存）、ログ。
# API キーは使わない（bitbank の公開データだけを使う）。
set -euo pipefail

REGION="${TA_REGION:-ap-northeast-1}"
STACK="${TA_DASHBOARD_STACK:-bitbank-profile-dashboard}"
REPO="${TA_REPO_URL:-https://github.com/shunshun0904/trade_agent.git}"
BRANCH="${TA_BRANCH:-claude/bitbank-trade-agent-7ers55}"
WORKDIR="${HOME}/bitbank-profile-dashboard"

cd "$HOME"
if [ -d "$WORKDIR/.git" ]; then
    git -C "$WORKDIR" fetch -q origin "$BRANCH" && git -C "$WORKDIR" checkout -q -B "$BRANCH" "origin/$BRANCH"
else
    git clone -q --depth 1 -b "$BRANCH" "$REPO" "$WORKDIR"
fi
cd "$WORKDIR/dashboard"

if ! command -v sam >/dev/null; then
    python3 -m pip install --quiet --user --upgrade aws-sam-cli
    export PATH="$HOME/.local/bin:$PATH"
fi

CURRENT="$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
    --query "Stacks[0].Parameters[?ParameterKey=='AllowedCidr'].ParameterValue" --output text 2>/dev/null || true)"
[ -n "$CURRENT" ] && [ "$CURRENT" != "None" ] && echo "現在許可しているアドレス: $CURRENT"
echo "ダッシュボードを開く自宅の IP アドレスを入力してください（例: 203.0.113.10）。"
echo "CloudShell の IP ではなく、自宅のブラウザで IP 確認サイトを開いて調べたアドレスです。"
read -r -p "IP アドレス: " IP < /dev/tty
if [[ "$IP" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
    CIDR="$IP/32"
elif [[ "$IP" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}/([0-9]|[12][0-9]|3[0-2])$ ]]; then
    CIDR="$IP"
else
    echo "IP アドレスの形式ではありません: $IP" >&2
    exit 1
fi

sam build --region "$REGION"
sam deploy --stack-name "$STACK" --region "$REGION" --resolve-s3 --capabilities CAPABILITY_IAM \
    --parameter-overrides "AllowedCidr=$CIDR" --no-confirm-changeset --no-fail-on-empty-changeset

URL="$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='DashboardUrl'].OutputValue" --output text)"
echo
echo "ダッシュボード: $URL"
echo "（$CIDR からのみ開けます。IP が変わったら、このスクリプトをもう一度実行してください）"
