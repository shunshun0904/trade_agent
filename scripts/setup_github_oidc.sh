#!/usr/bin/env bash
# GitHub Actions が AWS SSM の bitbank キーを読めるようにする（AWS CloudShell で実行）。
#
#   1. GitHub の OIDC プロバイダを IAM に登録する（既にあれば何もしない）
#   2. IAM ロール trade-agent-bitbank-gha を作る。引き受けられるのは
#      shunshun0904/trade_agent の指定ブランチへの push で動いたワークフローだけ
#   3. ロールに付ける権限は /trade-agent/bitbank/* の読み取りと、SSM 経由の復号だけ
#      （発注・出金の権限は bitbank 側のキーの権限で決まる。出金権限のないキーであること）
#
# 何度実行してもよい。最後にロール ARN を表示するので、GitHub のリポジトリ変数
# AWS_ROLE_ARN（Settings → Secrets and variables → Actions → Variables）に登録する。
set -euo pipefail

REGION="${TA_REGION:-ap-northeast-1}"
REPO="${TA_GITHUB_REPO:-shunshun0904/trade_agent}"
BRANCHES="${TA_GITHUB_BRANCHES:-claude/bitbank-trade-agent-7ers55}"   # 空白区切り
ROLE="${TA_ROLE_NAME:-trade-agent-bitbank-gha}"
PROVIDER_HOST="token.actions.githubusercontent.com"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
PROVIDER_ARN="arn:aws:iam::${ACCOUNT_ID}:oidc-provider/${PROVIDER_HOST}"
echo "account ${ACCOUNT_ID}, region ${REGION}, repo ${REPO}"

# 1. OIDC プロバイダ
if aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$PROVIDER_ARN" >/dev/null 2>&1; then
    echo "[1/3] OIDC プロバイダは登録済み"
else
    aws iam create-open-id-connect-provider --url "https://${PROVIDER_HOST}" --client-id-list sts.amazonaws.com \
        >/dev/null 2>&1 \
    || aws iam create-open-id-connect-provider --url "https://${PROVIDER_HOST}" --client-id-list sts.amazonaws.com \
        --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 >/dev/null
    echo "[1/3] OIDC プロバイダを登録した"
fi

# 2. ロール（信頼ポリシーは指定ブランチの push に限定）
SUBS=""
for b in $BRANCHES; do
    SUBS="${SUBS:+${SUBS},}\"repo:${REPO}:ref:refs/heads/${b}\""
done
TRUST=$(cat <<JSON
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Federated": "${PROVIDER_ARN}"},
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "${PROVIDER_HOST}:aud": "sts.amazonaws.com",
        "${PROVIDER_HOST}:sub": [${SUBS}]
      }
    }
  }]
}
JSON
)
if aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
    aws iam update-assume-role-policy --role-name "$ROLE" --policy-document "$TRUST"
    echo "[2/3] ロール ${ROLE} の信頼ポリシーを更新した"
else
    aws iam create-role --role-name "$ROLE" --assume-role-policy-document "$TRUST" \
        --description "GitHub Actions (${REPO}) reads bitbank keys from SSM" --max-session-duration 3600 >/dev/null
    echo "[2/3] ロール ${ROLE} を作成した"
fi

# 3. 権限（読み取りのみ）
POLICY=$(cat <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["ssm:GetParameter", "ssm:GetParameters"],
      "Resource": "arn:aws:ssm:${REGION}:${ACCOUNT_ID}:parameter/trade-agent/bitbank/*"
    },
    {
      "Effect": "Allow",
      "Action": "kms:Decrypt",
      "Resource": "*",
      "Condition": {"StringEquals": {"kms:ViaService": "ssm.${REGION}.amazonaws.com"}}
    }
  ]
}
JSON
)
aws iam put-role-policy --role-name "$ROLE" --policy-name bitbank-ssm-read --policy-document "$POLICY"
echo "[3/3] 権限を設定した（/trade-agent/bitbank/* の読み取りのみ）"

echo
echo "ロール ARN（GitHub のリポジトリ変数 AWS_ROLE_ARN に登録する）:"
aws iam get-role --role-name "$ROLE" --query Role.Arn --output text
