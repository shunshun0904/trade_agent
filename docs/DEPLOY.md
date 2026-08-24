# デプロイとフェーズ移行

## 前提

- AWS アカウントは **有料プラン(従量課金)** で作成する。クレジット制の
  「無料プラン」は6ヶ月で閉鎖されるため使用禁止(仕様 §17.2)
- リージョンは東京(`ap-northeast-1`)
- SAM CLI と Python 3.11
- SES で送信元・送信先アドレスを検証済みにしておく(サンドボックス状態でも
  検証済みアドレス同士なら送信できる)

## 1. シークレットを SSM に置く

コードとリポジトリへの直書きは禁止(仕様 §12)。

```bash
aws ssm put-parameter --type SecureString --name /trade-agent/bitbank/api-key    --value '...'
aws ssm put-parameter --type SecureString --name /trade-agent/bitbank/api-secret --value '...'
aws ssm put-parameter --type SecureString --name /trade-agent/anthropic/api-key  --value '...'
aws ssm put-parameter --type SecureString --name /trade-agent/mcp/bearer-token   --value "$(openssl rand -base64 32)"
```

bitbank のキーは **参照 + 取引のみ。出金権限を付けないこと。**

## 2. デプロイ

```bash
sam build
sam deploy --guided \
  --parameter-overrides \
    Environment=prod \
    OwnerEmail=you@example.com \
    SenderEmail=bot@example.com \
    PaperTrading=true \
    Phase=1
```

`PaperTrading=true` の間、執行層は bitbank の Private 注文 API に到達できない。

出力される `McpEndpoint` を claude.ai の「カスタムコネクタ」として登録し、
Bearer トークンに上で生成した値を設定する。

コンソールでの手作業変更は禁止(仕様 §17.3)。変更は必ず IaC 経由で行う。

## 3. デプロイ直後の確認

```bash
# 取引所定数が設定と一致しているか(Phase 2 前の必須項目)
make verify-pair

# エージェントが実際に受け取る JSON
PYTHONPATH=src python -m trade_agent.cli snapshot --prompt

# MCP が生きているか
curl -sS -X POST "$MCP_URL" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | head -40
```

CloudWatch で `trade-agent-prod-tick-heartbeat` アラームが `OK` になることを
確認する。`INSUFFICIENT_DATA` のままなら tick が動いていない。

## 4. フェーズ移行

各移行にはオーナーの明示承認が必要(仕様 §13)。

### Phase 1 → Phase 2

最低1ヶ月のペーパートレードを終え、次を満たしていること。

- [ ] 手数料込みの期待値がプラス
- [ ] 冪等性・再起動テストに合格(`make test` の `test_acceptance.py`)
- [ ] 3日ルールの発動が月2回以下
- [ ] `make verify-pair` が差分なし
- [ ] `agent_calls` の `cache_read_tokens` を確認し、想定どおりのコストか把握した
- [ ] 月次 LLM 費の実績が予算内

```bash
sam deploy --parameter-overrides \
  Environment=prod OwnerEmail=... SenderEmail=... \
  PaperTrading=false Phase=2
```

Phase 2 では全トレードが最小ロット(0.0001 BTC)固定になる
(`risk/rules.py` の `position_size` が `phase == 2` を見ている)。

### Phase 2 → Phase 3

1ヶ月の実弾運用を終え、オーナーが承認したら `Phase=3`。
以降は仕様どおりの資金管理(equity の 1%)でサイズが決まる。

## ロールバック

```bash
# 取引だけ即座に止める(建玉監視は継続)
PYTHONPATH=src python -m trade_agent.cli mcp pause_trading --args '{"confirm":true,"reason":"..."}'

# ペーパーに戻す
sam deploy --parameter-overrides ... PaperTrading=true Phase=1
```

## バックアップ

DynamoDB は Point-in-Time Recovery を有効化してある。
S3 はバージョニング有効、非現行バージョンは90日で失効。
