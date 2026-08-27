# デプロイとフェーズ移行

## 前提

- **AWSアカウントは有料プラン(従量課金)で作成する。** クレジット制の
  「無料プラン」は6ヶ月で閉鎖されるため使用禁止(仕様 §17.2)。
  運用自体は常時無料枠に収まる設計である
- **AWS CloudShell を使う。** ローカルに AWS CLI も SAM CLI も入れる必要はない
- bitbank の API キー(**参照 + 取引のみ。出金権限なし**)
- Anthropic の API キー

## 1. 一発デプロイ(AWS CloudShell)

AWS マネジメントコンソール右上の **CloudShell** アイコンを開き、次の1行を貼る。
対話形式で必要事項を聞かれる。

```bash
curl -fsSL https://raw.githubusercontent.com/shunshun0904/trade_agent/claude/trade-agent-spec-mtwldk/scripts/deploy.sh | bash
```

CloudShell を使う理由は、AWS 認証情報が最初から通っていること、そして
**ビルドがそこで完結すること**である。Lambda に載せる `pydantic-core` と
`jiter` はアーキテクチャ別のバイナリを持つため、テンプレートは CloudShell と
同じ x86_64 を指定してある(無料枠は 400,000 GB秒で arm64 と同じ)。

**CloudShell の Python が 3.11 でなくても動く。** ビルドは SAM 標準の Python
ビルダーではなく `scripts/build_lambda.sh`(makefile ビルダー)を使う。詳細は
[1.1](#11-ビルド方式makefile-ビルダー)。

聞かれるのは4つだけ:

| 入力 | 備考 |
|---|---|
| bitbank API キー | **参照 + 取引のみ。出金権限は付けない** |
| bitbank API シークレット | 入力は画面に表示されない |
| Anthropic API キー | `sk-ant-...` |
| 通知先メールアドレス | 送信元は既定で同じアドレス |

MCP の Bearer トークンは自動生成される。

スクリプトが行うこと:

```
0/5  環境      リージョン・認証情報の確認、SAM CLI が無ければ導入
1/5  ソース    リポジトリの clone / 更新
2/5  シークレット  SSM Parameter Store へ SecureString で保存
3/5  メール    SES のアドレス検証(検証メールのリンクを押す)
4/5  デプロイ  sam build → 成果物の検証 → sam deploy
5/5  検証      取引所定数・APIキー・残高の preflight
```

**何度でも再実行できる。** 既にある SSM パラメータは上書き前に確認され、
CloudFormation スタックは差分だけが適用される。

### 途中で失敗したら

| 症状 | 対処 |
|---|---|
| `no usable AWS credentials` | CloudShell 以外で実行している。CloudShell から実行する |
| `clone failed` | リポジトリが private。CloudShell に git 認証を設定する |
| ディスク不足 | `rm -rf ~/trade_agent/.aws-sam` して再実行 |
| `refusing to deploy a package that will not start` | 成果物の検証で落ちた。表示される問題(ABIタグ不一致・欠損ファイル)をそのまま直す |
| `PythonPipBuilder:Validation - Binary validation failed for python ... runtime: python3.11` | 古い版のスクリプトで実行している。`rm -rf ~/trade_agent` して貼り直す(現在は makefile ビルダーを使うため、ホストの Python 版は不問) |
| `sam deploy failed` | スクリプトが失敗したリソースと理由を抽出して表示する。下の「スタックがロールバックしたとき」も参照 |
| `Waiter StackCreateComplete failed ... ROLLBACK_COMPLETE` | 初回作成が失敗した。スタックは空の抜け殻として残り、**更新できない**。再実行すれば自動で削除される |
| `Stack ... is in ROLLBACK_COMPLETE state and can not be updated` | 同上。古い版のスクリプトで手が止まっている場合は下のコマンドで削除する |
| `... already exists` | 前回の失敗が残したリソースを掴んでいる。ロググループなら再実行時に削除を提案される |

### スタックがロールバックしたとき

CloudFormation の**初回作成**が失敗すると、スタックは全リソースを巻き戻した
うえで `ROLLBACK_COMPLETE` という空の抜け殻として残る。この状態のスタックは
**更新できない**(AWS の仕様)。削除して作り直すしかないため、そのまま
`sam deploy` を再実行すると

```
Stack:trade-agent-prod is in ROLLBACK_COMPLETE state and can not be updated.
```

という、元の失敗とは無関係なエラーに変わって原因が見えなくなる。

スクリプトは 4/5 の冒頭でスタックの状態を見て、この状態なら**元の失敗理由を
表示してから削除を提案する**。したがって普通は貼り直すだけでよい。

手で調べる場合:

```bash
# なぜ落ちたか(「Resource creation cancelled」は巻き添えなので除く)
aws cloudformation describe-stack-events --stack-name trade-agent-prod \
  --region ap-northeast-1 \
  --query "StackEvents[?ResourceStatus=='CREATE_FAILED'].[LogicalResourceId,ResourceStatusReason]" \
  --output text | grep -v 'Resource creation cancelled'

# 抜け殻を消す
aws cloudformation delete-stack --stack-name trade-agent-prod --region ap-northeast-1
aws cloudformation wait stack-delete-complete --stack-name trade-agent-prod --region ap-northeast-1
```

巻き戻しは CloudFormation が作ったリソースだけを消す。**CloudFormation が
作っていないものは残る**。この構成で該当するのは Lambda のロググループで、
関数は初回実行時に `/aws/lambda/<関数名>` を自分で作るため、デプロイ中に
スケジュールが発火するとテンプレート側の作成と競合して
`already exists` で落ちうる。しかも残ったロググループは以後のデプロイを
同じ理由で落とし続ける。

テンプレートでは各関数に `DependsOn: <対応するロググループ>` を付けて、
ロググループが先に存在することを保証してある。すでに孤児が残っている場合は、
スクリプトが再実行時に一覧を出して削除を提案する。

環境変数で挙動を変えられる(通常は不要):

```bash
TA_REGION=ap-northeast-1 TA_ENVIRONMENT=prod \
TA_BRANCH=claude/trade-agent-spec-mtwldk bash scripts/deploy.sh
```

## 1.1 ビルド方式(makefile ビルダー)

SAM 標準の Python ビルダーは、pip を回すために **ランタイムと同じ版の
`python3.11` バイナリ**をホストに要求する。CloudShell が積んでいる Python は
その時々の版(執筆時点で 3.13)なので、標準ビルダーだと何もしないうちに

```
PythonPipBuilder:Validation - Binary validation failed for python,
searched for python in following locations : [...] which did not satisfy
constraints for runtime: python3.11
```

で止まる。CloudShell に別版の Python を入れるのは、常に無料枠で完結させる
という前提に合わない。

そこで `template.yaml` の5つの関数はすべて `BuildMethod: makefile` を宣言し、
`Makefile` の `build-<LogicalId>` ターゲット経由で `scripts/build_lambda.sh`
を呼ぶ。pip は **ターゲットの interpreter である必要はなく**、wheel の解決先を
指定できる:

```
--platform manylinux2014_x86_64 --implementation cp
--python-version 3.11 --only-binary=:all: --target <artifacts>
```

これで、ホストが 3.11 でも 3.13 でもその先でも、成果物には
`cpython-311-x86_64-linux-gnu` の拡張モジュールが入る。

副作用として、ホスト側で誤った版の wheel を掴んでも **デプロイは成功して
しまう**(壊れるのは数分後、最初の invoke で import が落ちたとき)。それを
ビルド時のエラーに変えるのが `scripts/verify_artifact.py` で、アップロード前に
次を確認する。

| 検査 | 落ちたときに本番で起きること |
|---|---|
| `.so` の ABI タグがランタイム版と一致するか | 全関数が import 時に落ちる |
| `.so` の ELF `e_machine` がアーキテクチャと一致するか | 同上 |
| `config/default.yaml` があるか | 設定を読めず起動時に落ちる |
| 5つの handler モジュールがあるか | 該当関数だけが `Runtime.ImportModuleError` |
| 依存パッケージがあるか | import 時に落ちる |

成果物を **import せずに** 判定しているのは意図的で、ホストの Python が
ランタイムと違う場合、正しい成果物ほど import に失敗するからである。
ABI タグはどのホストからでも読める。

単体でも回せる:

```bash
make build-TickFunction ARTIFACTS_DIR=.aws-sam/build/TickFunction
make verify-artifact
```

ランタイム版やアーキテクチャを変えるときは、`template.yaml` の
`Runtime` / `Architectures`、`scripts/build_lambda.sh` の
`TA_LAMBDA_PYTHON` / `TA_LAMBDA_PLATFORM` の既定値、
`verify_artifact.py` の `--runtime` / `--arch` を揃える。

## 2. デプロイ後の確認

スクリプトの 5/5 が自動で実行するが、いつでも単体で回せる。

```bash
cd ~/trade_agent
PYTHONPATH=src python3 -m trade_agent.cli preflight
```

チェック内容:

```
[1/4] public market data              板と価格が取れるか
[2/4] exchange constants vs config    最小注文数量・手数料の差分(仕様 §2)
[3/4] bitbank private API credentials キーが通るか、残高はいくらか
[4/4] Anthropic API key               SSM から読めるか
```

**Phase 2(実弾)へ移行する前に、[2/4] が `DRIFT` なしであることを必ず確認する。**
この値は全注文のサイズ計算に効く。

そのうえで、CloudWatch のハートビートアラームを確認する。

```bash
aws cloudwatch describe-alarms --alarm-names trade-agent-prod-tick-heartbeat \
  --region ap-northeast-1 --query "MetricAlarms[0].StateValue" --output text
```

最初の5分tickが回るまでは `INSUFFICIENT_DATA` で正常。15分ほどで `OK` になる。

## 3. MCP コネクタの登録

デプロイ出力の `MCP endpoint` と `Bearer token` を、claude.ai の
「カスタムコネクタ」に登録する。トークンを再表示するには:

```bash
aws ssm get-parameter --with-decryption \
  --name /trade-agent/mcp/bearer-token --region ap-northeast-1 \
  --query Parameter.Value --output text
```

疎通確認:

```bash
curl -sS -X POST "$MCP_URL" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | head -40
```

## 3.5 手動でデプロイする場合

スクリプトを使わない場合の等価な手順。

```bash
aws ssm put-parameter --type SecureString --name /trade-agent/bitbank/api-key    --value '...'
aws ssm put-parameter --type SecureString --name /trade-agent/bitbank/api-secret --value '...'
aws ssm put-parameter --type SecureString --name /trade-agent/anthropic/api-key  --value '...'
aws ssm put-parameter --type SecureString --name /trade-agent/mcp/bearer-token   --value "$(openssl rand -base64 32)"

aws ses verify-email-identity --email-address you@example.com --region ap-northeast-1

sam build
sam deploy --stack-name trade-agent-prod --region ap-northeast-1 \
  --capabilities CAPABILITY_IAM --resolve-s3 --no-confirm-changeset \
  --parameter-overrides \
    Environment=prod OwnerEmail=you@example.com SenderEmail=you@example.com \
    PaperTrading=true Phase=1
```

コンソールでの手作業変更は禁止(仕様 §17.3)。変更は必ず IaC 経由で行う。

## 4. フェーズ移行

各移行にはオーナーの明示承認が必要(仕様 §13)。

### Phase 1 → Phase 2

最低1ヶ月のペーパートレードを終え、次を満たしていること。

- [ ] 手数料込みの期待値がプラス
- [ ] 冪等性・再起動テストに合格(`make test` の `test_acceptance.py`)
- [ ] 3日ルールの発動が月2回以下
- [ ] `trade-agent preflight` が全項目 OK(特に取引所定数の DRIFT なし)
- [ ] `agent_calls` の `cache_read_tokens` を確認し、想定どおりのコストか把握した
- [ ] 月次 LLM 費の実績が予算内

```bash
cd ~/trade_agent
sam deploy --stack-name trade-agent-prod --region ap-northeast-1 \
  --capabilities CAPABILITY_IAM --resolve-s3 --no-confirm-changeset \
  --parameter-overrides \
    Environment=prod OwnerEmail=you@example.com SenderEmail=you@example.com \
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
