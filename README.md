# trade-agent — BTC/JPY マルチエージェント自動売買システム

bitbank 現物 BTC/JPY を対象とした、LLM マルチエージェント方式のデイトレード
システム。AWS サーバーレス構成で動作する。

> **現在のフェーズ: Phase 1(ペーパートレード)**
> `system.paper_trading` が `true` の間、執行層は bitbank の Private 注文 API に
> **構造的に到達できない**。Phase 2 への移行にはオーナーの明示承認が必要
> (仕様 §13)。

---

## 0. 安全に関する前提(必読)

### bitbank API キーには出金権限を付与しない

本システムが必要とするのは **参照(照会)** と **取引** の権限だけである。
出金権限を持つキーを設定してはならない。コード側にも出金 API を呼ぶ経路は
一切存在しない(`exchange/base.py` の `ExchangeClient` に出金メソッドがない)。

キー・シークレットは **SSM Parameter Store(SecureString)** にのみ置く。
リポジトリ、設定ファイル、環境変数ファイル、チャットに貼らないこと。

```bash
aws ssm put-parameter --type SecureString --name /trade-agent/bitbank/api-key    --value '...'
aws ssm put-parameter --type SecureString --name /trade-agent/bitbank/api-secret --value '...'
aws ssm put-parameter --type SecureString --name /trade-agent/anthropic/api-key  --value '...'
aws ssm put-parameter --type SecureString --name /trade-agent/mcp/bearer-token   --value "$(openssl rand -hex 32)"
```

**キーが第三者の目に触れた場合(スクリーンショット共有を含む)は、
必ず bitbank 管理画面で削除し、新しいキーを再発行すること。**
一度でも露出したキーは、権限が限定されていても再利用してはならない。

### 退屈防止ルール(3日ルール)について

**現在は無効(`boredom.enabled: false`)。** 合議閾値
`screening.consensus_min` を 1 にした時点で、このルールの唯一のレバー
(合議を `relaxed_consensus_min` まで緩める)が通常時と同じ値になり、
発火しても何も変えられなくなった。72時間無取引という前提も、頻度を上げた
現在の設定では成立しない。コードは残してあるが、動作しない。

以下は有効化した場合の挙動である。72時間無取引を作らないという
**オーナーのエンタメ要求** に基づく機能であり、
**統計的な優位性を持たない**(仕様 §7)。実装上は次のように隔離してある。

- 発注は最小ロット固定(0.0001 BTC)、リスク上限 0.5%、損切りは entry から -0.7% 以内
- `probe=true` フラグが付き、戦略成績の集計から常に除外される
- キルスイッチ・連敗ブレーキ・日次損失上限・急変動停止のいずれかが作動中は**発火しない**
- probe の月間累計損失が equity の 2% に達した時点で、当月は自動停止しオーナーへ通知

### 収益について

月利10%は **目標(target)** であって保証でも制約でもない。本システムのいかなる
出力もそれを約束しない。第1期(3ヶ月)は検証・学習フェーズであり、稼働費を含めた
黒字化は目標に含まれない(仕様 §1)。

---

## 1. これは何をするか

30分ごとに市況を機械的にスクリーニングし(LLM 不使用・コスト0円)、条件が
成立したときだけ 7体の LLM エージェントによるフル議論を1サイクル回す。

```
市況分析(A1)
   ↓
戦略3案(A2a 順張り / A2b 逆張り / A2c 悲観)  ← 相互不可視で独立提案
   ↓
相互批判(匿名化した他2案を1ラウンド)
   ↓
合意ルール(Python が判定: 3案中2案以上が buy でなければ no_trade)
   ↓
裁定(A3) → サイズ算出(Python) → リスク査定(A4)
   ↓
決定論ガード → 再クオート → 冪等発注
   ↓
自作OCO(損切りを取引所側に発注、利確はローカル評価)
```

決済後は A7 が **直近20件以上の集計統計** から教訓を抽出し、次サイクルの
コンテキストに載せる。1トレードから断定させることはしない。

### エージェントは7体

仕様は9体(A1〜A7)を定義しているが、**検査官(A5)と指揮官(A6)は置いていない。**

| | 役割 | 1サイクルのコール数 |
|---|---|---|
| A1 | 市況分析 | 1 |
| A2a/b/c | 戦略3案(独立提案) | 3 |
| A2 | 相互批判(匿名化) | 3 |
| A3 | 裁定 | 1 |
| A4 | リスク査定 | 1 |
| A7 | 自己分析(決済後・非同期) | — |
| | **合計** | **9コール** |

外した理由:

- **検査官(A5)** は LLM が LLM を検査する層だった。数値検証は決定論ガードが
  全部やっており、残るのは定性的な矛盾探しだけで、そこが一番当てにならない。
- **指揮官(A6)** の GO/NO-GO は、合意ルール(Python)・サイジング(Python)・
  リスク査定(A4)・決定論ガード(Python)を全部通った後に乗る5つ目の拒否権だった。

指揮官が唯一やっていた冗長でない仕事 —— オーナー向けレポート本文 —— は
**Python が組み立てる**(`orchestrator/report.py`)。材料はすべて構造化済みの
フィールドなので、モデルに書き直させる必要がない。むしろ数値が必ず入り、
実際に起きたことと食い違わない。

### 損切りは取引所側に置く

bitbank に OCO エンドポイントは無いため自作している(仕様 §8)。
現物残高は売り注文を1つしか裏付けられないので、**損切りを取引所側の
`stop` 注文として置き、利確は5分tickがローカル評価する**。

これにより **このプロセスが停止しても損切りは働く**。
ローカル評価はバックストップとして残り、取引所側の脚が拒否・消失した
水準だけを引き継ぐ(二重売却はしない)。

詳細と未検証事項は [docs/OPEN-QUESTIONS.md](docs/OPEN-QUESTIONS.md) A-1。

### 設計の中心にある3つの決定

1. **数値は LLM に計算させない。** 価格・指標・数量・損益はすべて Python が
   確定させ、JSON でモデルに渡す。モデルの仕事は解釈・判断・批判・文章化だけ。
   引用された指標値は決定論ガードがスナップショットの実値と照合し、
   食い違えば出力を棄却する。
2. **合意ルールは Python が数える。** 裁定者が自分で言いくるめられるルールは
   ルールではない。
3. **安全装置は LLM の前に評価する。** 停止中のシステムは1トークンも消費しない。

---

## 2. すぐ試す(AWS 不要)

```bash
make install
make test                                    # 287 tests
PYTHONPATH=src python -m trade_agent.cli --local decide    # 1サイクルをオフライン実行
```

`--local` はインメモリストアと決定論的なオフライン LLM を使う。API キーも
AWS アカウントも不要で、仕様 §14 の受け入れ基準はすべてこの状態で再現できる。

### 取引所定数の再確認(仕様 §2)

最小注文数量と手数料は `config/default.yaml` に置いてあるが、正しさの根拠は
取引所側にある。次のコマンドが `GET /v1/spot/pairs`(認証不要)と設定値を
突き合わせ、差分があれば終了コード 2 を返す。

```bash
make verify-pair
```

**Phase 2(実弾)へ移行する前に必ず実行すること。** この値は全注文の
サイズ計算に効く。

---

## 3. 主なコマンド

| コマンド | 内容 |
|---|---|
| `trade-agent verify-pair` | 取引所定数と設定値の差分検査 |
| `trade-agent snapshot --prompt` | エージェントが実際に受け取る JSON を表示 |
| `trade-agent status` | オーナー向けステータス(MCP の `get_status` と同じ) |
| `trade-agent tick` | 5分監視パスを1回 |
| `trade-agent screen` | 30分スクリーニングを1回 |
| `trade-agent decide` | フル議論を1サイクル |
| `trade-agent reflect` | 決済後の集計分析 |
| `trade-agent backfill --days 60` | ヒストリカルローソク足の取得 |
| `trade-agent backtest` | 決定論レイヤのリプレイ |
| `trade-agent mcp get_status` | MCP ツールをローカル実行 |

---

## 4. 構成

| 層 | モジュール | LLM | 概要 |
|---|---|---|---|
| ① データ | `data/`, `exchange/` | — | ローソク足・板・残高 → MarketSnapshot |
| ② オーケストレータ | `orchestrator/` | — | 状態遷移、差し戻しループ、ロック |
| ③ エージェント | `agents/` | ✓ | A1〜A4 + A7(7体、下記) |
| ④ 決定論ガード | `guards/` | — | スキーマ検証と数値照合 |
| ⑤ 執行 | `execution/` | — | 冪等発注、再クオート、SL/TP 監視 |
| ⑥ オーナー対話 | `mcp/` | — | リモート MCP サーバー(プル型) |
| ⑥' 緊急通知 | `notify/` | — | SES メール(プッシュ) |
| ⑦ 記憶 | `storage/` | — | DynamoDB + S3 |

Lambda は5本(`tick` / `screen` / `decide` / `reflect` / `mcp`)。
詳細は [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

---

## 5. コスト

| 項目 | 月額 |
|---|---|
| LLM(Anthropic 直 API) | 1サイクル7〜9コール。予算 **2,900円/月** |
| AWS インフラ | 0〜100円(常時無料枠内の設計) |
| **合計上限** | **3,000円** |

**1日の議論回数に上限はない。** 支出を日割りでペース配分する:

```
本日の許容額 = (月間予算 - 当月実績) / 当月の残り日数 × 2.0
```

回数は金額の代理変数として質が悪い(1サイクルは合議に届くかで7〜9コールに
変動し、キャッシュの効き方でも実費が変わる)。荒れた日は多く、静かな日は少なく
使ってよく、**使いすぎた分は翌日以降の取り分が自動的に縮む**ので月間予算は
破られない。100% で当月の LLM 呼び出しを停止する。
**停止しても5分tickと損切り監視は動き続ける**(仕様 §11)。
AWS Budgets が $2 超過で警告メールを送る。

> **既知の注意点:** `claude-haiku-4-5` は共通プレフィックスが 4,096 トークン
> 未満だとキャッシュエントリを作らない(エラーにはならず、静かに全額課金される)。
> 現状の共通プレフィックスは約 2,000 トークンで、この閾値を下回る。
> `agent_calls` テーブルの `cache_read_tokens` が 0 のままなら効いていない。
> 対策は docs/ARCHITECTURE.md の「プロンプトキャッシュ」節を参照。

---

## 6. オーナーとの対話(MCP)

`mcp` Lambda の Function URL を claude.ai の「カスタムコネクタ」として登録する。
**Authentication は「None」を選び、トークンは URL のパスに載せる**
(`.../mcp/<トークン>`) — 理由と手順は [docs/MCP.md](docs/MCP.md)。
`Authorization: Bearer` ヘッダも受け付ける(curl / Claude Code 用)。
認証は必須で、トークンのないリクエストは常に 401 になる。

**プル型**であり、オーナーが質問したときだけ情報が流れる。
緊急イベントの即時通知はメールが担う(仕様 §16.1)。

| ツール | 種別 |
|---|---|
| `get_status` / `get_daily_report` / `get_trades` / `get_agent_log` / `get_lessons` / `get_cycles` | 読取 |
| `pause_trading` / `resume_trading` | 操作(`confirm=true` 必須) |

`get_cycles` は仕様 §16 の7ツールに対する8つ目で、意図的に足したもの。サイクルが約定に至らない理由は8種類あり(合議不成立・裁定者の見送り・サイズ計算の却下・リスク査定の却下・構造チェック・取引所の拒否…)、**それぞれ打つべき手が違う。** 従来その理由は Lambda の戻り値と CloudWatch の1行にしか存在せず、日次レポートが拾うのは21時をまたいだ1サイクルだけだった。

MCP から発注はできない。できる最大の操作は「停止」と「再開」まで。
`mcp` Lambda の IAM ロールには bitbank 秘密鍵の読取権限を与えていない(仕様 §12/§16.3)。

---

## 7. やらないこと(仕様 §15)

- ショート・レバレッジ・信用取引(現物ロングオンリー)
- BTC/JPY 以外の通貨ペア
- 秒単位の高頻度取引
- VPS・常駐サーバー・Kubernetes での運用
- AWS Bedrock 経由の LLM 呼び出し(Anthropic 直 API を使う)
- 収益の保証

---

## 8. デプロイ

AWS CloudShell を開いて1行貼るだけ。対話形式で必要事項を聞かれる。

```bash
curl -fsSL https://raw.githubusercontent.com/shunshun0904/trade_agent/claude/trade-agent-spec-mtwldk/scripts/deploy.sh | bash
```

シークレットのSSM登録、SESのアドレス検証、ビルド、デプロイ、デプロイ後検証
までを通しで行う。**何度でも再実行できる。**
`PaperTrading=true` 固定で入るため、この時点で実発注は起こらない。

デプロイ後の健全性は1コマンドで確認できる:

```bash
PYTHONPATH=src python -m trade_agent.cli preflight
```

詳細と手動手順は [docs/DEPLOY.md](docs/DEPLOY.md)。

## 9. ドキュメント

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — 設計判断とその理由
- [docs/DEPLOY.md](docs/DEPLOY.md) — デプロイ手順とフェーズ移行
- [docs/RUNBOOK.md](docs/RUNBOOK.md) — 障害対応
- [docs/OPEN-QUESTIONS.md](docs/OPEN-QUESTIONS.md) — **仕様の曖昧点とこちらの暫定判断(要確認)**
