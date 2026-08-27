# claude.ai にカスタムコネクタとして登録する

## 結論(先に手順だけ)

1. claude.ai → 設定 → コネクタ → **カスタムコネクタを追加**
2. **URL** に、Function URL の末尾に `mcp/<トークン>` を足したものを入れる:

   ```
   https://<関数ID>.lambda-url.ap-northeast-1.on.aws/mcp/<トークン>
   ```

3. **Authentication は「None」を選ぶ**
4. OAuth client の欄は、None を選べば無視される。何も入れない

デプロイスクリプトの最後に、この完成形の URL がそのまま表示される。

トークンを後から取り出す:

```bash
aws ssm get-parameter --with-decryption \
  --name /trade-agent/mcp/bearer-token --region ap-northeast-1 \
  --query Parameter.Value --output text
```

## なぜ「None」なのか

登録画面の Authentication は3択で、上2つは **OAuth 専用**である。

| 選択肢 | 意味 |
|---|---|
| Always required | 各ユーザーがサーバーの **OAuth フロー**でサインインする |
| Required when the server asks | 認証なしで接続し、サーバーが要求したらサインインさせる |
| **None** | サインインなし。**API キーを使うサーバー**もこれ |

このサーバーは OAuth 認可サーバーを持っていない。共有シークレット(ベアラートークン)1本で認証する設計である。したがって上2つは選べない。

そして **None を選ぶと、claude.ai は `Authorization` ヘッダを送らない。** 個人アカウントの登録画面には、固定ヘッダを入力する欄がそもそも存在しない(組織管理者向けのベータ機能としては別に存在する)。

## それが引き起こしていた問題

**当初の実装は `Authorization: Bearer <token>` ヘッダしか受け付けなかった。** つまり、この画面から登録する方法が存在しなかった。私の設計ミスである。

`curl` や Claude Code からはヘッダを送れるので気づきにくいが、claude.ai の登録画面という**本来の想定利用経路**で詰んでいた。

## 直し方と、そのトレードオフ

トークンを **URL のパスに載せる**経路を追加した(`/mcp/<token>`)。ヘッダも従来どおり受け付ける。両方来た場合はヘッダを優先する。

```
Authorization: Bearer <token>     ← curl、Claude Code
/mcp/<token>                      ← claude.ai のカスタムコネクタ
```

**これはヘッダより弱い。** URL はリクエストの中で最も書き残されやすい部分だからである — プロキシのログ、ブラウザの履歴、設定画面のスクリーンショット。ヘッダはそうならない。

それでもこちらを選んだ理由は、代替案が「ヘッダを使う」ではなく「**取引システムの唯一の公開コンポーネントに OAuth 2.1 認可サーバーを立てる**」だからである。`/authorize`、`/token`、PKCE、動的クライアント登録、そのすべてを Lambda に載せることになる。攻撃面と実装量の増加が、得られる強度に見合わない。

緩和策は2つ:

- **このリポジトリのコードはリクエストパスを一切ログに出さない。** `mcp_handler.py` にその旨のコメントを置いてある
- **トークンは 256 ビットの URL セーフな乱数**(`openssl rand -hex 32`)。推測は不可能で、URL の中で「いかにも秘密」に見える形にもならない

**この URL 全体を認証情報として扱うこと。** 貼り付け先は claude.ai の設定画面だけにする。

## トークンが base64 だった場合

以前の版のスクリプトは `openssl rand -base64 32` を使っていた。base64 は `/` と `+` を含みうるため、**URL のパスに載せられない**(`/` がパス区切りとして解釈される)。

デプロイスクリプトは、保存済みトークンが URL セーフでない場合にそれを検出し、入れ替えを提案する。手でやる場合:

```bash
NEW_TOKEN="$(openssl rand -hex 32)"
aws ssm put-parameter --overwrite --type SecureString \
  --name /trade-agent/mcp/bearer-token --value "$NEW_TOKEN" \
  --region ap-northeast-1
echo "$NEW_TOKEN"
```

トークンは SSM から毎回読まれるので、**Lambda の再デプロイは不要**。入れ替えたら claude.ai 側の URL を新しいトークンで登録し直す。

## 疎通確認

**AWS CloudShell で、これを実行する。**

```bash
cd ~/trade_agent && bash scripts/check_mcp.sh
```

URL とトークンを自分で組み立てる必要はない。スクリプトが CloudFormation の
出力と SSM から取ってきて、3つを確認する。

```
✓ token in the path  (claude.ai):          HTTP 200, 7 tool(s)
✓ Authorization header (curl / Claude Code): HTTP 200, 7 tool(s)
✓ no token: HTTP 401 (correctly refused)
```

最後に、claude.ai に貼る完成形の URL を表示する。

3つ目が 401 以外なら、**登録してはいけない。** 誰でも URL さえ知っていれば
停止・再開を叩ける状態ということなので、先に原因を潰す。

### なぜスクリプトなのか(curl の落とし穴)

同じことを手で curl すると、シェルによっては次のエラーになる。

```
curl: (3) URL rejected: Port number was not a decimal number between 0 and 65535
```

**ポートの話ではない。** リクエストボディの JSON が `-d` のデータではなく
**URL として** curl に渡っていて、`"jsonrpc":"2.0"` の `:` をホストとポートの
区切りと解釈し、`"2.0"` はポート番号ではないと言っている。

原因は単一引用符が効いていないこと。Windows の `cmd` は `'...'` を引用符として
扱わないため、JSON が複数の引数にばらけて curl に渡る。PowerShell も引用の
規則が違う。**bash(CloudShell)で実行すれば起きない。**

`check_mcp.sh` はボディを一時ファイルに書いて `-d @file` で渡すので、
引用の問題自体が発生しない。

### どうしても手で叩く場合

bash(CloudShell、macOS、Linux)なら:

```bash
URL="$(aws cloudformation describe-stacks --stack-name trade-agent-prod \
  --region ap-northeast-1 \
  --query "Stacks[0].Outputs[?OutputKey=='McpEndpoint'].OutputValue" --output text)"
TOKEN="$(aws ssm get-parameter --with-decryption \
  --name /trade-agent/mcp/bearer-token --region ap-northeast-1 \
  --query Parameter.Value --output text)"

printf '%s' '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' > /tmp/mcp.json
curl -sS -X POST "${URL%/}/mcp/${TOKEN}" \
  -H 'Content-Type: application/json' -d @/tmp/mcp.json
```

Windows の `cmd` なら、JSON をファイルに置いて `-d @` で渡す。
`"` を `\"` にエスケープして1行に押し込むのは、ここでは避けたほうがよい。

```cmd
echo {"jsonrpc":"2.0","id":1,"method":"tools/list"} > mcp.json
curl -sS -X POST "https://<関数ID>.lambda-url.ap-northeast-1.on.aws/mcp/<トークン>" -H "Content-Type: application/json" -d @mcp.json
```

7つのツールが並べば成功。`{"error":"missing bearer token"}` が返るなら、
パスの形が `/mcp/<token>` になっていない。

## 登録後にできること

読み取り専用ツールは自由に呼べる。`pause_trading` と `resume_trading` は `confirm=true` が必須で、監査ログに残る(仕様 §16)。

注文の発注・取消を行うツールは**存在しない**。MCP 経由で新規建玉を作ることはできない。この関数の IAM ロールは bitbank の認証情報を読む権限自体を持っていない(仕様 §12)。
