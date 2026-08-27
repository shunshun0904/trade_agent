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

登録前に、URL が生きていることを確かめられる。

```bash
URL="https://<関数ID>.lambda-url.ap-northeast-1.on.aws/mcp/<トークン>"

curl -sS -X POST "$URL" \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | head -40
```

7つのツールが並べば成功。`{"error":"missing bearer token"}` が返るなら、パスの形が `/mcp/<token>` になっていない。

ヘッダ経由でも同じ結果になる(Claude Code から使う場合はこちら):

```bash
curl -sS -X POST "https://<関数ID>.lambda-url.ap-northeast-1.on.aws/" \
  -H "Authorization: Bearer <トークン>" \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | head -40
```

## 登録後にできること

読み取り専用ツールは自由に呼べる。`pause_trading` と `resume_trading` は `confirm=true` が必須で、監査ログに残る(仕様 §16)。

注文の発注・取消を行うツールは**存在しない**。MCP 経由で新規建玉を作ることはできない。この関数の IAM ロールは bitbank の認証情報を読む権限自体を持っていない(仕様 §12)。
