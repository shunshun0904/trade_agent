# TPO・価格帯別出来高ダッシュボード

直近 24 時間の BTC/JPY の価格帯別出来高と TPO を、10 秒ごとに更新して表示する。bitbank の公開データだけを使い、API キーは使わない。

- 構成: API Gateway（REST API。許可した IP アドレスからだけ受け付ける）→ Lambda（呼ばれたときだけ計算）→ S3（約定の一時保存、非公開）
- 画面: 15 分足と POC・VAH・VAL の線、価格帯別出来高、TPO。タブが裏にある間は更新しない
- 計算の定義は `bbresearch/profile.py` と同じ（`tests/test_dashboard.py` で照合）。σ だけは表示用に直近 96 本の標準偏差を使う

## デプロイ（AWS CloudShell、ap-northeast-1）

```bash
curl -fsSL https://raw.githubusercontent.com/shunshun0904/trade_agent/claude/bitbank-trade-agent-7ers55/dashboard/deploy.sh | bash
```

途中で自宅の IP アドレスを聞かれる（CloudShell の IP ではない）。最後に表示される URL を自宅のブラウザで開く。
IP アドレスが変わったら、同じコマンドをもう一度実行する。更新間隔を変えるには `sam deploy` の
`--parameter-overrides` に `RefreshSeconds=30` などを足す（最小 5 秒）。

## React 版の画面（`web/`）

TPO・価格帯別出来高に、1 分ごとのシグナル（モデル B の上げ確率と生の水準）の直近 60 分の推移と、毎正時の判断を
並べる画面。2026-09-26 オーナー指示。今の Lambda の `/api/profile` にそのままつながる。

```bash
cd dashboard/web
npm install
VITE_API_BASE=https://<api-id>.execute-api.ap-northeast-1.amazonaws.com/prod npm run dev   # 手元で見る（/api を転送）
npm run build            # dist/ に静的ファイル
npm run build:preview    # dist-preview/index.html に 1 ファイル。ダミーデータで動く（プレビュー用）
```

- `/api/signals` の形は `web/src/api.js` の冒頭に書いた。Lambda にモデルを載せるまでは 404 を返すので、画面は「未接続」と出す。
- 右の列の 5 つの要約（平均・最新・変化・ばらつき・直近 15 分）は `bbresearch/direction.py` の `stack_features` と同じ定義。
- タブが裏にある間は取得しない。更新間隔は `VITE_REFRESH_SECONDS`（既定 10 秒）。
- 配信は未定（S3 + CloudFront か、Lambda から `dist/index.html` を返す）。決まったら `deploy.sh` に足す。

## 削除

```bash
aws cloudformation delete-stack --stack-name bitbank-profile-dashboard --region ap-northeast-1
```

S3 バケットに約定の一時保存が残っていると削除に失敗するので、先にバケットを空にする。
