# TPO・価格帯別出来高ダッシュボード

直近 3 時間の BTC/JPY の価格帯別出来高と TPO を 1 分足の粒度で、30 秒ごとに更新して表示する。bitbank の公開データだけを使い、API キーは使わない。

- 構成: API Gateway（REST API。許可した IP アドレスからだけ受け付ける）→ Lambda（呼ばれたときだけ計算）→ S3（約定の一時保存、非公開）
- 画面（React、`web/`）: 左に 1 分足（直近 3 時間）と POC・VAH・VAL の線、価格帯別出来高、TPO（5 分区間）。右に直近 60 分の 1 分ごとの水準（POC までの距離、バリューエリア内の位置、価格帯の厚さ、シングルプリントの割合）の推移。タブが裏にある間は更新しない
- API: `/api/profile`（左の列）と `/api/signals`（右の列）。どちらも標準ライブラリだけで計算する。分ごとの水準は実行環境の中に残し、新しい分だけ計算する（初回 60 分で約 1.4 秒、以後 0.1 秒）
- 計算の定義は `bbresearch/profile.py` と同じ（`tests/test_dashboard.py` で照合）。窓は 3 時間、TPO は 1 分足 5 本の区間（研究用の 24 時間・15 分足 2 本と異なる。2026-09-26 オーナー決定）。σ は直近 180 本の 1 分足のリターンの標準偏差（刻みは 0.25σ ≈ 0.025%）

## デプロイ（AWS CloudShell、ap-northeast-1）

```bash
curl -fsSL https://raw.githubusercontent.com/shunshun0904/trade_agent/claude/bitbank-trade-agent-7ers55/dashboard/deploy.sh | bash
```

途中で自宅の IP アドレスを聞かれる（CloudShell の IP ではない）。最後に表示される URL を自宅のブラウザで開く。
IP アドレスが変わったら、同じコマンドをもう一度実行する。更新間隔を変えるには `sam deploy` の
`--parameter-overrides` に `RefreshSeconds=10` などを足す（最小 5 秒）。

## 画面の変更（`web/`）

React（Vite）で書いてある。ビルドした 1 ファイルを `app/page.html` に置き、Lambda がそれを返す。`app/page.html` は生成物だが
コミットする（CloudShell でのデプロイに Node.js を要らなくするため）。画面を変えたら次を実行してコミットする。

```bash
dashboard/web/build.sh                 # npm ci → build → app/page.html に置く
cd dashboard/web && VITE_API_BASE=https://<api-id>.execute-api.ap-northeast-1.amazonaws.com/prod npm run dev   # 手元で見る
cd dashboard/web && npm run build:preview    # dist-preview/index.html にダミーデータ版（API に接続しない）
```

- API の形は `web/src/api.js` の冒頭に書いた。
- モデルの予測は載せない（2026-09-26 オーナー決定。方向の予測は費用を超えなかった。`docs/SPEC.md` §1.3）。
- 更新間隔は Lambda の `REFRESH_SECONDS`（既定 30 秒）。画面は 1 回の更新で `/api/profile` と `/api/signals` を 1 回ずつ呼ぶ。

## 削除

```bash
aws cloudformation delete-stack --stack-name bitbank-profile-dashboard --region ap-northeast-1
```

S3 バケットに約定の一時保存が残っていると削除に失敗するので、先にバケットを空にする。
