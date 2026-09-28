# プロジェクト: bitbank 現物トレードエージェント

作業を始める前に `docs/SPEC.md` を読むこと。設計判断・API 仕様の要点・フェーズごとの仕様・未決事項はすべてそこにある。

## 現在の状態

- Phase 1（`bbdata/`）、Phase 0（`scripts/phase0.py`、結果は SPEC.md §9）、Phase 2〜6（`bbresearch/`）は実装済み。
- このブランチは 2026-09-27 に既定ブランチになった（schedule 付きワークフローが自動で動く。60 日間 push がないと GitHub が schedule を止める）。
- 実 API へのアクセスは GitHub Actions で行う。開発環境からは行わない。
  - どのワークフローも、API の workflow_dispatch でこのブランチを ref に指定して実行できる（既定ブランチになくてもよいことを 2026-09-23 に確認）。ただし workflow_dispatch だけで push のきっかけを持たないワークフローは、既定ブランチにないと API が 404 を返した（`swing.yml`、2026-09-28）。push のきっかけも付けておく
  - `phase0.yml`: `scripts/phase0.py` の変更を push すると実行し、`reports/phase0/` をコミットする
  - `research.yml`: `configs/research.yaml` の変更を push すると（`run_on_actions: true` のとき）実行し、`reports/research/` をコミットする
  - `search.yml`: `configs/search.yaml` の変更を push すると（`run_on_actions: true` のとき）パラメータ探索を実行し、`reports/search/` をコミットする（開発期間のみ。ホールドアウトは評価しない）
  - `diagnose.yml`: 損失の要因分解・ホライズン・エントリー方法の比較（`configs/diagnose.yaml`）
  - `maker_search.yml`: 深い指値・短い保有・売りも指値の決済条件の探索と確認期間での評価（`configs/maker_search.yaml`）
  - `val_rule.yml`: ルール A（VAL での反発）を固定した数値で全期間1回評価（`configs/val_rule.yaml`）
  - `direction.yml`: 15 分後の上げ下げを予測する二段構えのモデル（A: 15 分ごと、B: 1 分ごと）を 2024 年より前で学習し、2024 年以降で1回評価（`configs/direction.yaml`）
  - `direction_1h.yml`: 同じモデルの 1 時間版。毎正時に判断し、1 分ごとの TPO×価格帯別出来高のシグナルの直近 60 分の推移を加える（`configs/direction_1h.yaml`）
  - `direction_cost.yml`: 目的変数を「成行往復の費用（約 0.3%）を超えて上がるか」にし、下げ側も学習して差を信号にする版を 15 分と 1 時間で実行（`configs/direction_cost.yaml`、`direction_1h_cost.yaml`）
  - `activity.yml`: 全ペアの前日の約定数・出来高（`reports/activity/`）。`weekend_range.yml`: 曜日 × 時間帯の値幅（`reports/weekend_range/`）。`portfolio.yml`: 最小分散ポートフォリオの前向き検証（`scripts/portfolio.py`、`reports/portfolio/`）
  - `rebalance.yml`: リバランス計画（ドライラン、発注しない。`scripts/rebalance.py`、`configs/rebalance.yaml`）。毎月 1 日 00:33 UTC に銘柄と相対の重みを選び直し、毎週月曜 00:33 UTC に JPY の割合だけ見直す（目標ボラ 30%、BTC が 200 日線を下回る間は全額 JPY。2026-09-27 オーナー決定）。ログは割合だけ
  - `recorder.yml`: 板・約定の記録。約 5 時間 45 分ごとに次のジョブを自分で起動して連続させ、成果物（90 日）に保存する。止めるには `configs/recorder.yaml` の `enabled: false`
  - `monitor.yml`: 毎日 00:47 UTC に残高・評価額・目標の重みを `docs/monitor/daily.jsonl` に記録し、`docs/monitor/index.html`（モニター画面）を作り直してコミットする（`scripts/snapshot.py`、`bbresearch/monitor.py`）。`rebalance.yml` は計画を `docs/monitor/rebalances.jsonl` に追記する
  - `margin.yml`: 信用取引の対応ペアと条件（建玉金利・手数料・保証金率）を `/spot/pairs` から一覧にする（`reports/margin/`）
  - `dist.yml`: 1 時間足のテクニカル指標（373 個。`bbresearch/indicators.py`、部品は `bbresearch/ta.py`）と約定フローの特徴量（34 個。`bbresearch/tradeflow.py`、約定履歴は Actions のキャッシュ）から 4 時間後までの収益率の条件付き分布を分位点回帰フォレストで推定し、無条件・直近 n 本の経験分布と精度を比べ、群ごとの寄与も出す（`scripts/dist_forecast.py`、`configs/dist.yaml`、`reports/dist/`）
  - `vol_daily.yml`: ポートフォリオの JPY 比率を日次で調整する案の前向き検証（BTC 1 時間足のフォレストによる翌日ボラ予測 ÷ 長期平均を推定ボラに掛ける。`scripts/vol_daily.py`、`configs/vol_daily.yaml`、`reports/vol_daily/`）
  - `auth-check.yml`: 認証付き API の疎通確認（参照系のみ）。キーは Secrets `bitbank_API` / `bitbank_secret` から読む
  - `swing.yml`: スイング（4 時間足で判断し 1〜3 日保有）の方向の研究。mode `check`（データの確認。損益は出さない）と `eval`（事前登録した 45 通りを全期間で 1 回。`configs/swing.yaml` の `owner_approved` が必要）。eval は workflow_dispatch だけで動き、`scripts/swing.py` かワークフローの変更の push では check だけが動く（`scripts/swing.py`、`bbresearch/swing.py`、`reports/swing/`）。workflow_dispatch だけのワークフローは既定ブランチにないと API から起動できない（404、2026-09-28 に確認）
- `dashboard/`: TPO・価格帯別出来高のダッシュボード（AWS: API Gateway + Lambda + S3、SAM）。`dashboard/deploy.sh` を AWS CloudShell で実行してデプロイする。Lambda は標準ライブラリだけで書き、計算が `bbresearch/profile.py` と一致することを `tests/test_dashboard.py` で照合している。画面は `dashboard/web/`（React、Vite）で書き、`dashboard/web/build.sh` でビルドして `dashboard/app/page.html`（生成物、コミットする）に置く。左に直近 24 時間・15 分足の TPO・価格帯別出来高（ダーク配色、5 分ごとに更新）、右に 1 分ごとの水準の 60 分の推移（`/api/signals`）。モデルの予測は載せない
- リポジトリは公開（2026-09-27 に非公開へ切り替え、2026-09-28 に公開へ戻した）。ログやレポートに残高・注文の内容・キーを出さない（例外: `docs/monitor/` の記録は金額を残す。公開のまま記録する。2026-09-28 オーナー決定）。
- 上げ下げの予測モデル（`direction*.yml`）は 2026-09-26 に研究を区切った（目的変数 3 通り・ホライズン 2 通りとも費用を超えない。SPEC §1.3）。自動売買には使わない。
- 収益率の分布推定（`dist.yml`、`vol_daily.yml`）は 2026-09-27 に研究を区切った（幅の推定はフォレストが最良だが、向きは指標 373 個にも約定履歴にもなく、幅を日次の JPY 調整に使っても改善しない。SPEC の該当節）。自動売買には使わない。
- 2026-09-28 オーナー決定: 方向の研究をスイング（4 時間足で判断し 1〜3 日保有、JPY の現物ペアすべて）で続ける。仮説はトレンド（時系列・横断モメンタム）と急落後の反発。数値は `configs/swing.yaml` に事前に固定し、全期間で 1 回だけ評価する（SPEC の「スイング」の節）。
- スイングの研究の結果（2026-09-28、run 36341530578、`reports/swing/report.md`）: 事前登録の判定では 3 つとも「支持しない」。時系列モメンタムは 24 通りすべてでアルファが正（t 1.65〜3.28、費用 2 倍でも正、アクティブが正の年 75%）だが、45 試行のデフレートシャープが 0.717（基準 0.95）。横断モメンタムは費用に負ける。急落後の反発は保有率 1% で割引後に届かない。評価に使った数値は変えない（変えた実行は新しい試行として数える）。時系列モメンタムの超過リターンは 2017〜2021 年に集中し、2022〜2026 年は 24 通りの平均で +5.9%（正は 12/24）で、直近 5 年はほぼ効いていない。2026-09-28 オーナー決定で不支持を確定し、前向きのドライランは行わない（研究を区切った）。自動売買には使わない。
- 2026-09-28 オーナー指示で、期待収益率の分布推定の先行研究を調べた（Notion のページ https://app.notion.com/p/3e9036495bee81d2971cfbe20a527d17 と付録 A〜E、SPEC の「分布推定の先行研究の調査」の節）。文献の結論はこれまでの結果と同じ（幅は予測でき、向きは費用に届かない）。次の研究の候補は 3 つ（評価の土台、24 時間の新しい情報源、分布から配分への変換）。
- 次の作業: 先行研究の調査で絞った候補から、次の研究の方向をオーナーが決める。オーナーが `dashboard/deploy.sh` でデプロイし、React 版の画面と `/api/signals` を実機で確認する。Phase 7 は保留。

## コマンド

```bash
pip install -r requirements.txt
python -m pytest -q
python -m bbdata download   --pairs btc_jpy --start 2026-09-15 --end 2026-09-22 --candles 15min
python -m bbdata build-bars --pairs btc_jpy --start 2026-09-15 --end 2026-09-22 --freqs 15min 1h
python -m bbdata validate   --pair  btc_jpy --start 2026-09-15 --end 2026-09-22
python -m bbresearch run --config configs/research.yaml --out reports/research
python -m bbresearch search --config configs/research.yaml --grid configs/search.yaml --out reports/search
python -m bbresearch direction --config configs/research.yaml --spec configs/direction.yaml --out reports/direction
```

## 作業ルール

- SPEC.md の【確定】事項は、オーナーの確認なしに変更しない。
- 【要検証】事項に依存する実装は、検証してから行う。結果は SPEC.md §9 に追記する。
- 【未決】事項は推測で埋めず、オーナーに質問する。
- 時刻は内部ですべて UTC。足の index は開始時刻、区間は左閉右開。
- 判断時刻 `t0` の特徴量・シグナルには `t0` より後のデータを使わない。先読みがないことを確かめるテストを必ず書く。
- 足の集計と特徴量の計算は、研究用と本番用で同じ関数を使う。
- 単体テストはネットワークにアクセスしない（HTTP はモックする）。
- 公開 API に大量のリクエストを送らない。
- 実注文を出すコードは `dry-run` を既定にする。API キー・シークレットはコミットもログ出力もしない。
- フェーズを終えるたびに、テストと `README.md`・`docs/SPEC.md` を更新する。
