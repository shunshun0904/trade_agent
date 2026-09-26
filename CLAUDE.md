# プロジェクト: bitbank 現物トレードエージェント

作業を始める前に `docs/SPEC.md` を読むこと。設計判断・API 仕様の要点・フェーズごとの仕様・未決事項はすべてそこにある。

## 現在の状態

- Phase 1（`bbdata/`）、Phase 0（`scripts/phase0.py`、結果は SPEC.md §9）、Phase 2〜6（`bbresearch/`）は実装済み。
- 実 API へのアクセスは GitHub Actions で行う。開発環境からは行わない。
  - どのワークフローも、API の workflow_dispatch でこのブランチを ref に指定して実行できる（既定ブランチになくてもよいことを 2026-09-23 に確認）
  - `phase0.yml`: `scripts/phase0.py` の変更を push すると実行し、`reports/phase0/` をコミットする
  - `research.yml`: `configs/research.yaml` の変更を push すると（`run_on_actions: true` のとき）実行し、`reports/research/` をコミットする
  - `search.yml`: `configs/search.yaml` の変更を push すると（`run_on_actions: true` のとき）パラメータ探索を実行し、`reports/search/` をコミットする（開発期間のみ。ホールドアウトは評価しない）
  - `diagnose.yml`: 損失の要因分解・ホライズン・エントリー方法の比較（`configs/diagnose.yaml`）
  - `maker_search.yml`: 深い指値・短い保有・売りも指値の決済条件の探索と確認期間での評価（`configs/maker_search.yaml`）
  - `val_rule.yml`: ルール A（VAL での反発）を固定した数値で全期間1回評価（`configs/val_rule.yaml`）
  - `direction.yml`: 15 分後の上げ下げを予測する二段構えのモデル（A: 15 分ごと、B: 1 分ごと）を 2024 年より前で学習し、2024 年以降で1回評価（`configs/direction.yaml`）
  - `direction_1h.yml`: 同じモデルの 1 時間版。毎正時に判断し、1 分ごとの TPO×価格帯別出来高のシグナルの直近 60 分の推移を加える（`configs/direction_1h.yaml`）
  - `direction_cost.yml`: 目的変数を「成行往復の費用（約 0.3%）を超えて上がるか」にした版を 15 分と 1 時間で実行（`configs/direction_cost.yaml`、`direction_1h_cost.yaml`）
  - `recorder.yml`: 板・約定の記録。約 5 時間 45 分ごとに次のジョブを自分で起動して連続させ、成果物（90 日）に保存する。止めるには `configs/recorder.yaml` の `enabled: false`
  - `auth-check.yml`: 認証付き API の疎通確認（参照系のみ）。キーは Secrets `bitbank_API` / `bitbank_secret` から読む
- `dashboard/`: TPO・価格帯別出来高のダッシュボード（AWS: API Gateway + Lambda + S3、SAM）。`dashboard/deploy.sh` を AWS CloudShell で実行してデプロイする。Lambda は標準ライブラリだけで書き、計算が `bbresearch/profile.py` と一致することを `tests/test_dashboard.py` で照合している。`dashboard/web/` は React 版の画面（Vite）。TPO・価格帯別出来高に 1 分ごとのシグナルの 60 分の推移と毎正時の判断を並べる。`/api/signals` は未実装（形は `web/src/api.js`）
- リポジトリは公開。ログやコミットするレポートに残高・注文の内容・キーを出さない。
- 次の作業: Phase 6 の結果をオーナーが確認する。Phase 7 はその後。

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
