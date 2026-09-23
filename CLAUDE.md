# プロジェクト: bitbank 現物トレードエージェント

作業を始める前に `docs/SPEC.md` を読むこと。設計判断・API 仕様の要点・フェーズごとの仕様・未決事項はすべてそこにある。

## 現在の状態

- Phase 1（`bbdata/`）、Phase 0（`scripts/phase0.py`、結果は SPEC.md §9）、Phase 2〜6（`bbresearch/`）は実装済み。
- 実 API へのアクセスは GitHub Actions で行う。開発環境からは行わない。
  - `phase0.yml`: `scripts/phase0.py` の変更を push すると実行し、`reports/phase0/` をコミットする
  - `research.yml`: `configs/research.yaml` の変更を push すると（`run_on_actions: true` のとき）実行し、`reports/research/` をコミットする
  - `auth-check.yml`: 認証付き API の疎通確認（参照系のみ）。キーは SSM から OIDC で読む。リポジトリ変数 `AWS_ROLE_ARN` が必要
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
