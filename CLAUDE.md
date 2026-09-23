# プロジェクト: bitbank 現物トレードエージェント

作業を始める前に `docs/SPEC.md` を読むこと。設計判断・API 仕様の要点・フェーズごとの仕様・未決事項はすべてそこにある。

## 現在の状態

- Phase 1（`bbdata/`: 過去データ取得と足の構築）は実装済み。実 API では未検証。
- 次の作業は Phase 0（実データ検証、SPEC.md §5）。

## コマンド

```bash
pip install -r requirements.txt
python -m pytest -q
python -m bbdata download   --pairs btc_jpy --start 2026-09-15 --end 2026-09-22 --candles 15min
python -m bbdata build-bars --pairs btc_jpy --start 2026-09-15 --end 2026-09-22 --freqs 15min 1h
python -m bbdata validate   --pair  btc_jpy --start 2026-09-15 --end 2026-09-22
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
