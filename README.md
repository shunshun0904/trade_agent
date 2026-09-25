# bitbank 現物トレードエージェント（研究段階）

- `bbdata/`: 過去データ取得と足・約定集計（Phase 1）
- `scripts/phase0.py`: 実データ検証（Phase 0、結果は `docs/SPEC.md` §9）
- `bbresearch/`: CUSUM → 約定考慮トリプルバリア → 特徴量 → メタモデル → バックテスト（Phase 2〜6）

実 API へのアクセスは GitHub Actions で行う（`.github/workflows/phase0.yml`、`research.yml`）。

## 研究パイプライン（bbresearch）

```bash
python -m bbresearch run --config configs/research.yaml --out reports/research
```

設定は `configs/research.yaml`。`run_on_actions: true` の状態でこのファイルを変更して push すると、
GitHub Actions が約定履歴を取得してパイプラインを実行し、`reports/research/report.md` をコミットする。
試行ごとの結果は `reports/experiments.jsonl` に追記され、Deflated Sharpe Ratio の試行数に数えられる。

パラメータ探索（SPEC §7 の範囲、開発期間のみ）:

```bash
python -m bbresearch search --config configs/research.yaml --grid configs/search.yaml --out reports/search
```

`configs/search.yaml` を変更して push すると GitHub Actions（`search.yml`）で実行し、`reports/search/`
（`report.md`、`screen.csv`、`selected.yaml`）をコミットする。

### 15 分後の上げ下げの予測（二段構え）

```bash
python -m bbresearch direction --config configs/research.yaml --spec configs/direction.yaml --out reports/direction
```

モデル B が 1 分ごとに、モデル A が 15 分ごとに「15 分後の価格が今より高いか」を予測する。A には直近 15 分の
B の予測を特徴量として加える（A+B）。特徴量は 1 分足の流れ・15 分足・価格帯別出来高と TPO・時刻。
2024 年より前で学習し、2024 年以降で1回だけ評価して A 単独と A+B を比べる。`configs/direction.yaml` を変更して
push すると GitHub Actions（`direction.yml`）で実行し、`reports/direction/` をコミットする。

# bbdata — bitbank 過去データ取得と足・約定集計

bitbank の公開API（認証不要）から約定履歴を日付単位で取得して Parquet に保存し、
そこから時間足と約定フローの集計を作ります。売買ロジック（CUSUM 一次シグナル →
約定考慮トリプルバリア → メタモデル）の入力データを用意する段階です。

## セットアップ

```bash
pip install -r requirements.txt
python -m pytest -q        # HTTP はモック、ネットワークにアクセスしない
```

## 使い方

日付はすべて UTC です。`--start` は含み、`--end` は含みません。

```bash
# 1. 約定履歴を取得（検証用に公式15分足も取得）
python -m bbdata download --pairs btc_jpy --start 2025-01-01 --end 2026-09-24 --candles 15min

# 2. 足を作る（15分足と1時間足）
python -m bbdata build-bars --pairs btc_jpy --start 2025-01-01 --end 2026-09-24 \
    --freqs 15min 1h --large-trade-amount 0.1

# 3. 公式ロウソク足と突き合わせる（小さい期間で先に実行を推奨）
python -m bbdata validate --pair btc_jpy --start 2026-09-01 --end 2026-09-08
```

保存先:

```
data/raw/transactions/{pair}/{YYYYMMDD}.parquet   約定の生データ（1日1ファイル）
data/raw/candles/{pair}/{type}/{period}.parquet   公式ロウソク足（検証用）
data/bars/{pair}/{freq}.parquet                   集計済みの足
```

足の列の定義は `bbdata/bars.py` の冒頭に書いています。

## 設計上の判断

- **生の約定を保存する**: 次の段階の約定判定（指値が約定したか）は約定単位の時刻で行う必要があるため、足だけでなく生データを残します。
- **前後1日を余分に取得**: `transactions/{YYYYMMDD}` の日付境界のタイムゾーンは公式ドキュメントに記載がありません。前後1日を取得し、`transaction_id` で重複を除いてから `executed_at`（UnixTime ミリ秒）で切り出すので、境界がどのタイムゾーンでも結果は変わりません。
- **直近2日は毎回取り直す**: 当日分はまだ確定していない可能性があるためです。それより古い日付は、ファイルがあればスキップします（`--overwrite` で強制取得）。
- **約定なしの足**: close は直前の足から前方埋めし（過去方向のみ）、OHLC と vwap も close にそろえます。`max_buy_price` / `min_sell_price` は NaN のままです（その足では指値が約定しなかった、という意味を保つため）。
- **大口約定の閾値は固定値**: 全期間の分位点で決めると先読みになるため、数量の固定値で指定します。
- **レート制限**: 公開APIのレート制限は公式ドキュメントに数値の記載がありません。既定では 0.3 秒間隔でリクエストし、429 と 5xx は指数バックオフで最大5回再試行します。

## 実データで確認が必要だった点

以下は Phase 0 で確認済み（結果は `docs/SPEC.md` §9）。side はテイカー側、公式足の timestamp は開始時刻、
最古日は 2017-02-14、日付境界は UTC。

1. **`side` の意味**: テイカー側の売買方向という前提で `min_sell_price`（売りテイカーが買い板に当たった最安値）を作っています。約定と板のスナップショットを同時に記録し、`buy` の約定価格が売り気配付近にあるかを見れば確認できます。
2. **公式ロウソク足の timestamp**: 足の開始時刻か終了時刻かが不明です。`validate` の出力で `ohlc_match_rate` が最も高い `shift_bars` を見れば分かります（0 なら開始時刻、-1 なら終了時刻の表記）。
3. **過去データの遡れる範囲**: 何年前まで取得できるかはドキュメントに記載がありません。取得できない日付は警告を出してスキップします。
4. **出来高の一致**: 公式足と自前集計の出来高が一致しない場合、約定履歴の欠けか集計定義の違いが考えられます。
