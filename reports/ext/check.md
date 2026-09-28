# 候補 2: 外部データの確認（GitHub Actions から届くか、履歴がどこまであるか）

- 実行 2026-09-28T17:56:36+00:00（UTC）。認証なしの公開 API とアーカイブだけ。発注はしない。
- 状態 200 以外（451・403 など）は、その場所（Actions の機械の国）からは使えないことを示す。

| 名前 | 内容 | 状態 | 結果 |
|---|---|---|---|
| binance_spot_klines | Binance 現物 BTCUSDT の 1 時間足（最も古い足） | 451 | {   "code": 0,   "msg": "Service unavailable from a restricted location according to 'b. Eligibility' in https://www.bin |
| binance_data_api_klines | Binance の市場データ専用の窓口（同じ現物の足） | 200 | 最も古い足 2017-08-17 04:00（UTC） |
| binance_fut_funding | Binance 永久先物 BTCUSDT の資金調達率（最も古い値） | 451 | {   "code": 0,   "msg": "Service unavailable from a restricted location according to 'b. Eligibility' in https://www.bin |
| binance_fut_klines | Binance 永久先物の 1 時間足 | 451 | {   "code": 0,   "msg": "Service unavailable from a restricted location according to 'b. Eligibility' in https://www.bin |
| binance_fut_premium | Binance 永久先物のプレミアム指数（先物と現物の価格差）の 1 時間足 | 451 | {   "code": 0,   "msg": "Service unavailable from a restricted location according to 'b. Eligibility' in https://www.bin |
| binance_fut_oi_hist | Binance 永久先物の建玉の履歴（直近の分だけの窓口） | 451 | {   "code": 0,   "msg": "Service unavailable from a restricted location according to 'b. Eligibility' in https://www.bin |
| vision_funding | Binance のアーカイブ: 資金調達率（2019-10 の月次ファイル） | 404 | - |
| vision_um_klines | Binance のアーカイブ: 永久先物の 1 時間足（2019-10） | 404 | - |
| vision_spot_klines | Binance のアーカイブ: 現物の 1 時間足（2019-10） | 200 | 43,434 バイト |
| vision_premium | Binance のアーカイブ: プレミアム指数の 1 時間足（2020-01） | 200 | 16,478 バイト |
| vision_metrics_2020 | Binance のアーカイブ: 建玉などの 5 分ごとの記録（2020-09-01） | 200 | 12,191 バイト |
| vision_metrics_2021 | Binance のアーカイブ: 建玉などの 5 分ごとの記録（2021-12-01） | 200 | 14,792 バイト |
| vision_metrics_2023 | Binance のアーカイブ: 建玉などの 5 分ごとの記録（2023-01-01） | 200 | 14,337 バイト |
| bybit_funding_2020 | Bybit 永久先物 BTCUSDT の資金調達率（2020-04-01 の分） | 403 | {     error:The Amazon CloudFront distribution is configured to block access from your country } |
| bybit_funding_latest | Bybit の資金調達率（最新） | 403 | {     error:The Amazon CloudFront distribution is configured to block access from your country } |
| bybit_oi_2021 | Bybit の建玉の 1 時間ごとの履歴（2021-01-01 の分） | 403 | {     error:The Amazon CloudFront distribution is configured to block access from your country } |
| bybit_kline_2020 | Bybit 永久先物の 1 時間足（2020-04-01 の分） | 403 | {     error:The Amazon CloudFront distribution is configured to block access from your country } |
| bybit_archive | Bybit の約定のアーカイブ（一覧のページ） | 200 | 173,921 バイト |
| dukascopy_hour_2020_01 | Dukascopy のドル円（USDJPY）: 2020 年 1 月の 1 時間足の月次ファイル | 200 | 6,012 バイト、足 744 本、最初の足: ずれ 0 秒、始値 108.631、高値 108.631、安値 108.631、終値 108.631 |
| dukascopy_min_2020_01_02 | Dukascopy のドル円: 2020-01-02 の 1 分足の日次ファイル | 503 | <html><body><h1>503 Service Unavailable</h1> No server is available to handle this request. </body></html>  |
| fred_dexjpus | FRED のドル円（DEXJPUS、日次。ニューヨーク正午の値） | 200 | 14,535 行、最初 1971-01-04,357.73、最後 2026-09-18,156.87 |
| vision_funding_2019-09 | Binance のアーカイブ: 資金調達率（2019-09） | 404 | - |
| vision_funding_2019-12 | Binance のアーカイブ: 資金調達率（2019-12） | 404 | - |
| vision_funding_2020-01 | Binance のアーカイブ: 資金調達率（2020-01） | 200 | 825 バイト |
| vision_funding_2020-06 | Binance のアーカイブ: 資金調達率（2020-06） | 200 | 791 バイト |
| vision_funding_2021-01 | Binance のアーカイブ: 資金調達率（2021-01） | 200 | 981 バイト |
| vision_funding_2022-01 | Binance のアーカイブ: 資金調達率（2022-01） | 200 | 906 バイト |
| vision_funding_2024-01 | Binance のアーカイブ: 資金調達率（2024-01） | 200 | 696 バイト |
| vision_um_klines_2019-12 | Binance のアーカイブ: 永久先物の 1 時間足（2019-12） | 404 | - |
| vision_um_klines_2020-01 | Binance のアーカイブ: 永久先物の 1 時間足（2020-01） | 200 | 37,271 バイト |
| vision_premium_2019-09 | Binance のアーカイブ: プレミアム指数の 1 時間足（2019-09） | 404 | - |
| vision_premium_2019-12 | Binance のアーカイブ: プレミアム指数の 1 時間足（2019-12） | 404 | - |
| deribit_funding | Deribit の資金調達率（届くかだけ） | 200 | 255 バイト |
| okx_funding | OKX の資金調達率（届くかだけ） | 200 | 227 バイト |
| kraken_futures_funding | Kraken Futures の資金調達率（届くかだけ） | 200 | 1,020,357 バイト |
| bitmex_funding | BitMEX の資金調達率（届くかだけ） | 200 | 182 バイト |
| bitmex_funding_2019 | BitMEX XBTUSD の資金調達率（2019-06-01 以降の最初の 3 件） | 200 | 3 件、最初 2019-06-01T04:00、率 0.000116、間隔 2000-01-01T08:00:00.000Z |
| bitmex_funding_2020 | BitMEX XBTUSD の資金調達率（2020-01-01 以降の最初の 3 件） | 200 | 3 件、最初 2020-01-01T04:00、率 0.0001、間隔 2000-01-01T08:00:00.000Z |
| deribit_funding_2019 | Deribit BTC-PERPETUAL の資金調達率（2019-06-01 の 3 時間） | 200 | 3 件、最初 2019-06-01 01:00、8 時間の率 0.0004371521792057977、1 時間の率 8.142461913047441e-05 |
| deribit_funding_2020 | Deribit BTC-PERPETUAL の資金調達率（2020-01-01 の 3 時間） | 200 | 3 件、最初 2020-01-01 01:00、8 時間の率 0.00014033842756214416、1 時間の率 8.649043447829139e-07 |
