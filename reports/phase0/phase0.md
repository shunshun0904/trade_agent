# Phase 0 検証結果（2026-09-23T15:55:22+00:00）

pair: `btc_jpy`

## ペア仕様（/spot/pairs, /spot/status）

```json
{
  "pair_spec": {
    "name": "btc_jpy",
    "base_asset": "btc",
    "quote_asset": "jpy",
    "maker_fee_rate_base": "0",
    "taker_fee_rate_base": "0",
    "maker_fee_rate_quote": "0",
    "taker_fee_rate_quote": "0.001",
    "margin_open_maker_fee_rate_quote": "0",
    "margin_open_taker_fee_rate_quote": "0.001",
    "margin_close_maker_fee_rate_quote": "0",
    "margin_close_taker_fee_rate_quote": "0.001",
    "margin_long_interest": "0.0004",
    "margin_short_interest": "0.0004",
    "margin_current_individual_ratio": "0.5",
    "margin_current_individual_until": 1790693999999,
    "margin_current_company_ratio": "0.0818",
    "margin_current_company_until": 1790693999999,
    "margin_next_individual_ratio": "0.5",
    "margin_next_individual_until": 1791298799999,
    "margin_next_company_ratio": "0.0818",
    "margin_next_company_until": 1791298799999,
    "unit_amount": "0.0001",
    "limit_max_amount": "1000",
    "market_max_amount": "10",
    "market_allowance_rate": "0.2",
    "market_in_cb_allowance_rate": "0.2",
    "price_digits": 0,
    "amount_digits": 4,
    "is_enabled": true,
    "stop_order": false,
    "stop_order_and_cancel": false,
    "stop_market_order": false,
    "stop_stop_order": false,
    "stop_stop_limit_order": false,
    "stop_margin_long_order": false,
    "stop_margin_short_order": false,
    "stop_buy_order": false,
    "stop_sell_order": false
  },
  "spot_status": {
    "pair": "btc_jpy",
    "status": "NORMAL",
    "min_amount": "0.0001"
  }
}
```

## V1 公式ロウソク足との一致（2026-09-15 〜 2026-09-22、15min）

自前の足 672 本（約定なし 2 本）、公式 672 本。 ohlc_match_rate 最大の shift_bars = **0**


| shift_bars | n | close_match_rate | ohlc_match_rate | volume_rel_err_median | volume_rel_err_p99 |
|---|---|---|---|---|---|
| -1 | 669 | 0.0044843 | 0 | 0.716434 | 32.2848 |
| 0 | 670 | 1 | 1 | 0 | 2.06086e-16 |
| 1 | 669 | 0.00298954 | 0 | 0.719424 | 59.2485 |

## V5 日付境界

推定: **UTC**


| date | n | first_utc | last_utc | first_minus_day0_h | day0_plus_24h_minus_last_h |
|---|---|---|---|---|---|
| 2026-09-15 | 9074 | 2026-09-15T00:00:01.337000+00:00 | 2026-09-15T23:58:50.617000+00:00 | 0 | 0.019 |
| 2026-09-16 | 7255 | 2026-09-16T00:00:07.107000+00:00 | 2026-09-16T23:58:45.950000+00:00 | 0.002 | 0.021 |
| 2026-09-17 | 4631 | 2026-09-17T00:00:07.186000+00:00 | 2026-09-17T23:57:30.289000+00:00 | 0.002 | 0.042 |
| 2026-09-18 | 11216 | 2026-09-18T00:00:02.999000+00:00 | 2026-09-18T23:59:52.276000+00:00 | 0.001 | 0.002 |
| 2026-09-19 | 4621 | 2026-09-19T00:00:06.052000+00:00 | 2026-09-19T23:58:34.669000+00:00 | 0.002 | 0.024 |
| 2026-09-20 | 4461 | 2026-09-20T00:00:08.740000+00:00 | 2026-09-20T23:56:06.561000+00:00 | 0.002 | 0.065 |
| 2026-09-21 | 15222 | 2026-09-21T00:00:01.996000+00:00 | 2026-09-21T23:59:36.489000+00:00 | 0.001 | 0.007 |

## V2 約定の side

板 182 回、約定 112 件。depth のキー: ['ask_market', 'asks', 'asks_over', 'asks_under', 'bid_market', 'bids', 'bids_over', 'bids_under', 'sequenceId', 'timestamp']


age_ms は約定時刻と直前の板の timestamp の差。


| max_age_ms | n_buy | buy_ge_prev_ask | buy_gt_prev_mid | n_sell | sell_le_prev_bid | sell_lt_prev_mid |
|---|---|---|---|---|---|---|
| 500 | 2 | 1 | 1 | 23 | 1 | 1 |
| 1000 | 18 | 1 | 1 | 24 | 1 | 1 |
| 2000 | 23 | 0.913043 | 1 | 29 | 0.896552 | 0.896552 |
| 5000 | 23 | 0.913043 | 1 | 29 | 0.896552 | 0.896552 |

## V3 最古の約定日

最古: **2017-02-14**

```
{"oldest": "2017-02-14", "probes": [["2020-05-11", true], ["2017-03-07", true], ["2015-08-04", false], ["2016-05-20", false], ["2016-10-12", false], ["2016-12-24", false], ["2017-01-29", false], ["2017-02-16", true], ["2017-02-07", false], ["2017-02-11", false], ["2017-02-13", false], ["2017-02-14", true]], "after_oldest_check": [["2017-02-15", true], ["2017-02-21", true], ["2017-03-16", true]]}
```

## V4 Public REST の応答ステータス


```
{"200": 391, "404": 8}
```
