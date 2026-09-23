# 研究パイプライン結果（2026-09-23T20:48:04+00:00）

- pair: `btc_jpy`、期間 2017-02-14 〜 2026-09-22（ホールドアウト 180 日）
- 足 336672 本（約定なし 4237）、約定 72488969 件、config_hash `e5a957e68ec9`
- 手数料率: メイカー 0.0、テイカー 0.001
- ホールドアウトは評価していない（使用済み。`split.evaluate_holdout: false`）
- 価格帯別出来高・TPO の特徴量: あり（直近 24 時間、刻み 0.25σ）
- 試行数（DSR 用、過去の実験ログを含む）: 1668

## イベントとラベル

| signal | period | events | fill_rate | y_rate | mean_ret_net | exit_types |
|---|---|---|---|---|---|---|
| dip | dev | 9200 | 0.594 | 0.512 | -0.00102 | {'tp': 2306, 'sl': 1916, 'time': 1241} |
| breakout | dev | 9899 | 0.563 | 0.456 | -0.00187 | {'tp': 2022, 'time': 2020, 'sl': 1533} |

## メタモデル（OOF / ホールドアウト）

| signal | model | n_train | base_rate | OOF logloss | OOF AUC | OOF brier | HO n | HO AUC | HO logloss |
|---|---|---|---|---|---|---|---|---|---|
| dip | logit | 5463 | 0.512 | 0.690 | 0.563 | 0.248 | 0 | - | - |
| dip | lgbm | 5463 | 0.512 | 0.695 | 0.548 | 0.251 | 0 | - | - |
| breakout | logit | 5575 | 0.456 | 0.693 | 0.530 | 0.249 | 0 | - | - |
| breakout | lgbm | 5575 | 0.456 | 0.694 | 0.533 | 0.250 | 0 | - | - |

## バックテスト（dev）

| strategy | orders | trades | fill_rate | total_return | max_dd | sharpe | win_rate | avg_ret_net | time_in_mkt | maker_fee | taker_fee | DSR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dip/primary | 8837 | 5218 | 0.590 | -0.9977 | -0.9979 | -1.750 | 0.509 | -0.00111 | 0.127 | 0 | 642257 | 0.000 |
| dip/threshold_0.5/logit | 4917 | 3045 | 0.619 | -0.7771 | -0.8387 | -0.375 | 0.548 | -0.00036 | 0.073 | 0 | 1072965 | 0.000 |
| dip/threshold_0.5/lgbm | 4866 | 2987 | 0.614 | -0.7502 | -0.8336 | -0.350 | 0.542 | -0.00033 | 0.072 | 0 | 1205102 | 0.000 |
| dip/threshold_0.55/logit | 2773 | 1761 | 0.635 | 1.2546 | -0.3789 | 0.458 | 0.578 | 0.00064 | 0.041 | 0 | 1725615 | 0.000 |
| dip/threshold_0.55/lgbm | 3009 | 1864 | 0.619 | -0.5264 | -0.6054 | -0.176 | 0.545 | -0.00024 | 0.043 | 0 | 800884 | 0.000 |
| dip/threshold_0.6/logit | 1206 | 776 | 0.643 | 1.8301 | -0.4193 | 0.610 | 0.595 | 0.00162 | 0.017 | 0 | 561294 | 0.000 |
| dip/threshold_0.6/lgbm | 1506 | 939 | 0.624 | 0.4866 | -0.4372 | 0.314 | 0.559 | 0.00062 | 0.021 | 0 | 725764 | 0.000 |
| dip/meta/logit | 4904 | 3040 | 0.620 | 0.5281 | -0.1627 | 0.523 | 0.548 | -0.00036 | 0.073 | 0 | 242703 | 0.000 |
| dip/meta/lgbm | 4853 | 2979 | 0.614 | 0.1170 | -0.1107 | 0.211 | 0.542 | -0.00032 | 0.071 | 0 | 225598 | 0.000 |
| breakout/primary | 6393 | 3883 | 0.607 | -0.9996 | -0.9996 | -2.954 | 0.457 | -0.00206 | 0.119 | 0 | 152372 | 0.000 |
| breakout/threshold_0.5/logit | 2301 | 1433 | 0.623 | -0.9076 | -0.9175 | -1.149 | 0.491 | -0.00151 | 0.038 | 0 | 195704 | 0.000 |
| breakout/threshold_0.5/lgbm | 2714 | 1600 | 0.590 | -0.8715 | -0.8784 | -1.186 | 0.487 | -0.00117 | 0.047 | 0 | 355912 | 0.000 |
| breakout/threshold_0.55/logit | 749 | 485 | 0.648 | -0.6961 | -0.7076 | -0.729 | 0.501 | -0.00217 | 0.012 | 0 | 125432 | 0.000 |
| breakout/threshold_0.55/lgbm | 1206 | 712 | 0.590 | -0.5545 | -0.6020 | -0.604 | 0.504 | -0.00099 | 0.019 | 0 | 263442 | 0.000 |
| breakout/threshold_0.6/logit | 285 | 183 | 0.642 | -0.4424 | -0.4886 | -0.405 | 0.508 | -0.00261 | 0.004 | 0 | 76257 | 0.000 |
| breakout/threshold_0.6/lgbm | 430 | 231 | 0.537 | -0.5156 | -0.5302 | -0.875 | 0.472 | -0.00294 | 0.006 | 0 | 90565 | 0.000 |
| breakout/meta/logit | 2284 | 1426 | 0.624 | -0.1539 | -0.1947 | -0.241 | 0.489 | -0.00153 | 0.038 | 0 | 65420 | 0.000 |
| breakout/meta/lgbm | 2701 | 1593 | 0.590 | -0.1630 | -0.1836 | -0.852 | 0.487 | -0.00118 | 0.046 | 0 | 72207 | 0.000 |

exit_type 別:

- dip/primary: sl n=1843 pnl=-7856957 avg_ret=-0.01558, time n=1187 pnl=-346034 avg_ret=-0.00137, tp n=2188 pnl=7205306 avg_ret=0.01123
- dip/threshold_0.5/logit: sl n=989 pnl=-12580376 avg_ret=-0.01803, time n=671 pnl=-568574 avg_ret=-0.00133, tp n=1385 pnl=12371813 avg_ret=0.01273
- dip/threshold_0.5/lgbm: sl n=980 pnl=-14027016 avg_ret=-0.01785, time n=643 pnl=-619854 avg_ret=-0.00133, tp n=1364 pnl=13896676 avg_ret=0.01273
- dip/threshold_0.55/logit: sl n=548 pnl=-19343600 avg_ret=-0.02027, time n=359 pnl=-730131 avg_ret=-0.00098, tp n=854 pnl=21328323 avg_ret=0.01475
- dip/threshold_0.55/lgbm: sl n=608 pnl=-9865355 avg_ret=-0.01965, time n=395 pnl=-347764 avg_ret=-0.00109, tp n=861 pnl=9686722 avg_ret=0.01387
- dip/threshold_0.6/logit: sl n=231 pnl=-7539602 avg_ret=-0.02555, time n=145 pnl=-236391 avg_ret=-0.00080, tp n=400 pnl=9606127 avg_ret=0.01819
- dip/threshold_0.6/lgbm: sl n=295 pnl=-8857277 avg_ret=-0.02134, time n=195 pnl=-386528 avg_ret=-0.00131, tp n=449 pnl=9730443 avg_ret=0.01590
- dip/meta/logit: sl n=987 pnl=-3170083 avg_ret=-0.01804, time n=670 pnl=-82762 avg_ret=-0.00134, tp n=1383 pnl=3780926 avg_ret=0.01274
- dip/meta/lgbm: sl n=976 pnl=-2784228 avg_ret=-0.01786, time n=643 pnl=-106374 avg_ret=-0.00133, tp n=1360 pnl=3007636 avg_ret=0.01274
- breakout/primary: sl n=1040 pnl=-2137734 avg_ret=-0.01874, time n=1436 pnl=-384363 avg_ret=-0.00366, tp n=1407 pnl=1522467 avg_ret=0.01191
- breakout/threshold_0.5/logit: sl n=421 pnl=-2504777 avg_ret=-0.01866, time n=416 pnl=-353736 avg_ret=-0.00332, tp n=596 pnl=1950930 avg_ret=0.01186
- breakout/threshold_0.5/lgbm: sl n=430 pnl=-3187836 avg_ret=-0.01670, time n=535 pnl=-765332 avg_ret=-0.00326, tp n=635 pnl=3081678 avg_ret=0.01111
- breakout/threshold_0.55/logit: sl n=151 pnl=-2031056 avg_ret=-0.02532, time n=127 pnl=-218887 avg_ret=-0.00361, tp n=207 pnl=1553823 avg_ret=0.01561
- breakout/threshold_0.55/lgbm: sl n=195 pnl=-2372936 avg_ret=-0.01706, time n=219 pnl=-554634 avg_ret=-0.00363, tp n=298 pnl=2373029 avg_ret=0.01145
- breakout/threshold_0.6/logit: sl n=62 pnl=-1745215 avg_ret=-0.03717, time n=44 pnl=-65106 avg_ret=-0.00168, tp n=77 pnl=1367876 avg_ret=0.02470
- breakout/threshold_0.6/lgbm: sl n=74 pnl=-982129 avg_ret=-0.01845, time n=66 pnl=-152601 avg_ret=-0.00346, tp n=91 pnl=619141 avg_ret=0.01003
- breakout/meta/logit: sl n=419 pnl=-1121898 avg_ret=-0.01871, time n=416 pnl=-49396 avg_ret=-0.00332, tp n=591 pnl=1017425 avg_ret=0.01191
- breakout/meta/lgbm: sl n=429 pnl=-578491 avg_ret=-0.01671, time n=533 pnl=-137666 avg_ret=-0.00327, tp n=631 pnl=553169 avg_ret=0.01115

比較対象:

- dip/random: n_candidates=8837, n_orders_mean=7840.6500, total_return_mean=-0.9951, total_return_p05=-0.9985, total_return_p95=-0.9907
- breakout/random: n_candidates=6393, n_orders_mean=5856.5000, total_return_mean=-0.9753, total_return_p05=-0.9944, total_return_p95=-0.9362
- buy_and_hold: total_return=99.8135, sharpe=1.0697, max_drawdown=-0.8513
