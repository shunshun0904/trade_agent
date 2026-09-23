# 研究パイプライン結果（2026-09-23T17:46:09+00:00）

- pair: `btc_jpy`、期間 2017-02-14 〜 2026-09-22（ホールドアウト 180 日）
- 足 336672 本（約定なし 4237）、約定 72488969 件、config_hash `f7788a7e6e52`
- 手数料率: メイカー 0.0、テイカー 0.001
- 試行数（DSR 用、過去の実験ログを含む）: 18

## イベントとラベル

| signal | period | events | fill_rate | y_rate | mean_ret_net | exit_types |
|---|---|---|---|---|---|---|
| dip | dev | 23979 | 0.820 | 0.425 | -0.00135 | {'sl': 8718, 'tp': 6934, 'time': 4017} |
| dip | holdout | 1381 | 0.763 | 0.361 | -0.00160 | {'sl': 500, 'tp': 324, 'time': 230} |
| breakout | dev | 25239 | 0.818 | 0.389 | -0.00185 | {'sl': 8567, 'tp': 6791, 'time': 5293} |
| breakout | holdout | 1419 | 0.766 | 0.371 | -0.00140 | {'sl': 453, 'tp': 367, 'time': 267} |

## メタモデル（OOF / ホールドアウト）

| signal | model | n_train | base_rate | OOF logloss | OOF AUC | OOF brier | HO n | HO AUC | HO logloss |
|---|---|---|---|---|---|---|---|---|---|
| dip | logit | 19669 | 0.425 | 0.681 | 0.535 | 0.244 | 1054 | 0.568 | 0.654 |
| dip | lgbm | 19669 | 0.425 | 0.679 | 0.549 | 0.243 | 1054 | 0.543 | 0.652 |
| breakout | logit | 20650 | 0.389 | 0.670 | 0.529 | 0.238 | 1087 | 0.547 | 0.657 |
| breakout | lgbm | 20650 | 0.389 | 0.666 | 0.546 | 0.237 | 1087 | 0.550 | 0.658 |

## バックテスト（dev）

| strategy | orders | trades | fill_rate | total_return | max_dd | sharpe | win_rate | avg_ret_net | time_in_mkt | maker_fee | taker_fee | DSR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dip/primary | 8799 | 6971 | 0.792 | -0.9998 | -0.9998 | -3.124 | 0.451 | -0.00124 | 0.079 | 0 | 247934 | 0.000 |
| dip/threshold_0.5/logit | 1516 | 1220 | 0.805 | -0.6652 | -0.8103 | -0.397 | 0.513 | -0.00069 | 0.011 | 0 | 264887 | 0.000 |
| dip/threshold_0.5/lgbm | 2696 | 2215 | 0.822 | -0.8764 | -0.8771 | -0.884 | 0.492 | -0.00083 | 0.022 | 0 | 443160 | 0.000 |
| dip/threshold_0.55/logit | 660 | 524 | 0.794 | -0.1638 | -0.6553 | -0.000 | 0.531 | -0.00001 | 0.005 | 0 | 165679 | 0.000 |
| dip/threshold_0.55/lgbm | 728 | 591 | 0.812 | -0.0108 | -0.3283 | 0.079 | 0.531 | 0.00020 | 0.005 | 0 | 299516 | 0.000 |
| dip/threshold_0.6/logit | 343 | 273 | 0.796 | -0.2098 | -0.6180 | -0.065 | 0.516 | -0.00041 | 0.003 | 0 | 100191 | 0.000 |
| dip/threshold_0.6/lgbm | 204 | 161 | 0.789 | 0.0739 | -0.3278 | 0.126 | 0.553 | 0.00079 | 0.001 | 0 | 76238 | 0.000 |
| dip/meta/logit | 1511 | 1215 | 0.804 | -0.0804 | -0.3141 | -0.086 | 0.514 | -0.00067 | 0.011 | 0 | 63991 | 0.000 |
| dip/meta/lgbm | 2675 | 2196 | 0.821 | -0.0589 | -0.1178 | -0.215 | 0.492 | -0.00081 | 0.022 | 0 | 69765 | 0.000 |
| breakout/primary | 4495 | 3269 | 0.727 | -1.0000 | -1.0000 | -4.002 | 0.387 | -0.00329 | 0.036 | 0 | 119436 | 0.000 |
| breakout/threshold_0.5/logit | 901 | 684 | 0.759 | -0.9569 | -0.9605 | -1.573 | 0.420 | -0.00437 | 0.006 | 0 | 85767 | 0.000 |
| breakout/threshold_0.5/lgbm | 772 | 678 | 0.878 | -0.7675 | -0.7687 | -1.556 | 0.450 | -0.00208 | 0.006 | 0 | 165008 | 0.000 |
| breakout/threshold_0.55/logit | 417 | 305 | 0.731 | -0.8062 | -0.8348 | -0.903 | 0.410 | -0.00503 | 0.003 | 0 | 70847 | 0.000 |
| breakout/threshold_0.55/lgbm | 124 | 112 | 0.903 | -0.1163 | -0.2013 | -0.282 | 0.536 | -0.00103 | 0.001 | 0 | 50825 | 0.000 |
| breakout/threshold_0.6/logit | 231 | 170 | 0.736 | -0.6790 | -0.6802 | -0.819 | 0.394 | -0.00621 | 0.002 | 0 | 58504 | 0.000 |
| breakout/threshold_0.6/lgbm | 10 | 10 | 1.000 | 0.0109 | -0.0372 | 0.077 | 0.600 | 0.00122 | 0.000 | 0 | 4082 | 0.000 |
| breakout/meta/logit | 898 | 682 | 0.759 | -0.2976 | -0.3666 | -0.543 | 0.418 | -0.00439 | 0.006 | 0 | 40669 | 0.000 |
| breakout/meta/lgbm | 764 | 672 | 0.880 | -0.0530 | -0.0541 | -0.879 | 0.451 | -0.00208 | 0.006 | 0 | 15902 | 0.000 |

exit_type 別:

- dip/primary: sl n=2960 pnl=-2881029 avg_ret=-0.00971, time n=1408 pnl=-44218 avg_ret=-0.00061, tp n=2603 pnl=1925418 avg_ret=0.00805
- dip/threshold_0.5/logit: sl n=510 pnl=-4344992 avg_ret=-0.02071, time n=173 pnl=-52302 avg_ret=-0.00032, tp n=537 pnl=3732131 avg_ret=0.01821
- dip/threshold_0.5/lgbm: sl n=946 pnl=-5321105 avg_ret=-0.01447, time n=372 pnl=-32466 avg_ret=-0.00012, tp n=897 pnl=4477220 avg_ret=0.01326
- dip/threshold_0.55/logit: sl n=216 pnl=-3271657 avg_ret=-0.02590, time n=71 pnl=52115 avg_ret=0.00157, tp n=237 pnl=3055695 avg_ret=0.02311
- dip/threshold_0.55/lgbm: sl n=234 pnl=-4204828 avg_ret=-0.02019, time n=90 pnl=-10388 avg_ret=0.00013, tp n=267 pnl=4204445 avg_ret=0.01809
- dip/threshold_0.6/logit: sl n=116 pnl=-2195289 avg_ret=-0.03034, time n=45 pnl=85466 avg_ret=0.00358, tp n=112 pnl=1900018 avg_ret=0.02900
- dip/threshold_0.6/lgbm: sl n=62 pnl=-1493913 avg_ret=-0.02683, time n=21 pnl=-9596 avg_ret=-0.00070, tp n=78 pnl=1577396 avg_ret=0.02316
- dip/meta/logit: sl n=507 pnl=-1425073 avg_ret=-0.02076, time n=172 pnl=9459 avg_ret=-0.00032, tp n=536 pnl=1335171 avg_ret=0.01822
- dip/meta/lgbm: sl n=936 pnl=-948686 avg_ret=-0.01450, time n=370 pnl=-3799 avg_ret=-0.00011, tp n=890 pnl=893587 avg_ret=0.01330
- breakout/primary: sl n=1537 pnl=-1225051 avg_ret=-0.01306, time n=611 pnl=-98639 avg_ret=-0.00247, tp n=1121 pnl=323727 avg_ret=0.00967
- breakout/threshold_0.5/logit: sl n=343 pnl=-1486000 avg_ret=-0.02013, time n=92 pnl=-56351 avg_ret=-0.00150, tp n=249 pnl=585449 avg_ret=0.01628
- breakout/threshold_0.5/lgbm: sl n=311 pnl=-1710511 avg_ret=-0.01141, time n=89 pnl=-56983 avg_ret=-0.00173, tp n=278 pnl=1000031 avg_ret=0.00824
- breakout/threshold_0.55/logit: sl n=155 pnl=-1453502 avg_ret=-0.02508, time n=44 pnl=-25446 avg_ret=-0.00068, tp n=106 pnl=672756 avg_ret=0.02248
- breakout/threshold_0.55/lgbm: sl n=46 pnl=-522731 avg_ret=-0.01178, time n=8 pnl=-19005 avg_ret=-0.00246, tp n=58 pnl=425483 avg_ret=0.00768
- breakout/threshold_0.6/logit: sl n=88 pnl=-1357865 avg_ret=-0.02922, time n=27 pnl=-8528 avg_ret=0.00014, tp n=55 pnl=687419 avg_ret=0.02749
- breakout/threshold_0.6/lgbm: sl n=4 pnl=-56013 avg_ret=-0.01347, tp n=6 pnl=66931 avg_ret=0.01102
- breakout/meta/logit: sl n=343 pnl=-872452 avg_ret=-0.02013, time n=92 pnl=-2726 avg_ret=-0.00150, tp n=247 pnl=577596 avg_ret=0.01638
- breakout/meta/lgbm: sl n=307 pnl=-155287 avg_ret=-0.01147, time n=89 pnl=-6498 avg_ret=-0.00173, tp n=276 pnl=108753 avg_ret=0.00824

比較対象:

- dip/random: n_candidates=8799, n_orders_mean=5899.9500, total_return_mean=-0.9995, total_return_p05=-0.9996, total_return_p95=-0.9993
- breakout/random: n_candidates=4495, n_orders_mean=4287.3500, total_return_mean=-0.9958, total_return_p05=-0.9974, total_return_p95=-0.9940
- buy_and_hold: total_return=99.8135, sharpe=1.0697, max_drawdown=-0.8513

## バックテスト（holdout）

| strategy | orders | trades | fill_rate | total_return | max_dd | sharpe | win_rate | avg_ret_net | time_in_mkt | maker_fee | taker_fee | DSR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dip/primary | 1328 | 1010 | 0.761 | -0.8110 | -0.8096 | -15.082 | 0.354 | -0.00164 | 0.219 | 0 | 341459 | - |
| dip/threshold_0.5/logit | 0 | 0 | - | 0.0000 | 0.0000 | - | - | - | - | - | - | - |
| dip/threshold_0.5/lgbm | 18 | 16 | 0.889 | -0.0216 | -0.0435 | -0.972 | 0.438 | -0.00134 | 0.004 | 0 | 8772 | - |
| dip/threshold_0.55/logit | 0 | 0 | - | 0.0000 | 0.0000 | - | - | - | - | - | - | - |
| dip/threshold_0.55/lgbm | 3 | 3 | 1.000 | -0.0073 | -0.0134 | -0.863 | 0.333 | -0.00242 | 0.001 | 0 | 1978 | - |
| dip/threshold_0.6/logit | 0 | 0 | - | 0.0000 | 0.0000 | - | - | - | - | - | - | - |
| dip/threshold_0.6/lgbm | 0 | 0 | - | 0.0000 | 0.0000 | - | - | - | - | - | - | - |
| dip/meta/logit | 0 | 0 | - | 0.0000 | 0.0000 | - | - | - | - | - | - | - |
| dip/meta/lgbm | 18 | 16 | 0.889 | -0.0024 | -0.0033 | -1.595 | 0.438 | -0.00134 | 0.004 | 0 | 505 | - |
| breakout/primary | 1347 | 1033 | 0.767 | -0.7644 | -0.7696 | -15.417 | 0.370 | -0.00139 | 0.246 | 0 | 342999 | - |
| breakout/threshold_0.5/logit | 5 | 4 | 0.800 | -0.0143 | -0.0199 | -1.529 | 0.250 | -0.00358 | 0.000 | 0 | 2966 | - |
| breakout/threshold_0.5/lgbm | 20 | 16 | 0.800 | -0.0265 | -0.0376 | -1.565 | 0.375 | -0.00166 | 0.003 | 0 | 9776 | - |
| breakout/threshold_0.55/logit | 0 | 0 | - | 0.0000 | 0.0000 | - | - | - | - | - | - | - |
| breakout/threshold_0.55/lgbm | 3 | 2 | 0.667 | -0.0026 | -0.0071 | -0.435 | 0.500 | -0.00129 | 0.000 | 0 | 994 | - |
| breakout/threshold_0.6/logit | 0 | 0 | - | 0.0000 | 0.0000 | - | - | - | - | - | - | - |
| breakout/threshold_0.6/lgbm | 0 | 0 | - | 0.0000 | 0.0000 | - | - | - | - | - | - | - |
| breakout/meta/logit | 5 | 4 | 0.800 | 0.0000 | -0.0001 | 0.030 | 0.250 | -0.00358 | 0.000 | 0 | 13 | - |
| breakout/meta/lgbm | 19 | 15 | 0.789 | -0.0009 | -0.0014 | -0.965 | 0.400 | -0.00135 | 0.003 | 0 | 342 | - |

exit_type 別:

- dip/primary: sl n=484 pnl=-1227043 avg_ret=-0.00526, time n=220 pnl=-130896 avg_ret=-0.00120, tp n=306 pnl=546923 avg_ret=0.00377
- dip/threshold_0.5/lgbm: sl n=6 pnl=-50965 avg_ret=-0.00859, time n=3 pnl=-9308 avg_ret=-0.00320, tp n=7 pnl=38668 avg_ret=0.00567
- dip/threshold_0.55/lgbm: sl n=1 pnl=-9383 avg_ret=-0.00939, time n=1 pnl=-3993 avg_ret=-0.00403, tp n=1 pnl=6086 avg_ret=0.00617
- dip/meta/lgbm: sl n=6 pnl=-3426 avg_ret=-0.00859, time n=3 pnl=-486 avg_ret=-0.00320, tp n=7 pnl=1498 avg_ret=0.00567
- breakout/primary: sl n=434 pnl=-1208576 avg_ret=-0.00540, time n=249 pnl=-229205 avg_ret=-0.00185, tp n=350 pnl=673333 avg_ret=0.00391
- breakout/threshold_0.5/logit: sl n=3 pnl=-19906 avg_ret=-0.00668, tp n=1 pnl=5613 avg_ret=0.00573
- breakout/threshold_0.5/lgbm: sl n=8 pnl=-49530 avg_ret=-0.00628, time n=2 pnl=-5753 avg_ret=-0.00297, tp n=6 pnl=28793 avg_ret=0.00493
- breakout/threshold_0.55/lgbm: sl n=1 pnl=-7089 avg_ret=-0.00709, tp n=1 pnl=4483 avg_ret=0.00452
- breakout/meta/logit: sl n=3 pnl=-95 avg_ret=-0.00668, tp n=1 pnl=98 avg_ret=0.00573
- breakout/meta/lgbm: sl n=7 pnl=-1922 avg_ret=-0.00628, time n=2 pnl=-204 avg_ret=-0.00297, tp n=6 pnl=1211 avg_ret=0.00493

比較対象:

- dip/random: n_candidates=1328, n_orders_mean=1063.5500, total_return_mean=-0.7082, total_return_p05=-0.7387, total_return_p95=-0.6721
- breakout/random: n_candidates=1347, n_orders_mean=1073.5000, total_return_mean=-0.7080, total_return_p05=-0.7312, total_return_p95=-0.6708
- buy_and_hold: total_return=0.1954, sharpe=1.3484, max_drawdown=-0.2709
