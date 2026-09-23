# 研究パイプライン結果（2026-09-23T20:40:28+00:00）

- pair: `btc_jpy`、期間 2017-02-14 〜 2026-09-22（ホールドアウト 180 日）
- 足 336672 本（約定なし 4237）、約定 72488969 件、config_hash `3a5ccf669564`
- 手数料率: メイカー 0.0、テイカー 0.001
- 試行数（DSR 用、過去の実験ログを含む）: 1650

## イベントとラベル

| signal | period | events | fill_rate | y_rate | mean_ret_net | exit_types |
|---|---|---|---|---|---|---|
| dip | dev | 9200 | 0.594 | 0.512 | -0.00102 | {'tp': 2306, 'sl': 1916, 'time': 1241} |
| dip | holdout | 532 | 0.461 | 0.498 | -0.00078 | {'tp': 105, 'sl': 83, 'time': 57} |
| breakout | dev | 9899 | 0.563 | 0.456 | -0.00187 | {'tp': 2022, 'time': 2020, 'sl': 1533} |
| breakout | holdout | 562 | 0.452 | 0.437 | -0.00110 | {'tp': 101, 'time': 88, 'sl': 65} |

## メタモデル（OOF / ホールドアウト）

| signal | model | n_train | base_rate | OOF logloss | OOF AUC | OOF brier | HO n | HO AUC | HO logloss |
|---|---|---|---|---|---|---|---|---|---|
| dip | logit | 5463 | 0.512 | 0.693 | 0.539 | 0.250 | 245 | 0.526 | 0.693 |
| dip | lgbm | 5463 | 0.512 | 0.694 | 0.549 | 0.250 | 245 | 0.535 | 0.698 |
| breakout | logit | 5575 | 0.456 | 0.690 | 0.534 | 0.248 | 254 | 0.504 | 0.687 |
| breakout | lgbm | 5575 | 0.456 | 0.697 | 0.519 | 0.252 | 254 | 0.489 | 0.695 |

## バックテスト（dev）

| strategy | orders | trades | fill_rate | total_return | max_dd | sharpe | win_rate | avg_ret_net | time_in_mkt | maker_fee | taker_fee | DSR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dip/primary | 8837 | 5218 | 0.590 | -0.9977 | -0.9979 | -1.750 | 0.509 | -0.00111 | 0.127 | 0 | 642257 | 0.000 |
| dip/threshold_0.5/logit | 5224 | 3266 | 0.625 | -0.9514 | -0.9571 | -0.867 | 0.525 | -0.00080 | 0.080 | 0 | 632677 | 0.000 |
| dip/threshold_0.5/lgbm | 4724 | 2887 | 0.611 | -0.8122 | -0.8400 | -0.480 | 0.536 | -0.00045 | 0.069 | 0 | 982455 | 0.000 |
| dip/threshold_0.55/logit | 2046 | 1354 | 0.662 | -0.3765 | -0.5820 | -0.047 | 0.547 | -0.00012 | 0.031 | 0 | 607664 | 0.000 |
| dip/threshold_0.55/lgbm | 2826 | 1737 | 0.615 | -0.2551 | -0.5624 | -0.001 | 0.553 | -0.00000 | 0.040 | 0 | 981072 | 0.000 |
| dip/threshold_0.6/logit | 637 | 424 | 0.666 | 1.1594 | -0.3675 | 0.484 | 0.583 | 0.00228 | 0.009 | 0 | 292435 | 0.000 |
| dip/threshold_0.6/lgbm | 1432 | 899 | 0.628 | 1.2273 | -0.2582 | 0.553 | 0.585 | 0.00109 | 0.021 | 0 | 827967 | 0.000 |
| dip/meta/logit | 5208 | 3259 | 0.626 | 0.2012 | -0.2449 | 0.241 | 0.525 | -0.00081 | 0.080 | 0 | 173515 | 0.000 |
| dip/meta/lgbm | 4714 | 2880 | 0.611 | 0.0987 | -0.1044 | 0.190 | 0.536 | -0.00045 | 0.069 | 0 | 210576 | 0.000 |
| breakout/primary | 6393 | 3883 | 0.607 | -0.9996 | -0.9996 | -2.954 | 0.457 | -0.00206 | 0.119 | 0 | 152372 | 0.000 |
| breakout/threshold_0.5/logit | 1644 | 1073 | 0.653 | -0.7727 | -0.7946 | -0.750 | 0.511 | -0.00120 | 0.026 | 0 | 232677 | 0.000 |
| breakout/threshold_0.5/lgbm | 2613 | 1546 | 0.592 | -0.9526 | -0.9582 | -1.867 | 0.466 | -0.00186 | 0.044 | 0 | 215833 | 0.000 |
| breakout/threshold_0.55/logit | 460 | 301 | 0.654 | -0.5987 | -0.6036 | -0.586 | 0.512 | -0.00264 | 0.007 | 0 | 108642 | 0.000 |
| breakout/threshold_0.55/lgbm | 1119 | 645 | 0.576 | -0.4268 | -0.5680 | -0.411 | 0.496 | -0.00073 | 0.017 | 0 | 297692 | 0.000 |
| breakout/threshold_0.6/logit | 190 | 115 | 0.605 | -0.3559 | -0.4518 | -0.344 | 0.478 | -0.00309 | 0.002 | 0 | 55879 | 0.000 |
| breakout/threshold_0.6/lgbm | 354 | 202 | 0.571 | -0.1677 | -0.2872 | -0.194 | 0.505 | -0.00072 | 0.005 | 0 | 105422 | 0.000 |
| breakout/meta/logit | 1636 | 1068 | 0.653 | -0.1618 | -0.1684 | -0.312 | 0.511 | -0.00120 | 0.026 | 0 | 42159 | 0.000 |
| breakout/meta/lgbm | 2597 | 1537 | 0.592 | -0.1263 | -0.1654 | -0.695 | 0.466 | -0.00186 | 0.044 | 0 | 68202 | 0.000 |

exit_type 別:

- dip/primary: sl n=1843 pnl=-7856957 avg_ret=-0.01558, time n=1187 pnl=-346034 avg_ret=-0.00137, tp n=2188 pnl=7205306 avg_ret=0.01123
- dip/threshold_0.5/logit: sl n=1110 pnl=-8104981 avg_ret=-0.01792, time n=740 pnl=-361033 avg_ret=-0.00140, tp n=1416 pnl=7514576 avg_ret=0.01295
- dip/threshold_0.5/lgbm: sl n=953 pnl=-11463062 avg_ret=-0.01777, time n=650 pnl=-484810 avg_ret=-0.00119, tp n=1284 pnl=11135633 avg_ret=0.01279
- dip/threshold_0.55/logit: sl n=455 pnl=-8998998 avg_ret=-0.02293, time n=281 pnl=-274472 avg_ret=-0.00124, tp n=618 pnl=8896931 avg_ret=0.01719
- dip/threshold_0.55/lgbm: sl n=561 pnl=-11829222 avg_ret=-0.01973, time n=367 pnl=-484245 avg_ret=-0.00119, tp n=809 pnl=12058380 avg_ret=0.01421
- dip/threshold_0.6/logit: sl n=137 pnl=-5773142 avg_ret=-0.03267, time n=76 pnl=13626 avg_ret=0.00003, tp n=211 pnl=6918957 avg_ret=0.02578
- dip/threshold_0.6/lgbm: sl n=267 pnl=-9606954 avg_ret=-0.02176, time n=194 pnl=-364346 avg_ret=-0.00086, tp n=438 pnl=11198592 avg_ret=0.01589
- dip/meta/logit: sl n=1109 pnl=-2634845 avg_ret=-0.01793, time n=739 pnl=-53495 avg_ret=-0.00140, tp n=1411 pnl=2889515 avg_ret=0.01296
- dip/meta/lgbm: sl n=951 pnl=-2542360 avg_ret=-0.01779, time n=649 pnl=-95538 avg_ret=-0.00119, tp n=1280 pnl=2736625 avg_ret=0.01280
- breakout/primary: sl n=1040 pnl=-2137734 avg_ret=-0.01874, time n=1436 pnl=-384363 avg_ret=-0.00366, tp n=1407 pnl=1522467 avg_ret=0.01191
- breakout/threshold_0.5/logit: sl n=319 pnl=-2966942 avg_ret=-0.01970, time n=279 pnl=-346216 avg_ret=-0.00324, tp n=475 pnl=2540497 avg_ret=0.01242
- breakout/threshold_0.5/lgbm: sl n=436 pnl=-2242400 avg_ret=-0.01705, time n=510 pnl=-538083 avg_ret=-0.00385, tp n=600 pnl=1827917 avg_ret=0.01088
- breakout/threshold_0.55/logit: sl n=99 pnl=-2082015 avg_ret=-0.03064, time n=76 pnl=-94470 avg_ret=-0.00236, tp n=126 pnl=1577780 avg_ret=0.01920
- breakout/threshold_0.55/lgbm: sl n=184 pnl=-2640299 avg_ret=-0.01677, time n=190 pnl=-441939 avg_ret=-0.00297, tp n=271 pnl=2655479 avg_ret=0.01172
- breakout/threshold_0.6/logit: sl n=40 pnl=-1454503 avg_ret=-0.04157, time n=27 pnl=-76017 avg_ret=-0.00435, tp n=48 pnl=1174634 avg_ret=0.02970
- breakout/threshold_0.6/lgbm: sl n=62 pnl=-1049166 avg_ret=-0.01757, time n=53 pnl=-176284 avg_ret=-0.00355, tp n=87 pnl=1057763 avg_ret=0.01301
- breakout/meta/logit: sl n=317 pnl=-855210 avg_ret=-0.01979, time n=278 pnl=-12482 avg_ret=-0.00325, tp n=473 pnl=705893 avg_ret=0.01245
- breakout/meta/lgbm: sl n=432 pnl=-583417 avg_ret=-0.01707, time n=509 pnl=-118180 avg_ret=-0.00384, tp n=596 pnl=575292 avg_ret=0.01085

比較対象:

- dip/random: n_candidates=8837, n_orders_mean=7840.6500, total_return_mean=-0.9951, total_return_p05=-0.9985, total_return_p95=-0.9907
- breakout/random: n_candidates=6393, n_orders_mean=5856.8500, total_return_mean=-0.9858, total_return_p05=-0.9968, total_return_p95=-0.9722
- buy_and_hold: total_return=99.8135, sharpe=1.0697, max_drawdown=-0.8513

## バックテスト（holdout）

| strategy | orders | trades | fill_rate | total_return | max_dd | sharpe | win_rate | avg_ret_net | time_in_mkt | maker_fee | taker_fee | DSR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dip/primary | 512 | 236 | 0.461 | -0.1855 | -0.1813 | -2.544 | 0.492 | -0.00084 | 0.108 | 0 | 119915 | - |
| dip/threshold_0.5/logit | 185 | 86 | 0.465 | -0.0707 | -0.0971 | -1.383 | 0.488 | -0.00082 | 0.037 | 0 | 49885 | - |
| dip/threshold_0.5/lgbm | 215 | 102 | 0.474 | -0.0425 | -0.0965 | -0.647 | 0.520 | -0.00040 | 0.043 | 0 | 51992 | - |
| dip/threshold_0.55/logit | 45 | 14 | 0.311 | -0.0016 | -0.0233 | -0.070 | 0.571 | -0.00008 | 0.006 | 0 | 7006 | - |
| dip/threshold_0.55/lgbm | 107 | 53 | 0.495 | 0.0385 | -0.0385 | 0.880 | 0.585 | 0.00074 | 0.023 | 0 | 27014 | - |
| dip/threshold_0.6/logit | 2 | 0 | 0.000 | 0.0000 | 0.0000 | - | - | - | - | - | - | - |
| dip/threshold_0.6/lgbm | 39 | 19 | 0.487 | 0.0488 | -0.0153 | 1.757 | 0.579 | 0.00254 | 0.008 | 0 | 9130 | - |
| dip/meta/logit | 185 | 86 | 0.465 | -0.0010 | -0.0041 | -0.486 | 0.488 | -0.00082 | 0.037 | 0 | 2005 | - |
| dip/meta/lgbm | 214 | 102 | 0.477 | 0.0078 | -0.0066 | 0.931 | 0.520 | -0.00040 | 0.043 | 0 | 5448 | - |
| breakout/primary | 554 | 251 | 0.453 | -0.2413 | -0.2447 | -3.658 | 0.438 | -0.00108 | 0.129 | 0 | 129787 | - |
| breakout/threshold_0.5/logit | 49 | 27 | 0.551 | -0.0035 | -0.0377 | -0.142 | 0.519 | -0.00010 | 0.013 | 0 | 12947 | - |
| breakout/threshold_0.5/lgbm | 114 | 55 | 0.482 | -0.1471 | -0.1616 | -3.365 | 0.327 | -0.00286 | 0.029 | 0 | 35435 | - |
| breakout/threshold_0.55/logit | 1 | 0 | 0.000 | 0.0000 | 0.0000 | - | - | - | - | - | - | - |
| breakout/threshold_0.55/lgbm | 29 | 15 | 0.517 | -0.0541 | -0.0653 | -2.342 | 0.267 | -0.00367 | 0.008 | 0 | 11643 | - |
| breakout/threshold_0.6/logit | 0 | 0 | - | 0.0000 | 0.0000 | - | - | - | - | - | - | - |
| breakout/threshold_0.6/lgbm | 3 | 0 | 0.000 | 0.0000 | 0.0000 | - | - | - | - | - | - | - |
| breakout/meta/logit | 47 | 25 | 0.532 | -0.0005 | -0.0014 | -0.888 | 0.560 | 0.00018 | 0.011 | 0 | 304 | - |
| breakout/meta/lgbm | 113 | 55 | 0.487 | -0.0108 | -0.0122 | -2.739 | 0.327 | -0.00286 | 0.029 | 0 | 2341 | - |

exit_type 別:

- dip/primary: sl n=82 pnl=-656252 avg_ret=-0.00900, time n=54 pnl=-70526 avg_ret=-0.00153, tp n=100 pnl=541268 avg_ret=0.00621
- dip/threshold_0.5/logit: sl n=32 pnl=-290997 avg_ret=-0.00946, time n=20 pnl=-7581 avg_ret=-0.00042, tp n=34 pnl=227871 avg_ret=0.00707
- dip/threshold_0.5/lgbm: sl n=35 pnl=-297990 avg_ret=-0.00876, time n=19 pnl=-44877 avg_ret=-0.00250, tp n=48 pnl=300405 avg_ret=0.00653
- dip/threshold_0.55/logit: sl n=5 pnl=-52474 avg_ret=-0.01040, time n=2 pnl=1625 avg_ret=0.00082, tp n=7 pnl=49214 avg_ret=0.00703
- dip/threshold_0.55/lgbm: sl n=15 pnl=-113904 avg_ret=-0.00757, time n=12 pnl=-26310 avg_ret=-0.00218, tp n=26 pnl=178702 avg_ret=0.00688
- dip/threshold_0.6/lgbm: sl n=4 pnl=-24956 avg_ret=-0.00617, time n=5 pnl=-10627 avg_ret=-0.00207, tp n=10 pnl=84388 avg_ret=0.00832
- dip/meta/logit: sl n=32 pnl=-12870 avg_ret=-0.00946, time n=20 pnl=-378 avg_ret=-0.00042, tp n=34 pnl=12235 avg_ret=0.00707
- dip/meta/lgbm: sl n=35 pnl=-24521 avg_ret=-0.00876, time n=19 pnl=-4413 avg_ret=-0.00250, tp n=48 pnl=36758 avg_ret=0.00653
- breakout/primary: sl n=65 pnl=-534549 avg_ret=-0.00947, time n=85 pnl=-247389 avg_ret=-0.00332, tp n=101 pnl=540679 avg_ret=0.00622
- breakout/threshold_0.5/logit: sl n=5 pnl=-54550 avg_ret=-0.01091, time n=8 pnl=-31990 avg_ret=-0.00399, tp n=14 pnl=83085 avg_ret=0.00598
- breakout/threshold_0.5/lgbm: sl n=18 pnl=-182971 avg_ret=-0.01117, time n=21 pnl=-57909 avg_ret=-0.00299, tp n=16 pnl=93738 avg_ret=0.00664
- breakout/threshold_0.55/lgbm: sl n=5 pnl=-60472 avg_ret=-0.01247, time n=7 pnl=-13763 avg_ret=-0.00198, tp n=3 pnl=20128 avg_ret=0.00704
- breakout/meta/logit: sl n=5 pnl=-1573 avg_ret=-0.01091, time n=6 pnl=-612 avg_ret=-0.00412, tp n=14 pnl=1645 avg_ret=0.00598
- breakout/meta/lgbm: sl n=18 pnl=-13609 avg_ret=-0.01117, time n=21 pnl=-2656 avg_ret=-0.00299, tp n=16 pnl=5458 avg_ret=0.00664

比較対象:

- dip/random: n_candidates=512, n_orders_mean=468.2500, total_return_mean=-0.1866, total_return_p05=-0.2688, total_return_p95=-0.0992
- breakout/random: n_candidates=554, n_orders_mean=502.5500, total_return_mean=-0.2001, total_return_p05=-0.2724, total_return_p95=-0.1431
- buy_and_hold: total_return=0.1954, sharpe=1.3484, max_drawdown=-0.2709
