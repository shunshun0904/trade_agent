# 研究パイプライン結果（2026-09-23T20:57:19+00:00）

- pair: `btc_jpy`、期間 2017-02-14 〜 2026-09-22（ホールドアウト 180 日）
- 足 336672 本（約定なし 4237）、約定 72488969 件、config_hash `080201e7f42d`
- 手数料率: メイカー 0.0、テイカー 0.001
- ホールドアウトは評価していない（使用済み。`split.evaluate_holdout: false`）
- 収益率の分布の特徴量: あり（窓 [4, 16, 96] 本）
- 価格帯別出来高・TPO の特徴量: あり（直近 24 時間、刻み 0.25σ）
- 試行数（DSR 用、過去の実験ログを含む）: 1686

## イベントとラベル

| signal | period | events | fill_rate | y_rate | mean_ret_net | exit_types |
|---|---|---|---|---|---|---|
| dip | dev | 9200 | 0.594 | 0.512 | -0.00102 | {'tp': 2306, 'sl': 1916, 'time': 1241} |
| breakout | dev | 9899 | 0.563 | 0.456 | -0.00187 | {'tp': 2022, 'time': 2020, 'sl': 1533} |

## メタモデル（OOF / ホールドアウト）

| signal | model | n_train | base_rate | OOF logloss | OOF AUC | OOF brier | HO n | HO AUC | HO logloss |
|---|---|---|---|---|---|---|---|---|---|
| dip | logit | 5463 | 0.512 | 0.692 | 0.558 | 0.249 | 0 | - | - |
| dip | lgbm | 5463 | 0.512 | 0.693 | 0.550 | 0.250 | 0 | - | - |
| breakout | logit | 5575 | 0.456 | 0.695 | 0.522 | 0.251 | 0 | - | - |
| breakout | lgbm | 5575 | 0.456 | 0.694 | 0.532 | 0.250 | 0 | - | - |

## バックテスト（dev）

| strategy | orders | trades | fill_rate | total_return | max_dd | sharpe | win_rate | avg_ret_net | time_in_mkt | maker_fee | taker_fee | DSR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dip/primary | 8837 | 5218 | 0.590 | -0.9977 | -0.9979 | -1.750 | 0.509 | -0.00111 | 0.127 | 0 | 642257 | 0.000 |
| dip/threshold_0.5/logit | 4844 | 3012 | 0.622 | -0.7180 | -0.8378 | -0.289 | 0.547 | -0.00029 | 0.072 | 0 | 1424911 | 0.000 |
| dip/threshold_0.5/lgbm | 4804 | 2968 | 0.618 | -0.8377 | -0.8594 | -0.523 | 0.536 | -0.00048 | 0.071 | 0 | 935879 | 0.000 |
| dip/threshold_0.55/logit | 2884 | 1842 | 0.639 | 0.6137 | -0.4378 | 0.329 | 0.570 | 0.00043 | 0.043 | 0 | 1463526 | 0.000 |
| dip/threshold_0.55/lgbm | 3032 | 1887 | 0.622 | 0.0430 | -0.4163 | 0.152 | 0.561 | 0.00018 | 0.044 | 0 | 1172990 | 0.000 |
| dip/threshold_0.6/logit | 1390 | 896 | 0.645 | 2.4028 | -0.3716 | 0.671 | 0.600 | 0.00162 | 0.020 | 0 | 973672 | 0.000 |
| dip/threshold_0.6/lgbm | 1539 | 953 | 0.619 | 0.7118 | -0.3517 | 0.401 | 0.577 | 0.00076 | 0.021 | 0 | 685894 | 0.000 |
| dip/meta/logit | 4837 | 3010 | 0.622 | 0.5331 | -0.1335 | 0.534 | 0.547 | -0.00029 | 0.072 | 0 | 271503 | 0.000 |
| dip/meta/lgbm | 4797 | 2965 | 0.618 | 0.2838 | -0.0848 | 0.477 | 0.537 | -0.00047 | 0.071 | 0 | 244559 | 0.000 |
| breakout/primary | 6393 | 3883 | 0.607 | -0.9996 | -0.9996 | -2.954 | 0.457 | -0.00206 | 0.119 | 0 | 152372 | 0.000 |
| breakout/threshold_0.5/logit | 2382 | 1489 | 0.625 | -0.9144 | -0.9223 | -1.223 | 0.487 | -0.00150 | 0.040 | 0 | 207744 | 0.000 |
| breakout/threshold_0.5/lgbm | 2675 | 1599 | 0.598 | -0.9133 | -0.9186 | -1.435 | 0.478 | -0.00142 | 0.045 | 0 | 305646 | 0.000 |
| breakout/threshold_0.55/logit | 852 | 561 | 0.658 | -0.7271 | -0.7438 | -0.817 | 0.506 | -0.00206 | 0.014 | 0 | 137226 | 0.000 |
| breakout/threshold_0.55/lgbm | 1208 | 718 | 0.594 | -0.5166 | -0.6111 | -0.511 | 0.510 | -0.00087 | 0.019 | 0 | 270137 | 0.000 |
| breakout/threshold_0.6/logit | 296 | 196 | 0.662 | -0.3967 | -0.4327 | -0.381 | 0.520 | -0.00207 | 0.005 | 0 | 86385 | 0.000 |
| breakout/threshold_0.6/lgbm | 387 | 234 | 0.605 | -0.1238 | -0.2945 | -0.096 | 0.513 | -0.00037 | 0.006 | 0 | 130864 | 0.000 |
| breakout/meta/logit | 2364 | 1480 | 0.626 | -0.1844 | -0.2325 | -0.297 | 0.487 | -0.00149 | 0.039 | 0 | 69224 | 0.000 |
| breakout/meta/lgbm | 2661 | 1587 | 0.596 | -0.1224 | -0.1394 | -0.610 | 0.478 | -0.00144 | 0.045 | 0 | 76617 | 0.000 |

exit_type 別:

- dip/primary: sl n=1843 pnl=-7856957 avg_ret=-0.01558, time n=1187 pnl=-346034 avg_ret=-0.00137, tp n=2188 pnl=7205306 avg_ret=0.01123
- dip/threshold_0.5/logit: sl n=994 pnl=-16207711 avg_ret=-0.01769, time n=649 pnl=-646220 avg_ret=-0.00117, tp n=1369 pnl=16135965 avg_ret=0.01277
- dip/threshold_0.5/lgbm: sl n=991 pnl=-11087129 avg_ret=-0.01776, time n=641 pnl=-442405 avg_ret=-0.00122, tp n=1336 pnl=10691797 avg_ret=0.01270
- dip/threshold_0.55/logit: sl n=575 pnl=-16130801 avg_ret=-0.02001, time n=379 pnl=-631359 avg_ret=-0.00104, tp n=888 pnl=17375829 avg_ret=0.01430
- dip/threshold_0.55/lgbm: sl n=603 pnl=-13146378 avg_ret=-0.01925, time n=395 pnl=-474301 avg_ret=-0.00102, tp n=889 pnl=13663705 avg_ret=0.01390
- dip/threshold_0.6/logit: sl n=269 pnl=-11816684 avg_ret=-0.02348, time n=168 pnl=-254762 avg_ret=-0.00029, tp n=459 pnl=14474296 avg_ret=0.01702
- dip/threshold_0.6/lgbm: sl n=297 pnl=-8252358 avg_ret=-0.02170, time n=186 pnl=-132189 avg_ret=-0.00020, tp n=470 pnl=9096366 avg_ret=0.01534
- dip/meta/logit: sl n=993 pnl=-3414834 avg_ret=-0.01769, time n=649 pnl=-73214 avg_ret=-0.00117, tp n=1368 pnl=4021161 avg_ret=0.01277
- dip/meta/lgbm: sl n=989 pnl=-2885650 avg_ret=-0.01776, time n=640 pnl=-71558 avg_ret=-0.00123, tp n=1336 pnl=3240998 avg_ret=0.01270
- breakout/primary: sl n=1040 pnl=-2137734 avg_ret=-0.01874, time n=1436 pnl=-384363 avg_ret=-0.00366, tp n=1407 pnl=1522467 avg_ret=0.01191
- breakout/threshold_0.5/logit: sl n=436 pnl=-2612478 avg_ret=-0.01854, time n=442 pnl=-374185 avg_ret=-0.00329, tp n=611 pnl=2072291 avg_ret=0.01195
- breakout/threshold_0.5/lgbm: sl n=457 pnl=-2881792 avg_ret=-0.01665, time n=523 pnl=-591662 avg_ret=-0.00316, tp n=619 pnl=2560122 avg_ret=0.01131
- breakout/threshold_0.55/logit: sl n=168 pnl=-2185085 avg_ret=-0.02455, time n=153 pnl=-262077 avg_ret=-0.00369, tp n=240 pnl=1720014 avg_ret=0.01473
- breakout/threshold_0.55/lgbm: sl n=206 pnl=-2599701 avg_ret=-0.01769, time n=212 pnl=-417521 avg_ret=-0.00281, tp n=300 pnl=2500582 avg_ret=0.01205
- breakout/threshold_0.6/logit: sl n=66 pnl=-1856989 avg_ret=-0.03356, time n=44 pnl=-78877 avg_ret=-0.00175, tp n=86 pnl=1539139 avg_ret=0.02193
- breakout/threshold_0.6/lgbm: sl n=63 pnl=-1136334 avg_ret=-0.01768, time n=71 pnl=-247512 avg_ret=-0.00350, tp n=100 pnl=1260062 avg_ret=0.01275
- breakout/meta/logit: sl n=432 pnl=-1142988 avg_ret=-0.01862, time n=440 pnl=-56105 avg_ret=-0.00330, tp n=608 pnl=1014695 avg_ret=0.01198
- breakout/meta/lgbm: sl n=454 pnl=-628144 avg_ret=-0.01669, time n=518 pnl=-123910 avg_ret=-0.00320, tp n=615 pnl=629639 avg_ret=0.01129

比較対象:

- dip/random: n_candidates=8837, n_orders_mean=7840.6500, total_return_mean=-0.9951, total_return_p05=-0.9985, total_return_p95=-0.9907
- breakout/random: n_candidates=6393, n_orders_mean=5856.5000, total_return_mean=-0.9753, total_return_p05=-0.9944, total_return_p95=-0.9362
- buy_and_hold: total_return=99.8135, sharpe=1.0697, max_drawdown=-0.8513
