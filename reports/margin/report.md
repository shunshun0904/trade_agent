# bitbank の信用取引の対応ペア（2026-09-26 23:02 UTC、/spot/pairs から）

- 信用取引できると判定したペア（5）: btc_jpy、xrp_jpy、eth_jpy、sol_jpy、doge_jpy
- 判定: 建玉金利（long / short）が入っていて、信用の新規建て停止フラグが両方 false のもの
- 金利は 1 日あたり。委託保証金率（個人）は現在の値と、次に予定されている値

| ペア | 現物 | 信用 | 建玉金利 long/日 | short/日 | 新規 maker | 新規 taker | 返済 maker | 返済 taker | 保証金率（個人）現在 | まで | 次 | から | 停止 long | 停止 short |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| btc_jpy | ○ | ○ | 0.040% | 0.040% | 0.000% | 0.100% | 0.000% | 0.100% | 50.000% | 2026-09-29 23:59 JST | 50.000% | 2026-10-06 23:59 JST | False | False |
| doge_jpy | ○ | ○ | 0.040% | 0.040% | -0.020% | 0.120% | -0.020% | 0.120% | 50.000% | 期限なし | - | - | False | False |
| eth_jpy | ○ | ○ | 0.040% | 0.040% | -0.020% | 0.120% | -0.020% | 0.120% | 50.000% | 2026-09-29 23:59 JST | 50.000% | 2026-10-06 23:59 JST | False | False |
| sol_jpy | ○ | ○ | 0.040% | 0.040% | -0.020% | 0.120% | -0.020% | 0.120% | 50.000% | 期限なし | - | - | False | False |
| xrp_jpy | ○ | ○ | 0.040% | 0.040% | -0.020% | 0.120% | -0.020% | 0.120% | 50.000% | 2026-09-29 23:59 JST | 50.000% | 2026-10-06 23:59 JST | False | False |
| ada_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| ape_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| arb_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| astr_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| atom_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| avax_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| axs_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| bat_btc | ○ | - | - | - | - | - | - | - | - | - | - | - | True | True |
| bat_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| bcc_btc | ○ | - | - | - | - | - | - | - | - | - | - | - | True | True |
| bcc_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| bnb_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| boba_btc | ○ | - | - | - | - | - | - | - | - | - | - | - | True | True |
| boba_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| chz_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| cyber_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| dai_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| dot_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| enj_btc | ○ | - | - | - | - | - | - | - | - | - | - | - | True | True |
| enj_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| eth_btc | ○ | - | - | - | - | - | - | - | - | - | - | - | True | True |
| flr_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| gala_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| grt_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| imx_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| klay_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| link_btc | ○ | - | - | - | - | - | - | - | - | - | - | - | True | True |
| link_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| lpt_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| ltc_btc | ○ | - | - | - | - | - | - | - | - | - | - | - | True | True |
| ltc_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| mana_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| mask_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| matic_btc | ○ | - | - | - | - | - | - | - | - | - | - | - | True | True |
| matic_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | True | True |
| mkr_btc | ○ | - | - | - | - | - | - | - | - | - | - | - | True | True |
| mkr_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | True | True |
| mona_btc | ○ | - | - | - | - | - | - | - | - | - | - | - | True | True |
| mona_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| oas_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| omg_btc | ○ | - | - | - | - | - | - | - | - | - | - | - | True | True |
| omg_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| op_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| pol_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| qtum_btc | ○ | - | - | - | - | - | - | - | - | - | - | - | True | True |
| qtum_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| render_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| rndr_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | True | True |
| sand_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| sky_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| sui_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| trx_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| xlm_btc | ○ | - | - | - | - | - | - | - | - | - | - | - | True | True |
| xlm_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
| xrp_btc | ○ | - | - | - | - | - | - | - | - | - | - | - | True | True |
| xym_btc | ○ | - | - | - | - | - | - | - | - | - | - | - | True | True |
| xym_jpy | ○ | - | - | - | - | - | - | - | - | - | - | - | False | False |
