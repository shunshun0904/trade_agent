/* API の形（どちらも dashboard/app/app.py が返す）。

/api/profile
  { now_ms, price, sigma, bin_width, window_h, n_trades_window, data_fetched_ms,
    vp_levels: {poc, vah, val}, vp: [{lo, hi, v}],      // v は BTC
    tpo_levels: {poc, vah, val}, tpo: [{lo, hi, n}],    // n は 30 分区間の数
    candles: [[t_ms, o, h, l, c]] }

/api/signals（直近 60 分の 1 分ごとの水準。dashboard/app/profile_core.py の signals）
  { now_ms, window_min: 60, keys: [...],
    minutes: [{ t, price, sigma, vp_poc_dist, vp_va_pos, vp_at_price, tpo_poc_dist, tpo_va_pos, tpo_single_up, p_big_1h }],
      // t は分の開始（ms）。各行は t より前の約定と t 以前に確定した足だけから計算する。距離は σ 単位
      // p_big_1h: 1 時間後に 0.3% を超えて上がる確率（ボラティリティの目安。方向の予測としては使えない。SPEC §1.3）。
      //           モデル（reports/direction_deploy/model_B.json）がなければ null
    model: { horizon_min, target_min_return, train_end, test_auc, base_rate_test } | 無し }
*/
import { mockProfile, mockSignals } from "./mock.js";

const SOURCE = import.meta.env.VITE_DATA_SOURCE || "api";
const base = () => (typeof location === "undefined" ? "/" : location.href.split("?")[0].split("#")[0].replace(/\/?$/, "/"));

async function getJson(path) {
  const r = await fetch(base() + path, { cache: "no-store" });
  let d = null;
  try { d = await r.json(); } catch { /* 本文が JSON でない */ }
  if (!r.ok || (d && d.error)) throw new Error((d && d.error) || `HTTP ${r.status}`);
  return d;
}

export const isMock = SOURCE === "mock";

export async function fetchProfile() {
  if (isMock) return mockProfile(Date.now());
  return getJson("api/profile");
}

export async function fetchSignals() {
  if (isMock) return mockSignals(Date.now());
  return getJson("api/signals");
}
