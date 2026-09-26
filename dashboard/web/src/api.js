/* API の形。

/api/profile（実装済み、dashboard/app/app.py）
  { now_ms, price, sigma, bin_width, window_h, n_trades_window, data_fetched_ms,
    vp_levels: {poc, vah, val}, vp: [{lo, hi, v}],      // v は BTC
    tpo_levels: {poc, vah, val}, tpo: [{lo, hi, n}],    // n は 30 分区間の数
    candles: [[t_ms, o, h, l, c]] }

/api/signals（この画面のために決めた形。Lambda にモデルを載せるまでは未実装で、画面は「未接続」と出す）
  { now_ms, horizon_min: 60, threshold: 0.6, next_decision_ms,
    minutes: [{ t, p_up, price, vp_poc_dist, vp_va_pos, vp_at_price, tpo_poc_dist, tpo_va_pos, tpo_single_up }],
      // 直近 60 分。t は分の開始（ms）。p_up はモデル B の「60 分後に上がる確率」。距離は σ 単位
    summary: { B_mean, B_last, B_slope, B_std, B_mean_recent },   // モデル A に渡す要約（bbresearch/direction.py と同じ定義）
    decision: { t, p_up, action: "buy" | "hold" } | null,         // 直近の正時の A の判断
    history: [{ t, p_up, action, r }],                             // 過去の判断と 1 時間後の対数リターン（r は未確定なら null）
    model: { trained_to, test_auc } }
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
  try {
    return await getJson("api/signals");
  } catch (e) {
    // 404 は「まだ載せていない」。それ以外は本当の失敗
    if (/HTTP 404|403/.test(e.message)) return { unavailable: true };
    throw e;
  }
}
