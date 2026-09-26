/* プレビュー用のダミーデータ。形は api.js のとおり。乱数の種を固定し、時刻だけ現在に合わせる。 */
const MIN = 60_000, BAR = 15 * MIN;

function rng(seed) {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}
const gauss = (r) => { const u = 1 - r(), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };

// 価格帯の配列 counts から POC と 70% のバリューエリア（bbresearch/profile.py と同じ規則）
function valueArea(counts, share = 0.7) {
  const tot = counts.reduce((a, b) => a + b, 0);
  let poc = 0;
  counts.forEach((c, i) => { if (c > counts[poc]) poc = i; });
  let lo = poc, hi = poc, acc = counts[poc];
  while (acc < share * tot && (lo > 0 || hi < counts.length - 1)) {
    const up = hi < counts.length - 1 ? counts[hi + 1] : -1;
    const dn = lo > 0 ? counts[lo - 1] : -1;
    if (up >= dn) { hi += 1; acc += up; } else { lo -= 1; acc += dn; }
  }
  return { poc, lo, hi };
}

function candlesFrom(now, r, p0) {
  const end = Math.floor(now / BAR) * BAR + BAR;
  const out = [];
  let p = p0;
  for (let t = end - 96 * BAR; t < end; t += BAR) {
    const o = p;
    const steps = Array.from({ length: 15 }, () => gauss(r) * 0.0011);
    const path = steps.map((s) => (p *= Math.exp(s)));
    out.push([t, o, Math.max(o, ...path), Math.min(o, ...path), p]);
  }
  return out;
}

export function mockProfile(now) {
  const r = rng(20260926);
  const candles = candlesFrom(now, r, 16_400_000);
  const price = candles[candles.length - 1][4];
  const sigma = 0.0042;
  const w = 0.25 * sigma * price;
  const lows = candles.map((c) => c[3]), highs = candles.map((c) => c[2]);
  const pmin = Math.min(...lows) - w, pmax = Math.max(...highs) + w;
  const k0 = Math.floor(pmin / w), n = Math.ceil(pmax / w) - k0;
  const vol = new Array(n).fill(0), tpo = new Array(n).fill(0);
  candles.forEach((c, i) => {
    const a = Math.floor(c[3] / w) - k0, b = Math.floor(c[2] / w) - k0;
    const v = (0.6 + 2.2 * r()) * (i % 7 === 0 ? 3 : 1);
    for (let k = a; k <= b; k++) vol[k] += v / (b - a + 1);
    if (i % 2 === 1) {
      const a2 = Math.floor(Math.min(c[3], candles[i - 1][3]) / w) - k0, b2 = Math.floor(Math.max(c[2], candles[i - 1][2]) / w) - k0;
      for (let k = a2; k <= b2; k++) tpo[k] += 1;
    }
  });
  const mk = (counts, key) => {
    const va = valueArea(counts);
    return {
      levels: { poc: (k0 + va.poc + 0.5) * w, val: (k0 + va.lo) * w, vah: (k0 + va.hi + 1) * w },
      rows: counts.map((c, i) => ({ lo: (k0 + i) * w, hi: (k0 + i + 1) * w, [key]: key === "n" ? c : +c.toFixed(4) })),
    };
  };
  const vp = mk(vol, "v"), tp = mk(tpo, "n");
  return {
    now_ms: now, price, sigma, bin_width: w, window_h: 24, n_trades_window: 61_842, data_fetched_ms: now - 2_400,
    vp_levels: vp.levels, vp: vp.rows, tpo_levels: tp.levels, tpo: tp.rows, candles,
  };
}

export function mockSignals(now) {
  const r = rng(777);
  const t0 = Math.floor(now / MIN) * MIN - 59 * MIN;
  let poc = 0.9, tpoc = 0.4, pos = 0.35, tpos = 0.5, thick = 1.1, single = 0.25, price = 15_700_000, pb = 0.16;
  const minutes = [];
  for (let i = 0; i < 60; i++) {
    pb = Math.min(0.8, Math.max(0.03, pb + 0.004 * (i - 25) / 35 + gauss(r) * 0.02));
    poc += gauss(r) * 0.06 - 0.008; tpoc += gauss(r) * 0.05; pos += gauss(r) * 0.03; tpos += gauss(r) * 0.03;
    thick = Math.max(0.2, thick + gauss(r) * 0.08); single = Math.min(1, Math.max(0, single + gauss(r) * 0.05));
    price *= Math.exp(gauss(r) * 0.0006);
    minutes.push({ t: t0 + i * MIN, price: Math.round(price), sigma: 0.0042, vp_poc_dist: +poc.toFixed(3),
      vp_va_pos: +pos.toFixed(3), vp_at_price: +thick.toFixed(3), tpo_poc_dist: +tpoc.toFixed(3),
      tpo_va_pos: +tpos.toFixed(3), tpo_single_up: +single.toFixed(3), p_big_1h: +pb.toFixed(3) });
  }
  return { now_ms: now, window_min: 60, keys: ["vp_poc_dist", "vp_va_pos", "vp_at_price", "tpo_poc_dist", "tpo_va_pos", "tpo_single_up"], minutes,
    model: { horizon_min: 60, target_min_return: 0.003, train_end: "2024-01-01", test_auc: 0.670, base_rate_test: 0.189 } };
}
