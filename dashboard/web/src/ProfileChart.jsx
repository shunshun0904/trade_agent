import { useMemo, useState } from "react";
import { useSize } from "./useSize.js";
import { yen, hm } from "./format.js";

const M = { top: 22, right: 14, bottom: 22, left: 72 };
const GAP = 14;

function lin(d0, d1, r0, r1) {
  const f = (x) => r0 + ((x - d0) / (d1 - d0)) * (r1 - r0);
  f.inv = (y) => d0 + ((y - r0) / (r1 - r0)) * (d1 - d0);
  return f;
}

function niceTicks(lo, hi, n) {
  const span = hi - lo, raw = span / n, p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * p).find((s) => span / s <= n) || 10 * p;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(v);
  return out;
}

/** 15 分足・価格帯別出来高・TPO を同じ価格軸に並べる。 */
export default function ProfileChart({ d, height = 520 }) {
  const [ref, W] = useSize();
  const [tip, setTip] = useState(null);

  const g = useMemo(() => {
    if (!W || !d) return null;
    const inner = W - M.left - M.right;
    const wC = Math.round(inner * 0.58), wP = Math.round((inner - wC - 2 * GAP) / 2);
    const xC0 = M.left, xV0 = xC0 + wC + GAP, xT0 = xV0 + wP + GAP;
    const H = height - M.top - M.bottom;
    const lows = d.candles.map((c) => c[3]), highs = d.candles.map((c) => c[2]);
    const pLo = Math.min(...lows, ...d.vp.map((r) => r.lo)), pHi = Math.max(...highs, ...d.vp.map((r) => r.hi));
    const pad = (pHi - pLo) * 0.03;
    const y = lin(pLo - pad, pHi + pad, M.top + H, M.top);
    const t0 = d.candles[0][0], t1 = d.candles[d.candles.length - 1][0] + 15 * 60_000;
    const x = lin(t0, t1, xC0, xC0 + wC);
    const cw = Math.max(1, (wC / d.candles.length) * 0.7);
    const vMax = Math.max(...d.vp.map((r) => r.v)), nMax = Math.max(...d.tpo.map((r) => r.n));
    const xv = lin(0, vMax, xV0, xV0 + wP), xt = lin(0, nMax, xT0, xT0 + wP);
    const tickH = [3, 6, 12].find((h) => (wC / 24) * h >= 56) || 12;  // 目盛の間隔が 56px 以上になる時間数
    return { inner, wC, wP, xC0, xV0, xT0, H, y, x, cw, xv, xt, t0, t1, pLo: pLo - pad, pHi: pHi + pad, vMax, nMax, tickH };
  }, [W, d, height]);

  if (!d) return <div ref={ref} className="chart-wrap" style={{ height }} />;

  const onMove = (e) => {
    if (!g) return;
    const box = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - box.left, py = e.clientY - box.top;
    const price = g.y.inv(py);
    let t = null;
    if (px >= g.xC0 && px <= g.xC0 + g.wC) {
      const tm = g.x.inv(px), i = Math.min(d.candles.length - 1, Math.max(0, Math.floor((tm - g.t0) / (15 * 60_000))));
      const c = d.candles[i];
      t = { px, py, rows: [["時刻", hm(c[0])], ["始値", yen(c[1])], ["高値", yen(c[2])], ["安値", yen(c[3])], ["終値", yen(c[4])]] };
    } else if (px >= g.xV0 && px <= g.xV0 + g.wP) {
      const r = d.vp.find((q) => price >= q.lo && price < q.hi);
      if (r) t = { px, py, rows: [["価格帯", `${yen(r.lo)}〜${yen(r.hi)}`], ["出来高", `${r.v.toFixed(3)} BTC`]] };
    } else if (px >= g.xT0 && px <= g.xT0 + g.wP) {
      const r = d.tpo.find((q) => price >= q.lo && price < q.hi);
      if (r) t = { px, py, rows: [["価格帯", `${yen(r.lo)}〜${yen(r.hi)}`], ["TPO", `${r.n} 区間`]] };
    }
    setTip(t);
  };

  const vl = d.vp_levels, tl = d.tpo_levels;
  const inVA = (r, lv) => (r.lo + r.hi) / 2 >= lv.val && (r.lo + r.hi) / 2 <= lv.vah;
  const bh = g ? Math.max(1, g.y(d.vp[0].lo) - g.y(d.vp[0].hi) - 1) : 0;

  return (
    <div ref={ref} className="chart-wrap" style={{ height }}>
      {g && (
        <svg width={W} height={height} onMouseMove={onMove} onMouseLeave={() => setTip(null)} role="img"
             aria-label="15 分足、価格帯別出来高、TPO">
          {/* 価格の目盛とグリッド */}
          <g className="grid">
            {niceTicks(g.pLo, g.pHi, 8).map((v) => (
              <g key={v}>
                <line x1={M.left} x2={W - M.right} y1={g.y(v)} y2={g.y(v)} />
                <text x={M.left - 8} y={g.y(v) + 3.5} textAnchor="end">{yen(v)}</text>
              </g>
            ))}
          </g>
          {/* 時刻の目盛（3 時間ごと） */}
          <g className="axis">
            <path d={`M${g.xC0},${M.top + g.H}H${g.xC0 + g.wC}`} />
            {d.candles.filter((c) => c[0] % (g.tickH * 3_600_000) === 0).map((c) => (
              <text key={c[0]} x={g.x(c[0])} y={height - 6} textAnchor="middle">{hm(c[0])}</text>
            ))}
            <text x={g.xV0} y={M.top - 8}>出来高（BTC）</text>
            <text x={g.xT0} y={M.top - 8}>TPO（30 分区間）</text>
          </g>
          {/* 価格帯別出来高 */}
          {d.vp.map((r) => (
            <rect key={`v${r.lo}`} x={g.xV0} y={g.y(r.hi) + 0.5} width={Math.max(0, g.xv(r.v) - g.xV0)} height={bh}
                  fill={inVA(r, vl) ? "var(--vp)" : "var(--bar-dim)"} rx={1} />
          ))}
          {/* TPO */}
          {d.tpo.map((r) => (
            <rect key={`t${r.lo}`} x={g.xT0} y={g.y(r.hi) + 0.5} width={Math.max(0, g.xt(r.n) - g.xT0)} height={bh}
                  fill={inVA(r, tl) ? "var(--tpo)" : "var(--bar-dim)"} rx={1} />
          ))}
          {/* ローソク足 */}
          {d.candles.map((c) => {
            const up = c[4] >= c[1], xc = g.x(c[0] + 7.5 * 60_000);
            const col = up ? "var(--up)" : "var(--down)";
            return (
              <g key={c[0]}>
                <line x1={xc} x2={xc} y1={g.y(c[2])} y2={g.y(c[3])} stroke={col} strokeWidth={1} />
                <rect x={xc - g.cw / 2} y={g.y(Math.max(c[1], c[4]))} width={g.cw}
                      height={Math.max(1, Math.abs(g.y(c[1]) - g.y(c[4])))} fill={col} />
              </g>
            );
          })}
          {/* 水準の線: 出来高は実線、TPO は点線。ラベルは足の左端 */}
          {[["poc", "POC", "var(--poc)"], ["vah", "VAH", "var(--vp)"], ["val", "VAL", "var(--vp)"]].map(([k, lab, col]) => (
            <g key={`vp-${k}`}>
              <line x1={g.xC0} x2={W - M.right} y1={g.y(vl[k])} y2={g.y(vl[k])} stroke={col} strokeWidth={1.5} />
              <rect x={g.xC0 + 2} y={g.y(vl[k]) - 14} width={(lab.length + yen(vl[k]).length + 1) * 6.6 + 6} height={13} rx={2}
                    fill="var(--panel)" opacity={0.85} />
              <text x={g.xC0 + 5} y={g.y(vl[k]) - 4} style={{ fill: col, fontWeight: 500 }}>{lab} {yen(vl[k])}</text>
            </g>
          ))}
          {[["poc", "var(--poc)"], ["vah", "var(--tpo)"], ["val", "var(--tpo)"]].map(([k, col]) => (
            <line key={`tpo-${k}`} x1={g.xC0} x2={W - M.right} y1={g.y(tl[k])} y2={g.y(tl[k])}
                  stroke={col} strokeWidth={1.5} strokeDasharray="2 4" />
          ))}
          {/* 現在値 */}
          <line x1={M.left} x2={W - M.right} y1={g.y(d.price)} y2={g.y(d.price)} stroke="var(--ink)" strokeWidth={1} strokeDasharray="6 4" />
          <rect x={4} y={g.y(d.price) - 8} width={M.left - 10} height={16} rx={3} fill="var(--ink)" />
          <text x={M.left - 8} y={g.y(d.price) + 3.5} textAnchor="end" style={{ fill: "var(--panel)", fontWeight: 500 }}>{yen(d.price)}</text>
          {tip && <line x1={M.left} x2={W - M.right} y1={tip.py} y2={tip.py} stroke="var(--ink-3)" strokeWidth={0.75} />}
        </svg>
      )}
      {tip && (
        <div className="tip" style={{ left: Math.min(tip.px + 14, W - 180), top: Math.max(0, tip.py - 10) }}>
          {tip.rows.map(([k, v]) => <div key={k}><b>{k}</b> <span className="num">{v}</span></div>)}
        </div>
      )}
    </div>
  );
}
