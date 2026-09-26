import { useMemo, useState } from "react";
import { useSize } from "./useSize.js";
import { fix, signed, hm, mmss, pct, yen } from "./format.js";

const MIN = 60_000;

/** 直近 60 分のモデル B の上げ確率。しきい値より上を帯で示し、最新点を強調する。 */
function ProbabilityChart({ minutes, threshold, height = 170 }) {
  const [ref, W] = useSize();
  const [hover, setHover] = useState(null);
  const M = { top: 8, right: 10, bottom: 20, left: 34 };
  const g = useMemo(() => {
    if (!W || !minutes?.length) return null;
    const innerW = W - M.left - M.right, H = height - M.top - M.bottom;
    const t0 = minutes[0].t, t1 = minutes[minutes.length - 1].t;
    const x = (t) => M.left + ((t - t0) / (t1 - t0 || 1)) * innerW;
    const y = (p) => M.top + (1 - p) * H;
    const path = minutes.map((m, i) => `${i ? "L" : "M"}${x(m.t).toFixed(1)},${y(m.p_up).toFixed(1)}`).join("");
    const area = `${path}L${x(t1).toFixed(1)},${y(0).toFixed(1)}L${x(t0).toFixed(1)},${y(0).toFixed(1)}Z`;
    return { innerW, H, x, y, path, area, t0, t1 };
  }, [W, minutes, height]);
  if (!minutes?.length) return <div ref={ref} style={{ height }} />;
  const last = minutes[minutes.length - 1];
  const onMove = (e) => {
    if (!g) return;
    const px = e.clientX - e.currentTarget.getBoundingClientRect().left;
    const i = Math.round(((px - M.left) / g.innerW) * (minutes.length - 1));
    setHover(minutes[Math.min(minutes.length - 1, Math.max(0, i))]);
  };
  const above = g && Math.max(0, g.y(threshold) - M.top);
  return (
    <div ref={ref} className="chart-wrap" style={{ height }}>
      {g && (
        <svg width={W} height={height} onMouseMove={onMove} onMouseLeave={() => setHover(null)} role="img"
             aria-label="直近 60 分のモデル B の上げ確率">
          <rect x={M.left} y={M.top} width={g.innerW} height={above} fill="var(--band)" />
          <g className="grid">
            {[0.3, 0.4, 0.5, 0.6, 0.7].map((p) => (
              <g key={p}>
                <line x1={M.left} x2={W - M.right} y1={g.y(p)} y2={g.y(p)} strokeDasharray={p === 0.5 ? "" : "2 3"} />
                <text x={M.left - 6} y={g.y(p) + 3.5} textAnchor="end">{p.toFixed(1)}</text>
              </g>
            ))}
          </g>
          <line x1={M.left} x2={W - M.right} y1={g.y(threshold)} y2={g.y(threshold)} stroke="var(--vp)" strokeWidth={1} strokeDasharray="4 3" />
          <text x={M.left + 4} y={g.y(threshold) - 4} style={{ fill: "var(--vp)" }}>しきい値 {threshold.toFixed(2)}</text>
          <path d={g.area} fill="var(--vp-soft)" opacity={0.55} />
          <path d={g.path} fill="none" stroke="var(--vp)" strokeWidth={2} strokeLinejoin="round" />
          <circle cx={g.x(last.t)} cy={g.y(last.p_up)} r={4} fill="var(--vp)" stroke="var(--panel)" strokeWidth={2} />
          <g className="axis">
            {minutes.filter((m) => m.t % (15 * MIN) === 0).map((m) => (
              <text key={m.t} x={g.x(m.t)} y={height - 5} textAnchor="middle">{hm(m.t)}</text>
            ))}
          </g>
          {hover && (
            <g>
              <line x1={g.x(hover.t)} x2={g.x(hover.t)} y1={M.top} y2={M.top + g.H} stroke="var(--ink-3)" strokeWidth={0.75} />
              <circle cx={g.x(hover.t)} cy={g.y(hover.p_up)} r={3.5} fill="var(--panel)" stroke="var(--vp)" strokeWidth={2} />
            </g>
          )}
        </svg>
      )}
      {hover && g && (
        <div className="tip" style={{ left: Math.min(g.x(hover.t) + 12, W - 150), top: 4 }}>
          <div><b>{hm(hover.t)}</b></div>
          <div>上げ確率 <span className="num">{fix(hover.p_up, 3)}</span></div>
          <div>価格 <span className="num">{yen(hover.price)}</span></div>
        </div>
      )}
    </div>
  );
}

/** 生の水準 1 つの 60 分の推移（小さな折れ線）。 */
function Spark({ label, values, unit, digits = 2, zero = false }) {
  const [ref, W] = useSize();
  const H = 34, P = 3;
  const v = values.filter((x) => x != null && isFinite(x));
  if (!v.length) return null;
  const lo = Math.min(...v, zero ? 0 : Infinity), hi = Math.max(...v, zero ? 0 : -Infinity);
  const y = (x) => P + (1 - (x - lo) / (hi - lo || 1)) * (H - 2 * P);
  const x = (i) => (W ? (i / (values.length - 1)) * (W - 2) + 1 : 0);
  const path = values.map((q, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(q).toFixed(1)}`).join("");
  const last = values[values.length - 1], chg = last - values[0];
  return (
    <div className="lv" ref={ref}>
      <div className="k">{label}</div>
      {W > 0 && (
        <svg width={W} height={H} aria-hidden="true">
          {zero && lo < 0 && hi > 0 && <line x1={0} x2={W} y1={y(0)} y2={y(0)} stroke="var(--line)" />}
          <path d={path} fill="none" stroke="var(--tpo)" strokeWidth={1.5} strokeLinejoin="round" />
          <circle cx={x(values.length - 1)} cy={y(last)} r={3} fill="var(--tpo)" stroke="var(--panel)" strokeWidth={1.5} />
        </svg>
      )}
      <div className="v">{fix(last, digits)}{unit}<span className="chg">60 分で {signed(chg, digits)}</span></div>
    </div>
  );
}

export default function SignalPanel({ s, now }) {
  if (!s) return null;
  if (s.unavailable) {
    return (
      <div className="panel">
        <div className="panel-head"><h2>1 分ごとのシグナル</h2><span className="pill wait">未接続</span></div>
        <p className="note">/api/signals がまだありません。モデルを Lambda に載せると、この列に直近 60 分の上げ確率と水準の推移が出ます。</p>
      </div>
    );
  }
  const dec = s.decision, sm = s.summary;
  const remain = s.next_decision_ms - now;
  const candidate = sm && sm.B_last >= s.threshold;
  const cells = [["平均", sm?.B_mean], ["最新", sm?.B_last], ["変化", sm?.B_slope], ["ばらつき", sm?.B_std], ["直近 15 分", sm?.B_mean_recent]];
  const col = (k) => s.minutes.map((m) => m[k]);
  return (
    <div className="signals">
      <div className="panel">
        <div className="panel-head">
          <h2>次の判断</h2>
          <span className={`pill ${candidate ? "ok" : "wait"}`}>{candidate ? "買い候補" : "待機"}</span>
        </div>
        <div className="decision">
          <div>
            <div className="label">直近 60 分の上げ確率（最新）</div>
            <div className="big">{fix(sm?.B_last, 3)}<small>しきい値 {fix(s.threshold, 2)}</small></div>
          </div>
          <div className="countdown">
            <div className="label">{hm(s.next_decision_ms)} の判断まで</div>
            <div className="big num">{mmss(remain)}</div>
          </div>
        </div>
        <div className="summary">
          {cells.map(([k, v]) => (
            <div className="cell" key={k}><div className="k">{k}</div><div className="v">{k === "変化" ? signed(v, 3) : fix(v, 3)}</div></div>
          ))}
        </div>
        <p className="note">この 5 つの要約と水準の推移を、正時に判断するモデル A に渡す。</p>
      </div>

      <div className="panel">
        <div className="panel-head">
          <h2>上げ確率の推移</h2>
          <span className="sub">モデル B、1 分ごと、{s.horizon_min} 分後に上がる確率</span>
        </div>
        <ProbabilityChart minutes={s.minutes} threshold={s.threshold} />
      </div>

      <div className="panel">
        <div className="panel-head">
          <h2>水準の推移</h2>
          <span className="sub">現在値からの距離は σ 単位、上が高い</span>
        </div>
        <div className="levels">
          <Spark label="出来高 POC までの距離" values={col("vp_poc_dist")} unit="σ" zero />
          <Spark label="TPO POC までの距離" values={col("tpo_poc_dist")} unit="σ" zero />
          <Spark label="出来高バリューエリア内の位置（0=VAL, 1=VAH）" values={col("vp_va_pos")} />
          <Spark label="TPO バリューエリア内の位置" values={col("tpo_va_pos")} />
          <Spark label="現在の価格帯の厚さ（平均比）" values={col("vp_at_price")} />
          <Spark label="上 1σ のシングルプリントの割合" values={col("tpo_single_up")} />
        </div>
      </div>

      {s.history?.length > 0 && (
        <div className="panel">
          <div className="panel-head"><h2>過去の判断</h2><span className="sub">正時の A の確率と 1 時間後の値動き</span></div>
          <div style={{ overflowX: "auto" }}>
            <table className="hist">
              <thead><tr><th>時刻</th><th>確率</th><th>判断</th><th>1 時間後</th></tr></thead>
              <tbody>
                {[...s.history].reverse().map((h) => (
                  <tr key={h.t}>
                    <td className="num">{hm(h.t)}</td>
                    <td className="num">{fix(h.p_up, 3)}</td>
                    <td><span className={`pill ${h.action === "buy" ? "ok" : "wait"}`}>{h.action === "buy" ? "買い" : "見送り"}</span></td>
                    <td className="num" style={{ color: h.r == null ? "var(--ink-3)" : h.r >= 0 ? "var(--up)" : "var(--down)" }}>
                      {h.r == null ? "未確定" : pct(h.r, 2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {dec && <p className="note">直近の判断 {hm(dec.t)}: 確率 {fix(dec.p_up, 3)}、{dec.action === "buy" ? "買い" : "見送り"}。
        モデルは {s.model?.trained_to} まで学習。</p>}
    </div>
  );
}
