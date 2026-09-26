import { useMemo, useState } from "react";
import { useSize } from "./useSize.js";
import { fix, pct, signed, hm, yen } from "./format.js";

/** 1 時間後に 0.3% を超えて上がる確率（ボラティリティの目安）の 60 分の折れ線。 */
function ProbabilityChart({ minutes, base, height = 150 }) {
  const [ref, W] = useSize();
  const [hover, setHover] = useState(null);
  const M = { top: 8, right: 10, bottom: 20, left: 36 };
  const pts = minutes.filter((m) => m.p_big_1h != null);
  const g = useMemo(() => {
    if (!W || pts.length < 2) return null;
    const innerW = W - M.left - M.right, H = height - M.top - M.bottom;
    const t0 = minutes[0].t, t1 = minutes[minutes.length - 1].t;
    const hi = Math.max(0.5, ...pts.map((m) => m.p_big_1h));
    const x = (t) => M.left + ((t - t0) / (t1 - t0 || 1)) * innerW;
    const y = (p) => M.top + (1 - p / hi) * H;
    const path = pts.map((m, i) => `${i ? "L" : "M"}${x(m.t).toFixed(1)},${y(m.p_big_1h).toFixed(1)}`).join("");
    const area = `${path}L${x(pts[pts.length - 1].t).toFixed(1)},${y(0).toFixed(1)}L${x(pts[0].t).toFixed(1)},${y(0).toFixed(1)}Z`;
    const ticks = [0.2, 0.4, 0.6, 0.8].filter((p) => p <= hi);
    return { innerW, H, x, y, path, area, hi, ticks };
  }, [W, minutes, height]);
  if (pts.length < 2) return <div ref={ref} style={{ height }} />;
  const last = pts[pts.length - 1];
  const onMove = (e) => {
    if (!g) return;
    const px = e.clientX - e.currentTarget.getBoundingClientRect().left;
    const i = Math.round(((px - M.left) / g.innerW) * (minutes.length - 1));
    const m = minutes[Math.min(minutes.length - 1, Math.max(0, i))];
    setHover(m.p_big_1h == null ? null : m);
  };
  return (
    <div ref={ref} className="chart-wrap" style={{ height }}>
      {g && (
        <svg width={W} height={height} onMouseMove={onMove} onMouseLeave={() => setHover(null)} role="img"
             aria-label="1 時間後に 0.3% を超えて上がる確率の推移">
          <g className="grid">
            {g.ticks.map((p) => (
              <g key={p}>
                <line x1={M.left} x2={W - M.right} y1={g.y(p)} y2={g.y(p)} strokeDasharray="2 3" />
                <text x={M.left - 6} y={g.y(p) + 3.5} textAnchor="end">{p.toFixed(1)}</text>
              </g>
            ))}
          </g>
          {base != null && (
            <>
              <line x1={M.left} x2={W - M.right} y1={g.y(base)} y2={g.y(base)} stroke="var(--ink-3)" strokeWidth={1} strokeDasharray="4 3" />
              <text x={M.left + 4} y={g.y(base) - 4}>平均 {pct(base, 0)}（学習後の評価期間）</text>
            </>
          )}
          <path d={g.area} fill="var(--vp-soft)" opacity={0.55} />
          <path d={g.path} fill="none" stroke="var(--vp)" strokeWidth={2} strokeLinejoin="round" />
          <circle cx={g.x(last.t)} cy={g.y(last.p_big_1h)} r={4} fill="var(--vp)" stroke="var(--panel)" strokeWidth={2} />
          <g className="axis">
            {minutes.filter((m) => m.t % (15 * 60_000) === 0).map((m) => (
              <text key={m.t} x={g.x(m.t)} y={height - 5} textAnchor="middle">{hm(m.t)}</text>
            ))}
          </g>
          {hover && (
            <g>
              <line x1={g.x(hover.t)} x2={g.x(hover.t)} y1={M.top} y2={M.top + g.H} stroke="var(--ink-3)" strokeWidth={0.75} />
              <circle cx={g.x(hover.t)} cy={g.y(hover.p_big_1h)} r={3.5} fill="var(--panel)" stroke="var(--vp)" strokeWidth={2} />
            </g>
          )}
        </svg>
      )}
      {hover && g && (
        <div className="tip" style={{ left: Math.min(g.x(hover.t) + 12, W - 150), top: 4 }}>
          <div><b>{hm(hover.t)}</b></div>
          <div>確率 <span className="num">{pct(hover.p_big_1h, 1)}</span></div>
          <div>価格 <span className="num">{yen(hover.price)}</span></div>
        </div>
      )}
    </div>
  );
}

/** 1 つの水準の 60 分の推移（小さな折れ線）。 */
function Spark({ label, values, unit = "", digits = 2, zero = false, fmt }) {
  const [ref, W] = useSize();
  const H = 34, P = 3;
  const v = values.filter((x) => x != null && isFinite(x));
  if (!v.length) return null;
  const lo = Math.min(...v, zero ? 0 : Infinity), hi = Math.max(...v, zero ? 0 : -Infinity);
  const y = (x) => P + (1 - (x - lo) / (hi - lo || 1)) * (H - 2 * P);
  const x = (i) => (W ? (i / (values.length - 1)) * (W - 2) + 1 : 0);
  let path = "", pen = false;
  values.forEach((q, i) => {
    if (q == null || !isFinite(q)) { pen = false; return; }
    path += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(q).toFixed(1)}`;
    pen = true;
  });
  const last = v[v.length - 1], chg = last - v[0];
  const show = fmt || ((q) => fix(q, digits) + unit);
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
      <div className="v">{show(last)}<span className="chg">60 分で {fmt ? fmt(chg, true) : signed(chg, digits)}</span></div>
    </div>
  );
}

export default function SignalPanel({ s }) {
  if (!s) return null;
  const m = s.minutes || [];
  if (!m.length) {
    return (
      <div className="panel">
        <div className="panel-head"><h2>水準の推移</h2><span className="pill wait">データ待ち</span></div>
        <p className="note">σ を計算できる足がそろうまで（約 24 時間）表示できません。</p>
      </div>
    );
  }
  const col = (k) => m.map((r) => r[k]);
  const last = m[m.length - 1];
  const first = m[0];
  const hasModel = m.some((r) => r.p_big_1h != null);
  const md = s.model || {};
  return (
    <div className="signals">
      <div className="panel">
        <div className="panel-head">
          <h2>1 時間後に 0.3% 超上がる確率</h2>
          {hasModel
            ? <span className="sub">ボラティリティの目安。方向の予測には使えない</span>
            : <span className="pill wait">モデル未配置</span>}
        </div>
        {hasModel ? (
          <>
            <div className="summary" style={{ marginTop: 0, marginBottom: 8 }}>
              <div className="cell"><div className="k">最新</div><div className="v">{pct(last.p_big_1h, 1)}</div></div>
              <div className="cell"><div className="k">60 分前</div><div className="v">{pct(first.p_big_1h, 1)}</div></div>
              <div className="cell"><div className="k">評価期間の平均</div><div className="v">{pct(md.base_rate_test, 1)}</div></div>
              <div className="cell"><div className="k">AUC</div><div className="v">{fix(md.test_auc, 3)}</div></div>
              <div className="cell"><div className="k">学習</div><div className="v" style={{ fontSize: 12 }}>〜{(md.train_end || "").slice(0, 10)}</div></div>
            </div>
            <ProbabilityChart minutes={m} base={md.base_rate_test} />
            <p className="note">
              この確率が高いときは、同じくらいの確率で 0.3% 超下がりもする（上げ確率と下げ確率の相関 0.7〜0.8）。
              大きく動きそうかの目安として見る。
            </p>
          </>
        ) : (
          <p className="note">reports/direction_deploy/model_B.json を含めてデプロイすると表示されます。</p>
        )}
      </div>
      <div className="panel">
        <div className="panel-head">
          <h2>直近 60 分</h2>
          <span className="sub">{hm(first.t)}〜{hm(last.t)}、1 分ごと</span>
        </div>
        <div className="summary">
          <div className="cell"><div className="k">現在値</div><div className="v">{yen(last.price)}</div></div>
          <div className="cell"><div className="k">60 分の変化</div><div className="v">{signed((last.price / first.price - 1) * 100, 2)}%</div></div>
          <div className="cell"><div className="k">σ（15 分）</div><div className="v">{fix(last.sigma * 100, 3)}%</div></div>
          <div className="cell"><div className="k">出来高 POC まで</div><div className="v">{signed(last.vp_poc_dist, 2)}σ</div></div>
          <div className="cell"><div className="k">TPO POC まで</div><div className="v">{signed(last.tpo_poc_dist, 2)}σ</div></div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <h2>水準の推移</h2>
          <span className="sub">現在値からの距離は σ 単位、上が高い</span>
        </div>
        <div className="levels">
          <Spark label="価格" values={col("price")} fmt={(q, d) => (d ? signed(q, 0) : yen(q))} />
          <Spark label="出来高 POC までの距離" values={col("vp_poc_dist")} unit="σ" zero />
          <Spark label="TPO POC までの距離" values={col("tpo_poc_dist")} unit="σ" zero />
          <Spark label="出来高バリューエリア内の位置（0=VAL, 1=VAH）" values={col("vp_va_pos")} />
          <Spark label="TPO バリューエリア内の位置" values={col("tpo_va_pos")} />
          <Spark label="現在の価格帯の厚さ（平均比）" values={col("vp_at_price")} />
          <Spark label="上 1σ のシングルプリントの割合" values={col("tpo_single_up")} />
        </div>
        <p className="note">
          POC までの距離が 0 に近いほど現在値は最も取引された価格帯にある。バリューエリア内の位置が 0 未満・1 超なら
          エリアの外。シングルプリントの割合が高いほど、上側は薄く動きやすい。
        </p>
      </div>
    </div>
  );
}
