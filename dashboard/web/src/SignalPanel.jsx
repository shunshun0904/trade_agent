import { useSize } from "./useSize.js";
import { fix, signed, hm, yen } from "./format.js";

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
  return (
    <div className="signals">
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
