import { useEffect, useState } from "react";
import { fetchProfile, fetchSignals, isMock } from "./api.js";
import ProfileChart from "./ProfileChart.jsx";
import SignalPanel from "./SignalPanel.jsx";
import { yen, hms } from "./format.js";

// Lambda が配信するとき "__REFRESH_SECONDS__" を設定値に置き換える（プレビューでは置き換わらず 30 秒）
const REFRESH_MS = (Number("__REFRESH_SECONDS__") || 30) * 1000;

export default function App() {
  const [profile, setProfile] = useState(null);
  const [signals, setSignals] = useState(null);
  const [err, setErr] = useState(null);

  // タブが裏にある間は取得しない（呼び出し回数＝費用を増やさない）。表に戻ったらすぐ取得する
  useEffect(() => {
    let timer = null, alive = true;
    const load = async () => {
      try {
        const [p, s] = await Promise.all([fetchProfile(), fetchSignals()]);
        if (!alive) return;
        setProfile(p); setSignals(s); setErr(null);
      } catch (e) {
        if (alive) setErr(e.message);
      }
    };
    const start = () => { if (!timer) { load(); timer = setInterval(load, REFRESH_MS); } };
    const stop = () => { if (timer) { clearInterval(timer); timer = null; } };
    const onVis = () => (document.hidden ? stop() : start());
    document.addEventListener("visibilitychange", onVis);
    if (!document.hidden) start();
    return () => { alive = false; stop(); document.removeEventListener("visibilitychange", onVis); };
  }, []);

  const vl = profile?.vp_levels, tl = profile?.tpo_levels;
  return (
    <div className="app">
      <header className="topbar">
        <h1>BTC/JPY 価格帯別出来高・TPO</h1>
        {profile && <span className="price">{yen(profile.price)}<small>円</small></span>}
        <span className="meta">
          {isMock && <span className="pill wait">ダミーデータ</span>}
          {!isMock && !err && profile && <span className="pill ok live">{REFRESH_MS / 1000} 秒ごとに更新</span>}
          {err && <span className="pill err">取得に失敗</span>}
          {profile && <span>更新 <span className="num">{hms(profile.now_ms)}</span></span>}
          {profile && <span>刻み <span className="num">{yen(profile.bin_width)}</span> 円（0.25σ）</span>}
          {profile && <span>約定 <span className="num">{profile.n_trades_window.toLocaleString("ja-JP")}</span> 件 / {profile.window_h}h</span>}
        </span>
      </header>
      {err && <div className="err">取得に失敗しました: {err}（{REFRESH_MS / 1000} 秒後に再試行）</div>}
      {profile && profile.first_trade_ms != null && profile.first_trade_ms > profile.now_ms - profile.window_h * 3_600_000 && (
        <div className="err">保持している約定は {hms(profile.first_trade_ms)} からです。それより前の時間帯は集計に入っていません
          {profile.warnings ? `（取得できなかった日付: ${profile.warnings.join("、")}）` : ""}。</div>
      )}
      <div className="main">
        <section className="panel">
          <div className="panel-head">
            <h2>直近 {profile ? profile.window_h : 3} 時間（{profile && profile.candle_ms ? profile.candle_ms / 60_000 : 1} 分足）</h2>
            <div className="legend">
              <span><i className="sw box" style={{ background: "var(--up)" }} />陽線</span>
              <span><i className="sw box" style={{ background: "var(--down)" }} />陰線</span>
              <span><i className="sw" style={{ borderColor: "var(--poc)" }} />POC</span>
              <span><i className="sw" style={{ borderColor: "var(--vp)" }} />出来高 VAH / VAL</span>
              <span><i className="sw dot" style={{ borderColor: "var(--tpo)" }} />TPO VAH / VAL</span>
              <span><i className="sw box" style={{ background: "var(--vp)" }} />出来高のバリューエリア</span>
              <span><i className="sw box" style={{ background: "var(--tpo)" }} />TPO のバリューエリア</span>
            </div>
          </div>
          <ProfileChart d={profile} />
          {profile && (
            <div className="legend" style={{ marginTop: 8 }}>
              <span>出来高 POC <b className="num">{yen(vl.poc)}</b></span>
              <span>VAH <b className="num">{yen(vl.vah)}</b></span>
              <span>VAL <b className="num">{yen(vl.val)}</b></span>
              <span>TPO POC <b className="num">{yen(tl.poc)}</b></span>
              <span>VAH <b className="num">{yen(tl.vah)}</b></span>
              <span>VAL <b className="num">{yen(tl.val)}</b></span>
            </div>
          )}
        </section>
        <SignalPanel s={signals} />
      </div>
    </div>
  );
}
