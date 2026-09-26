export const yen = (x) => (x == null || !isFinite(x) ? "-" : Math.round(x).toLocaleString("ja-JP"));
export const pct = (x, d = 1) => (x == null || !isFinite(x) ? "-" : `${(x * 100).toFixed(d)}%`);
export const fix = (x, d = 2) => (x == null || !isFinite(x) ? "-" : x.toFixed(d));
export const signed = (x, d = 2) => (x == null || !isFinite(x) ? "-" : `${x >= 0 ? "+" : ""}${x.toFixed(d)}`);
export const hm = (ms) => new Date(ms).toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit" });
export const hms = (ms) => new Date(ms).toLocaleTimeString("ja-JP");
export const mmss = (ms) => {
  const s = Math.max(0, Math.floor(ms / 1000));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
};
