import { useEffect, useRef, useState } from "react";

/** 要素の幅を追う（描画は px 座標で行い、viewBox の引き伸ばしで文字がゆがまないようにする）。 */
export function useSize() {
  const ref = useRef(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((es) => setW(Math.floor(es[0].contentRect.width)));
    ro.observe(el);
    setW(Math.floor(el.getBoundingClientRect().width));
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}
