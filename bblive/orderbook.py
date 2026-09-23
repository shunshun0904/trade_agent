"""Public Stream の depth_whole / depth_diff から板を組み立てる。

公式ドキュメント public-stream_JP.md「板情報の処理方法」の手順に従う:
- depth_diff を受け取ったら、数量が非 0 なら追加・上書き、0 なら削除する。バッファにも残す。
- depth_whole を受け取ったら板を置き換え、バッファした diff のうち s が sequenceId より大きいものを s の昇順で再適用する。
- s（シーケンス ID）は単調増加だが連続とは限らない。whole の sequenceId 以下の diff はバッファから捨ててよい。
"""
from __future__ import annotations


class OrderBook:
    def __init__(self) -> None:
        self.asks: dict[float, float] = {}
        self.bids: dict[float, float] = {}
        self.seq: int | None = None          # 最後に置き換えた whole の sequenceId
        self.last_s: int | None = None       # 最後に適用した diff の s
        self._buffer: list[tuple[int, dict]] = []
        self.has_whole = False

    @staticmethod
    def _apply(side: dict[float, float], levels) -> None:
        for price, amount in levels:
            p, a = float(price), float(amount)
            if a == 0:
                side.pop(p, None)
            else:
                side[p] = a

    def on_diff(self, data: dict) -> None:
        s = int(data["s"])
        self._buffer.append((s, data))
        if self.seq is not None and s <= self.seq:
            return
        self._apply(self.asks, data.get("a", []))
        self._apply(self.bids, data.get("b", []))
        self.last_s = s

    def on_whole(self, data: dict) -> None:
        seq = int(data["sequenceId"])
        self.asks = {float(p): float(a) for p, a in data.get("asks", [])}
        self.bids = {float(p): float(a) for p, a in data.get("bids", [])}
        self.seq = seq
        self.has_whole = True
        self._buffer = sorted((x for x in self._buffer if x[0] > seq), key=lambda x: x[0])
        for s, d in self._buffer:
            self._apply(self.asks, d.get("a", []))
            self._apply(self.bids, d.get("b", []))
            self.last_s = s

    def best(self) -> tuple[float | None, float | None]:
        """(最良買気配, 最良売気配)"""
        return (max(self.bids) if self.bids else None, min(self.asks) if self.asks else None)
