"""Public Stream（Socket.IO）の板・約定をそのまま記録する（認証不要）。

    python -m bblive.recorder --pair btc_jpy --hours 5.75 --out rec

受信したメッセージを、受信時刻（UTC、ミリ秒）と一緒に 1 行 1 メッセージの JSON として、
{out}/{YYYYMMDD}/{HH}.jsonl.gz に書く（受信時刻の UTC の時で切り替える）。
板は後から bblive.orderbook.OrderBook で組み立て直せるよう、加工せずに残す。

購読するルーム: depth_whole_{pair}, depth_diff_{pair}, transactions_{pair}, circuit_break_info_{pair}
切断されたら指数バックオフで再接続し、切断していた区間を stats に残す。
仕様の出典: https://github.com/bitbankinc/bitbank-api-docs/blob/master/public-stream_JP.md
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

STREAM_URL = "wss://stream.bitbank.cc"
ROOMS = ("depth_whole_{pair}", "depth_diff_{pair}", "transactions_{pair}", "circuit_break_info_{pair}")

log = logging.getLogger("recorder")


def now_ms() -> int:
    return int(time.time() * 1000)


class HourlyWriter:
    """受信時刻の UTC の時ごとに gzip の JSONL を切り替えて書く。"""

    def __init__(self, out: Path) -> None:
        self.out = Path(out)
        self._key: str | None = None
        self._fh = None
        self.files: list[str] = []

    def write(self, recv_ms: int, room: str, message: dict) -> None:
        t = datetime.fromtimestamp(recv_ms / 1000, tz=timezone.utc)
        key = t.strftime("%Y%m%d/%H")
        if key != self._key:
            self.close()
            path = self.out / f"{key}.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = gzip.open(path, "at", encoding="utf-8")
            self._key = key
            self.files.append(str(path))
        self._fh.write(json.dumps({"r": recv_ms, "room": room, "m": message}, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class Recorder:
    def __init__(self, pair: str, out: Path, hours: float) -> None:
        self.pair = pair
        self.rooms = [r.format(pair=pair) for r in ROOMS]
        self.writer = HourlyWriter(out)
        self.deadline = time.monotonic() + hours * 3600
        self.counts: Counter = Counter()
        self.disconnects: list[dict] = []
        self.started_ms = now_ms()
        self._down_since: int | None = None

    def on_message(self, payload: dict) -> None:
        room = payload.get("room_name")
        if room not in self.rooms:
            return
        self.counts[room] += 1
        self.writer.write(now_ms(), room, payload.get("message"))

    async def run(self) -> dict:
        import socketio  # python-socketio[asyncio_client]

        backoff = 1.0
        while time.monotonic() < self.deadline:
            sio = socketio.AsyncClient(reconnection=False)
            closed = asyncio.Event()

            @sio.event
            async def connect():
                for room in self.rooms:
                    await sio.emit("join-room", room)
                log.info("接続、%s を購読", self.rooms)

            @sio.event
            async def disconnect(*_):
                closed.set()

            @sio.on("message")
            async def message(data):
                self.on_message(data)

            try:
                await sio.connect(STREAM_URL, transports=["websocket"])
                if self._down_since is not None:
                    self.disconnects.append({"from_ms": self._down_since, "to_ms": now_ms()})
                    self._down_since = None
                backoff = 1.0
                remaining = self.deadline - time.monotonic()
                try:
                    await asyncio.wait_for(closed.wait(), timeout=max(0.0, remaining))
                    log.warning("切断された")
                except asyncio.TimeoutError:
                    pass
            except Exception as exc:  # 接続失敗も再試行する
                log.warning("接続できない: %r", exc)
            finally:
                if sio.connected:
                    await sio.disconnect()
            if time.monotonic() >= self.deadline:
                break
            if self._down_since is None:
                self._down_since = now_ms()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
        self.writer.close()
        if self._down_since is not None:
            self.disconnects.append({"from_ms": self._down_since, "to_ms": now_ms()})
        return self.stats()

    def stats(self) -> dict:
        return {"pair": self.pair, "started_ms": self.started_ms, "ended_ms": now_ms(),
                "counts": dict(self.counts), "disconnects": self.disconnects, "files": self.writer.files}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="bblive.recorder")
    ap.add_argument("--pair", default="btc_jpy")
    ap.add_argument("--hours", type=float, default=5.75)
    ap.add_argument("--out", default="rec")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    stats = asyncio.run(Recorder(args.pair, Path(args.out), args.hours).run())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"stats_{stats['started_ms']}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
