"""TPO・価格帯別出来高ダッシュボードの Lambda（API Gateway の REST API から呼ばれる）。

- GET /             画面（page.html）
- GET /api/profile  直近 3 時間の価格帯別出来高・TPO・1 分足（JSON）
- GET /api/signals  直近 60 分の 1 分ごとの水準（POC までの距離など。JSON）。分ごとの結果は実行環境の中に残し、新しい分だけ計算する

画面が REFRESH_SECONDS（既定 30 秒、2026-09-26 オーナー決定）ごとに /api/profile を呼ぶ。タブが裏にある間は呼ばない。約定は bitbank の公開 REST API（認証不要）から取り、S3 に保存して
次回は差分だけ取る。表示の窓 3 時間と σ（1 分足 180 本）に足りるように直近 4 時間分を保持する（初回の取得を軽くするため。2026-09-26）。
- 保存がない・古いとき: 日付指定（UTC の日付、Phase 0 V5）で必要な日を取る
- それ以外: 最新 60 件を取り、保存済みの最新より新しいものを足す。60 件の中に保存済みの最新が含まれなければ
  取りこぼしの可能性があるので、当日（と必要なら前日）の日付指定で取り直す
- 公開 API への負荷を抑えるため、前回の取得から MIN_FETCH_INTERVAL 秒以内なら取りに行かない
- 保存データは Lambda の実行環境の中にも持ち、S3 への書き込みは SAVE_INTERVAL 秒に 1 回までにする
  （呼び出しごとの S3 の読み書きを減らして費用を抑える）
"""
from __future__ import annotations

import gzip
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from profile_core import compute, signals

PAIR = os.environ.get("PAIR", "btc_jpy")
BUCKET = os.environ.get("CACHE_BUCKET", "")
CACHE_KEY = f"cache/{PAIR}.json.gz"
PUBLIC = "https://public.bitbank.cc"
KEEP_MS = 4 * 3_600_000  # 窓 3 時間 + σ の 180 分 + TPO の 5 分に足りる。日付指定の取得は多くても 2 日分
REFRESH_SECONDS = int(os.environ.get("REFRESH_SECONDS", "30"))
MIN_FETCH_INTERVAL = REFRESH_SECONDS
SAVE_INTERVAL = 60
PAGE = (Path(__file__).parent / "page.html").read_text(encoding="utf-8").replace(
    "__REFRESH_SECONDS__", str(REFRESH_SECONDS))
_MEM: dict = {}  # 同じ実行環境で続けて呼ばれたときに使い回す保存データ


class NoData(RuntimeError):
    """その日付のデータがない（HTTP 404 か success != 1）。日付指定の取得で、当日分がまだないときなどに起きる。"""


def http_get_json(path: str) -> dict:
    req = urllib.request.Request(PUBLIC + path, headers={"User-Agent": "bitbank-profile-dashboard"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            body = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise NoData(f"{path}: HTTP 404") from None
        raise
    if body.get("success") != 1:
        raise NoData(f"{path}: code={(body.get('data') or {}).get('code')}")
    return body["data"]


def _rows(txs: list[dict]) -> list[list]:
    """[transaction_id, executed_at, price, amount]"""
    return [[int(t["transaction_id"]), int(t["executed_at"]), float(t["price"]), float(t["amount"])] for t in txs]


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y%m%d")


class Store:
    """S3 に置く保存データ（テストでは差し替える）。"""

    def __init__(self, bucket: str) -> None:
        import boto3

        self.s3 = boto3.client("s3")
        self.bucket = bucket

    def load(self) -> dict | None:
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=CACHE_KEY)
        except self.s3.exceptions.NoSuchKey:
            return None
        return json.loads(gzip.decompress(obj["Body"].read()))

    def save(self, data: dict) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=CACHE_KEY, Body=gzip.compress(json.dumps(data).encode()),
                           ContentType="application/json", ContentEncoding="gzip")


def refresh(cache: dict | None, now_ms: int, get=http_get_json) -> dict:
    """保存データを now 時点まで更新して返す。"""
    if cache and now_ms - cache.get("fetched_ms", 0) < MIN_FETCH_INTERVAL * 1000:
        return cache
    rows: dict[int, list] = {r[0]: r for r in (cache or {}).get("rows", [])}
    covered_from = (cache or {}).get("from_ms")
    need_days: set[str] = set()
    if not cache or covered_from is None or covered_from > now_ms - KEEP_MS:
        d = datetime.fromtimestamp((now_ms - KEEP_MS) / 1000, tz=timezone.utc).date()
        while d <= datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).date():
            need_days.add(d.strftime("%Y%m%d"))
            d += timedelta(days=1)
        covered_from = now_ms - KEEP_MS
    else:
        latest = _rows(get(f"/{PAIR}/transactions")["transactions"])
        last_id = max(rows) if rows else -1
        for r in latest:
            rows[r[0]] = r
        if latest and min(r[0] for r in latest) > last_id:  # 保存済みの最新が 60 件に入っていない
            last_ts = max(r[1] for r in rows.values() if r[0] <= last_id) if last_id >= 0 else now_ms - KEEP_MS
            need_days.update({_day(last_ts), _day(now_ms)})
    skipped = []
    for day in sorted(need_days):
        try:
            for r in _rows(get(f"/{PAIR}/transactions/{day}")["transactions"]):
                rows[r[0]] = r
        except NoData as exc:  # その日のデータがない（当日分がまだない、など）。飛ばして続ける
            skipped.append(f"{day}: {exc}")
    if need_days:  # 日付指定で取ったときも最新 60 件を足す（当日分の日付指定が使えない場合の備え）
        for r in _rows(get(f"/{PAIR}/transactions")["transactions"]):
            rows[r[0]] = r
    keep = [r for r in rows.values() if r[1] >= now_ms - KEEP_MS]
    keep.sort(key=lambda r: (r[1], r[0]))
    return {"rows": keep, "from_ms": max(covered_from, now_ms - KEEP_MS), "fetched_ms": now_ms, "skipped": skipped}


def _resp(status: int, body: str, ctype: str) -> dict:
    return {"statusCode": status, "headers": {"Content-Type": ctype, "Cache-Control": "no-store"}, "body": body}


def handler(event, context, store=None, get=http_get_json, now_ms: int | None = None):
    path = (event or {}).get("path") or "/"
    which = path.rstrip("/").rsplit("/", 1)[-1]
    if which not in ("profile", "signals"):
        return _resp(200, PAGE, "text/html; charset=utf-8")
    now_ms = now_ms or int(time.time() * 1000)
    mem = _MEM if store is None else {}
    store = store or Store(BUCKET)
    try:
        before = mem.get("cache") or store.load()
        cache = refresh(before, now_ms, get)
        mem["cache"] = cache
        if cache is not before and now_ms - mem.get("saved_ms", 0) >= SAVE_INTERVAL * 1000:
            store.save(cache)
            mem["saved_ms"] = now_ms
        trades = [(r[1], r[2], r[3]) for r in cache["rows"]]
        if which == "signals":
            res = signals(trades, now_ms, known=mem.setdefault("signals", {}))
            if not res["minutes"]:
                res["error"] = "σ を計算するデータが足りない"
        else:
            res = compute(trades, now_ms)
        if "error" in res:
            return _resp(502, json.dumps(res), "application/json")
        res["pair"] = PAIR
        res["data_fetched_ms"] = cache["fetched_ms"]
        res["first_trade_ms"] = cache["rows"][0][1] if cache["rows"] else None
        if cache.get("skipped"):
            res["warnings"] = cache["skipped"]
        return _resp(200, json.dumps(res), "application/json")
    except Exception as exc:  # 画面に理由を出す（キーや残高は扱わないので出して問題ない）
        return _resp(502, json.dumps({"error": repr(exc)}), "application/json")
