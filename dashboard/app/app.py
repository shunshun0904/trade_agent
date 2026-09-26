"""TPO・価格帯別出来高ダッシュボードの Lambda（API Gateway の REST API から呼ばれる）。

- GET /             画面（page.html）
- GET /api/profile  直近 24 時間の価格帯別出来高・TPO・15 分足（JSON）
- GET /api/signals  直近 60 分の 1 分ごとの水準（POC までの距離など）と、1 時間後に 0.3% を超えて上がる確率（p_big_1h、
                    ボラティリティの目安。inference.py、モデルは reports/direction_deploy/model_B.json）。分ごとの結果は
                    実行環境の中に残し、新しい分だけ計算する

画面が REFRESH_SECONDS（既定 10 秒）ごとに /api/profile を呼ぶ。タブが裏にある間は呼ばない。約定は bitbank の公開 REST API（認証不要）から取り、S3 に保存して
次回は差分だけ取る。σ の計算に 24 時間より前の足も要るので、直近 49 時間分を保持する。
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
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from inference import load_model, predict_minutes
from profile_core import compute, signals

PAIR = os.environ.get("PAIR", "btc_jpy")
BUCKET = os.environ.get("CACHE_BUCKET", "")
CACHE_KEY = f"cache/{PAIR}.json.gz"
PUBLIC = "https://public.bitbank.cc"
KEEP_MS = 52 * 3_600_000  # 特徴量のローリング窓（最大 192 本 = 48 時間）と 24 時間の窓のため
ROW_LEN = 5  # [transaction_id, executed_at, price, amount, is_buy]
REFRESH_SECONDS = int(os.environ.get("REFRESH_SECONDS", "10"))
MIN_FETCH_INTERVAL = REFRESH_SECONDS
SAVE_INTERVAL = 60
PAGE = (Path(__file__).parent / "page.html").read_text(encoding="utf-8").replace(
    "__REFRESH_SECONDS__", str(REFRESH_SECONDS))
_MEM: dict = {}  # 同じ実行環境で続けて呼ばれたときに使い回す保存データ


def http_get_json(path: str) -> dict:
    req = urllib.request.Request(PUBLIC + path, headers={"User-Agent": "bitbank-profile-dashboard"})
    with urllib.request.urlopen(req, timeout=10) as r:
        body = json.loads(r.read())
    if body.get("success") != 1:
        raise RuntimeError(f"{path}: code={(body.get('data') or {}).get('code')}")
    return body["data"]


def _rows(txs: list[dict]) -> list[list]:
    """[transaction_id, executed_at, price, amount, is_buy]"""
    return [[int(t["transaction_id"]), int(t["executed_at"]), float(t["price"]), float(t["amount"]), t.get("side") == "buy"]
            for t in txs]


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
    if cache and any(len(r) != ROW_LEN for r in cache.get("rows", [])):
        cache = None  # 古い形式（side なし）の保存データは捨てて取り直す
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
    today = _day(now_ms)
    for day in sorted(need_days):
        try:
            for r in _rows(get(f"/{PAIR}/transactions/{day}")["transactions"]):
                rows[r[0]] = r
        except RuntimeError:
            if day != today:  # 当日分は日付が変わった直後だとまだないことがある。それ以外の失敗は上に伝える
                raise
    keep = [r for r in rows.values() if r[1] >= now_ms - KEEP_MS]
    keep.sort(key=lambda r: (r[1], r[0]))
    return {"rows": keep, "from_ms": max(covered_from, now_ms - KEEP_MS), "fetched_ms": now_ms}


def _add_predictions(res: dict, rows: list[list], mem: dict) -> None:
    """res["minutes"] の各行に p_big_1h を足す。分ごとの結果は mem["pred"] に残す。モデルがなければ何もしない。"""
    model = mem.get("model")
    if model is None:
        model = mem["model"] = load_model() or {}
    if not model:
        return
    known: dict = mem.setdefault("pred", {})
    todo = [m["t"] for m in res["minutes"] if m["t"] not in known]
    if todo:
        known.update(predict_minutes(rows, todo, model))
    for m in res["minutes"]:
        m["p_big_1h"] = known.get(m["t"])
    for t in [k for k in known if k < res["minutes"][0]["t"]]:
        del known[t]
    res["model"] = {"horizon_min": model.get("horizon_min"), "target_min_return": model.get("target_min_return"),
                    "train_end": model.get("train_end"), "test_auc": model.get("test_auc"),
                    "base_rate_test": model.get("base_rate_test")}


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
                _add_predictions(res, cache["rows"], mem)
        else:
            res = compute(trades, now_ms)
        if "error" in res:
            return _resp(502, json.dumps(res), "application/json")
        res["pair"] = PAIR
        res["data_fetched_ms"] = cache["fetched_ms"]
        return _resp(200, json.dumps(res), "application/json")
    except Exception as exc:  # 画面に理由を出す（キーや残高は扱わないので出して問題ない）
        return _resp(502, json.dumps({"error": repr(exc)}), "application/json")
