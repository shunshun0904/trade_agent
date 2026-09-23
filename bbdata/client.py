"""bitbank Public API（認証不要）の最小クライアント。

仕様の出典: https://github.com/bitbankinc/bitbank-api-docs/blob/master/public-api_JP.md
"""
from __future__ import annotations

import logging
import random
import time
from collections import Counter
from typing import Any, Callable

import requests

PUBLIC_BASE_URL = "https://public.bitbank.cc"

# 公開APIのレートリミットは公式ドキュメントに数値の記載がない。
# そのためリクエスト間隔を空け、429 / 5xx は指数バックオフで再試行する。
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

log = logging.getLogger(__name__)


class BitbankAPIError(RuntimeError):
    """success != 1 のレスポンス（再試行しないエラー）。"""

    def __init__(self, path: str, status: int, code: Any):
        super().__init__(f"{path}: HTTP {status}, code={code}")
        self.path = path
        self.status = status
        self.code = code


class PublicClient:
    def __init__(
        self,
        base_url: str = PUBLIC_BASE_URL,
        min_interval: float = 0.3,
        max_retries: int = 5,
        timeout: float = 15.0,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = session or requests.Session()
        self._sleep = sleep
        self._last_request = float("-inf")
        # HTTP ステータス（通信例外は "error"）ごとの応答数。レート制限の観察（V4）に使う
        self.status_counts: Counter = Counter()

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            self._sleep(wait)
        self._last_request = time.monotonic()

    def get(self, path: str) -> dict:
        url = self.base_url + path
        reason = ""
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                self.status_counts["error"] += 1
                reason = repr(exc)
            else:
                self.status_counts[resp.status_code] += 1
                if resp.status_code not in RETRYABLE_STATUS:
                    return self._parse(path, resp)
                reason = f"HTTP {resp.status_code}"
            if attempt < self.max_retries:
                backoff = min(2.0**attempt, 30.0) + random.uniform(0.0, 0.5)
                log.warning(
                    "%s: %s, %.1f秒後に再試行 (%d/%d)",
                    path, reason, backoff, attempt + 1, self.max_retries,
                )
                self._sleep(backoff)
        raise RuntimeError(f"{path}: 再試行上限に到達 ({reason})")

    @staticmethod
    def _parse(path: str, resp: requests.Response) -> dict:
        try:
            body = resp.json()
        except ValueError:
            raise BitbankAPIError(path, resp.status_code, "non-json response") from None
        if body.get("success") == 1:
            return body["data"]
        raise BitbankAPIError(path, resp.status_code, (body.get("data") or {}).get("code"))

    def transactions(self, pair: str, yyyymmdd: str | None = None) -> list[dict]:
        """指定日の全約定。yyyymmdd を省略すると最新60件。"""
        path = f"/{pair}/transactions" + (f"/{yyyymmdd}" if yyyymmdd else "")
        return self.get(path)["transactions"]

    def candlestick(self, pair: str, candle_type: str, period: str) -> list[list]:
        """ohlcv 行 [始値, 高値, 安値, 終値, 出来高, UnixTimeミリ秒] のリスト。

        period は 1min〜1hour なら YYYYMMDD、4hour 以上なら YYYY。
        """
        data = self.get(f"/{pair}/candlestick/{candle_type}/{period}")
        blocks = data.get("candlestick") or []
        return blocks[0]["ohlcv"] if blocks else []

    def depth(self, pair: str) -> dict:
        """現在の板。asks / bids は [価格, 数量] の文字列ペア、timestamp は UnixTime ミリ秒。"""
        return self.get(f"/{pair}/depth")

    def ticker(self, pair: str) -> dict:
        return self.get(f"/{pair}/ticker")

    def circuit_break_info(self, pair: str) -> dict:
        return self.get(f"/{pair}/circuit_break_info")
