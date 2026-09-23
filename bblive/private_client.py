"""bitbank Private REST（認証付き）の最小クライアント。

認証は ACCESS-TIME-WINDOW 方式（SPEC.md §3.2）。
    GET : 署名対象 = REQUEST-TIME + TIME-WINDOW + パス（/v1 を含む）+ クエリ文字列
    POST: 署名対象 = REQUEST-TIME + TIME-WINDOW + リクエストボディの JSON 文字列
POST は署名した JSON 文字列をそのまま送る（シリアライズは1回だけ）。

API キー・シークレットはログに出さない。__repr__ にも含めない。
出金 API を呼ぶメソッドは作らない。
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any, Callable
from urllib.parse import urlencode

import requests

from bbdata.client import BitbankAPIError

PRIVATE_BASE_URL = "https://api.bitbank.cc"
API_PREFIX = "/v1"

log = logging.getLogger(__name__)


def sign(secret: str, message: str) -> str:
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


class PrivateClient:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = PRIVATE_BASE_URL,
        time_window_ms: int = 5000,
        timeout: float = 10.0,
        session: requests.Session | None = None,
        clock_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    ) -> None:
        if not api_key or not api_secret:
            raise ValueError("API キーとシークレットが必要です")
        self._key = api_key
        self._secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.time_window_ms = int(time_window_ms)
        self.timeout = timeout
        self.session = session or requests.Session()
        self._clock_ms = clock_ms

    def __repr__(self) -> str:  # キーを表示しない
        return f"PrivateClient(base_url={self.base_url!r})"

    def _headers(self, message_tail: str) -> dict[str, str]:
        t = str(self._clock_ms())
        w = str(self.time_window_ms)
        return {
            "ACCESS-KEY": self._key,
            "ACCESS-REQUEST-TIME": t,
            "ACCESS-TIME-WINDOW": w,
            "ACCESS-SIGNATURE": sign(self._secret, t + w + message_tail),
            "Content-Type": "application/json",
        }

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        full = API_PREFIX + path
        query = ("?" + urlencode(params)) if params else ""
        resp = self.session.get(self.base_url + full + query, headers=self._headers(full + query), timeout=self.timeout)
        return self._parse(path, resp)

    def post(self, path: str, body: dict[str, Any]) -> dict:
        payload = json.dumps(body, separators=(",", ":"))
        resp = self.session.post(self.base_url + API_PREFIX + path, data=payload,
                                 headers=self._headers(payload), timeout=self.timeout)
        return self._parse(path, resp)

    @staticmethod
    def _parse(path: str, resp: requests.Response) -> dict:
        try:
            body = resp.json()
        except ValueError:
            raise BitbankAPIError(path, resp.status_code, "non-json response") from None
        if body.get("success") == 1:
            return body["data"]
        raise BitbankAPIError(path, resp.status_code, (body.get("data") or {}).get("code"))

    # --------------------------------------------------------------- 参照系のみ
    def assets(self) -> list[dict]:
        return self.get("/user/assets")["assets"]

    def active_orders(self, pair: str) -> list[dict]:
        return self.get("/user/spot/active_orders", {"pair": pair})["orders"]
