"""bitbank REST client (public + private, spot only).

Endpoints and the signing scheme follow
https://github.com/bitbankinc/bitbank-api-docs (public-api.md, rest-api.md).

Signing uses the ACCESS-TIME-WINDOW variant:
    signature = HMAC_SHA256(secret, request_time + time_window + path_or_body)
It is preferred over ACCESS-NONCE because a Lambda that retries does not have
to keep a monotonically increasing counter anywhere.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from datetime import date, datetime
from decimal import Decimal

import requests

from ..errors import ExchangeError, ExchangeRateLimited, InsufficientFunds
from ..models.market import Candle
from ..models.trading import OrderIntent, OrderType
from ..money import dec, to_str
from .base import Balance, PairSettings
from .ratelimit import RateLimiter

log = logging.getLogger(__name__)

# Candle types whose history endpoint is addressed by day rather than by year.
DAILY_ADDRESSED = {"1min", "5min", "15min", "30min", "1hour"}

# bitbank error codes worth branching on. Full list: errors.md in the API docs.
CODE_INSUFFICIENT_FUNDS = {60001, 60002, 60011}
CODE_ORDER_NOT_FOUND = {50008, 50009, 50010}


class BitbankClient:
    """One client for both the public and the private surface.

    `credentials` is a `(key, secret)` pair or None. With None only the public
    calls work — which is exactly what the `mcp` Lambda gets (spec 12).
    """

    def __init__(self, *, public_base_url: str, private_base_url: str, pair: str,
                 credentials: tuple[str, str] | None = None,
                 rate_limiter: RateLimiter | None = None,
                 timeout: float = 10.0, max_retries: int = 4,
                 retry_base_delay: float = 1.0,
                 session: requests.Session | None = None,
                 sleep=time.sleep):
        self.public_base_url = public_base_url.rstrip("/")
        self.private_base_url = private_base_url.rstrip("/")
        self.pair = pair
        self._credentials = credentials
        self.rate_limiter = rate_limiter or RateLimiter(10, 6)
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.session = session or requests.Session()
        self._sleep = sleep

    # -- public ------------------------------------------------------------

    def get_ticker(self) -> dict:
        return self._public(f"/{self.pair}/ticker")

    def get_depth(self) -> dict:
        return self._public(f"/{self.pair}/depth")

    def get_transactions(self, day: date | None = None) -> list[dict]:
        path = f"/{self.pair}/transactions"
        if day is not None:
            path += f"/{day.strftime('%Y%m%d')}"
        return self._public(path)["transactions"]

    def get_circuit_break_info(self) -> dict:
        return self._public(f"/{self.pair}/circuit_break_info")

    def get_candles(self, candle_type: str, day: date | int) -> list[Candle]:
        """`day` is a date for intraday types and a year (int) for 4hour+."""
        if candle_type in DAILY_ADDRESSED:
            key = day.strftime("%Y%m%d") if isinstance(day, date) else str(day)
        else:
            key = str(day.year if isinstance(day, date) else day)
        data = self._public(f"/{self.pair}/candlestick/{candle_type}/{key}")
        rows: list[Candle] = []
        for entry in data.get("candlestick", []):
            for ohlcv in entry.get("ohlcv", []):
                rows.append(Candle.from_bitbank(ohlcv))
        rows.sort(key=lambda c: c.opened_at)
        return rows

    def get_pair_settings(self) -> PairSettings:
        """`GET /v1/spot/pairs` — unauthenticated despite living under /v1."""
        data = self._request("GET", f"{self.private_base_url}/spot/pairs",
                             category="query", signed=False)
        for row in data.get("pairs", []):
            if row.get("name") == self.pair:
                return PairSettings(
                    name=row["name"],
                    min_order_btc=dec(row["unit_amount"]),
                    price_digits=int(row["price_digits"]),
                    amount_digits=int(row["amount_digits"]),
                    maker_fee_rate=dec(row["maker_fee_rate_quote"]),
                    taker_fee_rate=dec(row["taker_fee_rate_quote"]),
                    limit_max_amount=dec(row["limit_max_amount"])
                    if row.get("limit_max_amount") else None,
                    is_enabled=bool(row.get("is_enabled", True)),
                    stop_order=bool(row.get("stop_order", False)),
                    stop_order_and_cancel=bool(row.get("stop_order_and_cancel", False)),
                )
        raise ExchangeError(f"pair {self.pair} not listed by the exchange")

    # -- private -----------------------------------------------------------

    def get_balances(self) -> dict[str, Balance]:
        data = self._private("GET", "/user/assets")
        out: dict[str, Balance] = {}
        for row in data.get("assets", []):
            out[row["asset"]] = Balance(
                asset=row["asset"],
                free=dec(row["free_amount"]),
                locked=dec(row["locked_amount"]),
                onhand=dec(row["onhand_amount"]),
            )
        return out

    def create_order(self, intent: OrderIntent) -> dict:
        body: dict[str, object] = {
            "pair": intent.pair,
            "amount": to_str(intent.qty_btc),
            "side": str(intent.side),
            "type": str(intent.order_type),
        }
        if intent.order_type is OrderType.LIMIT:
            if intent.price is None:
                raise ExchangeError("limit order requires a price")
            body["price"] = to_str(intent.price)
            # post_only is only meaningful on a limit order, and bitbank ignores
            # it outside NORMAL circuit-break mode.
            body["post_only"] = bool(intent.post_only)
        return self._private("POST", "/user/spot/order", body=body)

    def cancel_order(self, exchange_order_id: str) -> dict:
        return self._private("POST", "/user/spot/cancel_order",
                             body={"pair": self.pair,
                                   "order_id": int(exchange_order_id)})

    def get_order(self, exchange_order_id: str) -> dict:
        return self._private("GET", "/user/spot/order",
                             params={"pair": self.pair,
                                     "order_id": int(exchange_order_id)})

    def get_active_orders(self) -> list[dict]:
        data = self._private("GET", "/user/spot/active_orders",
                             params={"pair": self.pair})
        return data.get("orders", [])

    def get_trades_for_order(self, exchange_order_id: str) -> list[dict]:
        data = self._private("GET", "/user/spot/trade_history",
                             params={"pair": self.pair,
                                     "order_id": int(exchange_order_id)})
        return data.get("trades", [])

    def get_recent_trades(self, since: datetime | None = None,
                          count: int = 100) -> list[dict]:
        params: dict[str, object] = {"pair": self.pair, "count": count, "order": "desc"}
        if since is not None:
            params["since"] = int(since.timestamp() * 1000)
        return self._private("GET", "/user/spot/trade_history",
                             params=params).get("trades", [])

    # -- transport ---------------------------------------------------------

    def _public(self, path: str) -> dict:
        return self._request("GET", f"{self.public_base_url}{path}",
                             category="query", signed=False)

    def _private(self, method: str, path: str, *, params: dict | None = None,
                 body: dict | None = None) -> dict:
        if self._credentials is None:
            raise ExchangeError(
                f"private endpoint {path} requested without credentials; this "
                "function is not permitted to trade")
        category = "update" if method == "POST" else "query"
        url = f"{self.private_base_url}{path}"
        return self._request(method, url, category=category, signed=True,
                             params=params, body=body, sign_path=f"/v1{path}")

    def _sign_headers(self, *, sign_path: str, params: dict | None,
                      body: dict | None) -> tuple[dict, str | None]:
        key, secret = self._credentials  # type: ignore[misc]
        request_time = str(int(time.time() * 1000))
        time_window = "5000"
        if body is not None:
            payload = json.dumps(body, separators=(", ", ": "))
            message = request_time + time_window + payload
        else:
            query = _query_string(params)
            payload = None
            message = request_time + time_window + sign_path + query
        signature = hmac.new(secret.encode(), message.encode(),
                             hashlib.sha256).hexdigest()
        headers = {
            "ACCESS-KEY": key,
            "ACCESS-REQUEST-TIME": request_time,
            "ACCESS-TIME-WINDOW": time_window,
            "ACCESS-SIGNATURE": signature,
        }
        if payload is not None:
            headers["Content-Type"] = "application/json"
        return headers, payload

    def _request(self, method: str, url: str, *, category: str, signed: bool,
                 params: dict | None = None, body: dict | None = None,
                 sign_path: str | None = None) -> dict:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.rate_limiter.acquire(category, sleep=self._sleep)
            headers: dict[str, str] = {}
            payload: str | None = None
            if signed:
                headers, payload = self._sign_headers(
                    sign_path=sign_path or "", params=params, body=body)
            request_url = url
            query_params = params
            if signed and body is None:
                # Sign exactly the bytes that go on the wire: build the query
                # string ourselves instead of letting requests encode it.
                request_url = url + _query_string(params)
                query_params = None
            try:
                response = self.session.request(
                    method, request_url, headers=headers, params=query_params,
                    data=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = ExchangeError(f"network error calling {url}: {exc}")
            else:
                try:
                    return self._unwrap(response)
                except ExchangeRateLimited as exc:
                    last_error = exc
                except ExchangeError as exc:
                    # 5xx is worth another try; a rejected order is not.
                    if exc.status is not None and exc.status >= 500:
                        last_error = exc
                    else:
                        raise
            if attempt < self.max_retries:
                delay = self.retry_base_delay * (2 ** attempt)
                log.warning("bitbank call failed (attempt %d/%d), retrying in %.1fs: %s",
                            attempt + 1, self.max_retries + 1, delay, last_error)
                self._sleep(delay)
        raise last_error or ExchangeError(f"request to {url} failed")

    @staticmethod
    def _unwrap(response: requests.Response) -> dict:
        if response.status_code == 429:
            raise ExchangeRateLimited("rate limited by bitbank", status=429)
        if response.status_code >= 500:
            raise ExchangeError(f"bitbank server error {response.status_code}",
                                status=response.status_code)
        try:
            payload = response.json()
        except ValueError as exc:
            raise ExchangeError(
                f"non-JSON response ({response.status_code}) from bitbank") from exc
        if payload.get("success") == 1:
            return payload.get("data", {})
        code = (payload.get("data") or {}).get("code")
        message = f"bitbank error code {code}"
        if code in CODE_INSUFFICIENT_FUNDS:
            raise InsufficientFunds(message, code=code, status=response.status_code)
        raise ExchangeError(message, code=code, status=response.status_code)


def _query_string(params: dict | None) -> str:
    if not params:
        return ""
    parts = "&".join(f"{k}={v}" for k, v in params.items())
    return f"?{parts}"


def signature_for(secret: str, request_time: str, time_window: str, message: str) -> str:
    """Exposed for tests: the exact HMAC bitbank's docs specify."""
    return hmac.new(secret.encode(), (request_time + time_window + message).encode(),
                    hashlib.sha256).hexdigest()
