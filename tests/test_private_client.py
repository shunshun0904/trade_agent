import hashlib
import hmac
import json

import pytest

from bbdata.client import BitbankAPIError
from bblive.private_client import PrivateClient


class Resp:
    def __init__(self, body, status=200):
        self._b, self.status_code = body, status

    def json(self):
        return self._b


class Sess:
    def __init__(self, body):
        self.body, self.calls = body, []

    def get(self, url, headers, timeout):
        self.calls.append(("GET", url, headers, None))
        return Resp(self.body)

    def post(self, url, data, headers, timeout):
        self.calls.append(("POST", url, headers, data))
        return Resp(self.body)


def mk(sess):
    return PrivateClient("KEY", "SECRET", session=sess, clock_ms=lambda: 1700000000000, time_window_ms=5000)


def expected(msg):
    return hmac.new(b"SECRET", msg.encode(), hashlib.sha256).hexdigest()


def test_get_signature_includes_v1_path_and_query():
    s = Sess({"success": 1, "data": {"orders": []}})
    mk(s).active_orders("btc_jpy")
    _, url, h, _ = s.calls[0]
    assert url == "https://api.bitbank.cc/v1/user/spot/active_orders?pair=btc_jpy"
    assert h["ACCESS-REQUEST-TIME"] == "1700000000000" and h["ACCESS-TIME-WINDOW"] == "5000"
    assert h["ACCESS-SIGNATURE"] == expected("17000000000005000/v1/user/spot/active_orders?pair=btc_jpy")


def test_post_signs_exactly_the_sent_body():
    s = Sess({"success": 1, "data": {}})
    mk(s).post("/user/spot/orders_info", {"pair": "btc_jpy", "order_ids": [1, 2]})
    _, url, h, data = s.calls[0]
    assert h["ACCESS-SIGNATURE"] == expected("17000000000005000" + data)
    assert json.loads(data) == {"pair": "btc_jpy", "order_ids": [1, 2]}


def test_error_and_repr_hide_secrets():
    c = mk(Sess({"success": 0, "data": {"code": 20001}}))
    with pytest.raises(BitbankAPIError) as e:
        c.assets()
    assert e.value.code == 20001
    assert "KEY" not in repr(c) and "SECRET" not in repr(c)
    assert not any("withdraw" in n for n in dir(c))
