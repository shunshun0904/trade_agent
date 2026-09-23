"""認証の疎通確認（参照系のみ、発注しない）。

環境変数 BITBANK_API_KEY / BITBANK_API_SECRET を使う（GitHub Actions では
Secrets の bitbank_API / bitbank_secret から渡す）。
リポジトリは公開なので、残高の数値や注文の内容は出力しない。出すのは成否と件数だけ。
"""
from __future__ import annotations

import os
import sys

from bbdata.client import BitbankAPIError
from bblive.private_client import PrivateClient

REQUIRED = {"BITBANK_API_KEY": "bitbank_API", "BITBANK_API_SECRET": "bitbank_secret"}


def main() -> int:
    missing = [f"{env}（Secret: {secret}）" for env, secret in REQUIRED.items() if not os.environ.get(env)]
    if missing:
        print("未設定: " + "、".join(missing))
        return 1
    c = PrivateClient(os.environ["BITBANK_API_KEY"], os.environ["BITBANK_API_SECRET"])
    ok = True
    try:
        assets = c.assets()
        names = sorted(a["asset"] for a in assets)
        print(f"GET /user/assets: OK（{len(assets)} 資産。jpy: {'jpy' in names}、btc: {'btc' in names}）")
    except BitbankAPIError as exc:
        print(f"GET /user/assets: NG（HTTP {exc.status}, code={exc.code}）")
        ok = False
    try:
        orders = c.active_orders("btc_jpy")
        print(f"GET /user/spot/active_orders(btc_jpy): OK（未約定 {len(orders)} 件）")
    except BitbankAPIError as exc:
        print(f"GET /user/spot/active_orders: NG（HTTP {exc.status}, code={exc.code}）")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
