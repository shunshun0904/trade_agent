#!/usr/bin/env bash
# React 版の画面をビルドして Lambda が返す app/page.html に置く。Node.js 18 以上が要る。
# 生成物（app/page.html）はコミットする。CloudShell でのデプロイに Node.js を要らなくするため。
set -euo pipefail
cd "$(dirname "$0")"
npm ci --no-audit --no-fund
npm run build
cp dist/index.html ../app/page.html
echo "wrote ../app/page.html ($(wc -c < ../app/page.html) bytes)"
