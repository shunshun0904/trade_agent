#!/usr/bin/env bash
# GitHub Actions 用: 生成したレポートをブランチの最新に載せ直して commit・push する。
#
#   scripts/commit_report.sh <実行前の実験ログの行数> <コミットメッセージ> <ファイル>...
#
# 実行中に別の実行のレポートが push されていても衝突しないよう、ブランチの最新をチェックアウトしてから
# レポートを上書きする。実験ログ（reports/experiments.jsonl）は、この実行で追記した行だけを足す。
set -euo pipefail
n0="$1"; msg="$2"; shift 2
log=reports/experiments.jsonl
branch="${GITHUB_REF_NAME:?}"
tmp="$(mktemp -d)"
for f in "$@"; do mkdir -p "$tmp/$(dirname "$f")"; cp "$f" "$tmp/$f"; done
if [ -f "$log" ]; then tail -n "+$(( n0 + 1 ))" "$log" > "$tmp/new_trials.jsonl"; else : > "$tmp/new_trials.jsonl"; fi
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
for attempt in 1 2 3; do
  git fetch -q origin "$branch"
  git reset -q --hard
  # チェックアウト時点のブランチになかったファイルは未追跡として残り checkout を妨げるので消す
  # （退避済み。追跡済みのファイルは消さない。消すと checkout 後も削除されたままになる）
  git clean -fq -- "$@" "$log"
  git checkout -q -B report-update "origin/$branch"
  for f in "$@"; do mkdir -p "$(dirname "$f")"; cp "$tmp/$f" "$f"; done
  cat "$tmp/new_trials.jsonl" >> "$log"
  git add -- "$@" "$log"
  git commit -q -m "$msg" || exit 0
  if git push -q origin "HEAD:$branch"; then exit 0; fi
  sleep $(( attempt * 5 ))
done
exit 1
