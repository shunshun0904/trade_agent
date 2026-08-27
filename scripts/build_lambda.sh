#!/usr/bin/env bash
#
# Build one Lambda artifact, independently of the host's Python version.
#
#   scripts/build_lambda.sh <artifacts-dir>
#
# SAM's own Python builder shells out to a `python3.11` binary to run pip, and
# refuses to build when it cannot find one. AWS CloudShell ships whatever Python
# it ships — 3.13 at the time of writing — so that binary is usually absent, and
# the build fails with a validation error before it does any work.
#
# pip does not actually need to *be* the target interpreter to resolve wheels
# for it. `--platform` + `--python-version` + `--only-binary=:all:` select
# wheels for the Lambda runtime from any host, which is what this does. The
# result carries `cpython-311-x86_64-linux-gnu` extensions whether it was built
# on 3.11, 3.13 or something later.
#
# Layout produced (matching the PYTHONPATH in template.yaml):
#
#   <artifacts>/                dependencies
#   <artifacts>/src/            the trade_agent package
#   <artifacts>/config/         default.yaml
#
set -euo pipefail

ARTIFACTS_DIR="${1:?usage: build_lambda.sh <artifacts-dir>}"
RUNTIME_PYTHON="${TA_LAMBDA_PYTHON:-3.11}"
RUNTIME_PLATFORM="${TA_LAMBDA_PLATFORM:-manylinux2014_x86_64}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$ARTIFACTS_DIR"

python3 -m pip install \
    --disable-pip-version-check \
    --quiet \
    --target "$ARTIFACTS_DIR" \
    --platform "$RUNTIME_PLATFORM" \
    --implementation cp \
    --python-version "$RUNTIME_PYTHON" \
    --only-binary=:all: \
    --upgrade \
    -r "$ROOT/requirements.txt"

# The package and its configuration. `config/` has to travel with the code:
# `config.py` resolves it relative to the package, which lands it at
# <artifacts>/config once PYTHONPATH points at <artifacts>/src.
rm -rf "${ARTIFACTS_DIR:?}/src" "${ARTIFACTS_DIR:?}/config"
cp -r "$ROOT/src" "$ARTIFACTS_DIR/src"
cp -r "$ROOT/config" "$ARTIFACTS_DIR/config"

# Bytecode compiled by the host would be for the host's Python, and is dead
# weight in the upload either way.
find "$ARTIFACTS_DIR" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$ARTIFACTS_DIR" -name '*.pyc' -delete 2>/dev/null || true
