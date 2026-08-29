.PHONY: install test cov lint build deploy verify-pair snapshot backfill backtest \
        build-TickFunction build-ScreenFunction build-DecideFunction \
        build-ReflectFunction build-McpFunction verify-artifact check-mcp status-live \
        tool

install:
	pip install -r requirements-dev.txt
	pip install -e .

test:
	PYTHONPATH=src python -m pytest

cov:
	PYTHONPATH=src python -m pytest --cov=trade_agent --cov-report=term-missing

# Re-check bitbank pair constants (min order size, fees) against the live
# exchange and report any drift from config/default.yaml (spec 2).
verify-pair:
	PYTHONPATH=src python -m trade_agent.cli verify-pair

snapshot:
	PYTHONPATH=src python -m trade_agent.cli snapshot

backfill:
	PYTHONPATH=src python -m trade_agent.cli backfill --days 30 --out data/candles

backtest:
	PYTHONPATH=src python -m trade_agent.cli backtest --candles data/candles

build:
	sam build

# SAM's makefile builder. It runs `make build-<FunctionLogicalId>` with
# ARTIFACTS_DIR set, once per function. We use it instead of SAM's built-in
# Python builder because that one requires a `python3.11` binary on PATH to run
# pip, and the build host rarely has the same Python as the Lambda runtime —
# AWS CloudShell ships 3.13. `build_lambda.sh` resolves wheels for the target
# runtime from whatever Python is available.
build-TickFunction build-ScreenFunction build-DecideFunction \
build-ReflectFunction build-McpFunction:
	bash scripts/build_lambda.sh "$(ARTIFACTS_DIR)"

# Same check the deploy script runs before uploading.
# Is the deployed system actually running? Read-only.
status-live:
	bash scripts/status.sh

# Check the deployed MCP endpoint and print the claude.ai connector URL.
check-mcp:
	bash scripts/check_mcp.sh

# Run one read-only MCP tool against the deployed system:
#   make tool TOOL=get_cycles
#   make tool TOOL=get_trades ARGS='{"limit": 5}'
tool:
	bash scripts/tool.sh "$(TOOL)" '$(ARGS)'

verify-artifact:
	python3 scripts/verify_artifact.py .aws-sam/build/TickFunction

deploy:
	sam deploy --guided
