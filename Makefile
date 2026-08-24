.PHONY: install test lint build deploy verify-pair backtest

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

deploy:
	sam deploy --guided
