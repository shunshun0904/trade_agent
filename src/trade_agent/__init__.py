"""BTC/JPY multi-agent trading system.

Layering (spec section 3):

    exchange/   raw bitbank access
    data/       indicators + MarketSnapshot assembly      (deterministic)
    llm/        Anthropic transport, budget accounting
    agents/     A1..A7, JSON in / JSON out
    guards/     deterministic verification of agent output (spec 5)
    risk/       position sizing, circuit breakers, boredom rule (spec 6, 7)
    execution/  idempotent ordering, requote, SL/TP management (spec 8)
    orchestrator/  the state machine that runs a decision cycle (spec 4.1)
    storage/    DynamoDB + S3 repositories (spec 10)
    handlers/   Lambda entry points (spec 17.1)
"""

__version__ = "0.1.0"
