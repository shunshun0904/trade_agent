"""`tick` Lambda — every 5 minutes, no LLM (spec 9, 17.1).

This is the function that must never stop running. It owns the stop loss, and
because there is no exchange-side OCO (see position_manager), a tick that does
not fire is a position with no stop. The CloudWatch alarm on its invocation
count is therefore a safety control of the same rank as the kill switch
(spec 17.3).

Everything here is deterministic and free: no model is consulted, so the tick
keeps working after the monthly LLM budget is spent (spec 11).
"""

from __future__ import annotations

import logging

from ..errors import ExchangeError
from ..models.state import SystemState
from ..money import dec, jpy
from ..orchestrator.context import AppContext
from ..storage.base import EquityPoint, LOCK_EXECUTION
from ..timeutil import jst_date_str
from .common import run_handler

log = logging.getLogger(__name__)


def handler(event=None, context=None, *, ctx: AppContext | None = None) -> dict:
    return run_handler("tick", run, ctx=ctx)


def run(ctx: AppContext) -> dict:
    now = ctx.clock.now()
    state = ctx.load_state()
    state.last_tick_at = now
    notes: list[str] = []

    executor = ctx.executor(owner=f"tick-{now.timestamp():.0f}")
    if not executor.acquire_lock(LOCK_EXECUTION):
        # A decide cycle is placing an order right now. Record the heartbeat
        # and get out of its way rather than reading half-written state.
        ctx.save_state(state)
        return {"skipped": "execution lock held by another invocation",
                "heartbeat": True}

    try:
        result = _work(ctx, state, notes, executor)
    finally:
        executor.release_lock(LOCK_EXECUTION)
    return result


def _work(ctx: AppContext, state: SystemState, notes: list[str], executor) -> dict:
    now = ctx.clock.now()
    notes.extend(executor.reconcile_pending())
    notes.extend(executor.expire_stale_entries())

    try:
        ticker = ctx.exchange.get_ticker()
        depth = ctx.exchange.get_depth()
        balances = ctx.exchange.get_balances()
        state.consecutive_private_api_failures = 0
    except ExchangeError as exc:
        state.consecutive_private_api_failures += 1
        ctx.save_state(state)
        _maybe_warn_api(ctx, state, exc)
        return {"error": f"market/account data unavailable: {exc}",
                "consecutive_failures": state.consecutive_private_api_failures}

    last = dec(ticker["last"])
    best_bid = dec(depth["bids"][0][0]) if depth.get("bids") else last
    best_ask = dec(depth["asks"][0][0]) if depth.get("asks") else last

    jpy_balance = balances.get(ctx.config.exchange.quote_asset)
    btc_balance = balances.get(ctx.config.exchange.base_asset)
    equity = ((jpy_balance.free + jpy_balance.locked) if jpy_balance else dec(0)) + \
             (((btc_balance.free + btc_balance.locked) * last) if btc_balance else dec(0))
    state.equity_jpy = equity
    state.peak_equity_jpy = max(state.peak_equity_jpy, equity)

    manager = ctx.position_manager(executor)
    update = manager.step(state, last_price=last, best_bid=best_bid,
                          best_ask=best_ask)
    notes.extend(update.notes)

    if ctx.risk.check_flash_move(state, now, _change_15m(ctx, last)):
        notes.append("flash-move pause armed")

    _check_kill_switch(ctx, state, notes)
    _check_daily_loss(ctx, state, notes)
    _poll_reflect_batch(ctx, state, notes)
    _write_equity_point(ctx, state)

    ctx.save_state(state)
    return {
        "equity_jpy": float(state.equity_jpy),
        "position": bool(state.open_position),
        "kill_switch": state.kill_switch,
        "closed_trade": update.closed.trade_id if update.closed else None,
        "opened_position": bool(update.opened),
        "notes": notes,
    }


def _change_15m(ctx: AppContext, last):
    """15-minute change from short candles, for the flash-move breaker.

    Computed here rather than reusing the snapshot builder: the tick must stay
    cheap, and this needs one candle call rather than the full fan-out.
    """
    from ..data.indicators import change_pct, closes

    cfg = ctx.config
    try:
        candles = ctx.exchange.get_candles(cfg.snapshot.short_candle_type,
                                           ctx.clock.now().date())
    except ExchangeError:
        return None
    if len(candles) < 4:
        return None
    per_candle_minutes = 5 if cfg.snapshot.short_candle_type == "5min" else 1
    periods = max(1, cfg.risk.flash_move_window_minutes // per_candle_minutes)
    return change_pct(closes(candles), periods)


def _check_kill_switch(ctx: AppContext, state: SystemState,
                       notes: list[str]) -> None:
    if state.kill_switch or not ctx.risk.should_kill(state):
        return
    drawdown = state.drawdown_pct(ctx.config.capital.initial_equity_jpy)
    ctx.risk.engage_kill_switch(
        state, ctx.clock.now(),
        f"equity {jpy(state.equity_jpy)} JPY is {drawdown:.1f}% below initial capital")
    notes.append("kill switch engaged")

    # Spec 6: close everything, then stop. The close is attempted before the
    # notification so the owner's first email already reflects a flat book.
    position = state.open_position
    if position is not None:
        notes.extend(ctx.position_manager().force_close(
            state, position, "kill_switch").notes)

    if ctx.notifier is not None:
        ctx.notifier.send(
            "キルスイッチ発動",
            f"equity {jpy(state.equity_jpy)} 円(初期資金比 -{drawdown:.1f}%)。\n"
            "全建玉のクローズを試行し、システムを完全停止しました。\n"
            "再開には Claude から resume_trading を confirm=true で実行してください。")


def _check_daily_loss(ctx: AppContext, state: SystemState,
                      notes: list[str]) -> None:
    limit = ctx.config.risk.daily_loss_limit_pct
    if state.daily_loss_pct() < limit:
        return
    key = f"daily-loss-notified:{state.daily.jst_date}"
    if key in notes:
        return
    notes.append(f"daily loss limit reached ({state.daily_loss_pct():.2f}% >= {limit}%)")
    if ctx.notifier is not None:
        ctx.notifier.send(
            "日次最大損失に到達",
            f"当日の実現損益 {jpy(state.daily.realized_pnl_jpy)} 円 "
            f"({state.daily_loss_pct():.2f}%) が上限 {limit}% に達しました。\n"
            "本日の新規建ては停止します。建玉管理は継続します。")


def _poll_reflect_batch(ctx: AppContext, state: SystemState,
                        notes: list[str]) -> None:
    """Collect an A7 batch result if one is outstanding (spec 11)."""
    if not state.pending_reflect_batch_id:
        return
    from .reflect import collect_batch

    try:
        collected = collect_batch(ctx, state)
    except Exception as exc:  # noqa: BLE001 - never let reflection break the tick
        log.warning("reflect batch poll failed: %s", exc)
        return
    if collected:
        notes.append(f"reflection batch collected: {collected} lessons stored")


def _write_equity_point(ctx: AppContext, state: SystemState) -> None:
    today = jst_date_str(ctx.clock.now())
    trades = ctx.store.trades.list_recent(200)
    todays = [t for t in trades if t.exit_at and jst_date_str(t.exit_at) == today]
    ctx.store.equity.put(EquityPoint(
        jst_date=today,
        equity_jpy=state.equity_jpy,
        realized_pnl_jpy=state.daily.realized_pnl_jpy,
        cumulative_llm_cost_jpy=state.monthly.llm_cost_jpy,
        infra_cost_jpy=ctx.config.cost.infra_cost_jpy,
        kill_switch=state.kill_switch,
        trades=len([t for t in todays if not t.probe]),
        probe_trades=len([t for t in todays if t.probe]),
        updated_at=ctx.clock.now()))


def _maybe_warn_api(ctx: AppContext, state: SystemState, exc: Exception) -> None:
    threshold = ctx.config.exchange.private_failure_threshold
    if state.consecutive_private_api_failures != threshold:
        return
    if ctx.notifier is not None:
        ctx.notifier.send(
            "取引所APIの連続失敗",
            f"{threshold} 回連続で取引所APIの呼び出しに失敗しました。\n"
            f"直近のエラー: {exc}\n"
            "口座状態が確認できないため、新規建ては停止します。"
            "建玉がある場合、損切り監視も停止している可能性があります。")
