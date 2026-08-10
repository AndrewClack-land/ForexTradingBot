"""Crash-consistency contracts for the persistent LIMIT ENTER outbox.

These tests keep MT5 and Telegram outside the process.  They exercise the real
Core state transitions, atomic JSON stores and SQLite trade journal at the
three unsafe boundaries around a broker fill announcement.
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import main
from bot.telegram_bot import TelegramBot
from core.persistent_limit import (
    PendingLimitLeg,
    PendingLimitPlan,
    load_pending_limits,
    save_pending_limits,
)
from core.state_store import load_active_trades
from core.strategy_narrative import ActiveTrade
from core.trade_journal import TradeJournal


PLAN_ID = "enter-outbox-plan-00000000000001"
CHAT_ID = -1001234567890
MESSAGE_ID = 731_337


def _signal() -> dict:
    return {
        "signal": "ENTER",
        "side": "LONG",
        "entry_price": 1.1000,
        "entry_min": 1.0998,
        "entry_max": 1.1002,
        "stop_price": 1.0950,
        "tp_price": 1.1150,
        "tp_prices": [1.1150],
        "tf": "15M",
        "narrative": "durable ENTER outbox fixture",
        "idea_id": "idea-enter-outbox-0001",
    }


def _filled_plan() -> PendingLimitPlan:
    now = time.time()
    signal = _signal()
    plan = PendingLimitPlan.create(
        symbol="EURUSD",
        signal=signal,
        trigger_signature="LONG|pivot|EURUSD|closed-M15",
        expires_at=now + 3600.0,
        magic=4242,
        legs=(
            PendingLimitLeg(
                index=1,
                target_price=signal["tp_prices"][0],
                requested_volume=0.1,
            ),
        ),
        now=now,
        plan_id=PLAN_ID,
    )
    plan = plan.attach_order(1, 101, now=now + 0.1)
    return plan.record_fill(
        1,
        0.1,
        1.0999,
        deal_ticket=501,
        position_id=9001,
        now=now + 0.2,
    )


class _SettledFillExecutor:
    """Authoritative broker view after the sole LIMIT leg has filled."""

    def __init__(self, plan: PendingLimitPlan):
        self.plan = plan
        self.cancelled: list[int] = []

    @staticmethod
    def list_pending_orders() -> list[dict]:
        return []

    def list_positions(self) -> list[dict]:
        return [
            {
                "ticket": 9001,
                "symbol": self.plan.symbol,
                "comment": self.plan.broker_comment_for_leg(1),
                "volume": 0.1,
                "entry_price": 1.0999,
            }
        ]

    @staticmethod
    def get_pending_order_fills(
        _tickets,
        *,
        comments,
        lookback_days,
    ) -> dict:
        assert comments
        assert lookback_days >= 1
        return {}

    def get_pending_order_histories(
        self,
        comments,
        *,
        lookback_days,
    ) -> dict:
        assert lookback_days >= 1
        comment = self.plan.broker_comment_for_leg(1)
        assert comment in comments
        return {
            comment: [
                {
                    "ticket": 101,
                    "terminal": True,
                    "fill_expected": True,
                    "had_execution": True,
                }
            ]
        }

    @staticmethod
    def get_pending_order_history(_ticket: int) -> dict:
        return {
            "ticket": 101,
            "terminal": True,
            "fill_expected": True,
            "had_execution": True,
        }

    def cancel_pending_order(self, ticket, *, expected_comment_prefix):
        raise AssertionError(
            "a completely filled leg has no cancellable remainder: "
            f"{ticket} {expected_comment_prefix}"
        )


def _core(
    executor: _SettledFillExecutor,
    plan: PendingLimitPlan,
    *,
    active_trades: dict | None = None,
) -> main.Core:
    core = main.Core.__new__(main.Core)
    core.mt5_executor = executor
    core.pending_limits = {plan.symbol: plan}
    core._pending_state_safe = True
    core._pending_entry_events = {}
    core._pending_reconcile_error_ts = 0.0
    core._management_lock = threading.RLock()
    core.active_trades = dict(active_trades or {})
    core._entries_today = {}
    core._trigger_signatures = {}
    core._entry_cooldowns = {}
    core.shadow_tick_watcher = None
    core._roll_daily_counters = lambda: None
    core._capture_quote = lambda _symbol: None
    return core


def test_restart_requeues_enter_after_crash_between_active_save_and_queue(
    tmp_path,
    monkeypatch,
):
    plan = _filled_plan()
    pending_path = tmp_path / "pending_limits.json"
    active_path = tmp_path / "active_trades.json"
    save_pending_limits({plan.symbol: plan}, pending_path)
    monkeypatch.setattr(main, "AI_DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "PERSISTENT_LIMIT_STATE_PATH", pending_path)

    crashing = _core(_SettledFillExecutor(plan), plan)

    class SimulatedProcessCrash(RuntimeError):
        pass

    def crash_before_queue() -> None:
        raise SimulatedProcessCrash("after ActiveTrade save, before ENTER queue")

    crashing._roll_daily_counters = crash_before_queue
    with pytest.raises(SimulatedProcessCrash, match="before ENTER queue"):
        crashing._materialize_pending_trade(
            plan,
            queue_entry_event=True,
        )

    # The exact crash window has an ActiveTrade on disk, no volatile event,
    # and the unacknowledged plan still durably owns the ENTER outbox item.
    assert active_path.exists()
    assert crashing._pending_entry_events == {}
    assert load_pending_limits(pending_path)[plan.symbol].entry_announcement_pending

    restarted_plan = load_pending_limits(pending_path)[plan.symbol]
    restarted = _core(
        _SettledFillExecutor(restarted_plan),
        restarted_plan,
        active_trades=load_active_trades(active_path),
    )

    restarted._manage_pending_limits(startup=True)

    assert plan.symbol in restarted.pending_limits
    assert restarted.pending_limits[plan.symbol].entry_announcement_pending
    event = restarted._pending_entry_events[plan.symbol]
    assert event["signal"] == "ENTER"
    assert event["pending_limit"]["plan_id"] == plan.plan_id
    assert event["idea_id"] == _signal()["idea_id"]


def test_durable_ack_is_written_before_next_reconcile_retires_plan(
    tmp_path,
    monkeypatch,
):
    plan = _filled_plan()
    pending_path = tmp_path / "pending_limits.json"
    save_pending_limits({plan.symbol: plan}, pending_path)
    monkeypatch.setattr(main, "AI_DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "PERSISTENT_LIMIT_STATE_PATH", pending_path)
    core = _core(_SettledFillExecutor(plan), plan)

    core._manage_pending_limits(startup=True)
    assert core.pending_limits[plan.symbol].entry_announcement_pending
    assert plan.symbol in core._pending_entry_events

    # Telegram consumes the volatile event before acknowledging its durable
    # source.  The acknowledgement itself must reach disk before retirement.
    core._pending_entry_events.clear()
    assert core.ack_pending_limit_entry(plan.plan_id) is True
    acknowledged = load_pending_limits(pending_path)[plan.symbol]
    assert acknowledged.entry_announced_at is not None
    assert acknowledged.entry_announcement_pending is False
    assert plan.symbol in core.pending_limits

    core._manage_pending_limits()

    assert plan.symbol not in core.pending_limits
    assert load_pending_limits(pending_path) == {}
    assert plan.symbol in core.active_trades


class _Profiler:
    @staticmethod
    def section(_name):
        return nullcontext()


def test_restored_message_identity_skips_duplicate_send_and_repairs_journal(
    tmp_path,
    monkeypatch,
):
    plan = _filled_plan()
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(main, "AI_DATA_DIR", runtime_dir)
    trade = ActiveTrade(
        side="LONG",
        entry=1.0999,
        stop=1.0950,
        tp_prices=[1.1150],
        tf="15M",
        narrative="durable ENTER outbox fixture",
        symbol=plan.symbol,
    )
    trade.idea_id = _signal()["idea_id"]
    core = _core(
        _SettledFillExecutor(plan),
        plan,
        active_trades={plan.symbol: trade},
    )

    # This is the second crash window: Telegram succeeded and Core persisted
    # its immutable identity, but journal repair/outbox ack did not run.
    assert core.bind_pending_entry_message(
        plan.symbol,
        plan.plan_id,
        chat_id=CHAT_ID,
        message_id=MESSAGE_ID,
    )
    restored = load_active_trades(runtime_dir / "active_trades.json")
    assert restored[plan.symbol].telegram_chat_id == CHAT_ID
    assert restored[plan.symbol].telegram_message_id == MESSAGE_ID

    journal = TradeJournal(
        str(tmp_path / "trades.db"),
        export_on_each_event=False,
    )
    bot = TelegramBot.__new__(TelegramBot)
    bot.journal = journal
    bot.profiler = _Profiler()
    bot.channel_id = CHAT_ID
    bot.core = SimpleNamespace(active_trades=restored)
    bot._format_signal = lambda symbol, _sig: f"{symbol} ENTER"
    sends: list[dict] = []

    async def unexpected_duplicate_send(_app, **kwargs):
        sends.append(dict(kwargs))
        return SimpleNamespace(message_id=MESSAGE_ID + 1)

    bot._safe_send_message = unexpected_duplicate_send
    signal = dict(plan.signal_payload)
    signal.update(
        {
            "entry_price": 1.0999,
            "execution": {"volume": 0.1},
            "pending_limit": {"plan_id": plan.plan_id},
        }
    )

    try:
        delivered = asyncio.run(
            bot._post_enter(object(), plan.symbol, signal)
        )
        row = journal._conn.execute(
            """
            SELECT telegram_chat_id_open, telegram_message_id_open
            FROM trades
            WHERE symbol = ?
            """,
            (plan.symbol,),
        ).fetchone()
        assert delivered is True
        assert sends == []
        assert row is not None
        assert tuple(row) == (CHAT_ID, MESSAGE_ID)
        assert journal.setup_metrics()["total_setups"] == 1
    finally:
        journal.close()
