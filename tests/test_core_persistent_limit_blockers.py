"""Regression contracts for proven Core persistent-LIMIT race windows.

These tests are intentionally integration-shaped and broker-free.  They use a
``Core.__new__`` shell plus mutable broker evidence so publication order can be
controlled exactly.  A failing test means Core can lose ownership, duplicate a
fill, or retain risk after a restart; it must not be weakened to fit the current
implementation.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import main
from core.persistent_limit import PendingLimitLeg, PendingLimitPlan


NOW = 2_000_000_000.0
PLAN_ID = "core-blocker-plan-000000000001"


def _signal() -> dict:
    return {
        "signal": "ENTER",
        "side": "LONG",
        "entry_price": 1.1000,
        "entry_min": 1.0998,
        "entry_max": 1.1002,
        "stop_price": 1.0950,
        "tp_price": 1.1150,
        "tp_prices": [1.1050, 1.1100, 1.1150],
        "tf": "15M",
        "narrative": "persistent LIMIT blocker fixture",
        "idea_id": "pending-idea",
    }


def _plan(*, attached_indexes=(1, 2, 3)) -> PendingLimitPlan:
    signal = _signal()
    plan = PendingLimitPlan.create(
        symbol="EURUSD",
        signal=signal,
        trigger_signature="LONG|pivot|blocker",
        expires_at=NOW + 3600.0,
        magic=4242,
        legs=tuple(
            PendingLimitLeg(index, target, 0.1)
            for index, target in enumerate(signal["tp_prices"], start=1)
        ),
        now=NOW,
        plan_id=PLAN_ID,
    )
    for offset, index in enumerate(attached_indexes, start=1):
        plan = plan.attach_order(index, 100 + index, now=NOW + offset)
    return plan


def _single_leg_plan() -> PendingLimitPlan:
    signal = _signal()
    return PendingLimitPlan.create(
        symbol="EURUSD",
        signal=signal,
        trigger_signature="LONG|pivot|cancel-delay",
        expires_at=NOW + 3600.0,
        magic=4242,
        legs=(PendingLimitLeg(3, signal["tp_prices"][2], 0.3),),
        now=NOW,
        plan_id="core-cancel-delay-plan-000000001",
    ).attach_order(3, 103, now=NOW + 1)


def _core(executor, plan):
    core = main.Core.__new__(main.Core)
    core.mt5_executor = executor
    core.pending_limits = {"EURUSD": plan}
    core._pending_state_safe = True
    core._pending_entry_events = {}
    core._pending_reconcile_error_ts = 0.0
    core._management_lock = threading.RLock()
    core.active_trades = {}
    core._entries_today = {}
    core._trigger_signatures = {}
    core._entry_cooldowns = {}
    core.shadow_tick_watcher = None
    core._roll_daily_counters = lambda: None
    core._capture_quote = lambda _symbol: None
    core._save_pending_limit_state = lambda: None
    return core


class MutableBroker:
    def __init__(self, *, live_orders=(), positions=(), fills=None, history=None):
        self.live_orders = [dict(row) for row in live_orders]
        self.positions = [dict(row) for row in positions]
        self.fills = dict(fills or {})
        self.history = dict(history or {})
        self.cancelled = []
        self.cancel_error = None
        self.placement_calls = 0

    def list_pending_orders(self):
        return [dict(row) for row in self.live_orders]

    def list_positions(self):
        return [dict(row) for row in self.positions]

    def get_pending_order_fills(self, _tickets, *, comments, lookback_days):
        assert comments
        assert lookback_days == 1
        return self.fills

    def cancel_pending_order(self, ticket, *, expected_comment_prefix):
        assert expected_comment_prefix
        if self.cancel_error is not None:
            raise self.cancel_error
        ticket = int(ticket)
        self.cancelled.append(ticket)
        self.live_orders = [
            row
            for row in self.live_orders
            if int(row.get("ticket") or 0) != ticket
        ]
        return True

    def get_pending_order_history(self, ticket):
        return self.history.get(int(ticket))

    def place_limit_leg(self, *_args, **_kwargs):
        self.placement_calls += 1
        raise AssertionError("reconciliation must not submit another order")


def test_position_fallback_then_same_deal_does_not_double_count(monkeypatch):
    plan = _plan()
    comments = {
        index: plan.broker_comment_for_leg(index) for index in (1, 2, 3)
    }
    broker = MutableBroker(
        live_orders=[
            {"ticket": 102, "comment": comments[2]},
            {"ticket": 103, "comment": comments[3]},
        ],
        positions=[
            {
                "ticket": 9001,
                "comment": comments[1],
                "volume": 0.1,
                "entry_price": 1.0999,
            }
        ],
    )
    broker.cancel_error = RuntimeError("synthetic cancellation delay")
    core = _core(broker, plan)
    monkeypatch.setattr(main, "save_active_trades", lambda *_args: None)

    # The position appears before its opening deal. Core cancels siblings but
    # does not invent an unkeyed synthetic fill that the later deal would
    # double-count.
    core._manage_pending_limits()
    awaiting_deal = core.pending_limits["EURUSD"].legs[0]
    assert awaiting_deal.filled_volume == 0.0
    assert awaiting_deal.deal_tickets == ()

    # The same economic fill is published in deal history on the next poll.
    # It must adopt the immutable deal id without adding another 0.1 lot.
    broker.fills = {
        101: {
            "order_ticket": 101,
            "deals": [
                {
                    "deal_ticket": 501,
                    "position_id": 9001,
                    "volume": 0.1,
                    "price": 1.0999,
                    "comment": comments[1],
                }
            ],
        }
    }

    core._manage_pending_limits()

    reconciled = core.pending_limits["EURUSD"].legs[0]
    assert reconciled.filled_volume == pytest.approx(0.1)
    assert reconciled.deal_tickets == (501,)


def test_cancelled_order_is_not_retired_before_delayed_deal_grace():
    plan = _single_leg_plan()
    broker = MutableBroker(
        history={
            103: {
                "state": "CANCELED",
                "terminal": True,
                "fill_expected": False,
                "time_done_msc": 2_000_000_000_000,
            }
        }
    )
    core = _core(broker, plan)
    core._pending_cancel_reason = lambda _plan, *, now, quote: None

    core._manage_pending_limits(startup=True)

    # MetaTrader can publish terminal order history before the opening deal.
    # Ownership must survive at least one causal reconciliation/grace window.
    assert "EURUSD" in core.pending_limits
    assert not core.pending_limits["EURUSD"].is_terminal


def test_incomplete_startup_placement_cancels_every_known_leg():
    plan = _plan(attached_indexes=(1,))
    comment = plan.broker_comment_for_leg(1)
    broker = MutableBroker(
        live_orders=[{"ticket": 101, "comment": comment}],
        history={101: {"terminal": True, "fill_expected": False}},
    )
    core = _core(broker, plan)
    core._pending_cancel_reason = lambda _plan, *, now, quote: None

    core._manage_pending_limits(startup=True)

    # A crash between leg submissions must never leave the accepted subset
    # working while Core waits forever for tickets that were never submitted.
    assert broker.cancelled == [101]
    assert broker.placement_calls == 0


def test_incompatible_active_idea_cancels_pending_before_any_fill():
    plan = _plan()
    live_orders = [
        {
            "ticket": int(leg.broker_order_ticket),
            "comment": plan.broker_comment_for_leg(leg.index),
        }
        for leg in plan.legs
    ]
    broker = MutableBroker(
        live_orders=live_orders,
        history={
            int(leg.broker_order_ticket): {
                "terminal": True,
                "fill_expected": False,
            }
            for leg in plan.legs
        },
    )
    core = _core(broker, plan)
    core.active_trades["EURUSD"] = SimpleNamespace(
        idea_id="different-active-idea",
        side="SHORT",
        symbol="EURUSD",
    )
    core._pending_cancel_reason = lambda _plan, *, now, quote: None

    core._manage_pending_limits(startup=True)

    assert set(broker.cancelled) == {101, 102, 103}
    assert broker.placement_calls == 0
