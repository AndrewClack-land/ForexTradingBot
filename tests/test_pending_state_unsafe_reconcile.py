"""Unsafe admission state must never freeze already-owned broker risk."""

from __future__ import annotations

import threading
import time

import main
from core.persistent_limit import (
    BROKER_COMMENT_PREFIX,
    PendingLimitLeg,
    PendingLimitPlan,
)


def _plan() -> PendingLimitPlan:
    now = time.time() - 60.0
    signal = {
        "signal": "ENTER",
        "side": "LONG",
        "entry_price": 1.1000,
        "entry_min": 1.0998,
        "entry_max": 1.1002,
        "stop_price": 1.0950,
        "tp_price": 1.1150,
        "tp_prices": [1.1050, 1.1100, 1.1150],
        "idea_id": "unsafe-reconcile-idea",
    }
    plan = PendingLimitPlan.create(
        symbol="EURUSD",
        signal=signal,
        trigger_signature="LONG|unsafe-reconcile",
        expires_at=now + 900.0,
        magic=4242,
        legs=(PendingLimitLeg(3, 1.1150, 0.1),),
        now=now,
        plan_id="unsafe-reconcile-plan-00000001",
    )
    return plan.attach_order(3, 103, now=now + 1.0)


class _Broker:
    def __init__(self, orders):
        self.orders = [dict(row) for row in orders]
        self.cancelled = []

    def list_pending_orders(self):
        return [dict(row) for row in self.orders]

    @staticmethod
    def list_positions():
        return []

    @staticmethod
    def get_pending_order_fills(_tickets, *, comments, lookback_days):
        assert comments
        assert lookback_days >= 1
        return {}

    def cancel_pending_order(self, ticket, *, expected_comment_prefix):
        assert expected_comment_prefix.startswith(BROKER_COMMENT_PREFIX)
        ticket = int(ticket)
        self.cancelled.append(ticket)
        self.orders = [
            row
            for row in self.orders
            if int(row.get("ticket") or 0) != ticket
        ]
        return True

    @staticmethod
    def get_pending_order_history(_ticket):
        return {
            "terminal": True,
            "fill_expected": False,
            "had_execution": False,
            "time_done_msc": 0,
        }


def _core(broker, plan):
    core = main.Core.__new__(main.Core)
    core.mt5_executor = broker
    core.pending_limits = {"EURUSD": plan}
    core._pending_state_safe = False
    core._pending_entry_events = {}
    core._pending_reconcile_error_ts = 0.0
    core._management_lock = threading.RLock()
    core.active_trades = {}
    core.shadow_tick_watcher = None
    core._save_pending_limit_state = lambda: None
    core._capture_quote = lambda _symbol: None
    core._pending_cancel_reason = (
        lambda _plan, *, now, quote: "test risk cancellation"
    )
    return core


def test_unsafe_admission_state_still_cancels_and_retires_known_plan():
    plan = _plan()
    broker = _Broker(
        [
            {
                "ticket": 103,
                "comment": plan.broker_comment_for_leg(3),
            }
        ]
    )
    core = _core(broker, plan)

    core._manage_pending_limits()

    assert broker.cancelled == [103]
    assert core.pending_limits == {}
    assert core._pending_state_safe is False

    # The same flag remains a strict admission brake after risk cleanup.
    signal = {"signal": "ENTER"}
    assert core._apply_entry_guards(
        "EURUSD",
        signal,
        side="LONG",
        trig_sig="new-risk",
    )
    assert signal["signal"] == "WAIT_PENDING_STATE"


def test_orphan_blocks_new_risk_but_not_known_plan_reconciliation():
    plan = _plan()
    orphan_ticket = 999
    broker = _Broker(
        [
            {
                "ticket": 103,
                "comment": plan.broker_comment_for_leg(3),
            },
            {
                "ticket": orphan_ticket,
                "comment": f"{BROKER_COMMENT_PREFIX}orphan:T1",
            },
        ]
    )
    core = _core(broker, plan)
    core._pending_state_safe = True

    core._manage_pending_limits(startup=True)

    assert core._pending_state_safe is False
    assert broker.cancelled == [103]
    assert [row["ticket"] for row in broker.orders] == [orphan_ticket]
    assert core.pending_limits == {}
