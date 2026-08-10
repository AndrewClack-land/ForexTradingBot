"""Characterization contract for durable persistent-LIMIT state.

These tests deliberately exercise only the broker-agnostic model.  MT5 request
construction and reconciliation belong to adapter tests; the invariants here
must survive any adapter implementation and a process restart.
"""

from __future__ import annotations

import json

import pytest

import core.persistent_limit as persistent_limit
from core.persistent_limit import (
    PendingLimitLeg,
    PendingLimitPlan,
    PendingLimitState,
    PendingLimitValidationError,
    load_pending_limits,
    save_pending_limits,
)


NOW = 1_786_291_200.0
EXPIRES_AT = NOW + 15 * 60
PLAN_ID = "limit-plan-0001"


def _plan() -> PendingLimitPlan:
    signal = {
        "setup_id": PLAN_ID,
        "side": "LONG",
        "entry_price": 1.1000,
        "entry_min": 1.0998,
        "entry_max": 1.1002,
        "stop_price": 1.0950,
        "tp_prices": [1.1050, 1.1100, 1.1150],
        "trigger_kind": "h1_pivot_reclaim_15m",
    }
    legs = tuple(
        PendingLimitLeg(index=index, target_price=target, requested_volume=0.1)
        for index, target in enumerate(signal["tp_prices"], start=1)
    )
    return PendingLimitPlan.create(
        symbol="eurusd",
        signal=signal,
        trigger_signature="pivot:EURUSD:2026-08-10T12:00Z",
        expires_at=EXPIRES_AT,
        magic=260810,
        legs=legs,
        now=NOW,
    )


def _placed() -> PendingLimitPlan:
    plan = _plan()
    for offset, (index, ticket) in enumerate(
        ((1, 1001), (2, 1002), (3, 1003)),
        start=1,
    ):
        plan = plan.attach_order(
            index,
            ticket,
            now=NOW + offset,
        )
    assert plan.state is PendingLimitState.PLACED
    return plan


def test_plan_identity_and_comment_survive_transitions_and_restart():
    placing = _plan()
    placed = _placed()
    restored = PendingLimitPlan.from_dict(placed.to_dict())

    assert placing.plan_id == placed.plan_id == restored.plan_id == PLAN_ID
    assert placing.broker_comment == placed.broker_comment == restored.broker_comment
    assert restored.signal_payload["setup_id"] == PLAN_ID


def test_lifecycle_operations_never_renew_the_original_ttl():
    placing = _plan()
    placed = _placed()
    partially_filled = placed.record_fill(
        1,
        0.1,
        1.1000,
        deal_ticket=2001,
        position_id=3001,
        now=NOW + 10,
    )
    cancelling = partially_filled.transition(
        PendingLimitState.CANCELLING,
        now=NOW + 11,
        reason="cancel unfilled sibling legs",
    )
    restored = PendingLimitPlan.from_dict(cancelling.to_dict())

    assert {
        plan.expires_at
        for plan in (placing, placed, partially_filled, cancelling, restored)
    } == {EXPIRES_AT}


def test_store_round_trip_atomically_replaces_the_previous_snapshot(tmp_path):
    path = tmp_path / "pending-limits.json"
    placing = _plan()
    save_pending_limits({"EURUSD": placing}, path)
    assert load_pending_limits(path) == {"EURUSD": placing}

    placed = _placed()
    save_pending_limits({"EURUSD": placed}, path)

    assert load_pending_limits(path) == {"EURUSD": placed}
    assert not list(tmp_path.glob(".pending-limits.json.*.tmp"))


def test_failed_atomic_replace_keeps_the_last_valid_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "pending-limits.json"
    placing = _plan()
    save_pending_limits({"EURUSD": placing}, path)

    def fail_replace(_source, _destination):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(persistent_limit.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        save_pending_limits({"EURUSD": _placed()}, path)

    assert load_pending_limits(path) == {"EURUSD": placing}
    assert not list(tmp_path.glob(".pending-limits.json.*.tmp"))


@pytest.mark.parametrize(
    "contents",
    (
        "{not-json",
        json.dumps({"schema_version": 999, "plans": {}}),
        json.dumps({"schema_version": 1, "plans": []}),
    ),
)
def test_existing_corrupt_state_fails_closed_instead_of_becoming_empty(
    tmp_path,
    contents,
):
    path = tmp_path / "pending-limits.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(PendingLimitValidationError):
        load_pending_limits(path)


def test_expiry_deadline_is_exclusive_and_never_renews():
    plan = _plan()

    assert plan.is_due(EXPIRES_AT - 1e-6) is False
    assert plan.expire_if_due(EXPIRES_AT - 1e-6) is plan

    expired = plan.expire_if_due(EXPIRES_AT)
    assert expired.state is PendingLimitState.EXPIRED
    assert expired.is_terminal is True
    assert expired.expires_at == EXPIRES_AT


def test_due_placed_plan_requests_cancellation_before_claiming_expiry():
    placed = _placed()

    due = placed.expire_if_due(EXPIRES_AT)

    assert due.state is PendingLimitState.CANCELLING
    assert due.is_terminal is False
    assert due.working_order_tickets == (1001, 1002, 1003)
    assert due.expires_at == EXPIRES_AT


def test_any_fill_is_atomic_and_exposes_remaining_leg_cancellation_intent():
    placed = _placed()

    filled_one_leg = placed.record_fill(
        1,
        0.1,
        1.1000,
        deal_ticket=2001,
        position_id=3001,
        now=NOW + 10,
    )

    assert filled_one_leg.state is PendingLimitState.PARTIAL
    assert filled_one_leg.has_fills is True
    assert filled_one_leg.working_order_tickets == (1002, 1003)
    assert filled_one_leg.legs[0].filled_volume == pytest.approx(0.1)
    assert filled_one_leg.legs[0].deal_tickets == (2001,)

    cancelling = filled_one_leg.transition(
        PendingLimitState.CANCELLING,
        now=NOW + 11,
        reason="a fill requires cancellation of every unfilled sibling",
    )
    assert cancelling.working_order_tickets == (1002, 1003)

