from __future__ import annotations

import json
from dataclasses import replace

import pytest

from core.persistent_limit import (
    BROKER_COMMENT_PREFIX,
    PENDING_LIMIT_SCHEMA_VERSION,
    PendingLimitLeg,
    PendingLimitPlan,
    PendingLimitState,
    PendingLimitValidationError,
    load_pending_limits,
    make_json_safe_signal,
    save_pending_limits,
)


def _signal() -> dict:
    return {
        "signal": "ENTER",
        "side": "LONG",
        "entry_price": 1.1000,
        "entry_min": 1.0995,
        "entry_max": 1.1000,
        "stop_price": 1.0950,
        "tp_price": 1.1150,
        "tp_prices": [1.1050, 1.1100, 1.1150],
        "tf": "15M",
        "narrative": "test",
        "trigger_meta": {"labels": ("a", "b")},
    }


def _plan(*, volume: float = 0.3, now: float = 100.0) -> PendingLimitPlan:
    return PendingLimitPlan.create(
        symbol="eurusd",
        signal=_signal(),
        trigger_signature="LONG|pivot|z1.09950-1.10000",
        expires_at=now + 900.0,
        magic=20260318,
        legs=[
            PendingLimitLeg(
                index=3,
                target_price=1.1150,
                requested_volume=volume,
            )
        ],
        now=now,
        plan_id="0123456789abcdef0123456789abcdef",
    )


def test_create_has_stable_identity_and_detached_json_payload() -> None:
    signal = _signal()
    plan = PendingLimitPlan.create(
        symbol="eurusd",
        signal=signal,
        trigger_signature="LONG|pivot|zone",
        expires_at=1000.0,
        magic=7,
        legs=[PendingLimitLeg(3, 1.1150, 0.3)],
        now=100.0,
        plan_id="abcdef0123456789abcdef0123456789",
    )

    assert plan.symbol == "EURUSD"
    assert plan.state is PendingLimitState.PLACING
    assert plan.signal_payload["setup_id"] == plan.plan_id
    assert plan.signal_payload["trigger_meta"]["labels"] == ["a", "b"]
    assert "setup_id" not in signal
    assert plan.broker_comment.startswith(BROKER_COMMENT_PREFIX)
    assert len(plan.broker_comment) <= 31
    assert plan.to_dict()["plan_id"] == plan.plan_id


def test_per_leg_comments_share_ownership_prefix_but_are_distinct() -> None:
    signal = _signal()
    legs = [
        PendingLimitLeg(index=i, target_price=target, requested_volume=0.1)
        for i, target in enumerate(signal["tp_prices"], start=1)
    ]
    plan = PendingLimitPlan.create(
        symbol="EURUSD",
        signal=signal,
        trigger_signature="LONG|pivot|zone",
        expires_at=1000.0,
        magic=7,
        legs=legs,
        now=100.0,
        plan_id="abcdef0123456789abcdef0123456789",
    )

    comments = [plan.broker_comment_for_leg(i) for i in (1, 2, 3)]
    assert len(set(comments)) == 3
    assert all(comment.startswith(BROKER_COMMENT_PREFIX) for comment in comments)
    assert all(len(comment) <= 31 for comment in comments)
    with pytest.raises(PendingLimitValidationError, match="unknown leg"):
        plan.broker_comment_for_leg(4)


def test_attach_order_is_atomic_for_a_multileg_placement() -> None:
    signal = _signal()
    plan = PendingLimitPlan.create(
        symbol="EURUSD",
        signal=signal,
        trigger_signature="LONG|pivot|zone",
        expires_at=1000.0,
        magic=7,
        legs=[
            PendingLimitLeg(i, target, 0.1)
            for i, target in enumerate(signal["tp_prices"], start=1)
        ],
        now=100.0,
        plan_id="abcdef0123456789abcdef0123456789",
    )

    plan = plan.attach_order(1, 101, now=101.0)
    assert plan.state is PendingLimitState.PLACING
    plan = plan.attach_order(2, 102, now=102.0)
    assert plan.state is PendingLimitState.PLACING
    plan = plan.attach_order(3, 103, now=103.0)
    assert plan.state is PendingLimitState.PLACED
    assert plan.working_order_tickets == (101, 102, 103)
    assert plan.attach_order(3, 103, now=104.0).state is PendingLimitState.PLACED


def test_record_fill_is_atomic_idempotent_and_handles_cancel_race() -> None:
    plan = _plan().attach_order(3, 101, now=101.0)

    partial = plan.record_fill(
        3,
        0.1,
        1.0999,
        deal_ticket=501,
        position_id=901,
        now=102.0,
    )
    assert partial.state is PendingLimitState.PARTIAL
    assert partial.legs[0].filled_volume == pytest.approx(0.1)
    assert partial.record_fill(
        3,
        0.1,
        1.0999,
        deal_ticket=501,
        position_id=901,
        now=103.0,
    ) is partial

    cancelling = partial.transition(
        PendingLimitState.CANCELLING,
        now=104.0,
        reason="TTL",
    )
    raced = cancelling.record_fill(
        3,
        0.1,
        1.0998,
        deal_ticket=502,
        position_id=901,
        now=105.0,
    )
    assert raced.state is PendingLimitState.CANCELLING
    assert raced.legs[0].average_fill_price == pytest.approx(1.09985)

    # Broker confirmed the unfilled 0.1 remainder cancelled. The partial
    # exposure is promoted to a filled ActiveTrade, never called cancelled.
    filled = raced.transition(PendingLimitState.FILLED, now=106.0)
    assert filled.is_terminal
    assert filled.has_fills
    assert filled.working_order_tickets == ()


def test_complete_fill_promotes_directly_to_terminal_filled() -> None:
    plan = _plan(volume=0.1).attach_order(3, 101, now=101.0)
    filled = plan.record_fill(
        3,
        0.1,
        1.1,
        deal_ticket=501,
        position_id=901,
        now=102.0,
    )

    assert filled.state is PendingLimitState.FILLED
    with pytest.raises(PendingLimitValidationError, match="terminal"):
        filled.record_fill(3, 0.01, 1.1, deal_ticket=502, now=103.0)


def test_ttl_is_immutable_and_elapsed_ticket_requires_cancellation() -> None:
    plan = _plan(now=100.0)
    expiry = plan.expires_at

    with pytest.raises(PendingLimitValidationError, match="before"):
        plan.transition(PendingLimitState.EXPIRED, now=expiry - 1.0)
    assert plan.expire_if_due(expiry - 1.0) is plan

    unsubmitted = plan.expire_if_due(expiry)
    assert unsubmitted.state is PendingLimitState.EXPIRED
    assert unsubmitted.expires_at == expiry

    placed = plan.attach_order(3, 101, now=101.0)
    cancelling = placed.expire_if_due(expiry)
    assert cancelling.state is PendingLimitState.CANCELLING
    assert cancelling.expires_at == expiry
    assert cancelling.working_order_tickets == (101,)


def test_submission_attempt_is_durable_before_ticket_or_fill() -> None:
    plan = _plan(now=100.0)

    attempted = plan.mark_submission_attempted(3, now=101.0)

    assert attempted.state is PendingLimitState.PLACING
    assert attempted.legs[0].submission_attempted is True
    assert attempted.legs[0].broker_order_ticket is None
    assert PendingLimitPlan.from_dict(attempted.to_dict()) == attempted
    assert attempted.mark_submission_attempted(3, now=102.0) is attempted

    # An attempted request with no returned ticket is ambiguous. Reaching TTL
    # starts cancellation/reconciliation; it must never claim no order existed.
    due = attempted.expire_if_due(attempted.expires_at)
    assert due.state is PendingLimitState.CANCELLING
    assert due.legs[0].submission_attempted is True


def test_entry_announcement_is_a_durable_terminal_outbox_ack() -> None:
    unfilled = _plan(volume=0.1)
    with pytest.raises(PendingLimitValidationError, match="before a broker fill"):
        unfilled.mark_entry_announced(now=101.0)

    filled = unfilled.attach_order(3, 101, now=101.0).record_fill(
        3,
        0.1,
        1.1,
        deal_ticket=501,
        position_id=901,
        now=102.0,
    )
    assert filled.state is PendingLimitState.FILLED
    assert filled.entry_announcement_pending is True
    assert filled.entry_announced_at is None

    restored_pending = PendingLimitPlan.from_dict(filled.to_dict())
    assert restored_pending.entry_announcement_pending is True
    announced = restored_pending.mark_entry_announced(now=103.0)
    assert announced.state is PendingLimitState.FILLED
    assert announced.entry_announced_at == pytest.approx(103.0)
    assert announced.entry_announcement_pending is False
    assert PendingLimitPlan.from_dict(announced.to_dict()) == announced
    assert announced.mark_entry_announced(now=104.0) is announced


def test_strict_geometry_and_state_validation_fail_closed() -> None:
    bad_signal = _signal()
    bad_signal["stop_price"] = 1.1010
    with pytest.raises(PendingLimitValidationError, match="LONG geometry"):
        PendingLimitPlan.create(
            symbol="EURUSD",
            signal=bad_signal,
            trigger_signature="bad",
            expires_at=1000.0,
            magic=7,
            legs=[PendingLimitLeg(3, 1.1150, 0.1)],
            now=100.0,
        )

    plan = _plan().attach_order(3, 101, now=101.0)
    with pytest.raises(PendingLimitValidationError, match="failed plan"):
        replace(plan, state=PendingLimitState.FAILED)


def test_json_safe_signal_rejects_nan_unknown_objects_and_cycles() -> None:
    with pytest.raises(PendingLimitValidationError, match="non-finite"):
        make_json_safe_signal({"value": float("nan")})
    with pytest.raises(PendingLimitValidationError, match="unsupported"):
        make_json_safe_signal({"value": object()})

    cyclic: dict = {}
    cyclic["self"] = cyclic
    with pytest.raises(PendingLimitValidationError, match="cycle"):
        make_json_safe_signal(cyclic)


def test_versioned_atomic_store_round_trip(tmp_path) -> None:
    path = tmp_path / "pending_limits.json"
    plan = _plan().attach_order(3, 101, now=101.0)

    save_pending_limits({"EURUSD": plan}, path)
    restored = load_pending_limits(path)

    assert restored == {"EURUSD": plan}
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == PENDING_LIMIT_SCHEMA_VERSION
    assert raw["plans"]["EURUSD"]["expires_at"] == plan.expires_at
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        json.dumps({"schema_version": 999, "plans": {}}),
        json.dumps(
            {
                "schema_version": PENDING_LIMIT_SCHEMA_VERSION,
                "plans": [],
            }
        ),
    ],
)
def test_existing_corrupt_store_never_becomes_an_empty_book(tmp_path, payload) -> None:
    path = tmp_path / "pending_limits.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(PendingLimitValidationError):
        load_pending_limits(path)


def test_missing_store_is_the_only_empty_store_case(tmp_path) -> None:
    assert load_pending_limits(tmp_path / "missing.json") == {}
