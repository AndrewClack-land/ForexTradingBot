from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from backtest.paired_entry_arms import (
    DecisionCandidate,
    FoldWindow,
    PairedEntryArmError,
    evaluate_four_arms,
    four_arm_specs,
    observe_entry,
    select_candidate,
)


DECISION = pd.Timestamp("2026-01-01T00:10:00Z")
FOLD = FoldWindow(
    fold_index=3,
    test_start=pd.Timestamp("2026-01-01T00:00:00Z"),
    test_end=pd.Timestamp("2026-01-01T01:00:00Z"),
)


def _candidate(
    opportunity_id: str,
    *,
    priority: int,
    expected_net_r: float | None,
    entry_min: float = 99.0,
    entry_max: float = 101.0,
    planned_entry: float = 100.0,
    expires_at: str = "2026-01-01T00:14:00Z",
    model_train_end: str | None = "2025-12-31T23:00:00Z",
    known_at: str | None = "2026-01-01T00:10:00Z",
    cost_profile_id: str | None = "cost-1",
    cost_profile_sha256: str | None = "a" * 64,
    cost_profile_created_at_utc: str | None = "2025-12-31T22:30:00Z",
    cost_profile_measured_through: str | None = "2025-12-31T22:00:00Z",
) -> DecisionCandidate:
    return DecisionCandidate(
        parent_opportunity_id=opportunity_id,
        decision_event_id="decision-1",
        arbitration_group_id="arb-1",
        fold_index=3,
        symbol="EURUSD",
        trigger_kind=f"trigger_{opportunity_id}",
        side="LONG",
        bar_close_time=pd.Timestamp("2026-01-01T00:09:00Z"),
        decision_time=DECISION,
        expires_at=pd.Timestamp(expires_at),
        entry_min=entry_min,
        entry_max=entry_max,
        planned_entry=planned_entry,
        stop=95.0,
        tp_prices=(110.0, 120.0, 130.0),
        production_priority=priority,
        expected_gross_r=(expected_net_r + 0.1 if expected_net_r is not None else None),
        estimated_cost_r=0.1 if expected_net_r is not None else None,
        expected_net_r=expected_net_r,
        ranking_basis="expected_net_r" if expected_net_r is not None else None,
        model_id="model-1" if expected_net_r is not None else None,
        model_train_end=(
            pd.Timestamp(model_train_end) if model_train_end is not None else None
        ),
        known_at=pd.Timestamp(known_at) if known_at is not None else None,
        cost_profile_id=(
            cost_profile_id if expected_net_r is not None else None
        ),
        cost_profile_sha256=(
            cost_profile_sha256 if expected_net_r is not None else None
        ),
        cost_profile_created_at_utc=(
            pd.Timestamp(cost_profile_created_at_utc)
            if expected_net_r is not None
            and cost_profile_created_at_utc is not None
            else None
        ),
        cost_profile_measured_through=(
            pd.Timestamp(cost_profile_measured_through)
            if expected_net_r is not None
            and cost_profile_measured_through is not None
            else None
        ),
    )


def _m1(*rows: tuple[str, float, float, float, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"open": open_, "high": high, "low": low, "close": close}
            for _, open_, high, low, close in rows
        ],
        index=pd.DatetimeIndex([timestamp for timestamp, *_ in rows]),
    )


def _by_arm(results):
    return {result.arm: result for result in results}


def test_fixed_design_is_exactly_two_by_two():
    assert [spec.arm for spec in four_arm_specs()] == [
        "production_priority_one__m1_open_in_range",
        "production_priority_one__planned_limit_touch",
        "net_rank_one__m1_open_in_range",
        "net_rank_one__planned_limit_touch",
    ]
    assert [spec.sensitivity_only for spec in four_arm_specs()] == [
        False,
        True,
        False,
        True,
    ]


def test_candidate_policies_choose_priority_or_expected_net_at_decision():
    priority = _candidate("priority", priority=0, expected_net_r=0.1)
    net = _candidate("net", priority=4, expected_net_r=0.8)

    assert (
        select_candidate(
            [net, priority],
            policy="production_priority_one",
            fold=FOLD,
        ).parent_opportunity_id
        == "priority"
    )
    assert (
        select_candidate(
            [priority, net],
            policy="net_rank_one",
            fold=FOLD,
        ).parent_opportunity_id
        == "net"
    )


def test_net_rank_never_falls_back_to_candidate_that_fills_in_future():
    future_filler = _candidate(
        "future-filler",
        priority=0,
        expected_net_r=0.1,
        entry_min=101.0,
        entry_max=102.0,
        planned_entry=101.5,
    )
    rank_one_no_fill = _candidate(
        "rank-one-no-fill",
        priority=1,
        expected_net_r=0.9,
        entry_min=99.0,
        entry_max=100.0,
        planned_entry=99.5,
    )
    bars = _m1(
        ("2026-01-01T00:11:00Z", 101.5, 102.0, 101.0, 101.5),
        ("2026-01-01T00:12:00Z", 101.5, 102.0, 101.0, 101.5),
        ("2026-01-01T00:13:00Z", 101.5, 102.0, 101.0, 101.5),
    )

    results = _by_arm(
        evaluate_four_arms(
            [future_filler, rank_one_no_fill],
            bars,
            fold=FOLD,
        )
    )

    assert (
        results[
            "production_priority_one__m1_open_in_range"
        ].candidate.parent_opportunity_id
        == "future-filler"
    )
    assert results["production_priority_one__m1_open_in_range"].status == "FILLED"
    for arm in (
        "net_rank_one__m1_open_in_range",
        "net_rank_one__planned_limit_touch",
    ):
        assert results[arm].candidate.parent_opportunity_id == ("rank-one-no-fill")
        assert results[arm].status == "NO_FILL"


def test_planned_limit_touch_is_labeled_m1_range_sensitivity():
    candidate = _candidate("one", priority=0, expected_net_r=0.5)
    bars = _m1(
        ("2026-01-01T00:11:00Z", 102.0, 103.0, 100.0, 102.0),
        ("2026-01-01T00:12:00Z", 102.0, 103.0, 101.5, 102.0),
        ("2026-01-01T00:13:00Z", 102.0, 103.0, 101.5, 102.0),
    )

    open_result = observe_entry(
        candidate,
        bars,
        policy="m1_open_in_range",
        fold=FOLD,
    )
    touch_result = observe_entry(
        candidate,
        bars,
        policy="planned_limit_touch",
        fold=FOLD,
    )

    assert open_result.status == "NO_FILL"
    assert touch_result.status == "FILLED"
    assert touch_result.entry_price == 100.0
    assert touch_result.observed_at == pd.Timestamp("2026-01-01T00:12:00Z")

    rows = [
        result.to_audit_row()
        for result in evaluate_four_arms([candidate], bars, fold=FOLD)
        if result.spec.entry_policy == "planned_limit_touch"
    ]
    assert all(row["sensitivity_only"] is True for row in rows)
    assert all(row["entry_time_precision"] == "M1_RANGE_WINDOW" for row in rows)
    assert all(row["m1_trigger_time"] is None for row in rows)


def test_same_bar_planned_entry_and_stop_is_never_a_clean_fill():
    candidate = _candidate("one", priority=0, expected_net_r=0.5)
    bars = _m1(
        ("2026-01-01T00:11:00Z", 102.0, 103.0, 94.0, 96.0),
        ("2026-01-01T00:12:00Z", 102.0, 103.0, 101.5, 102.0),
        ("2026-01-01T00:13:00Z", 102.0, 103.0, 101.5, 102.0),
    )

    result = observe_entry(
        candidate,
        bars,
        policy="planned_limit_touch",
        fold=FOLD,
    )

    assert result.status == "AMBIGUOUS_FILL_STOP"
    assert result.same_bar_stop_ambiguous is True
    assert "intrabar order is unknown" in result.reason


def test_decision_bar_and_expiry_open_are_both_excluded():
    candidate = _candidate("one", priority=0, expected_net_r=0.5)
    bars = _m1(
        ("2026-01-01T00:10:00Z", 100.0, 100.5, 99.5, 100.0),
        ("2026-01-01T00:11:00Z", 102.0, 103.0, 101.5, 102.0),
        ("2026-01-01T00:12:00Z", 102.0, 103.0, 101.5, 102.0),
        ("2026-01-01T00:13:00Z", 102.0, 103.0, 101.5, 102.0),
        ("2026-01-01T00:14:00Z", 100.0, 100.5, 99.5, 100.0),
    )

    assert (
        observe_entry(
            candidate,
            bars,
            policy="m1_open_in_range",
            fold=FOLD,
        ).status
        == "NO_FILL"
    )
    assert (
        observe_entry(
            candidate,
            bars,
            policy="planned_limit_touch",
            fold=FOLD,
        ).status
        == "NO_FILL"
    )


def test_fold_cutoff_and_partial_final_bar_are_censored():
    fold = FoldWindow(
        fold_index=3,
        test_start=FOLD.test_start,
        test_end=pd.Timestamp("2026-01-01T00:14:00Z"),
    )
    fold_clipped = _candidate(
        "fold-clipped",
        priority=0,
        expected_net_r=0.5,
        expires_at="2026-01-01T00:16:00Z",
    )
    complete_to_fold = _m1(
        ("2026-01-01T00:11:00Z", 102.0, 103.0, 101.5, 102.0),
        ("2026-01-01T00:12:00Z", 102.0, 103.0, 101.5, 102.0),
        ("2026-01-01T00:13:00Z", 102.0, 103.0, 101.5, 102.0),
    )
    result = observe_entry(
        fold_clipped,
        complete_to_fold,
        policy="m1_open_in_range",
        fold=fold,
    )
    assert result.status == "CENSORED"
    assert result.fold_clipped is True

    partial = _candidate(
        "partial",
        priority=0,
        expected_net_r=0.5,
        expires_at="2026-01-01T00:13:30Z",
    )
    partial_bars = _m1(
        ("2026-01-01T00:11:00Z", 102.0, 103.0, 101.5, 102.0),
        ("2026-01-01T00:12:00Z", 102.0, 103.0, 101.5, 102.0),
        # This range straddles the exclusive 00:13:30 deadline and cannot be
        # used as evidence of an earlier touch.
        ("2026-01-01T00:13:00Z", 102.0, 103.0, 100.0, 102.0),
    )
    result = observe_entry(
        partial,
        partial_bars,
        policy="planned_limit_touch",
        fold=FOLD,
    )
    assert result.status == "CENSORED"
    assert result.entry_price is None

    # Open-based parity can use the 00:13 open before a 00:13:30 cutoff, so
    # omitting that row is also explicit censorship rather than a false expiry.
    result = observe_entry(
        partial,
        partial_bars.iloc[:2],
        policy="m1_open_in_range",
        fold=FOLD,
    )
    assert result.status == "CENSORED"

    inside_bar = replace(
        _candidate("inside-bar", priority=0, expected_net_r=0.5),
        decision_time=pd.Timestamp("2026-01-01T00:10:30Z"),
    )
    result = observe_entry(
        inside_bar,
        partial_bars,
        policy="planned_limit_touch",
        fold=FOLD,
    )
    assert result.status == "CENSORED"
    assert "starts inside" in result.reason


def test_net_rank_requires_frozen_causal_score_provenance():
    missing_known_at = _candidate(
        "missing-known-at",
        priority=0,
        expected_net_r=0.5,
        known_at=None,
    )
    with pytest.raises(PairedEntryArmError, match="model_train_end and known_at"):
        select_candidate(
            [missing_known_at],
            policy="net_rank_one",
            fold=FOLD,
        )

    with pytest.raises(PairedEntryArmError, match="model_train_end cannot follow"):
        _candidate(
            "future-model",
            priority=0,
            expected_net_r=0.5,
            model_train_end="2026-01-01T00:11:00Z",
        )


def test_net_rank_requires_complete_causal_cost_provenance():
    missing_profile_time = _candidate(
        "missing-profile-time",
        priority=0,
        expected_net_r=0.5,
        cost_profile_created_at_utc=None,
    )
    with pytest.raises(
        PairedEntryArmError,
        match="cost_profile_created_at_utc and cost_profile_measured_through",
    ):
        select_candidate(
            [missing_profile_time],
            policy="net_rank_one",
            fold=FOLD,
        )

    future_profile = _candidate(
        "future-profile",
        priority=0,
        expected_net_r=0.5,
        cost_profile_created_at_utc="2026-01-01T00:05:00Z",
        cost_profile_measured_through="2025-12-31T23:59:00Z",
    )
    with pytest.raises(PairedEntryArmError, match="before the OOS fold"):
        select_candidate(
            [future_profile],
            policy="net_rank_one",
            fold=FOLD,
        )


def test_net_rank_requires_one_profile_and_consistent_net_arithmetic():
    one = _candidate("one", priority=0, expected_net_r=0.5)
    other_profile = _candidate(
        "other-profile",
        priority=1,
        expected_net_r=0.4,
        cost_profile_id="cost-2",
        cost_profile_sha256="b" * 64,
    )
    with pytest.raises(PairedEntryArmError, match="one identical cost profile"):
        select_candidate(
            [one, other_profile],
            policy="net_rank_one",
            fold=FOLD,
        )

    inconsistent = replace(one, expected_net_r=0.7)
    with pytest.raises(
        PairedEntryArmError,
        match="expected_gross_r - estimated_cost_r",
    ):
        select_candidate(
            [inconsistent],
            policy="net_rank_one",
            fold=FOLD,
        )


def test_audit_preserves_parent_and_arbitration_identity():
    candidate = _candidate("parent-7", priority=2, expected_net_r=0.5)
    bars = _m1(
        ("2026-01-01T00:11:00Z", 100.5, 101.0, 100.0, 100.5),
        ("2026-01-01T00:12:00Z", 102.0, 103.0, 101.5, 102.0),
        ("2026-01-01T00:13:00Z", 102.0, 103.0, 101.5, 102.0),
    )

    row = evaluate_four_arms([candidate], bars, fold=FOLD)[0].to_audit_row()

    assert row["parent_opportunity_id"] == "parent-7"
    assert row["decision_event_id"] == "decision-1"
    assert row["arbitration_group_id"] == "arb-1"
    assert row["arbitration_scope"] == (
        "single_decision_only_no_timeline_state"
    )
    assert row["context_decision_time"] == DECISION.isoformat()
    assert row["model_train_end"] == "2025-12-31T23:00:00+00:00"
    assert row["known_at"] == DECISION.isoformat()
    assert row["cost_profile_created_at_utc"] == (
        "2025-12-31T22:30:00+00:00"
    )
    assert row["cost_profile_measured_through"] == (
        "2025-12-31T22:00:00+00:00"
    )
