from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from backtest.cost_model import CostProfile
from backtest.global_policy_replay import (
    GlobalReplayError,
    PendingInvalidation,
    ReplayPolicyConfig,
    UnsupportedReplayFeature,
    run_four_arm_global_replay,
    run_global_policy_replay,
)
from backtest.paired_entry_arms import ArmSpec, DecisionCandidate, FoldWindow
from core.shadow_candidate_ranker import PortfolioCorrelationProfile


FOLD = FoldWindow(
    fold_index=1,
    test_start=pd.Timestamp("2026-01-01T00:00:00Z"),
    test_end=pd.Timestamp("2026-01-01T00:20:00Z"),
)
PRODUCTION_OPEN = ArmSpec("production_priority_one", "m1_open_in_range")


def _cost_profile(
    *,
    created_at: str = "2025-12-31T22:30:00Z",
    measured_through: str = "2025-12-31T22:00:00Z",
) -> CostProfile:
    symbol_template = {
        "contract_size": 100_000,
        "spread_price": 0.10,
        "slippage_price_per_side": 0.02,
        "commission_round_turn_per_lot": 4.0,
        "swap_long_per_lot_rollover": -1.0,
        "swap_short_per_lot_rollover": -1.0,
    }
    return CostProfile.from_mapping(
        {
            "schema": "fx-cost-profile-v1",
            "profile_id": "cost-test-v1",
            "account_currency": "USD",
            "measured_from": "test fixture",
            "created_at_utc": created_at,
            "measured_through_utc": measured_through,
            "rollover_timezone": "UTC",
            "rollover_hour": 22,
            "triple_swap_weekday": 2,
            "symbols": {
                "EURUSD": {
                    **symbol_template,
                    "base_currency": "EUR",
                    "quote_currency": "USD",
                },
                "GBPUSD": {
                    **symbol_template,
                    "base_currency": "GBP",
                    "quote_currency": "USD",
                },
            },
        }
    )


def _correlation_profile() -> PortfolioCorrelationProfile:
    return PortfolioCorrelationProfile.from_mapping(
        {
            "schema": "portfolio-correlation-profile-v1",
            "profile_id": "corr-test-v1",
            "trained_start_utc": "2025-01-01T00:00:00Z",
            "trained_end_utc": "2025-12-31T22:00:00Z",
            "return_timeframe": "H1",
            "method": "pearson-shrunk",
            "shrinkage": 0.1,
            "penalty_scale_r": 1.0,
            "max_penalty_r": 1.0,
            "correlations": {
                "EURUSD": {"EURUSD": 1.0, "GBPUSD": 0.9},
                "GBPUSD": {"EURUSD": 0.9, "GBPUSD": 1.0},
            },
        }
    )


def _candidate(
    parent: str,
    *,
    decision: str,
    expires: str,
    arbitration: str,
    decision_event: str | None = None,
    symbol: str = "EURUSD",
    side: str = "LONG",
    trigger: str = "order_block_1h",
    priority: int = 0,
    entry_min: float = 99.0,
    entry_max: float = 101.0,
    planned_entry: float = 100.0,
    expected_net_r: float | None = None,
    fold: FoldWindow = FOLD,
    profile: CostProfile | None = None,
) -> DecisionCandidate:
    is_long = side == "LONG"
    stop = entry_min - 4.0 if is_long else entry_max + 4.0
    tps = (
        (entry_max + 9.0, entry_max + 19.0, entry_max + 29.0)
        if is_long
        else (entry_min - 9.0, entry_min - 19.0, entry_min - 29.0)
    )
    decision_time = pd.Timestamp(decision)
    if expected_net_r is not None:
        profile = profile or _cost_profile()
    return DecisionCandidate(
        parent_opportunity_id=parent,
        decision_event_id=decision_event or f"decision-{arbitration}",
        arbitration_group_id=arbitration,
        fold_index=fold.fold_index,
        symbol=symbol,
        trigger_kind=trigger,
        side=side,
        bar_close_time=decision_time - pd.Timedelta(minutes=1),
        decision_time=decision_time,
        expires_at=pd.Timestamp(expires),
        entry_min=entry_min,
        entry_max=entry_max,
        planned_entry=planned_entry,
        stop=stop,
        tp_prices=tps,
        production_priority=priority,
        expected_gross_r=(
            expected_net_r + 0.1 if expected_net_r is not None else None
        ),
        estimated_cost_r=0.1 if expected_net_r is not None else None,
        expected_net_r=expected_net_r,
        ranking_basis="expected_net_r" if expected_net_r is not None else None,
        model_id="model-test-v1" if expected_net_r is not None else None,
        model_train_end=(
            pd.Timestamp("2025-12-31T21:00:00Z")
            if expected_net_r is not None
            else None
        ),
        known_at=decision_time if expected_net_r is not None else None,
        cost_profile_id=(
            profile.profile_id
            if expected_net_r is not None and profile is not None
            else None
        ),
        cost_profile_sha256=(
            profile.profile_sha256
            if expected_net_r is not None and profile is not None
            else None
        ),
        cost_profile_created_at_utc=(
            pd.Timestamp(profile.created_at_utc)
            if expected_net_r is not None and profile is not None
            else None
        ),
        cost_profile_measured_through=(
            pd.Timestamp(profile.measured_through_utc)
            if expected_net_r is not None and profile is not None
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


def _run(
    groups,
    frame,
    *,
    config: ReplayPolicyConfig = ReplayPolicyConfig(),
    folds=(FOLD,),
    invalidations=(),
):
    return run_global_policy_replay(
        decision_groups=groups,
        folds=folds,
        m1_by_symbol={"EURUSD": frame},
        data_snapshot_id="test-m1-v1",
        spec=PRODUCTION_OPEN,
        config=config,
        invalidations=invalidations,
    )


def test_fill_is_processed_before_later_decision_at_same_utc_timestamp():
    first = _candidate(
        "first",
        decision="2026-01-01T00:10:00Z",
        expires="2026-01-01T00:14:00Z",
        arbitration="arb-1",
    )
    later = _candidate(
        "later",
        decision="2026-01-01T00:11:00Z",
        expires="2026-01-01T00:15:00Z",
        arbitration="arb-2",
    )
    frame = _m1(
        ("2026-01-01T00:11:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T00:12:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T00:13:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T00:14:00Z", 100.0, 101.0, 99.0, 100.0),
    )

    result = _run([(later,), (first,)], frame)

    same_time = [
        row
        for row in result.events
        if row["timestamp"] == "2026-01-01T00:11:00+00:00"
    ]
    assert [row["kind"] for row in same_time] == ["FILL", "DECISION_REJECTED"]
    assert same_time[1]["reason"] == "SYMBOL_HAS_ACTIVE_IDEA"
    assert len(result.setups) == 1
    assert result.setups[0]["parent_opportunity_id"] == "first"


def test_expiry_and_external_invalidation_precede_equal_time_decisions_and_fill():
    expiring = _candidate(
        "expiring",
        decision="2026-01-01T00:10:00Z",
        expires="2026-01-01T00:12:00Z",
        arbitration="arb-expire",
        entry_min=90.0,
        entry_max=91.0,
        planned_entry=90.5,
    )
    after_expiry = _candidate(
        "after-expiry",
        decision="2026-01-01T00:12:00Z",
        expires="2026-01-01T00:16:00Z",
        arbitration="arb-after",
    )
    frame = _m1(
        ("2026-01-01T00:11:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T00:13:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T00:14:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T00:15:00Z", 100.0, 101.0, 99.0, 100.0),
    )
    result = _run([(after_expiry,), (expiring,)], frame)
    at_expiry = [
        row
        for row in result.events
        if row["timestamp"] == "2026-01-01T00:12:00+00:00"
    ]
    assert [row["kind"] for row in at_expiry] == [
        "PENDING_TERMINAL",
        "DECISION_ARMED",
    ]

    invalidated = _candidate(
        "invalidated",
        decision="2026-01-01T00:10:00Z",
        expires="2026-01-01T00:14:00Z",
        arbitration="arb-invalidated",
    )
    result = _run(
        [(invalidated,)],
        frame,
        invalidations=(
            PendingInvalidation(
                fold_index=1,
                parent_opportunity_id="invalidated",
                observed_at=pd.Timestamp("2026-01-01T00:11:00Z"),
                reason="structure changed",
            ),
        ),
    )
    terminal = next(
        row for row in result.events if row["kind"] == "PENDING_TERMINAL"
    )
    assert terminal["event_priority"] < 30
    assert result.setups[0]["status"] == "INVALIDATED_EXTERNAL"
    assert result.setups[0]["fill_time"] is None


def test_net_rank_uses_only_causal_active_exposure_not_future_fill():
    cost = _cost_profile()
    correlation = _correlation_profile()
    active_parent = _candidate(
        "eur-active",
        decision="2026-01-01T00:05:00Z",
        expires="2026-01-01T00:09:00Z",
        arbitration="arb-eur",
        expected_net_r=0.5,
        profile=cost,
    )
    future_filler = _candidate(
        "gbp-long-fills",
        decision="2026-01-01T00:07:00Z",
        expires="2026-01-01T00:11:00Z",
        arbitration="arb-gbp",
        symbol="GBPUSD",
        side="LONG",
        priority=0,
        expected_net_r=0.8,
        profile=cost,
    )
    rank_one_no_fill = _candidate(
        "gbp-short-no-fill",
        decision="2026-01-01T00:07:00Z",
        expires="2026-01-01T00:11:00Z",
        arbitration="arb-gbp",
        symbol="GBPUSD",
        side="SHORT",
        priority=1,
        entry_min=109.0,
        entry_max=111.0,
        planned_entry=110.0,
        expected_net_r=0.7,
        profile=cost,
    )
    eur = _m1(
        ("2026-01-01T00:06:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T00:07:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T00:08:00Z", 100.0, 101.0, 99.0, 100.0),
    )
    gbp = _m1(
        ("2026-01-01T00:08:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T00:09:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T00:10:00Z", 100.0, 101.0, 99.0, 100.0),
    )

    result = run_global_policy_replay(
        decision_groups=[
            (future_filler, rank_one_no_fill),
            (active_parent,),
        ],
        folds=[FOLD],
        m1_by_symbol={"EURUSD": eur, "GBPUSD": gbp},
        data_snapshot_id="test-m1-v1",
        spec=ArmSpec("net_rank_one", "m1_open_in_range"),
        cost_profile=cost,
        correlation_profile=correlation,
    )

    gbp_setup = next(row for row in result.setups if row["symbol"] == "GBPUSD")
    assert gbp_setup["parent_opportunity_id"] == "gbp-short-no-fill"
    assert gbp_setup["correlation_penalty_r"] == 0.0
    assert gbp_setup["status"] == "NO_FILL_EXPIRED"
    alternatives = {row["parent_opportunity_id"]: row for row in gbp_setup["alternatives"]}
    assert alternatives["gbp-long-fills"]["correlation_penalty_r"] == pytest.approx(0.9)
    assert alternatives["gbp-long-fills"]["selected"] is False


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        (
            ReplayPolicyConfig(cooldown_after_exit=pd.Timedelta(minutes=5)),
            "SYMBOL_COOLDOWN",
        ),
        (ReplayPolicyConfig(daily_loss_cap_r=0.5), "DAILY_LOSS_CAP"),
    ],
)
def test_exit_precedes_equal_time_decision_and_applies_stateful_gate(config, reason):
    loser = _candidate(
        "loser",
        decision="2026-01-01T00:10:00Z",
        expires="2026-01-01T00:14:00Z",
        arbitration="arb-loss",
    )
    next_signal = _candidate(
        "next",
        decision="2026-01-01T00:12:00Z",
        expires="2026-01-01T00:16:00Z",
        arbitration="arb-next",
    )
    frame = _m1(
        ("2026-01-01T00:11:00Z", 100.0, 101.0, 94.0, 95.0),
        ("2026-01-01T00:13:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T00:14:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T00:15:00Z", 100.0, 101.0, 99.0, 100.0),
    )
    result = _run([(next_signal,), (loser,)], frame, config=config)
    at_exit = [
        row
        for row in result.events
        if row["timestamp"] == "2026-01-01T00:12:00+00:00"
    ]
    assert [row["kind"] for row in at_exit] == [
        "POSITION_EXIT",
        "DECISION_REJECTED",
    ]
    assert at_exit[1]["reason"] == reason


def test_duplicate_gate_is_explicit_and_recorded_on_arm():
    first = _candidate(
        "dup-1",
        decision="2026-01-01T00:10:00Z",
        expires="2026-01-01T00:12:00Z",
        arbitration="arb-dup-1",
        entry_min=90.0,
        entry_max=91.0,
        planned_entry=90.5,
    )
    duplicate = _candidate(
        "dup-2",
        decision="2026-01-01T00:12:00Z",
        expires="2026-01-01T00:16:00Z",
        arbitration="arb-dup-2",
    )
    frame = _m1(
        ("2026-01-01T00:11:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T00:13:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T00:14:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T00:15:00Z", 100.0, 101.0, 99.0, 100.0),
    )
    result = _run(
        [(duplicate,), (first,)],
        frame,
        config=ReplayPolicyConfig(
            duplicate_window=pd.Timedelta(hours=1),
            duplicate_key_policy="symbol_trigger_side",
            duplicate_record_on="arm",
        ),
    )
    rejected = next(row for row in result.events if row["kind"] == "DECISION_REJECTED")
    assert rejected["reason"] == "DUPLICATE_TRIGGER_WINDOW"


def test_fold_boundary_censors_without_carry_then_same_symbol_can_trade():
    first_fold = FoldWindow(
        fold_index=1,
        test_start=pd.Timestamp("2026-01-01T00:00:00Z"),
        test_end=pd.Timestamp("2026-01-01T00:15:00Z"),
    )
    second_fold = FoldWindow(
        fold_index=2,
        test_start=pd.Timestamp("2026-01-01T00:15:00Z"),
        test_end=pd.Timestamp("2026-01-01T00:30:00Z"),
    )
    first = _candidate(
        "fold-1",
        decision="2026-01-01T00:10:00Z",
        expires="2026-01-01T00:14:00Z",
        arbitration="arb-fold-1",
        fold=first_fold,
    )
    second = _candidate(
        "fold-2",
        decision="2026-01-01T00:15:00Z",
        expires="2026-01-01T00:19:00Z",
        arbitration="arb-fold-2",
        fold=second_fold,
    )
    frame = _m1(
        ("2026-01-01T00:11:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T00:12:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T00:16:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T00:17:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T00:18:00Z", 100.0, 101.0, 99.0, 100.0),
    )
    result = _run(
        [(second,), (first,)],
        frame,
        folds=(second_fold, first_fold),
    )
    by_parent = {row["parent_opportunity_id"]: row for row in result.setups}
    assert by_parent["fold-1"]["status"] == "CENSORED_ACTIVE_FOLD_END"
    assert by_parent["fold-2"]["fill_time"] == "2026-01-01T00:16:00+00:00"
    boundary = [
        row
        for row in result.events
        if row["timestamp"] == "2026-01-01T00:15:00+00:00"
    ]
    assert [row["kind"] for row in boundary] == [
        "FOLD_END_ACTIVE_CENSOR",
        "DECISION_ARMED",
    ]


def test_four_arm_factorial_reports_closed_gross_and_net_cost_metrics():
    cost = _cost_profile()
    correlation = _correlation_profile()
    candidate = _candidate(
        "four-arm",
        decision="2026-01-01T00:10:00Z",
        expires="2026-01-01T00:15:00Z",
        arbitration="arb-four",
        expected_net_r=0.5,
        profile=cost,
    )
    frame = _m1(
        ("2026-01-01T00:11:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T00:12:00Z", 100.0, 131.0, 99.0, 130.0),
        ("2026-01-01T00:13:00Z", 130.0, 131.0, 129.0, 130.0),
        ("2026-01-01T00:14:00Z", 130.0, 131.0, 129.0, 130.0),
    )
    results = run_four_arm_global_replay(
        decision_groups=[(candidate,)],
        folds=[FOLD],
        m1_by_symbol={"EURUSD": frame},
        data_snapshot_id="test-m1-v1",
        cost_profile=cost,
        correlation_profile=correlation,
    )
    assert [result.arm for result in results] == [
        "production_priority_one__m1_open_in_range",
        "production_priority_one__planned_limit_touch",
        "net_rank_one__m1_open_in_range",
        "net_rank_one__planned_limit_touch",
    ]
    for result in results:
        assert result.metrics["gross"]["closed_setups"] == 1
        assert result.metrics["gross"]["net_r"] == pytest.approx(3.4)
        assert result.metrics["net_after_cost"]["net_r"] < 3.4
        assert result.setups[0]["arbitration_group_id"] == "arb-four"


def test_experiment_input_id_is_shared_by_arms_and_ignores_group_order():
    cost = _cost_profile()
    correlation = _correlation_profile()
    candidate = _candidate(
        "input-id",
        decision="2026-01-01T00:10:00Z",
        expires="2026-01-01T00:15:00Z",
        arbitration="arb-a",
        expected_net_r=0.5,
        profile=cost,
    )
    other = _candidate(
        "input-id-other",
        decision="2026-01-01T00:12:00Z",
        expires="2026-01-01T00:16:00Z",
        arbitration="arb-b",
        expected_net_r=0.4,
        profile=cost,
    )
    frame = _m1(
        ("2026-01-01T00:11:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T00:12:00Z", 100.0, 131.0, 99.0, 130.0),
        ("2026-01-01T00:13:00Z", 130.0, 131.0, 129.0, 130.0),
        ("2026-01-01T00:14:00Z", 130.0, 131.0, 129.0, 130.0),
    )

    def _run_four(groups):
        return run_four_arm_global_replay(
            decision_groups=groups,
            folds=[FOLD],
            m1_by_symbol={"EURUSD": frame},
            data_snapshot_id="test-m1-v1",
            cost_profile=cost,
            correlation_profile=correlation,
        )

    results = _run_four([(candidate,), (other,)])
    input_ids = {result.experiment_input_id for result in results}
    assert len(input_ids) == 1
    assert len({result.replay_id for result in results}) == 4
    for result in results:
        assert result.metrics["experiment_input_id"] == result.experiment_input_id
        assert result.to_dict()["experiment_input_id"] == result.experiment_input_id

    permuted = _run_four([(other,), (candidate,)])
    assert {row.experiment_input_id for row in permuted} == input_ids

    changed = run_four_arm_global_replay(
        decision_groups=[(candidate,)],
        folds=[FOLD],
        m1_by_symbol={"EURUSD": frame},
        data_snapshot_id="test-m1-v1",
        cost_profile=cost,
        correlation_profile=correlation,
    )
    assert {row.experiment_input_id for row in changed} != input_ids


def test_noncausal_cost_is_post_hoc_stress_only_for_production_arm():
    future_cost = _cost_profile(
        created_at="2026-01-01T00:05:00Z",
        measured_through="2026-01-01T00:04:00Z",
    )
    production_candidate = _candidate(
        "static-stress",
        decision="2026-01-01T00:10:00Z",
        expires="2026-01-01T00:15:00Z",
        arbitration="arb-stress",
    )
    frame = _m1(
        ("2026-01-01T00:11:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T00:12:00Z", 100.0, 131.0, 99.0, 130.0),
        ("2026-01-01T00:13:00Z", 130.0, 131.0, 129.0, 130.0),
        ("2026-01-01T00:14:00Z", 130.0, 131.0, 129.0, 130.0),
    )
    result = run_global_policy_replay(
        decision_groups=[(production_candidate,)],
        folds=[FOLD],
        m1_by_symbol={"EURUSD": frame},
        data_snapshot_id="test-m1-v1",
        spec=PRODUCTION_OPEN,
        cost_profile=future_cost,
    )
    assert result.metrics["cost_profile_usage"] == "POST_HOC_STATIC_STRESS"
    assert (
        result.metrics["cost_profile_point_in_time"][
            "causal_for_all_fold_test_starts"
        ]
        is False
    )
    assert result.metrics["net_after_cost"] is not None
    decision = next(
        row for row in result.events if row["kind"] == "DECISION_ARMED"
    )
    assert decision["selected_ranking_score"] is None
    assert decision["correlation_penalty_r"] is None

    net_candidate = _candidate(
        "future-cost-rank",
        decision="2026-01-01T00:10:00Z",
        expires="2026-01-01T00:15:00Z",
        arbitration="arb-future-cost",
        expected_net_r=0.5,
        profile=future_cost,
    )
    with pytest.raises(GlobalReplayError, match="before the OOS fold"):
        run_global_policy_replay(
            decision_groups=[(net_candidate,)],
            folds=[FOLD],
            m1_by_symbol={"EURUSD": frame},
            data_snapshot_id="test-m1-v1",
            spec=ArmSpec("net_rank_one", "m1_open_in_range"),
            cost_profile=future_cost,
            correlation_profile=_correlation_profile(),
        )


def test_unsupported_policy_features_fail_closed():
    with pytest.raises(UnsupportedReplayFeature, match="carry_across_folds"):
        ReplayPolicyConfig(carry_across_folds=True)
    with pytest.raises(UnsupportedReplayFeature, match="position adding"):
        ReplayPolicyConfig(allow_position_adding=True)
    with pytest.raises(GlobalReplayError, match="net daily cap"):
        run_global_policy_replay(
            decision_groups=[],
            folds=[FOLD],
            m1_by_symbol={},
            data_snapshot_id="test-m1-v1",
            spec=PRODUCTION_OPEN,
            config=ReplayPolicyConfig(
                daily_loss_cap_r=1.0,
                daily_cap_basis="net_after_cost_r",
            ),
        )


def test_non_three_leg_candidate_is_explicitly_unsupported():
    candidate = _candidate(
        "bad-legs",
        decision="2026-01-01T00:10:00Z",
        expires="2026-01-01T00:14:00Z",
        arbitration="arb-bad",
    )
    candidate = replace(candidate, tp_prices=(110.0, 120.0))
    with pytest.raises(UnsupportedReplayFeature, match="three TP"):
        _run([(candidate,)], _m1())
