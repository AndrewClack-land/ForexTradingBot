"""Tests for the causal hierarchical quality scorer."""

from __future__ import annotations

import copy
import math
from datetime import datetime, timedelta, timezone

import pytest

from core.hierarchical_quality_score import (
    QUALITY_PROFILE_SCHEMA,
    R_BASIS_GROSS,
    R_BASIS_NET,
    HierarchicalQualityProfile,
    LiveHierarchicalQualityScorer,
    build_quality_features,
    fit_quality_profile,
    rank_quality_candidates,
)


TRAIN_START = datetime(2026, 5, 4, 9, 30, tzinfo=timezone.utc)
SCORE_TIME = "2026-06-10T09:30:00Z"


def _row(
    index: int,
    *,
    trigger: str,
    symbol: str,
    side: str,
    tp: int | None,
    net_r: float | None,
    **overrides,
):
    decision = TRAIN_START + timedelta(hours=6 * index)
    row = {
        "decision_time_utc": decision.isoformat().replace("+00:00", "Z"),
        "symbol": symbol,
        "side": side,
        "trigger_kind": trigger,
        "stop_atr_h1": 1.0 + 0.02 * (index % 5),
        "spread_r": 0.04 + 0.01 * (index % 3),
        "fvg_relative_state": "NEUTRAL",
        "fvg_age_bars": 4 + (index % 6),
        "intended_execution_mode": "MARKET",
        "label_complete": True,
        "label_tp1_hit": tp,
        "label_realized_net_r": net_r,
    }
    row.update(overrides)
    return row


def _observations():
    rows = []
    index = 0
    # Rich cell: PIVOT_RECLAIM|EURUSD|LONG (12 rows, mixed outcomes).
    for _ in range(8):
        rows.append(
            _row(index, trigger="PIVOT_RECLAIM", symbol="EURUSD",
                 side="LONG", tp=1, net_r=0.9)
        )
        index += 1
    for _ in range(4):
        rows.append(
            _row(index, trigger="PIVOT_RECLAIM", symbol="EURUSD",
                 side="LONG", tp=0, net_r=-1.0)
        )
        index += 1
    # Sparse sides under a supported symbol: PIVOT_RECLAIM|GBPUSD.
    for side, tp, net_r in (
        ("LONG", 1, 0.5),
        ("LONG", 0, -1.0),
        ("SHORT", 1, 0.6),
        ("SHORT", 0, -1.0),
    ):
        rows.append(
            _row(index, trigger="PIVOT_RECLAIM", symbol="GBPUSD",
                 side=side, tp=tp, net_r=net_r)
        )
        index += 1
    # Second trigger family: ORDERBLOCK_1H|GBPUSD|SHORT (8 rows).
    for i in range(8):
        rows.append(
            _row(index, trigger="ORDERBLOCK_1H", symbol="GBPUSD",
                 side="SHORT", tp=int(i % 2 == 0),
                 net_r=0.7 if i % 2 == 0 else -1.0)
        )
        index += 1
    return rows


@pytest.fixture(scope="module")
def profile() -> HierarchicalQualityProfile:
    return fit_quality_profile(_observations())


@pytest.fixture(scope="module")
def scorer(profile) -> LiveHierarchicalQualityScorer:
    return LiveHierarchicalQualityScorer(profile)


def _candidate(**overrides):
    base = {
        "side": "LONG",
        "trigger_kind": "PIVOT_RECLAIM",
        "decision_time_utc": SCORE_TIME,
        "stop_atr_h1": 1.0,
        "spread_r": 0.05,
        "fvg_relative_state": "NEUTRAL",
        "fvg_age_bars": 6,
        "intended_execution_mode": "MARKET",
    }
    base.update(overrides)
    return base


def test_build_features_derives_context_from_decision_prices() -> None:
    context = build_quality_features(
        {
            "side": "LONG",
            "trigger_kind": "pivot_reclaim",
            "decision_time_utc": "2026-06-05T09:30:00Z",
            "planned_entry": 1.1000,
            "planned_stop": 1.0950,
            "atr_h1_14": 0.0040,
            "decision_bid": 1.1001,
            "decision_ask": 1.1003,
            "fvg_side": "LONG",
            "fvg_age_bars": 3,
        },
        symbol="eurusd",
    )
    assert context["symbol"] == "EURUSD"
    assert context["trigger_kind"] == "PIVOT_RECLAIM"
    assert context["weekday"] == "FRI"
    assert context["session"] == "LONDON"
    assert context["stop_atr_h1"] == pytest.approx(0.0050 / 0.0040)
    assert context["spread_r"] == pytest.approx(0.0002 / 0.0050)
    assert context["fvg_relative_state"] == "ALIGNED"
    assert context["fvg_age_log"] == pytest.approx(math.log1p(3.0))


def test_build_features_rejects_post_decision_fields() -> None:
    with pytest.raises(ValueError, match="post-decision"):
        build_quality_features(
            _candidate(first_touch_at_utc="2026-06-10T09:31:00Z"),
            symbol="EURUSD",
        )
    with pytest.raises(ValueError, match="post-decision"):
        build_quality_features(
            _candidate(
                trigger_meta={"broker_fill_time_utc": "2026-06-10T09:31:00Z"}
            ),
            symbol="EURUSD",
        )


def test_build_features_rejects_weekday_mismatch() -> None:
    with pytest.raises(ValueError, match="weekday"):
        build_quality_features(
            _candidate(weekday="MONDAY"),
            symbol="EURUSD",
        )


def test_exporter_shaped_rows_fit_directly() -> None:
    rows = [
        _row(
            i,
            trigger="PIVOT_RECLAIM",
            symbol="EURUSD",
            side="LONG",
            tp=i % 2,
            net_r=0.8 if i % 2 else -1.0,
            fvg_relative_state="NONE",
            fvg_age_bars=None,
            weekday_utc="IGNORED_BY_CONTRACT",
        )
        for i in range(6)
    ]
    fitted = fit_quality_profile(rows)
    assert fitted.support_count("global", "*") == 6
    assert fitted.r_label_basis == R_BASIS_NET
    assert fitted.r_cost_status == "ALREADY_NET_NO_REDEDUCTION"


def test_fit_is_deterministic_and_round_trips(profile) -> None:
    refit = fit_quality_profile(_observations())
    assert refit.profile_id == profile.profile_id
    assert refit.profile_sha256 == profile.profile_sha256
    reloaded = HierarchicalQualityProfile.from_mapping(profile.to_mapping())
    assert reloaded.profile_sha256 == profile.profile_sha256
    assert reloaded.feature_names == profile.feature_names
    assert reloaded.support_rows == profile.support_rows


def test_incomplete_and_unlabelled_rows_are_excluded(profile) -> None:
    noisy = _observations() + [
        _row(90, trigger="PIVOT_RECLAIM", symbol="EURUSD", side="LONG",
             tp=1, net_r=5.0, label_complete=False),
        _row(91, trigger="PIVOT_RECLAIM", symbol="EURUSD", side="LONG",
             tp=None, net_r=None),
    ]
    fitted = fit_quality_profile(noisy)
    assert fitted.support_count("global", "*") == profile.support_count(
        "global", "*"
    )
    assert fitted.profile_sha256 == profile.profile_sha256


def test_gross_basis_requires_causal_cost() -> None:
    rows = [
        _row(i, trigger="PIVOT_RECLAIM", symbol="EURUSD", side="LONG",
             tp=1, net_r=None, label_gross_r=1.0)
        for i in range(4)
    ]
    with pytest.raises(ValueError, match="label_causal_cost_r"):
        fit_quality_profile(rows, r_label_basis=R_BASIS_GROSS)
    for row in rows:
        row["label_causal_cost_r"] = 0.1
    fitted = fit_quality_profile(rows, r_label_basis=R_BASIS_GROSS)
    assert fitted.r_label_basis == R_BASIS_GROSS
    assert fitted.r_cost_status == (
        "GROSS_MINUS_EMBEDDED_CAUSAL_COST_NO_REDEDUCTION"
    )
    assert fitted.net_r_model is not None


def test_outcome_fields_cannot_leak_into_fit_features(profile) -> None:
    leaky = copy.deepcopy(_observations())
    for row in leaky:
        row["first_touch_at_utc"] = "2026-05-04T09:31:00Z"
        row["fill_price"] = 1.234
        row["exact_retest_seconds_ignored"] = 17.0
    fitted = fit_quality_profile(leaky)
    assert fitted.profile_sha256 == profile.profile_sha256


def test_scorer_reports_hierarchy_backoff(scorer) -> None:
    rich = scorer.score(_candidate(), symbol="EURUSD")
    assert rich["quality_status"] == "SCORED"
    assert rich["quality_hierarchy_level"] == "trigger_symbol_side"
    assert rich["quality_backoff_reason"] == "NONE"
    assert rich["quality_hierarchy_support"] == 12

    sparse_side = scorer.score(_candidate(), symbol="GBPUSD")
    assert sparse_side["quality_hierarchy_level"] == "trigger_symbol"
    assert sparse_side["quality_backoff_reason"] == (
        "BACKOFF_TO_TRIGGER_SYMBOL"
    )
    assert sparse_side["quality_hierarchy_support"] == 4

    unseen = scorer.score(
        _candidate(trigger_kind="UNSEEN_TRIGGER"),
        symbol="EURUSD",
    )
    assert unseen["quality_hierarchy_level"] == "global"
    assert unseen["quality_backoff_reason"] == "BACKOFF_TO_GLOBAL"
    assert unseen["quality_hierarchy_support"] == 24


def test_sparse_cells_pool_toward_the_coarser_level(scorer) -> None:
    long_score = scorer.score(_candidate(side="LONG"), symbol="GBPUSD")
    short_score = scorer.score(_candidate(side="SHORT"), symbol="GBPUSD")
    assert long_score["quality_hierarchy_level"] == "trigger_symbol"
    assert short_score["quality_hierarchy_level"] == "trigger_symbol"
    assert long_score["quality_expected_net_r"] == pytest.approx(
        short_score["quality_expected_net_r"], abs=1e-12
    )
    assert long_score["quality_tp1_probability"] == pytest.approx(
        short_score["quality_tp1_probability"], abs=1e-12
    )


def test_score_output_contract(scorer) -> None:
    candidate = _candidate(spread_r=None)
    frozen = copy.deepcopy(candidate)
    output = scorer.score(candidate, symbol="EURUSD")
    assert candidate == frozen
    assert all(key.startswith("quality_") for key in output)
    assert output["quality_executing"] is False
    assert output["quality_phase"] == "DECISION"
    assert output["quality_profile_schema"] == QUALITY_PROFILE_SCHEMA
    assert (
        output["quality_tp1_probability_lcb"]
        <= output["quality_tp1_probability"]
        <= output["quality_tp1_probability_ucb"]
    )
    assert output["quality_expected_net_r_std_error"] > 0.0
    assert output["quality_conservative_net_r"] < output[
        "quality_expected_net_r"
    ]
    assert output["quality_feature_missing"] == ["spread_r"]


def test_scorer_rejects_future_dated_profile(scorer) -> None:
    with pytest.raises(ValueError, match="future-dated"):
        scorer.score(
            _candidate(decision_time_utc="2026-05-01T09:30:00Z"),
            symbol="EURUSD",
        )


def test_stop_atr_band_penalty_applies_outside_preferred_band() -> None:
    fitted = fit_quality_profile(
        _observations(),
        outside_stop_atr_penalty_r=0.25,
    )
    banded = LiveHierarchicalQualityScorer(fitted)
    inside = banded.score(_candidate(stop_atr_h1=1.0), symbol="EURUSD")
    outside = banded.score(_candidate(stop_atr_h1=2.2), symbol="EURUSD")
    assert inside["quality_stop_atr_in_preferred_band"] is True
    assert inside["quality_stop_atr_penalty_r"] == 0.0
    assert outside["quality_stop_atr_in_preferred_band"] is False
    assert outside["quality_stop_atr_penalty_r"] == 0.25
    assert outside["quality_conservative_net_r"] == pytest.approx(
        outside["quality_expected_net_r_lcb"] - 0.25
    )


def test_rank_orders_by_conservative_net_r_and_never_mutates() -> None:
    rows = [
        {"opportunity_id": "b", "quality_conservative_net_r": 0.4},
        {"opportunity_id": "a", "quality_conservative_net_r": 0.4},
        {
            "opportunity_id": "c",
            "quality_conservative_net_r": 0.9,
            "quality_correlation_penalty_r": 0.6,
        },
        {"opportunity_id": "d", "quality_status": "UNSCORED:ValueError"},
    ]
    frozen = copy.deepcopy(rows)
    ranked = rank_quality_candidates(rows)
    assert rows == frozen
    assert [row["opportunity_id"] for row in ranked] == ["a", "b", "c", "d"]
    assert [row["quality_rank_position"] for row in ranked] == [1, 2, 3, 4]
    assert [row["quality_selected"] for row in ranked] == [
        True, False, False, False,
    ]
    assert ranked[2]["quality_rank_score_r"] == pytest.approx(0.3)
    assert ranked[3]["quality_rank_score_r"] is None

    with pytest.raises(ValueError, match="penalty"):
        rank_quality_candidates(
            [{"opportunity_id": "x", "quality_correlation_penalty_r": -1.0}]
        )


def test_tp_only_labels_produce_partial_model() -> None:
    rows = [
        _row(i, trigger="PIVOT_RECLAIM", symbol="EURUSD", side="LONG",
             tp=i % 2, net_r=None)
        for i in range(6)
    ]
    fitted = fit_quality_profile(rows)
    assert fitted.net_r_model is None
    partial = LiveHierarchicalQualityScorer(fitted)
    output = partial.score(_candidate(), symbol="EURUSD")
    assert output["quality_status"] == "PARTIAL_MODEL"
    assert output["quality_tp1_probability"] is not None
    assert output["quality_expected_net_r"] is None
    assert output["quality_conservative_net_r"] is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update(schema="wrong"), "schema"),
        (
            lambda p: p["models"]["tp1_probability"]["coefficients"].pop(),
            "dimension",
        ),
        (lambda p: p["support"]["global"].update({"*": 0}), "positive"),
        (lambda p: p["min_support"].pop("trigger"), "integer"),
        (
            lambda p: p["inference"].update(r_label_basis="MID"),
            "r_label_basis",
        ),
    ],
)
def test_profile_validation_rejects_tampering(profile, mutate, message) -> None:
    payload = profile.to_mapping()
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        HierarchicalQualityProfile.from_mapping(payload)
