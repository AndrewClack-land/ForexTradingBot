from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math

import pytest

from backtest.rf_candidate_optimizer import (
    CandidateForestConfig,
    CandidateRFTrainingError,
    CandidateTargetConfig,
    CandidateWalkForwardConfig,
    SINGLE_TP_TARGET_CONTRACT,
    fit_rf_candidate_profile,
)
from core.rf_candidate_contract import RFCandidateProfile


UTC = timezone.utc


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _rows(days: int = 34) -> list[dict[str, object]]:
    start = datetime(2026, 1, 1, 16, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for day in range(days):
        decision = start + timedelta(days=day)
        event_id = f"event-{day:03d}"
        for candidate in range(2):
            filled = (day + candidate) % 3 != 0
            tp_hit = filled and (day + candidate) % 2 == 0
            trigger = "UNSEEN" if day == 11 else "BASE"
            row: dict[str, object] = {
                "outcome_status": "RESOLVED",
                "decision_event_id": event_id,
                "candidate_id": f"{event_id}-candidate-{candidate}",
                "decision_time_utc": _iso(decision),
                "features_known_at_utc": _iso(decision),
                "label_known_at_utc": _iso(decision + timedelta(hours=10)),
                "target_contract": SINGLE_TP_TARGET_CONTRACT,
                "tp_count": 1,
                "tp_reward_r": 1.2,
                "filled": filled,
                "causal_cost_r": 0.04 + 0.01 * candidate,
                "sequence_index": day,
                "feature_values": {
                    "score_for_side": float((day % 7) - 3),
                    "score_against_side": float((day + candidate) % 4),
                    "side_sign": 1.0 if candidate == 0 else -1.0,
                    f"trigger:{trigger}": 1.0,
                    f"symbol:{'EURUSD' if candidate == 0 else 'GBPUSD'}": 1.0,
                },
            }
            if filled:
                row.update(
                    {
                        "exit_time_utc": _iso(decision + timedelta(hours=8)),
                        "tp_hit": tp_hit,
                        "gross_r": 1.2 if tp_hit else -1.0,
                    }
                )
            else:
                row.update(
                    {
                        "exit_time_utc": None,
                        "tp_hit": None,
                        "gross_r": None,
                    }
                )
            rows.append(row)
    return rows


def _walk_forward() -> CandidateWalkForwardConfig:
    return CandidateWalkForwardConfig(
        entry_ttl=timedelta(hours=12),
        max_hold=timedelta(hours=12),
        purge=timedelta(days=1),
        embargo=timedelta(days=1),
        train_span=timedelta(days=10),
        test_span=timedelta(days=4),
        step_span=timedelta(days=4),
        min_train_events=6,
        min_test_events=2,
    )


def _forest() -> CandidateForestConfig:
    return CandidateForestConfig(
        n_estimators=7,
        max_depth=3,
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=91,
    )


def _fit(rows: list[dict[str, object]] | None = None):
    return fit_rf_candidate_profile(
        rows or _rows(),
        walk_forward=_walk_forward(),
        forest=_forest(),
        target=CandidateTargetConfig(tp_reward_r=1.2),
        strategy_contract={
            "strategy_version": "single-tp-production-a9ee196",
            "factor_contract": "factor-vector-v1",
        },
        uncertainty_z=1.25,
    )


def test_profile_is_portable_sealed_deterministic_and_never_promoted() -> None:
    first = _fit()
    second = _fit()

    assert first.profile == second.profile
    parsed = RFCandidateProfile.from_mapping(first.profile)
    assert parsed.profile_sha256 == first.profile["profile_sha256"]
    assert parsed.strategy_contract["target_contract"] == (SINGLE_TP_TARGET_CONTRACT)
    definition = parsed.strategy_contract["target_definition"]
    assert definition["tp_count"] == 1
    assert definition["tp_reward_r"] == pytest.approx(1.2)
    assert parsed.strategy_contract["promotion_policy"] == (
        "manual-shadow-challenger-only"
    )
    assert first.auto_promoted is False
    assert first.to_mapping()["auto_promotion"]["enabled"] is False


def test_fold_registry_is_train_only_and_unseen_category_maps_to_other() -> None:
    result = _fit()
    first = result.folds[0]

    assert "trigger:BASE" in first["feature_registry"]
    assert "trigger:__OTHER__" in first["feature_registry"]
    assert "trigger:UNSEEN" not in first["feature_registry"]
    assert first["preprocessing"] == {
        "fit_scope": "fold-train-only",
        "numeric": "contract-zero-with-explicit-missing-indicators",
        "categorical": "train-only-one-hot-with-__OTHER__",
        "scaling": "none",
    }
    assert "trigger:UNSEEN" in result.profile["feature_registry"]


def test_walk_forward_cutoffs_cover_horizon_and_keep_oos_non_overlapping() -> None:
    result = _fit()
    previous_end = None
    for fold in result.folds:
        decision_cutoff = datetime.fromisoformat(
            fold["train_decision_cutoff_utc"].replace("Z", "+00:00")
        )
        label_cutoff = datetime.fromisoformat(
            fold["train_label_cutoff_utc"].replace("Z", "+00:00")
        )
        test_start = datetime.fromisoformat(
            fold["test_start_utc"].replace("Z", "+00:00")
        )
        test_end = datetime.fromisoformat(fold["test_end_utc"].replace("Z", "+00:00"))
        assert test_start - decision_cutoff >= timedelta(days=1)
        assert test_start - label_cutoff >= timedelta(days=1)
        if previous_end is not None:
            assert test_start >= previous_end
        previous_end = test_end


def test_each_head_gives_each_eligible_event_total_weight_one() -> None:
    result = _fit()

    for fold in result.folds:
        audits = fold["sample_weight_audit"]
        for head in (
            "fill_head",
            "tp_given_fill_head",
            "gross_r_given_fill_head",
        ):
            audit = audits[head]
            assert audit["max_event_weight_error"] <= 1e-12
            assert audit["total_weight"] == pytest.approx(audit["event_count"])


def test_oos_metrics_include_cost_aware_net_r_auc_brier_and_drawdown() -> None:
    result = _fit()
    metrics = result.aggregate_metrics

    assert 0.0 <= metrics["fill_auc"] <= 1.0
    assert 0.0 <= metrics["fill_brier"] <= 1.0
    assert 0.0 <= metrics["tp_given_fill_auc"] <= 1.0
    assert 0.0 <= metrics["tp_given_fill_brier"] <= 1.0
    assert math.isfinite(metrics["expected_net_r_sum"])
    assert math.isfinite(metrics["realized_net_r_sum"])
    assert metrics["cost_application"] == "filled_once"
    assert metrics["selected_max_drawdown_r"] is not None
    assert metrics["selected_max_drawdown_r"] >= 0.0
    assert metrics["selection_policy"] == (
        "max_tree_dispersion_penalized_score_abstain_le_zero"
    )
    assert metrics["baseline_policy"] == "max_raw_expected_net_no_abstain"
    assert metrics["baseline_selected_event_count"] == metrics["decision_event_count"]
    assert (
        metrics["selected_event_count"] + metrics["no_selection_event_count"]
        == metrics["decision_event_count"]
    )
    for row in result.oos_predictions:
        expected = row["fill_probability"] * (
            row["expected_gross_r_given_fill"] - row["causal_cost_r"]
        )
        assert row["expected_net_r"] == pytest.approx(expected)
        assert row["uncertainty_z"] == pytest.approx(1.25)
        assert row["tree_dispersion_penalty_multiplier"] == pytest.approx(1.25)
        assert row["tree_dispersion_penalty_r"] == pytest.approx(
            1.25 * row["tree_dispersion_r"]
        )
        assert row["conservative_intrinsic_score_r"] == pytest.approx(
            row["expected_net_r"] - row["tree_dispersion_penalty_r"]
        )
        assert isinstance(row["policy_selected"], bool)
        if not row["filled"]:
            assert row["realized_net_r"] == 0.0


def test_oos_policy_matches_live_dispersion_order_and_positive_threshold() -> None:
    result = _fit()
    grouped: dict[str, list[dict[str, object]]] = {}
    for source in result.oos_predictions:
        row = dict(source)
        grouped.setdefault(str(row["decision_event_id"]), []).append(row)

    for candidates in grouped.values():
        ordered = sorted(
            candidates,
            key=lambda row: (
                -float(row["conservative_intrinsic_score_r"]),
                -float(row["tp_probability"]),
                -float(row["expected_net_r"]),
                str(row["candidate_id"]),
            ),
        )
        should_select = float(ordered[0]["conservative_intrinsic_score_r"]) > 0.0
        selected = [row for row in candidates if row["policy_selected"]]
        assert len(selected) == int(should_select)
        if should_select:
            assert selected[0]["candidate_id"] == ordered[0]["candidate_id"]
            assert selected[0]["policy_rank"] == 1
        else:
            assert all(
                row["policy_event_status"] == "ABSTAINED_NONPOSITIVE_SCORE"
                for row in candidates
            )


def test_all_negative_oos_scores_abstain_but_keep_baseline_metrics() -> None:
    rows = _rows()
    for row in rows:
        row["causal_cost_r"] = 10.0
    result = _fit(rows)
    metrics = result.aggregate_metrics

    assert metrics["selected_event_count"] == 0
    assert metrics["no_selection_event_count"] == metrics["decision_event_count"]
    assert metrics["selection_rate"] == pytest.approx(0.0)
    assert metrics["selected_expected_net_r_sum"] == pytest.approx(0.0)
    assert metrics["selected_realized_net_r_sum"] == pytest.approx(0.0)
    assert metrics["selected_max_drawdown_r"] is None
    assert metrics["baseline_selected_event_count"] == metrics["decision_event_count"]
    assert metrics["baseline_selected_expected_net_r_sum"] < 0.0
    assert all(not row["policy_selected"] for row in result.oos_predictions)


def test_censored_and_invalid_rows_are_excluded_not_labelled() -> None:
    rows = _rows()
    decision = datetime(2025, 12, 20, 16, tzinfo=UTC)
    rows.extend(
        [
            {
                "outcome_status": "CENSORED",
                "decision_time_utc": _iso(decision),
                "features_known_at_utc": _iso(decision),
            },
            {
                "outcome_status": "INVALID",
                "decision_time_utc": _iso(decision),
                "features_known_at_utc": _iso(decision),
            },
        ]
    )

    result = _fit(rows)

    assert result.excluded_counts == {"censored": 1, "invalid": 1}


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("tp_count", 3, "single-TP"),
        ("target_contract", "legacy-three-tp", "target_contract"),
    ],
)
def test_non_single_tp_labels_are_rejected(
    field: str, value: object, match: str
) -> None:
    rows = _rows()
    rows[0][field] = value

    with pytest.raises(CandidateRFTrainingError, match=match):
        _fit(rows)


def test_future_dated_features_are_rejected() -> None:
    rows = _rows()
    decision = datetime.fromisoformat(
        str(rows[0]["decision_time_utc"]).replace("Z", "+00:00")
    )
    rows[0]["features_known_at_utc"] = _iso(decision + timedelta(seconds=1))

    with pytest.raises(CandidateRFTrainingError, match="not known"):
        _fit(rows)


def test_exit_after_label_known_is_rejected() -> None:
    rows = _rows()
    filled_index = next(index for index, row in enumerate(rows) if row["filled"])
    label_known = datetime.fromisoformat(
        str(rows[filled_index]["label_known_at_utc"]).replace("Z", "+00:00")
    )
    rows[filled_index]["exit_time_utc"] = _iso(label_known + timedelta(seconds=1))

    with pytest.raises(CandidateRFTrainingError, match="exit_time"):
        _fit(rows)


def test_non_fill_cannot_carry_post_fill_labels() -> None:
    rows = _rows()
    index = next(index for index, row in enumerate(rows) if not row["filled"])
    rows[index]["gross_r"] = 0.0

    with pytest.raises(CandidateRFTrainingError, match="non-fill"):
        _fit(rows)


@pytest.mark.parametrize("boundary", ["purge", "embargo"])
def test_purge_and_embargo_each_cover_ttl_plus_max_hold(boundary: str) -> None:
    kwargs = {
        "entry_ttl": timedelta(hours=12),
        "max_hold": timedelta(hours=12),
        "purge": timedelta(days=1),
        "embargo": timedelta(days=1),
    }
    kwargs[boundary] = timedelta(hours=23)

    with pytest.raises(CandidateRFTrainingError, match=boundary):
        CandidateWalkForwardConfig(**kwargs)


def test_same_event_has_one_decision_time_and_sequence() -> None:
    rows = _rows()
    rows[1]["sequence_index"] = 999

    with pytest.raises(CandidateRFTrainingError, match="inconsistent"):
        _fit(rows)


def test_created_at_cannot_precede_latest_known_label() -> None:
    rows = _rows()

    with pytest.raises(CandidateRFTrainingError, match="newest training label"):
        fit_rf_candidate_profile(
            rows,
            walk_forward=_walk_forward(),
            forest=_forest(),
            strategy_contract={
                "strategy_version": "test",
                "factor_contract": "test",
            },
            created_at_utc="2026-01-02T00:00:00Z",
        )


def test_nested_causal_feature_record_is_supported() -> None:
    rows = deepcopy(_rows())
    for row in rows:
        values = row.pop("feature_values")
        known_at = row.pop("features_known_at_utc")
        row["features"] = {
            "decision_time_utc": known_at,
            "values": values,
        }

    result = _fit(rows)

    assert result.profile["profile_sha256"]
