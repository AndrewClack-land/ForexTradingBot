from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math

import pytest

from core.rf_candidate_contract import (
    RF_CANDIDATE_FEATURE_SCHEMA,
    RF_CANDIDATE_PROFILE_SCHEMA,
    LiveRFCandidateScorer,
    RFCandidateProfile,
    build_rf_candidate_features,
    canonical_rf_profile_sha256,
    rank_rf_candidates,
    seal_rf_candidate_profile,
    tree_dispersion_penalized_score,
)
from core.rf_shadow_bridge import RFShadowBridge


UTC = timezone.utc
DECISION = datetime(2026, 1, 3, 12, 30, tzinfo=UTC)
EXPECTED_CONTRACT = {
    "expected_strategy_version": "single-tp-v1",
    "expected_factor_contract": "v1-fvg-margin",
    "expected_target_contract": "single-tp-1.2r",
}


def _leaf(value: float) -> dict:
    return {
        "feature_index": [-1],
        "threshold": [0.0],
        "left_child": [-1],
        "right_child": [-1],
        "value": [value],
    }


def _split(
    *,
    feature: int,
    threshold: float,
    left: float,
    right: float,
) -> dict:
    return {
        "feature_index": [feature, -1, -1],
        "threshold": [threshold, 0.0, 0.0],
        "left_child": [1, -1, -1],
        "right_child": [2, -1, -1],
        "value": [0.0, left, right],
    }


def _profile_mapping(
    *,
    registry: tuple[str, ...] = ("score_long",),
    fill_trees: tuple[dict, ...] | None = None,
    tp_trees: tuple[dict, ...] | None = None,
    gross_trees: tuple[dict, ...] | None = None,
    created: str = "2026-01-02T00:00:00Z",
    trained: str = "2026-01-01T00:00:00Z",
    uncertainty_z: float = 1.0,
) -> dict:
    raw = {
        "schema": RF_CANDIDATE_PROFILE_SCHEMA,
        "created_at_utc": created,
        "trained_through_utc": trained,
        "strategy_contract": {
            "strategy_version": "single-tp-v1",
            "factor_contract": "v1-fvg-margin",
            "target_contract": "single-tp-1.2r",
        },
        "feature_registry": list(registry),
        "uncertainty_z": uncertainty_z,
        "forests": {
            "fill_probability": {
                "trees": list(
                    fill_trees
                    or (
                        _split(
                            feature=0,
                            threshold=1.5,
                            left=0.2,
                            right=0.8,
                        ),
                        _leaf(0.6),
                    )
                )
            },
            "tp_given_fill_probability": {
                "trees": list(tp_trees or (_leaf(0.5), _leaf(0.7)))
            },
            "gross_r_given_fill": {
                "trees": list(gross_trees or (_leaf(1.0), _leaf(-0.2)))
            },
        },
    }
    return seal_rf_candidate_profile(raw)


def _profile(**kwargs) -> RFCandidateProfile:
    return RFCandidateProfile.from_mapping(_profile_mapping(**kwargs))


def _scorer(profile: RFCandidateProfile | None = None) -> LiveRFCandidateScorer:
    return LiveRFCandidateScorer(profile or _profile(), **EXPECTED_CONTRACT)


def _factor_vector(score_long: float = 2.0) -> dict:
    return {
        "schema": "narrative-factor-vector/v2",
        "bias": "LONG",
        "score_long": score_long,
        "score_short": 1.0,
        "score_delta_long_minus_short": score_long - 1.0,
        "margin_long": 2.0,
        "margin_short": 2.0,
        "fvg_side": "LONG",
        "factors": [
            {
                "key": "h1_premium_discount",
                "present": True,
                "vote_side": "LONG",
            },
            {
                "key": "false_breakout_4h",
                "present": True,
                "vote_side": "SHORT",
            },
        ],
    }


def _candidate(**updates) -> dict:
    candidate = {
        "opportunity_id": "opp-a",
        "symbol": "EURUSD",
        "side": "LONG",
        "trigger_kind": "h1_pivot_reclaim_15m",
        "production_priority": 4,
        "decision_time_utc": DECISION,
        "entry_price": 1.1000,
        "entry_min": 1.0998,
        "entry_max": 1.1002,
        "stop_price": 1.0980,
        "tp_prices": [1.1024],
        "atr_h1_14": 0.0020,
        "atr_m15_14": 0.0010,
        "decision_bid": 1.1003,
        "decision_ask": 1.1005,
        "fvg_regime": "LONG",
        "fvg_age_bars": 3,
        "vol_R": 0.8,
        "vol_em_1d": 0.008,
        "vol_tp1_em_ratio": 0.3,
        "intended_execution_mode": "exact-retest-limit",
        "factor_vector": _factor_vector(),
    }
    candidate.update(updates)
    return candidate


def test_manual_tree_math_and_only_rf_annotations() -> None:
    scorer = _scorer()
    result = scorer.score(_candidate(), causal_cost_r=0.1)

    assert set(result) and all(key.startswith("rf_") for key in result)
    assert result["rf_executing"] is False
    assert result["rf_fill_probability"] == pytest.approx(0.7)
    assert result["rf_tp_given_fill_probability"] == pytest.approx(0.6)
    assert result["rf_tp_probability"] == pytest.approx(0.42)
    assert result["rf_expected_gross_r_given_fill"] == pytest.approx(0.4)
    assert result["rf_expected_net_r"] == pytest.approx(0.21)

    expected_variance = 0.5 * 0.45 - 0.21**2
    assert result["rf_tree_dispersion_r"] == pytest.approx(math.sqrt(expected_variance))
    assert result["rf_conservative_intrinsic_score_r"] == pytest.approx(
        0.21 - math.sqrt(expected_variance)
    )


def test_profile_hash_is_canonical_and_tampering_is_rejected() -> None:
    mapping = _profile_mapping()
    reordered = {key: mapping[key] for key in reversed(tuple(mapping))}
    assert canonical_rf_profile_sha256(mapping) == canonical_rf_profile_sha256(
        reordered
    )
    assert RFCandidateProfile.from_mapping(mapping).to_mapping() == mapping

    tampered = deepcopy(mapping)
    tampered["forests"]["gross_r_given_fill"]["trees"][0]["value"][0] = 99.0
    with pytest.raises(ValueError, match="hash mismatch"):
        RFCandidateProfile.from_mapping(tampered)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("expected_strategy_version", "other-strategy"),
        ("expected_factor_contract", "other-factors"),
        ("expected_target_contract", "other-target"),
    ],
)
def test_scorer_fails_at_construction_on_contract_mismatch(
    field: str,
    bad_value: str,
) -> None:
    expected = dict(EXPECTED_CONTRACT)
    expected[field] = bad_value
    with pytest.raises(ValueError, match="contract mismatch"):
        LiveRFCandidateScorer(_profile(), **expected)


def test_load_requires_and_enforces_exact_contract_before_prediction(tmp_path) -> None:
    path = tmp_path / "rf.json"
    path.write_text(
        json.dumps(_profile_mapping()),
        encoding="utf-8",
    )
    assert LiveRFCandidateScorer.load(path, **EXPECTED_CONTRACT) is not None
    mismatch = dict(EXPECTED_CONTRACT, expected_factor_contract="wrong")
    with pytest.raises(ValueError, match="contract mismatch"):
        LiveRFCandidateScorer.load(path, **mismatch)

    bridge = RFShadowBridge.load(
        path,
        cost_profile=None,
        **EXPECTED_CONTRACT,
    )
    assert bridge.model_id == _profile().profile_sha256[:24]
    with pytest.raises(ValueError, match="contract mismatch"):
        RFShadowBridge.load(
            path,
            cost_profile=None,
            **mismatch,
        )


def test_strict_profile_rejects_unknown_feature_and_bad_tree() -> None:
    unknown = _profile_mapping(registry=("future_magic",))
    with pytest.raises(ValueError, match="unsupported feature"):
        RFCandidateProfile.from_mapping(unknown)

    bad = _profile_mapping()
    bad["forests"]["fill_probability"]["trees"][0]["feature_index"][0] = 9
    bad = seal_rf_candidate_profile(bad)
    with pytest.raises(ValueError, match="invalid feature index"):
        RFCandidateProfile.from_mapping(bad)


@pytest.mark.parametrize(
    "created,trained,message",
    [
        (
            "2026-01-04T00:00:00Z",
            "2026-01-01T00:00:00Z",
            "not created",
        ),
        (
            "2026-01-05T00:00:00Z",
            "2026-01-04T00:00:00Z",
            "not created",
        ),
    ],
)
def test_future_model_is_rejected(created: str, trained: str, message: str) -> None:
    scorer = _scorer(_profile(created=created, trained=trained))
    with pytest.raises(ValueError, match=message):
        scorer.score(_candidate(), causal_cost_r=0.0)


@pytest.mark.parametrize(
    "forbidden",
    [
        {"actual_fill_price": 1.1},
        {"nested": {"first_touch_time": "2026-01-03T12:31:00Z"}},
        {"nested": [{"exact_retest_seconds": 60}]},
        {"outcome": {"tp1_hit": True}},
        {"exit_time": "2026-01-03T13:00:00Z"},
    ],
)
def test_postdecision_fields_are_rejected_recursively(forbidden: dict) -> None:
    candidate = _candidate(**forbidden)
    with pytest.raises(ValueError, match="post-decision"):
        build_rf_candidate_features(candidate, feature_registry=("score_long",))


def test_missing_and_unknown_categories_are_explicit() -> None:
    registry = (
        "symbol:EURUSD",
        "symbol:__OTHER__",
        "spread_r",
        "spread_r_missing",
        "orca_snapshot_available",
        "orca_absorption_ratio",
        "orca_absorption_ratio_missing",
    )
    candidate = _candidate(
        symbol="XAUUSD",
        decision_bid=None,
        decision_ask=None,
    )
    features = build_rf_candidate_features(candidate, feature_registry=registry)

    assert features["schema"] == RF_CANDIDATE_FEATURE_SCHEMA
    assert features["values"]["symbol:EURUSD"] == 0.0
    assert features["values"]["symbol:__OTHER__"] == 1.0
    assert features["values"]["spread_r_missing"] == 1.0
    assert features["values"]["orca_snapshot_available"] == 0.0
    assert "symbol:XAUUSD" in features["oov"]
    assert "spread_r" in features["missing"]
    assert "orca_absorption_ratio" in features["missing"]


def test_session_is_explicit_and_never_invented_from_utc() -> None:
    registry = (
        "session:OVERLAP",
        "session:NY_ONLY",
        "session:MISSING",
    )
    missing = build_rf_candidate_features(_candidate(), feature_registry=registry)
    assert missing["values"]["session:MISSING"] == 1.0
    assert missing["values"]["session:OVERLAP"] == 0.0

    explicit = build_rf_candidate_features(
        _candidate(),
        feature_registry=registry,
        session_label="NY_ONLY",
    )
    assert explicit["values"]["session:NY_ONLY"] == 1.0
    assert explicit["values"]["session:MISSING"] == 0.0
    with pytest.raises(ValueError, match="explicit session_label"):
        build_rf_candidate_features(
            _candidate(),
            feature_registry=registry,
            session_label="MADE_UP_DST_SESSION",
        )


def test_optional_orca_fields_require_point_in_time_provenance() -> None:
    registry = (
        "orca_snapshot_available",
        "orca_absorption_ratio",
        "orca_absorption_ratio_missing",
        "orca_mode:RISK_OFF",
    )
    snapshot = {
        "known_at_utc": "2026-01-03T12:29:00Z",
        "trained_through_utc": "2026-01-03T12:00:00Z",
        "market_mode": "RISK_OFF",
        "absorption_ratio": 0.72,
    }
    features = build_rf_candidate_features(
        _candidate(), feature_registry=registry, orca_snapshot=snapshot
    )
    assert features["values"]["orca_snapshot_available"] == 1.0
    assert features["values"]["orca_absorption_ratio"] == pytest.approx(0.72)
    assert features["values"]["orca_mode:RISK_OFF"] == 1.0

    future = dict(snapshot, known_at_utc="2026-01-03T12:31:00Z")
    with pytest.raises(ValueError, match="future-dated"):
        build_rf_candidate_features(
            _candidate(), feature_registry=registry, orca_snapshot=future
        )


def test_bcd_auc_is_diagnostic_and_never_aliases_crisis_probability() -> None:
    registry = (
        "orca_crisis_probability",
        "orca_crisis_probability_missing",
        "orca_bcd_auc",
        "orca_bcd_auc_missing",
    )
    lineage = {
        "known_at_utc": "2026-01-03T12:29:00Z",
        "trained_through_utc": "2026-01-03T12:00:00Z",
    }
    ambiguous = build_rf_candidate_features(
        _candidate(),
        feature_registry=registry,
        orca_snapshot={**lineage, "bcd_score": 0.93},
    )
    assert ambiguous["values"]["orca_crisis_probability_missing"] == 1.0
    assert ambiguous["values"]["orca_bcd_auc_missing"] == 1.0

    explicit = build_rf_candidate_features(
        _candidate(),
        feature_registry=registry,
        orca_snapshot={
            **lineage,
            "crisis_probability": 0.17,
            "bcd_auc": 0.74,
        },
    )
    assert explicit["values"]["orca_crisis_probability"] == pytest.approx(0.17)
    assert explicit["values"]["orca_bcd_auc"] == pytest.approx(0.74)
    with pytest.raises(ValueError, match="at most 1"):
        build_rf_candidate_features(
            _candidate(),
            feature_registry=registry,
            orca_snapshot={**lineage, "bcd_auc": 1.01},
        )


def test_cost_is_supplied_once_and_subtracted_once() -> None:
    scorer = _scorer()
    first = scorer.score(_candidate(), causal_cost_r=0.1)
    second = scorer.score(_candidate(), causal_cost_r=0.2)

    assert first["rf_causal_cost_r"] == 0.1
    assert second["rf_causal_cost_r"] == 0.2
    assert first["rf_expected_net_r"] - second["rf_expected_net_r"] == (
        pytest.approx(first["rf_fill_probability"] * 0.1)
    )
    with pytest.raises(ValueError, match="nonnegative"):
        scorer.score(_candidate(), causal_cost_r=-0.01)


def test_identical_trees_have_zero_dispersion() -> None:
    scorer = _scorer(
        _profile(
            fill_trees=(_leaf(0.5), _leaf(0.5)),
            tp_trees=(_leaf(0.4), _leaf(0.4)),
            gross_trees=(_leaf(1.0), _leaf(1.0)),
        )
    )
    result = scorer.score(_candidate(), causal_cost_r=0.2)
    assert result["rf_fill_tree_std"] == pytest.approx(0.0)
    assert result["rf_gross_r_tree_std"] == pytest.approx(0.0)
    assert result["rf_tree_dispersion_r"] == pytest.approx(0.0)
    assert result["rf_conservative_intrinsic_score_r"] == pytest.approx(0.4)


def test_shared_tree_dispersion_formula_matches_live_output() -> None:
    scorer = _scorer()
    result = scorer.score(_candidate(), causal_cost_r=0.1)
    shared = tree_dispersion_penalized_score(
        (0.8, 0.6),
        (1.0, -0.2),
        causal_cost_r=0.1,
        penalty_multiplier=1.0,
    )
    assert result["rf_expected_net_r"] == pytest.approx(shared["expected_net_r"])
    assert result["rf_tree_dispersion_r"] == pytest.approx(shared["tree_dispersion_r"])
    assert result["rf_tree_dispersion_penalty_r"] == pytest.approx(
        shared["tree_dispersion_penalty_r"]
    )
    assert result["rf_conservative_intrinsic_score_r"] == pytest.approx(
        shared["dispersion_penalized_score_r"]
    )


def test_rank_is_permutation_stable_and_never_mutates_candidates() -> None:
    first = {
        "opportunity_id": "b",
        "side": "SHORT",
        "entry_price": 1.2,
        "rf_conservative_intrinsic_score_r": 0.1,
        "rf_tp_probability": 0.7,
        "rf_expected_net_r": 0.2,
    }
    second = {
        "opportunity_id": "a",
        "side": "LONG",
        "entry_price": 1.1,
        "rf_conservative_intrinsic_score_r": 0.3,
        "rf_tp_probability": 0.5,
        "rf_expected_net_r": 0.4,
    }
    originals = deepcopy([first, second])
    forward = rank_rf_candidates([first, second])
    reverse = rank_rf_candidates([second, first])

    assert [row["opportunity_id"] for row in forward] == ["a", "b"]
    assert [row["opportunity_id"] for row in reverse] == ["a", "b"]
    assert forward[0]["rf_selected"] is True
    assert forward[0]["rf_rank_position"] == 1
    assert [first, second] == originals


@pytest.mark.parametrize("best_score", [0.0, -0.01])
def test_rank_abstains_when_best_dispersion_penalized_score_is_not_positive(
    best_score: float,
) -> None:
    ranked = rank_rf_candidates(
        [
            {
                "candidate_id": "best",
                "rf_conservative_intrinsic_score_r": best_score,
            },
            {
                "candidate_id": "worse",
                "rf_conservative_intrinsic_score_r": -1.0,
            },
        ]
    )
    assert not any(row["rf_selected"] for row in ranked)
    assert ranked[0]["rf_selection_status"] == "ABSTAINED_NONPOSITIVE_SCORE"


def test_malicious_scorer_cannot_overwrite_execution_fields() -> None:
    class MaliciousScorer:
        def score(self, candidate, **kwargs):
            return {"side": "SHORT", "entry_price": 9.9}

    candidate = _candidate()
    original = deepcopy(candidate)
    with pytest.raises(ValueError, match="non-rf"):
        rank_rf_candidates(
            [candidate],
            scorer=MaliciousScorer(),  # type: ignore[arg-type]
            causal_costs_r=[0.0],
        )
    assert candidate == original


def test_rank_with_real_scorer_copies_signal_geometry() -> None:
    scorer = _scorer(_profile(uncertainty_z=0.0))
    candidate = _candidate()
    original = deepcopy(candidate)
    ranked = rank_rf_candidates([candidate], scorer=scorer, causal_costs_r=[0.1])
    assert candidate == original
    assert ranked[0]["side"] == original["side"]
    assert ranked[0]["entry_price"] == original["entry_price"]
    assert ranked[0]["stop_price"] == original["stop_price"]
    assert ranked[0]["tp_prices"] == original["tp_prices"]
    assert ranked[0]["rf_selected"] is True
