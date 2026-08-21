from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import pytest

from backtest.orca_random_forest import (
    FoldWindow,
    ORCARandomForestError,
    RandomForestConfig,
    TargetConfig,
    artifact_to_json,
    build_orca_targets,
    compute_artifact_hash,
    fit_orca_random_forest,
    load_orca_artifact,
    predict_orca_frozen,
    resolve_fold_windows,
    validate_orca_artifact,
)


def _small_forest() -> RandomForestConfig:
    return RandomForestConfig(
        n_estimators=9,
        max_depth=4,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=73,
        n_jobs=1,
    )


def _sample(rows: int = 84) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    returns = np.resize(
        np.asarray(
            [
                0.00,
                0.04,
                0.035,
                -0.09,
                0.025,
                0.045,
                -0.08,
                0.035,
                0.025,
                -0.015,
                0.01,
                0.00,
            ],
            dtype=float,
        ),
        rows,
    )
    close = pd.Series(100.0 * np.cumprod(1.0 + returns), index=index, name="SPY")
    rng = np.random.default_rng(20260419)
    features = pd.DataFrame(
        {
            "return_1d": close.pct_change().fillna(0.0),
            "cycle": np.sin(np.arange(rows, dtype=float) / 3.0),
            "causal_noise": rng.normal(size=rows),
        },
        index=index,
    )
    return features, close


def _fit(
    features: pd.DataFrame,
    close: pd.Series,
    *,
    folds: int = 2,
) -> dict:
    return fit_orca_random_forest(
        features,
        close,
        fold_window=FoldWindow(
            mode="rolling",
            train_size=30,
            gap_size=3,
            test_size=12,
            folds=folds,
        ),
        target_config=TargetConfig(horizon=3),
        forest_config=_small_forest(),
    )


def test_target_builder_uses_strict_boundaries_and_known_at():
    index = pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC")
    rally_close = pd.Series([100.0, 103.0, 106.1, 106.0], index=index)
    rally = build_orca_targets(
        rally_close,
        config=TargetConfig(horizon=1, rally_threshold=0.03),
    )
    assert rally["rally"].iloc[0] == 0  # exactly +3%
    assert rally["rally"].iloc[1] == 1  # strictly above +3%
    assert rally["label_known_at"].iloc[0] == index[1]
    assert pd.isna(rally["label_known_at"].iloc[-1])

    crash_close = pd.Series([100.0, 93.0, 100.0, 92.999], index=index)
    crash = build_orca_targets(
        crash_close,
        config=TargetConfig(horizon=1, crash_threshold=-0.07),
    )
    assert crash["crash"].iloc[0] == 0  # exactly -7%
    assert crash["crash"].iloc[2] == 1  # strictly below -7%


def test_fold_window_defaults_match_paper_and_rejects_oos_overlap():
    assert FoldWindow().to_dict() == {
        "mode": "rolling",
        "train_size": 756,
        "gap_size": 10,
        "test_size": 126,
        "folds": 8,
        "step_size": 126,
    }
    with pytest.raises(ORCARandomForestError, match="overlapping OOS"):
        FoldWindow(train_size=20, test_size=5, step_size=4)


def test_resolved_rolling_and_expanding_windows_are_explicit():
    rolling = resolve_fold_windows(
        sample_count=80,
        labelled_count=75,
        window=FoldWindow(
            mode="rolling", train_size=20, gap_size=3, test_size=5, folds=2
        ),
    )
    assert rolling[0].to_dict() == {
        "index": 0,
        "train_start": 0,
        "train_end": 20,
        "gap_start": 20,
        "gap_end": 23,
        "test_start": 23,
        "test_end": 28,
    }
    assert rolling[1].train_start == 5
    assert rolling[0].test_end == rolling[1].test_start

    expanding = resolve_fold_windows(
        sample_count=80,
        labelled_count=75,
        window=FoldWindow(
            mode="expanding", train_size=20, gap_size=3, test_size=5, folds=2
        ),
    )
    assert expanding[0].train_start == expanding[1].train_start == 0
    assert expanding[1].train_end == 25


def test_gap_and_label_maturity_are_recorded_and_purged():
    features, close = _sample(60)
    artifact = fit_orca_random_forest(
        features,
        close,
        fold_window=FoldWindow(train_size=20, gap_size=3, test_size=5, folds=1),
        target_config=TargetConfig(horizon=3),
        forest_config=_small_forest(),
    )
    lineage = artifact["folds"][0]["lineage"]
    assert lineage["train_end"] == lineage["gap_start"] == 20
    assert lineage["gap_end"] == lineage["test_start"] == 23
    assert lineage["train_rows_before_label_purge"] == 20
    assert lineage["train_rows_mature"] == 17
    assert lineage["train_rows_purged_unresolved"] == 3
    assert pd.Timestamp(lineage["train_last_label_known_at"]) < pd.Timestamp(
        lineage["train_end_exclusive_time"]
    )


def test_mutating_future_features_cannot_change_an_earlier_fold():
    features, close = _sample()
    original = _fit(features, close)
    fold_zero_end = original["folds"][0]["lineage"]["test_end"]
    mutated = features.copy()
    mutated.iloc[fold_zero_end:, :] += 10_000.0
    refit = _fit(mutated, close)
    assert original["folds"][0] == refit["folds"][0]


def test_artifact_and_hash_are_deterministic_and_json_round_trips():
    features, close = _sample()
    first = _fit(features, close)
    second = _fit(features, close)
    assert first["artifact_hash"] == second["artifact_hash"]
    assert artifact_to_json(first) == artifact_to_json(second)
    loaded = load_orca_artifact(artifact_to_json(first))
    assert loaded == first
    assert first["calibration"]["status"] == "uncalibrated"


def test_frozen_numpy_predictions_match_sklearn_oos_predictions():
    features, close = _sample()
    artifact = _fit(features, close)
    for fold in artifact["folds"]:
        start = fold["lineage"]["test_start"]
        end = fold["lineage"]["test_end"]
        frozen = predict_orca_frozen(
            artifact,
            features.iloc[start:end],
            fold_index=fold["index"],
        )
        expected_rally = [row["rally_probability"] for row in fold["oos"]]
        expected_crash = [row["crash_probability"] for row in fold["oos"]]
        np.testing.assert_allclose(
            frozen["rally_probability"], expected_rally, rtol=0.0, atol=1e-14
        )
        np.testing.assert_allclose(
            frozen["crash_probability"], expected_crash, rtol=0.0, atol=1e-14
        )


def test_pooled_metrics_and_bcd_formula_are_finite_when_both_classes_exist():
    features, close = _sample()
    metrics = _fit(features, close)["metrics"]
    assert metrics["status"] == "ok"
    assert metrics["rally"]["status"] == "ok"
    assert metrics["crash"]["status"] == "ok"
    assert metrics["bcd_auc"] == pytest.approx(
        math.sqrt(metrics["rally"]["auc"] * metrics["crash"]["auc"])
    )
    for task in ("rally", "crash"):
        for name in ("auc", "average_precision", "brier", "log_loss"):
            assert math.isfinite(metrics[task][name])


def test_metrics_are_explicitly_undefined_when_oos_class_is_missing():
    features, _ = _sample(50)
    close = pd.Series(100.0, index=features.index)
    artifact = fit_orca_random_forest(
        features,
        close,
        fold_window=FoldWindow(train_size=20, gap_size=2, test_size=5, folds=1),
        target_config=TargetConfig(horizon=2),
        forest_config=_small_forest(),
    )
    metrics = artifact["metrics"]
    assert metrics["status"] == "undefined_missing_class"
    assert metrics["bcd_auc"] is None
    for task in ("rally", "crash"):
        assert metrics[task]["status"] == "undefined_missing_class"
        for name in ("auc", "average_precision", "brier", "log_loss"):
            assert metrics[task][name] is None


def test_frozen_prediction_requires_exact_feature_order():
    features, close = _sample()
    artifact = _fit(features, close)
    with pytest.raises(ORCARandomForestError, match="names and order"):
        predict_orca_frozen(artifact, features.iloc[:3, ::-1], fold_index=0)


def test_nonfinite_features_use_train_only_medians_and_remain_predictable():
    features, close = _sample()
    features = features.copy()
    features.iloc[2, 0] = np.nan
    features.iloc[4, 1] = np.inf
    features.iloc[33, 2] = -np.inf
    artifact = _fit(features, close)
    output = predict_orca_frozen(artifact, features.iloc[30:36], fold_index=0)
    assert np.isfinite(output.to_numpy()).all()
    assert ((output >= 0.0) & (output <= 1.0)).all().all()


def test_tamper_and_nonfinite_artifacts_fail_closed():
    features, close = _sample()
    artifact = _fit(features, close)

    tampered = copy.deepcopy(artifact)
    tampered["folds"][0]["rally_forest"]["trees"][0]["threshold"][0] += 0.5
    with pytest.raises(ORCARandomForestError, match="hash mismatch"):
        validate_orca_artifact(tampered)

    registry_tamper = copy.deepcopy(artifact)
    registry_tamper["feature_registry"]["names"][0] = "renamed"
    registry_tamper["artifact_hash"] = compute_artifact_hash(registry_tamper)
    with pytest.raises(ORCARandomForestError, match="feature registry"):
        validate_orca_artifact(registry_tamper)

    nonfinite = copy.deepcopy(artifact)
    nonfinite["folds"][0]["preprocessor"]["center"][0] = float("nan")
    with pytest.raises(ORCARandomForestError, match="canonical JSON"):
        compute_artifact_hash(nonfinite)


def test_fold_resolution_fails_closed_when_full_request_does_not_fit():
    with pytest.raises(ORCARandomForestError, match="insufficient labelled rows"):
        resolve_fold_windows(
            sample_count=30,
            labelled_count=27,
            window=FoldWindow(train_size=20, gap_size=3, test_size=5, folds=2),
        )
