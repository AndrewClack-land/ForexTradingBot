"""Leakage-resistant ORCA-style Random Forest research model.

The module deliberately starts from an already prepared *causal* feature
frame.  It does not fetch market data, construct spectral features, or touch
the live signal path.  scikit-learn is imported only inside
``fit_orca_random_forest``; frozen artifacts are scored with NumPy alone.

The Random Forest probabilities are explicitly uncalibrated.  Calibration is
not inferred merely because ``predict_proba`` is available.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd


ORCA_RANDOM_FOREST_SCHEMA = "orca-random-forest-v1"
ORCA_FEATURE_REGISTRY_SCHEMA = "orca-causal-feature-registry-v1"


class ORCARandomForestError(ValueError):
    """Raised when an ORCA research artifact cannot be produced safely."""


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ORCARandomForestError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ORCARandomForestError(f"{name} must be a positive integer") from exc
    if parsed != value or parsed <= 0:
        raise ORCARandomForestError(f"{name} must be a positive integer")
    return parsed


def _non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ORCARandomForestError(f"{name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ORCARandomForestError(f"{name} must be a non-negative integer") from exc
    if parsed != value or parsed < 0:
        raise ORCARandomForestError(f"{name} must be a non-negative integer")
    return parsed


@dataclass(frozen=True)
class FoldWindow:
    """Row-count walk-forward configuration.

    The paper-compatible default is an explicit rolling window with 756
    training rows, a 10-row embargo, 126 OOS rows, and eight folds.  ``step``
    may be increased, but values smaller than ``test_size`` are rejected so
    OOS windows can never overlap silently.
    """

    mode: Literal["rolling", "expanding"] = "rolling"
    train_size: int = 756
    gap_size: int = 10
    test_size: int = 126
    folds: int = 8
    step_size: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"rolling", "expanding"}:
            raise ORCARandomForestError("mode must be 'rolling' or 'expanding'")
        _positive_int(self.train_size, name="train_size")
        _non_negative_int(self.gap_size, name="gap_size")
        _positive_int(self.test_size, name="test_size")
        _positive_int(self.folds, name="folds")
        if self.step_size is not None:
            step = _positive_int(self.step_size, name="step_size")
            if step < self.test_size:
                raise ORCARandomForestError(
                    "step_size must be at least test_size; overlapping OOS "
                    "windows are forbidden"
                )

    @property
    def step(self) -> int:
        return self.test_size if self.step_size is None else self.step_size

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "train_size": self.train_size,
            "gap_size": self.gap_size,
            "test_size": self.test_size,
            "folds": self.folds,
            "step_size": self.step,
        }


@dataclass(frozen=True)
class TargetConfig:
    horizon: int = 10
    rally_threshold: float = 0.03
    crash_threshold: float = -0.07

    def __post_init__(self) -> None:
        _positive_int(self.horizon, name="horizon")
        if not math.isfinite(float(self.rally_threshold)):
            raise ORCARandomForestError("rally_threshold must be finite")
        if not math.isfinite(float(self.crash_threshold)):
            raise ORCARandomForestError("crash_threshold must be finite")
        if self.rally_threshold <= 0.0:
            raise ORCARandomForestError("rally_threshold must be positive")
        if not -1.0 < self.crash_threshold < 0.0:
            raise ORCARandomForestError(
                "crash_threshold must be strictly between -1 and 0"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "rally_threshold": float(self.rally_threshold),
            "crash_threshold": float(self.crash_threshold),
            "rally_definition": "endpoint_return_strictly_greater",
            "crash_definition": "minimum_forward_path_return_strictly_less",
            "label_known_at": "t_plus_horizon",
        }


@dataclass(frozen=True)
class RandomForestConfig:
    n_estimators: int = 200
    max_depth: int | None = 6
    min_samples_leaf: int = 30
    min_samples_split: int = 60
    max_features: str | int | float | None = "sqrt"
    class_weight: str | None = "balanced_subsample"
    random_state: int = 20260419
    n_jobs: int = 1

    def __post_init__(self) -> None:
        _positive_int(self.n_estimators, name="n_estimators")
        if self.max_depth is not None:
            _positive_int(self.max_depth, name="max_depth")
        _positive_int(self.min_samples_leaf, name="min_samples_leaf")
        if _positive_int(self.min_samples_split, name="min_samples_split") < 2:
            raise ORCARandomForestError("min_samples_split must be at least 2")
        if isinstance(self.max_features, str):
            if self.max_features not in {"sqrt", "log2"}:
                raise ORCARandomForestError(
                    "max_features string must be 'sqrt' or 'log2'"
                )
        elif self.max_features is not None:
            if isinstance(self.max_features, bool):
                raise ORCARandomForestError("max_features cannot be boolean")
            if isinstance(self.max_features, int):
                _positive_int(self.max_features, name="max_features")
            else:
                parsed = float(self.max_features)
                if not math.isfinite(parsed) or not 0.0 < parsed <= 1.0:
                    raise ORCARandomForestError("float max_features must be in (0, 1]")
        if self.class_weight not in {None, "balanced", "balanced_subsample"}:
            raise ORCARandomForestError(
                "class_weight must be None, 'balanced', or 'balanced_subsample'"
            )
        if isinstance(self.random_state, bool) or not isinstance(
            self.random_state, int
        ):
            raise ORCARandomForestError("random_state must be an integer")
        if isinstance(self.n_jobs, bool) or not isinstance(self.n_jobs, int):
            raise ORCARandomForestError("n_jobs must be a non-zero integer")
        if self.n_jobs == 0:
            raise ORCARandomForestError("n_jobs must be a non-zero integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "min_samples_split": self.min_samples_split,
            "max_features": self.max_features,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }


@dataclass(frozen=True)
class ResolvedFold:
    index: int
    train_start: int
    train_end: int
    gap_start: int
    gap_end: int
    test_start: int
    test_end: int

    def to_dict(self) -> dict[str, int]:
        return {
            "index": self.index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "gap_start": self.gap_start,
            "gap_end": self.gap_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
        }


def _canonical_json(payload: Any) -> str:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ORCARandomForestError("payload is not finite canonical JSON") from exc


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _payload_without_hash(artifact: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(artifact)
    payload.pop("artifact_hash", None)
    return payload


def compute_artifact_hash(artifact: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 after excluding ``artifact_hash``."""

    return _canonical_hash(_payload_without_hash(artifact))


def _validate_time_index(index: pd.Index) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise ORCARandomForestError("features and close must use a DatetimeIndex")
    if index.hasnans:
        raise ORCARandomForestError("time index contains NaT")
    if index.has_duplicates:
        raise ORCARandomForestError("time index must be unique")
    if not index.is_monotonic_increasing:
        raise ORCARandomForestError("time index must be monotonically increasing")
    return index


def _timestamp(value: Any) -> str:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ORCARandomForestError("timestamp cannot be NaT")
    return parsed.isoformat()


def build_orca_targets(
    benchmark_close: pd.Series,
    *,
    config: TargetConfig = TargetConfig(),
) -> pd.DataFrame:
    """Build strict rally/crash labels and their causal availability time.

    A row at ``t`` is labelled only when the close at ``t + horizon`` is
    available.  Equality at either threshold is a negative label, matching
    the paper's strict ``> +3%`` and ``< -7%`` definitions.
    """

    if not isinstance(benchmark_close, pd.Series):
        raise ORCARandomForestError("benchmark_close must be a pandas Series")
    index = _validate_time_index(benchmark_close.index)
    try:
        close = benchmark_close.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ORCARandomForestError("benchmark_close must be numeric") from exc
    if close.ndim != 1 or len(close) <= config.horizon:
        raise ORCARandomForestError(
            "benchmark_close must contain more rows than the target horizon"
        )
    if not np.isfinite(close).all() or np.any(close <= 0.0):
        raise ORCARandomForestError(
            "benchmark_close must contain finite positive values"
        )

    labelled = len(close) - config.horizon
    entry = close[:labelled]
    endpoint = close[config.horizon :]
    rally_cutoff = entry * (1.0 + float(config.rally_threshold))
    rally_values = (endpoint > rally_cutoff).astype(np.int8)

    forward_windows = np.lib.stride_tricks.sliding_window_view(
        close[1:], config.horizon
    )[:labelled]
    path_min = np.min(forward_windows, axis=1)
    crash_cutoff = entry * (1.0 + float(config.crash_threshold))
    crash_values = (path_min < crash_cutoff).astype(np.int8)

    rally = pd.Series(pd.NA, index=index, dtype="Int8", name="rally")
    crash = pd.Series(pd.NA, index=index, dtype="Int8", name="crash")
    rally.iloc[:labelled] = rally_values
    crash.iloc[:labelled] = crash_values
    known_at = index.to_series(index=index).shift(-config.horizon)
    known_at.name = "label_known_at"
    return pd.concat([rally, crash, known_at], axis=1)


def resolve_fold_windows(
    *,
    sample_count: int,
    labelled_count: int,
    window: FoldWindow = FoldWindow(),
) -> tuple[ResolvedFold, ...]:
    """Resolve explicit half-open row windows and reject incomplete folds."""

    total = _positive_int(sample_count, name="sample_count")
    labelled = _positive_int(labelled_count, name="labelled_count")
    if labelled > total:
        raise ORCARandomForestError("labelled_count cannot exceed sample_count")

    resolved: list[ResolvedFold] = []
    for fold_index in range(window.folds):
        offset = fold_index * window.step
        train_start = offset if window.mode == "rolling" else 0
        train_end = window.train_size + offset
        gap_start = train_end
        gap_end = gap_start + window.gap_size
        test_start = gap_end
        test_end = test_start + window.test_size
        if test_end > labelled:
            raise ORCARandomForestError(
                "insufficient labelled rows for the requested fold window: "
                f"fold={fold_index}, required={test_end}, available={labelled}"
            )
        fold = ResolvedFold(
            index=fold_index,
            train_start=train_start,
            train_end=train_end,
            gap_start=gap_start,
            gap_end=gap_end,
            test_start=test_start,
            test_end=test_end,
        )
        if not (
            0
            <= fold.train_start
            < fold.train_end
            == fold.gap_start
            <= fold.gap_end
            == fold.test_start
            < fold.test_end
            <= total
        ):
            raise ORCARandomForestError("invalid or overlapping fold boundaries")
        if resolved and resolved[-1].test_end > fold.test_start:
            raise ORCARandomForestError("overlapping OOS windows are forbidden")
        resolved.append(fold)
    return tuple(resolved)


def _validate_features(features: pd.DataFrame) -> tuple[pd.DatetimeIndex, list[str]]:
    if not isinstance(features, pd.DataFrame):
        raise ORCARandomForestError("features must be a pandas DataFrame")
    index = _validate_time_index(features.index)
    if features.empty or features.shape[1] == 0:
        raise ORCARandomForestError("features must not be empty")
    names = list(features.columns)
    if any(not isinstance(name, str) or not name for name in names):
        raise ORCARandomForestError("feature names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ORCARandomForestError("feature names must be unique")
    try:
        matrix = features.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ORCARandomForestError("all causal features must be numeric") from exc
    if matrix.ndim != 2 or matrix.shape != features.shape:
        raise ORCARandomForestError("invalid feature matrix shape")
    return index, names


def _feature_matrix(features: pd.DataFrame) -> np.ndarray:
    try:
        return features.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ORCARandomForestError("all causal features must be numeric") from exc


def _feature_registry(names: Sequence[str]) -> dict[str, Any]:
    body = {
        "schema": ORCA_FEATURE_REGISTRY_SCHEMA,
        "names": list(names),
        "count": len(names),
        "order_is_contractual": True,
    }
    return {**body, "hash": _canonical_hash(body)}


def _fit_preprocessor(matrix: np.ndarray) -> dict[str, list[float]]:
    finite = np.where(np.isfinite(matrix), matrix, np.nan)
    centers: list[float] = []
    scales: list[float] = []
    for column in range(finite.shape[1]):
        values = finite[:, column]
        available = values[np.isfinite(values)]
        if available.size == 0:
            center = 0.0
            scale = 1.0
        else:
            center = float(np.median(available))
            q1, q3 = np.percentile(available, [25.0, 75.0])
            scale = float(q3 - q1)
            if not math.isfinite(scale) or scale <= 0.0:
                scale = 1.0
        centers.append(center)
        scales.append(scale)
    return {"center": centers, "scale": scales, "finite_policy": "median_then_iqr"}


def _transform(matrix: np.ndarray, preprocessor: Mapping[str, Any]) -> np.ndarray:
    center = np.asarray(preprocessor["center"], dtype=float)
    scale = np.asarray(preprocessor["scale"], dtype=float)
    if center.ndim != 1 or scale.shape != center.shape:
        raise ORCARandomForestError("invalid frozen preprocessor shape")
    if matrix.ndim != 2 or matrix.shape[1] != len(center):
        raise ORCARandomForestError("feature count does not match preprocessor")
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ORCARandomForestError("preprocessor contains non-finite values")
    if np.any(scale <= 0.0):
        raise ORCARandomForestError("preprocessor scale must be positive")
    filled = np.where(np.isfinite(matrix), matrix, center)
    transformed = (filled - center) / scale
    if not np.isfinite(transformed).all():
        raise ORCARandomForestError("feature transformation produced non-finite values")
    return transformed


def _positive_probability(model: Any, matrix: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(matrix), dtype=float)
    classes = [int(value) for value in np.asarray(model.classes_).tolist()]
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ORCARandomForestError("unexpected sklearn probability shape")
    if 1 not in classes:
        return np.zeros(len(matrix), dtype=float)
    return probabilities[:, classes.index(1)]


def _portable_forest(model: Any) -> dict[str, Any]:
    classes = [int(value) for value in np.asarray(model.classes_).tolist()]
    trees: list[dict[str, Any]] = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        values = np.asarray(tree.value, dtype=float).reshape(tree.node_count, -1)
        if values.shape[1] != len(classes):
            raise ORCARandomForestError("tree class dimension does not match forest")
        trees.append(
            {
                "children_left": [int(value) for value in tree.children_left],
                "children_right": [int(value) for value in tree.children_right],
                "feature": [int(value) for value in tree.feature],
                "threshold": [float(value) for value in tree.threshold],
                "value": [[float(component) for component in row] for row in values],
            }
        )
    return {
        "classes": classes,
        "trees": trees,
        "probability": "mean_leaf_class_distribution",
    }


def _tree_probability(
    tree: Mapping[str, Any],
    row: np.ndarray,
    *,
    classes: Sequence[int],
) -> float:
    left = tree["children_left"]
    right = tree["children_right"]
    feature = tree["feature"]
    threshold = tree["threshold"]
    values = tree["value"]
    node = 0
    hops = 0
    while int(left[node]) != -1:
        feature_index = int(feature[node])
        node = (
            int(left[node])
            if row[feature_index] <= float(threshold[node])
            else int(right[node])
        )
        hops += 1
        if hops > len(left):
            raise ORCARandomForestError("tree contains a cycle")
    leaf = np.asarray(values[node], dtype=float)
    total = float(leaf.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ORCARandomForestError("tree leaf has invalid class mass")
    if 1 not in classes:
        return 0.0
    return float(leaf[classes.index(1)] / total)


def _frozen_forest_probability(
    forest: Mapping[str, Any], matrix: np.ndarray
) -> np.ndarray:
    classes = [int(value) for value in forest["classes"]]
    trees = forest["trees"]
    if not trees:
        raise ORCARandomForestError("frozen forest has no trees")
    result = np.zeros(len(matrix), dtype=float)
    for tree in trees:
        result += np.asarray(
            [_tree_probability(tree, row, classes=classes) for row in matrix],
            dtype=float,
        )
    result /= float(len(trees))
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
        raise ORCARandomForestError("frozen forest produced invalid probabilities")
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        average = (position + 1 + end) / 2.0
        ranks[order[position:end]] = average
        position = end
    return ranks


def _binary_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    probability = np.asarray(scores, dtype=float)
    support = {
        "rows": int(len(y)),
        "positive": int(np.sum(y == 1)),
        "negative": int(np.sum(y == 0)),
    }
    undefined = {
        "auc": None,
        "average_precision": None,
        "brier": None,
        "log_loss": None,
        "support": support,
        "status": "undefined_missing_class",
    }
    if len(y) == 0 or set(np.unique(y).tolist()) != {0, 1}:
        return undefined
    if not np.isfinite(probability).all() or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ORCARandomForestError("metric probabilities must be finite in [0, 1]")

    positive = support["positive"]
    negative = support["negative"]
    ranks = _average_ranks(probability)
    auc = (float(ranks[y == 1].sum()) - positive * (positive + 1) / 2.0) / (
        positive * negative
    )

    order = np.argsort(-probability, kind="mergesort")
    sorted_scores = probability[order]
    sorted_labels = y[order]
    true_positive = 0
    false_positive = 0
    average_precision = 0.0
    position = 0
    while position < len(y):
        end = position + 1
        while end < len(y) and sorted_scores[end] == sorted_scores[position]:
            end += 1
        group = sorted_labels[position:end]
        previous_true_positive = true_positive
        true_positive += int(np.sum(group == 1))
        false_positive += int(np.sum(group == 0))
        precision = true_positive / (true_positive + false_positive)
        average_precision += (
            (true_positive - previous_true_positive) / positive
        ) * precision
        position = end

    brier = float(np.mean((probability - y) ** 2))
    clipped = np.clip(probability, 1e-15, 1.0 - 1e-15)
    log_loss = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1.0 - clipped)))
    return {
        "auc": float(auc),
        "average_precision": float(average_precision),
        "brier": brier,
        "log_loss": log_loss,
        "support": support,
        "status": "ok",
    }


def _paired_metrics(
    rally_labels: Sequence[int],
    rally_scores: Sequence[float],
    crash_labels: Sequence[int],
    crash_scores: Sequence[float],
) -> dict[str, Any]:
    rally = _binary_metrics(rally_labels, rally_scores)
    crash = _binary_metrics(crash_labels, crash_scores)
    if rally["auc"] is None or crash["auc"] is None:
        bcd_auc = None
        status = "undefined_missing_class"
    else:
        bcd_auc = math.sqrt(float(rally["auc"]) * float(crash["auc"]))
        status = "ok"
    return {
        "rally": rally,
        "crash": crash,
        "bcd_auc": bcd_auc,
        "bcd_formula": "sqrt(rally_auc * crash_auc)",
        "status": status,
    }


def _fold_timestamp_lineage(
    fold: ResolvedFold,
    index: pd.DatetimeIndex,
) -> dict[str, Any]:
    result = fold.to_dict()
    result.update(
        {
            "train_start_time": _timestamp(index[fold.train_start]),
            "train_end_exclusive_time": _timestamp(index[fold.train_end]),
            "gap_start_time": _timestamp(index[fold.gap_start]),
            "gap_end_exclusive_time": _timestamp(index[fold.gap_end]),
            "test_start_time": _timestamp(index[fold.test_start]),
            "test_end_exclusive_time": _timestamp(index[fold.test_end]),
        }
    )
    return result


def fit_orca_random_forest(
    features: pd.DataFrame,
    benchmark_close: pd.Series,
    *,
    fold_window: FoldWindow = FoldWindow(),
    target_config: TargetConfig = TargetConfig(),
    forest_config: RandomForestConfig = RandomForestConfig(),
) -> dict[str, Any]:
    """Fit two fold-frozen RF classifiers and return a canonical artifact.

    The only scikit-learn import in this module intentionally lives here.
    All preprocessing is estimated on mature training labels for the current
    fold.  Test features are scored once by the frozen fold model.
    """

    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:  # pragma: no cover - exercised without dependency
        raise ORCARandomForestError(
            "scikit-learn is required only for fitting; install the pinned "
            "requirements-backtest.txt environment"
        ) from exc

    index, feature_names = _validate_features(features)
    close_index = _validate_time_index(benchmark_close.index)
    if not index.equals(close_index):
        raise ORCARandomForestError(
            "features and benchmark_close must have identical ordered indices"
        )
    matrix = _feature_matrix(features)
    targets = build_orca_targets(benchmark_close, config=target_config)
    labelled_count = len(index) - target_config.horizon
    folds = resolve_fold_windows(
        sample_count=len(index),
        labelled_count=labelled_count,
        window=fold_window,
    )

    fold_payloads: list[dict[str, Any]] = []
    pooled_rally_labels: list[int] = []
    pooled_crash_labels: list[int] = []
    pooled_rally_probability: list[float] = []
    pooled_crash_probability: list[float] = []

    forest_kwargs = forest_config.to_dict()
    for fold in folds:
        train_positions = np.arange(fold.train_start, fold.train_end, dtype=int)
        train_cutoff = index[fold.train_end]
        known_at = targets["label_known_at"].iloc[train_positions]
        mature_mask = known_at.notna().to_numpy() & (
            known_at.to_numpy() < train_cutoff.to_datetime64()
            if train_cutoff.tzinfo is None
            else known_at.array < train_cutoff
        )
        mature_positions = train_positions[mature_mask]
        if mature_positions.size == 0:
            raise ORCARandomForestError(
                f"fold {fold.index} has no labels resolved before train_end"
            )

        train_targets = targets.iloc[mature_positions]
        if train_targets[["rally", "crash"]].isna().any().any():
            raise ORCARandomForestError("mature training targets cannot be missing")
        preprocessor = _fit_preprocessor(matrix[mature_positions])
        train_matrix = _transform(matrix[mature_positions], preprocessor)
        test_positions = np.arange(fold.test_start, fold.test_end, dtype=int)
        test_matrix = _transform(matrix[test_positions], preprocessor)

        rally_model = RandomForestClassifier(**forest_kwargs)
        crash_model = RandomForestClassifier(**forest_kwargs)
        rally_train = train_targets["rally"].astype(int).to_numpy()
        crash_train = train_targets["crash"].astype(int).to_numpy()
        rally_model.fit(train_matrix, rally_train)
        crash_model.fit(train_matrix, crash_train)

        rally_probability = _positive_probability(rally_model, test_matrix)
        crash_probability = _positive_probability(crash_model, test_matrix)
        test_targets = targets.iloc[test_positions]
        if test_targets[["rally", "crash", "label_known_at"]].isna().any().any():
            raise ORCARandomForestError("OOS targets must be fully resolved")
        rally_test = test_targets["rally"].astype(int).to_numpy()
        crash_test = test_targets["crash"].astype(int).to_numpy()

        oos_rows: list[dict[str, Any]] = []
        for offset, position in enumerate(test_positions):
            oos_rows.append(
                {
                    "position": int(position),
                    "timestamp": _timestamp(index[position]),
                    "label_known_at": _timestamp(
                        test_targets["label_known_at"].iloc[offset]
                    ),
                    "rally_label": int(rally_test[offset]),
                    "crash_label": int(crash_test[offset]),
                    "rally_probability": float(rally_probability[offset]),
                    "crash_probability": float(crash_probability[offset]),
                }
            )

        lineage = _fold_timestamp_lineage(fold, index)
        lineage.update(
            {
                "train_rows_before_label_purge": int(len(train_positions)),
                "train_rows_mature": int(len(mature_positions)),
                "train_rows_purged_unresolved": int(
                    len(train_positions) - len(mature_positions)
                ),
                "train_label_rule": "label_known_at < train_end_exclusive_time",
                "train_last_label_known_at": _timestamp(
                    targets["label_known_at"].iloc[mature_positions[-1]]
                ),
                "test_rows": int(len(test_positions)),
                "test_last_label_known_at": _timestamp(
                    test_targets["label_known_at"].iloc[-1]
                ),
            }
        )
        fold_payloads.append(
            {
                "index": fold.index,
                "lineage": lineage,
                "preprocessor": preprocessor,
                "rally_forest": _portable_forest(rally_model),
                "crash_forest": _portable_forest(crash_model),
                "oos": oos_rows,
                "metrics": _paired_metrics(
                    rally_test,
                    rally_probability,
                    crash_test,
                    crash_probability,
                ),
            }
        )
        pooled_rally_labels.extend(int(value) for value in rally_test)
        pooled_crash_labels.extend(int(value) for value in crash_test)
        pooled_rally_probability.extend(float(value) for value in rally_probability)
        pooled_crash_probability.extend(float(value) for value in crash_probability)

    artifact: dict[str, Any] = {
        "schema": ORCA_RANDOM_FOREST_SCHEMA,
        "calibration": {
            "status": "uncalibrated",
            "method": None,
            "claim": "RandomForest predict_proba is not treated as calibrated",
        },
        "feature_registry": _feature_registry(feature_names),
        "target_config": target_config.to_dict(),
        "fold_window": fold_window.to_dict(),
        "forest_config": forest_config.to_dict(),
        "lineage": {
            "input_rows": len(index),
            "labelled_rows": labelled_count,
            "input_start": _timestamp(index[0]),
            "input_end": _timestamp(index[-1]),
            "fit_dependency": "scikit-learn",
            "frozen_predict_dependency": "numpy",
        },
        "folds": fold_payloads,
        "metrics": _paired_metrics(
            pooled_rally_labels,
            pooled_rally_probability,
            pooled_crash_labels,
            pooled_crash_probability,
        ),
    }
    artifact["artifact_hash"] = compute_artifact_hash(artifact)
    validate_orca_artifact(artifact)
    return artifact


def _validate_tree(
    tree: Mapping[str, Any], *, feature_count: int, class_count: int
) -> None:
    keys = ("children_left", "children_right", "feature", "threshold", "value")
    if any(key not in tree for key in keys):
        raise ORCARandomForestError("tree is missing a required array")
    lengths = [len(tree[key]) for key in keys]
    if not lengths[0] or len(set(lengths)) != 1:
        raise ORCARandomForestError("tree arrays must have equal non-zero length")
    node_count = lengths[0]
    for node in range(node_count):
        left = int(tree["children_left"][node])
        right = int(tree["children_right"][node])
        feature = int(tree["feature"][node])
        threshold = float(tree["threshold"][node])
        value = np.asarray(tree["value"][node], dtype=float)
        if not math.isfinite(threshold):
            raise ORCARandomForestError("tree threshold must be finite")
        if value.shape != (class_count,) or not np.isfinite(value).all():
            raise ORCARandomForestError("tree class values are invalid")
        if np.any(value < 0.0) or float(value.sum()) <= 0.0:
            raise ORCARandomForestError("tree class values must have positive mass")
        if left == -1 or right == -1:
            if left != -1 or right != -1 or feature != -2:
                raise ORCARandomForestError("invalid leaf encoding")
        else:
            if not (0 <= left < node_count and 0 <= right < node_count):
                raise ORCARandomForestError("tree child index is out of bounds")
            if not 0 <= feature < feature_count:
                raise ORCARandomForestError("tree feature index is out of bounds")


def _validate_forest(forest: Mapping[str, Any], *, feature_count: int) -> None:
    classes = [int(value) for value in forest.get("classes", [])]
    if not classes or len(classes) != len(set(classes)):
        raise ORCARandomForestError("forest classes are invalid")
    if any(value not in {0, 1} for value in classes):
        raise ORCARandomForestError("forest classes must be binary")
    trees = forest.get("trees")
    if not isinstance(trees, list) or not trees:
        raise ORCARandomForestError("forest must contain portable trees")
    for tree in trees:
        _validate_tree(tree, feature_count=feature_count, class_count=len(classes))


def validate_orca_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed on hash, feature-contract, lineage, or tree corruption."""

    if not isinstance(artifact, Mapping):
        raise ORCARandomForestError("artifact must be a mapping")
    if artifact.get("schema") != ORCA_RANDOM_FOREST_SCHEMA:
        raise ORCARandomForestError("unsupported ORCA artifact schema")
    calibration = artifact.get("calibration", {})
    if calibration.get("status") != "uncalibrated":
        raise ORCARandomForestError("artifact must state uncalibrated status")
    expected_hash = artifact.get("artifact_hash")
    if not isinstance(expected_hash, str) or expected_hash != compute_artifact_hash(
        artifact
    ):
        raise ORCARandomForestError("artifact hash mismatch")

    registry = artifact.get("feature_registry")
    if not isinstance(registry, Mapping):
        raise ORCARandomForestError("feature registry is missing")
    registry_body = dict(registry)
    registry_hash = registry_body.pop("hash", None)
    if registry_body.get("schema") != ORCA_FEATURE_REGISTRY_SCHEMA:
        raise ORCARandomForestError("unsupported feature registry schema")
    names = registry_body.get("names")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or registry_body.get("count") != len(names)
        or registry_hash != _canonical_hash(registry_body)
    ):
        raise ORCARandomForestError("feature registry hash or names are invalid")

    folds = artifact.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ORCARandomForestError("artifact has no folds")
    previous_test_end = -1
    for expected_index, fold in enumerate(folds):
        if fold.get("index") != expected_index:
            raise ORCARandomForestError("fold indices must be contiguous")
        lineage = fold.get("lineage", {})
        test_start = int(lineage.get("test_start", -1))
        test_end = int(lineage.get("test_end", -1))
        if test_start < previous_test_end or test_end <= test_start:
            raise ORCARandomForestError("fold OOS lineage overlaps or is invalid")
        previous_test_end = test_end
        preprocessor = fold.get("preprocessor", {})
        center = np.asarray(preprocessor.get("center", []), dtype=float)
        scale = np.asarray(preprocessor.get("scale", []), dtype=float)
        if (
            center.shape != (len(names),)
            or scale.shape != center.shape
            or not np.isfinite(center).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0.0)
        ):
            raise ORCARandomForestError("fold preprocessor is invalid")
        _validate_forest(fold.get("rally_forest", {}), feature_count=len(names))
        _validate_forest(fold.get("crash_forest", {}), feature_count=len(names))


def artifact_to_json(artifact: Mapping[str, Any]) -> str:
    """Validate and return stable canonical JSON."""

    validate_orca_artifact(artifact)
    return _canonical_json(artifact)


def load_orca_artifact(payload: str | bytes | Mapping[str, Any]) -> dict[str, Any]:
    """Load and validate a canonical artifact without importing sklearn."""

    if isinstance(payload, Mapping):
        artifact = dict(payload)
    else:
        try:
            artifact = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ORCARandomForestError("invalid ORCA artifact JSON") from exc
    validate_orca_artifact(artifact)
    return artifact


def predict_orca_frozen(
    artifact: Mapping[str, Any],
    features: pd.DataFrame,
    *,
    fold_index: int = -1,
) -> pd.DataFrame:
    """Score a feature frame from portable arrays, with no sklearn import."""

    validate_orca_artifact(artifact)
    _, names = _validate_features(features)
    expected = list(artifact["feature_registry"]["names"])
    if names != expected:
        raise ORCARandomForestError(
            "feature names and order must exactly match the frozen registry"
        )
    folds = artifact["folds"]
    resolved_index = fold_index if fold_index >= 0 else len(folds) + fold_index
    if not 0 <= resolved_index < len(folds):
        raise ORCARandomForestError("fold_index is out of bounds")
    fold = folds[resolved_index]
    matrix = _transform(_feature_matrix(features), fold["preprocessor"])
    rally = _frozen_forest_probability(fold["rally_forest"], matrix)
    crash = _frozen_forest_probability(fold["crash_forest"], matrix)
    return pd.DataFrame(
        {
            "rally_probability": rally,
            "crash_probability": crash,
        },
        index=features.index,
    )


__all__ = [
    "FoldWindow",
    "ORCARandomForestError",
    "ORCA_RANDOM_FOREST_SCHEMA",
    "RandomForestConfig",
    "ResolvedFold",
    "TargetConfig",
    "artifact_to_json",
    "build_orca_targets",
    "compute_artifact_hash",
    "fit_orca_random_forest",
    "load_orca_artifact",
    "predict_orca_frozen",
    "resolve_fold_windows",
    "validate_orca_artifact",
]
