"""Train-only factor-weight fitting for counterfactual opportunities.

This module deliberately has no strategy or execution side effects.  Its input
is an already-generated technical-opportunity universe: both directions and
all trigger families must have been detected independently of the production
bias before calling this code.

Row contract
------------
Every opportunity row must contain:

``opportunity_id``
    Stable identifier independent of an outer walk-forward fold.
``decision_time``
    UTC-compatible timestamp at which every feature was known.
``side``
    ``LONG`` or ``SHORT``.
``factor_vector``
    A frozen five-factor vector using ``core.narrative_scoring`` keys.
``label_status``
    ``FILLED``, ``NO_FILL``, ``PENDING``, ``CENSORED``, or ``INVALID``.
``opportunity_r``
    Finite realised setup R for ``FILLED``; exactly zero for ``NO_FILL``;
    otherwise ``None``.
``exit_time``
    Label-maturity time for ``FILLED`` and ``NO_FILL``.  It may be ``None`` for
    non-mature rows.

``decision_event_id`` is required so correlated trigger rows from the same
M15 decision and side can share one unit of training weight.  Optional
``symbol`` and ``trigger_kind`` plus a positive base ``sample_weight`` are
copied into audit artifacts.  No score threshold is applied: every OOS
opportunity receives the frozen fold score whenever its fold has enough
independent decision-side clusters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from core.narrative_scoring import FACTOR_DEFINITIONS, FACTOR_VECTOR_SCHEMA


WEIGHT_OPTIMIZER_SCHEMA = "counterfactual-factor-weight-optimizer/v1"
# The fit shrinks toward the live contract, so this follows the definitions'
# configured weights and stays [2, 2, 1, 1, 1] with a zero-weight FVG row.
REFERENCE_WEIGHTS = {
    definition.key: float(definition.configured_weight)
    for definition in FACTOR_DEFINITIONS
}
WEIGHT_MIN = 0.0
WEIGHT_MAX = 3.0
WEIGHT_SUM = float(sum(REFERENCE_WEIGHTS.values()))
DEFAULT_RIDGE_ALPHA = 1.0
DEFAULT_MIN_TRAIN_OPPORTUNITIES = 400
DEFAULT_MAX_ITERATIONS = 10_000
DEFAULT_TOLERANCE = 1e-12

_MATURE_LABEL_STATUSES = frozenset({"FILLED", "NO_FILL"})
_NON_MATURE_LABEL_STATUSES = frozenset(
    {"PENDING", "CENSORED", "INVALID"}
)
_LABEL_STATUSES = _MATURE_LABEL_STATUSES | _NON_MATURE_LABEL_STATUSES
_FACTOR_KEYS = tuple(definition.key for definition in FACTOR_DEFINITIONS)


class WeightOptimizerError(ValueError):
    """Raised when the opportunity universe cannot be audited safely."""


@dataclass(frozen=True)
class _Opportunity:
    opportunity_id: str
    decision_event_id: str
    decision_time: pd.Timestamp
    exit_time: pd.Timestamp | None
    side: str
    symbol: str
    trigger_kind: str
    label_status: str
    opportunity_r: float | None
    sample_weight: float
    alignment: tuple[float, ...]


@dataclass(frozen=True)
class _Period:
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


@dataclass(frozen=True)
class _Fit:
    intercept: float
    weights: np.ndarray
    iterations: int
    converged: bool
    objective: float


def _timestamp(value: Any, *, name: str) -> pd.Timestamp:
    if value is None:
        raise WeightOptimizerError(f"{name} is required")
    try:
        parsed = pd.Timestamp(value)
    except Exception as exc:
        raise WeightOptimizerError(f"{name} is invalid: {value!r}") from exc
    if pd.isna(parsed):
        raise WeightOptimizerError(f"{name} is invalid: {value!r}")
    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def _optional_timestamp(value: Any, *, name: str) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    return _timestamp(value, name=name)


def _duration(value: Any, *, name: str) -> pd.Timedelta:
    try:
        duration = pd.Timedelta(value)
    except Exception as exc:
        raise WeightOptimizerError(
            f"{name} is invalid: {value!r}"
        ) from exc
    if duration <= pd.Timedelta(0):
        raise WeightOptimizerError(f"{name} must be positive")
    return duration


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_number(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise WeightOptimizerError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise WeightOptimizerError(f"{name} must be finite")
    return parsed


def _factor_alignment(
    factor_vector: Mapping[str, Any],
    *,
    side: str,
    opportunity_id: str,
) -> tuple[float, ...]:
    if factor_vector.get("schema") != FACTOR_VECTOR_SCHEMA:
        raise WeightOptimizerError(
            f"{opportunity_id}: unexpected factor-vector schema"
        )
    rows = factor_vector.get("factors")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise WeightOptimizerError(
            f"{opportunity_id}: factor_vector.factors must be a sequence"
        )
    by_key: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("key"):
            raise WeightOptimizerError(
                f"{opportunity_id}: malformed factor row"
            )
        key = str(row["key"])
        if key in by_key:
            raise WeightOptimizerError(
                f"{opportunity_id}: duplicate factor {key}"
            )
        by_key[key] = row
    if set(by_key) != set(_FACTOR_KEYS):
        raise WeightOptimizerError(
            f"{opportunity_id}: factor vector must contain exactly "
            f"{list(_FACTOR_KEYS)}"
        )

    values: list[float] = []
    for key in _FACTOR_KEYS:
        row = by_key[key]
        present = bool(row.get("present"))
        vote_side = str(row.get("vote_side") or "NEUTRAL").upper()
        if not present or vote_side == "NEUTRAL":
            values.append(0.0)
        elif vote_side == side:
            values.append(1.0)
        elif vote_side in {"LONG", "SHORT"}:
            values.append(-1.0)
        else:
            raise WeightOptimizerError(
                f"{opportunity_id}: invalid vote_side for {key}: "
                f"{vote_side!r}"
            )
    return tuple(values)


def _prepare_opportunities(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[_Opportunity, ...]:
    prepared: list[_Opportunity] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise WeightOptimizerError(
                f"opportunity row {index} must be a mapping"
            )
        opportunity_id = str(row.get("opportunity_id") or "").strip()
        if not opportunity_id:
            raise WeightOptimizerError(
                f"opportunity row {index} is missing opportunity_id"
            )
        if opportunity_id in seen_ids:
            raise WeightOptimizerError(
                f"duplicate opportunity_id: {opportunity_id}"
            )
        seen_ids.add(opportunity_id)

        side = str(row.get("side") or "").upper()
        if side not in {"LONG", "SHORT"}:
            raise WeightOptimizerError(
                f"{opportunity_id}: side must be LONG or SHORT"
            )
        label_status = str(row.get("label_status") or "").upper()
        if label_status not in _LABEL_STATUSES:
            raise WeightOptimizerError(
                f"{opportunity_id}: unsupported label_status "
                f"{label_status!r}"
            )
        decision_time = _timestamp(
            row.get("decision_time"),
            name=f"{opportunity_id}.decision_time",
        )
        exit_time = _optional_timestamp(
            row.get("exit_time"),
            name=f"{opportunity_id}.exit_time",
        )
        if exit_time is not None and exit_time < decision_time:
            raise WeightOptimizerError(
                f"{opportunity_id}: exit_time cannot precede decision_time"
            )

        raw_target = row.get("opportunity_r")
        target: float | None
        if label_status in _MATURE_LABEL_STATUSES:
            if exit_time is None:
                raise WeightOptimizerError(
                    f"{opportunity_id}: mature label requires exit_time"
                )
            target = _finite_number(
                raw_target,
                name=f"{opportunity_id}.opportunity_r",
            )
            if label_status == "NO_FILL" and not math.isclose(
                target,
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise WeightOptimizerError(
                    f"{opportunity_id}: NO_FILL opportunity_r must be zero"
                )
        else:
            if raw_target is not None:
                raise WeightOptimizerError(
                    f"{opportunity_id}: non-mature label must not contain "
                    "opportunity_r"
                )
            target = None

        sample_weight = _finite_number(
            row.get("sample_weight", 1.0),
            name=f"{opportunity_id}.sample_weight",
        )
        if sample_weight <= 0:
            raise WeightOptimizerError(
                f"{opportunity_id}: sample_weight must be positive"
            )
        vector = row.get("factor_vector")
        if not isinstance(vector, Mapping):
            raise WeightOptimizerError(
                f"{opportunity_id}: factor_vector is required"
            )
        alignment = _factor_alignment(
            vector,
            side=side,
            opportunity_id=opportunity_id,
        )
        decision_event_id = str(
            row.get("decision_event_id") or ""
        ).strip()
        if not decision_event_id:
            raise WeightOptimizerError(
                f"{opportunity_id}: decision_event_id is required"
            )
        prepared.append(
            _Opportunity(
                opportunity_id=opportunity_id,
                decision_event_id=decision_event_id,
                decision_time=decision_time,
                exit_time=exit_time,
                side=side,
                symbol=str(row.get("symbol") or ""),
                trigger_kind=str(row.get("trigger_kind") or "unknown"),
                label_status=label_status,
                opportunity_r=target,
                sample_weight=sample_weight,
                alignment=alignment,
            )
        )
    return tuple(
        sorted(
            prepared,
            key=lambda item: (
                item.decision_time,
                item.symbol,
                item.side,
                item.trigger_kind,
                item.opportunity_id,
            ),
        )
    )


def _prepare_periods(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[_Period, ...]:
    periods: list[_Period] = []
    fold_indices: set[int] = set()
    for index, row in enumerate(rows):
        try:
            fold_index = int(row["fold_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WeightOptimizerError(
                f"period row {index} has invalid fold_index"
            ) from exc
        if fold_index in fold_indices:
            raise WeightOptimizerError(
                f"duplicate fold_index: {fold_index}"
            )
        fold_indices.add(fold_index)
        period = _Period(
            fold_index=fold_index,
            train_start=_timestamp(
                row.get("train_start"),
                name=f"fold {fold_index}.train_start",
            ),
            train_end=_timestamp(
                row.get("train_end"),
                name=f"fold {fold_index}.train_end",
            ),
            test_start=_timestamp(
                row.get("test_start"),
                name=f"fold {fold_index}.test_start",
            ),
            test_end=_timestamp(
                row.get("test_end"),
                name=f"fold {fold_index}.test_end",
            ),
        )
        if not (
            period.train_start
            < period.train_end
            <= period.test_start
            < period.test_end
        ):
            raise WeightOptimizerError(
                f"fold {fold_index}: require train_start < train_end <= "
                "test_start < test_end"
            )
        periods.append(period)

    ordered = tuple(sorted(periods, key=lambda item: item.fold_index))
    chronological = sorted(ordered, key=lambda item: item.test_start)
    for earlier, later in zip(chronological, chronological[1:]):
        if earlier.test_end > later.test_start:
            raise WeightOptimizerError(
                "outer test windows must not overlap"
            )
    return ordered


def _project_capped_simplex(values: np.ndarray) -> np.ndarray:
    """Project onto ``WEIGHT_MIN <= w <= WEIGHT_MAX, sum(w)=WEIGHT_SUM``."""

    vector = np.asarray(values, dtype=float)
    if vector.shape != (len(_FACTOR_KEYS),):
        raise WeightOptimizerError("weight vector has unexpected shape")
    if not np.all(np.isfinite(vector)):
        raise WeightOptimizerError("weight vector must be finite")
    count = len(vector)
    if not (
        count * WEIGHT_MIN <= WEIGHT_SUM <= count * WEIGHT_MAX
    ):
        raise WeightOptimizerError("weight constraints are infeasible")

    low = float(np.min(vector - WEIGHT_MAX))
    high = float(np.max(vector - WEIGHT_MIN))
    for _ in range(200):
        threshold = (low + high) / 2.0
        projected = np.clip(
            vector - threshold,
            WEIGHT_MIN,
            WEIGHT_MAX,
        )
        if float(np.sum(projected)) > WEIGHT_SUM:
            low = threshold
        else:
            high = threshold
    projected = np.clip(
        vector - (low + high) / 2.0,
        WEIGHT_MIN,
        WEIGHT_MAX,
    )
    residual = WEIGHT_SUM - float(np.sum(projected))
    if abs(residual) > 1e-10:
        raise WeightOptimizerError(
            "capped-simplex projection did not converge"
        )
    if abs(residual) > 0:
        adjustable = np.flatnonzero(
            (projected > WEIGHT_MIN + 1e-12)
            & (projected < WEIGHT_MAX - 1e-12)
        )
        if len(adjustable):
            projected[adjustable[0]] += residual
    return projected


def _fit_weights(
    rows: Sequence[_Opportunity],
    *,
    ridge_alpha: float,
    max_iterations: int,
    tolerance: float,
) -> _Fit:
    matrix = np.asarray(
        [row.alignment for row in rows],
        dtype=float,
    ) / WEIGHT_SUM
    target = np.asarray(
        [float(row.opportunity_r) for row in rows],
        dtype=float,
    )
    sample_weights = np.asarray(
        [row.sample_weight for row in rows],
        dtype=float,
    )
    sample_weights /= float(np.sum(sample_weights))

    feature_mean = np.sum(matrix * sample_weights[:, None], axis=0)
    target_mean = float(np.sum(target * sample_weights))
    centered_matrix = matrix - feature_mean
    centered_target = target - target_mean
    reference = np.asarray(
        [REFERENCE_WEIGHTS[key] for key in _FACTOR_KEYS],
        dtype=float,
    )
    weights = _project_capped_simplex(reference)

    weighted_matrix = centered_matrix * np.sqrt(
        sample_weights[:, None]
    )
    gram = weighted_matrix.T @ weighted_matrix
    smoothness = (
        2.0 * float(np.max(np.linalg.eigvalsh(gram)))
        + 2.0 * ridge_alpha / len(_FACTOR_KEYS)
    )
    step = 1.0 / max(smoothness, 1e-12)

    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        residual = centered_matrix @ weights - centered_target
        gradient = (
            2.0
            * centered_matrix.T
            @ (sample_weights * residual)
            + 2.0
            * ridge_alpha
            * (weights - reference)
            / len(_FACTOR_KEYS)
        )
        updated = _project_capped_simplex(weights - step * gradient)
        if float(np.max(np.abs(updated - weights))) <= tolerance:
            weights = updated
            converged = True
            break
        weights = updated

    intercept = float(target_mean - feature_mean @ weights)
    predictions = intercept + matrix @ weights
    squared_error = float(
        np.sum(sample_weights * (predictions - target) ** 2)
    )
    penalty = float(
        ridge_alpha
        * np.mean((weights - reference) ** 2)
    )
    objective = squared_error + penalty
    return _Fit(
        intercept=intercept,
        weights=weights,
        iterations=iterations,
        converged=converged,
        objective=objective,
    )


def _training_payload(
    rows: Sequence[_Opportunity],
) -> list[dict[str, Any]]:
    return [
        {
            "opportunity_id": row.opportunity_id,
            "decision_time": row.decision_time.isoformat(),
            "exit_time": (
                row.exit_time.isoformat()
                if row.exit_time is not None
                else None
            ),
            "side": row.side,
            "label_status": row.label_status,
            "opportunity_r": row.opportunity_r,
            "sample_weight": row.sample_weight,
            "alignment": list(row.alignment),
        }
        for row in rows
    ]


def _rank_correlation(
    predicted: np.ndarray,
    actual: np.ndarray,
) -> float | None:
    if len(predicted) < 3:
        return None
    predicted_rank = pd.Series(predicted).rank(
        method="average"
    ).to_numpy()
    actual_rank = pd.Series(actual).rank(method="average").to_numpy()
    if np.std(predicted_rank) == 0 or np.std(actual_rank) == 0:
        return None
    return float(np.corrcoef(predicted_rank, actual_rank)[0, 1])


def _score_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scored = [
        row
        for row in rows
        if row.get("predicted_opportunity_r") is not None
    ]
    labeled = [
        row
        for row in scored
        if row.get("actual_opportunity_r") is not None
    ]
    if not labeled:
        return {
            "oos_opportunities": len(rows),
            "opportunities_scored": len(scored),
            "labeled_opportunities": 0,
            "filled_labels": 0,
            "no_fill_labels": 0,
            "actual_mean_r": None,
            "predicted_mean_r": None,
            "mae_r": None,
            "rmse_r": None,
            "rank_ic": None,
        }
    predicted = np.asarray(
        [float(row["predicted_opportunity_r"]) for row in labeled],
        dtype=float,
    )
    actual = np.asarray(
        [float(row["actual_opportunity_r"]) for row in labeled],
        dtype=float,
    )
    error = predicted - actual
    return {
        "oos_opportunities": len(rows),
        "opportunities_scored": len(scored),
        "labeled_opportunities": len(labeled),
        "filled_labels": sum(
            row.get("actual_label_status") == "FILLED"
            for row in labeled
        ),
        "no_fill_labels": sum(
            row.get("actual_label_status") == "NO_FILL"
            for row in labeled
        ),
        "actual_mean_r": float(np.mean(actual)),
        "predicted_mean_r": float(np.mean(predicted)),
        "mae_r": float(np.mean(np.abs(error))),
        "rmse_r": float(np.sqrt(np.mean(error**2))),
        "rank_ic": _rank_correlation(predicted, actual),
    }


def _metric_rows(
    predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in predictions:
        groups[("overall", "ALL")].append(row)
        groups[("fold", str(row.get("fold_index")))].append(row)
        groups[("symbol", str(row.get("symbol") or ""))].append(row)
        groups[("side", str(row.get("side")))].append(row)
        groups[("trigger", str(row.get("trigger_kind")))].append(row)
    return [
        {
            "schema": WEIGHT_OPTIMIZER_SCHEMA,
            "dimension": dimension,
            "dimension_value": value,
            **_score_metrics(rows),
        }
        for (dimension, value), rows in sorted(groups.items())
    ]


def build_counterfactual_weight_scores(
    *,
    opportunities: Sequence[Mapping[str, Any]],
    periods: Sequence[Mapping[str, Any]],
    entry_ttl: Any,
    max_holding: Any,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    min_train_opportunities: int = DEFAULT_MIN_TRAIN_OPPORTUNITIES,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Fit constrained factor weights in train and score every OOS row.

    The function does not generate candidates, choose a direction, reject an
    opportunity, change risk, or replay execution.  It only fits on mature
    train labels and applies the frozen fold model as a continuous OOS score.
    """

    alpha = _finite_number(ridge_alpha, name="ridge_alpha")
    if alpha <= 0:
        raise WeightOptimizerError("ridge_alpha must be positive")
    if int(min_train_opportunities) <= 0:
        raise WeightOptimizerError(
            "min_train_opportunities must be positive"
        )
    if int(max_iterations) <= 0:
        raise WeightOptimizerError("max_iterations must be positive")
    fit_tolerance = _finite_number(tolerance, name="tolerance")
    if fit_tolerance <= 0:
        raise WeightOptimizerError("tolerance must be positive")

    entry_delta = _duration(entry_ttl, name="entry_ttl")
    holding_delta = _duration(max_holding, name="max_holding")
    purge = entry_delta + holding_delta
    prepared = _prepare_opportunities(opportunities)
    prepared_periods = _prepare_periods(periods)

    models: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []

    for period in prepared_periods:
        purge_cutoff = period.train_end - purge
        raw_train_rows = [
            row
            for row in prepared
            if period.train_start <= row.decision_time < purge_cutoff
            and row.label_status in _MATURE_LABEL_STATUSES
            and row.exit_time is not None
            and row.exit_time < period.train_end
        ]
        test_rows = [
            row
            for row in prepared
            if period.test_start <= row.decision_time < period.test_end
        ]
        cluster_counts = Counter(
            (row.decision_event_id, row.side)
            for row in raw_train_rows
        )
        train_rows = [
            replace(
                row,
                sample_weight=(
                    row.sample_weight
                    / float(
                        cluster_counts[
                            (row.decision_event_id, row.side)
                        ]
                    )
                ),
            )
            for row in raw_train_rows
        ]
        train_clusters = len(cluster_counts)
        effective_weights = np.asarray(
            [row.sample_weight for row in train_rows],
            dtype=float,
        )
        effective_sample_size = (
            float(effective_weights.sum() ** 2)
            / float(np.square(effective_weights).sum())
            if len(effective_weights)
            else 0.0
        )
        status = (
            "FIT"
            if train_clusters >= int(min_train_opportunities)
            else "NOT_FIT"
        )
        training_payload = _training_payload(train_rows)
        training_sha256 = _canonical_hash(training_payload)
        reference_array = np.asarray(
            [REFERENCE_WEIGHTS[key] for key in _FACTOR_KEYS],
            dtype=float,
        )
        fit: _Fit | None = None
        if status == "FIT":
            fit = _fit_weights(
                train_rows,
                ridge_alpha=alpha,
                max_iterations=int(max_iterations),
                tolerance=fit_tolerance,
            )
            if not fit.converged:
                raise WeightOptimizerError(
                    f"fold {period.fold_index}: projected-gradient fit "
                    f"did not converge within {int(max_iterations)} iterations"
                )
            frozen_weights = fit.weights
            intercept: float | None = fit.intercept
        else:
            frozen_weights = reference_array
            intercept = None

        model_identity = {
            "schema": WEIGHT_OPTIMIZER_SCHEMA,
            "fold_index": period.fold_index,
            "status": status,
            "period": period.to_dict(),
            "purge_duration": purge.isoformat(),
            "training_sha256": training_sha256,
            "ridge_alpha": alpha,
            "min_train_opportunities": int(min_train_opportunities),
            "minimum_train_unit": "unique_decision_event_id_x_side",
            "train_opportunity_rows": len(raw_train_rows),
            "train_decision_side_clusters": train_clusters,
            "train_effective_sample_size": effective_sample_size,
            "constraints": {
                "minimum": WEIGHT_MIN,
                "maximum": WEIGHT_MAX,
                "sum": WEIGHT_SUM,
            },
            "intercept": intercept,
            "weights": {
                key: float(value)
                for key, value in zip(_FACTOR_KEYS, frozen_weights)
            },
        }
        model_id = _canonical_hash(model_identity)[:24]
        model = {
            **model_identity,
            "model_id": model_id,
            "train_start": period.train_start.isoformat(),
            "train_end": period.train_end.isoformat(),
            "purge_cutoff": purge_cutoff.isoformat(),
            "test_start": period.test_start.isoformat(),
            "test_end": period.test_end.isoformat(),
            "train_opportunities": len(raw_train_rows),
            "train_decision_side_clusters": train_clusters,
            "train_effective_sample_size": effective_sample_size,
            "test_opportunities": len(test_rows),
            "reference_weights": dict(REFERENCE_WEIGHTS),
            "iterations": fit.iterations if fit is not None else 0,
            "converged": fit.converged if fit is not None else False,
            "training_objective": (
                fit.objective if fit is not None else None
            ),
        }
        models.append(model)

        coefficients.append(
            {
                "schema": WEIGHT_OPTIMIZER_SCHEMA,
                "model_id": model_id,
                "fold_index": period.fold_index,
                "model_status": status,
                "feature": "intercept",
                "reference_weight": None,
                "fitted_weight": intercept,
                "delta_from_reference": None,
            }
        )
        coefficients.extend(
            {
                "schema": WEIGHT_OPTIMIZER_SCHEMA,
                "model_id": model_id,
                "fold_index": period.fold_index,
                "model_status": status,
                "feature": key,
                "reference_weight": REFERENCE_WEIGHTS[key],
                "fitted_weight": float(value),
                "delta_from_reference": (
                    float(value) - REFERENCE_WEIGHTS[key]
                ),
            }
            for key, value in zip(_FACTOR_KEYS, frozen_weights)
        )

        for row in test_rows:
            alignment = np.asarray(row.alignment, dtype=float)
            weighted_factor_score = float(alignment @ frozen_weights)
            predicted = (
                float(intercept + weighted_factor_score / WEIGHT_SUM)
                if intercept is not None
                else None
            )
            actual = (
                row.opportunity_r
                if row.label_status in _MATURE_LABEL_STATUSES
                and row.exit_time is not None
                and row.exit_time < period.test_end
                else None
            )
            predictions.append(
                {
                    "schema": WEIGHT_OPTIMIZER_SCHEMA,
                    "opportunity_id": row.opportunity_id,
                    "decision_event_id": row.decision_event_id,
                    "fold_index": period.fold_index,
                    "symbol": row.symbol,
                    "decision_time": row.decision_time.isoformat(),
                    "side": row.side,
                    "trigger_kind": row.trigger_kind,
                    "model_status": status,
                    "model_id": model_id,
                    "train_opportunities": len(raw_train_rows),
                    "train_decision_side_clusters": train_clusters,
                    "train_effective_sample_size": effective_sample_size,
                    "factor_alignment": {
                        key: float(value)
                        for key, value in zip(
                            _FACTOR_KEYS,
                            row.alignment,
                        )
                    },
                    "weighted_factor_score": weighted_factor_score,
                    "normalized_factor_score": (
                        weighted_factor_score / WEIGHT_SUM
                    ),
                    "predicted_opportunity_r": predicted,
                    "actual_label_status": row.label_status,
                    "actual_label_known_before_test_end": (
                        actual is not None
                    ),
                    "actual_opportunity_r": actual,
                    "prediction_error_r": (
                        predicted - actual
                        if predicted is not None and actual is not None
                        else None
                    ),
                    "score_threshold_applied": False,
                }
            )

    predictions.sort(
        key=lambda row: (
            int(row["fold_index"]),
            str(row["decision_time"]),
            str(row["symbol"]),
            str(row["side"]),
            str(row["trigger_kind"]),
            str(row["opportunity_id"]),
        )
    )
    metrics = _metric_rows(predictions)
    summary = {
        "schema": WEIGHT_OPTIMIZER_SCHEMA,
        "mode": "counterfactual-all-opportunity-frozen-score",
        "target": "opportunity_r_including_zero_for_no_fill",
        "factor_features": list(_FACTOR_KEYS),
        "reference_weights": dict(REFERENCE_WEIGHTS),
        "constraints": {
            "minimum": WEIGHT_MIN,
            "maximum": WEIGHT_MAX,
            "sum": WEIGHT_SUM,
        },
        "ridge_alpha": alpha,
        "min_train_opportunities": int(min_train_opportunities),
        "minimum_train_unit": "unique_decision_event_id_x_side",
        "purge_duration": purge.isoformat(),
        "models_fit": sum(model["status"] == "FIT" for model in models),
        "models_not_fit": sum(
            model["status"] == "NOT_FIT" for model in models
        ),
        "oos_opportunities": len(predictions),
        "oos_scored": sum(
            row["predicted_opportunity_r"] is not None
            for row in predictions
        ),
        "no_hard_threshold": True,
        "execution_side_effects_in_this_module": False,
        "risk_changed_by_score": False,
        "training_population": (
            "all supplied technical opportunities with mature train-only "
            "FILLED/NO_FILL labels; correlated triggers share one "
            "decision-side cluster weight; no production-bias selection"
        ),
        "oos_label_metric_horizon": (
            "actual labels are scored only when exit_time < test_end"
        ),
    }
    return {
        "summary": summary,
        "models": models,
        "predictions": predictions,
        "coefficients": coefficients,
        "metrics": metrics,
    }


__all__ = [
    "DEFAULT_MIN_TRAIN_OPPORTUNITIES",
    "DEFAULT_RIDGE_ALPHA",
    "REFERENCE_WEIGHTS",
    "WEIGHT_MAX",
    "WEIGHT_MIN",
    "WEIGHT_OPTIMIZER_SCHEMA",
    "WEIGHT_SUM",
    "WeightOptimizerError",
    "build_counterfactual_weight_scores",
]
