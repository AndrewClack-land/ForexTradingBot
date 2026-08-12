"""Leakage-aware conditional shadow scoring for attributed baseline setups.

This is intentionally not a replacement-weight optimizer.  It learns only
inside the candidate population admitted by the current +2/+1 formula and
never changes an entry, fill, stop, target, disposition, or setup count.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from core.narrative_scoring import FACTOR_DEFINITIONS


SHADOW_SCORE_SCHEMA = "narrative-shadow-score/v1"
PRIMARY_POLICY = "stop-first"
DEFAULT_RIDGE_ALPHA = 25.0
DEFAULT_MIN_TRAIN_SETUPS = 400
_TRIGGERS = (
    "h1_pivot_reclaim_15m",
    "absorption_15m",
    "order_block_1h",
    "rejection_block_15m",
    "turtle_soup_15m",
    "unknown",
)


def _timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _setup_sort_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("decision_time") or ""),
        str(row.get("exit_time") or ""),
        str(row.get("symbol") or ""),
        str(row.get("candidate_id") or ""),
        str(row.get("setup_id") or ""),
        str(row.get("policy") or ""),
    )


def _factor_maps(row: Mapping[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    vector = row.get("factor_vector")
    factors = (
        vector.get("factors", ())
        if isinstance(vector, Mapping)
        else ()
    )
    candidate_side = str(row.get("side") or "NEUTRAL").upper()
    alignments: dict[str, int] = {}
    presence: dict[str, int] = {}
    for item in factors:
        if not isinstance(item, Mapping) or not item.get("key"):
            continue
        key = str(item["key"])
        present = int(bool(item.get("present")))
        vote_side = str(item.get("vote_side") or "NEUTRAL").upper()
        if not present:
            alignment = 0
        elif vote_side == candidate_side:
            alignment = 1
        elif vote_side in {"LONG", "SHORT"}:
            alignment = -1
        else:
            alignment = 0
        alignments[key] = alignment
        presence[key] = present
    return alignments, presence


def _feature_names(symbols: Sequence[str]) -> tuple[str, ...]:
    normalized_symbols = tuple(sorted({str(symbol) for symbol in symbols}))
    names = [
        f"factor_alignment:{definition.key}"
        for definition in FACTOR_DEFINITIONS
    ]
    names.extend(
        f"factor_present:{definition.key}"
        for definition in FACTOR_DEFINITIONS
    )
    names.extend(
        (
            "side:LONG",
            *(f"symbol:{symbol}" for symbol in normalized_symbols[1:]),
            *(f"trigger:{trigger}" for trigger in _TRIGGERS[1:]),
            "fvg_alignment",
            "fvg_age_log_scaled",
            "fvg_age_missing",
            "vol_r_scaled",
            "vol_r_missing",
            "tp1_em_ratio_scaled",
            "tp1_em_ratio_missing",
        )
    )
    return tuple(names)


def _raw_number(
    row: Mapping[str, Any],
    field: str,
) -> tuple[float, int]:
    value = row.get(field)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0, 1
    if not math.isfinite(parsed):
        return 0.0, 1
    return parsed, 0


def _feature_vector(
    row: Mapping[str, Any],
    *,
    names: Sequence[str],
    vol_mean: float,
    vol_scale: float,
    ratio_mean: float,
    ratio_scale: float,
) -> np.ndarray:
    alignments, presence = _factor_maps(row)
    side = str(row.get("side") or "NEUTRAL").upper()
    symbol = str(row.get("symbol") or "")
    trigger = str(row.get("trigger_kind") or "unknown")
    vector = row.get("factor_vector")
    fvg_side = (
        str(vector.get("fvg_side") or "NEUTRAL").upper()
        if isinstance(vector, Mapping)
        else "NEUTRAL"
    )
    fvg_alignment = (
        1.0
        if fvg_side == side and side in {"LONG", "SHORT"}
        else -1.0
        if fvg_side in {"LONG", "SHORT"}
        else 0.0
    )
    vol_r, vol_missing = _raw_number(row, "vol_r")
    tp1_em_ratio, ratio_missing = _raw_number(
        row,
        "vol_tp1_em_ratio",
    )
    fvg_age, fvg_age_missing = _raw_number(row, "fvg_age_bars")
    fvg_age_log_scaled = (
        math.log1p(max(0.0, fvg_age)) / math.log1p(300.0)
    )
    values: dict[str, float] = {
        **{
            f"factor_alignment:{definition.key}": float(
                alignments.get(definition.key, 0)
            )
            for definition in FACTOR_DEFINITIONS
        },
        **{
            f"factor_present:{definition.key}": float(
                presence.get(definition.key, 0)
            )
            for definition in FACTOR_DEFINITIONS
        },
        "side:LONG": float(side == "LONG"),
        "fvg_alignment": fvg_alignment,
        "fvg_age_log_scaled": fvg_age_log_scaled,
        "fvg_age_missing": float(fvg_age_missing),
        "vol_r_scaled": (vol_r - vol_mean) / vol_scale,
        "vol_r_missing": float(vol_missing),
        "tp1_em_ratio_scaled": (
            (tp1_em_ratio - ratio_mean) / ratio_scale
        ),
        "tp1_em_ratio_missing": float(ratio_missing),
    }
    for name in names:
        if name.startswith("symbol:"):
            values[name] = float(symbol == name.split(":", 1)[1])
        elif name.startswith("trigger:"):
            values[name] = float(trigger == name.split(":", 1)[1])
    return np.asarray([values.get(name, 0.0) for name in names], dtype=float)


def _fit_ridge(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    design = np.column_stack((np.ones(len(matrix)), matrix))
    penalty = np.eye(design.shape[1], dtype=float) * float(alpha)
    penalty[0, 0] = 0.0
    return np.linalg.pinv(design.T @ design + penalty) @ design.T @ target


def _rank_correlation(predicted: np.ndarray, actual: np.ndarray) -> float | None:
    if len(predicted) < 3:
        return None
    pred_rank = pd.Series(predicted).rank(method="average").to_numpy()
    actual_rank = pd.Series(actual).rank(method="average").to_numpy()
    if np.std(pred_rank) == 0 or np.std(actual_rank) == 0:
        return None
    return float(np.corrcoef(pred_rank, actual_rank)[0, 1])


def _score_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fitted = [row for row in rows if row.get("model_status") == "FIT"]
    if not fitted:
        return {
            "setups": 0,
            "actual_expectancy_r": None,
            "predicted_mean_r": None,
            "mae_r": None,
            "rmse_r": None,
            "rank_ic": None,
            "top_minus_bottom_quintile_r": None,
        }
    predicted = np.asarray(
        [float(row["predicted_expected_r"]) for row in fitted],
        dtype=float,
    )
    actual = np.asarray(
        [float(row["actual_net_r"]) for row in fitted],
        dtype=float,
    )
    error = predicted - actual
    spread: float | None = None
    if len(fitted) >= 20 and np.ptp(predicted) > 0:
        order = np.argsort(predicted, kind="stable")
        quintiles = np.array_split(order, 5)
        spread = float(
            np.mean(actual[quintiles[-1]])
            - np.mean(actual[quintiles[0]])
        )
    return {
        "setups": len(fitted),
        "actual_expectancy_r": float(np.mean(actual)),
        "predicted_mean_r": float(np.mean(predicted)),
        "mae_r": float(np.mean(np.abs(error))),
        "rmse_r": float(np.sqrt(np.mean(error**2))),
        "rank_ic": _rank_correlation(predicted, actual),
        "top_minus_bottom_quintile_r": spread,
    }


def _metric_rows(
    predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        groups[("overall", "ALL")].append(row)
        groups[("fold", str(row.get("fold_index")))].append(row)
        groups[("symbol", str(row.get("symbol")))].append(row)
        groups[("side", str(row.get("side")))].append(row)
        groups[("trigger", str(row.get("trigger_kind")))].append(row)
    return [
        {
            "schema": SHADOW_SCORE_SCHEMA,
            "dimension": dimension,
            "dimension_value": value,
            **_score_metrics(rows),
        }
        for (dimension, value), rows in sorted(groups.items())
    ]


def build_shadow_scores(
    *,
    setups: Sequence[Mapping[str, Any]],
    periods: Sequence[Mapping[str, Any]],
    symbols: Sequence[str],
    entry_ttl: pd.Timedelta,
    max_holding: pd.Timedelta,
    primary_policy: str = PRIMARY_POLICY,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    min_train_setups: int = DEFAULT_MIN_TRAIN_SETUPS,
) -> dict[str, Any]:
    """Fit rolling shadow models only on earlier, fully matured labels."""

    names = _feature_names(symbols)
    primary_rows = sorted(
        (
            row
            for row in setups
            if str(row.get("policy")) == primary_policy
            and isinstance(row.get("factor_vector"), Mapping)
            and row.get("status") == "CLOSED"
            and row.get("net_r") is not None
        ),
        key=_setup_sort_key,
    )
    models: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    purge = pd.Timedelta(entry_ttl) + pd.Timedelta(max_holding)

    for period in sorted(periods, key=lambda item: int(item["fold_index"])):
        fold_index = int(period["fold_index"])
        test_start = _timestamp(period["test_start"])
        test_end = _timestamp(period["test_end"])
        train_start_raw = period.get("train_start")
        train_start = (
            _timestamp(train_start_raw)
            if train_start_raw is not None
            else min(
                (_timestamp(row["decision_time"]) for row in primary_rows),
                default=test_start,
            )
        )
        purge_cutoff = test_start - purge
        train_rows = [
            row
            for row in primary_rows
            if train_start <= _timestamp(row["decision_time"]) < purge_cutoff
            and _timestamp(row["exit_time"]) < test_start
            and int(row.get("fold_index") or 0) < fold_index
        ]
        test_rows = [
            row
            for row in primary_rows
            if int(row.get("fold_index") or 0) == fold_index
            and test_start <= _timestamp(row["decision_time"]) < test_end
        ]
        status = (
            "FIT" if len(train_rows) >= int(min_train_setups) else "NOT_FIT"
        )
        model: dict[str, Any] = {
            "schema": SHADOW_SCORE_SCHEMA,
            "fold_index": fold_index,
            "status": status,
            "training_source": "previous_baseline_oos_only",
            "target": "net_r_given_production_fill",
            "train_start": train_start.isoformat(),
            "purge_cutoff": purge_cutoff.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "train_setups": len(train_rows),
            "test_setups": len(test_rows),
            "min_train_setups": int(min_train_setups),
            "ridge_alpha": float(ridge_alpha),
            "feature_names": list(names),
        }
        if status == "NOT_FIT":
            model["model_id"] = None
            models.append(model)
            predictions.extend(
                {
                    "schema": SHADOW_SCORE_SCHEMA,
                    "setup_id": row.get("setup_id"),
                    "candidate_id": row.get("candidate_id"),
                    "fold_index": fold_index,
                    "symbol": row.get("symbol"),
                    "decision_time": row.get("decision_time"),
                    "side": row.get("side"),
                    "trigger_kind": row.get("trigger_kind"),
                    "policy": primary_policy,
                    "model_status": status,
                    "model_id": None,
                    "train_setups": len(train_rows),
                    "predicted_expected_r": None,
                    "actual_net_r": row.get("net_r"),
                    "prediction_error_r": None,
                }
                for row in test_rows
            )
            continue

        train_vol = np.asarray(
            [_raw_number(row, "vol_r")[0] for row in train_rows]
        )
        vol_mean = float(np.mean(train_vol))
        vol_scale = float(np.std(train_vol))
        if not math.isfinite(vol_scale) or vol_scale < 1e-12:
            vol_scale = 1.0
        train_ratio = np.asarray(
            [
                _raw_number(row, "vol_tp1_em_ratio")[0]
                for row in train_rows
            ]
        )
        ratio_mean = float(np.mean(train_ratio))
        ratio_scale = float(np.std(train_ratio))
        if not math.isfinite(ratio_scale) or ratio_scale < 1e-12:
            ratio_scale = 1.0
        matrix = np.vstack(
            [
                _feature_vector(
                    row,
                    names=names,
                    vol_mean=vol_mean,
                    vol_scale=vol_scale,
                    ratio_mean=ratio_mean,
                    ratio_scale=ratio_scale,
                )
                for row in train_rows
            ]
        )
        target = np.asarray(
            [float(row["net_r"]) for row in train_rows],
            dtype=float,
        )
        coefficients = _fit_ridge(
            matrix,
            target,
            alpha=float(ridge_alpha),
        )
        model_identity = {
            "schema": SHADOW_SCORE_SCHEMA,
            "fold_index": fold_index,
            "train_setup_ids": sorted(
                str(row.get("setup_id")) for row in train_rows
            ),
            "feature_names": list(names),
            "coefficients": [float(value) for value in coefficients],
            "vol_mean": vol_mean,
            "vol_scale": vol_scale,
            "ratio_mean": ratio_mean,
            "ratio_scale": ratio_scale,
            "ridge_alpha": float(ridge_alpha),
        }
        model_id = _canonical_hash(model_identity)[:24]
        model.update(
            {
                "model_id": model_id,
                "vol_mean": vol_mean,
                "vol_scale": vol_scale,
                "ratio_mean": ratio_mean,
                "ratio_scale": ratio_scale,
                "intercept": float(coefficients[0]),
                "coefficients": {
                    name: float(value)
                    for name, value in zip(names, coefficients[1:])
                },
            }
        )
        models.append(model)
        coefficient_rows.extend(
            {
                "schema": SHADOW_SCORE_SCHEMA,
                "model_id": model_id,
                "fold_index": fold_index,
                "train_start": train_start.isoformat(),
                "purge_cutoff": purge_cutoff.isoformat(),
                "test_start": test_start.isoformat(),
                "train_setups": len(train_rows),
                "ridge_alpha": float(ridge_alpha),
                "feature": feature,
                "coefficient": float(value),
            }
            for feature, value in (
                ("intercept", coefficients[0]),
                *zip(names, coefficients[1:]),
            )
        )
        for row in test_rows:
            features = _feature_vector(
                row,
                names=names,
                vol_mean=vol_mean,
                vol_scale=vol_scale,
                ratio_mean=ratio_mean,
                ratio_scale=ratio_scale,
            )
            predicted = float(coefficients[0] + features @ coefficients[1:])
            predictions.append(
                {
                    "schema": SHADOW_SCORE_SCHEMA,
                    "setup_id": row.get("setup_id"),
                    "candidate_id": row.get("candidate_id"),
                    "fold_index": fold_index,
                    "symbol": row.get("symbol"),
                    "decision_time": row.get("decision_time"),
                    "side": row.get("side"),
                    "trigger_kind": row.get("trigger_kind"),
                    "policy": primary_policy,
                    "model_status": status,
                    "model_id": model_id,
                    "train_setups": len(train_rows),
                    "predicted_expected_r": predicted,
                    "actual_net_r": row.get("net_r"),
                    "prediction_error_r": predicted - float(row["net_r"]),
                }
            )

    metrics = _metric_rows(predictions)
    fitted = [row for row in predictions if row["model_status"] == "FIT"]
    summary = {
        "schema": SHADOW_SCORE_SCHEMA,
        "mode": "conditional-shadow-soft-score",
        "primary_label_policy": primary_policy,
        "target": "net_r_given_production_fill",
        "execution_changed_by_score": False,
        "no_hard_rejection": True,
        "candidate_ids_changed": False,
        "setups_changed": False,
        "risk_changed": False,
        "ridge_alpha": float(ridge_alpha),
        "min_train_setups": int(min_train_setups),
        "purge_duration": purge.isoformat(),
        "models_fit": sum(model["status"] == "FIT" for model in models),
        "models_not_fit": sum(
            model["status"] == "NOT_FIT" for model in models
        ),
        "oos_predictions": len(fitted),
        "oos_not_scored": len(predictions) - len(fitted),
        "overall_metrics": _score_metrics(predictions),
        "selection_bias_warning": (
            "The model is trained only on prior OOS raw ENTER signals "
            "(baseline bias plus a detected trigger) that then survived "
            "execution gates and filled. Its coefficients estimate net R "
            "conditional on that selected population, not unbiased "
            "replacement weights."
        ),
    }
    return {
        "summary": summary,
        "models": models,
        "predictions": predictions,
        "coefficients": coefficient_rows,
        "metrics": metrics,
    }
