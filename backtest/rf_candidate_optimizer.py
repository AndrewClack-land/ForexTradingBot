"""Causal, offline Random-Forest training for shadow entry candidates.

This module is deliberately research-only.  It imports scikit-learn only
inside the fitting function and exports the portable JSON tree format consumed
by :mod:`core.rf_candidate_contract`.  It never promotes a model or changes a
live decision.

The target contract is intentionally narrow: one production take-profit per
filled candidate.  Fill is learned on all resolved candidates; TP and gross R
are learned only after fill.  Costs are supplied causally per candidate and are
subtracted exactly once outside the intrinsic forests.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import re
from statistics import fmean
from typing import Any, Literal, Optional

from core.rf_candidate_contract import (
    RF_CANDIDATE_PROFILE_SCHEMA,
    FrozenForest,
    RFCandidateProfile,
    default_rf_feature_registry,
    seal_rf_candidate_profile,
    tree_dispersion_penalized_score,
)


RF_CANDIDATE_OPTIMIZER_SCHEMA = "rf-candidate-optimizer-v1"
SINGLE_TP_TARGET_CONTRACT = "production-single-tp-v1"

_CATEGORY_PREFIXES = (
    "symbol:",
    "trigger:",
    "session:",
    "weekday:",
    "execution:",
    "fvg_state:",
    "orca_mode:",
)
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class CandidateRFTrainingError(ValueError):
    """Raised when a causal candidate profile cannot be trained safely."""


def _utc(value: Any, *, name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise CandidateRFTrainingError(
                f"{name} must be a timezone-aware ISO timestamp"
            ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CandidateRFTrainingError(f"{name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CandidateRFTrainingError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise CandidateRFTrainingError(f"{name} must be finite")
    return parsed


def _positive_duration(value: timedelta, *, name: str) -> None:
    if not isinstance(value, timedelta) or value <= timedelta(0):
        raise CandidateRFTrainingError(f"{name} must be a positive timedelta")


@dataclass(frozen=True, kw_only=True)
class CandidateTargetConfig:
    """Production-compatible label contract for one take-profit."""

    contract: str = SINGLE_TP_TARGET_CONTRACT
    tp_reward_r: float = 1.2

    def __post_init__(self) -> None:
        if self.contract != SINGLE_TP_TARGET_CONTRACT:
            raise CandidateRFTrainingError(
                f"target contract must be {SINGLE_TP_TARGET_CONTRACT!r}"
            )
        reward = _finite(self.tp_reward_r, name="tp_reward_r")
        if reward <= 0.0:
            raise CandidateRFTrainingError("tp_reward_r must be positive")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.contract,
            "tp_count": 1,
            "tp_reward_r": float(self.tp_reward_r),
            "fill_label": "resolved candidate filled before entry expiry",
            "tp_label": "sole production TP hit after fill",
            "gross_r_label": "realized gross R after fill and before costs",
        }


@dataclass(frozen=True, kw_only=True)
class CandidateForestConfig:
    n_estimators: int = 200
    max_depth: Optional[int] = 6
    min_samples_leaf: int = 30
    min_samples_split: int = 60
    max_features: str | int | float | None = "sqrt"
    class_weight: Optional[str] = "balanced_subsample"
    random_state: int = 1729
    n_jobs: int = 1

    def __post_init__(self) -> None:
        for name in ("n_estimators", "min_samples_leaf", "min_samples_split"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise CandidateRFTrainingError(f"{name} must be a positive integer")
        if self.min_samples_split < 2:
            raise CandidateRFTrainingError("min_samples_split must be at least 2")
        if self.max_depth is not None and (
            isinstance(self.max_depth, bool)
            or not isinstance(self.max_depth, int)
            or self.max_depth <= 0
        ):
            raise CandidateRFTrainingError("max_depth must be positive or None")
        if isinstance(self.random_state, bool) or not isinstance(
            self.random_state, int
        ):
            raise CandidateRFTrainingError("random_state must be an integer")
        if self.n_jobs != 1:
            raise CandidateRFTrainingError(
                "n_jobs must be 1 for reproducible portable profiles"
            )


@dataclass(frozen=True, kw_only=True)
class CandidateWalkForwardConfig:
    """Time-based expanding/rolling walk-forward with two causal cutoffs.

    ``purge`` limits training decision timestamps.  ``embargo`` independently
    limits when a training label may have become known.  Both must cover the
    complete entry-TTL plus maximum-hold horizon.
    """

    entry_ttl: timedelta
    max_hold: timedelta
    purge: timedelta
    embargo: timedelta
    train_span: timedelta = timedelta(days=730)
    test_span: timedelta = timedelta(days=180)
    step_span: timedelta = timedelta(days=180)
    mode: Literal["expanding", "rolling"] = "expanding"
    min_train_events: int = 30
    min_test_events: int = 5

    def __post_init__(self) -> None:
        for name in (
            "entry_ttl",
            "max_hold",
            "purge",
            "embargo",
            "train_span",
            "test_span",
            "step_span",
        ):
            _positive_duration(getattr(self, name), name=name)
        horizon = self.entry_ttl + self.max_hold
        if self.purge < horizon:
            raise CandidateRFTrainingError("purge must be >= entry_ttl + max_hold")
        if self.embargo < horizon:
            raise CandidateRFTrainingError("embargo must be >= entry_ttl + max_hold")
        if self.step_span < self.test_span:
            raise CandidateRFTrainingError(
                "step_span must be >= test_span to prevent overlapping OOS rows"
            )
        if self.mode not in {"expanding", "rolling"}:
            raise CandidateRFTrainingError("mode must be expanding or rolling")
        for name in ("min_train_events", "min_test_events"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise CandidateRFTrainingError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class CandidateWalkForwardFold:
    index: int
    train_start_utc: datetime
    train_decision_cutoff_utc: datetime
    train_label_cutoff_utc: datetime
    test_start_utc: datetime
    test_end_utc: datetime

    def to_mapping(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "train_start_utc": _iso(self.train_start_utc),
            "train_decision_cutoff_utc": _iso(self.train_decision_cutoff_utc),
            "train_label_cutoff_utc": _iso(self.train_label_cutoff_utc),
            "test_start_utc": _iso(self.test_start_utc),
            "test_end_utc": _iso(self.test_end_utc),
        }


@dataclass(frozen=True)
class CandidateRFTrainingResult:
    profile: Mapping[str, Any]
    folds: tuple[Mapping[str, Any], ...]
    oos_predictions: tuple[Mapping[str, Any], ...]
    aggregate_metrics: Mapping[str, Any]
    excluded_counts: Mapping[str, int]

    @property
    def auto_promoted(self) -> bool:
        return False

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": RF_CANDIDATE_OPTIMIZER_SCHEMA,
            "profile": dict(self.profile),
            "folds": [dict(fold) for fold in self.folds],
            "oos_predictions": [dict(row) for row in self.oos_predictions],
            "aggregate_metrics": dict(self.aggregate_metrics),
            "excluded_counts": dict(self.excluded_counts),
            "auto_promotion": {
                "enabled": False,
                "reason": "manual champion/challenger review is required",
            },
        }


@dataclass(frozen=True)
class _CandidateRow:
    event_id: str
    candidate_id: str
    decision_time: datetime
    features_known_at: datetime
    label_known_at: datetime
    exit_time: Optional[datetime]
    filled: bool
    tp_hit: Optional[bool]
    gross_r: Optional[float]
    causal_cost_r: float
    sequence_index: Optional[int]
    feature_values: Mapping[str, float]


def _status(raw: Mapping[str, Any], *, index: int) -> str:
    value = str(raw.get("outcome_status") or "").strip().upper()
    if value not in {"RESOLVED", "CENSORED", "INVALID"}:
        raise CandidateRFTrainingError(
            f"rows[{index}].outcome_status must be RESOLVED, CENSORED, or INVALID"
        )
    return value


def _feature_payload(
    raw: Mapping[str, Any], *, index: int
) -> tuple[Mapping[str, Any], Any]:
    payload = raw.get("feature_values")
    known_at = raw.get("features_known_at_utc")
    nested = raw.get("features")
    if payload is None and isinstance(nested, Mapping):
        payload = nested.get("values")
        known_at = known_at or nested.get("decision_time_utc")
    if not isinstance(payload, Mapping) or not payload:
        raise CandidateRFTrainingError(
            f"rows[{index}].feature_values must be a non-empty mapping"
        )
    return payload, known_at


def _valid_feature_name(name: str) -> bool:
    defaults = set(default_rf_feature_registry())
    if name in defaults:
        return True
    for prefix in _CATEGORY_PREFIXES:
        if name.startswith(prefix):
            return bool(_TOKEN_RE.fullmatch(name[len(prefix) :]))
    if name.startswith(("factor_alignment:", "factor_present:")):
        return bool(_TOKEN_RE.fullmatch(name.partition(":")[2]))
    return False


def _parse_feature_values(
    payload: Mapping[str, Any], *, index: int
) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw_name, raw_value in payload.items():
        name = str(raw_name)
        if not _valid_feature_name(name):
            raise CandidateRFTrainingError(
                f"rows[{index}] contains unsupported feature {name!r}"
            )
        values[name] = _finite(
            raw_value, name=f"rows[{index}].feature_values[{name!r}]"
        )
    for prefix in _CATEGORY_PREFIXES:
        active = [
            name
            for name, value in values.items()
            if name.startswith(prefix) and value != 0.0
        ]
        if any(values[name] not in {0.0, 1.0} for name in active):
            raise CandidateRFTrainingError(
                f"rows[{index}] category {prefix!r} must be one-hot"
            )
        if len(active) > 1:
            raise CandidateRFTrainingError(
                f"rows[{index}] category {prefix!r} has multiple active values"
            )
    return values


def _parse_resolved_row(
    raw: Mapping[str, Any],
    *,
    index: int,
    target: CandidateTargetConfig,
) -> _CandidateRow:
    event_id = str(raw.get("decision_event_id") or "").strip()
    candidate_id = str(raw.get("candidate_id") or "").strip()
    if not event_id or not candidate_id:
        raise CandidateRFTrainingError(
            f"rows[{index}] requires decision_event_id and candidate_id"
        )
    decision = _utc(
        raw.get("decision_time_utc"),
        name=f"rows[{index}].decision_time_utc",
    )
    payload, known_raw = _feature_payload(raw, index=index)
    known = _utc(known_raw, name=f"rows[{index}].features_known_at_utc")
    if known > decision:
        raise CandidateRFTrainingError(
            f"rows[{index}] features were not known by the decision"
        )
    label_known = _utc(
        raw.get("label_known_at_utc"),
        name=f"rows[{index}].label_known_at_utc",
    )
    if label_known <= decision:
        raise CandidateRFTrainingError(
            f"rows[{index}] label must become known after the decision"
        )
    if raw.get("target_contract") != target.contract:
        raise CandidateRFTrainingError(
            f"rows[{index}] must declare target_contract={target.contract!r}"
        )
    if raw.get("tp_count", 1) != 1:
        raise CandidateRFTrainingError(
            f"rows[{index}] violates the single-TP target contract"
        )
    if "tp_reward_r" in raw and not math.isclose(
        _finite(raw["tp_reward_r"], name=f"rows[{index}].tp_reward_r"),
        float(target.tp_reward_r),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise CandidateRFTrainingError(
            f"rows[{index}] tp_reward_r differs from the target contract"
        )
    filled = raw.get("filled")
    if not isinstance(filled, bool):
        raise CandidateRFTrainingError(f"rows[{index}].filled must be boolean")
    exit_raw = raw.get("exit_time_utc")
    tp_hit = raw.get("tp_hit")
    gross_raw = raw.get("gross_r")
    exit_time: Optional[datetime]
    gross_r: Optional[float]
    if filled:
        exit_time = _utc(exit_raw, name=f"rows[{index}].exit_time_utc")
        if not decision < exit_time <= label_known:
            raise CandidateRFTrainingError(
                f"rows[{index}] exit_time must be after decision and no later than label_known"
            )
        if not isinstance(tp_hit, bool):
            raise CandidateRFTrainingError(
                f"rows[{index}].tp_hit must be boolean after fill"
            )
        gross_r = _finite(gross_raw, name=f"rows[{index}].gross_r")
    else:
        if exit_raw is not None or tp_hit is not None or gross_raw is not None:
            raise CandidateRFTrainingError(
                f"rows[{index}] non-fill cannot carry exit, TP, or gross-R labels"
            )
        exit_time = None
        tp_hit = None
        gross_r = None
    cost = _finite(raw.get("causal_cost_r"), name=f"rows[{index}].causal_cost_r")
    if cost < 0.0:
        raise CandidateRFTrainingError(
            f"rows[{index}].causal_cost_r must be nonnegative"
        )
    sequence = raw.get("sequence_index")
    if sequence is not None:
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise CandidateRFTrainingError(
                f"rows[{index}].sequence_index must be an integer"
            )
    return _CandidateRow(
        event_id=event_id,
        candidate_id=candidate_id,
        decision_time=decision,
        features_known_at=known,
        label_known_at=label_known,
        exit_time=exit_time,
        filled=filled,
        tp_hit=tp_hit,
        gross_r=gross_r,
        causal_cost_r=cost,
        sequence_index=sequence,
        feature_values=_parse_feature_values(payload, index=index),
    )


def _validate_excluded_row(raw: Mapping[str, Any], *, index: int) -> None:
    decision = _utc(
        raw.get("decision_time_utc"),
        name=f"rows[{index}].decision_time_utc",
    )
    known_raw = raw.get("features_known_at_utc")
    nested = raw.get("features")
    if known_raw is None and isinstance(nested, Mapping):
        known_raw = nested.get("decision_time_utc")
    if (
        known_raw is not None
        and _utc(known_raw, name=f"rows[{index}].features_known_at_utc") > decision
    ):
        raise CandidateRFTrainingError(
            f"rows[{index}] excluded row has future-dated features"
        )


def _prepare_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: CandidateTargetConfig,
) -> tuple[list[_CandidateRow], Counter[str]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise CandidateRFTrainingError("rows must be a sequence")
    resolved: list[_CandidateRow] = []
    excluded: Counter[str] = Counter()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise CandidateRFTrainingError(f"rows[{index}] must be a mapping")
        status = _status(raw, index=index)
        if status != "RESOLVED":
            _validate_excluded_row(raw, index=index)
            excluded[status.lower()] += 1
            continue
        resolved.append(_parse_resolved_row(raw, index=index, target=target))
    if not resolved:
        raise CandidateRFTrainingError("no resolved candidate rows remain")
    seen: set[tuple[str, str]] = set()
    event_context: dict[str, tuple[datetime, Optional[int]]] = {}
    sequence_events: dict[int, str] = {}
    for row in resolved:
        key = (row.event_id, row.candidate_id)
        if key in seen:
            raise CandidateRFTrainingError(
                f"duplicate candidate {row.candidate_id!r} in event {row.event_id!r}"
            )
        seen.add(key)
        context = (row.decision_time, row.sequence_index)
        previous = event_context.setdefault(row.event_id, context)
        if previous != context:
            raise CandidateRFTrainingError(
                f"event {row.event_id!r} has inconsistent decision time or sequence"
            )
        if row.sequence_index is not None:
            owner = sequence_events.setdefault(row.sequence_index, row.event_id)
            if owner != row.event_id:
                raise CandidateRFTrainingError(
                    "sequence_index must identify exactly one decision event"
                )
    return sorted(
        resolved,
        key=lambda row: (row.decision_time, row.event_id, row.candidate_id),
    ), excluded


def _event_count(rows: Sequence[_CandidateRow]) -> int:
    return len({row.event_id for row in rows})


def resolve_candidate_walk_forward_folds(
    rows: Sequence[_CandidateRow],
    config: CandidateWalkForwardConfig,
) -> tuple[CandidateWalkForwardFold, ...]:
    """Resolve non-overlapping OOS windows from already validated rows."""

    first = min(row.decision_time for row in rows)
    last = max(row.decision_time for row in rows)
    gap = max(config.purge, config.embargo)
    test_start = first + config.train_span + gap
    folds: list[CandidateWalkForwardFold] = []
    while test_start <= last:
        decision_cutoff = test_start - config.purge
        label_cutoff = test_start - config.embargo
        train_start = (
            first if config.mode == "expanding" else decision_cutoff - config.train_span
        )
        test_end = test_start + config.test_span
        train = [
            row
            for row in rows
            if train_start <= row.decision_time < decision_cutoff
            and row.label_known_at <= label_cutoff
        ]
        test = [row for row in rows if test_start <= row.decision_time < test_end]
        if (
            _event_count(train) >= config.min_train_events
            and _event_count(test) >= config.min_test_events
            and any(row.filled for row in train)
        ):
            folds.append(
                CandidateWalkForwardFold(
                    index=len(folds),
                    train_start_utc=train_start,
                    train_decision_cutoff_utc=decision_cutoff,
                    train_label_cutoff_utc=label_cutoff,
                    test_start_utc=test_start,
                    test_end_utc=test_end,
                )
            )
        test_start += config.step_span
    if not folds:
        raise CandidateRFTrainingError(
            "no walk-forward fold satisfies the causal row and event minima"
        )
    return tuple(folds)


def _fold_rows(
    rows: Sequence[_CandidateRow], fold: CandidateWalkForwardFold
) -> tuple[list[_CandidateRow], list[_CandidateRow]]:
    train = [
        row
        for row in rows
        if fold.train_start_utc <= row.decision_time < fold.train_decision_cutoff_utc
        and row.label_known_at <= fold.train_label_cutoff_utc
    ]
    test = [
        row
        for row in rows
        if fold.test_start_utc <= row.decision_time < fold.test_end_utc
    ]
    return train, test


def _category_prefix(name: str) -> Optional[str]:
    return next(
        (prefix for prefix in _CATEGORY_PREFIXES if name.startswith(prefix)), None
    )


def _build_feature_registry(rows: Sequence[_CandidateRow]) -> tuple[str, ...]:
    numeric: set[str] = set()
    category_tokens: dict[str, set[str]] = defaultdict(set)
    category_present: set[str] = set()
    for row in rows:
        for name, value in row.feature_values.items():
            prefix = _category_prefix(name)
            if prefix is None:
                numeric.add(name)
            else:
                category_present.add(prefix)
                if value == 1.0:
                    category_tokens[prefix].add(name[len(prefix) :])
    registry = sorted(numeric)
    for prefix in _CATEGORY_PREFIXES:
        if prefix not in category_present:
            continue
        registry.extend(
            f"{prefix}{token}"
            for token in sorted(
                token for token in category_tokens[prefix] if token != "__OTHER__"
            )
        )
        registry.append(f"{prefix}__OTHER__")
    if not registry:
        raise CandidateRFTrainingError("training rows have no causal features")
    return tuple(registry)


def _vector(row: _CandidateRow, registry: Sequence[str]) -> list[float]:
    vector: list[float] = []
    known_by_prefix: dict[str, set[str]] = defaultdict(set)
    for name in registry:
        prefix = _category_prefix(name)
        if prefix is not None and not name.endswith(":__OTHER__"):
            known_by_prefix[prefix].add(name[len(prefix) :])
    actual_by_prefix: dict[str, Optional[str]] = {}
    for prefix in _CATEGORY_PREFIXES:
        actual = [
            name[len(prefix) :]
            for name, value in row.feature_values.items()
            if name.startswith(prefix) and value == 1.0
        ]
        actual_by_prefix[prefix] = actual[0] if actual else None
    for name in registry:
        prefix = _category_prefix(name)
        if prefix is None:
            vector.append(float(row.feature_values.get(name, 0.0)))
            continue
        token = name[len(prefix) :]
        actual = actual_by_prefix[prefix]
        if token == "__OTHER__":
            vector.append(
                float(actual is not None and actual not in known_by_prefix[prefix])
            )
        else:
            vector.append(float(actual == token))
    return vector


def _sample_weights(rows: Sequence[_CandidateRow]) -> list[float]:
    counts = Counter(row.event_id for row in rows)
    return [1.0 / counts[row.event_id] for row in rows]


def _weight_audit(rows: Sequence[_CandidateRow]) -> dict[str, Any]:
    totals: dict[str, float] = defaultdict(float)
    for row, weight in zip(rows, _sample_weights(rows)):
        totals[row.event_id] += weight
    deviations = [abs(total - 1.0) for total in totals.values()]
    return {
        "event_count": len(totals),
        "total_weight": float(sum(totals.values())),
        "max_event_weight_error": max(deviations, default=0.0),
    }


@dataclass(frozen=True)
class _FittedHeads:
    fill: Any
    tp_given_fill: Any
    gross_r_given_fill: Any


def _fit_heads(
    rows: Sequence[_CandidateRow],
    registry: Sequence[str],
    config: CandidateForestConfig,
    *,
    seed_offset: int,
) -> _FittedHeads:
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise CandidateRFTrainingError(
            "scikit-learn is required only in the offline backtest environment"
        ) from exc
    filled = [row for row in rows if row.filled]
    if not filled:
        raise CandidateRFTrainingError("TP and gross-R heads require filled rows")
    common = {
        "n_estimators": config.n_estimators,
        "max_depth": config.max_depth,
        "min_samples_leaf": config.min_samples_leaf,
        "min_samples_split": config.min_samples_split,
        "max_features": config.max_features,
        "n_jobs": config.n_jobs,
    }
    fill_model = RandomForestClassifier(
        **common,
        class_weight=config.class_weight,
        random_state=config.random_state + seed_offset,
    )
    tp_model = RandomForestClassifier(
        **common,
        class_weight=config.class_weight,
        random_state=config.random_state + seed_offset + 1,
    )
    gross_model = RandomForestRegressor(
        **common,
        random_state=config.random_state + seed_offset + 2,
    )
    fill_model.fit(
        [_vector(row, registry) for row in rows],
        [int(row.filled) for row in rows],
        sample_weight=_sample_weights(rows),
    )
    tp_model.fit(
        [_vector(row, registry) for row in filled],
        [int(bool(row.tp_hit)) for row in filled],
        sample_weight=_sample_weights(filled),
    )
    gross_model.fit(
        [_vector(row, registry) for row in filled],
        [float(row.gross_r) for row in filled],
        sample_weight=_sample_weights(filled),
    )
    return _FittedHeads(fill_model, tp_model, gross_model)


def _positive_probability(model: Any, matrix: Sequence[Sequence[float]]) -> list[float]:
    probabilities = model.predict_proba(matrix)
    classes = [int(value) for value in model.classes_]
    if 1 not in classes:
        return [0.0] * len(matrix)
    column = classes.index(1)
    return [float(row[column]) for row in probabilities]


def _portable_classifier_forest(model: Any) -> dict[str, Any]:
    classes = [int(value) for value in model.classes_]
    positive_index = classes.index(1) if 1 in classes else None
    trees = []
    for estimator in model.estimators_:
        source = estimator.tree_
        feature_index = []
        threshold = []
        left_child = []
        right_child = []
        values = []
        for node in range(source.node_count):
            leaf = int(source.children_left[node]) == -1
            feature_index.append(-1 if leaf else int(source.feature[node]))
            threshold.append(0.0 if leaf else float(source.threshold[node]))
            left_child.append(-1 if leaf else int(source.children_left[node]))
            right_child.append(-1 if leaf else int(source.children_right[node]))
            if leaf:
                masses = [float(value) for value in source.value[node].reshape(-1)]
                total = sum(masses)
                probability = (
                    masses[positive_index] / total
                    if positive_index is not None and total > 0.0
                    else 0.0
                )
                values.append(float(probability))
            else:
                values.append(0.0)
        trees.append(
            {
                "feature_index": feature_index,
                "threshold": threshold,
                "left_child": left_child,
                "right_child": right_child,
                "value": values,
            }
        )
    return {"trees": trees}


def _portable_regression_forest(model: Any) -> dict[str, Any]:
    trees = []
    for estimator in model.estimators_:
        source = estimator.tree_
        feature_index = []
        threshold = []
        left_child = []
        right_child = []
        values = []
        for node in range(source.node_count):
            leaf = int(source.children_left[node]) == -1
            feature_index.append(-1 if leaf else int(source.feature[node]))
            threshold.append(0.0 if leaf else float(source.threshold[node]))
            left_child.append(-1 if leaf else int(source.children_left[node]))
            right_child.append(-1 if leaf else int(source.children_right[node]))
            values.append(float(source.value[node].reshape(-1)[0]) if leaf else 0.0)
        trees.append(
            {
                "feature_index": feature_index,
                "threshold": threshold,
                "left_child": left_child,
                "right_child": right_child,
                "value": values,
            }
        )
    return {"trees": trees}


def _auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _brier(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    if not labels:
        return None
    return fmean((float(label) - score) ** 2 for label, score in zip(labels, scores))


def _prediction_rows(
    rows: Sequence[_CandidateRow],
    registry: Sequence[str],
    heads: _FittedHeads,
    *,
    fold_index: int,
    uncertainty_z: float,
) -> list[dict[str, Any]]:
    matrix = [_vector(row, registry) for row in rows]
    feature_count = len(registry)
    fill_forest = FrozenForest.from_mapping(
        _portable_classifier_forest(heads.fill),
        feature_count=feature_count,
        probability=True,
        name="fold.fill_probability",
    )
    tp_forest = FrozenForest.from_mapping(
        _portable_classifier_forest(heads.tp_given_fill),
        feature_count=feature_count,
        probability=True,
        name="fold.tp_given_fill_probability",
    )
    gross_forest = FrozenForest.from_mapping(
        _portable_regression_forest(heads.gross_r_given_fill),
        feature_count=feature_count,
        probability=False,
        name="fold.gross_r_given_fill",
    )
    predictions = []
    for row, vector in zip(rows, matrix):
        fill_trees = fill_forest.predictions(vector)
        tp_trees = tp_forest.predictions(vector)
        gross_trees = gross_forest.predictions(vector)
        fill_probability = fmean(fill_trees)
        tp_probability = fmean(tp_trees)
        score = tree_dispersion_penalized_score(
            fill_trees,
            gross_trees,
            causal_cost_r=row.causal_cost_r,
            penalty_multiplier=uncertainty_z,
        )
        expected_gross = score["expected_gross_r_given_fill"]
        expected_net = score["expected_net_r"]
        realized_net = float(row.gross_r) - row.causal_cost_r if row.filled else 0.0
        predictions.append(
            {
                "fold_index": fold_index,
                "decision_event_id": row.event_id,
                "candidate_id": row.candidate_id,
                "decision_time_utc": _iso(row.decision_time),
                "sequence_index": row.sequence_index,
                "filled": row.filled,
                "tp_hit": row.tp_hit,
                "causal_cost_r": row.causal_cost_r,
                "realized_net_r": realized_net,
                "fill_probability": fill_probability,
                "tp_given_fill_probability": tp_probability,
                "tp_probability": fill_probability * tp_probability,
                "expected_gross_r_given_fill": expected_gross,
                "expected_net_r": expected_net,
                "uncertainty_z": uncertainty_z,
                "tree_dispersion_penalty_multiplier": uncertainty_z,
                "tree_dispersion_r": score["tree_dispersion_r"],
                "tree_dispersion_penalty_r": score["tree_dispersion_penalty_r"],
                "conservative_intrinsic_score_r": score["dispersion_penalized_score_r"],
            }
        )
    return _annotate_event_policy(predictions)


def _annotate_event_policy(
    predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    annotated = [dict(row) for row in predictions]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotated:
        grouped[str(row["decision_event_id"])].append(row)
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
        select = float(ordered[0]["conservative_intrinsic_score_r"]) > 0.0
        event_status = "SELECTED" if select else "ABSTAINED_NONPOSITIVE_SCORE"
        for rank, row in enumerate(ordered, start=1):
            row["policy_rank"] = rank
            row["policy_selected"] = bool(select and rank == 1)
            row["policy_event_status"] = event_status
    return annotated


def _max_drawdown(rows: Sequence[Mapping[str, Any]]) -> Optional[float]:
    if not rows or not all(row.get("sequence_index") is not None for row in rows):
        return None
    ordered = sorted(
        rows,
        key=lambda row: (int(row["sequence_index"]), str(row["candidate_id"])),
    )
    cumulative = 0.0
    peak = 0.0
    drawdown = 0.0
    for row in ordered:
        cumulative += float(row["realized_net_r"])
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    return drawdown


def _event_policy_metrics(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["decision_event_id"])].append(row)
    baseline_selected = [
        sorted(
            candidates,
            key=lambda row: (
                -float(row["expected_net_r"]),
                str(row["candidate_id"]),
            ),
        )[0]
        for candidates in grouped.values()
    ]
    policy_rows = (
        list(predictions)
        if all("policy_selected" in row for row in predictions)
        else _annotate_event_policy(predictions)
    )
    selected = [row for row in policy_rows if bool(row["policy_selected"])]
    expected_sum = sum(float(row["expected_net_r"]) for row in selected)
    realized_sum = sum(float(row["realized_net_r"]) for row in selected)
    event_count = len(grouped)
    baseline_expected = sum(float(row["expected_net_r"]) for row in baseline_selected)
    baseline_realized = sum(float(row["realized_net_r"]) for row in baseline_selected)
    return {
        "selection_policy": "max_tree_dispersion_penalized_score_abstain_le_zero",
        "selected_event_count": len(selected),
        "no_selection_event_count": event_count - len(selected),
        "selection_rate": len(selected) / event_count if event_count else None,
        "selected_expected_net_r_sum": expected_sum,
        "selected_realized_net_r_sum": realized_sum,
        "selected_max_drawdown_r": _max_drawdown(selected),
        "baseline_policy": "max_raw_expected_net_no_abstain",
        "baseline_selected_event_count": len(baseline_selected),
        "baseline_selected_expected_net_r_sum": baseline_expected,
        "baseline_selected_realized_net_r_sum": baseline_realized,
        "baseline_selected_max_drawdown_r": _max_drawdown(baseline_selected),
    }


def _metrics(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fill_labels = [int(bool(row["filled"])) for row in predictions]
    fill_scores = [float(row["fill_probability"]) for row in predictions]
    filled = [row for row in predictions if row["filled"]]
    tp_labels = [int(bool(row["tp_hit"])) for row in filled]
    tp_scores = [float(row["tp_given_fill_probability"]) for row in filled]
    result = {
        "candidate_count": len(predictions),
        "decision_event_count": len(
            {str(row["decision_event_id"]) for row in predictions}
        ),
        "fill_auc": _auc(fill_labels, fill_scores),
        "fill_brier": _brier(fill_labels, fill_scores),
        "tp_given_fill_auc": _auc(tp_labels, tp_scores),
        "tp_given_fill_brier": _brier(tp_labels, tp_scores),
        "expected_net_r_sum": sum(float(row["expected_net_r"]) for row in predictions),
        "expected_net_r_mean": (
            fmean(float(row["expected_net_r"]) for row in predictions)
            if predictions
            else None
        ),
        "realized_net_r_sum": sum(float(row["realized_net_r"]) for row in predictions),
        "cost_application": "filled_once",
    }
    result.update(_event_policy_metrics(predictions))
    return result


def _weight_audits(rows: Sequence[_CandidateRow]) -> dict[str, Any]:
    filled = [row for row in rows if row.filled]
    return {
        "fill_head": _weight_audit(rows),
        "tp_given_fill_head": _weight_audit(filled),
        "gross_r_given_fill_head": _weight_audit(filled),
    }


def _profile(
    rows: Sequence[_CandidateRow],
    *,
    registry: Sequence[str],
    heads: _FittedHeads,
    created_at: datetime,
    strategy_contract: Mapping[str, Any],
    target: CandidateTargetConfig,
    uncertainty_z: float,
) -> dict[str, Any]:
    contract = dict(strategy_contract)
    for required in ("strategy_version", "factor_contract"):
        if not str(contract.get(required) or "").strip():
            raise CandidateRFTrainingError(f"strategy_contract.{required} is required")
    if "target_contract" in contract and contract["target_contract"] != target.contract:
        raise CandidateRFTrainingError(
            "strategy_contract.target_contract conflicts with the trainer target"
        )
    contract.update(
        {
            "target_contract": target.contract,
            "target_definition": target.to_mapping(),
            "probability_semantics": "uncalibrated-random-forest-score",
            "cost_semantics": "causal_cost_r_subtracted_once_after_fill",
            "promotion_policy": "manual-shadow-challenger-only",
        }
    )
    trained_through = max(row.label_known_at for row in rows)
    if created_at < trained_through:
        raise CandidateRFTrainingError(
            "created_at_utc cannot precede the newest training label"
        )
    raw = {
        "schema": RF_CANDIDATE_PROFILE_SCHEMA,
        "created_at_utc": _iso(created_at),
        "trained_through_utc": _iso(trained_through),
        "strategy_contract": contract,
        "feature_registry": list(registry),
        "uncertainty_z": uncertainty_z,
        "forests": {
            "fill_probability": _portable_classifier_forest(heads.fill),
            "tp_given_fill_probability": _portable_classifier_forest(
                heads.tp_given_fill
            ),
            "gross_r_given_fill": _portable_regression_forest(heads.gross_r_given_fill),
        },
    }
    sealed = seal_rf_candidate_profile(raw)
    RFCandidateProfile.from_mapping(sealed)
    return sealed


def fit_rf_candidate_profile(
    rows: Sequence[Mapping[str, Any]],
    *,
    walk_forward: CandidateWalkForwardConfig,
    strategy_contract: Mapping[str, Any],
    forest: CandidateForestConfig = CandidateForestConfig(),
    target: CandidateTargetConfig = CandidateTargetConfig(),
    uncertainty_z: float = 1.0,
    created_at_utc: Any = None,
) -> CandidateRFTrainingResult:
    """Run purged WFO, refit all mature rows, and seal a portable profile.

    The returned profile is a diagnostic challenger.  No acceptance threshold,
    deployment, live mutation, or auto-promotion is performed here.
    """

    uncertainty = _finite(uncertainty_z, name="uncertainty_z")
    if uncertainty < 0.0:
        raise CandidateRFTrainingError("uncertainty_z must be nonnegative")
    prepared, excluded = _prepare_rows(rows, target=target)
    folds = resolve_candidate_walk_forward_folds(prepared, walk_forward)
    fold_reports: list[dict[str, Any]] = []
    oos: list[dict[str, Any]] = []
    for fold in folds:
        train, test = _fold_rows(prepared, fold)
        registry = _build_feature_registry(train)
        heads = _fit_heads(
            train,
            registry,
            forest,
            seed_offset=fold.index * 10,
        )
        predictions = _prediction_rows(
            test,
            registry,
            heads,
            fold_index=fold.index,
            uncertainty_z=uncertainty,
        )
        oos.extend(predictions)
        report = fold.to_mapping()
        report.update(
            {
                "train_candidate_count": len(train),
                "train_event_count": _event_count(train),
                "test_candidate_count": len(test),
                "test_event_count": _event_count(test),
                "tree_dispersion_penalty_multiplier": uncertainty,
                "feature_registry": list(registry),
                "preprocessing": {
                    "fit_scope": "fold-train-only",
                    "numeric": "contract-zero-with-explicit-missing-indicators",
                    "categorical": "train-only-one-hot-with-__OTHER__",
                    "scaling": "none",
                },
                "sample_weight_audit": _weight_audits(train),
                "metrics": _metrics(predictions),
            }
        )
        fold_reports.append(report)
    if not oos:
        raise CandidateRFTrainingError("walk-forward produced no OOS predictions")
    trained_through = max(row.label_known_at for row in prepared)
    created_at = (
        trained_through
        if created_at_utc is None
        else _utc(created_at_utc, name="created_at_utc")
    )
    final_registry = _build_feature_registry(prepared)
    final_heads = _fit_heads(
        prepared,
        final_registry,
        forest,
        seed_offset=1_000_000,
    )
    profile = _profile(
        prepared,
        registry=final_registry,
        heads=final_heads,
        created_at=created_at,
        strategy_contract=strategy_contract,
        target=target,
        uncertainty_z=uncertainty,
    )
    return CandidateRFTrainingResult(
        profile=profile,
        folds=tuple(fold_reports),
        oos_predictions=tuple(oos),
        aggregate_metrics=_metrics(oos),
        excluded_counts=dict(sorted(excluded.items())),
    )


__all__ = [
    "CandidateForestConfig",
    "CandidateRFTrainingError",
    "CandidateRFTrainingResult",
    "CandidateTargetConfig",
    "CandidateWalkForwardConfig",
    "CandidateWalkForwardFold",
    "RF_CANDIDATE_OPTIMIZER_SCHEMA",
    "SINGLE_TP_TARGET_CONTRACT",
    "fit_rf_candidate_profile",
    "resolve_candidate_walk_forward_folds",
]
