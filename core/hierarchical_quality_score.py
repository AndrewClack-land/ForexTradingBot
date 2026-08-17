"""Causal, non-executing hierarchical quality scoring.

The module owns both sides of the frozen ``hierarchical-quality-profile-v1``
contract: a deterministic offline fitter and a dependency-light live scorer.
It intentionally has no broker, database, clock, or order-management imports.

Canonical decision-time input fields
------------------------------------
``symbol``, ``side`` (LONG/SHORT), ``trigger_kind``,
``decision_time_utc`` (timezone-aware), ``stop_atr_h1``, ``session``,
``weekday``, ``fvg_relative_state``, ``fvg_age_bars``, ``spread_r`` and
``intended_execution_mode``.  ``weekday`` is checked against (or derived
from) ``decision_time_utc``.  A few documented legacy aliases are accepted
by :func:`build_quality_features`.

Offline labels are deliberately separate from features: ``label_complete``,
``label_tp1_hit``, ``label_realized_net_r`` (or ``label_gross_r`` with an
explicit causal ``label_causal_cost_r``), and the optional auxiliary target
``label_exact_retest_seconds``.  Actual touch/fill/retest fields are rejected
in the live DECISION phase.  Historical retest duration is only a target for
an auxiliary model; it can never become a same-trade predictor.

Every value returned by :meth:`LiveHierarchicalQualityScorer.score` starts
with ``quality_`` and ``quality_executing`` is always ``False``.  This is a
diagnostic/ranking component, not an execution gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


QUALITY_PROFILE_SCHEMA = "hierarchical-quality-profile-v1"
R_BASIS_NET = "NET_R_AFTER_BROKER_COSTS"
R_BASIS_GROSS = "GROSS_R_BEFORE_COSTS"
_VALID_R_BASES = frozenset({R_BASIS_NET, R_BASIS_GROSS})
_HIERARCHY = (
    "global",
    "trigger",
    "trigger_symbol",
    "trigger_symbol_side",
)
_NUMERIC_FEATURES = ("stop_atr_h1", "fvg_age_log", "spread_r")
_CATEGORICAL_FEATURES = (
    "session",
    "weekday",
    "fvg_relative_state",
    "intended_execution_mode",
)
_WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
_DEFAULT_MIN_SUPPORT = {
    "global": 1,
    "trigger": 4,
    "trigger_symbol": 3,
    "trigger_symbol_side": 3,
}
_DEFAULT_HIERARCHY_MULTIPLIER = {
    "trigger": 1.0,
    "trigger_symbol": 2.0,
    "trigger_symbol_side": 4.0,
}

# These are outcomes or post-decision observations.  The broader pattern
# check below catches spelling variants and nested watcher payloads.
_FORBIDDEN_DECISION_FIELDS = frozenset({
    "actual_touch_at_utc",
    "actual_touch_time_utc",
    "actual_entry_price",
    "actual_fill_price",
    "actual_fill_time_utc",
    "broker_fill_time_utc",
    "exact_retest_at_utc",
    "exact_retest_seconds",
    "fill_price",
    "fill_time_utc",
    "first_limit_touch_at_utc",
    "first_strict_touch_at_utc",
    "first_tolerant_touch_at_utc",
    "first_touch_at_utc",
    "strict_touch_at_utc",
    "time_to_exact_retest_seconds",
})


def _finite(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_finite(
    value: Any,
    *,
    name: str,
    minimum: Optional[float] = None,
) -> Optional[float]:
    if value is None or value == "":
        return None
    result = _finite(value, name=name)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _utc(value: Any, *, name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(
                str(value).strip().replace("Z", "+00:00")
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _normalized_token(value: Any, *, name: str) -> str:
    token = str(value or "").strip().upper().replace(" ", "_")
    if not token:
        raise ValueError(f"{name} is required")
    return token


def _first(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _nested_first(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    value = _first(mapping, names)
    if value is not None:
        return value
    for container_name in ("feature_context", "factor_vector", "metadata"):
        container = mapping.get(container_name)
        if isinstance(container, Mapping):
            value = _first(container, names)
            if value is not None:
                return value
    return None


def _looks_post_decision(name: str) -> bool:
    key = name.strip().lower()
    if key.startswith("quality_expected_"):
        return False
    if key in _FORBIDDEN_DECISION_FIELDS:
        return True
    if "touch" in key and any(
        marker in key
        for marker in ("actual", "first", "strict", "tolerant", "exact", "limit")
    ):
        return True
    if "retest" in key and any(
        marker in key for marker in ("actual", "time", "seconds", "at_utc")
    ):
        return True
    return "fill" in key and any(
        marker in key for marker in ("actual", "broker", "price", "time", "at_utc")
    )


def _reject_post_decision_values(
    value: Any,
    *,
    path: str = "candidate",
) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}"
            if child is not None and _looks_post_decision(key):
                raise ValueError(
                    f"DECISION features contain post-decision field {child_path}"
                )
            _reject_post_decision_values(child, path=child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_post_decision_values(child, path=f"{path}[{index}]")


def _session_for_hour(hour: int) -> str:
    if hour < 7:
        return "ASIA"
    if hour < 13:
        return "LONDON"
    if hour < 21:
        return "NEW_YORK"
    return "ROLLOVER"


def _fvg_relative_state(mapping: Mapping[str, Any], *, side: str) -> str:
    raw = _nested_first(mapping, ("fvg_relative_state", "fvg_state"))
    if raw is not None:
        token = _normalized_token(raw, name="fvg_relative_state")
        aliases = {
            "NONE": "NEUTRAL",
            "NO_FVG": "NEUTRAL",
            "MISSING": "MISSING",
            "ALIGNED": "ALIGNED",
            "OPPOSED": "OPPOSED",
            "OPPOSING": "OPPOSED",
            "NEUTRAL": "NEUTRAL",
        }
        if token in aliases:
            return aliases[token]
        if token in {"LONG", "SHORT", "BULLISH", "BEARISH"}:
            fvg_side = (
                "LONG" if token in {"LONG", "BULLISH"} else "SHORT"
            )
            return "ALIGNED" if fvg_side == side else "OPPOSED"
        raise ValueError("fvg_relative_state has an unsupported value")
    raw_side = _nested_first(mapping, ("fvg_side",))
    if raw_side is None:
        return "MISSING"
    token = _normalized_token(raw_side, name="fvg_side")
    if token in {"NEUTRAL", "NONE", "NO_FVG"}:
        return "NEUTRAL"
    fvg_side = "LONG" if token in {"LONG", "BULLISH"} else "SHORT"
    if token not in {"LONG", "SHORT", "BULLISH", "BEARISH"}:
        raise ValueError("fvg_side has an unsupported value")
    return "ALIGNED" if fvg_side == side else "OPPOSED"


def _spread_r(mapping: Mapping[str, Any]) -> Optional[float]:
    raw = _nested_first(
        mapping,
        ("spread_r", "decision_spread_r", "estimated_spread_r"),
    )
    if raw is not None:
        return _optional_finite(raw, name="spread_r", minimum=0.0)
    bid = _optional_finite(
        _nested_first(mapping, ("decision_bid", "bid")),
        name="decision_bid",
    )
    ask = _optional_finite(
        _nested_first(mapping, ("decision_ask", "ask")),
        name="decision_ask",
    )
    entry = _optional_finite(
        _nested_first(mapping, ("planned_entry", "entry_price")),
        name="planned_entry",
    )
    stop = _optional_finite(
        _nested_first(mapping, ("planned_stop", "stop_price")),
        name="planned_stop",
    )
    if None in {bid, ask, entry, stop}:
        return None
    risk = abs(float(entry) - float(stop))
    if risk <= 0.0:
        return None
    return abs(float(ask) - float(bid)) / risk


def _stop_atr_h1(mapping: Mapping[str, Any]) -> Optional[float]:
    raw = _nested_first(
        mapping,
        (
            "stop_atr_h1",
            "stop_distance_atr",
            "technical_risk_atr",
            "risk_atr",
        ),
    )
    if raw is not None:
        return _optional_finite(raw, name="stop_atr_h1", minimum=0.0)
    entry = _optional_finite(
        _nested_first(mapping, ("planned_entry", "entry_price")),
        name="planned_entry",
    )
    stop = _optional_finite(
        _nested_first(mapping, ("planned_stop", "stop_price")),
        name="planned_stop",
    )
    atr = _optional_finite(
        _nested_first(mapping, ("atr_h1_14", "atr_h1")),
        name="atr_h1_14",
        minimum=0.0,
    )
    if None in {entry, stop, atr} or float(atr) <= 0.0:
        return None
    return abs(float(entry) - float(stop)) / float(atr)


def build_quality_features(
    candidate: Mapping[str, Any],
    *,
    symbol: Optional[str] = None,
    decision_time_utc: Any = None,
    phase: str = "DECISION",
) -> dict[str, Any]:
    """Build the deterministic causal feature context for one candidate.

    Main integration should provide the canonical fields listed in the module
    docstring.  The builder performs normalization, derives weekday/session
    when needed, derives stop/spread R from decision-time prices when possible,
    and rejects populated actual touch/fill/time-to-retest fields.
    """

    if not isinstance(candidate, Mapping):
        raise ValueError("candidate must be a mapping")
    normalized_phase = _normalized_token(phase, name="phase")
    if normalized_phase != "DECISION":
        raise ValueError("hierarchical quality features support DECISION phase only")
    _reject_post_decision_values(candidate)

    resolved_symbol = _normalized_token(
        symbol if symbol is not None else candidate.get("symbol"),
        name="symbol",
    )
    side = _normalized_token(candidate.get("side"), name="side")
    if side not in {"LONG", "SHORT"}:
        raise ValueError("side must be LONG or SHORT")
    trigger = _normalized_token(
        _first(candidate, ("trigger_kind", "entry_trigger", "trigger")),
        name="trigger_kind",
    )
    decision_raw = (
        decision_time_utc
        if decision_time_utc is not None
        else _first(
            candidate,
            (
                "decision_time_utc",
                "decision_bar_close",
                "observed_at_utc",
                "timestamp_utc",
            ),
        )
    )
    decision = _utc(decision_raw, name="decision_time_utc")
    derived_weekday = _WEEKDAYS[decision.weekday()]
    raw_weekday = candidate.get("weekday")
    if raw_weekday is not None:
        if isinstance(raw_weekday, int) and 0 <= raw_weekday <= 6:
            supplied_weekday = _WEEKDAYS[raw_weekday]
        else:
            supplied_weekday = _normalized_token(raw_weekday, name="weekday")[:3]
        if supplied_weekday != derived_weekday:
            raise ValueError("weekday disagrees with decision_time_utc")

    raw_session = candidate.get("session")
    session = (
        _normalized_token(raw_session, name="session")
        if raw_session is not None
        else _session_for_hour(decision.hour)
    )
    execution = _normalized_token(
        _first(
            candidate,
            (
                "intended_execution_mode",
                "execution_method",
                "execution_mode",
                "entry_method",
                "order_type",
            ),
        )
        or "UNKNOWN",
        name="intended_execution_mode",
    )
    fvg_age = _optional_finite(
        _nested_first(candidate, ("fvg_age_bars",)),
        name="fvg_age_bars",
        minimum=0.0,
    )
    return {
        "phase": "DECISION",
        "decision_time_utc": decision,
        "symbol": resolved_symbol,
        "side": side,
        "trigger_kind": trigger,
        "stop_atr_h1": _stop_atr_h1(candidate),
        "session": session,
        "weekday": derived_weekday,
        "fvg_relative_state": _fvg_relative_state(candidate, side=side),
        "fvg_age_bars": fvg_age,
        "fvg_age_log": (
            math.log1p(fvg_age) if fvg_age is not None else None
        ),
        "spread_r": _spread_r(candidate),
        "intended_execution_mode": execution,
    }


# A descriptive alias for callers that prefer the longer name.
build_quality_feature_context = build_quality_features


@dataclass(frozen=True)
class _Normalizer:
    name: str
    mean: float
    scale: float


@dataclass(frozen=True)
class _FittedModel:
    kind: str
    sample_count: int
    coefficients: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    residual_scale: Optional[float] = None


@dataclass(frozen=True)
class HierarchicalQualityProfile:
    """Deeply immutable validated representation of the v1 profile."""

    profile_id: str
    profile_sha256: str
    trained_start_utc: datetime
    trained_end_utc: datetime
    feature_names: tuple[str, ...]
    normalizers: tuple[_Normalizer, ...]
    category_levels: tuple[tuple[str, tuple[str, ...]], ...]
    support_rows: tuple[tuple[str, str, int], ...]
    min_support_rows: tuple[tuple[str, int], ...]
    tp1_model: Optional[_FittedModel]
    net_r_model: Optional[_FittedModel]
    retest_model: Optional[_FittedModel]
    lcb_z: float
    r_label_basis: str
    r_cost_status: str
    preferred_stop_atr_min: float
    preferred_stop_atr_max: float
    outside_stop_atr_penalty_r: float
    logistic_l2: float
    net_r_l2: float
    huber_delta: float

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "HierarchicalQualityProfile":
        if not isinstance(payload, Mapping):
            raise ValueError("quality profile root must be an object")
        if str(payload.get("schema") or "") != QUALITY_PROFILE_SCHEMA:
            raise ValueError(
                f"quality profile schema must be {QUALITY_PROFILE_SCHEMA}"
            )
        profile_id = str(payload.get("profile_id") or "").strip()
        if not profile_id:
            raise ValueError("profile_id is required")
        trained_start = _utc(
            payload.get("trained_start_utc"), name="trained_start_utc"
        )
        trained_end = _utc(
            payload.get("trained_end_utc"), name="trained_end_utc"
        )
        if trained_end < trained_start:
            raise ValueError("trained_end_utc cannot precede trained_start_utc")

        raw_names = payload.get("feature_names")
        if not isinstance(raw_names, list) or not raw_names:
            raise ValueError("feature_names must be a non-empty list")
        feature_names = tuple(str(name) for name in raw_names)
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("feature_names must be unique")
        if feature_names[0] != "intercept":
            raise ValueError("the first feature must be intercept")

        raw_normalizers = payload.get("normalizers")
        if not isinstance(raw_normalizers, Mapping):
            raise ValueError("normalizers must be a mapping")
        normalizers: list[_Normalizer] = []
        for name in _NUMERIC_FEATURES:
            raw = raw_normalizers.get(name)
            if not isinstance(raw, Mapping):
                raise ValueError(f"normalizer {name} is required")
            mean = _finite(raw.get("mean"), name=f"normalizers.{name}.mean")
            scale = _finite(raw.get("scale"), name=f"normalizers.{name}.scale")
            if scale <= 0.0:
                raise ValueError(f"normalizers.{name}.scale must be positive")
            normalizers.append(_Normalizer(name, mean, scale))

        raw_categories = payload.get("category_levels")
        if not isinstance(raw_categories, Mapping):
            raise ValueError("category_levels must be a mapping")
        category_levels: list[tuple[str, tuple[str, ...]]] = []
        for name in _CATEGORICAL_FEATURES:
            raw = raw_categories.get(name)
            if not isinstance(raw, list) or not raw:
                raise ValueError(f"category_levels.{name} must be non-empty")
            levels = tuple(str(level) for level in raw)
            if len(set(levels)) != len(levels):
                raise ValueError(f"category_levels.{name} must be unique")
            category_levels.append((name, levels))

        raw_support = payload.get("support")
        if not isinstance(raw_support, Mapping):
            raise ValueError("support must be a mapping")
        support_rows: list[tuple[str, str, int]] = []
        for level in _HIERARCHY:
            raw_level = raw_support.get(level)
            if not isinstance(raw_level, Mapping):
                raise ValueError(f"support.{level} must be a mapping")
            for key, raw_count in raw_level.items():
                try:
                    count = int(raw_count)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"support.{level}.{key} must be an integer") from exc
                if count < 0 or float(raw_count) != count:
                    raise ValueError(f"support.{level}.{key} must be nonnegative")
                support_rows.append((level, str(key), count))
        if not any(
            level == "global" and key == "*" and count > 0
            for level, key, count in support_rows
        ):
            raise ValueError("support.global.* must be positive")

        raw_min_support = payload.get("min_support")
        if not isinstance(raw_min_support, Mapping):
            raise ValueError("min_support must be a mapping")
        min_support_rows: list[tuple[str, int]] = []
        for level in _HIERARCHY:
            try:
                count = int(raw_min_support[level])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"min_support.{level} must be an integer") from exc
            if count < 1 or float(raw_min_support[level]) != count:
                raise ValueError(f"min_support.{level} must be positive")
            min_support_rows.append((level, count))

        models = payload.get("models")
        if not isinstance(models, Mapping):
            raise ValueError("models must be a mapping")
        tp1_model = _parse_model(
            models.get("tp1_probability"),
            expected_kind="penalized-logistic-map",
            feature_count=len(feature_names),
            optional=True,
        )
        net_r_model = _parse_model(
            models.get("net_r"),
            expected_kind="huber-ridge-net-r",
            feature_count=len(feature_names),
            optional=True,
        )
        retest_model = _parse_model(
            models.get("historical_retest_seconds"),
            expected_kind="huber-ridge-log1p-seconds",
            feature_count=len(feature_names),
            optional=True,
        )
        if tp1_model is None and net_r_model is None:
            raise ValueError("profile needs a TP1 or net-R model")

        inference = payload.get("inference")
        if not isinstance(inference, Mapping):
            raise ValueError("inference must be a mapping")
        lcb_z = _finite(inference.get("one_sided_lcb_z"), name="one_sided_lcb_z")
        if lcb_z < 0.0:
            raise ValueError("one_sided_lcb_z cannot be negative")
        basis = str(inference.get("r_label_basis") or "").strip().upper()
        if basis not in _VALID_R_BASES:
            raise ValueError("r_label_basis is invalid")
        cost_status = str(inference.get("r_cost_status") or "").strip().upper()
        if not cost_status:
            raise ValueError("r_cost_status is required")

        policy = payload.get("ranking_policy")
        if not isinstance(policy, Mapping):
            raise ValueError("ranking_policy must be a mapping")
        stop_min = _finite(
            policy.get("preferred_stop_atr_min"),
            name="preferred_stop_atr_min",
        )
        stop_max = _finite(
            policy.get("preferred_stop_atr_max"),
            name="preferred_stop_atr_max",
        )
        stop_penalty = _finite(
            policy.get("outside_stop_atr_penalty_r"),
            name="outside_stop_atr_penalty_r",
        )
        if stop_min < 0.0 or stop_max < stop_min or stop_penalty < 0.0:
            raise ValueError("invalid preferred stop ATR ranking policy")

        fit = payload.get("fit")
        if not isinstance(fit, Mapping):
            raise ValueError("fit must be a mapping")
        logistic_l2 = _finite(fit.get("logistic_l2"), name="logistic_l2")
        net_r_l2 = _finite(fit.get("net_r_l2"), name="net_r_l2")
        huber_delta = _finite(fit.get("huber_delta"), name="huber_delta")
        if logistic_l2 <= 0.0 or net_r_l2 <= 0.0 or huber_delta <= 0.0:
            raise ValueError("fit penalties must be positive")

        canonical_payload = dict(payload)
        return cls(
            profile_id=profile_id,
            profile_sha256=_canonical_hash(canonical_payload),
            trained_start_utc=trained_start,
            trained_end_utc=trained_end,
            feature_names=feature_names,
            normalizers=tuple(normalizers),
            category_levels=tuple(category_levels),
            support_rows=tuple(sorted(support_rows)),
            min_support_rows=tuple(min_support_rows),
            tp1_model=tp1_model,
            net_r_model=net_r_model,
            retest_model=retest_model,
            lcb_z=lcb_z,
            r_label_basis=basis,
            r_cost_status=cost_status,
            preferred_stop_atr_min=stop_min,
            preferred_stop_atr_max=stop_max,
            outside_stop_atr_penalty_r=stop_penalty,
            logistic_l2=logistic_l2,
            net_r_l2=net_r_l2,
            huber_delta=huber_delta,
        )

    @classmethod
    def load(cls, path: Path | str) -> "HierarchicalQualityProfile":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("quality profile root must be an object")
        return cls.from_mapping(payload)

    def support_count(self, level: str, key: str) -> int:
        for row_level, row_key, count in self.support_rows:
            if row_level == level and row_key == key:
                return count
        return 0

    def min_support(self, level: str) -> int:
        for row_level, count in self.min_support_rows:
            if row_level == level:
                return count
        raise KeyError(level)

    def to_mapping(self) -> dict[str, Any]:
        support: dict[str, dict[str, int]] = {
            level: {} for level in _HIERARCHY
        }
        for level, key, count in self.support_rows:
            support[level][key] = count
        return {
            "schema": QUALITY_PROFILE_SCHEMA,
            "profile_id": self.profile_id,
            "trained_start_utc": _iso(self.trained_start_utc),
            "trained_end_utc": _iso(self.trained_end_utc),
            "feature_names": list(self.feature_names),
            "normalizers": {
                item.name: {"mean": item.mean, "scale": item.scale}
                for item in self.normalizers
            },
            "category_levels": {
                name: list(levels) for name, levels in self.category_levels
            },
            "support": support,
            "min_support": dict(self.min_support_rows),
            "models": {
                "tp1_probability": _model_mapping(self.tp1_model),
                "net_r": _model_mapping(self.net_r_model),
                "historical_retest_seconds": _model_mapping(self.retest_model),
            },
            "inference": {
                "one_sided_lcb_z": self.lcb_z,
                "r_label_basis": self.r_label_basis,
                "r_cost_status": self.r_cost_status,
            },
            "ranking_policy": {
                "preferred_stop_atr_min": self.preferred_stop_atr_min,
                "preferred_stop_atr_max": self.preferred_stop_atr_max,
                "outside_stop_atr_penalty_r": self.outside_stop_atr_penalty_r,
            },
            "fit": {
                "logistic_l2": self.logistic_l2,
                "net_r_l2": self.net_r_l2,
                "huber_delta": self.huber_delta,
            },
        }


def _parse_model(
    payload: Any,
    *,
    expected_kind: str,
    feature_count: int,
    optional: bool,
) -> Optional[_FittedModel]:
    if payload is None and optional:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError(f"model {expected_kind} must be an object")
    if str(payload.get("kind") or "") != expected_kind:
        raise ValueError(f"model kind must be {expected_kind}")
    try:
        sample_count = int(payload.get("sample_count"))
    except (TypeError, ValueError) as exc:
        raise ValueError("model sample_count must be an integer") from exc
    if sample_count < 1:
        raise ValueError("model sample_count must be positive")
    raw_coefficients = payload.get("coefficients")
    if not isinstance(raw_coefficients, list) or len(raw_coefficients) != feature_count:
        raise ValueError("model coefficient dimension mismatch")
    coefficients = tuple(
        _finite(value, name="model coefficient") for value in raw_coefficients
    )
    raw_covariance = payload.get("covariance")
    if not isinstance(raw_covariance, list) or len(raw_covariance) != feature_count:
        raise ValueError("model covariance dimension mismatch")
    matrix = np.asarray(raw_covariance, dtype=float)
    if matrix.shape != (feature_count, feature_count):
        raise ValueError("model covariance must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("model covariance must be finite")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-9):
        raise ValueError("model covariance must be symmetric")
    if float(np.min(np.linalg.eigvalsh(matrix))) < -1e-7:
        raise ValueError("model covariance must be positive semidefinite")
    residual = payload.get("residual_scale")
    residual_scale = (
        None
        if residual is None
        else _finite(residual, name="model residual_scale")
    )
    if residual_scale is not None and residual_scale < 0.0:
        raise ValueError("model residual_scale cannot be negative")
    return _FittedModel(
        kind=expected_kind,
        sample_count=sample_count,
        coefficients=coefficients,
        covariance=tuple(tuple(float(value) for value in row) for row in matrix),
        residual_scale=residual_scale,
    )


def _model_mapping(model: Optional[_FittedModel]) -> Optional[dict[str, Any]]:
    if model is None:
        return None
    return {
        "kind": model.kind,
        "sample_count": model.sample_count,
        "coefficients": list(model.coefficients),
        "covariance": [list(row) for row in model.covariance],
        "residual_scale": model.residual_scale,
    }


def _hierarchy_keys(context: Mapping[str, Any]) -> dict[str, str]:
    trigger = str(context["trigger_kind"])
    symbol = str(context["symbol"])
    side = str(context["side"])
    return {
        "global": "*",
        "trigger": trigger,
        "trigger_symbol": f"{trigger}|{symbol}",
        "trigger_symbol_side": f"{trigger}|{symbol}|{side}",
    }


def _normalizers(contexts: Sequence[Mapping[str, Any]]) -> tuple[_Normalizer, ...]:
    result: list[_Normalizer] = []
    for name in _NUMERIC_FEATURES:
        values = np.asarray(
            [float(row[name]) for row in contexts if row.get(name) is not None],
            dtype=float,
        )
        if values.size == 0:
            mean, scale = 0.0, 1.0
        else:
            mean = float(np.mean(values))
            scale = float(np.std(values))
            if not math.isfinite(scale) or scale < 1e-9:
                scale = 1.0
        result.append(_Normalizer(name, mean, scale))
    return tuple(result)


def _category_levels(
    contexts: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        (
            name,
            tuple(sorted({str(row[name]) for row in contexts} | {"MISSING"})),
        )
        for name in _CATEGORICAL_FEATURES
    )


def _feature_names(
    contexts: Sequence[Mapping[str, Any]],
    categories: Sequence[tuple[str, tuple[str, ...]]],
) -> tuple[str, ...]:
    names = ["intercept"]
    hierarchy_values: dict[str, set[str]] = {
        level: set() for level in _HIERARCHY[1:]
    }
    for context in contexts:
        keys = _hierarchy_keys(context)
        for level in _HIERARCHY[1:]:
            hierarchy_values[level].add(keys[level])
    for level in _HIERARCHY[1:]:
        names.extend(
            f"hierarchy:{level}:{key}"
            for key in sorted(hierarchy_values[level])
        )
    for name in _NUMERIC_FEATURES:
        names.extend((f"numeric:{name}", f"missing:{name}"))
    for name, levels in categories:
        names.extend(f"category:{name}:{level}" for level in levels)
    return tuple(names)


def _vectorize(
    context: Mapping[str, Any],
    *,
    feature_names: Sequence[str],
    normalizers: Sequence[_Normalizer],
    active_hierarchy_level: str = "trigger_symbol_side",
) -> tuple[np.ndarray, tuple[str, ...]]:
    active_index = _HIERARCHY.index(active_hierarchy_level)
    values: dict[str, float] = {"intercept": 1.0}
    keys = _hierarchy_keys(context)
    for index, level in enumerate(_HIERARCHY[1:], start=1):
        if index <= active_index:
            values[f"hierarchy:{level}:{keys[level]}"] = 1.0
    for normalizer in normalizers:
        raw = context.get(normalizer.name)
        values[f"numeric:{normalizer.name}"] = (
            0.0
            if raw is None
            else (float(raw) - normalizer.mean) / normalizer.scale
        )
        values[f"missing:{normalizer.name}"] = float(raw is None)
    oov: list[str] = []
    feature_name_set = set(feature_names)
    for name in _CATEGORICAL_FEATURES:
        level = str(context.get(name) or "MISSING")
        column = f"category:{name}:{level}"
        if column in feature_name_set:
            values[column] = 1.0
        else:
            oov.append(f"{name}={level}")
    return (
        np.asarray([values.get(name, 0.0) for name in feature_names], dtype=float),
        tuple(oov),
    )


def _penalty_diagonal(
    feature_names: Sequence[str],
    *,
    base_l2: float,
) -> np.ndarray:
    penalties = np.full(len(feature_names), base_l2, dtype=float)
    penalties[0] = max(1e-6, base_l2 * 1e-6)
    for index, name in enumerate(feature_names):
        for level, multiplier in _DEFAULT_HIERARCHY_MULTIPLIER.items():
            if name.startswith(f"hierarchy:{level}:"):
                penalties[index] = base_l2 * multiplier
                break
    return penalties


def _safe_inverse(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(matrix, rcond=1e-12, hermitian=True)


def _sigmoid(value: Any) -> Any:
    array = np.asarray(value, dtype=float)
    clipped = np.clip(array, -35.0, 35.0)
    result = 1.0 / (1.0 + np.exp(-clipped))
    if np.ndim(result) == 0:
        return float(result)
    return result


def _fit_logistic_map(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    feature_names: Sequence[str],
    l2: float,
    max_iterations: int,
) -> _FittedModel:
    penalty = _penalty_diagonal(feature_names, base_l2=l2)
    coefficients = np.zeros(matrix.shape[1], dtype=float)
    # Initialize the intercept with a weak Jeffreys-like pseudo-count so an
    # all-success/all-failure bootstrap remains finite.
    positives = float(np.sum(labels))
    probability = (positives + 0.5) / (len(labels) + 1.0)
    coefficients[0] = math.log(probability / (1.0 - probability))
    for _ in range(max_iterations):
        fitted = _sigmoid(matrix @ coefficients)
        weights = np.maximum(fitted * (1.0 - fitted), 1e-7)
        hessian = (matrix.T * weights) @ matrix + np.diag(penalty)
        gradient = matrix.T @ (fitted - labels) + penalty * coefficients
        step = _safe_inverse(hessian) @ gradient
        coefficients -= step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    fitted = _sigmoid(matrix @ coefficients)
    weights = np.maximum(fitted * (1.0 - fitted), 1e-7)
    hessian = (matrix.T * weights) @ matrix + np.diag(penalty)
    covariance = _safe_inverse(hessian)
    return _FittedModel(
        kind="penalized-logistic-map",
        sample_count=len(labels),
        coefficients=tuple(float(value) for value in coefficients),
        covariance=tuple(tuple(float(value) for value in row) for row in covariance),
    )


def _robust_scale(values: np.ndarray) -> float:
    median = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - median)))
    if not math.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(values))
    return scale if math.isfinite(scale) and scale >= 1e-6 else 1.0


def _fit_huber_ridge(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    feature_names: Sequence[str],
    l2: float,
    huber_delta: float,
    max_iterations: int,
    kind: str,
) -> _FittedModel:
    penalty = _penalty_diagonal(feature_names, base_l2=l2)
    coefficients = np.zeros(matrix.shape[1], dtype=float)
    coefficients[0] = float(np.median(labels))
    scale = _robust_scale(labels)
    weights = np.ones(len(labels), dtype=float)
    for _ in range(max_iterations):
        residuals = labels - matrix @ coefficients
        scale = _robust_scale(residuals)
        cutoff = max(1e-9, huber_delta * scale)
        absolute = np.abs(residuals)
        weights = np.where(absolute <= cutoff, 1.0, cutoff / np.maximum(absolute, 1e-12))
        hessian = (matrix.T * weights) @ matrix + np.diag(penalty)
        target = matrix.T @ (weights * labels)
        updated = _safe_inverse(hessian) @ target
        if float(np.max(np.abs(updated - coefficients))) < 1e-9:
            coefficients = updated
            break
        coefficients = updated
    residuals = labels - matrix @ coefficients
    scale = _robust_scale(residuals)
    hessian = (matrix.T * weights) @ matrix + np.diag(penalty)
    covariance = (scale * scale) * _safe_inverse(hessian)
    return _FittedModel(
        kind=kind,
        sample_count=len(labels),
        coefficients=tuple(float(value) for value in coefficients),
        covariance=tuple(tuple(float(value) for value in row) for row in covariance),
        residual_scale=scale,
    )


def _complete(observation: Mapping[str, Any]) -> bool:
    raw = _first(observation, ("label_complete", "pnl_complete", "complete"))
    if raw is None:
        return True
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "complete"}
    return bool(raw)


def _binary_label(value: Any, *, name: str) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    parsed = _finite(value, name=name)
    if parsed not in {0.0, 1.0}:
        raise ValueError(f"{name} must be 0/1")
    return parsed


def _fit_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist decision fields so labels can never leak into predictors."""

    allowed = {
        "symbol",
        "side",
        "trigger_kind",
        "entry_trigger",
        "trigger",
        "decision_time_utc",
        "decision_bar_close",
        "observed_at_utc",
        "timestamp_utc",
        "stop_atr_h1",
        "stop_distance_atr",
        "technical_risk_atr",
        "risk_atr",
        "session",
        "weekday",
        "fvg_relative_state",
        "fvg_state",
        "fvg_side",
        "fvg_age_bars",
        "spread_r",
        "decision_spread_r",
        "estimated_spread_r",
        "decision_bid",
        "decision_ask",
        "bid",
        "ask",
        "planned_entry",
        "entry_price",
        "planned_stop",
        "stop_price",
        "atr_h1_14",
        "atr_h1",
        "intended_execution_mode",
        "execution_method",
        "execution_mode",
        "entry_method",
        "order_type",
        "factor_vector",
        "feature_context",
        "metadata",
    }
    return {key: value for key, value in observation.items() if key in allowed}


def fit_quality_profile(
    observations: Sequence[Mapping[str, Any]],
    *,
    profile_id: Optional[str] = None,
    r_label_basis: str = R_BASIS_NET,
    min_support: Optional[Mapping[str, int]] = None,
    logistic_l2: float = 4.0,
    net_r_l2: float = 4.0,
    huber_delta: float = 1.345,
    one_sided_lcb_z: float = 1.2815515655446004,
    preferred_stop_atr_min: float = 0.75,
    preferred_stop_atr_max: float = 1.50,
    outside_stop_atr_penalty_r: float = 0.0,
    max_iterations: int = 100,
) -> HierarchicalQualityProfile:
    """Fit a deterministic v1 profile from independent-idea rows.

    TP and R labels may be independently censored.  For
    ``NET_R_AFTER_BROKER_COSTS`` use ``label_realized_net_r``.  For
    ``GROSS_R_BEFORE_COSTS`` use ``label_gross_r`` *and*
    ``label_causal_cost_r``; the latter is subtracted before fitting so live
    scoring never double-deducts costs.  Incomplete rows are ignored.
    """

    if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
        raise ValueError("observations must be a sequence of mappings")
    basis = str(r_label_basis or "").strip().upper()
    if basis not in _VALID_R_BASES:
        raise ValueError("r_label_basis is invalid")
    logistic_l2 = _finite(logistic_l2, name="logistic_l2")
    net_r_l2 = _finite(net_r_l2, name="net_r_l2")
    huber_delta = _finite(huber_delta, name="huber_delta")
    lcb_z = _finite(one_sided_lcb_z, name="one_sided_lcb_z")
    stop_min = _finite(preferred_stop_atr_min, name="preferred_stop_atr_min")
    stop_max = _finite(preferred_stop_atr_max, name="preferred_stop_atr_max")
    stop_penalty = _finite(
        outside_stop_atr_penalty_r,
        name="outside_stop_atr_penalty_r",
    )
    if logistic_l2 <= 0.0 or net_r_l2 <= 0.0 or huber_delta <= 0.0:
        raise ValueError("fit penalties must be positive")
    if lcb_z < 0.0 or stop_min < 0.0 or stop_max < stop_min or stop_penalty < 0.0:
        raise ValueError("invalid inference or ranking policy")
    try:
        iterations = int(max_iterations)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_iterations must be an integer") from exc
    if iterations < 1:
        raise ValueError("max_iterations must be positive")

    contexts: list[dict[str, Any]] = []
    tp_labels: list[Optional[float]] = []
    r_labels: list[Optional[float]] = []
    retest_labels: list[Optional[float]] = []
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            raise ValueError(f"observation {index} must be a mapping")
        if not _complete(observation):
            continue
        context = build_quality_features(_fit_payload(observation))
        tp = _binary_label(
            _first(observation, ("label_tp1_hit", "tp1_hit")),
            name=f"observations[{index}].label_tp1_hit",
        )
        if basis == R_BASIS_NET:
            r_value = _optional_finite(
                _first(
                    observation,
                    (
                        "label_realized_net_r",
                        "broker_realized_net_r",
                        "realized_net_r",
                        "net_r",
                    ),
                ),
                name=f"observations[{index}].label_realized_net_r",
            )
        else:
            gross = _optional_finite(
                _first(observation, ("label_gross_r", "gross_r")),
                name=f"observations[{index}].label_gross_r",
            )
            if gross is None:
                r_value = None
            else:
                cost = _optional_finite(
                    _first(
                        observation,
                        (
                            "label_causal_cost_r",
                            "causal_cost_r",
                            "estimated_cost_r",
                        ),
                    ),
                    name=f"observations[{index}].label_causal_cost_r",
                    minimum=0.0,
                )
                if cost is None:
                    raise ValueError(
                        "gross R labels require label_causal_cost_r"
                    )
                r_value = gross - cost
        retest = _optional_finite(
            _first(
                observation,
                (
                    "label_exact_retest_seconds",
                    "historical_exact_retest_seconds",
                    "time_to_exact_retest_seconds",
                ),
            ),
            name=f"observations[{index}].label_exact_retest_seconds",
            minimum=0.0,
        )
        if tp is None and r_value is None and retest is None:
            continue
        contexts.append(context)
        tp_labels.append(tp)
        r_labels.append(r_value)
        retest_labels.append(retest)

    if not contexts:
        raise ValueError("no complete labelled observations")
    if not any(value is not None for value in tp_labels) and not any(
        value is not None for value in r_labels
    ):
        raise ValueError("profile needs at least one TP1 or R label")

    minimums = dict(_DEFAULT_MIN_SUPPORT)
    if min_support is not None:
        unknown = set(min_support) - set(_HIERARCHY)
        if unknown:
            raise ValueError(f"unknown min_support levels: {sorted(unknown)}")
        minimums.update(min_support)
    for level in _HIERARCHY:
        try:
            minimums[level] = int(minimums[level])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"min_support.{level} must be an integer") from exc
        if minimums[level] < 1:
            raise ValueError(f"min_support.{level} must be positive")

    normalizers = _normalizers(contexts)
    categories = _category_levels(contexts)
    names = _feature_names(contexts, categories)
    full_matrix = np.vstack([
        _vectorize(
            context,
            feature_names=names,
            normalizers=normalizers,
        )[0]
        for context in contexts
    ])

    tp_indices = [i for i, value in enumerate(tp_labels) if value is not None]
    r_indices = [i for i, value in enumerate(r_labels) if value is not None]
    retest_indices = [
        i for i, value in enumerate(retest_labels) if value is not None
    ]
    tp_model = (
        _fit_logistic_map(
            full_matrix[tp_indices],
            np.asarray([tp_labels[i] for i in tp_indices], dtype=float),
            feature_names=names,
            l2=logistic_l2,
            max_iterations=iterations,
        )
        if tp_indices
        else None
    )
    net_model = (
        _fit_huber_ridge(
            full_matrix[r_indices],
            np.asarray([r_labels[i] for i in r_indices], dtype=float),
            feature_names=names,
            l2=net_r_l2,
            huber_delta=huber_delta,
            max_iterations=iterations,
            kind="huber-ridge-net-r",
        )
        if r_indices
        else None
    )
    retest_model = (
        _fit_huber_ridge(
            full_matrix[retest_indices],
            np.log1p(
                np.asarray([retest_labels[i] for i in retest_indices], dtype=float)
            ),
            feature_names=names,
            l2=net_r_l2,
            huber_delta=huber_delta,
            max_iterations=iterations,
            kind="huber-ridge-log1p-seconds",
        )
        if retest_indices
        else None
    )

    support: dict[str, dict[str, int]] = {
        level: {} for level in _HIERARCHY
    }
    support["global"]["*"] = len(contexts)
    for context in contexts:
        keys = _hierarchy_keys(context)
        for level in _HIERARCHY[1:]:
            key = keys[level]
            support[level][key] = support[level].get(key, 0) + 1
    decisions = [context["decision_time_utc"] for context in contexts]
    raw_profile: dict[str, Any] = {
        "schema": QUALITY_PROFILE_SCHEMA,
        "profile_id": "PENDING",
        "trained_start_utc": _iso(min(decisions)),
        "trained_end_utc": _iso(max(decisions)),
        "feature_names": list(names),
        "normalizers": {
            item.name: {"mean": item.mean, "scale": item.scale}
            for item in normalizers
        },
        "category_levels": {
            name: list(levels) for name, levels in categories
        },
        "support": support,
        "min_support": minimums,
        "models": {
            "tp1_probability": _model_mapping(tp_model),
            "net_r": _model_mapping(net_model),
            "historical_retest_seconds": _model_mapping(retest_model),
        },
        "inference": {
            "one_sided_lcb_z": lcb_z,
            "r_label_basis": basis,
            "r_cost_status": (
                "ALREADY_NET_NO_REDEDUCTION"
                if basis == R_BASIS_NET
                else "GROSS_MINUS_EMBEDDED_CAUSAL_COST_NO_REDEDUCTION"
            ),
        },
        "ranking_policy": {
            "preferred_stop_atr_min": stop_min,
            "preferred_stop_atr_max": stop_max,
            "outside_stop_atr_penalty_r": stop_penalty,
        },
        "fit": {
            "logistic_l2": logistic_l2,
            "net_r_l2": net_r_l2,
            "huber_delta": huber_delta,
        },
    }
    generated_id = _canonical_hash(raw_profile)[:24]
    raw_profile["profile_id"] = str(profile_id or generated_id)
    return HierarchicalQualityProfile.from_mapping(raw_profile)


fit_hierarchical_quality_profile = fit_quality_profile


def _model_prediction(
    model: _FittedModel,
    vector: np.ndarray,
) -> tuple[float, float]:
    coefficients = np.asarray(model.coefficients, dtype=float)
    covariance = np.asarray(model.covariance, dtype=float)
    mean = float(vector @ coefficients)
    variance = float(vector @ covariance @ vector)
    return mean, math.sqrt(max(0.0, variance))


class LiveHierarchicalQualityScorer:
    """Frozen, causal scorer for one immutable quality profile."""

    def __init__(self, profile: HierarchicalQualityProfile) -> None:
        if not isinstance(profile, HierarchicalQualityProfile):
            raise TypeError("profile must be HierarchicalQualityProfile")
        self.profile = profile
        self.model_id = profile.profile_id

    @classmethod
    def load(
        cls,
        path: Path | str | None,
    ) -> Optional["LiveHierarchicalQualityScorer"]:
        """Fail open for live startup; use profile.load for strict validation."""

        if path is None:
            return None
        try:
            return cls(HierarchicalQualityProfile.load(path))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _backoff(
        self,
        context: Mapping[str, Any],
    ) -> tuple[str, int, str, tuple[str, ...]]:
        keys = _hierarchy_keys(context)
        path: list[str] = []
        for level in reversed(_HIERARCHY[1:]):
            count = self.profile.support_count(level, keys[level])
            required = self.profile.min_support(level)
            path.append(f"{level}:{keys[level]}:{count}/{required}")
            if count >= required:
                reason = (
                    "NONE"
                    if level == "trigger_symbol_side"
                    else f"BACKOFF_TO_{level.upper()}"
                )
                return level, count, reason, tuple(reversed(path))
        count = self.profile.support_count("global", "*")
        path.append(
            f"global:*:{count}/{self.profile.min_support('global')}"
        )
        return "global", count, "BACKOFF_TO_GLOBAL", tuple(reversed(path))

    def score(
        self,
        candidate: Mapping[str, Any],
        *,
        symbol: Optional[str] = None,
        decision_time_utc: Any = None,
        phase: str = "DECISION",
    ) -> dict[str, Any]:
        """Return diagnostic annotations; every returned key is ``quality_*``."""

        context = build_quality_features(
            candidate,
            symbol=symbol,
            decision_time_utc=decision_time_utc,
            phase=phase,
        )
        decision = context["decision_time_utc"]
        if self.profile.trained_end_utc > decision:
            raise ValueError("quality profile is future-dated for this decision")
        level, support, reason, support_path = self._backoff(context)
        vector, oov = _vectorize(
            context,
            feature_names=self.profile.feature_names,
            normalizers=self.profile.normalizers,
            active_hierarchy_level=level,
        )
        stop_atr = context.get("stop_atr_h1")
        stop_in_band: Optional[bool] = None
        stop_penalty = 0.0
        if stop_atr is not None:
            stop_in_band = (
                self.profile.preferred_stop_atr_min
                <= float(stop_atr)
                <= self.profile.preferred_stop_atr_max
            )
            if not stop_in_band:
                stop_penalty = self.profile.outside_stop_atr_penalty_r

        output: dict[str, Any] = {
            "quality_status": "SCORED",
            "quality_profile_schema": QUALITY_PROFILE_SCHEMA,
            "quality_profile_id": self.profile.profile_id,
            "quality_profile_sha256": self.profile.profile_sha256,
            "quality_phase": "DECISION",
            "quality_executing": False,
            "quality_hierarchy_level": level,
            "quality_hierarchy_support": support,
            "quality_hierarchy_path": list(support_path),
            "quality_backoff_reason": reason,
            "quality_feature_oov": list(oov),
            "quality_r_label_basis": self.profile.r_label_basis,
            "quality_r_cost_status": self.profile.r_cost_status,
            "quality_stop_atr_h1": stop_atr,
            "quality_stop_atr_in_preferred_band": stop_in_band,
            "quality_stop_atr_penalty_r": stop_penalty,
            "quality_feature_missing": [
                name
                for name in _NUMERIC_FEATURES
                if context.get(name) is None
            ],
            "quality_tp1_probability": None,
            "quality_tp1_probability_lcb": None,
            "quality_tp1_probability_ucb": None,
            "quality_tp1_logit_std_error": None,
            "quality_expected_net_r": None,
            "quality_expected_net_r_lcb": None,
            "quality_expected_net_r_std_error": None,
            "quality_conservative_net_r": None,
            "quality_expected_historical_retest_seconds": None,
            "quality_expected_historical_retest_std_error_seconds": None,
        }
        if self.profile.tp1_model is not None:
            logit, logit_se = _model_prediction(self.profile.tp1_model, vector)
            output.update({
                "quality_tp1_probability": _sigmoid(logit),
                "quality_tp1_probability_lcb": _sigmoid(
                    logit - self.profile.lcb_z * logit_se
                ),
                "quality_tp1_probability_ucb": _sigmoid(
                    logit + self.profile.lcb_z * logit_se
                ),
                "quality_tp1_logit_std_error": logit_se,
            })
        if self.profile.net_r_model is not None:
            expected, expected_se = _model_prediction(
                self.profile.net_r_model, vector
            )
            lcb = expected - self.profile.lcb_z * expected_se
            output.update({
                "quality_expected_net_r": expected,
                "quality_expected_net_r_lcb": lcb,
                "quality_expected_net_r_std_error": expected_se,
                "quality_conservative_net_r": lcb - stop_penalty,
            })
        if self.profile.retest_model is not None:
            log_seconds, log_seconds_se = _model_prediction(
                self.profile.retest_model, vector
            )
            clipped = min(30.0, max(0.0, log_seconds))
            expected_seconds = math.expm1(clipped)
            output.update({
                "quality_expected_historical_retest_seconds": expected_seconds,
                "quality_expected_historical_retest_std_error_seconds": (
                    math.exp(clipped) * log_seconds_se
                ),
            })
        if self.profile.tp1_model is None or self.profile.net_r_model is None:
            output["quality_status"] = "PARTIAL_MODEL"
        return output


def _ranking_number(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return -math.inf
    return parsed if math.isfinite(parsed) else -math.inf


def _stable_id(candidate: Mapping[str, Any]) -> str:
    for name in (
        "quality_stable_id",
        "opportunity_id",
        "candidate_id",
        "shadow_opportunity_id",
        "setup_id",
    ):
        value = candidate.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    canonical = {
        key: value
        for key, value in candidate.items()
        if not str(key).startswith("quality_")
    }
    return _canonical_hash(canonical)[:24]


def rank_quality_candidates(
    candidates: Sequence[Mapping[str, Any]],
    scorer: Optional[LiveHierarchicalQualityScorer] = None,
    *,
    symbol: Optional[str] = None,
    decision_time_utc: Any = None,
) -> list[dict[str, Any]]:
    """Purely rank copied candidates; never mutate or return an order action.

    Primary sort is ``quality_rank_score_r`` (conservative net R less an
    optional correlation penalty), followed by TP1 probability LCB, expected
    net R, and stable ID.  A stop-ATR penalty is already included by the
    scorer in ``quality_conservative_net_r``.
    """

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("every candidate must be a mapping")
        row = dict(candidate)
        if scorer is not None:
            annotations = scorer.score(
                candidate,
                symbol=symbol,
                decision_time_utc=decision_time_utc,
            )
            if any(not key.startswith("quality_") for key in annotations):
                raise ValueError("quality scorer returned a non-quality field")
            row.update(annotations)
        stable_id = _stable_id(row)
        conservative = _ranking_number(row.get("quality_conservative_net_r"))
        raw_penalty = row.get(
            "quality_correlation_penalty_r",
            row.get("correlation_penalty_r", 0.0),
        )
        try:
            correlation_penalty = float(raw_penalty or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("correlation penalty must be numeric") from exc
        if not math.isfinite(correlation_penalty) or correlation_penalty < 0.0:
            raise ValueError("correlation penalty must be finite and nonnegative")
        rank_score = (
            conservative - correlation_penalty
            if math.isfinite(conservative)
            else -math.inf
        )
        row.update({
            "quality_stable_id": stable_id,
            "quality_correlation_penalty_r": correlation_penalty,
            "quality_rank_score_r": (
                rank_score if math.isfinite(rank_score) else None
            ),
            "quality_rank_position": None,
            "quality_selected": False,
        })
        rows.append(row)

    rows.sort(
        key=lambda row: (
            -_ranking_number(row.get("quality_rank_score_r")),
            -_ranking_number(row.get("quality_tp1_probability_lcb")),
            -_ranking_number(row.get("quality_expected_net_r")),
            str(row["quality_stable_id"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["quality_rank_position"] = rank
        row["quality_selected"] = rank == 1 and row["quality_rank_score_r"] is not None
    return rows


__all__ = [
    "QUALITY_PROFILE_SCHEMA",
    "R_BASIS_GROSS",
    "R_BASIS_NET",
    "HierarchicalQualityProfile",
    "LiveHierarchicalQualityScorer",
    "build_quality_feature_context",
    "build_quality_features",
    "fit_hierarchical_quality_profile",
    "fit_quality_profile",
    "rank_quality_candidates",
]
