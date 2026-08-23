"""Portable, causal Random-Forest contract for shadow candidate scoring.

The offline fitter is intentionally outside this module.  A fitted model is
transported as canonical JSON containing three forests whose trees are plain
arrays.  Live inference therefore needs neither scikit-learn nor pickle and
cannot execute code embedded in a model artifact.

The scorer is diagnostic.  It copies no execution fields and every annotation
it returns starts with ``rf_``.  Realized fill, touch, retest, exit and outcome
fields are rejected recursively before any feature is built.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean
from typing import Any, Optional


SINGLE_TP_TARGET_CONTRACT = "production-single-tp-v1"


RF_CANDIDATE_PROFILE_SCHEMA = "rf-candidate-profile-v1"
RF_CANDIDATE_FEATURE_SCHEMA = "rf-candidate-features-v1"

_FACTOR_KEYS = (
    "h1_premium_discount",
    "false_breakout_4h",
    "true_breakout_15m",
    "false_breakout_1h",
    "true_breakout_1h",
    "order_block_1h",
    "rejection_block_1h",
    "fvg_regime_1h",
)

_DIRECT_NUMERIC_FEATURES = frozenset(
    {
        "score_long",
        "score_short",
        "score_delta_long_minus_short",
        "score_for_side",
        "score_against_side",
        "required_margin_for_side",
        "score_edge_for_side",
        "production_bias_aligned",
        "side_sign",
        "production_priority",
        "production_priority_missing",
        "fvg_alignment",
        "fvg_age_log",
        "fvg_age_missing",
        "stop_atr_h1",
        "stop_atr_h1_missing",
        "entry_zone_atr_m15",
        "entry_zone_atr_m15_missing",
        "quote_to_entry_atr_m15",
        "quote_to_entry_atr_m15_missing",
        "tp1_distance_atr_h1",
        "tp1_distance_atr_h1_missing",
        "vol_r",
        "vol_r_missing",
        "vol_em_1d",
        "vol_em_1d_missing",
        "vol_tp1_em_ratio",
        "vol_tp1_em_ratio_missing",
        "spread_r",
        "spread_r_missing",
        "orca_snapshot_available",
        "orca_absorption_ratio",
        "orca_absorption_ratio_missing",
        "orca_systemic_score",
        "orca_systemic_score_missing",
        "orca_tail_event_probability",
        "orca_tail_event_probability_missing",
        "orca_largest_eigenvalue",
        "orca_largest_eigenvalue_missing",
        "orca_eigenvalue_ratio",
        "orca_eigenvalue_ratio_missing",
        "orca_spectral_gap",
        "orca_spectral_gap_missing",
        "orca_crisis_probability",
        "orca_crisis_probability_missing",
        "orca_bcd_auc",
        "orca_bcd_auc_missing",
    }
)

_CATEGORY_PREFIXES = (
    "symbol:",
    "trigger:",
    "session:",
    "weekday:",
    "execution:",
    "fvg_state:",
    "orca_mode:",
)

_DEFAULT_SYMBOLS = ("EURUSD", "GBPUSD", "USDCAD", "GOLD")
_DEFAULT_TRIGGERS = (
    "rejection_block_1h",
    "fxpro_cluster_rejection_15m",
    "fxpro_quote_pressure_rejection_15m",
    "h1_pivot_reclaim_15m",
    "order_block_1h",
    "__OTHER__",
)
_DEFAULT_SESSIONS = (
    "ASIA_ONLY",
    "LONDON_ONLY",
    "OVERLAP",
    "NY_ONLY",
    "OFF",
    "MISSING",
)
_DEFAULT_WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
_DEFAULT_EXECUTIONS = ("MARKET", "EXACT_LIMIT", "UNKNOWN", "__OTHER__")
_DEFAULT_FVG_STATES = ("ALIGNED", "OPPOSED", "NEUTRAL", "MISSING")
_DEFAULT_ORCA_MODES = ("RISK_ON", "RISK_OFF", "TRANSITION", "MISSING", "__OTHER__")

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

_EXACT_POST_DECISION_FIELDS = frozenset(
    {
        "actual_fill",
        "actual_fill_price",
        "actual_fill_time",
        "actual_fill_at_utc",
        "broker_fill_price",
        "broker_fill_time",
        "fill_price",
        "fill_time",
        "fill_at_utc",
        "first_touch",
        "first_touch_time",
        "touch_time",
        "touch_at_utc",
        "actual_retest_seconds",
        "exact_retest_seconds",
        "time_to_exact_retest_seconds",
        "retest_time",
        "retest_at_utc",
        "exit",
        "exit_time",
        "exit_price",
        "exit_reason",
        "outcome",
        "trade_outcome",
        "realized_net_r",
        "broker_realized_net_r",
        "pnl",
        "net_pnl",
        "tp_hit",
        "tp1_hit",
        "sl_hit",
    }
)

_TRIGGER_ALIASES = {
    "pivot_reclaim": "h1_pivot_reclaim_15m",
    "h1_pivot_reclaim": "h1_pivot_reclaim_15m",
    "orderblock_1h": "order_block_1h",
    "rejection_block_h1": "rejection_block_1h",
}


def _finite(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _optional_finite(
    value: Any,
    *,
    name: str,
    minimum: Optional[float] = None,
) -> Optional[float]:
    if value is None or value == "":
        return None
    parsed = _finite(value, name=name)
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return parsed


def _utc(value: Any, *, name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any, *, path: str = "profile") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, datetime):
        return _iso(_utc(value, name=path))
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise ValueError(f"{path} contains a non-string key")
            result[raw_key] = _json_safe(item, path=f"{path}.{raw_key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{path} contains a non-JSON value")


def _canonical_json(value: Mapping[str, Any]) -> str:
    safe = _json_safe(value)
    return json.dumps(
        safe,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_rf_profile_sha256(profile: Mapping[str, Any]) -> str:
    """Hash a profile while excluding its self-referential hash field."""

    if not isinstance(profile, Mapping):
        raise ValueError("profile must be a mapping")
    payload = dict(profile)
    payload.pop("profile_sha256", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def seal_rf_candidate_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe profile copy with its canonical SHA-256 attached."""

    payload = dict(_json_safe(profile))
    payload.pop("profile_sha256", None)
    payload["profile_sha256"] = canonical_rf_profile_sha256(payload)
    return payload


def default_rf_feature_registry() -> tuple[str, ...]:
    numeric = (
        *(f"factor_alignment:{key}" for key in _FACTOR_KEYS),
        *(f"factor_present:{key}" for key in _FACTOR_KEYS),
        *sorted(_DIRECT_NUMERIC_FEATURES),
    )
    categories = (
        *(f"symbol:{value}" for value in (*_DEFAULT_SYMBOLS, "__OTHER__")),
        *(f"trigger:{value}" for value in _DEFAULT_TRIGGERS),
        *(f"session:{value}" for value in _DEFAULT_SESSIONS),
        *(f"weekday:{value}" for value in _DEFAULT_WEEKDAYS),
        *(f"execution:{value}" for value in _DEFAULT_EXECUTIONS),
        *(f"fvg_state:{value}" for value in _DEFAULT_FVG_STATES),
        *(f"orca_mode:{value}" for value in _DEFAULT_ORCA_MODES),
    )
    return tuple(numeric + categories)


def _valid_feature_name(name: str) -> bool:
    if name in _DIRECT_NUMERIC_FEATURES:
        return True
    if name.startswith(("factor_alignment:", "factor_present:")):
        _, _, token = name.partition(":")
        return bool(token and _TOKEN_RE.fullmatch(token))
    for prefix in _CATEGORY_PREFIXES:
        if name.startswith(prefix):
            token = name[len(prefix) :]
            return bool(token and _TOKEN_RE.fullmatch(token))
    return False


def _is_post_decision_key(raw_name: Any) -> bool:
    name = str(raw_name).strip().lower().replace("-", "_")
    if name in _EXACT_POST_DECISION_FIELDS:
        return True
    if name.startswith("actual_") and any(
        token in name for token in ("fill", "touch", "retest", "exit", "outcome")
    ):
        return True
    if "touch" in name and any(
        token in name for token in ("actual", "first", "strict", "exact", "time")
    ):
        return True
    if "retest" in name and any(
        token in name for token in ("actual", "exact", "time", "seconds", "at_utc")
    ):
        return True
    if name.startswith("outcome") or name.endswith("_outcome"):
        return True
    return False


def _reject_post_decision_fields(value: Any, *, path: str = "candidate") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if _is_post_decision_key(key):
                raise ValueError(
                    f"RF decision features contain post-decision field {child_path}"
                )
            _reject_post_decision_fields(child, path=child_path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_post_decision_fields(child, path=f"{path}[{index}]")


def _strict_keys(
    mapping: Mapping[str, Any],
    *,
    expected: set[str],
    name: str,
) -> None:
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{name} keys mismatch; missing={missing}, unknown={unknown}")


@dataclass(frozen=True)
class FrozenTree:
    feature_index: tuple[int, ...]
    threshold: tuple[float, ...]
    left_child: tuple[int, ...]
    right_child: tuple[int, ...]
    value: tuple[float, ...]

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        feature_count: int,
        probability: bool,
        name: str,
    ) -> "FrozenTree":
        if not isinstance(raw, Mapping):
            raise ValueError(f"{name} must be a mapping")
        expected = {
            "feature_index",
            "threshold",
            "left_child",
            "right_child",
            "value",
        }
        _strict_keys(raw, expected=expected, name=name)
        arrays: dict[str, tuple[Any, ...]] = {}
        for field in expected:
            source = raw[field]
            if not isinstance(source, Sequence) or isinstance(
                source, (str, bytes, bytearray)
            ):
                raise ValueError(f"{name}.{field} must be an array")
            arrays[field] = tuple(source)
        lengths = {len(array) for array in arrays.values()}
        if lengths != {next(iter(lengths), 0)} or not lengths or 0 in lengths:
            raise ValueError(f"{name} arrays must have the same non-zero length")
        size = len(arrays["value"])
        try:
            feature_index = tuple(int(value) for value in arrays["feature_index"])
            left = tuple(int(value) for value in arrays["left_child"])
            right = tuple(int(value) for value in arrays["right_child"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} index arrays must contain integers") from exc
        for field_name, source, parsed in (
            ("feature_index", arrays["feature_index"], feature_index),
            ("left_child", arrays["left_child"], left),
            ("right_child", arrays["right_child"], right),
        ):
            if any(isinstance(value, bool) for value in source) or any(
                float(source_value) != parsed_value
                for source_value, parsed_value in zip(source, parsed)
            ):
                raise ValueError(f"{name}.{field_name} must contain integers")
        thresholds = tuple(
            _finite(value, name=f"{name}.threshold") for value in arrays["threshold"]
        )
        values = tuple(
            _finite(value, name=f"{name}.value") for value in arrays["value"]
        )
        parents = [0] * size
        for index, feature in enumerate(feature_index):
            is_leaf = feature == -1
            if is_leaf:
                if left[index] != -1 or right[index] != -1:
                    raise ValueError(f"{name} leaf {index} must have -1 children")
            else:
                if not 0 <= feature < feature_count:
                    raise ValueError(f"{name} node {index} has invalid feature index")
                for child in (left[index], right[index]):
                    if not 0 <= child < size or child == index:
                        raise ValueError(f"{name} node {index} has invalid child")
                    parents[child] += 1
                if left[index] == right[index]:
                    raise ValueError(f"{name} node {index} reuses one child")
            if probability and is_leaf and not 0.0 <= values[index] <= 1.0:
                raise ValueError(f"{name} probability leaf is outside [0,1]")
        if parents[0] != 0 or any(count != 1 for count in parents[1:]):
            raise ValueError(f"{name} must be one rooted tree")
        visited: set[int] = set()
        active: set[int] = set()

        def visit(index: int) -> None:
            if index in active:
                raise ValueError(f"{name} contains a cycle")
            if index in visited:
                return
            active.add(index)
            if feature_index[index] != -1:
                visit(left[index])
                visit(right[index])
            active.remove(index)
            visited.add(index)

        visit(0)
        if len(visited) != size:
            raise ValueError(f"{name} contains unreachable nodes")
        return cls(feature_index, thresholds, left, right, values)

    def predict(self, features: Sequence[float]) -> float:
        index = 0
        while self.feature_index[index] != -1:
            feature = self.feature_index[index]
            index = (
                self.left_child[index]
                if float(features[feature]) <= self.threshold[index]
                else self.right_child[index]
            )
        return self.value[index]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "feature_index": list(self.feature_index),
            "threshold": list(self.threshold),
            "left_child": list(self.left_child),
            "right_child": list(self.right_child),
            "value": list(self.value),
        }


@dataclass(frozen=True)
class FrozenForest:
    trees: tuple[FrozenTree, ...]

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        feature_count: int,
        probability: bool,
        name: str,
    ) -> "FrozenForest":
        if not isinstance(raw, Mapping):
            raise ValueError(f"{name} must be a mapping")
        _strict_keys(raw, expected={"trees"}, name=name)
        source = raw["trees"]
        if not isinstance(source, Sequence) or isinstance(
            source, (str, bytes, bytearray)
        ):
            raise ValueError(f"{name}.trees must be an array")
        if not source:
            raise ValueError(f"{name}.trees cannot be empty")
        return cls(
            tuple(
                FrozenTree.from_mapping(
                    tree,
                    feature_count=feature_count,
                    probability=probability,
                    name=f"{name}.trees[{index}]",
                )
                for index, tree in enumerate(source)
            )
        )

    def predictions(self, features: Sequence[float]) -> tuple[float, ...]:
        return tuple(tree.predict(features) for tree in self.trees)

    def to_mapping(self) -> dict[str, Any]:
        return {"trees": [tree.to_mapping() for tree in self.trees]}


@dataclass(frozen=True)
class RFCandidateProfile:
    profile_sha256: str
    created_at_utc: datetime
    trained_through_utc: datetime
    strategy_contract: Mapping[str, Any]
    feature_registry: tuple[str, ...]
    uncertainty_z: float
    fill_forest: FrozenForest
    tp_given_fill_forest: FrozenForest
    gross_r_forest: FrozenForest

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RFCandidateProfile":
        if not isinstance(raw, Mapping):
            raise ValueError("profile must be a mapping")
        expected = {
            "schema",
            "profile_sha256",
            "created_at_utc",
            "trained_through_utc",
            "strategy_contract",
            "feature_registry",
            "uncertainty_z",
            "forests",
        }
        _strict_keys(raw, expected=expected, name="profile")
        if raw["schema"] != RF_CANDIDATE_PROFILE_SCHEMA:
            raise ValueError("unexpected RF candidate profile schema")
        supplied_hash = str(raw["profile_sha256"] or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", supplied_hash):
            raise ValueError("profile_sha256 must be a SHA-256 hex digest")
        calculated_hash = canonical_rf_profile_sha256(raw)
        if supplied_hash != calculated_hash:
            raise ValueError("RF candidate profile hash mismatch")
        created = _utc(raw["created_at_utc"], name="created_at_utc")
        trained = _utc(raw["trained_through_utc"], name="trained_through_utc")
        if trained > created:
            raise ValueError("trained_through_utc cannot follow created_at_utc")
        contract = raw["strategy_contract"]
        if not isinstance(contract, Mapping) or not contract:
            raise ValueError("strategy_contract must be a non-empty mapping")
        contract_copy = dict(_json_safe(contract, path="strategy_contract"))
        for required in ("strategy_version", "factor_contract", "target_contract"):
            if not str(contract_copy.get(required) or "").strip():
                raise ValueError(f"strategy_contract.{required} is required")
        registry_raw = raw["feature_registry"]
        if not isinstance(registry_raw, Sequence) or isinstance(
            registry_raw, (str, bytes, bytearray)
        ):
            raise ValueError("feature_registry must be an array")
        registry = tuple(str(name) for name in registry_raw)
        if not registry or len(set(registry)) != len(registry):
            raise ValueError("feature_registry must be non-empty and unique")
        invalid = [name for name in registry if not _valid_feature_name(name)]
        if invalid:
            raise ValueError(f"unsupported feature registry entries: {invalid}")
        uncertainty_z = _finite(raw["uncertainty_z"], name="uncertainty_z")
        if uncertainty_z < 0.0:
            raise ValueError("uncertainty_z must be nonnegative")
        forests = raw["forests"]
        if not isinstance(forests, Mapping):
            raise ValueError("forests must be a mapping")
        _strict_keys(
            forests,
            expected={
                "fill_probability",
                "tp_given_fill_probability",
                "gross_r_given_fill",
            },
            name="forests",
        )
        return cls(
            profile_sha256=supplied_hash,
            created_at_utc=created,
            trained_through_utc=trained,
            strategy_contract=contract_copy,
            feature_registry=registry,
            uncertainty_z=uncertainty_z,
            fill_forest=FrozenForest.from_mapping(
                forests["fill_probability"],
                feature_count=len(registry),
                probability=True,
                name="forests.fill_probability",
            ),
            tp_given_fill_forest=FrozenForest.from_mapping(
                forests["tp_given_fill_probability"],
                feature_count=len(registry),
                probability=True,
                name="forests.tp_given_fill_probability",
            ),
            gross_r_forest=FrozenForest.from_mapping(
                forests["gross_r_given_fill"],
                feature_count=len(registry),
                probability=False,
                name="forests.gross_r_given_fill",
            ),
        )

    @classmethod
    def load(cls, path: Path | str) -> "RFCandidateProfile":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(payload)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": RF_CANDIDATE_PROFILE_SCHEMA,
            "profile_sha256": self.profile_sha256,
            "created_at_utc": _iso(self.created_at_utc),
            "trained_through_utc": _iso(self.trained_through_utc),
            "strategy_contract": dict(self.strategy_contract),
            "feature_registry": list(self.feature_registry),
            "uncertainty_z": self.uncertainty_z,
            "forests": {
                "fill_probability": self.fill_forest.to_mapping(),
                "tp_given_fill_probability": (self.tp_given_fill_forest.to_mapping()),
                "gross_r_given_fill": self.gross_r_forest.to_mapping(),
            },
        }


def _first(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _token(value: Any, *, name: str, upper: bool = True) -> str:
    token = str(value or "").strip()
    if not token:
        raise ValueError(f"{name} is required")
    token = token.upper() if upper else token.lower()
    if not _TOKEN_RE.fullmatch(token):
        raise ValueError(f"{name} contains unsupported characters")
    return token


def _session_label(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "MISSING"
    token = raw.upper().replace("-", "_").replace(" ", "_")
    if not _TOKEN_RE.fullmatch(token) or token not in _DEFAULT_SESSIONS:
        raise ValueError(f"unsupported explicit session_label: {raw!r}")
    return token


def _execution_token(value: Any) -> str:
    token = str(value or "UNKNOWN").strip().upper().replace("-", "_")
    if "LIMIT" in token or "RETEST" in token:
        return "EXACT_LIMIT"
    if "MARKET" in token:
        return "MARKET"
    return token if token and _TOKEN_RE.fullmatch(token) else "UNKNOWN"


def _factor_context(
    factor_vector: Any,
    *,
    side: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    numeric: dict[str, float] = {}
    meta = {
        "bias": "NEUTRAL",
        "fvg_side": "NEUTRAL",
        "score_long": 0.0,
        "score_short": 0.0,
        "margin_long": 0.0,
        "margin_short": 0.0,
    }
    rows: Sequence[Any] = ()
    if isinstance(factor_vector, Mapping):
        meta.update(
            {
                "bias": str(factor_vector.get("bias") or "NEUTRAL").upper(),
                "fvg_side": str(factor_vector.get("fvg_side") or "NEUTRAL").upper(),
                "score_long": _optional_finite(
                    factor_vector.get("score_long"), name="score_long"
                )
                or 0.0,
                "score_short": _optional_finite(
                    factor_vector.get("score_short"), name="score_short"
                )
                or 0.0,
                "margin_long": _optional_finite(
                    factor_vector.get("margin_long"), name="margin_long"
                )
                or 0.0,
                "margin_short": _optional_finite(
                    factor_vector.get("margin_short"), name="margin_short"
                )
                or 0.0,
            }
        )
        raw_rows = factor_vector.get("factors")
        if isinstance(raw_rows, Sequence) and not isinstance(
            raw_rows, (str, bytes, bytearray)
        ):
            rows = raw_rows
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("key"):
            raise ValueError("factor_vector contains a malformed factor row")
        key = _token(row["key"], name="factor key", upper=False)
        if key in seen:
            raise ValueError(f"factor_vector contains duplicate factor {key}")
        seen.add(key)
        present = bool(row.get("present"))
        vote = str(row.get("vote_side") or "NEUTRAL").upper()
        alignment = 0.0
        if present and vote in {"LONG", "SHORT"}:
            alignment = 1.0 if vote == side else -1.0
        numeric[f"factor_alignment:{key}"] = alignment
        numeric[f"factor_present:{key}"] = float(present)
    return numeric, meta


def _number_with_missing(
    values: dict[str, float],
    missing: list[str],
    name: str,
    raw: Any,
    *,
    minimum: Optional[float] = None,
) -> Optional[float]:
    parsed = _optional_finite(raw, name=name, minimum=minimum)
    if parsed is None:
        values[name] = 0.0
        values[f"{name}_missing"] = 1.0
        missing.append(name)
    else:
        values[name] = parsed
        values[f"{name}_missing"] = 0.0
    return parsed


def _orca_context(
    candidate: Mapping[str, Any],
    explicit: Optional[Mapping[str, Any]],
    *,
    decision: datetime,
    values: dict[str, float],
    missing: list[str],
) -> str:
    source: Any = explicit
    if source is None:
        source = candidate.get("orca_snapshot", candidate.get("orca"))
    fields = (
        "orca_absorption_ratio",
        "orca_systemic_score",
        "orca_tail_event_probability",
        "orca_largest_eigenvalue",
        "orca_eigenvalue_ratio",
        "orca_spectral_gap",
        "orca_crisis_probability",
        "orca_bcd_auc",
    )
    aliases = {
        "orca_absorption_ratio": ("absorption_ratio", "orca_absorption_ratio"),
        "orca_systemic_score": ("systemic_score", "orca_systemic_score"),
        "orca_tail_event_probability": (
            "tail_event_probability",
            "orca_tail_event_probability",
        ),
        "orca_largest_eigenvalue": (
            "largest_eigenvalue",
            "orca_largest_eigenvalue",
        ),
        "orca_eigenvalue_ratio": ("eigenvalue_ratio", "orca_eigenvalue_ratio"),
        "orca_spectral_gap": ("spectral_gap", "orca_spectral_gap"),
        "orca_crisis_probability": (
            "crisis_probability",
            "orca_crisis_probability",
        ),
        "orca_bcd_auc": ("bcd_auc", "orca_bcd_auc"),
    }
    if source is None:
        values["orca_snapshot_available"] = 0.0
        for field in fields:
            _number_with_missing(values, missing, field, None)
        return "MISSING"
    if not isinstance(source, Mapping):
        raise ValueError("orca_snapshot must be a mapping")
    known_raw = _first(source, ("known_at_utc", "known_at"))
    trained_raw = _first(
        source,
        ("trained_through_utc", "trained_through", "measured_through_utc"),
    )
    if known_raw is None or trained_raw is None:
        raise ValueError("ORCA features require known_at_utc and trained_through_utc")
    known = _utc(known_raw, name="orca.known_at_utc")
    trained = _utc(trained_raw, name="orca.trained_through_utc")
    if trained > known:
        raise ValueError("ORCA trained_through_utc cannot follow known_at_utc")
    if known > decision or trained > decision:
        raise ValueError("ORCA snapshot is future-dated for this decision")
    values["orca_snapshot_available"] = 1.0
    for field in fields:
        parsed = _number_with_missing(
            values,
            missing,
            field,
            _first(source, aliases[field]),
        )
        if (
            field
            in {
                "orca_absorption_ratio",
                "orca_tail_event_probability",
                "orca_eigenvalue_ratio",
                "orca_crisis_probability",
                "orca_bcd_auc",
            }
            and parsed is not None
            and parsed < 0.0
        ):
            raise ValueError(f"{field} must be nonnegative")
        if field == "orca_bcd_auc" and parsed is not None and parsed > 1.0:
            raise ValueError("orca_bcd_auc must be at most 1")
    mode = str(_first(source, ("market_mode", "regime", "mode")) or "MISSING")
    mode = mode.strip().upper().replace("-", "_").replace(" ", "_")
    return mode if mode and _TOKEN_RE.fullmatch(mode) else "MISSING"


def build_rf_candidate_features(
    candidate: Mapping[str, Any],
    *,
    feature_registry: Optional[Sequence[str]] = None,
    symbol: Optional[str] = None,
    decision_time_utc: Any = None,
    atr_h1_14: Any = None,
    atr_m15_14: Any = None,
    decision_bid: Any = None,
    decision_ask: Any = None,
    session_label: Any = None,
    intended_execution_mode: Any = None,
    orca_snapshot: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build one deterministic, decision-time feature vector."""

    if not isinstance(candidate, Mapping):
        raise ValueError("candidate must be a mapping")
    _reject_post_decision_fields(candidate)
    if orca_snapshot is not None:
        _reject_post_decision_fields(orca_snapshot, path="orca_snapshot")
    registry = tuple(feature_registry or default_rf_feature_registry())
    if not registry or len(set(registry)) != len(registry):
        raise ValueError("feature_registry must be non-empty and unique")
    invalid = [name for name in registry if not _valid_feature_name(str(name))]
    if invalid:
        raise ValueError(f"unsupported feature registry entries: {invalid}")
    resolved_symbol = _token(
        symbol if symbol is not None else candidate.get("symbol"),
        name="symbol",
    )
    side = _token(candidate.get("side"), name="side")
    if side not in {"LONG", "SHORT"}:
        raise ValueError("side must be LONG or SHORT")
    trigger = _token(
        _first(candidate, ("trigger_kind", "entry_trigger", "trigger")),
        name="trigger_kind",
        upper=False,
    )
    trigger = _TRIGGER_ALIASES.get(trigger, trigger)
    decision_raw = decision_time_utc
    if decision_raw is None:
        decision_raw = _first(
            candidate,
            (
                "decision_time_utc",
                "decision_bar_close",
                "observed_at_utc",
                "timestamp_utc",
            ),
        )
    decision = _utc(decision_raw, name="decision_time_utc")
    factor_values, factor_meta = _factor_context(
        candidate.get("factor_vector"), side=side
    )
    values = dict(factor_values)
    missing: list[str] = []
    score_long = float(factor_meta["score_long"])
    score_short = float(factor_meta["score_short"])
    margin_long = float(factor_meta["margin_long"])
    margin_short = float(factor_meta["margin_short"])
    for_side = score_long if side == "LONG" else score_short
    against = score_short if side == "LONG" else score_long
    required_margin = margin_long if side == "LONG" else margin_short
    values.update(
        {
            "score_long": score_long,
            "score_short": score_short,
            "score_delta_long_minus_short": score_long - score_short,
            "score_for_side": for_side,
            "score_against_side": against,
            "required_margin_for_side": required_margin,
            "score_edge_for_side": for_side - against - required_margin,
            "production_bias_aligned": float(factor_meta["bias"] == side),
            "side_sign": 1.0 if side == "LONG" else -1.0,
        }
    )
    priority = _number_with_missing(
        values,
        missing,
        "production_priority",
        _first(candidate, ("production_priority", "shadow_candidate_rank")),
        minimum=0.0,
    )
    if priority is None:
        values["production_priority"] = 0.0
    fvg_side = str(
        _first(candidate, ("fvg_regime", "fvg_side"))
        or factor_meta["fvg_side"]
        or "NEUTRAL"
    ).upper()
    if fvg_side in {"LONG", "SHORT"}:
        fvg_alignment = 1.0 if fvg_side == side else -1.0
        fvg_state = "ALIGNED" if fvg_side == side else "OPPOSED"
    elif fvg_side in {"NEUTRAL", "NONE", "NO_FVG"}:
        fvg_alignment = 0.0
        fvg_state = "NEUTRAL"
    else:
        fvg_alignment = 0.0
        fvg_state = "MISSING"
    values["fvg_alignment"] = fvg_alignment
    fvg_age = _optional_finite(
        candidate.get("fvg_age_bars"), name="fvg_age_bars", minimum=0.0
    )
    if fvg_age is None:
        values["fvg_age_log"] = 0.0
        values["fvg_age_missing"] = 1.0
        missing.append("fvg_age_log")
    else:
        values["fvg_age_log"] = math.log1p(fvg_age)
        values["fvg_age_missing"] = 0.0
    h1_atr = _optional_finite(
        atr_h1_14
        if atr_h1_14 is not None
        else _first(candidate, ("atr_h1_14", "atr_h1")),
        name="atr_h1_14",
        minimum=0.0,
    )
    m15_atr = _optional_finite(
        atr_m15_14
        if atr_m15_14 is not None
        else _first(candidate, ("atr_m15_14", "atr_m15")),
        name="atr_m15_14",
        minimum=0.0,
    )
    entry = _optional_finite(
        _first(candidate, ("planned_entry", "entry_price")),
        name="planned_entry",
    )
    stop = _optional_finite(
        _first(candidate, ("planned_stop", "stop_price")),
        name="planned_stop",
    )
    direct_stop_atr = _first(
        candidate,
        ("stop_atr_h1", "stop_distance_atr_h1", "technical_risk_atr"),
    )
    derived_stop_atr = direct_stop_atr
    if derived_stop_atr is None and None not in {entry, stop, h1_atr} and h1_atr:
        derived_stop_atr = abs(float(entry) - float(stop)) / float(h1_atr)
    _number_with_missing(
        values,
        missing,
        "stop_atr_h1",
        derived_stop_atr,
        minimum=0.0,
    )
    entry_min = _optional_finite(candidate.get("entry_min"), name="entry_min")
    entry_max = _optional_finite(candidate.get("entry_max"), name="entry_max")
    zone_atr: Optional[float] = None
    if None not in {entry_min, entry_max, m15_atr} and m15_atr:
        zone_atr = abs(float(entry_max) - float(entry_min)) / float(m15_atr)
    _number_with_missing(
        values,
        missing,
        "entry_zone_atr_m15",
        zone_atr,
        minimum=0.0,
    )
    bid = _optional_finite(
        decision_bid
        if decision_bid is not None
        else _first(candidate, ("decision_bid", "bid")),
        name="decision_bid",
    )
    ask = _optional_finite(
        decision_ask
        if decision_ask is not None
        else _first(candidate, ("decision_ask", "ask")),
        name="decision_ask",
    )
    quote_distance: Optional[float] = None
    if None not in {entry, m15_atr} and m15_atr:
        executable = ask if side == "LONG" else bid
        if executable is not None:
            quote_distance = (
                (float(executable) - float(entry)) / float(m15_atr)
                if side == "LONG"
                else (float(entry) - float(executable)) / float(m15_atr)
            )
    _number_with_missing(
        values,
        missing,
        "quote_to_entry_atr_m15",
        quote_distance,
    )
    tp_values = candidate.get("tp_prices")
    if not isinstance(tp_values, Sequence) or isinstance(
        tp_values, (str, bytes, bytearray)
    ):
        single_tp = candidate.get("tp_price")
        tp_values = () if single_tp is None else (single_tp,)
    tp_distances = []
    if entry is not None:
        for raw_tp in tp_values:
            tp = _optional_finite(raw_tp, name="tp_price")
            if tp is not None:
                tp_distances.append(abs(tp - entry))
    tp_atr = (
        min(tp_distances) / float(h1_atr)
        if tp_distances and h1_atr is not None and h1_atr > 0.0
        else None
    )
    _number_with_missing(
        values,
        missing,
        "tp1_distance_atr_h1",
        tp_atr,
        minimum=0.0,
    )
    _number_with_missing(
        values,
        missing,
        "vol_r",
        _first(candidate, ("vol_r", "vol_R")),
        minimum=0.0,
    )
    _number_with_missing(
        values,
        missing,
        "vol_em_1d",
        _first(candidate, ("vol_em_1d", "em_1d")),
        minimum=0.0,
    )
    _number_with_missing(
        values,
        missing,
        "vol_tp1_em_ratio",
        candidate.get("vol_tp1_em_ratio"),
        minimum=0.0,
    )
    spread_r = _first(candidate, ("spread_r", "decision_spread_r"))
    if spread_r is None and None not in {bid, ask, entry, stop}:
        risk = abs(float(entry) - float(stop))
        if risk > 0.0:
            spread_r = abs(float(ask) - float(bid)) / risk
    _number_with_missing(
        values,
        missing,
        "spread_r",
        spread_r,
        minimum=0.0,
    )
    orca_mode = _orca_context(
        candidate,
        orca_snapshot,
        decision=decision,
        values=values,
        missing=missing,
    )
    session = _session_label(
        session_label
        if session_label is not None
        else _first(
            candidate,
            ("session_label", "trading_session", "market_session", "session"),
        )
    )
    weekday = _DEFAULT_WEEKDAYS[decision.weekday()]
    execution = _execution_token(
        intended_execution_mode
        if intended_execution_mode is not None
        else _first(
            candidate,
            (
                "intended_execution_mode",
                "execution_method",
                "execution_mode",
                "entry_method",
                "order_type",
            ),
        )
    )
    actual_categories = {
        "symbol:": resolved_symbol,
        "trigger:": trigger,
        "session:": session,
        "weekday:": weekday,
        "execution:": execution,
        "fvg_state:": fvg_state,
        "orca_mode:": orca_mode,
    }
    oov: list[str] = []
    vector_values: dict[str, float] = {}
    for name in registry:
        feature_name = str(name)
        if any(feature_name.startswith(prefix) for prefix in _CATEGORY_PREFIXES):
            prefix = next(
                prefix
                for prefix in _CATEGORY_PREFIXES
                if feature_name.startswith(prefix)
            )
            expected = feature_name[len(prefix) :]
            actual = actual_categories[prefix]
            available = {
                item[len(prefix) :] for item in registry if str(item).startswith(prefix)
            }
            if actual not in available and f"{prefix}{actual}" not in oov:
                oov.append(f"{prefix}{actual}")
            vector_values[feature_name] = float(
                expected == actual
                or (expected == "__OTHER__" and actual not in available)
            )
        else:
            vector_values[feature_name] = float(values.get(feature_name, 0.0))
    return {
        "schema": RF_CANDIDATE_FEATURE_SCHEMA,
        "decision_time_utc": _iso(decision),
        "symbol": resolved_symbol,
        "side": side,
        "trigger_kind": trigger,
        "feature_registry": list(registry),
        "values": vector_values,
        "missing": sorted(set(missing)),
        "oov": sorted(set(oov)),
    }


def _population_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = fmean(values)
    return math.sqrt(max(0.0, fmean(value * value for value in values) - mean * mean))


def tree_dispersion_penalized_score(
    fill_tree_predictions: Sequence[Any],
    gross_r_tree_predictions: Sequence[Any],
    *,
    causal_cost_r: Any,
    penalty_multiplier: Any,
) -> dict[str, float]:
    """Return the shared empirical tree-dispersion ranking score.

    The penalty is a model-disagreement heuristic.  It is not a calibrated
    statistical coverage guarantee or calibrated uncertainty estimate.
    """

    fill_trees = tuple(
        _finite(value, name="fill_tree_prediction") for value in fill_tree_predictions
    )
    gross_trees = tuple(
        _finite(value, name="gross_r_tree_prediction")
        for value in gross_r_tree_predictions
    )
    if not fill_trees or not gross_trees:
        raise ValueError("tree prediction arrays must be non-empty")
    if any(not 0.0 <= value <= 1.0 for value in fill_trees):
        raise ValueError("fill tree predictions must be in [0, 1]")
    cost = _finite(causal_cost_r, name="causal_cost_r")
    multiplier = _finite(penalty_multiplier, name="penalty_multiplier")
    if cost < 0.0:
        raise ValueError("causal_cost_r must be nonnegative")
    if multiplier < 0.0:
        raise ValueError("penalty_multiplier must be nonnegative")

    p_fill = fmean(fill_trees)
    expected_gross = fmean(gross_trees)
    expected_net = p_fill * (expected_gross - cost)
    # This is the exact variance of F * (G - cost) under the empirical
    # Cartesian product of the fill-tree and gross-tree predictions.
    fill_second = fmean(value * value for value in fill_trees)
    net_gross = tuple(value - cost for value in gross_trees)
    gross_second = fmean(value * value for value in net_gross)
    dispersion = math.sqrt(
        max(0.0, fill_second * gross_second - expected_net * expected_net)
    )
    penalty = multiplier * dispersion
    return {
        "expected_gross_r_given_fill": expected_gross,
        "expected_net_r": expected_net,
        "tree_dispersion_r": dispersion,
        "tree_dispersion_penalty_r": penalty,
        "dispersion_penalized_score_r": expected_net - penalty,
    }


class LiveRFCandidateScorer:
    """Pure frozen-forest inference; never returns an execution action."""

    def __init__(
        self,
        profile: RFCandidateProfile,
        *,
        expected_strategy_version: str,
        expected_factor_contract: str,
        expected_target_contract: str,
    ) -> None:
        if not isinstance(profile, RFCandidateProfile):
            raise TypeError("profile must be RFCandidateProfile")
        expected = {
            "strategy_version": str(expected_strategy_version or "").strip(),
            "factor_contract": str(expected_factor_contract or "").strip(),
            "target_contract": str(expected_target_contract or "").strip(),
        }
        for name, value in expected.items():
            if not value:
                raise ValueError(f"expected_{name} is required")
            actual = str(profile.strategy_contract.get(name) or "").strip()
            if actual != value:
                raise ValueError(
                    f"RF strategy contract mismatch for {name}: "
                    f"expected {value!r}, profile has {actual!r}"
                )
        self.profile = profile
        self.expected_strategy_contract = expected
        self.model_id = profile.profile_sha256[:24]

    @classmethod
    def load(
        cls,
        path: Path | str | None,
        *,
        expected_strategy_version: str,
        expected_factor_contract: str,
        expected_target_contract: str,
    ) -> Optional["LiveRFCandidateScorer"]:
        if path is None:
            return None
        try:
            profile = RFCandidateProfile.load(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return cls(
            profile,
            expected_strategy_version=expected_strategy_version,
            expected_factor_contract=expected_factor_contract,
            expected_target_contract=expected_target_contract,
        )

    def score(
        self,
        candidate: Mapping[str, Any],
        *,
        causal_cost_r: Any,
        symbol: Optional[str] = None,
        decision_time_utc: Any = None,
        atr_h1_14: Any = None,
        atr_m15_14: Any = None,
        decision_bid: Any = None,
        decision_ask: Any = None,
        session_label: Any = None,
        intended_execution_mode: Any = None,
        orca_snapshot: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        cost = _finite(causal_cost_r, name="causal_cost_r")
        if cost < 0.0:
            raise ValueError("causal_cost_r must be nonnegative")
        features = build_rf_candidate_features(
            candidate,
            feature_registry=self.profile.feature_registry,
            symbol=symbol,
            decision_time_utc=decision_time_utc,
            atr_h1_14=atr_h1_14,
            atr_m15_14=atr_m15_14,
            decision_bid=decision_bid,
            decision_ask=decision_ask,
            session_label=session_label,
            intended_execution_mode=intended_execution_mode,
            orca_snapshot=orca_snapshot,
        )
        decision = _utc(features["decision_time_utc"], name="decision_time_utc")
        if self.profile.created_at_utc > decision:
            raise ValueError("RF profile was not created by this decision")
        if self.profile.trained_through_utc > decision:
            raise ValueError("RF profile is future-dated for this decision")
        vector = tuple(
            float(features["values"][name]) for name in self.profile.feature_registry
        )
        fill_trees = self.profile.fill_forest.predictions(vector)
        tp_trees = self.profile.tp_given_fill_forest.predictions(vector)
        gross_trees = self.profile.gross_r_forest.predictions(vector)
        p_fill = fmean(fill_trees)
        p_tp_given_fill = fmean(tp_trees)
        score = tree_dispersion_penalized_score(
            fill_trees,
            gross_trees,
            causal_cost_r=cost,
            penalty_multiplier=self.profile.uncertainty_z,
        )
        expected_gross = score["expected_gross_r_given_fill"]
        expected_net = score["expected_net_r"]
        utility_std = score["tree_dispersion_r"]
        conservative = score["dispersion_penalized_score_r"]
        status = "SCORED"
        if features["oov"]:
            status = "SCORED_WITH_OOV"
        output = {
            "rf_status": status,
            "rf_profile_schema": RF_CANDIDATE_PROFILE_SCHEMA,
            "rf_profile_sha256": self.profile.profile_sha256,
            "rf_model_id": self.model_id,
            "rf_executing": False,
            "rf_fill_probability": p_fill,
            "rf_tp_given_fill_probability": p_tp_given_fill,
            "rf_tp_probability": p_fill * p_tp_given_fill,
            "rf_expected_gross_r_given_fill": expected_gross,
            "rf_causal_cost_r": cost,
            "rf_expected_net_r": expected_net,
            "rf_fill_tree_std": _population_std(fill_trees),
            "rf_tp_given_fill_tree_std": _population_std(tp_trees),
            "rf_gross_r_tree_std": _population_std(gross_trees),
            "rf_tree_dispersion_r": utility_std,
            "rf_uncertainty_z": self.profile.uncertainty_z,
            "rf_tree_dispersion_penalty_multiplier": self.profile.uncertainty_z,
            "rf_tree_dispersion_penalty_r": score["tree_dispersion_penalty_r"],
            "rf_conservative_intrinsic_score_r": conservative,
            "rf_feature_missing": list(features["missing"]),
            "rf_feature_oov": list(features["oov"]),
        }
        if any(not key.startswith("rf_") for key in output):
            raise AssertionError("RF scorer emitted a non-rf annotation")
        return output


def _ranking_number(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return -math.inf
    return parsed if math.isfinite(parsed) else -math.inf


def _stable_candidate_id(candidate: Mapping[str, Any]) -> str:
    for name in (
        "rf_stable_id",
        "opportunity_id",
        "candidate_id",
        "shadow_opportunity_id",
        "setup_id",
    ):
        value = candidate.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    payload = {
        str(key): value
        for key, value in candidate.items()
        if not str(key).startswith("rf_")
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:24]


def rank_rf_candidates(
    candidates: Sequence[Mapping[str, Any]],
    scorer: Optional[LiveRFCandidateScorer] = None,
    *,
    causal_costs_r: Optional[Sequence[Any]] = None,
    symbol: Optional[str] = None,
    decision_time_utc: Any = None,
    **score_kwargs: Any,
) -> list[dict[str, Any]]:
    """Rank copies by dispersion-penalized score and abstain at score <= 0."""

    if scorer is not None:
        if causal_costs_r is None or len(causal_costs_r) != len(candidates):
            raise ValueError("one causal cost is required per scored candidate")
    elif causal_costs_r is not None:
        raise ValueError("causal_costs_r requires a scorer")
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError("every candidate must be a mapping")
        row = dict(candidate)
        if scorer is not None:
            annotations = scorer.score(
                candidate,
                causal_cost_r=causal_costs_r[index],
                symbol=symbol,
                decision_time_utc=decision_time_utc,
                **score_kwargs,
            )
            if any(not key.startswith("rf_") for key in annotations):
                raise ValueError("RF scorer returned a non-rf field")
            row.update(annotations)
        row["rf_stable_id"] = _stable_candidate_id(row)
        row["rf_rank_position"] = None
        row["rf_selected"] = False
        row["rf_selection_status"] = "NOT_SELECTED"
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -_ranking_number(row.get("rf_conservative_intrinsic_score_r")),
            -_ranking_number(row.get("rf_tp_probability")),
            -_ranking_number(row.get("rf_expected_net_r")),
            str(row["rf_stable_id"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rf_rank_position"] = rank
        selected = (
            rank == 1
            and _ranking_number(row.get("rf_conservative_intrinsic_score_r")) > 0.0
        )
        row["rf_selected"] = selected
        if selected:
            row["rf_selection_status"] = "SELECTED_POSITIVE_SCORE"
        elif rank == 1:
            row["rf_selection_status"] = "ABSTAINED_NONPOSITIVE_SCORE"
    return rows


__all__ = [
    "SINGLE_TP_TARGET_CONTRACT",
    "RF_CANDIDATE_FEATURE_SCHEMA",
    "RF_CANDIDATE_PROFILE_SCHEMA",
    "FrozenForest",
    "FrozenTree",
    "LiveRFCandidateScorer",
    "RFCandidateProfile",
    "build_rf_candidate_features",
    "canonical_rf_profile_sha256",
    "default_rf_feature_registry",
    "rank_rf_candidates",
    "seal_rf_candidate_profile",
    "tree_dispersion_penalized_score",
]
