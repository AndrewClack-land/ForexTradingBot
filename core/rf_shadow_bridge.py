"""Fail-open bridge from shadow candidates to the portable RF scorer.

This module is deliberately diagnostic-only.  It receives candidates after
the production decision is final, deep-copies them, computes decision-time
inputs, and returns copies carrying only ``rf_*`` annotations.  It imports no
executor and exposes no order action.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Optional

from backtest.cost_model import CostProfile, estimate_cost_r
from core.rf_candidate_contract import (
    LiveRFCandidateScorer,
    RFCandidateProfile,
    rank_rf_candidates,
)


ORCA_MONITOR_SNAPSHOT_SCHEMA = "orca-monitor-snapshot-v1"
DEFAULT_ORCA_MAX_AGE_SECONDS = 129_600.0
ALLOWED_ORCA_MODEL_STATUSES = frozenset({"OOS_VALIDATED"})

_SNAPSHOT_KEYS = frozenset(
    {
        "schema_version",
        "generated_at_utc",
        "as_of_market_utc",
        "calculation_as_of_utc",
        "returns_start_utc",
        "freshness",
        "age_seconds",
        "expected_assets",
        "asset_count",
        "full_universe",
        "symbols",
        "spectral_schema",
        "spectral_config_id",
        "feature_registry_hash",
        "feature_values_hash",
        "features",
        "networks",
        "model",
        "regime",
        "risk_mode",
        "suggested_equity_exposure",
        "exposure_status",
        "regime_eligible",
        "executing",
        "snapshot_hash",
    }
)

_MODEL_KEYS = frozenset(
    {
        "status",
        "model_id",
        "model_artifact_hash",
        "prediction_hash",
        "as_of_market_utc",
        "known_at_utc",
        "trained_through_utc",
        "p_rally",
        "p_crash",
        "rally_rank",
        "crash_rank",
        "validation",
        "calibration_status",
    }
)


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


def _finite(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _probability(value: Any, *, name: str) -> Optional[float]:
    if value in (None, ""):
        return None
    parsed = _finite(value, name=name)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return parsed


def _strict_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    actual = frozenset(str(key) for key in value)
    if actual != expected:
        raise ValueError(
            f"{name} keys mismatch; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _sha256_text(value: Any, *, name: str) -> str:
    parsed = str(value or "")
    if len(parsed) != 64 or any(
        character not in "0123456789abcdef" for character in parsed
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return parsed


def compute_orca_snapshot_hash(snapshot: Mapping[str, Any]) -> str:
    """Reproduce the monitor's canonical snapshot integrity hash."""

    payload = dict(snapshot)
    payload.pop("snapshot_hash", None)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _json_safe_rf_annotations(value: Any, *, path: str = "rf") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite value")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise ValueError(f"{path} contains a non-string key")
            result[raw_key] = _json_safe_rf_annotations(
                item,
                path=f"{path}.{raw_key}",
            )
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_safe_rf_annotations(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{path} contains a non-JSON value")


def _causal_atr(
    frame: Any,
    *,
    decision_time_utc: datetime,
    period: int = 14,
) -> Optional[float]:
    """Calculate the strategy's simple ATR on rows known by the decision."""

    if frame is None or getattr(frame, "empty", True):
        return None
    try:
        index = frame.index
        aware_index = []
        for raw_timestamp in index:
            if hasattr(raw_timestamp, "to_pydatetime"):
                raw_timestamp = raw_timestamp.to_pydatetime()
            aware_index.append(_utc(raw_timestamp, name="bar timestamp"))
        eligible_positions = [
            position
            for position, timestamp in enumerate(aware_index)
            if timestamp <= decision_time_utc
        ]
        if len(eligible_positions) < period + 2:
            return None
        causal = frame.iloc[eligible_positions]
        high = causal["high"].astype(float)
        low = causal["low"].astype(float)
        previous_close = causal["close"].astype(float).shift(1)
        true_range = (
            (high - low)
            .combine(
                (high - previous_close).abs(),
                max,
            )
            .combine(
                (low - previous_close).abs(),
                max,
            )
        )
        result = float(true_range.rolling(period).mean().iloc[-1])
    except Exception:
        return None
    return result if math.isfinite(result) and result > 0.0 else None


def _selected_network(networks: Any) -> Mapping[str, Any]:
    if (
        not isinstance(networks, Sequence)
        or isinstance(networks, (str, bytes, bytearray))
        or not networks
    ):
        raise ValueError("snapshot.networks must be a non-empty array")
    rows = []
    for index, raw in enumerate(networks):
        if not isinstance(raw, Mapping):
            raise ValueError(f"snapshot.networks[{index}] must be an object")
        estimator = str(raw.get("estimator") or "")
        observations = int(raw.get("observation_count") or 0)
        rows.append((raw, estimator, observations))
    ewm = [row for row in rows if row[1].startswith("ewm_")]
    candidates = ewm or rows
    return max(candidates, key=lambda row: (row[2], row[1]))[0]


def adapt_orca_rf_snapshot(
    snapshot: Mapping[str, Any],
    *,
    decision_time_utc: Any,
    max_age_seconds: float = DEFAULT_ORCA_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    """Authenticate and adapt one already-read point-in-time snapshot."""

    decision = _utc(decision_time_utc, name="decision_time_utc")
    maximum_age = _finite(max_age_seconds, name="max_age_seconds")
    if maximum_age <= 0.0:
        raise ValueError("max_age_seconds must be positive")
    if not isinstance(snapshot, Mapping):
        raise ValueError("ORCA snapshot root must be an object")
    payload = deepcopy(dict(snapshot))
    _strict_keys(payload, _SNAPSHOT_KEYS, name="snapshot")
    if payload.get("schema_version") != ORCA_MONITOR_SNAPSHOT_SCHEMA:
        raise ValueError("unexpected ORCA monitor snapshot schema")
    supplied_hash = _sha256_text(
        payload.get("snapshot_hash"),
        name="snapshot.snapshot_hash",
    )
    if supplied_hash != compute_orca_snapshot_hash(payload):
        raise ValueError("ORCA snapshot hash mismatch")
    if payload.get("executing") is not False:
        raise ValueError("ORCA snapshot must declare executing=false")
    if payload.get("freshness") != "FRESH":
        raise ValueError("ORCA snapshot is not FRESH")

    generated = _utc(
        payload.get("generated_at_utc"),
        name="snapshot.generated_at_utc",
    )
    market_as_of = _utc(
        payload.get("as_of_market_utc"),
        name="snapshot.as_of_market_utc",
    )
    calculation_as_of = _utc(
        payload.get("calculation_as_of_utc"),
        name="snapshot.calculation_as_of_utc",
    )
    returns_start = _utc(
        payload.get("returns_start_utc"),
        name="snapshot.returns_start_utc",
    )
    if calculation_as_of != generated:
        raise ValueError("ORCA calculation_as_of must equal generated_at")
    if not returns_start <= market_as_of <= generated <= decision:
        raise ValueError("ORCA snapshot timestamps are future or out of order")
    published_age = _finite(payload.get("age_seconds"), name="age_seconds")
    expected_published_age = (generated - market_as_of).total_seconds()
    if published_age < 0.0 or abs(published_age - expected_published_age) > 1e-6:
        raise ValueError("ORCA snapshot age_seconds is inconsistent")
    if (decision - market_as_of).total_seconds() > maximum_age:
        raise ValueError("ORCA snapshot is stale at this decision")

    _sha256_text(
        payload.get("feature_registry_hash"),
        name="snapshot.feature_registry_hash",
    )
    _sha256_text(
        payload.get("feature_values_hash"),
        name="snapshot.feature_values_hash",
    )
    features = payload.get("features")
    if not isinstance(features, Mapping) or not features:
        raise ValueError("snapshot.features must be a non-empty object")
    for name, value in features.items():
        _finite(value, name=f"snapshot.features.{name}")
    symbols = payload.get("symbols")
    if not isinstance(symbols, Sequence) or isinstance(
        symbols, (str, bytes, bytearray)
    ):
        raise ValueError("snapshot.symbols must be an array")
    asset_count = int(payload.get("asset_count") or 0)
    expected_assets = int(payload.get("expected_assets") or 0)
    if asset_count < 2 or asset_count != len(symbols):
        raise ValueError("snapshot.asset_count does not match symbols")
    if payload.get("full_universe") is not True or expected_assets != asset_count:
        raise ValueError("ORCA snapshot requires the complete expected universe")
    if payload.get("regime_eligible") is not True:
        raise ValueError("ORCA snapshot regime is not eligible")

    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("snapshot.model must be an object")
    _strict_keys(model, _MODEL_KEYS, name="snapshot.model")
    model_status = str(model.get("status") or "")
    if model_status not in ALLOWED_ORCA_MODEL_STATUSES:
        raise ValueError("ORCA model status is not allowed")
    crash_probability = _probability(
        model.get("p_crash"),
        name="snapshot.model.p_crash",
    )
    _probability(model.get("p_rally"), name="snapshot.model.p_rally")
    _probability(model.get("rally_rank"), name="snapshot.model.rally_rank")
    _probability(model.get("crash_rank"), name="snapshot.model.crash_rank")
    model_market_raw = model.get("as_of_market_utc")
    if model_market_raw not in (None, ""):
        model_market = _utc(
            model_market_raw,
            name="snapshot.model.as_of_market_utc",
        )
        if model_market != market_as_of:
            raise ValueError("ORCA model market as_of does not match snapshot")
    model_known_raw = model.get("known_at_utc")
    model_known = (
        _utc(model_known_raw, name="snapshot.model.known_at_utc")
        if model_known_raw not in (None, "")
        else None
    )
    trained_raw = model.get("trained_through_utc")
    trained = (
        _utc(trained_raw, name="snapshot.model.trained_through_utc")
        if trained_raw not in (None, "")
        else market_as_of
    )
    if trained > market_as_of or (
        model_known is not None and not trained <= model_known <= generated
    ):
        raise ValueError("ORCA model timestamps are future or out of order")
    if crash_probability is not None and model_known is None:
        raise ValueError("ORCA p_crash requires model known_at_utc")

    network = _selected_network(payload.get("networks"))
    eigenvalues_raw = network.get("eigenvalues")
    if (
        not isinstance(eigenvalues_raw, Sequence)
        or isinstance(eigenvalues_raw, (str, bytes, bytearray))
        or not eigenvalues_raw
    ):
        raise ValueError("ORCA network eigenvalues must be a non-empty array")
    eigenvalues = [
        _finite(value, name="snapshot.network.eigenvalue") for value in eigenvalues_raw
    ]
    eigenvalue_mass = sum(eigenvalues)
    if eigenvalues[0] < 0.0 or eigenvalue_mass <= 0.0:
        raise ValueError("ORCA network eigenvalues are invalid")
    absorption_raw = network.get("absorption_ratios")
    if not isinstance(absorption_raw, Mapping) or not absorption_raw:
        raise ValueError("ORCA absorption_ratios must be a non-empty object")
    absorption_rows = sorted(
        (
            int(rank),
            _probability(value, name=f"absorption_ratios.{rank}"),
        )
        for rank, value in absorption_raw.items()
    )
    absorption_by_rank = dict(absorption_rows)
    absorption = absorption_by_rank.get(5, absorption_rows[-1][1])
    if absorption is None:
        raise ValueError("ORCA absorption ratio is missing")
    spectral_gap_raw = network.get("lambda1_lambda2")
    spectral_gap = (
        _finite(spectral_gap_raw, name="lambda1_lambda2")
        if spectral_gap_raw not in (None, "")
        else None
    )
    systemic_raw = network.get("marchenko_pastur_lambda1_excess")
    systemic_score = (
        _finite(systemic_raw, name="marchenko_pastur_lambda1_excess")
        if systemic_raw not in (None, "")
        else None
    )
    raw_mode = str(payload.get("risk_mode") or "UNAVAILABLE").upper()
    market_mode = {
        "RISK_ON": "RISK_ON",
        "RISK_OFF": "RISK_OFF",
        "CAUTION": "TRANSITION",
        "NEUTRAL": "TRANSITION",
    }.get(raw_mode, "MISSING")

    adapted: dict[str, Any] = {
        "known_at_utc": _iso(generated),
        "trained_through_utc": _iso(trained),
        "market_mode": market_mode,
        "absorption_ratio": absorption,
        "systemic_score": systemic_score,
        "tail_event_probability": crash_probability,
        "largest_eigenvalue": eigenvalues[0],
        "eigenvalue_ratio": eigenvalues[0] / eigenvalue_mass,
        "spectral_gap": spectral_gap,
        "crisis_probability": crash_probability,
    }
    return adapted


def load_orca_rf_snapshot(
    path: Path | str,
    *,
    decision_time_utc: Any,
    max_age_seconds: float = DEFAULT_ORCA_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    """Read exactly once, then authenticate and adapt that immutable value."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("ORCA snapshot root must be an object")
    return adapt_orca_rf_snapshot(
        payload,
        decision_time_utc=decision_time_utc,
        max_age_seconds=max_age_seconds,
    )


@dataclass(frozen=True)
class RFShadowBridge:
    """Score copied candidates without any route back to production."""

    scorer: Any
    cost_profile: Optional[CostProfile]
    orca_snapshot_path: Optional[Path] = None
    orca_max_age_seconds: float = DEFAULT_ORCA_MAX_AGE_SECONDS

    @classmethod
    def load(
        cls,
        profile_path: Path | str,
        *,
        cost_profile: Optional[CostProfile],
        expected_strategy_version: str,
        expected_factor_contract: str,
        expected_target_contract: str,
        orca_snapshot_path: Path | str | None = None,
    ) -> "RFShadowBridge":
        profile = RFCandidateProfile.load(profile_path)
        return cls(
            scorer=LiveRFCandidateScorer(
                profile,
                expected_strategy_version=expected_strategy_version,
                expected_factor_contract=expected_factor_contract,
                expected_target_contract=expected_target_contract,
            ),
            cost_profile=cost_profile,
            orca_snapshot_path=(
                Path(orca_snapshot_path) if orca_snapshot_path is not None else None
            ),
        )

    @property
    def model_id(self) -> str:
        return str(getattr(self.scorer, "model_id", "unknown"))

    def enrich_candidates(
        self,
        *,
        symbol: str,
        candidates: Sequence[Mapping[str, Any]],
        observed_at_utc: Any,
        decision_bar_close: Any,
        strategy_data: Optional[Mapping[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Return diagnostic copies; raise to the worker's fail-open boundary."""

        observed = _utc(observed_at_utc, name="observed_at_utc")
        decision = (
            _utc(decision_bar_close, name="decision_bar_close")
            if decision_bar_close is not None
            else observed
        )
        if decision > observed:
            raise ValueError("decision_bar_close cannot follow observed_at_utc")
        originals: list[dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError("every RF shadow candidate must be a mapping")
            originals.append(deepcopy(dict(candidate)))
        if self.cost_profile is None:
            for row in originals:
                row["rf_status"] = "UNAVAILABLE_COST_PROFILE"
                row["rf_executing"] = False
            return originals

        frames = strategy_data or {}
        atr_h1 = _causal_atr(
            frames.get("1H"),
            decision_time_utc=decision,
        )
        atr_m15 = _causal_atr(
            frames.get("15M"),
            decision_time_utc=decision,
        )
        orca_context: Optional[dict[str, Any]] = None
        orca_status = "UNSET"
        orca_hash: Optional[str] = None
        if self.orca_snapshot_path is not None:
            try:
                raw_snapshot = json.loads(
                    self.orca_snapshot_path.read_text(encoding="utf-8")
                )
                if not isinstance(raw_snapshot, Mapping):
                    raise ValueError("ORCA snapshot root must be an object")
                orca_context = adapt_orca_rf_snapshot(
                    raw_snapshot,
                    decision_time_utc=decision,
                    max_age_seconds=self.orca_max_age_seconds,
                )
                orca_hash = _sha256_text(
                    raw_snapshot.get("snapshot_hash"),
                    name="snapshot.snapshot_hash",
                )
                orca_status = "AVAILABLE"
            except Exception as exc:
                orca_context = None
                orca_hash = None
                orca_status = f"UNAVAILABLE:{type(exc).__name__}"

        annotated: list[dict[str, Any]] = []
        for index, original in enumerate(originals):
            # This is the single cost-model call for this candidate.  The same
            # estimate supplies both RF utility cost and the spread feature.
            cost = estimate_cost_r(
                self.cost_profile,
                symbol=symbol,
                side=original.get("side"),
                entry_price=original.get("entry_price"),
                stop_price=original.get("stop_price"),
            )
            scoring_copy = deepcopy(original)
            scoring_copy.setdefault("spread_r", cost.spread_r)
            annotations = self.scorer.score(
                scoring_copy,
                causal_cost_r=cost.total_cost_r,
                symbol=symbol,
                decision_time_utc=decision,
                atr_h1_14=atr_h1,
                atr_m15_14=atr_m15,
                orca_snapshot=orca_context,
            )
            if not isinstance(annotations, Mapping):
                raise ValueError("RF scorer output must be a mapping")
            if any(not str(key).startswith("rf_") for key in annotations):
                raise ValueError("RF scorer returned a non-rf field")
            safe_annotations = _json_safe_rf_annotations(annotations)
            row = deepcopy(original)
            row.update(safe_annotations)
            row.update(
                {
                    "rf_status": "SCORED",
                    "rf_executing": False,
                    "rf_cost_profile_id": self.cost_profile.profile_id,
                    "rf_cost_profile_sha256": (self.cost_profile.profile_sha256),
                    "rf_orca_snapshot_status": orca_status,
                    "rf_orca_snapshot_hash": orca_hash,
                    "rf_bridge_source_index": index,
                    "rf_stable_id": str(
                        original.get("shadow_trigger_signature")
                        or original.get("opportunity_id")
                        or original.get("trigger_event_id")
                        or f"{symbol}:{index}"
                    ),
                }
            )
            annotated.append(row)

        ranked = rank_rf_candidates(annotated)
        ranked.sort(key=lambda row: int(row["rf_bridge_source_index"]))
        for row in ranked:
            row.pop("rf_bridge_source_index", None)
        for original, row in zip(originals, ranked):
            non_rf = {
                key: value
                for key, value in row.items()
                if not str(key).startswith("rf_")
            }
            if non_rf != original:
                raise AssertionError("RF bridge changed a non-rf candidate field")
        return ranked


__all__ = [
    "ALLOWED_ORCA_MODEL_STATUSES",
    "DEFAULT_ORCA_MAX_AGE_SECONDS",
    "ORCA_MONITOR_SNAPSHOT_SCHEMA",
    "RFShadowBridge",
    "adapt_orca_rf_snapshot",
    "compute_orca_snapshot_hash",
    "load_orca_rf_snapshot",
]
