"""Auto-refreshing ORCA spectral monitor and small HTTP service.

The monitor is deliberately outside the MT5 execution process.  It reads an
immutable wide D1 price panel, computes a causal spectral snapshot, optionally
joins a separately published model prediction, and atomically publishes one
JSON document for the dashboard.  It never imports the broker executor and it
never returns a trading action.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Mapping, Optional, Protocol

import pandas as pd

from core.orca_spectral import (
    OrcaSpectralConfig,
    build_orca_spectral_snapshots,
    spectral_feature_registry_hash,
    spectral_feature_row,
)


ORCA_MONITOR_SCHEMA = "orca-monitor-snapshot-v1"
ORCA_PREDICTION_SCHEMA = "orca-monitor-prediction-v1"


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


def _finite_optional(value: Any, *, name: str) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _hash_without(payload: Mapping[str, Any], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return sha256(_canonical_bytes(body)).hexdigest()


def compute_prediction_hash(payload: Mapping[str, Any]) -> str:
    """Hash a prediction document while excluding its own integrity field."""

    return _hash_without(payload, "prediction_hash")


def compute_snapshot_hash(payload: Mapping[str, Any]) -> str:
    """Hash a monitor snapshot while excluding its own integrity field."""

    return _hash_without(payload, "snapshot_hash")


def _feature_values_hash(features: Mapping[str, Any]) -> str:
    return sha256(_canonical_bytes(features)).hexdigest()


def _sha256_field(value: Any, *, name: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return text


@dataclass(frozen=True)
class OrcaMonitorConfig:
    refresh_seconds: float = 43_200.0
    stale_after_seconds: float = 129_600.0
    expected_assets: int = 24
    minimum_assets: int = 5
    allowed_model_statuses: tuple[str, ...] = ("OOS_VALIDATED",)

    def __post_init__(self) -> None:
        if not math.isfinite(self.refresh_seconds) or self.refresh_seconds <= 0:
            raise ValueError("refresh_seconds must be positive")
        if not math.isfinite(self.stale_after_seconds) or self.stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        if self.minimum_assets < 2:
            raise ValueError("minimum_assets must be >= 2")
        if self.expected_assets < self.minimum_assets:
            raise ValueError("expected_assets must be >= minimum_assets")
        statuses = tuple(str(value).strip() for value in self.allowed_model_statuses)
        if not statuses or any(not value for value in statuses):
            raise ValueError("allowed_model_statuses cannot be empty")
        if len(set(statuses)) != len(statuses):
            raise ValueError("allowed_model_statuses must be unique")
        object.__setattr__(self, "allowed_model_statuses", statuses)


class PredictionProvider(Protocol):
    def read_prediction(
        self,
        *,
        as_of_market_utc: datetime,
        known_at_utc: datetime,
        feature_registry_hash: str,
        feature_values_hash: str,
    ) -> Optional[Mapping[str, Any]]: ...


class JsonPredictionProvider:
    """Read a separately, atomically published ORCA model prediction."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def read_prediction(
        self,
        *,
        as_of_market_utc: datetime,
        known_at_utc: datetime,
        feature_registry_hash: str,
        feature_values_hash: str,
    ) -> Optional[Mapping[str, Any]]:
        if not self.path.exists():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("prediction must be a JSON object")
        return _validate_prediction(
            payload,
            as_of_market_utc=as_of_market_utc,
            known_at_utc=known_at_utc,
            feature_registry_hash=feature_registry_hash,
            feature_values_hash=feature_values_hash,
        )


def _validate_prediction(
    payload: Mapping[str, Any],
    *,
    as_of_market_utc: datetime,
    known_at_utc: datetime,
    feature_registry_hash: str,
    feature_values_hash: str,
) -> dict[str, Any]:
    if payload.get("schema_version") != ORCA_PREDICTION_SCHEMA:
        raise ValueError(f"prediction.schema_version must be {ORCA_PREDICTION_SCHEMA}")
    expected_hash = _sha256_field(
        payload.get("prediction_hash"),
        name="prediction.prediction_hash",
    )
    if expected_hash != compute_prediction_hash(payload):
        raise ValueError("prediction hash mismatch")

    market_as_of = _utc(as_of_market_utc, name="as_of_market_utc")
    decision_time = _utc(known_at_utc, name="known_at_utc")
    prediction_as_of = _utc(
        payload.get("as_of_market_utc"),
        name="prediction.as_of_market_utc",
    )
    prediction_known_at = _utc(
        payload.get("known_at_utc"),
        name="prediction.known_at_utc",
    )
    if prediction_as_of != market_as_of:
        raise ValueError("prediction market as_of does not exactly match prices")
    if prediction_known_at < prediction_as_of:
        raise ValueError("prediction known_at precedes market as_of")
    if prediction_known_at > decision_time:
        raise ValueError("prediction was not known at monitor decision time")

    registry = _sha256_field(
        payload.get("feature_registry_hash"),
        name="prediction.feature_registry_hash",
    )
    if registry != _sha256_field(
        feature_registry_hash,
        name="feature_registry_hash",
    ):
        raise ValueError("prediction feature registry does not match")
    values_hash = _sha256_field(
        payload.get("feature_values_hash"),
        name="prediction.feature_values_hash",
    )
    if values_hash != _sha256_field(
        feature_values_hash,
        name="feature_values_hash",
    ):
        raise ValueError("prediction feature values do not match")

    trained_raw = payload.get("trained_through_utc")
    if trained_raw not in (None, ""):
        trained_through = _utc(
            trained_raw,
            name="prediction.trained_through_utc",
        )
        if trained_through > prediction_as_of:
            raise ValueError("prediction model was trained beyond market as_of")
    artifact_hash = payload.get("model_artifact_hash")
    if artifact_hash not in (None, ""):
        _sha256_field(artifact_hash, name="prediction.model_artifact_hash")
    status = payload.get("status")
    if not isinstance(status, str) or not status.strip():
        raise ValueError("prediction.status must be a non-empty string")

    if payload.get("executing") is not False:
        raise ValueError("prediction must declare executing=false")
    result = dict(payload)
    for key in ("p_rally", "p_crash", "rally_rank", "crash_rank"):
        value = _finite_optional(result.get(key), name=f"prediction.{key}")
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"prediction.{key} must be in [0, 1]")
        result[key] = value
    return result


def classify_regime(
    *,
    rally_rank: Optional[float],
    crash_rank: Optional[float],
) -> dict[str, Any]:
    """Apply only the exposure zones explicitly stated in the ORCA paper.

    The paper does not publish the complete intermediate exposure table.  We
    therefore return ``None`` for intermediate exposure rather than inventing
    the missing 0.3/0.7/1.0/1.2 mapping.
    """

    if rally_rank is None or crash_rank is None:
        return {
            "regime": "UNAVAILABLE",
            "risk_mode": "UNAVAILABLE",
            "suggested_equity_exposure": None,
            "exposure_status": "MODEL_RANKS_MISSING",
        }
    if not (0.0 <= rally_rank <= 1.0 and 0.0 <= crash_rank <= 1.0):
        raise ValueError("rally_rank and crash_rank must be in [0, 1]")
    if crash_rank >= 0.60:
        return {
            "regime": "CRISIS",
            "risk_mode": "RISK_OFF",
            "suggested_equity_exposure": 0.0,
            "exposure_status": "PAPER_EXPLICIT_ZONE",
        }
    if rally_rank >= 0.90:
        return {
            "regime": "EUPHORIA",
            "risk_mode": "RISK_OFF",
            "suggested_equity_exposure": 0.0,
            "exposure_status": "PAPER_EXPLICIT_ZONE",
        }
    if 0.78 <= rally_rank < 0.90 and crash_rank < 0.40:
        return {
            "regime": "RALLY",
            "risk_mode": "RISK_ON",
            "suggested_equity_exposure": 1.5,
            "exposure_status": "PAPER_EXPLICIT_ZONE",
        }
    if crash_rank >= 0.40:
        regime = "CAUTION"
        risk_mode = "CAUTION"
    else:
        regime = "NORMAL"
        risk_mode = "NEUTRAL"
    return {
        "regime": regime,
        "risk_mode": risk_mode,
        "suggested_equity_exposure": None,
        "exposure_status": "INTERMEDIATE_MAP_NOT_PUBLISHED",
    }


def _closed_d1_index(values: Any, *, source_name: str) -> pd.DatetimeIndex:
    if isinstance(values, pd.DatetimeIndex):
        midnight = (
            (values.hour == 0)
            & (values.minute == 0)
            & (values.second == 0)
            & (values.microsecond == 0)
        )
        if bool(midnight.any()):
            raise ValueError(
                f"{source_name} rejects date-only or implicit-midnight D1 labels; "
                "use the actual timezone-aware bar-close availability timestamp"
            )
        if values.tz is None:
            raise ValueError(
                f"{source_name} timestamps must carry an explicit timezone"
            )
        parsed = values.tz_convert("UTC")
    else:
        timestamps: list[pd.Timestamp] = []
        for raw in values:
            try:
                timestamp = pd.Timestamp(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{source_name} contains an invalid D1 timestamp"
                ) from exc
            if pd.isna(timestamp):
                raise ValueError(f"{source_name} timestamps contain NaT")
            if (
                timestamp.hour == 0
                and timestamp.minute == 0
                and timestamp.second == 0
                and timestamp.microsecond == 0
            ):
                raise ValueError(
                    f"{source_name} rejects date-only or implicit-midnight "
                    "D1 labels; use the actual timezone-aware bar-close "
                    "availability timestamp"
                )
            if timestamp.tzinfo is None:
                raise ValueError(
                    f"{source_name} timestamps must carry an explicit timezone"
                )
            timestamps.append(timestamp.tz_convert("UTC"))
        parsed = pd.DatetimeIndex(timestamps)

    if parsed.hasnans:
        raise ValueError(f"{source_name} timestamps contain NaT")
    return parsed


def load_price_panel(path: Path | str) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
        if not isinstance(frame.index, pd.DatetimeIndex):
            time_column = next(
                (
                    name
                    for name in ("timestamp", "time", "date", "datetime")
                    if name in frame.columns
                ),
                None,
            )
            if time_column is None:
                raise ValueError(
                    "Parquet price panel needs a timezone-aware timestamp index "
                    "or timestamp column"
                )
            raw_time = frame.pop(time_column)
            frame.index = _closed_d1_index(
                raw_time,
                source_name="price panel",
            )
        else:
            frame.index = _closed_d1_index(
                frame.index,
                source_name="price panel",
            )
    elif suffix == ".csv":
        frame = pd.read_csv(source)
        time_column = next(
            (
                name
                for name in ("timestamp", "time", "date", "datetime")
                if name in frame.columns
            ),
            frame.columns[0] if len(frame.columns) else None,
        )
        if time_column is None:
            raise ValueError("CSV price panel has no columns")
        raw_time = frame.pop(time_column)
        frame.index = _closed_d1_index(
            raw_time,
            source_name="price panel",
        )
    else:
        raise ValueError("price panel must be .parquet, .pq, or .csv")
    return frame


def _snapshot_mapping(snapshot: Any) -> dict[str, Any]:
    graph_metrics = {
        format(float(threshold), ".12g"): {
            "threshold": float(metrics.threshold),
            "edge_count": int(metrics.edge_count),
            "edge_density": float(metrics.edge_density),
            "mean_degree": float(metrics.mean_degree),
            "degree_std": float(metrics.degree_std),
            "max_degree": int(metrics.max_degree),
            "isolated_nodes": int(metrics.isolated_nodes),
            "degree_centralization": float(metrics.degree_centralization),
            "global_clustering": float(metrics.global_clustering),
        }
        for threshold, metrics in snapshot.graph_metrics.items()
    }
    coordinates = {
        symbol: [float(value) for value in values]
        for symbol, values in snapshot.node_coordinates_3d.items()
    }
    signed_edges = {
        format(float(threshold), ".12g"): [
            {
                "threshold": float(edge.threshold),
                "source": edge.source,
                "target": edge.target,
                "correlation": float(edge.correlation),
                "sign": int(edge.sign),
                "absolute_correlation": float(edge.absolute_correlation),
            }
            for edge in edges
        ]
        for threshold, edges in snapshot.signed_edges.items()
    }
    return {
        "estimator": snapshot.estimator,
        "symbols": list(snapshot.symbols),
        "returns_start_utc": _iso(snapshot.returns_start_utc),
        "returns_end_utc": _iso(snapshot.returns_end_utc),
        "observation_count": int(snapshot.observation_count),
        "effective_sample_length": float(snapshot.effective_sample_length),
        "correlation": [
            [float(value) for value in row] for row in snapshot.correlation
        ],
        "eigenvalues": [float(value) for value in snapshot.eigenvalues],
        "eigenvectors": [
            [float(value) for value in row] for row in snapshot.eigenvectors
        ],
        "absorption_ratios": {
            str(rank): float(value)
            for rank, value in snapshot.absorption_ratios.items()
        },
        "eigenvalue_entropy": float(snapshot.eigenvalue_entropy),
        "effective_rank": float(snapshot.effective_rank),
        "lambda1_lambda2": (
            None
            if snapshot.lambda1_lambda2 is None
            else float(snapshot.lambda1_lambda2)
        ),
        "condition_number": (
            None
            if snapshot.condition_number is None
            else float(snapshot.condition_number)
        ),
        "marchenko_pastur_q": float(snapshot.marchenko_pastur_q),
        "marchenko_pastur_lower": float(snapshot.marchenko_pastur_lower),
        "marchenko_pastur_upper": float(snapshot.marchenko_pastur_upper),
        "marchenko_pastur_outlier_count": int(snapshot.marchenko_pastur_outlier_count),
        "marchenko_pastur_upper_outlier_count": int(
            snapshot.marchenko_pastur_upper_outlier_count
        ),
        "marchenko_pastur_lambda1_excess": float(
            snapshot.marchenko_pastur_lambda1_excess
        ),
        "dominant_eigenvector_hhi": float(snapshot.dominant_eigenvector_hhi),
        "dominant_eigenvector_participation": float(
            snapshot.dominant_eigenvector_participation
        ),
        "graph_metrics": graph_metrics,
        "node_coordinates_3d": coordinates,
        "signed_edges": signed_edges,
    }


def _market_freshness(
    *,
    age_seconds: float,
    asset_count: int,
    expected_assets: int,
    stale_after_seconds: float,
) -> str:
    if not math.isfinite(age_seconds) or age_seconds < 0.0:
        raise ValueError("market age must be finite and non-negative")
    if age_seconds > stale_after_seconds:
        return "STALE"
    if asset_count != expected_assets:
        return "DEGRADED_UNIVERSE"
    return "FRESH"


def _unavailable_regime(reason: str) -> dict[str, Any]:
    return {
        "regime": "UNAVAILABLE",
        "risk_mode": "UNAVAILABLE",
        "suggested_equity_exposure": None,
        "exposure_status": reason,
        "regime_eligible": False,
    }


def _gated_regime(
    *,
    freshness: str,
    model: Mapping[str, Any],
    monitor_config: OrcaMonitorConfig,
) -> dict[str, Any]:
    if freshness == "STALE":
        return _unavailable_regime("STALE_MARKET_DATA")
    if freshness != "FRESH":
        return _unavailable_regime("FULL_UNIVERSE_REQUIRED")
    status = str(model.get("status") or "UNAVAILABLE")
    if status == "UNAVAILABLE":
        return _unavailable_regime("MODEL_UNAVAILABLE")
    if status not in monitor_config.allowed_model_statuses:
        return _unavailable_regime("MODEL_STATUS_NOT_ALLOWED")
    result = classify_regime(
        rally_rank=_finite_optional(model.get("rally_rank"), name="rally_rank"),
        crash_rank=_finite_optional(model.get("crash_rank"), name="crash_rank"),
    )
    result["regime_eligible"] = result["regime"] != "UNAVAILABLE"
    return result


def build_monitor_snapshot(
    prices: pd.DataFrame,
    *,
    generated_at_utc: datetime,
    spectral_config: Optional[OrcaSpectralConfig] = None,
    monitor_config: Optional[OrcaMonitorConfig] = None,
    prediction: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    generated = _utc(generated_at_utc, name="generated_at_utc")
    monitor = monitor_config or OrcaMonitorConfig()
    if prices.shape[1] < monitor.minimum_assets:
        raise ValueError(
            f"ORCA requires at least {monitor.minimum_assets} assets; "
            f"received {prices.shape[1]}"
        )
    bundle = build_orca_spectral_snapshots(
        prices,
        as_of_utc=generated,
        config=spectral_config or OrcaSpectralConfig(),
    )
    features = spectral_feature_row(bundle)
    registry_hash = spectral_feature_registry_hash(bundle)
    values_hash = _feature_values_hash(features)
    prediction_row = (
        _validate_prediction(
            prediction,
            as_of_market_utc=bundle.returns_end_utc,
            known_at_utc=generated,
            feature_registry_hash=registry_hash,
            feature_values_hash=values_hash,
        )
        if prediction is not None
        else {}
    )
    age_seconds = (generated - bundle.returns_end_utc).total_seconds()
    freshness = _market_freshness(
        age_seconds=age_seconds,
        asset_count=len(bundle.symbols),
        expected_assets=monitor.expected_assets,
        stale_after_seconds=monitor.stale_after_seconds,
    )
    model: dict[str, Any] = {
        "status": (
            str(prediction_row.get("status") or "AVAILABLE")
            if prediction_row
            else "UNAVAILABLE"
        ),
        "model_id": prediction_row.get("model_id"),
        "model_artifact_hash": prediction_row.get("model_artifact_hash"),
        "prediction_hash": prediction_row.get("prediction_hash"),
        "as_of_market_utc": prediction_row.get("as_of_market_utc"),
        "known_at_utc": prediction_row.get("known_at_utc"),
        "trained_through_utc": prediction_row.get("trained_through_utc"),
        "p_rally": _finite_optional(prediction_row.get("p_rally"), name="p_rally"),
        "p_crash": _finite_optional(prediction_row.get("p_crash"), name="p_crash"),
        "rally_rank": _finite_optional(
            prediction_row.get("rally_rank"), name="rally_rank"
        ),
        "crash_rank": _finite_optional(
            prediction_row.get("crash_rank"), name="crash_rank"
        ),
        "validation": dict(prediction_row.get("validation") or {}),
        "calibration_status": prediction_row.get("calibration_status", "UNAVAILABLE"),
    }
    regime = _gated_regime(
        freshness=freshness,
        model=model,
        monitor_config=monitor,
    )
    payload: dict[str, Any] = {
        "schema_version": ORCA_MONITOR_SCHEMA,
        "generated_at_utc": _iso(generated),
        "as_of_market_utc": _iso(bundle.returns_end_utc),
        "calculation_as_of_utc": _iso(bundle.as_of_utc),
        "returns_start_utc": _iso(bundle.returns_start_utc),
        "freshness": freshness,
        "age_seconds": age_seconds,
        "expected_assets": monitor.expected_assets,
        "asset_count": len(bundle.symbols),
        "full_universe": len(bundle.symbols) == monitor.expected_assets,
        "symbols": list(bundle.symbols),
        "spectral_schema": bundle.schema_version,
        "spectral_config_id": bundle.config_id,
        "feature_registry_hash": registry_hash,
        "feature_values_hash": values_hash,
        "features": features,
        "networks": [_snapshot_mapping(snapshot) for snapshot in bundle.snapshots],
        "model": model,
        **regime,
        "executing": False,
    }
    _canonical_bytes(payload)
    payload["snapshot_hash"] = compute_snapshot_hash(payload)
    return payload


def materialize_current_snapshot(
    payload: Mapping[str, Any],
    *,
    current_time_utc: datetime,
    monitor_config: OrcaMonitorConfig,
) -> dict[str, Any]:
    """Recompute market age and safety gates at request time."""

    current = _utc(current_time_utc, name="current_time_utc")
    generated = _utc(payload.get("generated_at_utc"), name="generated_at_utc")
    market_as_of = _utc(payload.get("as_of_market_utc"), name="as_of_market_utc")
    if generated > current:
        raise ValueError("snapshot generation time is in the future")
    age_seconds = (current - market_as_of).total_seconds()
    expected_assets = int(payload.get("expected_assets", -1))
    asset_count = int(payload.get("asset_count", -1))
    if expected_assets != monitor_config.expected_assets:
        raise ValueError("snapshot expected-assets contract differs from monitor")
    freshness = _market_freshness(
        age_seconds=age_seconds,
        asset_count=asset_count,
        expected_assets=expected_assets,
        stale_after_seconds=monitor_config.stale_after_seconds,
    )
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("snapshot model must be an object")
    current_payload = dict(payload)
    current_payload["served_at_utc"] = _iso(current)
    current_payload["age_seconds"] = age_seconds
    current_payload["freshness"] = freshness
    current_payload["full_universe"] = asset_count == expected_assets
    current_payload.update(
        _gated_regime(
            freshness=freshness,
            model=model,
            monitor_config=monitor_config,
        )
    )
    current_payload["snapshot_hash"] = compute_snapshot_hash(current_payload)
    _canonical_bytes(current_payload)
    return current_payload


class AtomicSnapshotStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    def publish(self, payload: Mapping[str, Any]) -> None:
        self._validate_integrity(payload)
        encoded = _canonical_bytes(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
            )
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_name, self.path)
                if os.name != "nt":
                    directory_fd = os.open(str(self.path.parent), os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
            except Exception:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
                raise

    def read(self) -> dict[str, Any]:
        with self._lock:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("snapshot must be a JSON object")
        self._validate_integrity(payload)
        return payload

    @staticmethod
    def _validate_integrity(payload: Mapping[str, Any]) -> None:
        if payload.get("schema_version") != ORCA_MONITOR_SCHEMA:
            raise ValueError(f"snapshot.schema_version must be {ORCA_MONITOR_SCHEMA}")
        if payload.get("executing") is not False:
            raise ValueError("snapshot must declare executing=false")
        expected = _sha256_field(
            payload.get("snapshot_hash"),
            name="snapshot.snapshot_hash",
        )
        if expected != compute_snapshot_hash(payload):
            raise ValueError("snapshot hash mismatch")
        _canonical_bytes(payload)


class OrcaMonitorEngine:
    def __init__(
        self,
        *,
        prices_path: Path | str,
        snapshot_store: AtomicSnapshotStore,
        spectral_config: Optional[OrcaSpectralConfig] = None,
        monitor_config: Optional[OrcaMonitorConfig] = None,
        prediction_provider: Optional[PredictionProvider] = None,
        clock: Any = None,
    ) -> None:
        self.prices_path = Path(prices_path)
        self.snapshot_store = snapshot_store
        self.spectral_config = spectral_config or OrcaSpectralConfig()
        self.monitor_config = monitor_config or OrcaMonitorConfig()
        self.prediction_provider = prediction_provider
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.last_error: Optional[str] = None
        self.last_refresh_utc: Optional[datetime] = None

    def refresh(self) -> dict[str, Any]:
        try:
            payload = self._refresh_once()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise
        self.last_error = None
        self.last_refresh_utc = _utc(
            payload["generated_at_utc"],
            name="generated_at_utc",
        )
        return payload

    def current_snapshot(self) -> dict[str, Any]:
        payload = self.snapshot_store.read()
        return materialize_current_snapshot(
            payload,
            current_time_utc=_utc(self.clock(), name="clock"),
            monitor_config=self.monitor_config,
        )

    def health(self) -> tuple[HTTPStatus, dict[str, Any]]:
        base: dict[str, Any] = {
            "last_error": self.last_error,
            "last_refresh_utc": (
                _iso(self.last_refresh_utc) if self.last_refresh_utc else None
            ),
        }
        if self.last_refresh_utc is None:
            return HTTPStatus.SERVICE_UNAVAILABLE, {
                **base,
                "status": "STARTING",
                "reason": "NO_SUCCESSFUL_REFRESH",
            }
        try:
            snapshot = self.current_snapshot()
        except Exception as exc:
            return HTTPStatus.SERVICE_UNAVAILABLE, {
                **base,
                "status": "UNAVAILABLE",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        base.update(
            {
                "freshness": snapshot["freshness"],
                "age_seconds": snapshot["age_seconds"],
                "as_of_market_utc": snapshot["as_of_market_utc"],
                "model_status": snapshot["model"]["status"],
                "full_universe": snapshot["full_universe"],
            }
        )
        if self.last_error is not None:
            return HTTPStatus.SERVICE_UNAVAILABLE, {
                **base,
                "status": "DEGRADED",
                "reason": "LAST_REFRESH_FAILED",
            }
        if snapshot["freshness"] != "FRESH":
            return HTTPStatus.SERVICE_UNAVAILABLE, {
                **base,
                "status": "DEGRADED",
                "reason": snapshot["freshness"],
            }
        if (
            self.prediction_provider is not None
            and snapshot["model"]["status"]
            not in self.monitor_config.allowed_model_statuses
        ):
            return HTTPStatus.SERVICE_UNAVAILABLE, {
                **base,
                "status": "DEGRADED",
                "reason": "MODEL_STATUS_NOT_ALLOWED",
            }
        return HTTPStatus.OK, {
            **base,
            "status": "OK",
            "reason": None,
        }

    def _refresh_once(self) -> dict[str, Any]:
        generated = _utc(self.clock(), name="clock")
        prices = load_price_panel(self.prices_path)
        bundle = build_orca_spectral_snapshots(
            prices,
            as_of_utc=generated,
            config=self.spectral_config,
        )
        registry_hash = spectral_feature_registry_hash(bundle)
        values_hash = _feature_values_hash(spectral_feature_row(bundle))
        prediction: Optional[Mapping[str, Any]] = None
        if self.prediction_provider is not None:
            prediction = self.prediction_provider.read_prediction(
                as_of_market_utc=bundle.returns_end_utc,
                known_at_utc=generated,
                feature_registry_hash=registry_hash,
                feature_values_hash=values_hash,
            )
        payload = build_monitor_snapshot(
            prices,
            generated_at_utc=generated,
            spectral_config=self.spectral_config,
            monitor_config=self.monitor_config,
            prediction=prediction,
        )
        self.snapshot_store.publish(payload)
        return payload

    def run(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            started = time.monotonic()
            try:
                self.refresh()
            except Exception as exc:
                # Keep the last known-good snapshot; health exposes the error.
                self.last_error = f"{type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - started
            wait_seconds = max(
                0.1,
                self.monitor_config.refresh_seconds - elapsed,
            )
            stop_event.wait(wait_seconds)


class OrcaRequestHandler(SimpleHTTPRequestHandler):
    snapshot_store: AtomicSnapshotStore
    monitor_engine: OrcaMonitorEngine

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = self.path.split("?", 1)[0]
        if path == "/api/snapshot":
            try:
                payload = self.monitor_engine.current_snapshot()
            except Exception as exc:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"status": "UNAVAILABLE", "error": str(exc)},
                )
                return
            self._send_json(HTTPStatus.OK, payload)
            return
        if path == "/healthz":
            status, payload = self.monitor_engine.health()
            self._send_json(status, payload)
            return
        if path == "/":
            self.path = "/orca_dashboard.html"
        super().do_GET()

    def _send_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        encoded = _canonical_bytes(payload)
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.plot.ly; "
            "style-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; img-src 'self' data:; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        del format, args


def make_bound_handler(
    *,
    engine: OrcaMonitorEngine,
    dashboard_directory: Path,
) -> type[OrcaRequestHandler]:
    """Bind dependencies on a real handler subclass accepted by HTTPServer."""

    directory = str(Path(dashboard_directory).resolve())

    class BoundHandler(OrcaRequestHandler):
        snapshot_store = engine.snapshot_store
        monitor_engine = engine

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=directory, **kwargs)

    return BoundHandler


def serve(
    *,
    engine: OrcaMonitorEngine,
    dashboard_directory: Path,
    host: str,
    port: int,
) -> None:
    stop_event = threading.Event()
    worker = threading.Thread(
        target=engine.run,
        args=(stop_event,),
        name="orca-refresh",
        daemon=True,
    )
    worker.start()
    handler = make_bound_handler(
        engine=engine,
        dashboard_directory=dashboard_directory,
    )
    server = ThreadingHTTPServer((host, port), handler)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        stop_event.set()
        server.server_close()
        worker.join(timeout=5.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve ORCA monitoring")
    parser.add_argument(
        "--prices",
        default=os.getenv("ORCA_PRICES_PATH", ""),
        required=not bool(os.getenv("ORCA_PRICES_PATH")),
    )
    parser.add_argument(
        "--snapshot",
        default=os.getenv("ORCA_SNAPSHOT_PATH", "ai_data/orca/latest.json"),
    )
    parser.add_argument(
        "--prediction",
        default=os.getenv("ORCA_PREDICTION_PATH", ""),
    )
    parser.add_argument("--host", default=os.getenv("ORCA_MONITOR_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("ORCA_MONITOR_PORT", "8765"))
    )
    parser.add_argument(
        "--refresh-seconds",
        type=float,
        default=float(os.getenv("ORCA_REFRESH_SECONDS", "43200")),
    )
    parser.add_argument(
        "--expected-assets",
        type=int,
        default=int(os.getenv("ORCA_EXPECTED_ASSETS", "24")),
    )
    parser.add_argument(
        "--stale-after-seconds",
        type=float,
        default=float(os.getenv("ORCA_STALE_AFTER_SECONDS", "129600")),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parent
    monitor_config = OrcaMonitorConfig(
        refresh_seconds=args.refresh_seconds,
        stale_after_seconds=args.stale_after_seconds,
        expected_assets=args.expected_assets,
        minimum_assets=min(5, args.expected_assets),
    )
    provider = JsonPredictionProvider(args.prediction) if args.prediction else None
    engine = OrcaMonitorEngine(
        prices_path=args.prices,
        snapshot_store=AtomicSnapshotStore(args.snapshot),
        monitor_config=monitor_config,
        prediction_provider=provider,
    )
    serve(
        engine=engine,
        dashboard_directory=root,
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
