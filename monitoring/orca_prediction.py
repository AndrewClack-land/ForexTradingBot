"""Fail-closed publisher for frozen ORCA regime-model predictions.

This process is intentionally separate from both MT5 execution and model
training.  It reads a closed-D1 panel and an explicitly promoted, immutable
Random Forest artifact, scores the latest causal feature row with the portable
NumPy predictor, and atomically publishes a document consumed by
``JsonPredictionProvider``.

No model is fitted or promoted here.  Promotion is represented by two
operator-supplied SHA-256 pins: the frozen artifact hash and the complete
causal feature-frame registry hash.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from backtest.orca_feature_frame import (
    ORCA_PIPELINE_SCHEMA,
    OrcaFeatureFrame,
    build_orca_feature_frame,
    validate_feature_registry,
)
from backtest.orca_random_forest import (
    load_orca_artifact,
    predict_orca_frozen,
)
from core.orca_spectral import (
    BcdAucResult,
    OrcaSpectralConfig,
    balanced_crisis_detection_auc,
    build_orca_spectral_snapshots,
    spectral_feature_registry_hash,
    spectral_feature_row,
)
from monitoring.orca_monitor import (
    ORCA_PREDICTION_SCHEMA,
    compute_prediction_hash,
    load_price_panel,
)


ORCA_PREDICTION_PUBLISHER_SCHEMA = "orca-prediction-publisher-v1"
RANK_METHOD = "midrank_empirical_cdf_strictly_prior_oos"


class OrcaPredictionError(ValueError):
    """Raised when a prediction cannot be published without violating PIT."""


def _utc(value: Any, *, name: str) -> datetime:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise OrcaPredictionError(f"{name} must be a timestamp") from exc
    if pd.isna(parsed) or parsed.tzinfo is None:
        raise OrcaPredictionError(f"{name} must be timezone-aware")
    return parsed.tz_convert("UTC").to_pydatetime().astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(value: Any, *, name: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise OrcaPredictionError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _canonical_bytes(payload: Any, *, newline: bool = False) -> bytes:
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise OrcaPredictionError(
            "payload contains non-finite or non-JSON data"
        ) from exc
    return (text + ("\n" if newline else "")).encode("utf-8")


def compute_feature_values_hash(values: Mapping[str, Any]) -> str:
    """Match ``orca_monitor._feature_values_hash`` byte-for-byte."""

    return sha256(_canonical_bytes(values, newline=True)).hexdigest()


def _finite_probability(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OrcaPredictionError(f"{name} must be finite in [0, 1]") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise OrcaPredictionError(f"{name} must be finite in [0, 1]")
    return result


@dataclass(frozen=True)
class OrcaPredictionConfig:
    """Immutable promotion and freshness contract for one publisher."""

    promoted_artifact_hash: str
    promoted_feature_registry_hash: str
    stale_after_seconds: float = 129_600.0
    fold_index: int = -1
    model_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "promoted_artifact_hash",
            _sha256(self.promoted_artifact_hash, name="promoted_artifact_hash"),
        )
        object.__setattr__(
            self,
            "promoted_feature_registry_hash",
            _sha256(
                self.promoted_feature_registry_hash,
                name="promoted_feature_registry_hash",
            ),
        )
        try:
            stale = float(self.stale_after_seconds)
        except (TypeError, ValueError) as exc:
            raise OrcaPredictionError("stale_after_seconds must be positive") from exc
        if not math.isfinite(stale) or stale <= 0.0:
            raise OrcaPredictionError("stale_after_seconds must be positive")
        object.__setattr__(self, "stale_after_seconds", stale)
        if isinstance(self.fold_index, bool) or not isinstance(self.fold_index, int):
            raise OrcaPredictionError("fold_index must be an integer")
        if self.model_id is not None:
            model_id = self.model_id.strip()
            if not model_id:
                raise OrcaPredictionError("model_id cannot be blank")
            object.__setattr__(self, "model_id", model_id)


@dataclass(frozen=True)
class _PromotedArtifact:
    model: Mapping[str, Any]
    pipeline_registry: Optional[Mapping[str, Any]]


def load_promoted_artifact(
    path: Path | str,
    *,
    expected_artifact_hash: str,
) -> _PromotedArtifact:
    """Load a direct RF artifact or the nested artifact from a research pipeline."""

    source = Path(path)
    if not source.is_file():
        raise OrcaPredictionError(f"promoted artifact does not exist: {source}")
    try:
        outer = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OrcaPredictionError("promoted artifact is not valid UTF-8 JSON") from exc
    if not isinstance(outer, Mapping):
        raise OrcaPredictionError("promoted artifact must be a JSON object")

    pipeline_registry: Optional[Mapping[str, Any]] = None
    candidate: Any = outer
    if outer.get("schema") == ORCA_PIPELINE_SCHEMA:
        candidate = outer.get("model_artifact")
        pipeline_registry_raw = outer.get("feature_registry")
        if not isinstance(pipeline_registry_raw, Mapping):
            raise OrcaPredictionError("pipeline feature registry is missing")
        pipeline_registry = dict(pipeline_registry_raw)
    try:
        model = load_orca_artifact(candidate)
    except (TypeError, ValueError) as exc:
        raise OrcaPredictionError(f"promoted model is invalid: {exc}") from exc
    expected = _sha256(expected_artifact_hash, name="expected_artifact_hash")
    actual = _sha256(model.get("artifact_hash"), name="model.artifact_hash")
    if actual != expected:
        raise OrcaPredictionError("promoted artifact hash does not match its pin")
    return _PromotedArtifact(model=model, pipeline_registry=pipeline_registry)


def causal_empirical_rank(
    current_score: float,
    observations: Sequence[tuple[Any, float]],
    *,
    as_of_market_utc: datetime,
) -> tuple[Optional[float], int]:
    """Return a tie-aware empirical rank using OOS timestamps strictly before as-of."""

    score = _finite_probability(current_score, name="current_score")
    cutoff = _utc(as_of_market_utc, name="as_of_market_utc")
    eligible: list[float] = []
    for timestamp, value in observations:
        observed_at = _utc(timestamp, name="OOS timestamp")
        probability = _finite_probability(value, name="OOS probability")
        if observed_at < cutoff:
            eligible.append(probability)
    if not eligible:
        return None, 0
    history = np.asarray(eligible, dtype=float)
    less = int(np.sum(history < score))
    equal = int(np.sum(history == score))
    rank = (less + 0.5 * equal) / len(history)
    return float(rank), len(history)


def _selected_fold(
    model: Mapping[str, Any], fold_index: int
) -> tuple[int, Mapping[str, Any]]:
    folds = model["folds"]
    resolved = fold_index if fold_index >= 0 else len(folds) + fold_index
    if not 0 <= resolved < len(folds):
        raise OrcaPredictionError("fold_index is outside the promoted artifact")
    return resolved, folds[resolved]


def _trained_through(fold: Mapping[str, Any]) -> datetime:
    lineage = fold.get("lineage")
    if not isinstance(lineage, Mapping):
        raise OrcaPredictionError("selected fold lineage is missing")
    value = lineage.get("train_last_label_known_at")
    if value in (None, ""):
        raise OrcaPredictionError("selected fold lacks train_last_label_known_at")
    return _utc(value, name="train_last_label_known_at")


def _oos_history(model: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in model["folds"]:
        fold_rows = fold.get("oos")
        if not isinstance(fold_rows, list):
            raise OrcaPredictionError("fold OOS rows are missing")
        for raw in fold_rows:
            if not isinstance(raw, Mapping):
                raise OrcaPredictionError("OOS row must be an object")
            timestamp = _utc(raw.get("timestamp"), name="OOS timestamp")
            label_known_at = _utc(
                raw.get("label_known_at"),
                name="OOS label_known_at",
            )
            if label_known_at < timestamp:
                raise OrcaPredictionError("OOS label_known_at precedes its timestamp")
            rally_label = raw.get("rally_label")
            crash_label = raw.get("crash_label")
            if rally_label not in (0, 1) or crash_label not in (0, 1):
                raise OrcaPredictionError("OOS labels must be binary")
            rows.append(
                {
                    "timestamp": timestamp,
                    "label_known_at": label_known_at,
                    "rally_label": int(rally_label),
                    "crash_label": int(crash_label),
                    "rally_probability": _finite_probability(
                        raw.get("rally_probability"),
                        name="OOS rally_probability",
                    ),
                    "crash_probability": _finite_probability(
                        raw.get("crash_probability"),
                        name="OOS crash_probability",
                    ),
                }
            )
    return rows


def _validation_mapping(
    rows: Sequence[Mapping[str, Any]],
    *,
    as_of_market_utc: datetime,
) -> dict[str, Any]:
    # Labels are usable only after they are known, and both boundaries are
    # strict so the current D1 close cannot influence its own validation.
    eligible = [
        row
        for row in rows
        if row["timestamp"] < as_of_market_utc
        and row["label_known_at"] < as_of_market_utc
    ]
    if not eligible:
        return {
            "schema_version": "balanced-crisis-detection-auc-v1",
            "status": "UNDEFINED",
            "bcd_auc": None,
            "rally_auc": None,
            "crash_auc": None,
            "observation_count": 0,
            "rally_positive_count": 0,
            "rally_negative_count": 0,
            "crash_positive_count": 0,
            "crash_negative_count": 0,
            "reason": "no_strictly_prior_mature_oos_rows",
            "cutoff_rule": "timestamp < as_of and label_known_at < as_of",
        }
    result: BcdAucResult = balanced_crisis_detection_auc(
        [row["rally_label"] for row in eligible],
        [row["rally_probability"] for row in eligible],
        [row["crash_label"] for row in eligible],
        [row["crash_probability"] for row in eligible],
    )
    return {
        **asdict(result),
        "cutoff_rule": "timestamp < as_of and label_known_at < as_of",
    }


def _validate_model_registry(
    frame: OrcaFeatureFrame,
    promoted: _PromotedArtifact,
    *,
    expected_registry_hash: str,
) -> None:
    validate_feature_registry(frame)
    actual = _sha256(frame.registry.get("hash"), name="feature registry hash")
    expected = _sha256(expected_registry_hash, name="expected feature registry hash")
    if actual != expected:
        raise OrcaPredictionError(
            "causal feature registry does not match its promotion pin"
        )
    if promoted.pipeline_registry is not None and dict(
        promoted.pipeline_registry
    ) != dict(frame.registry):
        raise OrcaPredictionError(
            "pipeline feature registry does not match current builder"
        )
    names = list(frame.features.columns)
    model_names = list(promoted.model["feature_registry"]["names"])
    if names != model_names:
        raise OrcaPredictionError(
            "model feature registry does not match current builder"
        )


def build_prediction(
    prices: pd.DataFrame,
    promoted: _PromotedArtifact,
    *,
    benchmark_symbol: str,
    known_at_utc: datetime,
    config: OrcaPredictionConfig,
    spectral_config: OrcaSpectralConfig = OrcaSpectralConfig(),
    feature_builder: Callable[..., OrcaFeatureFrame] = build_orca_feature_frame,
    predictor: Callable[..., pd.DataFrame] = predict_orca_frozen,
) -> dict[str, Any]:
    """Build one provider-compatible prediction without changing external state."""

    known_at = _utc(known_at_utc, name="known_at_utc")
    bundle = build_orca_spectral_snapshots(
        prices,
        as_of_utc=known_at,
        config=spectral_config,
    )
    as_of_market = bundle.returns_end_utc
    age_seconds = (known_at - as_of_market).total_seconds()
    if age_seconds < 0.0:
        raise OrcaPredictionError("price panel contains a future D1 close")
    if age_seconds > config.stale_after_seconds:
        raise OrcaPredictionError("latest closed-D1 market data is stale")

    frame = feature_builder(
        prices,
        benchmark_symbol=benchmark_symbol,
        as_of_utc=known_at,
        spectral_config=spectral_config,
    )
    _validate_model_registry(
        frame,
        promoted,
        expected_registry_hash=config.promoted_feature_registry_hash,
    )
    if frame.features.index[-1] != pd.Timestamp(as_of_market):
        raise OrcaPredictionError(
            "model and spectral features have different market as-of"
        )
    current = frame.features.iloc[[-1]]
    current_values = current.to_numpy(dtype=float, copy=True)
    if np.isinf(current_values).any():
        raise OrcaPredictionError("current model feature row contains infinity")

    resolved_fold, fold = _selected_fold(promoted.model, config.fold_index)
    trained_through = _trained_through(fold)
    if trained_through > as_of_market:
        raise OrcaPredictionError("selected model was trained beyond market as-of")
    try:
        scored = predictor(promoted.model, current, fold_index=resolved_fold)
    except (TypeError, ValueError) as exc:
        raise OrcaPredictionError(f"frozen model inference failed: {exc}") from exc
    if (
        list(scored.columns) != ["rally_probability", "crash_probability"]
        or len(scored) != 1
    ):
        raise OrcaPredictionError("frozen predictor returned an incompatible shape")
    p_rally = _finite_probability(scored.iloc[0]["rally_probability"], name="p_rally")
    p_crash = _finite_probability(scored.iloc[0]["crash_probability"], name="p_crash")

    oos = _oos_history(promoted.model)
    rally_rank, rally_count = causal_empirical_rank(
        p_rally,
        [(row["timestamp"], row["rally_probability"]) for row in oos],
        as_of_market_utc=as_of_market,
    )
    crash_rank, crash_count = causal_empirical_rank(
        p_crash,
        [(row["timestamp"], row["crash_probability"]) for row in oos],
        as_of_market_utc=as_of_market,
    )
    validation = _validation_mapping(oos, as_of_market_utc=as_of_market)

    spectral_values = spectral_feature_row(bundle)
    spectral_registry_hash = spectral_feature_registry_hash(bundle)
    spectral_values_hash = compute_feature_values_hash(spectral_values)
    model_registry = dict(promoted.model["feature_registry"])
    model_registry_hash = _sha256(
        model_registry.get("hash"),
        name="model feature registry hash",
    )
    model_values = {
        name: (None if pd.isna(current.iloc[0][name]) else float(current.iloc[0][name]))
        for name in model_registry["names"]
    }
    artifact_hash = _sha256(
        promoted.model.get("artifact_hash"),
        name="model artifact hash",
    )
    model_id = config.model_id or f"orca-rf-{artifact_hash[:16]}"
    status = (
        "OOS_VALIDATED"
        if validation["status"] == "DEFINED"
        and rally_rank is not None
        and crash_rank is not None
        else "OOS_PARTIAL"
    )
    payload: dict[str, Any] = {
        "schema_version": ORCA_PREDICTION_SCHEMA,
        "publisher_schema_version": ORCA_PREDICTION_PUBLISHER_SCHEMA,
        "as_of_market_utc": _iso(as_of_market),
        "known_at_utc": _iso(known_at),
        "trained_through_utc": _iso(trained_through),
        # These two fields are the exact JsonPredictionProvider join contract.
        "feature_registry_hash": spectral_registry_hash,
        "feature_values_hash": spectral_values_hash,
        "feature_values": spectral_values,
        # The full model registry remains distinct from the monitor's static
        # spectral registry and is pinned independently at promotion time.
        "causal_feature_registry_hash": frame.registry["hash"],
        "model_feature_registry": model_registry,
        "model_feature_registry_hash": model_registry_hash,
        "model_feature_values_hash": compute_feature_values_hash(model_values),
        "model_feature_missing_count": sum(
            value is None for value in model_values.values()
        ),
        "model_missing_value_policy": "frozen_training_median_imputation",
        "model_id": model_id,
        "model_artifact_hash": artifact_hash,
        "model_fold_index": resolved_fold,
        "status": status,
        "p_rally": p_rally,
        "p_crash": p_crash,
        "rally_rank": rally_rank,
        "crash_rank": crash_rank,
        "rank_method": RANK_METHOD,
        "rank_oos_counts": {"rally": rally_count, "crash": crash_count},
        "calibration_status": "UNCALIBRATED",
        "validation": validation,
        "executing": False,
    }
    _canonical_bytes(payload)
    payload["prediction_hash"] = compute_prediction_hash(payload)
    _canonical_bytes(payload)
    return payload


class AtomicPredictionStore:
    """Atomic replace store; failures leave the previous prediction intact."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def publish(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema_version") != ORCA_PREDICTION_SCHEMA:
            raise OrcaPredictionError("unsupported prediction schema")
        if payload.get("executing") is not False:
            raise OrcaPredictionError("prediction must be non-executing")
        expected_hash = _sha256(payload.get("prediction_hash"), name="prediction_hash")
        if expected_hash != compute_prediction_hash(payload):
            raise OrcaPredictionError("prediction hash mismatch")
        encoded = _canonical_bytes(payload, newline=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def read(self) -> dict[str, Any]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise OrcaPredictionError("stored prediction must be an object")
        return payload


class OrcaPredictionEngine:
    """One-shot refresh engine suitable for a systemd timer."""

    def __init__(
        self,
        *,
        prices_path: Path | str,
        artifact_path: Path | str,
        output_path: Path | str,
        benchmark_symbol: str,
        config: OrcaPredictionConfig,
        spectral_config: OrcaSpectralConfig = OrcaSpectralConfig(),
        feature_builder: Callable[..., OrcaFeatureFrame] = build_orca_feature_frame,
        predictor: Callable[..., pd.DataFrame] = predict_orca_frozen,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.prices_path = Path(prices_path)
        self.artifact_path = Path(artifact_path)
        self.output = AtomicPredictionStore(output_path)
        self.benchmark_symbol = benchmark_symbol
        self.config = config
        self.spectral_config = spectral_config
        self.feature_builder = feature_builder
        self.predictor = predictor
        self.now = now

    def refresh(self) -> dict[str, Any]:
        known_at = _utc(self.now(), name="current time")
        prices = load_price_panel(self.prices_path)
        promoted = load_promoted_artifact(
            self.artifact_path,
            expected_artifact_hash=self.config.promoted_artifact_hash,
        )
        payload = build_prediction(
            prices,
            promoted,
            benchmark_symbol=self.benchmark_symbol,
            known_at_utc=known_at,
            config=self.config,
            spectral_config=self.spectral_config,
            feature_builder=self.feature_builder,
            predictor=self.predictor,
        )
        self.output.publish(payload)
        return payload


def _required(value: Optional[str], *, name: str) -> str:
    if value is None or not value.strip():
        raise OrcaPredictionError(f"{name} is required")
    return value.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish a frozen, non-executing ORCA prediction once"
    )
    parser.add_argument("--prices", default=os.getenv("ORCA_PRICES_PATH"))
    parser.add_argument("--artifact", default=os.getenv("ORCA_PROMOTED_ARTIFACT_PATH"))
    parser.add_argument("--output", default=os.getenv("ORCA_PREDICTION_PATH"))
    parser.add_argument(
        "--benchmark-symbol", default=os.getenv("ORCA_BENCHMARK_SYMBOL")
    )
    parser.add_argument(
        "--promoted-artifact-hash",
        default=os.getenv("ORCA_PROMOTED_ARTIFACT_HASH"),
    )
    parser.add_argument(
        "--promoted-feature-registry-hash",
        default=os.getenv("ORCA_PROMOTED_FEATURE_REGISTRY_HASH"),
    )
    parser.add_argument(
        "--stale-after-seconds",
        type=float,
        default=float(os.getenv("ORCA_PREDICTION_STALE_AFTER_SECONDS", "129600")),
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        default=int(os.getenv("ORCA_PROMOTED_FOLD_INDEX", "-1")),
    )
    parser.add_argument("--model-id", default=os.getenv("ORCA_PROMOTED_MODEL_ID"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    config = OrcaPredictionConfig(
        promoted_artifact_hash=_required(
            arguments.promoted_artifact_hash,
            name="promoted artifact hash",
        ),
        promoted_feature_registry_hash=_required(
            arguments.promoted_feature_registry_hash,
            name="promoted feature registry hash",
        ),
        stale_after_seconds=arguments.stale_after_seconds,
        fold_index=arguments.fold_index,
        model_id=arguments.model_id,
    )
    engine = OrcaPredictionEngine(
        prices_path=_required(arguments.prices, name="prices path"),
        artifact_path=_required(arguments.artifact, name="artifact path"),
        output_path=_required(arguments.output, name="output path"),
        benchmark_symbol=_required(
            arguments.benchmark_symbol,
            name="benchmark symbol",
        ),
        config=config,
    )
    payload = engine.refresh()
    print(
        json.dumps(
            {
                "status": payload["status"],
                "as_of_market_utc": payload["as_of_market_utc"],
                "model_id": payload["model_id"],
                "prediction_hash": payload["prediction_hash"],
                "executing": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())


__all__ = [
    "ORCA_PREDICTION_PUBLISHER_SCHEMA",
    "RANK_METHOD",
    "AtomicPredictionStore",
    "OrcaPredictionConfig",
    "OrcaPredictionEngine",
    "OrcaPredictionError",
    "build_prediction",
    "causal_empirical_rank",
    "compute_feature_values_hash",
    "load_promoted_artifact",
    "main",
]
