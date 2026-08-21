"""Point-in-time ORCA feature-frame construction for offline research.

This module joins the static primitives in :mod:`core.orca_spectral` with a
small, explicitly versioned traditional feature contract and causal temporal
derivatives.  It intentionally does **not** claim to reproduce the paper's
exact 206-column implementation: the paper does not publish a complete
machine-readable registry.  The ordered registry emitted here is the contract.

Input timestamps are the actual availability times of fully closed D1 bars.
No feature may observe a row later than its own timestamp.  Warm-up values are
represented by ``NaN``; infinity is never emitted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from backtest.orca_random_forest import (
    FoldWindow,
    RandomForestConfig,
    TargetConfig,
    artifact_to_json,
    fit_orca_random_forest,
)
from core.orca_spectral import (
    ORCA_SPECTRAL_SCHEMA,
    OrcaInsufficientDataError,
    OrcaSpectralConfig,
    OrcaValidationError,
    build_orca_spectral_snapshots,
    spectral_feature_registry_hash,
    spectral_feature_row,
    validate_d1_price_panel,
)


ORCA_FEATURE_FRAME_SCHEMA = "orca-causal-feature-frame-v1"
ORCA_TRADITIONAL_SCHEMA = "orca-traditional-features-v1"
ORCA_PIPELINE_SCHEMA = "orca-feature-rf-research-pipeline-v1"


class OrcaFeatureFrameError(ValueError):
    """Raised when a causal feature frame cannot be constructed safely."""


def _strict_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise OrcaFeatureFrameError(f"{name} must be an integer")
    parsed = int(value)
    if parsed <= 0:
        raise OrcaFeatureFrameError(f"{name} must be positive")
    return parsed


def _ordered_unique(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    parsed = tuple(_strict_positive_int(value, name=name) for value in values)
    if tuple(sorted(set(parsed))) != parsed:
        raise OrcaFeatureFrameError(f"{name} must be unique and strictly increasing")
    return parsed


@dataclass(frozen=True)
class TraditionalFeatureConfig:
    """Versioned, documented traditional feature family.

    These defaults cover the feature groups disclosed in the ORCA paper, but
    are not represented as the unpublished exact 79-feature registry.
    """

    schema_version: str = ORCA_TRADITIONAL_SCHEMA
    return_horizons: tuple[int, ...] = (1, 5, 10, 20, 60)
    realized_vol_windows: tuple[int, ...] = (5, 10, 20, 60)
    downside_vol_window: int = 20
    drawdown_windows: tuple[int, ...] = (20, 60)
    sma_windows: tuple[int, ...] = (10, 20, 50)
    moment_windows: tuple[int, ...] = (20, 60)
    rsi_window: int = 14
    vol_of_vol_inner: int = 5
    vol_of_vol_outer: int = 20
    annualization_days: int = 252

    def __post_init__(self) -> None:
        if self.schema_version != ORCA_TRADITIONAL_SCHEMA:
            raise OrcaFeatureFrameError(
                f"schema_version must be {ORCA_TRADITIONAL_SCHEMA}"
            )
        object.__setattr__(
            self,
            "return_horizons",
            _ordered_unique(self.return_horizons, name="return_horizons"),
        )
        object.__setattr__(
            self,
            "realized_vol_windows",
            _ordered_unique(
                self.realized_vol_windows,
                name="realized_vol_windows",
            ),
        )
        object.__setattr__(
            self,
            "drawdown_windows",
            _ordered_unique(self.drawdown_windows, name="drawdown_windows"),
        )
        object.__setattr__(
            self,
            "sma_windows",
            _ordered_unique(self.sma_windows, name="sma_windows"),
        )
        object.__setattr__(
            self,
            "moment_windows",
            _ordered_unique(self.moment_windows, name="moment_windows"),
        )
        for field in (
            "downside_vol_window",
            "rsi_window",
            "vol_of_vol_inner",
            "vol_of_vol_outer",
            "annualization_days",
        ):
            object.__setattr__(
                self,
                field,
                _strict_positive_int(getattr(self, field), name=field),
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "return_horizons": list(self.return_horizons),
            "realized_vol_windows": list(self.realized_vol_windows),
            "downside_vol_window": self.downside_vol_window,
            "drawdown_windows": list(self.drawdown_windows),
            "sma_windows": list(self.sma_windows),
            "moment_windows": list(self.moment_windows),
            "rsi_window": self.rsi_window,
            "vol_of_vol_inner": self.vol_of_vol_inner,
            "vol_of_vol_outer": self.vol_of_vol_outer,
            "annualization_days": self.annualization_days,
        }


@dataclass(frozen=True)
class DynamicsConfig:
    horizons: tuple[int, ...] = (5, 10, 20)
    percentile_window: int = 252

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "horizons",
            _ordered_unique(self.horizons, name="dynamics horizons"),
        )
        object.__setattr__(
            self,
            "percentile_window",
            _strict_positive_int(
                self.percentile_window,
                name="percentile_window",
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "horizons": list(self.horizons),
            "percentile_window": self.percentile_window,
            "zscore_window": "2 * horizon",
            "acceleration": "x_t - 2*x_t-h + x_t-2h",
            "percentile_ties": "average_rank",
        }


@dataclass(frozen=True)
class OrcaFeatureFrame:
    """Feature data plus its immutable-by-convention registry metadata."""

    features: pd.DataFrame
    registry: Mapping[str, Any]
    benchmark_symbol: str
    as_of_utc: datetime


def _canonical_hash(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OrcaFeatureFrameError("registry is not canonical finite JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _utc(value: Any, *, name: str) -> datetime:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise OrcaFeatureFrameError(f"{name} must be a timestamp") from exc
    if pd.isna(parsed) or parsed.tzinfo is None:
        raise OrcaFeatureFrameError(f"{name} must be timezone-aware")
    return parsed.tz_convert("UTC").to_pydatetime().astimezone(timezone.utc)


def _realized_volatility(
    returns: pd.Series,
    window: int,
    *,
    annualization_days: int,
) -> pd.Series:
    return returns.rolling(window, min_periods=window).std(ddof=0) * math.sqrt(
        annualization_days
    )


def _downside_volatility(
    returns: pd.Series,
    window: int,
    *,
    annualization_days: int,
) -> pd.Series:
    downside = returns.clip(upper=0.0)
    return downside.pow(2).rolling(window, min_periods=window).mean().pow(
        0.5
    ) * math.sqrt(annualization_days)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    average_gain = gain.rolling(window, min_periods=window).mean()
    average_loss = loss.rolling(window, min_periods=window).mean()
    ratio = average_gain / average_loss.where(average_loss > 0.0)
    result = 100.0 - 100.0 / (1.0 + ratio)
    result = result.where(~((average_loss == 0.0) & (average_gain > 0.0)), 100.0)
    result = result.where(~((average_gain == 0.0) & (average_loss > 0.0)), 0.0)
    result = result.where(~((average_gain == 0.0) & (average_loss == 0.0)), 50.0)
    return result


def build_traditional_feature_frame(
    prices: pd.DataFrame,
    *,
    benchmark_symbol: str,
    config: TraditionalFeatureConfig = TraditionalFeatureConfig(),
) -> pd.DataFrame:
    """Build the versioned causal traditional feature family."""

    if benchmark_symbol not in prices.columns:
        raise OrcaFeatureFrameError(
            f"benchmark_symbol {benchmark_symbol!r} is not in the price panel"
        )
    close = prices[benchmark_symbol].astype(float)
    one_day = close.pct_change(fill_method=None)
    asset_returns = prices.pct_change(fill_method=None)
    prefix = "traditional_v1"
    columns: dict[str, pd.Series] = {}

    for horizon in config.return_horizons:
        columns[f"{prefix}__return_{horizon:03d}d"] = close.pct_change(
            periods=horizon,
            fill_method=None,
        )

    realized: dict[int, pd.Series] = {}
    for window in config.realized_vol_windows:
        values = _realized_volatility(
            one_day,
            window,
            annualization_days=config.annualization_days,
        )
        realized[window] = values
        columns[f"{prefix}__realized_vol_{window:03d}d"] = values
    for numerator, denominator in ((5, 20), (10, 60)):
        if numerator in realized and denominator in realized:
            base = realized[denominator]
            columns[
                f"{prefix}__realized_vol_ratio_{numerator:03d}_{denominator:03d}d"
            ] = realized[numerator] / base.where(base.abs() > 1e-15)

    columns[f"{prefix}__downside_vol_{config.downside_vol_window:03d}d"] = (
        _downside_volatility(
            one_day,
            config.downside_vol_window,
            annualization_days=config.annualization_days,
        )
    )
    for window in config.drawdown_windows:
        rolling_high = close.rolling(window, min_periods=window).max()
        columns[f"{prefix}__drawdown_{window:03d}d"] = close / rolling_high - 1.0
    for window in config.sma_windows:
        average = close.rolling(window, min_periods=window).mean()
        columns[f"{prefix}__price_to_sma_{window:03d}d"] = close / average

    columns[f"{prefix}__rsi_{config.rsi_window:03d}d"] = _rsi(
        close,
        config.rsi_window,
    )
    for window in config.moment_windows:
        rolling = one_day.rolling(window, min_periods=window)
        columns[f"{prefix}__skew_{window:03d}d"] = rolling.skew()
        columns[f"{prefix}__kurtosis_{window:03d}d"] = rolling.kurt()

    inner_vol = _realized_volatility(
        one_day,
        config.vol_of_vol_inner,
        annualization_days=config.annualization_days,
    )
    columns[
        f"{prefix}__vol_of_vol_{config.vol_of_vol_inner:03d}_{config.vol_of_vol_outer:03d}d"
    ] = inner_vol.rolling(
        config.vol_of_vol_outer,
        min_periods=config.vol_of_vol_outer,
    ).std(ddof=0)
    columns[f"{prefix}__cross_asset_dispersion_001d"] = asset_returns.std(
        axis=1,
        ddof=0,
    )

    result = pd.DataFrame(columns, index=prices.index, dtype=float)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def _mean_absolute_correlation(correlation: np.ndarray) -> float:
    row, column = np.triu_indices(correlation.shape[0], k=1)
    if len(row) == 0:
        raise OrcaFeatureFrameError("mean correlation needs at least two symbols")
    return float(np.mean(np.abs(correlation[row, column])))


def _build_static_spectral_history(
    prices: pd.DataFrame,
    *,
    config: OrcaSpectralConfig,
) -> tuple[pd.DataFrame, tuple[str, ...], str, tuple[str, ...]]:
    rows: list[dict[str, float] | None] = [None] * len(prices)
    names: tuple[str, ...] | None = None
    core_names: tuple[str, ...] | None = None
    core_registry_hash: str | None = None
    estimators: tuple[str, ...] | None = None

    for end in range(2, len(prices) + 1):
        timestamp = prices.index[end - 1]
        try:
            bundle = build_orca_spectral_snapshots(
                prices.iloc[:end],
                as_of_utc=timestamp.to_pydatetime(),
                config=config,
            )
        except OrcaInsufficientDataError:
            continue
        static = spectral_feature_row(bundle)
        current_core_names = tuple(static)
        for snapshot in bundle.snapshots:
            static[f"{snapshot.estimator}__mean_abs_correlation"] = (
                _mean_absolute_correlation(snapshot.correlation)
            )
        current_names = tuple(static)
        if names is None:
            names = current_names
            core_names = current_core_names
            core_registry_hash = spectral_feature_registry_hash(bundle)
            estimators = tuple(snapshot.estimator for snapshot in bundle.snapshots)
        elif current_names != names or current_core_names != core_names:
            raise OrcaFeatureFrameError(
                "spectral feature registry changed inside one history"
            )
        rows[end - 1] = static

    if (
        names is None
        or core_names is None
        or core_registry_hash is None
        or estimators is None
    ):
        raise OrcaInsufficientDataError(
            "price panel never reaches the configured spectral warm-up"
        )
    result = pd.DataFrame(index=prices.index, columns=list(names), dtype=float)
    for position, row in enumerate(rows):
        if row is not None:
            result.iloc[position] = [row[name] for name in names]
    return result, core_names, core_registry_hash, estimators


def _last_percentile_rank(values: np.ndarray) -> float:
    if not np.isfinite(values).all():
        return float("nan")
    current = values[-1]
    less = int(np.count_nonzero(values < current))
    equal = int(np.count_nonzero(values == current))
    return float((less + (equal + 1.0) / 2.0) / len(values))


def _dynamic_key_columns(
    static: pd.DataFrame,
    *,
    estimators: Sequence[str],
) -> tuple[str, ...]:
    suffixes = (
        "lambda_01",
        "ar_1",
        "eigenvalue_entropy",
        "effective_rank",
        "lambda1_lambda2",
        "mean_abs_correlation",
        "graph_t050__edge_density",
        "graph_t050__global_clustering",
    )
    result: list[str] = []
    for estimator in estimators:
        for suffix in suffixes:
            name = f"{estimator}__{suffix}"
            if name in static.columns:
                result.append(name)
    if not result:
        raise OrcaFeatureFrameError("no configured key spectral metrics exist")
    return tuple(result)


def build_spectral_dynamics(
    static: pd.DataFrame,
    *,
    estimators: Sequence[str],
    config: DynamicsConfig = DynamicsConfig(),
) -> pd.DataFrame:
    """Build causal derivatives of selected spectral history columns."""

    columns: dict[str, pd.Series] = {}
    for name in _dynamic_key_columns(static, estimators=estimators):
        values = static[name]
        for horizon in config.horizons:
            previous = values.shift(horizon)
            columns[f"{name}__dyn_diff_{horizon:03d}d"] = values - previous
            columns[f"{name}__dyn_roc_{horizon:03d}d"] = (
                values / previous.where(previous.abs() > 1e-15) - 1.0
            )
            z_window = horizon * 2
            rolling = values.rolling(z_window, min_periods=z_window)
            mean = rolling.mean()
            deviation = rolling.std(ddof=0)
            columns[f"{name}__dyn_z_{horizon:03d}d"] = (
                values - mean
            ) / deviation.where(deviation > 1e-15)
            columns[f"{name}__dyn_acceleration_{horizon:03d}d"] = (
                values - 2.0 * previous + values.shift(horizon * 2)
            )
        columns[f"{name}__dyn_percentile_{config.percentile_window:03d}d"] = (
            values.rolling(
                config.percentile_window,
                min_periods=config.percentile_window,
            ).apply(_last_percentile_rank, raw=True)
        )
    result = pd.DataFrame(columns, index=static.index, dtype=float)
    return result.replace([np.inf, -np.inf], np.nan)


def _registry(
    *,
    features: pd.DataFrame,
    static_names: Sequence[str],
    core_names: Sequence[str],
    core_registry_hash: str,
    traditional_names: Sequence[str],
    dynamic_names: Sequence[str],
    spectral_config: OrcaSpectralConfig,
    traditional_config: TraditionalFeatureConfig,
    dynamics_config: DynamicsConfig,
    benchmark_symbol: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": ORCA_FEATURE_FRAME_SCHEMA,
        "paper_exact_206_replication": False,
        "causal_contract": "closed_D1_at_or_before_row_timestamp",
        "warmup_policy": "NaN_allowed_infinity_forbidden",
        "benchmark_symbol": benchmark_symbol,
        "spectral_schema": ORCA_SPECTRAL_SCHEMA,
        "spectral_config_id": spectral_config.config_id,
        "core_spectral_feature_names": list(core_names),
        "core_spectral_feature_registry_hash": core_registry_hash,
        "static_spectral_feature_names": list(static_names),
        "traditional_config": traditional_config.to_mapping(),
        "traditional_feature_names": list(traditional_names),
        "dynamics_config": dynamics_config.to_mapping(),
        "dynamic_feature_names": list(dynamic_names),
        "feature_names": list(features.columns),
        "feature_count": int(features.shape[1]),
    }
    return {**body, "hash": _canonical_hash(body)}


def validate_feature_registry(frame: OrcaFeatureFrame) -> None:
    if not isinstance(frame, OrcaFeatureFrame):
        raise OrcaFeatureFrameError("frame must be OrcaFeatureFrame")
    body = dict(frame.registry)
    expected_hash = body.pop("hash", None)
    if expected_hash != _canonical_hash(body):
        raise OrcaFeatureFrameError("feature registry hash mismatch")
    names = list(frame.features.columns)
    if body.get("feature_names") != names or body.get("feature_count") != len(names):
        raise OrcaFeatureFrameError("feature frame order differs from registry")
    values = frame.features.to_numpy(dtype=float, copy=False)
    if np.isinf(values).any():
        raise OrcaFeatureFrameError("feature frame contains infinity")


def build_orca_feature_frame(
    prices: pd.DataFrame,
    *,
    benchmark_symbol: str,
    as_of_utc: datetime,
    spectral_config: OrcaSpectralConfig = OrcaSpectralConfig(),
    traditional_config: TraditionalFeatureConfig = TraditionalFeatureConfig(),
    dynamics_config: DynamicsConfig = DynamicsConfig(),
) -> OrcaFeatureFrame:
    """Build a deterministic point-in-time closed-D1 ORCA feature frame."""

    as_of = _utc(as_of_utc, name="as_of_utc")
    try:
        panel = validate_d1_price_panel(prices, as_of_utc=as_of)
    except OrcaValidationError as exc:
        raise OrcaFeatureFrameError(str(exc)) from exc
    symbol = benchmark_symbol.strip() if isinstance(benchmark_symbol, str) else ""
    if not symbol or symbol not in panel.columns:
        raise OrcaFeatureFrameError(
            f"benchmark_symbol {benchmark_symbol!r} is not in the price panel"
        )

    static, core_names, core_hash, estimators = _build_static_spectral_history(
        panel,
        config=spectral_config,
    )
    traditional = build_traditional_feature_frame(
        panel,
        benchmark_symbol=symbol,
        config=traditional_config,
    )
    dynamics = build_spectral_dynamics(
        static,
        estimators=estimators,
        config=dynamics_config,
    )
    features = pd.concat([static, traditional, dynamics], axis=1)
    features = features.astype(float).replace([np.inf, -np.inf], np.nan)
    registry = _registry(
        features=features,
        static_names=tuple(static.columns),
        core_names=core_names,
        core_registry_hash=core_hash,
        traditional_names=tuple(traditional.columns),
        dynamic_names=tuple(dynamics.columns),
        spectral_config=spectral_config,
        traditional_config=traditional_config,
        dynamics_config=dynamics_config,
        benchmark_symbol=symbol,
    )
    result = OrcaFeatureFrame(
        features=features,
        registry=registry,
        benchmark_symbol=symbol,
        as_of_utc=as_of,
    )
    validate_feature_registry(result)
    return result


def build_and_fit_orca_pipeline(
    prices: pd.DataFrame,
    *,
    benchmark_symbol: str,
    as_of_utc: datetime,
    spectral_config: OrcaSpectralConfig = OrcaSpectralConfig(),
    traditional_config: TraditionalFeatureConfig = TraditionalFeatureConfig(),
    dynamics_config: DynamicsConfig = DynamicsConfig(),
    fold_window: FoldWindow = FoldWindow(),
    target_config: TargetConfig = TargetConfig(),
    forest_config: RandomForestConfig = RandomForestConfig(),
) -> dict[str, Any]:
    """Build features and fit the existing RF research model without promotion."""

    frame = build_orca_feature_frame(
        prices,
        benchmark_symbol=benchmark_symbol,
        as_of_utc=as_of_utc,
        spectral_config=spectral_config,
        traditional_config=traditional_config,
        dynamics_config=dynamics_config,
    )
    panel = validate_d1_price_panel(prices, as_of_utc=frame.as_of_utc)
    model = fit_orca_random_forest(
        frame.features,
        panel[frame.benchmark_symbol],
        fold_window=fold_window,
        target_config=target_config,
        forest_config=forest_config,
    )
    if model["feature_registry"]["names"] != list(frame.features.columns):
        raise OrcaFeatureFrameError("RF artifact did not preserve feature order")
    return {
        "schema": ORCA_PIPELINE_SCHEMA,
        "status": "research_only",
        "auto_promote": False,
        "feature_registry": dict(frame.registry),
        "model_artifact": model,
    }


def _read_price_csv(path: Path, *, timestamp_column: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    if timestamp_column not in data.columns:
        raise OrcaFeatureFrameError(f"timestamp column {timestamp_column!r} is missing")
    timestamp = pd.to_datetime(data.pop(timestamp_column), utc=True, errors="raise")
    data.index = pd.DatetimeIndex(timestamp)
    return data


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build closed-D1 ORCA features and fit a research-only RF artifact"
    )
    parser.add_argument("--prices-csv", required=True, type=Path)
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--benchmark-symbol", required=True)
    parser.add_argument("--as-of-utc", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)

    prices = _read_price_csv(
        arguments.prices_csv,
        timestamp_column=arguments.timestamp_column,
    )
    payload = build_and_fit_orca_pipeline(
        prices,
        benchmark_symbol=arguments.benchmark_symbol,
        as_of_utc=_utc(arguments.as_of_utc, name="as_of_utc"),
    )
    # The nested model is already validated by its canonical serializer.
    artifact_to_json(payload["model_artifact"])
    output = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())


__all__ = [
    "DynamicsConfig",
    "ORCA_FEATURE_FRAME_SCHEMA",
    "ORCA_PIPELINE_SCHEMA",
    "ORCA_TRADITIONAL_SCHEMA",
    "OrcaFeatureFrame",
    "OrcaFeatureFrameError",
    "TraditionalFeatureConfig",
    "build_and_fit_orca_pipeline",
    "build_orca_feature_frame",
    "build_spectral_dynamics",
    "build_traditional_feature_frame",
    "main",
    "validate_feature_registry",
]
