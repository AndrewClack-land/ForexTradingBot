"""Causal, production-independent spectral primitives for ORCA research.

The module deliberately has no broker, live-core, scikit-learn, NetworkX, or
configuration imports.  Timestamps in a price panel are interpreted as the
actual close/availability timestamps of fully closed D1 bars.  Callers must
therefore provide an aware ``as_of_utc`` and the latest panel timestamp may
not exceed it.

Only static spectral snapshots are built here.  Temporal derivatives over
5/10/20 observations belong to a point-in-time history layer so that this
module remains a small, deterministic mathematical boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


ORCA_SPECTRAL_SCHEMA = "orca-spectral-v1"
ORCA_UNIVERSE_PROFILE_SCHEMA = "orca-universe-profile-v1"
BCD_AUC_SCHEMA = "balanced-crisis-detection-auc-v1"


class OrcaSpectralError(ValueError):
    """Base error for an invalid or numerically unsafe ORCA calculation."""


class OrcaValidationError(OrcaSpectralError):
    """Raised when a public input violates the causal data contract."""


class OrcaInsufficientDataError(OrcaSpectralError):
    """Raised instead of manufacturing a feature from insufficient data."""


def _strict_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise OrcaValidationError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise OrcaValidationError(f"{name} must be positive")
    return result


def _finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OrcaValidationError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise OrcaValidationError(f"{name} must be finite")
    return result


def _aware_utc(value: Any, *, name: str) -> datetime:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise OrcaValidationError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _number_token(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return format(float(value), ".12g").replace(".", "p")


@dataclass(frozen=True)
class OrcaUniverseProfile:
    """Versioned exact-universe contract for a spectral feature family."""

    profile_id: str
    symbols: tuple[str, ...]
    absorption_ranks: tuple[int, ...]
    expected_assets: int
    minimum_assets: int
    schema_version: str = ORCA_UNIVERSE_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ORCA_UNIVERSE_PROFILE_SCHEMA:
            raise OrcaValidationError(
                f"schema_version must be {ORCA_UNIVERSE_PROFILE_SCHEMA}"
            )

        profile_id = self.profile_id.strip() if isinstance(self.profile_id, str) else ""
        if not profile_id or any(
            not (character.isalnum() or character in "-._")
            for character in profile_id
        ):
            raise OrcaValidationError(
                "profile_id must be a non-empty stable identifier"
            )

        symbols: list[str] = []
        for raw_symbol in self.symbols:
            if not isinstance(raw_symbol, str) or not raw_symbol.strip():
                raise OrcaValidationError(
                    "profile symbols must be non-empty strings"
                )
            symbols.append(raw_symbol.strip())
        if len(symbols) < 2 or len(set(symbols)) != len(symbols):
            raise OrcaValidationError(
                "profile symbols must contain at least two unique assets"
            )

        expected_assets = _strict_positive_int(
            self.expected_assets,
            name="expected_assets",
        )
        minimum_assets = _strict_positive_int(
            self.minimum_assets,
            name="minimum_assets",
        )
        if expected_assets != len(symbols):
            raise OrcaValidationError(
                "expected_assets must equal the exact profile symbol count"
            )
        if minimum_assets < 2 or minimum_assets > expected_assets:
            raise OrcaValidationError(
                "minimum_assets must lie between 2 and expected_assets"
            )

        ranks = tuple(
            _strict_positive_int(value, name="absorption_ranks")
            for value in self.absorption_ranks
        )
        if tuple(sorted(set(ranks))) != ranks:
            raise OrcaValidationError(
                "absorption_ranks must be unique and strictly increasing"
            )
        if ranks[-1] >= minimum_assets:
            raise OrcaValidationError(
                "profile absorption ranks must be below minimum_assets; "
                "a full-rank absorption ratio is identically 1.0"
            )

        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "symbols", tuple(symbols))
        object.__setattr__(self, "absorption_ranks", ranks)
        object.__setattr__(self, "expected_assets", expected_assets)
        object.__setattr__(self, "minimum_assets", minimum_assets)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "symbols": list(self.symbols),
            "absorption_ranks": list(self.absorption_ranks),
            "expected_assets": self.expected_assets,
            "minimum_assets": self.minimum_assets,
        }

    @property
    def contract_hash(self) -> str:
        canonical = json.dumps(
            self.to_mapping(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return sha256(canonical).hexdigest()

    def validate_symbols(self, symbols: Sequence[str]) -> None:
        actual = tuple(symbols)
        if actual != self.symbols:
            raise OrcaValidationError(
                f"universe profile {self.profile_id} requires exact ordered "
                f"symbols {self.symbols}; received {actual}"
            )


ORCA_FX_GOLD_4_PROFILE = OrcaUniverseProfile(
    profile_id="orca-fx-gold-4-v1",
    symbols=("EURUSD", "GBPUSD", "USDCAD", "GOLD"),
    absorption_ranks=(1, 2, 3),
    expected_assets=4,
    minimum_assets=4,
)


@dataclass(frozen=True)
class OrcaSpectralConfig:
    """Versioned feature contract; defaults reproduce declared ORCA scales."""

    schema_version: str = ORCA_SPECTRAL_SCHEMA
    rolling_windows: tuple[int, ...] = (60, 120)
    ewm_halflife: float = 30.0
    ewm_min_periods: int = 60
    absorption_ranks: tuple[int, ...] = (1, 3, 5)
    graph_thresholds: tuple[float, ...] = (0.3, 0.5, 0.7)
    numerical_tolerance: float = 1e-10
    universe_profile: Optional[OrcaUniverseProfile] = None

    def __post_init__(self) -> None:
        if self.schema_version != ORCA_SPECTRAL_SCHEMA:
            raise OrcaValidationError(f"schema_version must be {ORCA_SPECTRAL_SCHEMA}")

        windows = tuple(
            _strict_positive_int(value, name="rolling_windows")
            for value in self.rolling_windows
        )
        if any(value < 2 for value in windows):
            raise OrcaValidationError("rolling windows must be at least 2")
        if tuple(sorted(set(windows))) != windows:
            raise OrcaValidationError(
                "rolling_windows must be unique and strictly increasing"
            )

        halflife = _finite_float(self.ewm_halflife, name="ewm_halflife")
        if halflife <= 0.0:
            raise OrcaValidationError("ewm_halflife must be positive")
        min_periods = _strict_positive_int(
            self.ewm_min_periods,
            name="ewm_min_periods",
        )
        if min_periods < 2:
            raise OrcaValidationError("ewm_min_periods must be at least 2")

        ranks = tuple(
            _strict_positive_int(value, name="absorption_ranks")
            for value in self.absorption_ranks
        )
        if tuple(sorted(set(ranks))) != ranks:
            raise OrcaValidationError(
                "absorption_ranks must be unique and strictly increasing"
            )
        profile = self.universe_profile
        if profile is not None:
            if not isinstance(profile, OrcaUniverseProfile):
                raise OrcaValidationError(
                    "universe_profile must be OrcaUniverseProfile"
                )
            if ranks != profile.absorption_ranks:
                raise OrcaValidationError(
                    "absorption_ranks must match the universe profile"
                )

        thresholds = tuple(
            _finite_float(value, name="graph_thresholds")
            for value in self.graph_thresholds
        )
        if any(not 0.0 < value < 1.0 for value in thresholds):
            raise OrcaValidationError(
                "graph_thresholds must lie strictly inside (0, 1)"
            )
        if tuple(sorted(set(thresholds))) != thresholds:
            raise OrcaValidationError(
                "graph_thresholds must be unique and strictly increasing"
            )

        tolerance = _finite_float(
            self.numerical_tolerance,
            name="numerical_tolerance",
        )
        if not 0.0 < tolerance < 1.0:
            raise OrcaValidationError(
                "numerical_tolerance must lie strictly inside (0, 1)"
            )

        object.__setattr__(self, "rolling_windows", windows)
        object.__setattr__(self, "ewm_halflife", halflife)
        object.__setattr__(self, "ewm_min_periods", min_periods)
        object.__setattr__(self, "absorption_ranks", ranks)
        object.__setattr__(self, "graph_thresholds", thresholds)
        object.__setattr__(self, "numerical_tolerance", tolerance)

    def to_mapping(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "schema_version": self.schema_version,
            "rolling_windows": list(self.rolling_windows),
            "ewm_halflife": self.ewm_halflife,
            "ewm_min_periods": self.ewm_min_periods,
            "absorption_ranks": list(self.absorption_ranks),
            "graph_thresholds": list(self.graph_thresholds),
            "numerical_tolerance": self.numerical_tolerance,
        }
        if self.universe_profile is not None:
            mapping["universe_profile"] = self.universe_profile.to_mapping()
        return mapping

    @property
    def config_id(self) -> str:
        canonical = json.dumps(
            self.to_mapping(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return sha256(canonical).hexdigest()


ORCA_FX_GOLD_4_SPECTRAL_CONFIG = OrcaSpectralConfig(
    absorption_ranks=ORCA_FX_GOLD_4_PROFILE.absorption_ranks,
    universe_profile=ORCA_FX_GOLD_4_PROFILE,
)


ORCA_UNIVERSE_PROFILES: Mapping[str, OrcaUniverseProfile] = MappingProxyType(
    {ORCA_FX_GOLD_4_PROFILE.profile_id: ORCA_FX_GOLD_4_PROFILE}
)

_PROFILE_SPECTRAL_CONFIGS: Mapping[str, OrcaSpectralConfig] = MappingProxyType(
    {ORCA_FX_GOLD_4_PROFILE.profile_id: ORCA_FX_GOLD_4_SPECTRAL_CONFIG}
)


def resolve_universe_profile(name: str) -> OrcaUniverseProfile:
    """Return a registered exact-universe profile, failing closed on typos."""

    token = str(name or "").strip()
    profile = ORCA_UNIVERSE_PROFILES.get(token)
    if profile is None:
        known = ", ".join(sorted(ORCA_UNIVERSE_PROFILES)) or "<none>"
        raise OrcaValidationError(
            f"unknown ORCA universe profile {token!r}; registered profiles: {known}"
        )
    return profile


def spectral_config_for_profile(name: str) -> OrcaSpectralConfig:
    """Return the frozen spectral contract paired with a universe profile.

    The pairing is registered rather than derived so a profile can never be
    combined with absorption ranks it was not validated against.
    """

    profile = resolve_universe_profile(name)
    config = _PROFILE_SPECTRAL_CONFIGS.get(profile.profile_id)
    if config is None:  # pragma: no cover - registry is kept in lockstep
        raise OrcaValidationError(
            f"universe profile {profile.profile_id!r} has no registered "
            "spectral configuration"
        )
    return config


@dataclass(frozen=True)
class GraphMetrics:
    threshold: float
    edge_count: int
    edge_density: float
    mean_degree: float
    degree_std: float
    max_degree: int
    isolated_nodes: int
    degree_centralization: float
    global_clustering: float


@dataclass(frozen=True)
class SignedEdge:
    threshold: float
    source: str
    target: str
    correlation: float

    @property
    def sign(self) -> int:
        return 1 if self.correlation > 0.0 else -1

    @property
    def absolute_correlation(self) -> float:
        return abs(self.correlation)


@dataclass(frozen=True)
class CorrelationSnapshot:
    estimator: str
    symbols: tuple[str, ...]
    returns_start_utc: datetime
    returns_end_utc: datetime
    observation_count: int
    effective_sample_length: float
    correlation: np.ndarray


@dataclass(frozen=True)
class SpectralSnapshot:
    estimator: str
    symbols: tuple[str, ...]
    returns_start_utc: datetime
    returns_end_utc: datetime
    observation_count: int
    effective_sample_length: float
    correlation: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    absorption_ratios: Mapping[int, float]
    eigenvalue_entropy: float
    effective_rank: float
    lambda1_lambda2: Optional[float]
    condition_number: Optional[float]
    marchenko_pastur_q: float
    marchenko_pastur_lower: float
    marchenko_pastur_upper: float
    marchenko_pastur_outlier_count: int
    marchenko_pastur_upper_outlier_count: int
    marchenko_pastur_lambda1_excess: float
    dominant_eigenvector_hhi: float
    dominant_eigenvector_participation: float
    graph_metrics: Mapping[float, GraphMetrics]
    node_coordinates_3d: Mapping[str, tuple[float, float, float]]
    signed_edges: Mapping[float, tuple[SignedEdge, ...]]


@dataclass(frozen=True)
class OrcaSpectralBundle:
    schema_version: str
    config_id: str
    as_of_utc: datetime
    symbols: tuple[str, ...]
    returns_start_utc: datetime
    returns_end_utc: datetime
    snapshots: tuple[SpectralSnapshot, ...]
    profile_id: Optional[str] = None
    profile_contract_hash: Optional[str] = None

    def snapshot(self, estimator: str) -> SpectralSnapshot:
        for item in self.snapshots:
            if item.estimator == estimator:
                return item
        raise KeyError(estimator)


@dataclass(frozen=True)
class BcdAucResult:
    schema_version: str
    status: str
    bcd_auc: Optional[float]
    rally_auc: Optional[float]
    crash_auc: Optional[float]
    observation_count: int
    rally_positive_count: int
    rally_negative_count: int
    crash_positive_count: int
    crash_negative_count: int
    reason: Optional[str] = None


def validate_d1_price_panel(
    prices: pd.DataFrame,
    *,
    as_of_utc: datetime,
) -> pd.DataFrame:
    """Validate an aligned wide panel whose index is the D1 close timestamp.

    The function does not infer that a date label represents a closed candle.
    Instead, the index itself is contractually the bar-close/availability time,
    and it must be no later than ``as_of_utc``.
    """

    as_of = _aware_utc(as_of_utc, name="as_of_utc")
    if not isinstance(prices, pd.DataFrame):
        raise OrcaValidationError("prices must be a pandas DataFrame")
    if prices.empty or len(prices) < 2:
        raise OrcaInsufficientDataError("prices need at least two D1 rows")
    if isinstance(prices.columns, pd.MultiIndex):
        raise OrcaValidationError("prices must have a flat symbol axis")
    if len(prices.columns) < 2:
        raise OrcaInsufficientDataError("prices need at least two symbols")
    if prices.columns.has_duplicates:
        raise OrcaValidationError("price symbols must be unique")

    symbols: list[str] = []
    for raw_symbol in prices.columns:
        if not isinstance(raw_symbol, str) or not raw_symbol.strip():
            raise OrcaValidationError(
                "every price column must be a non-empty symbol string"
            )
        symbols.append(raw_symbol.strip())
    if len(set(symbols)) != len(symbols):
        raise OrcaValidationError("price symbols must remain unique after trimming")

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise OrcaValidationError("prices index must be a DatetimeIndex")
    if prices.index.tz is None:
        raise OrcaValidationError("prices index must be timezone-aware")
    index = prices.index.tz_convert("UTC")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise OrcaValidationError("prices index must be strictly increasing and unique")
    if index.normalize().has_duplicates:
        raise OrcaValidationError("prices may contain only one D1 bar per UTC date")
    latest = index[-1].to_pydatetime().astimezone(timezone.utc)
    if latest > as_of:
        raise OrcaValidationError("latest D1 bar closes after as_of_utc")

    try:
        values = prices.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise OrcaValidationError("all prices must be numeric") from exc
    if not np.isfinite(values).all():
        raise OrcaValidationError("prices must be fully aligned and finite")
    if np.any(values <= 0.0):
        raise OrcaValidationError("prices must be strictly positive")

    result = pd.DataFrame(values, index=index, columns=symbols)
    result.index.name = prices.index.name
    return result


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Return causal one-period simple returns without implicit filling."""

    if not isinstance(prices, pd.DataFrame) or len(prices) < 2:
        raise OrcaInsufficientDataError("prices need at least two rows")
    try:
        values = prices.to_numpy(dtype=float, copy=False)
    except (TypeError, ValueError) as exc:
        raise OrcaValidationError("all prices must be numeric") from exc
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise OrcaValidationError("prices must be finite and strictly positive")
    result = prices.pct_change(fill_method=None).iloc[1:].astype(float)
    if result.empty or not np.isfinite(result.to_numpy()).all():
        raise OrcaValidationError("simple returns are not finite")
    return result


def _frozen_array(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _weighted_correlation(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    if values.ndim != 2 or weights.ndim != 1:
        raise OrcaValidationError("correlation inputs have invalid dimensions")
    if len(values) != len(weights) or len(values) < 2:
        raise OrcaInsufficientDataError(
            "correlation needs at least two weighted observations"
        )
    normalized = np.asarray(weights, dtype=float)
    if not np.isfinite(normalized).all() or np.any(normalized < 0.0):
        raise OrcaValidationError("correlation weights are invalid")
    total = float(normalized.sum())
    if total <= 0.0:
        raise OrcaValidationError("correlation weights sum to zero")
    normalized = normalized / total
    mean = normalized @ values
    centered = values - mean
    covariance = (centered * normalized[:, None]).T @ centered
    variance = np.diag(covariance)
    if np.any(variance <= tolerance):
        raise OrcaValidationError(
            "correlation is undefined for a constant return series"
        )
    scale = np.sqrt(variance)
    correlation = covariance / np.outer(scale, scale)
    correlation = (correlation + correlation.T) / 2.0
    correlation = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    if not np.isfinite(correlation).all():
        raise OrcaValidationError("correlation matrix is not finite")
    return correlation


def build_correlation_snapshots(
    prices: pd.DataFrame,
    *,
    as_of_utc: datetime,
    config: OrcaSpectralConfig = OrcaSpectralConfig(),
) -> tuple[CorrelationSnapshot, ...]:
    """Build configured rolling and EWM correlation snapshots."""

    if not isinstance(config, OrcaSpectralConfig):
        raise OrcaValidationError("config must be OrcaSpectralConfig")
    panel = validate_d1_price_panel(prices, as_of_utc=as_of_utc)
    if config.universe_profile is not None:
        config.universe_profile.validate_symbols(tuple(panel.columns))
    returns = simple_returns(panel)
    symbols = tuple(str(column) for column in returns.columns)
    snapshots: list[CorrelationSnapshot] = []

    for window in config.rolling_windows:
        if len(returns) < window:
            raise OrcaInsufficientDataError(
                f"rolling_{window}d needs {window} returns; received {len(returns)}"
            )
        sample = returns.tail(window)
        weights = np.ones(window, dtype=float)
        correlation = _weighted_correlation(
            sample.to_numpy(dtype=float),
            weights,
            tolerance=config.numerical_tolerance,
        )
        snapshots.append(
            CorrelationSnapshot(
                estimator=f"rolling_{window}d",
                symbols=symbols,
                returns_start_utc=sample.index[0].to_pydatetime(),
                returns_end_utc=sample.index[-1].to_pydatetime(),
                observation_count=window,
                effective_sample_length=float(window),
                correlation=_frozen_array(correlation),
            )
        )

    if len(returns) < config.ewm_min_periods:
        raise OrcaInsufficientDataError(
            f"EWM needs {config.ewm_min_periods} returns; received {len(returns)}"
        )
    age = np.arange(len(returns) - 1, -1, -1, dtype=float)
    weights = np.exp2(-age / config.ewm_halflife)
    correlation = _weighted_correlation(
        returns.to_numpy(dtype=float),
        weights,
        tolerance=config.numerical_tolerance,
    )
    effective_length = float(weights.sum() ** 2 / np.square(weights).sum())
    snapshots.append(
        CorrelationSnapshot(
            estimator=f"ewm_hl{_number_token(config.ewm_halflife)}d",
            symbols=symbols,
            returns_start_utc=returns.index[0].to_pydatetime(),
            returns_end_utc=returns.index[-1].to_pydatetime(),
            observation_count=len(returns),
            effective_sample_length=effective_length,
            correlation=_frozen_array(correlation),
        )
    )
    return tuple(snapshots)


def _canonical_eigenbasis(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    """Choose deterministic signs and bases for degenerate eigenspaces."""

    result = np.array(eigenvectors, dtype=float, copy=True)
    size = len(eigenvalues)
    start = 0
    while start < size:
        end = start + 1
        scale = max(1.0, abs(float(eigenvalues[start])))
        while (
            end < size
            and abs(float(eigenvalues[end] - eigenvalues[start])) <= tolerance * scale
        ):
            end += 1

        width = end - start
        if width > 1:
            source = result[:, start:end]
            projector = source @ source.T
            basis: list[np.ndarray] = []
            for axis in range(size):
                candidate = np.array(projector[:, axis], copy=True)
                for previous in basis:
                    candidate -= previous * float(previous @ candidate)
                norm = float(np.linalg.norm(candidate))
                if norm > tolerance:
                    candidate /= norm
                    pivot = int(np.argmax(np.abs(candidate)))
                    if candidate[pivot] < 0.0:
                        candidate *= -1.0
                    basis.append(candidate)
                if len(basis) == width:
                    break
            if len(basis) != width:
                raise OrcaValidationError(
                    "could not canonicalize a degenerate eigenspace"
                )
            result[:, start:end] = np.column_stack(basis)
        else:
            vector = result[:, start]
            pivot = int(np.argmax(np.abs(vector)))
            if vector[pivot] < 0.0:
                result[:, start] *= -1.0
        start = end
    return result


def _graph_metrics(adjacency: np.ndarray, threshold: float) -> GraphMetrics:
    node_count = int(adjacency.shape[0])
    degree = adjacency.sum(axis=1).astype(float)
    edge_count = int(adjacency.sum() // 2)
    possible_edges = node_count * (node_count - 1) / 2.0
    edge_density = edge_count / possible_edges if possible_edges else 0.0
    mean_degree = float(degree.mean()) if node_count else 0.0
    degree_std = float(degree.std(ddof=0)) if node_count else 0.0
    max_degree = int(degree.max()) if node_count else 0
    isolated_nodes = int(np.count_nonzero(degree == 0.0))
    if node_count > 2:
        centralization = float(
            np.sum(max_degree - degree) / ((node_count - 1) * (node_count - 2))
        )
    else:
        centralization = 0.0

    triples_twice = float(np.sum(degree * (degree - 1.0)))
    if triples_twice > 0.0:
        closed_walks = float(np.trace(adjacency @ adjacency @ adjacency))
        clustering = closed_walks / triples_twice
    else:
        clustering = 0.0
    clustering = min(1.0, max(0.0, clustering))

    return GraphMetrics(
        threshold=threshold,
        edge_count=edge_count,
        edge_density=float(edge_density),
        mean_degree=mean_degree,
        degree_std=degree_std,
        max_degree=max_degree,
        isolated_nodes=isolated_nodes,
        degree_centralization=centralization,
        global_clustering=clustering,
    )


def _coordinates_from_non_dominant_eigenvectors(
    symbols: Sequence[str],
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> Mapping[str, tuple[float, float, float]]:
    node_count = len(symbols)
    coordinates = np.zeros((node_count, 3), dtype=float)
    available = min(3, max(0, node_count - 1))
    for dimension in range(available):
        component = dimension + 1
        coordinates[:, dimension] = eigenvectors[:, component] * math.sqrt(
            max(0.0, float(eigenvalues[component]))
        )
    maximum = float(np.max(np.abs(coordinates))) if coordinates.size else 0.0
    if maximum > 0.0:
        coordinates /= maximum
    return MappingProxyType(
        {
            symbol: tuple(float(value) for value in coordinates[index])
            for index, symbol in enumerate(symbols)
        }
    )


def _spectral_snapshot(
    source: CorrelationSnapshot,
    config: OrcaSpectralConfig,
) -> SpectralSnapshot:
    correlation = np.asarray(source.correlation, dtype=float)
    node_count = len(source.symbols)
    if correlation.shape != (node_count, node_count):
        raise OrcaValidationError("correlation shape does not match symbols")
    if config.absorption_ranks[-1] > node_count:
        raise OrcaInsufficientDataError(
            f"AR{config.absorption_ranks[-1]} needs at least "
            f"{config.absorption_ranks[-1]} assets; received {node_count}"
        )

    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=float)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=float)
    if float(eigenvalues[-1]) < -config.numerical_tolerance:
        raise OrcaValidationError("correlation matrix is not positive semidefinite")
    eigenvalues[np.abs(eigenvalues) <= config.numerical_tolerance] = 0.0
    eigenvectors = _canonical_eigenbasis(
        eigenvalues,
        eigenvectors,
        tolerance=config.numerical_tolerance,
    )

    total = float(eigenvalues.sum())
    if total <= config.numerical_tolerance:
        raise OrcaValidationError("eigenvalue mass is not positive")
    ratios = MappingProxyType(
        {
            rank: float(eigenvalues[:rank].sum() / total)
            for rank in config.absorption_ranks
        }
    )
    probabilities = np.clip(eigenvalues / total, 0.0, 1.0)
    nonzero = probabilities > config.numerical_tolerance
    entropy = float(-np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])))
    effective_rank = float(math.exp(entropy))

    lambda2 = float(eigenvalues[1]) if node_count >= 2 else 0.0
    spectral_gap = (
        float(eigenvalues[0] / lambda2)
        if lambda2 > config.numerical_tolerance
        else None
    )
    smallest = float(eigenvalues[-1])
    condition_number = (
        float(eigenvalues[0] / smallest)
        if smallest > config.numerical_tolerance
        else None
    )

    effective_length = float(source.effective_sample_length)
    if effective_length <= 1.0:
        raise OrcaInsufficientDataError(
            "Marchenko-Pastur bounds need effective sample length above 1"
        )
    q = float(node_count / effective_length)
    root_q = math.sqrt(q)
    mp_lower = float((1.0 - root_q) ** 2)
    mp_upper = float((1.0 + root_q) ** 2)
    below = eigenvalues < mp_lower - config.numerical_tolerance
    above = eigenvalues > mp_upper + config.numerical_tolerance
    mp_outliers = int(np.count_nonzero(below | above))
    mp_upper_outliers = int(np.count_nonzero(above))

    dominant = eigenvectors[:, 0]
    absolute_loading = np.abs(dominant)
    absolute_loading /= float(absolute_loading.sum())
    hhi = float(np.square(absolute_loading).sum())
    inverse_participation = float(np.power(dominant, 4).sum())
    participation = (
        1.0 / inverse_participation
        if inverse_participation > config.numerical_tolerance
        else 0.0
    )

    graph: dict[float, GraphMetrics] = {}
    edges: dict[float, tuple[SignedEdge, ...]] = {}
    absolute_correlation = np.abs(correlation)
    for threshold in config.graph_thresholds:
        adjacency = (absolute_correlation > threshold).astype(float)
        np.fill_diagonal(adjacency, 0.0)
        graph[threshold] = _graph_metrics(adjacency, threshold)
        threshold_edges: list[SignedEdge] = []
        for left in range(node_count):
            for right in range(left + 1, node_count):
                value = float(correlation[left, right])
                if abs(value) > threshold:
                    threshold_edges.append(
                        SignedEdge(
                            threshold=threshold,
                            source=source.symbols[left],
                            target=source.symbols[right],
                            correlation=value,
                        )
                    )
        edges[threshold] = tuple(threshold_edges)

    return SpectralSnapshot(
        estimator=source.estimator,
        symbols=source.symbols,
        returns_start_utc=source.returns_start_utc,
        returns_end_utc=source.returns_end_utc,
        observation_count=source.observation_count,
        effective_sample_length=effective_length,
        correlation=_frozen_array(correlation),
        eigenvalues=_frozen_array(eigenvalues),
        eigenvectors=_frozen_array(eigenvectors),
        absorption_ratios=ratios,
        eigenvalue_entropy=entropy,
        effective_rank=effective_rank,
        lambda1_lambda2=spectral_gap,
        condition_number=condition_number,
        marchenko_pastur_q=q,
        marchenko_pastur_lower=mp_lower,
        marchenko_pastur_upper=mp_upper,
        marchenko_pastur_outlier_count=mp_outliers,
        marchenko_pastur_upper_outlier_count=mp_upper_outliers,
        marchenko_pastur_lambda1_excess=float(eigenvalues[0] - mp_upper),
        dominant_eigenvector_hhi=hhi,
        dominant_eigenvector_participation=float(participation),
        graph_metrics=MappingProxyType(graph),
        node_coordinates_3d=_coordinates_from_non_dominant_eigenvectors(
            source.symbols,
            eigenvalues,
            eigenvectors,
        ),
        signed_edges=MappingProxyType(edges),
    )


def build_orca_spectral_snapshots(
    prices: pd.DataFrame,
    *,
    as_of_utc: datetime,
    config: OrcaSpectralConfig = OrcaSpectralConfig(),
) -> OrcaSpectralBundle:
    """Build all static causal snapshots under one frozen feature contract."""

    as_of = _aware_utc(as_of_utc, name="as_of_utc")
    correlation_snapshots = build_correlation_snapshots(
        prices,
        as_of_utc=as_of,
        config=config,
    )
    snapshots = tuple(
        _spectral_snapshot(item, config) for item in correlation_snapshots
    )
    return OrcaSpectralBundle(
        schema_version=ORCA_SPECTRAL_SCHEMA,
        config_id=config.config_id,
        as_of_utc=as_of,
        symbols=snapshots[0].symbols,
        returns_start_utc=min(item.returns_start_utc for item in snapshots),
        returns_end_utc=max(item.returns_end_utc for item in snapshots),
        snapshots=snapshots,
        profile_id=(
            config.universe_profile.profile_id
            if config.universe_profile is not None
            else None
        ),
        profile_contract_hash=(
            config.universe_profile.contract_hash
            if config.universe_profile is not None
            else None
        ),
    )


def _threshold_token(value: float) -> str:
    return f"t{int(round(value * 100)):03d}"


def spectral_feature_row(bundle: OrcaSpectralBundle) -> dict[str, float]:
    """Flatten a bundle into deterministic estimator-prefixed finite values."""

    if not isinstance(bundle, OrcaSpectralBundle):
        raise OrcaValidationError("bundle must be OrcaSpectralBundle")
    row: dict[str, float] = {}
    for snapshot in bundle.snapshots:
        prefix = snapshot.estimator
        row[f"{prefix}__effective_sample_length"] = float(
            snapshot.effective_sample_length
        )
        for index, value in enumerate(snapshot.eigenvalues, start=1):
            row[f"{prefix}__lambda_{index:02d}"] = float(value)
        for rank, value in sorted(snapshot.absorption_ratios.items()):
            row[f"{prefix}__ar_{rank}"] = float(value)
        row[f"{prefix}__eigenvalue_entropy"] = snapshot.eigenvalue_entropy
        row[f"{prefix}__effective_rank"] = snapshot.effective_rank

        gap_missing = snapshot.lambda1_lambda2 is None
        row[f"{prefix}__lambda1_lambda2"] = (
            0.0 if gap_missing else float(snapshot.lambda1_lambda2)
        )
        row[f"{prefix}__lambda1_lambda2_missing"] = float(gap_missing)
        condition_missing = snapshot.condition_number is None
        row[f"{prefix}__condition_number"] = (
            0.0 if condition_missing else float(snapshot.condition_number)
        )
        row[f"{prefix}__condition_number_missing"] = float(condition_missing)

        row[f"{prefix}__mp_q"] = snapshot.marchenko_pastur_q
        row[f"{prefix}__mp_lower"] = snapshot.marchenko_pastur_lower
        row[f"{prefix}__mp_upper"] = snapshot.marchenko_pastur_upper
        row[f"{prefix}__mp_outlier_count"] = float(
            snapshot.marchenko_pastur_outlier_count
        )
        row[f"{prefix}__mp_upper_outlier_count"] = float(
            snapshot.marchenko_pastur_upper_outlier_count
        )
        row[f"{prefix}__mp_lambda1_excess"] = snapshot.marchenko_pastur_lambda1_excess
        row[f"{prefix}__dominant_eigenvector_hhi"] = snapshot.dominant_eigenvector_hhi
        row[f"{prefix}__dominant_eigenvector_participation"] = (
            snapshot.dominant_eigenvector_participation
        )

        for threshold, metrics in sorted(snapshot.graph_metrics.items()):
            graph_prefix = f"{prefix}__graph_{_threshold_token(threshold)}"
            row[f"{graph_prefix}__edge_count"] = float(metrics.edge_count)
            row[f"{graph_prefix}__edge_density"] = metrics.edge_density
            row[f"{graph_prefix}__mean_degree"] = metrics.mean_degree
            row[f"{graph_prefix}__degree_std"] = metrics.degree_std
            row[f"{graph_prefix}__max_degree"] = float(metrics.max_degree)
            row[f"{graph_prefix}__isolated_nodes"] = float(metrics.isolated_nodes)
            row[f"{graph_prefix}__degree_centralization"] = (
                metrics.degree_centralization
            )
            row[f"{graph_prefix}__global_clustering"] = metrics.global_clustering

    if not all(math.isfinite(value) for value in row.values()):
        raise OrcaValidationError("spectral feature row contains non-finite data")
    return row


def spectral_feature_registry_hash(bundle: OrcaSpectralBundle) -> str:
    """Hash feature names and, for explicit profiles, the exact universe."""

    names = list(spectral_feature_row(bundle))
    if (bundle.profile_id is None) != (bundle.profile_contract_hash is None):
        raise OrcaValidationError("bundle universe profile metadata is incomplete")
    payload: Any = names
    if bundle.profile_id is not None:
        payload = {
            "feature_names": names,
            "ordered_universe": list(bundle.symbols),
            "profile_contract_hash": bundle.profile_contract_hash,
            "profile_id": bundle.profile_id,
        }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _binary_arrays(
    y_true: Sequence[Any],
    y_score: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1:
        raise OrcaValidationError("AUC inputs must be one-dimensional")
    if len(labels) == 0 or len(labels) != len(scores):
        raise OrcaValidationError(
            "AUC labels and scores must have one equal non-zero length"
        )
    try:
        numeric_labels = labels.astype(float)
    except (TypeError, ValueError) as exc:
        raise OrcaValidationError("AUC labels must be binary") from exc
    if not np.isfinite(numeric_labels).all() or not np.isfinite(scores).all():
        raise OrcaValidationError("AUC inputs must be finite")
    if not np.isin(numeric_labels, (0.0, 1.0)).all():
        raise OrcaValidationError("AUC labels must contain only 0 and 1")
    return numeric_labels.astype(np.int8), scores


def tie_aware_binary_auc(
    y_true: Sequence[Any],
    y_score: Sequence[Any],
) -> Optional[float]:
    """Exact Mann-Whitney ROC AUC using average ranks for score ties."""

    labels, scores = _binary_arrays(y_true, y_score)
    positive_count = int(labels.sum())
    negative_count = int(len(labels) - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(ranks[labels == 1].sum())
    statistic = positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    auc = statistic / (positive_count * negative_count)
    return float(min(1.0, max(0.0, auc)))


def balanced_crisis_detection_auc(
    y_rally: Sequence[Any],
    p_rally: Sequence[Any],
    y_crash: Sequence[Any],
    p_crash: Sequence[Any],
) -> BcdAucResult:
    """Compute the exact ORCA BCD-AUC or an explicit UNDEFINED result."""

    rally_labels, rally_scores = _binary_arrays(y_rally, p_rally)
    crash_labels, crash_scores = _binary_arrays(y_crash, p_crash)
    if len(rally_labels) != len(crash_labels):
        raise OrcaValidationError(
            "rally and crash OOS populations must be date-aligned"
        )

    rally_auc = tie_aware_binary_auc(rally_labels, rally_scores)
    crash_auc = tie_aware_binary_auc(crash_labels, crash_scores)
    rally_positive = int(rally_labels.sum())
    crash_positive = int(crash_labels.sum())
    missing: list[str] = []
    if rally_auc is None:
        missing.append("rally")
    if crash_auc is None:
        missing.append("crash")
    if missing:
        return BcdAucResult(
            schema_version=BCD_AUC_SCHEMA,
            status="UNDEFINED",
            bcd_auc=None,
            rally_auc=rally_auc,
            crash_auc=crash_auc,
            observation_count=len(rally_labels),
            rally_positive_count=rally_positive,
            rally_negative_count=len(rally_labels) - rally_positive,
            crash_positive_count=crash_positive,
            crash_negative_count=len(crash_labels) - crash_positive,
            reason=f"missing_class:{','.join(missing)}",
        )

    assert rally_auc is not None and crash_auc is not None
    return BcdAucResult(
        schema_version=BCD_AUC_SCHEMA,
        status="DEFINED",
        bcd_auc=float(math.sqrt(rally_auc * crash_auc)),
        rally_auc=rally_auc,
        crash_auc=crash_auc,
        observation_count=len(rally_labels),
        rally_positive_count=rally_positive,
        rally_negative_count=len(rally_labels) - rally_positive,
        crash_positive_count=crash_positive,
        crash_negative_count=len(crash_labels) - crash_positive,
        reason=None,
    )


__all__ = [
    "BCD_AUC_SCHEMA",
    "ORCA_SPECTRAL_SCHEMA",
    "ORCA_UNIVERSE_PROFILE_SCHEMA",
    "ORCA_FX_GOLD_4_PROFILE",
    "ORCA_FX_GOLD_4_SPECTRAL_CONFIG",
    "ORCA_UNIVERSE_PROFILES",
    "BcdAucResult",
    "CorrelationSnapshot",
    "GraphMetrics",
    "OrcaInsufficientDataError",
    "OrcaSpectralBundle",
    "OrcaSpectralConfig",
    "OrcaSpectralError",
    "OrcaUniverseProfile",
    "resolve_universe_profile",
    "spectral_config_for_profile",
    "OrcaValidationError",
    "SignedEdge",
    "SpectralSnapshot",
    "balanced_crisis_detection_auc",
    "build_correlation_snapshots",
    "build_orca_spectral_snapshots",
    "simple_returns",
    "spectral_feature_registry_hash",
    "spectral_feature_row",
    "tie_aware_binary_auc",
    "validate_d1_price_panel",
]
