"""FxPro/Quantower M15 cluster-rejection proxy.

The FxPro connector exposes reconstructed Bid/Ask tick history to Quantower,
not an exchange tape.  ``BuyVolume`` and ``SellVolume`` are therefore treated
as TickDirection classifications.  They never prove executions or aggressor
polarity and this module deliberately does not call the signal Absorption.

The detector is fail-closed.  A valid signal needs one checksummed, finalized
M15 event with complete PriceLevels, explicit availability semantics and a
matching FxPro candle.  Missing cluster data is never synthesized from OHLCV
or MT5 tick volume.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Sequence

import pandas as pd


CLUSTER_EVENT_SCHEMA_VERSION = 1
CLUSTER_TIMEFRAME = "15m"
CLUSTER_SOURCE = "quantower_fxpro_tick_cluster"
CLUSTER_VENUE = "FxPro"
CLUSTER_MARKET_TYPE = "otc_reconstructed_bidask_ticks"
CLUSTER_DATA_KIND = "tick_direction_cluster_proxy_no_execution_proof"
CLUSTER_CLASSIFICATION = "quantower_tickdirection_buy_sell"
CLUSTER_AVAILABILITY_MODES = frozenset(
    {"forward_observed", "research_assumption"}
)
ClusterSide = Literal["LONG", "SHORT"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ClusterRejectionThresholds:
    """Initial auditable thresholds; later fits must stay inside train windows."""

    edge_fraction: float = 0.20
    min_classified_volume: float = 20.0
    min_classification_ratio: float = 0.80
    min_edge_volume_share: float = 0.12
    min_edge_imbalance_ratio: float = 1.50
    min_rejection_fraction: float = 0.60
    min_wick_fraction: float = 0.20
    min_cluster_score: float = 0.60
    candle_match_tolerance_ticks: int = 2

    def __post_init__(self) -> None:
        finite = (
            self.edge_fraction,
            self.min_classified_volume,
            self.min_classification_ratio,
            self.min_edge_volume_share,
            self.min_edge_imbalance_ratio,
            self.min_rejection_fraction,
            self.min_wick_fraction,
            self.min_cluster_score,
        )
        if not all(math.isfinite(float(value)) for value in finite):
            raise ValueError("cluster-rejection thresholds must be finite")
        if not 0.0 < self.edge_fraction <= 0.5:
            raise ValueError("edge_fraction must be in (0, 0.5]")
        if self.min_classified_volume <= 0.0:
            raise ValueError("min_classified_volume must be positive")
        for name in (
            "min_classification_ratio",
            "min_edge_volume_share",
            "min_rejection_fraction",
            "min_wick_fraction",
            "min_cluster_score",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.min_edge_imbalance_ratio <= 1.0:
            raise ValueError("min_edge_imbalance_ratio must exceed 1")
        if (
            isinstance(self.candle_match_tolerance_ticks, bool)
            or int(self.candle_match_tolerance_ticks) < 0
        ):
            raise ValueError(
                "candle_match_tolerance_ticks must be a nonnegative integer"
            )


@dataclass(frozen=True)
class ClusterRejectionSignal:
    side: ClusterSide
    bar_open_time: str
    bar_close_time: str
    source: str
    source_bar_hash: str
    availability_mode: str
    classified_volume: float
    classification_ratio: float
    edge_volume: float
    edge_volume_share: float
    edge_imbalance_ratio: float
    edge_pressure_share: float
    rejection_fraction: float
    wick_fraction: float
    cluster_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "bar_open_time": self.bar_open_time,
            "bar_close_time": self.bar_close_time,
            "source": self.source,
            "source_bar_hash": self.source_bar_hash,
            "availability_mode": self.availability_mode,
            "classified_volume": self.classified_volume,
            "classification_ratio": self.classification_ratio,
            "edge_volume": self.edge_volume,
            "edge_volume_share": self.edge_volume_share,
            "edge_imbalance_ratio": self.edge_imbalance_ratio,
            "edge_pressure_share": self.edge_pressure_share,
            "rejection_fraction": self.rejection_fraction,
            "wick_fraction": self.wick_fraction,
            "cluster_score": self.cluster_score,
            "execution_proof_claimed": False,
            "aggressor_polarity_claimed": False,
        }


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def seal_cluster_event(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical event with a checksum over every other field."""

    event = dict(payload)
    event.pop("checksum", None)
    event["checksum"] = canonical_hash(event)
    return event


def _utc_timestamp(value: Any) -> Optional[pd.Timestamp]:
    try:
        result = pd.Timestamp(value)
    except Exception:
        return None
    if (
        pd.isna(result)
        or result.tzinfo is None
        or result.utcoffset() != pd.Timedelta(0)
    ):
        return None
    return result.tz_convert("UTC")


def _finite_number(
    value: Any,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    if positive and result <= 0.0:
        return None
    if nonnegative and result < 0.0:
        return None
    return result


def _integer(value: Any, *, nonnegative: bool = False) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    if result != value and not (
        isinstance(value, str) and str(result) == value.strip()
    ):
        return None
    if nonnegative and result < 0:
        return None
    return result


def _normalized_volume_fields(
    value: Any,
) -> Optional[dict[str, float | int]]:
    if not isinstance(value, Mapping):
        return None
    numeric: dict[str, float | int] = {}
    for field in ("volume", "buy_volume", "sell_volume", "delta"):
        parsed = _finite_number(value.get(field), nonnegative=(field != "delta"))
        if parsed is None:
            return None
        numeric[field] = parsed
    for field in ("trades", "buy_trades", "sell_trades"):
        parsed = _integer(value.get(field), nonnegative=True)
        if parsed is None:
            return None
        numeric[field] = parsed
    if (
        float(numeric["buy_volume"]) + float(numeric["sell_volume"])
        > float(numeric["volume"]) + 1e-9
        or int(numeric["buy_trades"]) + int(numeric["sell_trades"])
        > int(numeric["trades"])
        or abs(
            float(numeric["delta"])
            - (
                float(numeric["buy_volume"])
                - float(numeric["sell_volume"])
            )
        )
        > 1e-9
    ):
        return None
    return numeric


def validate_cluster_event(event: Any) -> Optional[dict[str, Any]]:
    """Normalize and validate one immutable cluster-proxy event."""

    if not isinstance(event, Mapping):
        return None
    raw = dict(event)
    checksum = raw.pop("checksum", None)
    if (
        not isinstance(checksum, str)
        or not _SHA256_RE.fullmatch(checksum)
        or canonical_hash(raw) != checksum
    ):
        return None
    expected = {
        "schema_version": CLUSTER_EVENT_SCHEMA_VERSION,
        "timeframe": CLUSTER_TIMEFRAME,
        "source": CLUSTER_SOURCE,
        "venue": CLUSTER_VENUE,
        "market_type": CLUSTER_MARKET_TYPE,
        "data_kind": CLUSTER_DATA_KIND,
        "classification": CLUSTER_CLASSIFICATION,
        "execution_proof": False,
        "aggressor_polarity_proven": False,
        "finalized": True,
    }
    for field, value in expected.items():
        if raw.get(field) != value:
            return None

    symbol = str(raw.get("symbol") or "").strip().upper()
    if not re.fullmatch(r"[A-Z0-9._-]{3,32}", symbol):
        return None
    tick_size = _finite_number(raw.get("tick_size"), positive=True)
    if tick_size is None:
        return None
    bar_open = _utc_timestamp(raw.get("bar_open"))
    bar_close = _utc_timestamp(raw.get("bar_close"))
    available_at = _utc_timestamp(raw.get("available_at"))
    if (
        bar_open is None
        or bar_close is None
        or available_at is None
        or bar_open.minute % 15
        or bar_open.second
        or bar_open.microsecond
        or bar_open.nanosecond
        or bar_close - bar_open != pd.Timedelta(minutes=15)
        or available_at < bar_close
    ):
        return None
    availability_mode = str(raw.get("availability_mode") or "")
    if availability_mode not in CLUSTER_AVAILABILITY_MODES:
        return None
    if not isinstance(raw.get("availability_evidence"), str) or not str(
        raw["availability_evidence"]
    ).strip():
        return None
    source_bar_hash = raw.get("source_bar_hash")
    source_manifest_sha256 = raw.get("source_manifest_sha256")
    if (
        not isinstance(source_bar_hash, str)
        or not _SHA256_RE.fullmatch(source_bar_hash)
        or not isinstance(source_manifest_sha256, str)
        or not _SHA256_RE.fullmatch(source_manifest_sha256)
    ):
        return None

    ohlc_raw = raw.get("ohlc")
    if not isinstance(ohlc_raw, Mapping):
        return None
    ohlc: dict[str, float] = {}
    for field in ("open", "high", "low", "close"):
        parsed = _finite_number(ohlc_raw.get(field), positive=True)
        if parsed is None:
            return None
        ohlc[field] = parsed
    if (
        ohlc["high"] <= ohlc["low"]
        or not ohlc["low"] <= ohlc["open"] <= ohlc["high"]
        or not ohlc["low"] <= ohlc["close"] <= ohlc["high"]
    ):
        return None

    total = _normalized_volume_fields(raw.get("total"))
    levels_raw = raw.get("price_levels")
    if total is None or not isinstance(levels_raw, Sequence) or not levels_raw:
        return None
    levels: list[dict[str, float | int]] = []
    previous_ticks: Optional[int] = None
    sums = {
        "volume": 0.0,
        "trades": 0,
        "buy_volume": 0.0,
        "sell_volume": 0.0,
        "buy_trades": 0,
        "sell_trades": 0,
        "delta": 0.0,
    }
    for raw_level in levels_raw:
        if not isinstance(raw_level, Mapping):
            return None
        price = _finite_number(raw_level.get("price"), positive=True)
        price_ticks = _integer(raw_level.get("price_ticks"))
        fields = _normalized_volume_fields(raw_level)
        if price is None or price_ticks is None or fields is None:
            return None
        if previous_ticks is not None and price_ticks <= previous_ticks:
            return None
        if abs(price - price_ticks * tick_size) > tick_size * 1e-6:
            return None
        previous_ticks = price_ticks
        normalized_level = {
            "price": price,
            "price_ticks": price_ticks,
            **fields,
        }
        levels.append(normalized_level)
        for field in sums:
            sums[field] += fields[field]
    for field in sums:
        if abs(float(sums[field]) - float(total[field])) > 1e-8:
            return None

    return {
        **raw,
        "symbol": symbol,
        "tick_size": tick_size,
        "bar_open": bar_open,
        "bar_close": bar_close,
        "available_at": available_at,
        "ohlc": ohlc,
        "total": total,
        "price_levels": levels,
        "checksum": checksum,
    }


def detect_fxpro_cluster_rejection(
    *,
    candle: Mapping[str, Any],
    candle_open_time: Any,
    event: Any,
    symbol: str,
    side: ClusterSide,
    decision_time: Any,
    thresholds: ClusterRejectionThresholds = ClusterRejectionThresholds(),
    allow_research_assumption: bool = False,
) -> Optional[ClusterRejectionSignal]:
    """Detect an M15 cluster rejection without claiming exchange absorption."""

    if side not in {"LONG", "SHORT"} or not isinstance(candle, Mapping):
        return None
    normalized = validate_cluster_event(event)
    if normalized is None:
        return None
    if (
        normalized["availability_mode"] == "research_assumption"
        and not allow_research_assumption
    ):
        return None
    expected_symbol = str(symbol or "").strip().upper()
    if not expected_symbol or normalized["symbol"] != expected_symbol:
        return None
    candle_open = _utc_timestamp(candle_open_time)
    decision = _utc_timestamp(decision_time)
    if (
        candle_open is None
        or decision is None
        or normalized["bar_open"] != candle_open
        or normalized["bar_close"] > decision
        or normalized["available_at"] > decision
    ):
        return None

    candle_prices: dict[str, float] = {}
    for field in ("open", "high", "low", "close"):
        parsed = _finite_number(candle.get(field), positive=True)
        if parsed is None:
            return None
        candle_prices[field] = parsed
    candle_range = candle_prices["high"] - candle_prices["low"]
    if candle_range <= 0.0:
        return None
    tolerance = (
        float(normalized["tick_size"])
        * int(thresholds.candle_match_tolerance_ticks)
    )
    if any(
        abs(candle_prices[field] - float(normalized["ohlc"][field]))
        > tolerance + 1e-12
        for field in candle_prices
    ):
        return None

    total = normalized["total"]
    classified_volume = float(total["buy_volume"]) + float(
        total["sell_volume"]
    )
    total_volume = float(total["volume"])
    if total_volume <= 0.0 or classified_volume < thresholds.min_classified_volume:
        return None
    classification_ratio = classified_volume / total_volume
    if classification_ratio < thresholds.min_classification_ratio:
        return None

    levels = normalized["price_levels"]
    low_ticks = int(levels[0]["price_ticks"])
    high_ticks = int(levels[-1]["price_ticks"])
    span_ticks = high_ticks - low_ticks
    if span_ticks <= 0:
        return None
    edge_width = max(1, int(math.ceil(span_ticks * thresholds.edge_fraction)))
    if side == "LONG":
        edge = [
            level
            for level in levels
            if int(level["price_ticks"]) <= low_ticks + edge_width
        ]
        dominant = sum(float(level["sell_volume"]) for level in edge)
        opposite = sum(float(level["buy_volume"]) for level in edge)
        rejection_fraction = (
            candle_prices["close"] - candle_prices["low"]
        ) / candle_range
        wick_fraction = (
            min(candle_prices["open"], candle_prices["close"])
            - candle_prices["low"]
        ) / candle_range
    else:
        edge = [
            level
            for level in levels
            if int(level["price_ticks"]) >= high_ticks - edge_width
        ]
        dominant = sum(float(level["buy_volume"]) for level in edge)
        opposite = sum(float(level["sell_volume"]) for level in edge)
        rejection_fraction = (
            candle_prices["high"] - candle_prices["close"]
        ) / candle_range
        wick_fraction = (
            candle_prices["high"]
            - max(candle_prices["open"], candle_prices["close"])
        ) / candle_range

    edge_volume = dominant + opposite
    if dominant <= 0.0 or edge_volume <= 0.0:
        return None
    edge_volume_share = edge_volume / classified_volume
    edge_imbalance_ratio = (
        dominant / opposite if opposite > 0.0 else float("inf")
    )
    edge_pressure_share = dominant / edge_volume
    if (
        edge_volume_share < thresholds.min_edge_volume_share
        or edge_imbalance_ratio < thresholds.min_edge_imbalance_ratio
        or rejection_fraction < thresholds.min_rejection_fraction
        or wick_fraction < thresholds.min_wick_fraction
    ):
        return None

    imbalance_strength = min(
        1.0,
        edge_pressure_share
        / (
            thresholds.min_edge_imbalance_ratio
            / (thresholds.min_edge_imbalance_ratio + 1.0)
        ),
    )
    concentration_strength = min(
        1.0,
        edge_volume_share / max(thresholds.min_edge_volume_share * 2.0, 1e-9),
    )
    cluster_score = (
        0.35 * imbalance_strength
        + 0.25 * concentration_strength
        + 0.25 * min(1.0, rejection_fraction)
        + 0.15 * min(1.0, wick_fraction)
    )
    if cluster_score < thresholds.min_cluster_score:
        return None

    return ClusterRejectionSignal(
        side=side,
        bar_open_time=normalized["bar_open"].isoformat(),
        bar_close_time=normalized["bar_close"].isoformat(),
        source=str(normalized["source"]),
        source_bar_hash=str(normalized["checksum"]),
        availability_mode=str(normalized["availability_mode"]),
        classified_volume=classified_volume,
        classification_ratio=classification_ratio,
        edge_volume=edge_volume,
        edge_volume_share=edge_volume_share,
        edge_imbalance_ratio=edge_imbalance_ratio,
        edge_pressure_share=edge_pressure_share,
        rejection_fraction=rejection_fraction,
        wick_fraction=wick_fraction,
        cluster_score=cluster_score,
    )


__all__ = [
    "CLUSTER_AVAILABILITY_MODES",
    "CLUSTER_CLASSIFICATION",
    "CLUSTER_DATA_KIND",
    "CLUSTER_EVENT_SCHEMA_VERSION",
    "CLUSTER_MARKET_TYPE",
    "CLUSTER_SOURCE",
    "CLUSTER_TIMEFRAME",
    "CLUSTER_VENUE",
    "ClusterRejectionSignal",
    "ClusterRejectionThresholds",
    "canonical_hash",
    "detect_fxpro_cluster_rejection",
    "seal_cluster_event",
    "validate_cluster_event",
]
