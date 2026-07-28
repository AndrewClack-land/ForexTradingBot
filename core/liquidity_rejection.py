"""Fail-closed FxPro OTC depth-liquidity rejection detection.

This module deliberately does *not* call the signal ``Absorption``.  FxPro
market depth is an aggregated executable-liquidity view supplied by the
broker; it is not a centralized exchange tape and does not prove that a quote
change was an execution rather than a cancellation.

The detector therefore consumes one finalized M15 summary of FxPro DOM quote
behaviour and looks for:

* directional quote pressure into a candle extreme;
* replenishment of the protective side of the broker book;
* poor price progress despite that pressure; and
* a causal close rejecting the extreme.

Missing, delayed, partial, malformed, non-FxPro, or checksum-invalid events
always return ``None``.  OHLCV and tick volume are never substituted.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional

import pandas as pd


LIQUIDITY_EVENT_SCHEMA_VERSION = 1
LIQUIDITY_TIMEFRAME = "15m"
LIQUIDITY_SOURCE = "fxpro_mt5_market_book"
LIQUIDITY_VENUE = "FxPro"
LIQUIDITY_MARKET_TYPE = "otc_aggregated_liquidity"
LIQUIDITY_DATA_KIND = "depth_quotes_no_execution_proof"
LiquiditySide = Literal["LONG", "SHORT"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NUMERIC_EVENT_FIELDS = (
    "coverage_ratio",
    "sample_count",
    "changed_snapshots",
    "max_gap_ms",
    "up_ticks",
    "down_ticks",
    "up_distance",
    "down_distance",
    "bid_depleted_volume",
    "bid_replenished_volume",
    "ask_depleted_volume",
    "ask_replenished_volume",
    "start_mid",
    "high_mid",
    "low_mid",
    "end_mid",
    "mean_spread",
    "max_spread",
    "min_bid_levels",
    "min_ask_levels",
)
_INTEGER_EVENT_FIELDS = (
    "sample_count",
    "changed_snapshots",
    "up_ticks",
    "down_ticks",
    "min_bid_levels",
    "min_ask_levels",
)


@dataclass(frozen=True)
class LiquidityRejectionThresholds:
    """Auditable initial thresholds; fit only inside WFO train windows later."""

    min_abs_quote_pressure: float = 0.15
    min_replenishment_ratio: float = 0.50
    min_replenishment_share: float = 0.55
    min_rejection_fraction: float = 0.60
    max_price_response_efficiency: float = 0.35
    min_coverage_ratio: float = 0.98
    max_gap_seconds: float = 5.0
    min_changed_snapshots: int = 20
    min_book_levels: int = 2
    max_mean_spread_bps: float = 5.0

    def __post_init__(self) -> None:
        finite = (
            self.min_abs_quote_pressure,
            self.min_replenishment_ratio,
            self.min_replenishment_share,
            self.min_rejection_fraction,
            self.max_price_response_efficiency,
            self.min_coverage_ratio,
            self.max_gap_seconds,
            self.max_mean_spread_bps,
        )
        if not all(math.isfinite(float(value)) for value in finite):
            raise ValueError("liquidity-rejection thresholds must be finite")
        if not 0.0 <= self.min_abs_quote_pressure <= 1.0:
            raise ValueError("min_abs_quote_pressure must be in [0, 1]")
        if self.min_replenishment_ratio < 0.0:
            raise ValueError("min_replenishment_ratio cannot be negative")
        if not 0.0 <= self.min_replenishment_share <= 1.0:
            raise ValueError("min_replenishment_share must be in [0, 1]")
        if not 0.5 < self.min_rejection_fraction <= 1.0:
            raise ValueError("min_rejection_fraction must be in (0.5, 1]")
        if not 0.0 <= self.max_price_response_efficiency <= 1.0:
            raise ValueError("max_price_response_efficiency must be in [0, 1]")
        if not 0.0 < self.min_coverage_ratio <= 1.0:
            raise ValueError("min_coverage_ratio must be in (0, 1]")
        if self.max_gap_seconds <= 0.0:
            raise ValueError("max_gap_seconds must be positive")
        if (
            isinstance(self.min_changed_snapshots, bool)
            or int(self.min_changed_snapshots) < 1
        ):
            raise ValueError("min_changed_snapshots must be a positive integer")
        if (
            isinstance(self.min_book_levels, bool)
            or int(self.min_book_levels) < 1
        ):
            raise ValueError("min_book_levels must be a positive integer")
        if self.max_mean_spread_bps <= 0.0:
            raise ValueError("max_mean_spread_bps must be positive")


@dataclass(frozen=True)
class LiquidityRejectionSignal:
    side: LiquiditySide
    bar_open_time: str
    bar_close_time: str
    source: str
    source_bar_hash: str
    quote_pressure: float
    replenishment_ratio: float
    replenishment_share: float
    rejection_fraction: float
    price_response_efficiency: float
    mean_spread_bps: float
    liquidity_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "bar_open_time": self.bar_open_time,
            "bar_close_time": self.bar_close_time,
            "source": self.source,
            "source_bar_hash": self.source_bar_hash,
            "quote_pressure": self.quote_pressure,
            "replenishment_ratio": self.replenishment_ratio,
            "replenishment_share": self.replenishment_share,
            "rejection_fraction": self.rejection_fraction,
            "price_response_efficiency": self.price_response_efficiency,
            "mean_spread_bps": self.mean_spread_bps,
            "liquidity_score": self.liquidity_score,
        }


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _event_value(event: Any, name: str) -> Any:
    if isinstance(event, Mapping):
        return event.get(name)
    return getattr(event, name, None)


def _utc_timestamp(value: Any) -> Optional[pd.Timestamp]:
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return None
    if (
        pd.isna(timestamp)
        or timestamp.tzinfo is None
        or timestamp.utcoffset() != pd.Timedelta(0)
    ):
        return None
    return timestamp.tz_convert("UTC")


def _finite_number(value: Any, *, nonnegative: bool = False) -> Optional[float]:
    if isinstance(value, (bool, str, bytes)) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    if nonnegative and number < 0.0:
        return None
    return number


def liquidity_event_checksum_payload(event: Any) -> Optional[dict[str, Any]]:
    """Return the normalized payload covered by an event checksum."""

    schema_version = _event_value(event, "schema_version")
    revision = _event_value(event, "revision")
    finalized = _event_value(event, "finalized")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or not isinstance(revision, int)
        or isinstance(revision, bool)
        or finalized is not True
    ):
        return None

    text_fields: dict[str, str] = {}
    for field in (
        "timeframe",
        "source",
        "venue",
        "market_type",
        "data_kind",
        "source_symbol",
        "symbol",
    ):
        value = _event_value(event, field)
        if not isinstance(value, str) or not value.strip():
            return None
        text_fields[field] = value.strip()
    text_fields["source_symbol"] = text_fields["source_symbol"].upper()
    text_fields["symbol"] = text_fields["symbol"].upper()

    timestamps: dict[str, str] = {}
    for field in ("bar_open", "bar_close", "available_at"):
        value = _utc_timestamp(_event_value(event, field))
        if value is None:
            return None
        timestamps[field] = value.isoformat()

    numbers: dict[str, float | int] = {}
    for field in _NUMERIC_EVENT_FIELDS:
        raw = _event_value(event, field)
        if field in _INTEGER_EVENT_FIELDS:
            if (
                not isinstance(raw, int)
                or isinstance(raw, bool)
                or raw < 0
            ):
                return None
            numbers[field] = int(raw)
            continue
        value = _finite_number(raw, nonnegative=True)
        if value is None:
            return None
        numbers[field] = float(value)

    return {
        "schema_version": schema_version,
        "revision": revision,
        **text_fields,
        **timestamps,
        "finalized": True,
        **numbers,
    }


def seal_liquidity_event(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and checksum a recorder-produced event payload."""

    normalized = liquidity_event_checksum_payload(payload)
    if normalized is None:
        raise ValueError("invalid FxPro liquidity event payload")
    return {**normalized, "checksum": _canonical_hash(normalized)}


def validate_liquidity_event(event: Any) -> Optional[dict[str, Any]]:
    """Return normalized event data, or ``None`` on any contract violation."""

    normalized = liquidity_event_checksum_payload(event)
    checksum = _event_value(event, "checksum")
    if normalized is None:
        return None
    if not isinstance(checksum, str) or not _SHA256_RE.fullmatch(checksum):
        return None
    if _canonical_hash(normalized) != checksum:
        return None
    if normalized["schema_version"] != LIQUIDITY_EVENT_SCHEMA_VERSION:
        return None
    if normalized["revision"] != 0:
        return None
    if normalized["timeframe"] != LIQUIDITY_TIMEFRAME:
        return None
    if normalized["source"] != LIQUIDITY_SOURCE:
        return None
    if normalized["venue"] != LIQUIDITY_VENUE:
        return None
    if normalized["market_type"] != LIQUIDITY_MARKET_TYPE:
        return None
    if normalized["data_kind"] != LIQUIDITY_DATA_KIND:
        return None
    if normalized["source_symbol"] != normalized["symbol"]:
        return None

    bar_open = pd.Timestamp(normalized["bar_open"])
    bar_close = pd.Timestamp(normalized["bar_close"])
    available_at = pd.Timestamp(normalized["available_at"])
    if (
        bar_open.minute % 15
        or bar_open.second
        or bar_open.microsecond
        or bar_open.nanosecond
        or bar_close - bar_open != pd.Timedelta(minutes=15)
        or available_at < bar_close
    ):
        return None
    if not 0.0 <= float(normalized["coverage_ratio"]) <= 1.0:
        return None
    sample_count = int(normalized["sample_count"])
    changed_snapshots = int(normalized["changed_snapshots"])
    directional_ticks = int(normalized["up_ticks"]) + int(
        normalized["down_ticks"]
    )
    if (
        sample_count < 1
        or changed_snapshots < 1
        or changed_snapshots > sample_count
        or directional_ticks > max(0, changed_snapshots - 1)
        or float(normalized["mean_spread"])
        > float(normalized["max_spread"])
    ):
        return None
    if float(normalized["low_mid"]) > float(normalized["high_mid"]):
        return None
    if not (
        float(normalized["low_mid"])
        <= float(normalized["start_mid"])
        <= float(normalized["high_mid"])
    ):
        return None
    if not (
        float(normalized["low_mid"])
        <= float(normalized["end_mid"])
        <= float(normalized["high_mid"])
    ):
        return None
    travelled = float(normalized["up_distance"]) + float(
        normalized["down_distance"]
    )
    if abs(
        float(normalized["end_mid"]) - float(normalized["start_mid"])
    ) > travelled + 1e-12:
        return None
    return {**normalized, "checksum": checksum}


def detect_fxpro_liquidity_rejection(
    *,
    candle: Mapping[str, Any],
    candle_open_time: Any,
    event: Any,
    symbol: str,
    side: LiquiditySide,
    decision_time: Any,
    thresholds: LiquidityRejectionThresholds = LiquidityRejectionThresholds(),
) -> Optional[LiquidityRejectionSignal]:
    """Detect a causal FxPro broker-liquidity rejection on one closed M15 bar."""

    if side not in {"LONG", "SHORT"} or not isinstance(candle, Mapping):
        return None
    normalized = validate_liquidity_event(event)
    if normalized is None:
        return None

    expected_symbol = str(symbol or "").strip().upper()
    if not expected_symbol or normalized["symbol"] != expected_symbol:
        return None

    candle_open = _utc_timestamp(candle_open_time)
    decision = _utc_timestamp(decision_time)
    if candle_open is None or decision is None:
        return None
    event_open = pd.Timestamp(normalized["bar_open"])
    event_close = pd.Timestamp(normalized["bar_close"])
    available_at = pd.Timestamp(normalized["available_at"])
    if (
        event_open != candle_open
        or event_close > decision
        or available_at > decision
    ):
        return None

    prices: dict[str, float] = {}
    for field in ("high", "low", "close"):
        value = _finite_number(candle.get(field))
        if value is None or value <= 0.0:
            return None
        prices[field] = value
    candle_range = prices["high"] - prices["low"]
    if (
        candle_range <= 0.0
        or not prices["low"] <= prices["close"] <= prices["high"]
    ):
        return None
    # MT5 candles are normally Bid while DOM summaries use the quote midpoint.
    # One maximum observed spread is a conservative alignment tolerance.
    feed_tolerance = float(normalized["max_spread"])
    if (
        float(normalized["low_mid"]) < prices["low"] - feed_tolerance
        or float(normalized["high_mid"])
        > prices["high"] + feed_tolerance
    ):
        return None

    coverage = float(normalized["coverage_ratio"])
    changed = int(normalized["changed_snapshots"])
    max_gap_seconds = float(normalized["max_gap_ms"]) / 1000.0
    if (
        coverage < thresholds.min_coverage_ratio
        or changed < thresholds.min_changed_snapshots
        or max_gap_seconds > thresholds.max_gap_seconds
        or int(normalized["min_bid_levels"]) < thresholds.min_book_levels
        or int(normalized["min_ask_levels"]) < thresholds.min_book_levels
    ):
        return None

    up_ticks = int(normalized["up_ticks"])
    down_ticks = int(normalized["down_ticks"])
    directional_ticks = up_ticks + down_ticks
    if directional_ticks <= 0:
        return None
    quote_pressure = (up_ticks - down_ticks) / directional_ticks

    up_distance = float(normalized["up_distance"])
    down_distance = float(normalized["down_distance"])
    total_distance = up_distance + down_distance
    if total_distance <= 0.0:
        return None
    price_response_efficiency = abs(
        float(normalized["end_mid"]) - float(normalized["start_mid"])
    ) / total_distance

    if side == "LONG":
        protective_replenished = float(normalized["bid_replenished_volume"])
        protective_depleted = float(normalized["bid_depleted_volume"])
        opposite_replenished = float(normalized["ask_replenished_volume"])
        rejection_fraction = (
            prices["close"] - prices["low"]
        ) / candle_range
        pressure_ok = quote_pressure <= -thresholds.min_abs_quote_pressure
    else:
        protective_replenished = float(normalized["ask_replenished_volume"])
        protective_depleted = float(normalized["ask_depleted_volume"])
        opposite_replenished = float(normalized["bid_replenished_volume"])
        rejection_fraction = (
            prices["high"] - prices["close"]
        ) / candle_range
        pressure_ok = quote_pressure >= thresholds.min_abs_quote_pressure

    if protective_depleted <= 0.0 or protective_replenished <= 0.0:
        return None
    replenishment_ratio = protective_replenished / protective_depleted
    replenishment_total = protective_replenished + opposite_replenished
    if replenishment_total <= 0.0:
        return None
    replenishment_share = protective_replenished / replenishment_total

    mean_mid = (
        float(normalized["start_mid"]) + float(normalized["end_mid"])
    ) / 2.0
    if mean_mid <= 0.0:
        return None
    mean_spread_bps = float(normalized["mean_spread"]) / mean_mid * 10_000.0

    if (
        not pressure_ok
        or replenishment_ratio < thresholds.min_replenishment_ratio
        or replenishment_share < thresholds.min_replenishment_share
        or rejection_fraction < thresholds.min_rejection_fraction
        or price_response_efficiency
        > thresholds.max_price_response_efficiency
        or mean_spread_bps > thresholds.max_mean_spread_bps
    ):
        return None

    pressure_strength = min(1.0, abs(quote_pressure))
    replenishment_strength = min(1.0, replenishment_ratio)
    inefficiency_strength = max(0.0, 1.0 - price_response_efficiency)
    liquidity_score = (
        0.30 * pressure_strength
        + 0.30 * replenishment_strength
        + 0.25 * rejection_fraction
        + 0.15 * inefficiency_strength
    )
    return LiquidityRejectionSignal(
        side=side,
        bar_open_time=event_open.isoformat(),
        bar_close_time=event_close.isoformat(),
        source=str(normalized["source"]),
        source_bar_hash=str(normalized["checksum"]),
        quote_pressure=float(quote_pressure),
        replenishment_ratio=float(replenishment_ratio),
        replenishment_share=float(replenishment_share),
        rejection_fraction=float(rejection_fraction),
        price_response_efficiency=float(price_response_efficiency),
        mean_spread_bps=float(mean_spread_bps),
        liquidity_score=float(liquidity_score),
    )
