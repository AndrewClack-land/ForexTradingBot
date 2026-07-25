"""Fail-closed 15-minute footprint absorption detection.

The detector is deliberately source-independent.  A feed adapter must provide
one finalized, provenance-bearing footprint event for the same closed M15 bar
as the price candle.  Missing or malformed order-flow data never falls back to
OHLC volume or tick-count approximations.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional

import pandas as pd


ABSORPTION_EVENT_SCHEMA_VERSION = 1
REQUIRED_POLARITY = (
    "buy_volume=aggressor_at_ask;sell_volume=aggressor_at_bid"
)
AbsorptionSide = Literal["LONG", "SHORT"]

_VOLUME_FIELDS = (
    "high_buy_volume",
    "high_sell_volume",
    "low_buy_volume",
    "low_sell_volume",
)


@dataclass(frozen=True)
class AbsorptionThresholds:
    """Thresholds matching the audited footprint absorption semantics."""

    min_imbalance_ratio: float = 2.0
    min_edge_volume: float = 80.0

    def __post_init__(self) -> None:
        values = (
            self.min_imbalance_ratio,
            self.min_edge_volume,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("absorption thresholds must be finite")
        if self.min_imbalance_ratio <= 1.0:
            raise ValueError("min_imbalance_ratio must exceed 1")
        if self.min_edge_volume < 0.0:
            raise ValueError("min_edge_volume cannot be negative")


@dataclass(frozen=True)
class AbsorptionSignal:
    side: AbsorptionSide
    bar_open_time: str
    bar_close_time: str
    source: str
    source_bar_hash: str
    imbalance_ratio: float
    rejection_fraction: float
    aggressive_edge_volume: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "bar_open_time": self.bar_open_time,
            "bar_close_time": self.bar_close_time,
            "source": self.source,
            "source_bar_hash": self.source_bar_hash,
            "imbalance_ratio": self.imbalance_ratio,
            "rejection_fraction": self.rejection_fraction,
            "aggressive_edge_volume": self.aggressive_edge_volume,
        }


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


def _finite_number(value: Any, *, positive: bool = False) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if positive and number <= 0.0:
        return None
    return number


def _event_value(event: Any, *names: str) -> Any:
    for name in names:
        if isinstance(event, Mapping):
            if name in event:
                return event[name]
        elif hasattr(event, name):
            return getattr(event, name)
    return None


def _validated_volumes(
    event: Any,
) -> Optional[dict[str, float]]:
    volumes: dict[str, float] = {}
    for field in _VOLUME_FIELDS:
        value = _finite_number(_event_value(event, field))
        if value is None or value < 0.0:
            return None
        volumes[field] = value
    return volumes


def detect_footprint_absorption(
    *,
    candle: Mapping[str, Any],
    candle_open_time: Any,
    event: Any,
    symbol: str,
    side: AbsorptionSide,
    decision_time: Any,
    thresholds: AbsorptionThresholds = AbsorptionThresholds(),
) -> Optional[AbsorptionSignal]:
    """Return a footprint absorption signal or ``None`` on any uncertainty.

    LONG requires aggressive selling concentrated at the lower footprint edge
    followed by a close rejecting that low.  SHORT is the exact upper-edge
    counterpart with aggressive buying.  A centered close is not enough:
    LONG must close strictly above the candle midpoint and SHORT strictly
    below it.
    """

    if side not in {"LONG", "SHORT"}:
        return None
    if not isinstance(candle, Mapping) or event is None:
        return None
    schema_version = _event_value(event, "schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != ABSORPTION_EVENT_SCHEMA_VERSION
    ):
        return None
    if _event_value(event, "timeframe") != "15m":
        return None
    if _event_value(event, "finalized") is not True:
        return None
    revision = _event_value(event, "revision")
    if (
        not isinstance(revision, int)
        or isinstance(revision, bool)
        or revision != 0
    ):
        return None

    expected_symbol = str(symbol or "").strip().upper()
    raw_event_symbol = _event_value(event, "symbol")
    raw_source_symbol = _event_value(event, "source_symbol")
    if not isinstance(raw_event_symbol, str):
        return None
    if not isinstance(raw_source_symbol, str):
        return None
    event_symbol = raw_event_symbol.strip().upper()
    source_symbol = raw_source_symbol.strip().upper()
    if (
        not expected_symbol
        or event_symbol != expected_symbol
        or source_symbol != event_symbol
    ):
        return None
    polarity = _event_value(event, "polarity")
    if polarity != REQUIRED_POLARITY:
        return None
    source_raw = _event_value(event, "source")
    venue_raw = _event_value(event, "venue")
    if not isinstance(source_raw, str) or not source_raw.strip():
        return None
    if not isinstance(venue_raw, str) or not venue_raw.strip():
        return None
    source = source_raw.strip()
    source_bar_hash = _event_value(event, "checksum")
    if (
        not isinstance(source_bar_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", source_bar_hash)
    ):
        return None

    candle_open = _utc_timestamp(candle_open_time)
    decision = _utc_timestamp(decision_time)
    event_open = _utc_timestamp(_event_value(event, "bar_open"))
    event_close = _utc_timestamp(_event_value(event, "bar_close"))
    available_at = _utc_timestamp(_event_value(event, "available_at"))
    if any(
        timestamp is None
        for timestamp in (
            candle_open,
            decision,
            event_open,
            event_close,
            available_at,
        )
    ):
        return None
    assert candle_open is not None
    assert decision is not None
    assert event_open is not None
    assert event_close is not None
    assert available_at is not None
    if (
        event_open.minute % 15
        or event_open.second
        or event_open.microsecond
        or event_open.nanosecond
    ):
        return None
    if event_open != candle_open:
        return None
    if event_close - event_open != pd.Timedelta(minutes=15):
        return None
    if available_at < event_close:
        return None
    if event_close > decision or available_at > decision:
        return None

    prices: dict[str, float] = {}
    for field in ("high", "low", "close"):
        value = _finite_number(candle.get(field), positive=True)
        if value is None:
            return None
        prices[field] = value
    price_range = prices["high"] - prices["low"]
    if price_range <= 0.0:
        return None
    if not prices["low"] <= prices["close"] <= prices["high"]:
        return None

    volumes = _validated_volumes(event)
    if volumes is None:
        return None
    checksum_payload = {
        "schema_version": schema_version,
        "revision": revision,
        "timeframe": "15m",
        "source_symbol": source_symbol,
        "symbol": event_symbol,
        "bar_open": event_open.isoformat(),
        "bar_close": event_close.isoformat(),
        "available_at": available_at.isoformat(),
        "finalized": True,
        **volumes,
    }
    expected_checksum = hashlib.sha256(
        json.dumps(
            checksum_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if source_bar_hash != expected_checksum:
        return None

    close_location = (prices["close"] - prices["low"]) / price_range
    if side == "LONG":
        aggressive_edge = volumes["low_sell_volume"]
        opposite_edge = volumes["low_buy_volume"]
        rejection_fraction = close_location
    else:
        aggressive_edge = volumes["high_buy_volume"]
        opposite_edge = volumes["high_sell_volume"]
        rejection_fraction = 1.0 - close_location

    if aggressive_edge <= 0.0:
        return None
    # Keep audit payloads strictly JSON-serializable even when the opposite
    # edge has zero prints.  The cap is far above any practical threshold.
    imbalance_ratio = (
        1_000_000_000.0
        if opposite_edge == 0.0
        else aggressive_edge / opposite_edge
    )
    if aggressive_edge < thresholds.min_edge_volume:
        return None
    if imbalance_ratio < thresholds.min_imbalance_ratio:
        return None
    if rejection_fraction <= 0.5:
        return None

    return AbsorptionSignal(
        side=side,
        bar_open_time=event_open.isoformat(),
        bar_close_time=event_close.isoformat(),
        source=source,
        source_bar_hash=source_bar_hash,
        imbalance_ratio=float(imbalance_ratio),
        rejection_fraction=float(rejection_fraction),
        aggressive_edge_volume=float(aggressive_edge),
    )
