"""Deterministic executed-trade-tape to sealed absorption-sidecar conversion.

This module is the *only* supported way to produce a
``forexbot.absorption-15m`` schema-v1 sidecar from a raw trade tape.  It is
fail-closed by construction:

* the aggressor side must be reported by the venue/exchange; tick-rule,
  quote-inferred and delta-inferred provenance are refused outright, because
  the engineering contract forbids inferring Buy/Sell from OHLCV, tick volume
  or quote changes;
* the operator must attest the closed interval the tape is known to cover, so
  a truncated tail can never be sealed as a ``finalized`` bar;
* ``available_at`` comes from a named, versioned rule with recorded evidence.
  There is no default rule and no implicit ``available_at = bar_close``;
* the symbol mapping stays identity-only.  A futures/proxy tape (``6E`` ->
  ``EURUSD``) is rejected here with an explicit message instead of being
  silently sealed; a proxy needs a separately versioned schema;
* input must be sorted, complete and parseable.  A bad row aborts the build,
  it is never dropped.

Only the bar's extreme *traded* price levels matter for absorption, so each
event records executed aggressor volume at the highest and lowest traded level
of its closed 15-minute bar.

One converter run produces one immutable, single-symbol sidecar root.  To
assemble a multi-symbol sidecar, place the per-symbol shard files under one
root and call :func:`backtest.orderflow_data.write_manifest` once with the
full identity mapping.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import pandas as pd

from .orderflow_data import (
    MANIFEST_FILENAME,
    OrderFlowDataValidationError,
    REQUIRED_POLARITY,
    event_checksum,
    write_manifest,
)


CONVERTER_ID = "forexbot.absorption-tape-converter"
CONVERTER_VERSION = 1
BAR_LENGTH = pd.Timedelta(minutes=15)

VOLUME_MEASURES = ("executed_size", "trade_count")

#: Provenance values that prove the aggressor was reported by the trading
#: venue rather than reconstructed locally.
ALLOWED_AGGRESSOR_PROVENANCE = (
    "exchange_reported_aggressor",
    "venue_reported_aggressor",
)
#: Values that are explicitly refused so an operator cannot re-label a
#: reconstruction as executed aggressor flow.
REFUSED_AGGRESSOR_PROVENANCE = (
    "tick_rule_reconstructed",
    "quote_inferred",
    "delta_inferred",
    "unknown",
)

TIMESTAMP_FORMATS = (
    "iso8601",
    "epoch_s",
    "epoch_ms",
    "epoch_us",
    "epoch_ns",
)

_EPOCH_UNITS = {
    "epoch_s": "s",
    "epoch_ms": "ms",
    "epoch_us": "us",
    "epoch_ns": "ns",
}

_SPOT_FX_SYMBOL_LENGTH = 6


class TapeIngestError(ValueError):
    """Raised when a trade tape cannot be sealed into a causal sidecar."""


# ---------------------------------------------------------------------------
# available_at rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AvailableAtRule:
    """A named, versioned rule mapping a closed bar to its availability."""

    rule_id: str
    descriptor: str
    requires_arrival: bool
    offset: pd.Timedelta
    evidence: str

    def resolve(
        self,
        *,
        bar_close: pd.Timestamp,
        arrival_max: pd.Timestamp | None,
    ) -> pd.Timestamp:
        if not self.requires_arrival:
            return bar_close + self.offset
        if arrival_max is None:
            raise TapeIngestError(
                f"{self.rule_id} requires a per-trade arrival timestamp for "
                f"every print; bar closing {bar_close.isoformat()} has none"
            )
        candidate = arrival_max + self.offset
        # A print can arrive before the bar closes; the footprint itself is
        # still not knowable until the bar is complete.
        return max(candidate, bar_close)


def _require_non_negative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TapeIngestError(f"{field} must be a non-negative integer")
    return int(value)


def _evidence_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_available_at_rule(
    rule_id: str,
    *,
    vendor_delay_ms: int | None = None,
    delay_measurement: str | None = None,
    arrival_margin_ms: int | None = None,
) -> AvailableAtRule:
    """Build a versioned ``available_at`` rule, or fail closed.

    ``bar_close_plus_measured_vendor_delay.v1``
        ``available_at = bar_close + delay_ms``.  The delay must be measured,
        not guessed, and the measurement text is sealed into the sidecar
        ``source`` descriptor by SHA-256 digest.

    ``max_observed_arrival_plus_margin.v1``
        ``available_at = max(bar_close, max(arrival in bar) + margin_ms)``.
        Requires the tape to carry a per-print arrival timestamp, which is
        strictly stronger evidence than a constant delay.
    """

    if rule_id == "bar_close_plus_measured_vendor_delay.v1":
        delay = _require_non_negative_int(vendor_delay_ms, field="vendor_delay_ms")
        if not isinstance(delay_measurement, str) or not delay_measurement.strip():
            raise TapeIngestError(
                "bar_close_plus_measured_vendor_delay.v1 requires a non-empty "
                "delay_measurement describing how the vendor delay was measured"
            )
        evidence = delay_measurement.strip()
        digest = _evidence_digest(evidence)
        return AvailableAtRule(
            rule_id=rule_id,
            descriptor=(
                f"{rule_id}(delay_ms={delay},"
                f"measurement_sha256={digest})"
            ),
            requires_arrival=False,
            offset=pd.Timedelta(milliseconds=delay),
            evidence=evidence,
        )
    if rule_id == "max_observed_arrival_plus_margin.v1":
        margin = _require_non_negative_int(
            arrival_margin_ms,
            field="arrival_margin_ms",
        )
        return AvailableAtRule(
            rule_id=rule_id,
            descriptor=f"{rule_id}(margin_ms={margin})",
            requires_arrival=True,
            offset=pd.Timedelta(milliseconds=margin),
            evidence="per-print arrival timestamps supplied by the tape",
        )
    raise TapeIngestError(
        f"unknown available_at rule {rule_id!r}; supported rules are "
        "['bar_close_plus_measured_vendor_delay.v1', "
        "'max_observed_arrival_plus_margin.v1']"
    )


# ---------------------------------------------------------------------------
# Tape normalization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TapeTrade:
    """One executed print, normalized onto an exact price-level grid."""

    timestamp: pd.Timestamp
    level: int
    volume: Decimal
    aggressor: str
    arrival: pd.Timestamp | None


@dataclass(frozen=True)
class TapeColumns:
    """Column names inside the source tape."""

    timestamp: str = "timestamp"
    price: str = "price"
    size: str = "size"
    aggressor: str = "aggressor"
    arrival: str | None = None


def _decimal(value: Any, *, field: str, location: str) -> Decimal:
    if value is None:
        raise TapeIngestError(f"{location}: {field} is missing")
    text = str(value).strip()
    if not text:
        raise TapeIngestError(f"{location}: {field} is empty")
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise TapeIngestError(
            f"{location}: {field} is not a decimal number: {text!r}"
        ) from exc
    if not number.is_finite():
        raise TapeIngestError(f"{location}: {field} must be finite: {text!r}")
    return number


def _parse_timestamp(
    value: Any,
    *,
    field: str,
    location: str,
    timestamp_format: str,
    naive_is_utc: bool,
) -> pd.Timestamp:
    if value is None or str(value).strip() == "":
        raise TapeIngestError(f"{location}: {field} is missing")
    text = str(value).strip()
    if timestamp_format in _EPOCH_UNITS:
        try:
            epoch = int(text)
        except ValueError as exc:
            raise TapeIngestError(
                f"{location}: {field} must be an integer epoch value for "
                f"format {timestamp_format!r}: {text!r}"
            ) from exc
        return pd.Timestamp(epoch, unit=_EPOCH_UNITS[timestamp_format], tz="UTC")
    if timestamp_format != "iso8601":
        raise TapeIngestError(f"unsupported timestamp format {timestamp_format!r}")
    try:
        timestamp = pd.Timestamp(text)
    except Exception as exc:  # pragma: no cover - pandas controls exact errors
        raise TapeIngestError(
            f"{location}: {field} is not a valid timestamp: {text!r}"
        ) from exc
    if pd.isna(timestamp):
        raise TapeIngestError(f"{location}: {field} is not a valid timestamp")
    if timestamp.tzinfo is None:
        if not naive_is_utc:
            raise TapeIngestError(
                f"{location}: {field} has no timezone; fix the tape or pass "
                "naive_timestamps_are_utc explicitly"
            )
        return timestamp.tz_localize("UTC")
    if timestamp.utcoffset() != pd.Timedelta(0):
        return timestamp.tz_convert("UTC")
    return timestamp.tz_convert("UTC")


def _price_level(price: Decimal, tick_size: Decimal, *, location: str) -> int:
    if price <= 0:
        raise TapeIngestError(f"{location}: price must be strictly positive")
    remainder = price % tick_size
    if remainder != 0:
        raise TapeIngestError(
            f"{location}: price {price} is not an exact multiple of tick size "
            f"{tick_size}; the declared tick size is wrong for this tape"
        )
    return int(price / tick_size)


def normalize_csv_tape(
    path: Path,
    *,
    columns: TapeColumns,
    tick_size: Decimal,
    volume_measure: str,
    buy_labels: Sequence[str],
    sell_labels: Sequence[str],
    timestamp_format: str,
    naive_is_utc: bool,
) -> Iterator[TapeTrade]:
    """Stream one CSV tape file as normalized, validated prints."""

    if volume_measure not in VOLUME_MEASURES:
        raise TapeIngestError(
            f"unsupported volume_measure {volume_measure!r}; expected one of "
            f"{list(VOLUME_MEASURES)}"
        )
    if any(not isinstance(label, str) or not label.strip() for label in buy_labels):
        raise TapeIngestError("buy aggressor labels must be non-empty strings")
    if any(not isinstance(label, str) or not label.strip() for label in sell_labels):
        raise TapeIngestError("sell aggressor labels must be non-empty strings")
    buy = {label.strip().upper() for label in buy_labels}
    sell = {label.strip().upper() for label in sell_labels}
    overlap = buy & sell
    if overlap:
        raise TapeIngestError(
            f"aggressor labels claimed as both buy and sell: {sorted(overlap)}"
        )
    if not buy or not sell:
        raise TapeIngestError("both buy and sell aggressor labels are required")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        required = [columns.timestamp, columns.price, columns.size, columns.aggressor]
        if columns.arrival is not None:
            required.append(columns.arrival)
        missing = [name for name in required if name not in header]
        if missing:
            raise TapeIngestError(f"{path}: tape is missing column(s): {missing}")
        for index, row in enumerate(reader, start=2):
            location = f"{path.name}:{index}"
            timestamp = _parse_timestamp(
                row.get(columns.timestamp),
                field="timestamp",
                location=location,
                timestamp_format=timestamp_format,
                naive_is_utc=naive_is_utc,
            )
            price = _decimal(row.get(columns.price), field="price", location=location)
            level = _price_level(price, tick_size, location=location)
            size = _decimal(row.get(columns.size), field="size", location=location)
            if size <= 0:
                raise TapeIngestError(
                    f"{location}: size must be strictly positive; a zero or "
                    "negative print is not an execution"
                )
            raw_side = str(row.get(columns.aggressor) or "").strip().upper()
            if raw_side in buy:
                aggressor = "BUY"
            elif raw_side in sell:
                aggressor = "SELL"
            else:
                raise TapeIngestError(
                    f"{location}: unmapped aggressor label {raw_side!r}; every "
                    "label must be explicitly declared as buy or sell"
                )
            arrival: pd.Timestamp | None = None
            if columns.arrival is not None:
                arrival = _parse_timestamp(
                    row.get(columns.arrival),
                    field="arrival",
                    location=location,
                    timestamp_format=timestamp_format,
                    naive_is_utc=naive_is_utc,
                )
                if arrival < timestamp:
                    raise TapeIngestError(
                        f"{location}: arrival precedes execution timestamp"
                    )
            volume = size if volume_measure == "executed_size" else Decimal(1)
            yield TapeTrade(
                timestamp=timestamp,
                level=level,
                volume=volume,
                aggressor=aggressor,
                arrival=arrival,
            )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class _BarAccumulator:
    bar_open: pd.Timestamp
    levels: Dict[int, List[Decimal]]
    trades: int
    last_timestamp: pd.Timestamp
    max_gap: pd.Timedelta
    arrival_max: pd.Timestamp | None


@dataclass(frozen=True)
class AggregationResult:
    """Sealed-ready event records plus the audit trail of what was excluded."""

    events: Tuple[Dict[str, Any], ...] = ()
    trades_read: int = 0
    bars_seen: int = 0
    excluded: Tuple[Dict[str, Any], ...] = ()

    def exclusion_counts(self) -> Dict[str, int]:
        counter: Counter[str] = Counter(str(item["reason"]) for item in self.excluded)
        return dict(sorted(counter.items()))


def _timestamp_text(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{value.microsecond // 1000:03d}Z"


def aggregate_trades(
    trades: Iterable[TapeTrade],
    *,
    symbol: str,
    rule: AvailableAtRule,
    covers_from: pd.Timestamp,
    covers_through: pd.Timestamp,
    min_trades_per_bar: int = 1,
    max_intrabar_gap: pd.Timedelta | None = None,
) -> AggregationResult:
    """Fold a sorted print stream into checksummed schema-v1 event records."""

    if min_trades_per_bar < 1:
        raise TapeIngestError("min_trades_per_bar must be at least 1")
    if covers_through <= covers_from:
        raise TapeIngestError("covers_through must be later than covers_from")
    if max_intrabar_gap is not None and (
        pd.isna(max_intrabar_gap) or max_intrabar_gap < pd.Timedelta(0)
    ):
        raise TapeIngestError("max_intrabar_gap must be finite and non-negative")

    normalized_symbol = require_identity_symbol(symbol)
    events: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    current: _BarAccumulator | None = None
    previous_timestamp: pd.Timestamp | None = None
    trades_read = 0
    bars_seen = 0

    def flush(accumulator: _BarAccumulator) -> None:
        nonlocal bars_seen
        bars_seen += 1
        bar_open = accumulator.bar_open
        bar_close = bar_open + BAR_LENGTH
        record_reason: str | None = None
        if bar_open < covers_from or bar_close > covers_through:
            record_reason = "outside_attested_coverage"
        elif accumulator.trades < min_trades_per_bar:
            record_reason = "below_min_trades_per_bar"
        elif max_intrabar_gap is not None and accumulator.max_gap > max_intrabar_gap:
            record_reason = "intrabar_gap_exceeds_limit"
        if record_reason is not None:
            excluded.append(
                {
                    "bar_open": _timestamp_text(bar_open),
                    "reason": record_reason,
                    "trades": accumulator.trades,
                }
            )
            return
        high_level = max(accumulator.levels)
        low_level = min(accumulator.levels)
        high_buy, high_sell = accumulator.levels[high_level]
        low_buy, low_sell = accumulator.levels[low_level]
        available_at = rule.resolve(
            bar_close=bar_close,
            arrival_max=accumulator.arrival_max,
        )
        record = {
            "schema_version": 1,
            "revision": 0,
            "timeframe": "15m",
            "source_symbol": normalized_symbol,
            "symbol": normalized_symbol,
            "bar_open": _timestamp_text(bar_open),
            "bar_close": _timestamp_text(bar_close),
            "available_at": _timestamp_text(available_at),
            "finalized": True,
            "high_buy_volume": float(high_buy),
            "high_sell_volume": float(high_sell),
            "low_buy_volume": float(low_buy),
            "low_sell_volume": float(low_sell),
        }
        record["checksum"] = event_checksum(record)
        events.append(record)

    for trade in trades:
        trades_read += 1
        if trade.aggressor not in {"BUY", "SELL"}:
            raise TapeIngestError(
                f"normalized aggressor must be BUY or SELL, got {trade.aggressor!r}"
            )
        if (
            not isinstance(trade.volume, Decimal)
            or not trade.volume.is_finite()
            or trade.volume <= 0
        ):
            raise TapeIngestError("normalized trade volume must be finite and positive")
        if isinstance(trade.level, bool) or not isinstance(trade.level, int):
            raise TapeIngestError("normalized trade price level must be an integer")
        if (
            not isinstance(trade.timestamp, pd.Timestamp)
            or pd.isna(trade.timestamp)
            or trade.timestamp.tzinfo is None
            or trade.timestamp.utcoffset() != pd.Timedelta(0)
        ):
            raise TapeIngestError(
                "normalized execution timestamp must be an explicit UTC timestamp"
            )
        if trade.arrival is not None and (
            not isinstance(trade.arrival, pd.Timestamp)
            or pd.isna(trade.arrival)
            or trade.arrival.tzinfo is None
            or trade.arrival.utcoffset() != pd.Timedelta(0)
            or trade.arrival < trade.timestamp
        ):
            raise TapeIngestError(
                "normalized arrival timestamp must be UTC and not precede execution"
            )
        if previous_timestamp is not None and trade.timestamp < previous_timestamp:
            raise TapeIngestError(
                "tape must be sorted by non-decreasing execution timestamp; "
                f"{trade.timestamp.isoformat()} follows "
                f"{previous_timestamp.isoformat()}"
            )
        previous_timestamp = trade.timestamp
        bar_open = trade.timestamp.floor("15min")
        if current is None or bar_open != current.bar_open:
            if current is not None:
                if bar_open < current.bar_open:  # pragma: no cover - sorted input
                    raise TapeIngestError("bar sequence regressed")
                flush(current)
            current = _BarAccumulator(
                bar_open=bar_open,
                levels={},
                trades=0,
                last_timestamp=trade.timestamp,
                max_gap=pd.Timedelta(0),
                arrival_max=None,
            )
        gap = trade.timestamp - current.last_timestamp
        if gap > current.max_gap:
            current.max_gap = gap
        current.last_timestamp = trade.timestamp
        current.trades += 1
        bucket = current.levels.get(trade.level)
        if bucket is None:
            bucket = [Decimal(0), Decimal(0)]
            current.levels[trade.level] = bucket
        bucket[0 if trade.aggressor == "BUY" else 1] += trade.volume
        if trade.arrival is not None:
            if current.arrival_max is None or trade.arrival > current.arrival_max:
                current.arrival_max = trade.arrival

    if current is not None:
        flush(current)

    return AggregationResult(
        events=tuple(events),
        trades_read=trades_read,
        bars_seen=bars_seen,
        excluded=tuple(excluded),
    )


def require_identity_symbol(symbol: Any, source_symbol: Any = None) -> str:
    """Validate a spot-FX identity mapping, or refuse with an explicit reason."""

    normalized = str(symbol or "").strip().upper()
    if not normalized:
        raise TapeIngestError("symbol must be a non-empty string")
    if source_symbol is not None:
        normalized_source = str(source_symbol or "").strip().upper()
        if normalized_source != normalized:
            raise TapeIngestError(
                f"symbol mapping {normalized_source!r} -> {normalized!r} is not "
                "identity; a futures or cross-market proxy tape requires a "
                "separately versioned proxy schema and is refused by "
                "forexbot.absorption-15m v1"
            )
    if (
        len(normalized) != _SPOT_FX_SYMBOL_LENGTH
        or not normalized.isascii()
        or not normalized.isalpha()
    ):
        raise TapeIngestError(
            f"{normalized!r} is not a six-letter spot FX symbol; schema v1 "
            "seals spot identity mappings only, a derivative or proxy "
            "instrument requires a separately versioned proxy schema"
        )
    return normalized


# ---------------------------------------------------------------------------
# Sidecar assembly
# ---------------------------------------------------------------------------


def build_source_descriptor(
    *,
    source_id: str,
    volume_measure: str,
    tick_size: Decimal,
    aggressor_provenance: str,
    rule: AvailableAtRule,
) -> str:
    """Compose the deterministic provenance string sealed into the manifest."""

    text = str(source_id).strip()
    if not text or ";" in text or "=" in text:
        raise TapeIngestError(
            "source_id must be a non-empty label without ';' or '=' characters"
        )
    return ";".join(
        (
            text,
            f"converter={CONVERTER_ID}/{CONVERTER_VERSION}",
            f"volume={volume_measure}",
            f"tick_size={tick_size.normalize():f}",
            f"aggressor={aggressor_provenance}",
            f"available_at={rule.descriptor}",
        )
    )


def require_aggressor_provenance(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in REFUSED_AGGRESSOR_PROVENANCE:
        raise TapeIngestError(
            f"aggressor provenance {normalized!r} is a local reconstruction, "
            "not executed aggressor flow; it can never be sealed as a "
            "forexbot.absorption-15m event"
        )
    if normalized not in ALLOWED_AGGRESSOR_PROVENANCE:
        raise TapeIngestError(
            f"unknown aggressor provenance {normalized!r}; allowed values are "
            f"{list(ALLOWED_AGGRESSOR_PROVENANCE)}"
        )
    return normalized


def _shard_records(
    records: Sequence[Mapping[str, Any]],
    *,
    shard: str,
) -> Dict[str, List[Mapping[str, Any]]]:
    if shard == "none":
        return {"events.json": list(records)}
    if shard != "month":
        raise TapeIngestError(f"unsupported shard mode {shard!r}")
    shards: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        month = str(record["bar_open"])[:7]
        shards.setdefault(f"events/{month}.json", []).append(record)
    return dict(sorted(shards.items()))


def write_sidecar(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    source: str,
    venue: str,
    symbol: str,
    shard: str = "none",
) -> Dict[str, Any]:
    """Build a sealed v1 sidecar in staging, then publish it atomically."""

    if not records:
        raise TapeIngestError(
            "no finalized bars survived validation; nothing can be sealed and "
            "Absorption stays DATA_UNAVAILABLE"
        )
    base = Path(root).expanduser().resolve()
    if base.exists():
        if not base.is_dir():
            raise TapeIngestError(f"{base} exists and is not a directory")
        if any(base.iterdir()):
            raise TapeIngestError(
                f"{base} is not empty; a sealed sidecar is immutable, build a "
                "new directory instead of writing over an existing one"
            )
    shards = _shard_records(records, shard=shard)
    base.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{base.name}.partial-", dir=str(base.parent))
    )
    try:
        for relative, shard_records in shards.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(
                    list(shard_records),
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        manifest_path = write_manifest(
            staging,
            source=source,
            venue=venue,
            symbol_mapping={symbol: symbol},
            polarity=REQUIRED_POLARITY,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if base.exists():
            if not base.is_dir() or any(base.iterdir()):
                raise TapeIngestError(
                    f"{base} changed while the sidecar was built; refusing publication"
                )
            base.rmdir()
        staging.replace(base)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_inputs(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    """Hash inputs and fail if a file changes while its digest is computed."""

    captured: List[Dict[str, Any]] = []
    for path in paths:
        before = path.stat()
        digest = _sha256_file(path)
        after = path.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise TapeIngestError(f"tape changed while hashing: {path}")
        captured.append(
            {
                "path": str(path),
                "size": int(after.st_size),
                "sha256": digest,
            }
        )
    return captured


def build_receipt(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    aggregation: AggregationResult,
    inputs: Sequence[Mapping[str, Any]],
    symbol: str,
    venue: str,
    source: str,
    tick_size: Decimal,
    volume_measure: str,
    aggressor_provenance: str,
    rule: AvailableAtRule,
    covers_from: pd.Timestamp,
    covers_through: pd.Timestamp,
    min_trades_per_bar: int,
    max_intrabar_gap: pd.Timedelta | None,
    naive_is_utc: bool,
    timestamp_format: str,
) -> Dict[str, Any]:
    """Build the out-of-band audit receipt for one converter run.

    The receipt intentionally lives outside the sealed root: schema v1 closes
    both the manifest field set and the sidecar file set, so an extra JSON file
    inside the root would break loading.
    """

    events = aggregation.events
    return {
        "converter": f"{CONVERTER_ID}/{CONVERTER_VERSION}",
        "sidecar_root": str(root),
        "manifest_sha256": manifest["manifest_sha256"],
        "content_sha256": manifest["content_sha256"],
        "symbol": symbol,
        "venue": venue,
        "source": source,
        "symbol_mapping": {symbol: symbol},
        "polarity": REQUIRED_POLARITY,
        "tick_size": f"{tick_size.normalize():f}",
        "volume_measure": volume_measure,
        "aggressor_provenance": aggressor_provenance,
        "timestamp_format": timestamp_format,
        "naive_timestamps_are_utc": bool(naive_is_utc),
        "available_at_rule": {
            "rule_id": rule.rule_id,
            "descriptor": rule.descriptor,
            "offset_ms": int(rule.offset / pd.Timedelta(milliseconds=1)),
            "requires_arrival_timestamps": rule.requires_arrival,
            "evidence": rule.evidence,
            "evidence_sha256": _evidence_digest(rule.evidence),
        },
        "attested_coverage": {
            "covers_from": _timestamp_text(covers_from),
            "covers_through": _timestamp_text(covers_through),
        },
        "quality_gates": {
            "min_trades_per_bar": int(min_trades_per_bar),
            "max_intrabar_gap_seconds": (
                None
                if max_intrabar_gap is None
                else max_intrabar_gap / pd.Timedelta(seconds=1)
            ),
        },
        "inputs": [dict(item) for item in inputs],
        "counts": {
            "trades_read": aggregation.trades_read,
            "bars_seen": aggregation.bars_seen,
            "events_sealed": len(events),
            "bars_excluded": len(aggregation.excluded),
        },
        "exclusion_reasons": aggregation.exclusion_counts(),
        "excluded_sample": list(aggregation.excluded[:200]),
        "first_event_bar_open": events[0]["bar_open"] if events else None,
        "last_event_bar_open": events[-1]["bar_open"] if events else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build-absorption-sidecar",
        description=(
            "Convert an executed-trade tape into a sealed "
            "forexbot.absorption-15m v1 sidecar. Fail-closed by design."
        ),
    )
    parser.add_argument(
        "--tape",
        action="append",
        required=True,
        help="CSV tape file, repeatable, supplied in chronological order",
    )
    parser.add_argument("--out", required=True, help="new, empty sidecar directory")
    parser.add_argument("--receipt", help="audit receipt path (must be outside --out)")
    parser.add_argument("--symbol", required=True, help="strategy symbol, e.g. EURUSD")
    parser.add_argument(
        "--source-symbol",
        help="symbol as written by the tape; must equal --symbol in schema v1",
    )
    parser.add_argument("--venue", required=True, help="execution venue label")
    parser.add_argument("--source-id", required=True, help="feed/product label")
    parser.add_argument("--tick-size", required=True, help="exact price grid, e.g. 0.00001")
    parser.add_argument(
        "--volume-measure",
        choices=VOLUME_MEASURES,
        required=True,
        help=(
            "executed_size sums traded size, trade_count counts prints; the "
            "audited min_edge_volume=80 threshold is scale-dependent"
        ),
    )
    parser.add_argument(
        "--aggressor-provenance",
        required=True,
        help=f"one of {list(ALLOWED_AGGRESSOR_PROVENANCE)}",
    )
    parser.add_argument("--column-timestamp", default="timestamp")
    parser.add_argument("--column-price", default="price")
    parser.add_argument("--column-size", default="size")
    parser.add_argument("--column-aggressor", default="aggressor")
    parser.add_argument(
        "--column-arrival",
        help="per-print arrival timestamp column, required by the arrival rule",
    )
    parser.add_argument("--buy-label", action="append", required=True)
    parser.add_argument("--sell-label", action="append", required=True)
    parser.add_argument(
        "--timestamp-format",
        choices=TIMESTAMP_FORMATS,
        required=True,
    )
    parser.add_argument(
        "--naive-timestamps-are-utc",
        action="store_true",
        help="explicitly accept tz-naive ISO timestamps as UTC",
    )
    parser.add_argument(
        "--covers-from",
        required=True,
        help="UTC instant from which the tape is attested complete",
    )
    parser.add_argument(
        "--covers-through",
        required=True,
        help="UTC instant through which the tape is attested complete",
    )
    parser.add_argument("--available-at-rule", required=True)
    parser.add_argument("--vendor-delay-ms", type=int)
    parser.add_argument("--delay-measurement")
    parser.add_argument("--arrival-margin-ms", type=int)
    parser.add_argument("--min-trades-per-bar", type=int, default=1)
    parser.add_argument("--max-intrabar-gap-seconds", type=float)
    parser.add_argument("--shard", choices=("none", "month"), default="month")
    return parser


def _coverage_timestamp(value: str, *, field: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise TapeIngestError(
            f"{field} is not a valid timestamp: {value!r}"
        ) from exc
    if pd.isna(timestamp):
        raise TapeIngestError(f"{field} is not a valid timestamp: {value!r}")
    if timestamp.tzinfo is None:
        raise TapeIngestError(f"{field} must be an explicit UTC timestamp")
    return timestamp.tz_convert("UTC")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute one converter run and return its receipt."""

    symbol = require_identity_symbol(args.symbol, args.source_symbol or args.symbol)
    provenance = require_aggressor_provenance(args.aggressor_provenance)
    tick_size = _decimal(args.tick_size, field="tick_size", location="--tick-size")
    if tick_size <= 0:
        raise TapeIngestError("--tick-size must be strictly positive")
    rule = resolve_available_at_rule(
        args.available_at_rule,
        vendor_delay_ms=args.vendor_delay_ms,
        delay_measurement=args.delay_measurement,
        arrival_margin_ms=args.arrival_margin_ms,
    )
    if rule.requires_arrival and not args.column_arrival:
        raise TapeIngestError(
            f"{rule.rule_id} requires --column-arrival on the tape"
        )
    covers_from = _coverage_timestamp(args.covers_from, field="--covers-from")
    covers_through = _coverage_timestamp(args.covers_through, field="--covers-through")
    max_gap_seconds = (
        None
        if args.max_intrabar_gap_seconds is None
        else float(args.max_intrabar_gap_seconds)
    )
    if max_gap_seconds is not None and (
        not math.isfinite(max_gap_seconds) or max_gap_seconds < 0
    ):
        raise TapeIngestError(
            "--max-intrabar-gap-seconds must be finite and non-negative"
        )
    max_gap = (
        None
        if max_gap_seconds is None
        else pd.Timedelta(seconds=max_gap_seconds)
    )
    tapes = [Path(item).expanduser().resolve() for item in args.tape]
    for path in tapes:
        if not path.is_file():
            raise TapeIngestError(f"tape file does not exist: {path}")
    input_snapshot = _snapshot_inputs(tapes)
    root = Path(args.out).expanduser().resolve()
    receipt_path = (
        Path(args.receipt).expanduser().resolve()
        if args.receipt
        else root.parent / f"{root.name}.build_receipt.json"
    )
    try:
        receipt_path.relative_to(root)
    except ValueError:
        pass
    else:
        raise TapeIngestError(
            "the receipt must live outside the sealed sidecar root; schema v1 "
            "seals the whole directory file set"
        )
    if receipt_path in tapes:
        raise TapeIngestError("the audit receipt must not overwrite an input tape")
    if receipt_path.exists():
        raise TapeIngestError(
            f"audit receipt already exists and is immutable: {receipt_path}"
        )

    columns = TapeColumns(
        timestamp=args.column_timestamp,
        price=args.column_price,
        size=args.column_size,
        aggressor=args.column_aggressor,
        arrival=args.column_arrival,
    )

    def stream() -> Iterator[TapeTrade]:
        for path in tapes:
            yield from normalize_csv_tape(
                path,
                columns=columns,
                tick_size=tick_size,
                volume_measure=args.volume_measure,
                buy_labels=args.buy_label,
                sell_labels=args.sell_label,
                timestamp_format=args.timestamp_format,
                naive_is_utc=bool(args.naive_timestamps_are_utc),
            )

    aggregation = aggregate_trades(
        stream(),
        symbol=symbol,
        rule=rule,
        covers_from=covers_from,
        covers_through=covers_through,
        min_trades_per_bar=int(args.min_trades_per_bar),
        max_intrabar_gap=max_gap,
    )
    if _snapshot_inputs(tapes) != input_snapshot:
        raise TapeIngestError(
            "one or more tape files changed during conversion; refusing to seal"
        )
    source = build_source_descriptor(
        source_id=args.source_id,
        volume_measure=args.volume_measure,
        tick_size=tick_size,
        aggressor_provenance=provenance,
        rule=rule,
    )
    manifest = write_sidecar(
        root,
        aggregation.events,
        source=source,
        venue=args.venue,
        symbol=symbol,
        shard=args.shard,
    )
    receipt = build_receipt(
        root=root,
        manifest=manifest,
        aggregation=aggregation,
        inputs=input_snapshot,
        symbol=symbol,
        venue=args.venue,
        source=source,
        tick_size=tick_size,
        volume_measure=args.volume_measure,
        aggressor_provenance=provenance,
        rule=rule,
        covers_from=covers_from,
        covers_through=covers_through,
        min_trades_per_bar=int(args.min_trades_per_bar),
        max_intrabar_gap=max_gap,
        naive_is_utc=bool(args.naive_timestamps_are_utc),
        timestamp_format=args.timestamp_format,
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                receipt,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        receipt = run(args)
    except (
        TapeIngestError,
        OrderFlowDataValidationError,
        FileNotFoundError,
        OSError,
    ) as exc:
        parser.exit(2, f"error: {exc}\n")
        return 2
    counts = receipt["counts"]
    print(f"sidecar: {receipt['sidecar_root']}")
    print(f"manifest: {receipt['sidecar_root']}/{MANIFEST_FILENAME}")
    print(f"manifest_sha256: {receipt['manifest_sha256']}")
    print(f"content_sha256: {receipt['content_sha256']}")
    print(
        "trades={trades_read} bars={bars_seen} sealed={events_sealed} "
        "excluded={bars_excluded}".format(**counts)
    )
    for reason, count in receipt["exclusion_reasons"].items():
        print(f"  excluded[{reason}]={count}")
    return 0


__all__ = [
    "ALLOWED_AGGRESSOR_PROVENANCE",
    "AggregationResult",
    "AvailableAtRule",
    "CONVERTER_ID",
    "CONVERTER_VERSION",
    "REFUSED_AGGRESSOR_PROVENANCE",
    "TapeColumns",
    "TapeIngestError",
    "TapeTrade",
    "VOLUME_MEASURES",
    "aggregate_trades",
    "build_parser",
    "build_receipt",
    "build_source_descriptor",
    "main",
    "normalize_csv_tape",
    "require_aggressor_provenance",
    "require_identity_symbol",
    "resolve_available_at_rule",
    "run",
    "write_sidecar",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
