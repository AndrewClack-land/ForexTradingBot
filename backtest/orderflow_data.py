"""Sealed, causal 15-minute absorption-event sidecar loading.

The sidecar is intentionally separate from OHLCV candles.  An event in this
dataset must come from a Bid/Ask footprint source whose volume polarity is
explicitly pinned in the manifest; candle volume is not a valid substitute.

Directory contract (schema version 1)::

    orderflow/
      manifest.json
      events.json          # or one/more Parquet shards

Every event is checksummed independently.  The manifest additionally pins all
source files and a canonical, serialization-independent content hash.  Loading
fails closed if provenance, mapping, polarity, timestamps, finalization, hashes
or event uniqueness cannot be proven.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Tuple

import pandas as pd


SCHEMA_NAME = "forexbot.absorption-15m"
SCHEMA_VERSION = 1
EVENT_REVISION = 0
TIMEFRAME = "15m"
MANIFEST_FILENAME = "manifest.json"

# Quantower's Bid/Ask volume convention used by the footprint exporter:
# BuyVolume is aggressive volume executed at ask, SellVolume at bid.
REQUIRED_POLARITY = "buy_volume=aggressor_at_ask;sell_volume=aggressor_at_bid"

_DATA_SUFFIXES = {".json", ".parquet", ".pq"}
_VOLUME_FIELDS = (
    "high_buy_volume",
    "high_sell_volume",
    "low_buy_volume",
    "low_sell_volume",
)
_EVENT_FIELDS = {
    "schema_version",
    "revision",
    "timeframe",
    "source_symbol",
    "symbol",
    "bar_open",
    "bar_close",
    "available_at",
    "finalized",
    *_VOLUME_FIELDS,
    "checksum",
}
_MANIFEST_FIELDS = {
    "schema",
    "schema_version",
    "timeframe",
    "source",
    "venue",
    "symbol_mapping",
    "polarity",
    "files",
    "content_sha256",
    "manifest_sha256",
}
_FILE_FIELDS = {"path", "size", "sha256"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class OrderFlowDataValidationError(ValueError):
    """Raised when an absorption sidecar is not sealed and causally valid."""


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise OrderFlowDataValidationError(
            f"{field} must be a lowercase 64-character SHA-256"
        )
    return value


def _require_nonempty_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OrderFlowDataValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _normalized_symbol(value: Any, *, field: str) -> str:
    return _require_nonempty_text(value, field=field).upper()


def _utc_timestamp(value: Any, *, field: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:  # pragma: no cover - pandas controls exact errors
        raise OrderFlowDataValidationError(
            f"{field} is not a valid timestamp: {value!r}"
        ) from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise OrderFlowDataValidationError(
            f"{field} must be an explicit timezone-aware UTC timestamp"
        )
    if timestamp.utcoffset() != pd.Timedelta(0):
        raise OrderFlowDataValidationError(
            f"{field} must use UTC (Z or +00:00), not a non-zero offset"
        )
    return timestamp.tz_convert("UTC")


def _aligned_bar_open(value: Any, *, field: str = "bar_open") -> pd.Timestamp:
    timestamp = _utc_timestamp(value, field=field)
    if (
        timestamp.minute % 15
        or timestamp.second
        or timestamp.microsecond
        or timestamp.nanosecond
    ):
        raise OrderFlowDataValidationError(
            f"{field} must be aligned to a 15-minute UTC boundary"
        )
    return timestamp


def _finite_nonnegative_volume(value: Any, *, field: str) -> float:
    if isinstance(value, (bool, str, bytes)) or value is None:
        raise OrderFlowDataValidationError(
            f"{field} must be a finite non-negative number"
        )
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OrderFlowDataValidationError(
            f"{field} must be a finite non-negative number"
        ) from exc
    if not math.isfinite(number) or number < 0:
        raise OrderFlowDataValidationError(
            f"{field} must be a finite non-negative number"
        )
    return number


def _normalize_mapping(value: Any) -> Dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise OrderFlowDataValidationError(
            "symbol_mapping must be a non-empty source-to-strategy mapping"
        )
    normalized: Dict[str, str] = {}
    targets: set[str] = set()
    for raw_source, raw_target in value.items():
        source_symbol = _normalized_symbol(
            raw_source,
            field="symbol_mapping source symbol",
        )
        target_symbol = _normalized_symbol(
            raw_target,
            field=f"symbol_mapping[{source_symbol!r}]",
        )
        if source_symbol in normalized:
            raise OrderFlowDataValidationError(
                f"symbol_mapping contains duplicate normalized source {source_symbol!r}"
            )
        if target_symbol in targets:
            raise OrderFlowDataValidationError(
                f"symbol_mapping contains duplicate target {target_symbol!r}; "
                "many-to-one mappings are forbidden"
            )
        if source_symbol != target_symbol:
            raise OrderFlowDataValidationError(
                "symbol_mapping must be identity-only until a separately "
                f"versioned proxy schema exists: {source_symbol!r} -> "
                f"{target_symbol!r}"
            )
        normalized[source_symbol] = target_symbol
        targets.add(target_symbol)
    return dict(sorted(normalized.items()))


@dataclass(frozen=True)
class AbsorptionEvent:
    """A finalized, provenance-pinned footprint event for one closed M15 bar."""

    schema_version: int
    revision: int
    timeframe: str
    source: str
    venue: str
    source_symbol: str
    symbol: str
    polarity: str
    bar_open: pd.Timestamp
    bar_close: pd.Timestamp
    available_at: pd.Timestamp
    finalized: bool
    high_buy_volume: float
    high_sell_volume: float
    low_buy_volume: float
    low_sell_volume: float
    checksum: str

    def checksum_payload(self) -> Dict[str, Any]:
        """Return the canonical row payload covered by ``checksum``."""

        return {
            "schema_version": self.schema_version,
            "revision": self.revision,
            "timeframe": self.timeframe,
            "source_symbol": self.source_symbol,
            "symbol": self.symbol,
            "bar_open": self.bar_open.isoformat(),
            "bar_close": self.bar_close.isoformat(),
            "available_at": self.available_at.isoformat(),
            "finalized": self.finalized,
            **{
                field: float(getattr(self, field))
                for field in _VOLUME_FIELDS
            },
        }

    def to_record(self) -> Dict[str, Any]:
        """Return the canonical event record, including its checksum."""

        return {**self.checksum_payload(), "checksum": self.checksum}


def _parse_event(
    raw: Mapping[str, Any],
    *,
    source: str,
    venue: str,
    polarity: str,
    symbol_mapping: Mapping[str, str] | None,
    location: str,
    require_checksum: bool,
) -> AbsorptionEvent:
    if not isinstance(raw, Mapping):
        raise OrderFlowDataValidationError(f"{location}: event must be an object")
    expected_fields = _EVENT_FIELDS if require_checksum else _EVENT_FIELDS - {"checksum"}
    actual_fields = set(raw)
    missing = sorted(expected_fields - actual_fields)
    extra = sorted(actual_fields - expected_fields)
    if missing:
        raise OrderFlowDataValidationError(
            f"{location}: missing event field(s): {missing}"
        )
    if extra:
        raise OrderFlowDataValidationError(
            f"{location}: unknown event field(s): {extra}"
        )

    version = raw["schema_version"]
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        raise OrderFlowDataValidationError(
            f"{location}: unsupported event schema_version {version!r}"
        )
    revision = raw["revision"]
    if (
        isinstance(revision, bool)
        or not isinstance(revision, Integral)
        or int(revision) != EVENT_REVISION
    ):
        raise OrderFlowDataValidationError(
            f"{location}: revision must be the integer {EVENT_REVISION}"
        )
    timeframe = raw["timeframe"]
    if timeframe != TIMEFRAME:
        raise OrderFlowDataValidationError(
            f"{location}: timeframe must be {TIMEFRAME!r}"
        )

    source_symbol = _normalized_symbol(
        raw["source_symbol"],
        field=f"{location}.source_symbol",
    )
    symbol = _normalized_symbol(raw["symbol"], field=f"{location}.symbol")
    if source_symbol != symbol:
        raise OrderFlowDataValidationError(
            f"{location}: source_symbol must equal symbol in schema v1; "
            "proxy mappings require a separately versioned schema"
        )
    if symbol_mapping is not None:
        mapped = symbol_mapping.get(source_symbol)
        if mapped is None:
            raise OrderFlowDataValidationError(
                f"{location}: source symbol {source_symbol!r} is not in symbol_mapping"
            )
        if mapped != symbol:
            raise OrderFlowDataValidationError(
                f"{location}: symbol {symbol!r} conflicts with mapping "
                f"{source_symbol!r} -> {mapped!r}"
            )

    bar_open = _aligned_bar_open(raw["bar_open"])
    bar_close = _utc_timestamp(raw["bar_close"], field=f"{location}.bar_close")
    expected_close = bar_open + pd.Timedelta(minutes=15)
    if bar_close != expected_close:
        raise OrderFlowDataValidationError(
            f"{location}: bar_close must equal bar_open + 15 minutes"
        )
    available_at = _utc_timestamp(
        raw["available_at"],
        field=f"{location}.available_at",
    )
    if available_at < bar_close:
        raise OrderFlowDataValidationError(
            f"{location}: available_at cannot precede bar_close"
        )

    finalized = raw["finalized"]
    if not isinstance(finalized, bool) or not finalized:
        raise OrderFlowDataValidationError(
            f"{location}: finalized must be the boolean true"
        )
    volumes = {
        field: _finite_nonnegative_volume(
            raw[field],
            field=f"{location}.{field}",
        )
        for field in _VOLUME_FIELDS
    }
    supplied_checksum = (
        _require_sha256(raw["checksum"], field=f"{location}.checksum")
        if require_checksum
        else ""
    )
    event = AbsorptionEvent(
        schema_version=SCHEMA_VERSION,
        revision=EVENT_REVISION,
        timeframe=TIMEFRAME,
        source=source,
        venue=venue,
        source_symbol=source_symbol,
        symbol=symbol,
        polarity=polarity,
        bar_open=bar_open,
        bar_close=bar_close,
        available_at=available_at,
        finalized=True,
        checksum=supplied_checksum,
        **volumes,
    )
    expected_checksum = _canonical_hash(event.checksum_payload())
    if require_checksum and supplied_checksum != expected_checksum:
        raise OrderFlowDataValidationError(
            f"{location}: event checksum does not match normalized content"
        )
    if not require_checksum:
        event = AbsorptionEvent(
            **{
                **event.__dict__,
                "checksum": expected_checksum,
            }
        )
    return event


def event_checksum(record: Mapping[str, Any]) -> str:
    """Compute a schema-v1 event checksum for an exporter before sealing.

    ``record`` must contain every event field except ``checksum``.  Provenance
    lives in the manifest, so placeholder source/venue values do not affect the
    checksum.
    """

    event = _parse_event(
        record,
        source="checksum",
        venue="checksum",
        polarity=REQUIRED_POLARITY,
        symbol_mapping=None,
        location="event",
        require_checksum=False,
    )
    return event.checksum


def _read_events(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise OrderFlowDataValidationError(
                f"{path}: invalid UTF-8 records JSON"
            ) from exc
        if not isinstance(payload, list):
            raise OrderFlowDataValidationError(
                f"{path}: JSON sidecar must be a list of event objects"
            )
        return payload
    if suffix in {".parquet", ".pq"}:
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise OrderFlowDataValidationError(
                f"{path}: cannot read Parquet: {exc}"
            ) from exc
        return frame.to_dict(orient="records")
    raise OrderFlowDataValidationError(f"{path}: unsupported sidecar file type")


def _discover_data_files(base: Path) -> list[Path]:
    manifest_path = (base / MANIFEST_FILENAME).resolve()
    return sorted(
        (
            path
            for path in base.rglob("*")
            if path.is_file()
            and path.suffix.lower() in _DATA_SUFFIXES
            and path.resolve() != manifest_path
        ),
        key=lambda path: path.relative_to(base).as_posix(),
    )


def _load_normalized_events(
    paths: Iterable[Path],
    *,
    base: Path,
    source: str,
    venue: str,
    polarity: str,
    symbol_mapping: Mapping[str, str],
) -> Tuple[AbsorptionEvent, ...]:
    events: list[AbsorptionEvent] = []
    for path in paths:
        relative = path.relative_to(base).as_posix()
        rows = _read_events(path)
        for index, raw in enumerate(rows):
            events.append(
                _parse_event(
                    raw,
                    source=source,
                    venue=venue,
                    polarity=polarity,
                    symbol_mapping=symbol_mapping,
                    location=f"{relative}[{index}]",
                    require_checksum=True,
                )
            )
    if not events:
        raise OrderFlowDataValidationError("Absorption sidecar contains no events")

    events.sort(key=lambda event: (event.symbol, event.bar_open, event.source_symbol))
    seen: Dict[Tuple[str, pd.Timestamp], AbsorptionEvent] = {}
    for event in events:
        key = (event.symbol, event.bar_open)
        previous = seen.get(key)
        if previous is not None:
            qualifier = "conflicting " if previous.to_record() != event.to_record() else ""
            raise OrderFlowDataValidationError(
                f"Duplicate {qualifier}event for {event.symbol} "
                f"{event.bar_open.isoformat()}"
            )
        seen[key] = event
    return tuple(events)


def _content_hash(events: Iterable[AbsorptionEvent]) -> str:
    return _canonical_hash(
        {
            "events": [
                event.to_record()
                for event in sorted(
                    events,
                    key=lambda item: (
                        item.symbol,
                        item.bar_open,
                        item.source_symbol,
                    ),
                )
            ]
        }
    )


def build_manifest(
    root: str | Path,
    *,
    source: str,
    venue: str,
    symbol_mapping: Mapping[str, str],
    polarity: str = REQUIRED_POLARITY,
) -> Dict[str, Any]:
    """Build (but do not write) a deterministic manifest for valid shards."""

    base = Path(root).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Order-flow directory does not exist: {base}")
    source_text = _require_nonempty_text(source, field="source")
    venue_text = _require_nonempty_text(venue, field="venue")
    mapping = _normalize_mapping(symbol_mapping)
    if polarity != REQUIRED_POLARITY:
        raise OrderFlowDataValidationError(
            "polarity must explicitly declare aggressive buy-at-ask and sell-at-bid"
        )

    paths = _discover_data_files(base)
    if not paths:
        raise FileNotFoundError(f"No JSON/Parquet absorption files found under {base}")
    events = _load_normalized_events(
        paths,
        base=base,
        source=source_text,
        venue=venue_text,
        polarity=polarity,
        symbol_mapping=mapping,
    )
    files = [
        {
            "path": path.relative_to(base).as_posix(),
            "size": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]
    payload: Dict[str, Any] = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "timeframe": TIMEFRAME,
        "source": source_text,
        "venue": venue_text,
        "symbol_mapping": mapping,
        "polarity": polarity,
        "files": files,
        "content_sha256": _content_hash(events),
    }
    return {**payload, "manifest_sha256": _canonical_hash(payload)}


def write_manifest(
    root: str | Path,
    *,
    source: str,
    venue: str,
    symbol_mapping: Mapping[str, str],
    polarity: str = REQUIRED_POLARITY,
) -> Path:
    """Validate shards and atomically write their schema-v1 manifest."""

    base = Path(root).expanduser().resolve()
    manifest = build_manifest(
        base,
        source=source,
        venue=venue,
        symbol_mapping=symbol_mapping,
        polarity=polarity,
    )
    destination = base / MANIFEST_FILENAME
    temporary = base / f".{MANIFEST_FILENAME}.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


class AbsorptionEventDataset:
    """Immutable view of a sealed 15-minute footprint-event sidecar."""

    def __init__(
        self,
        root: Path,
        events: Iterable[AbsorptionEvent],
        manifest: Mapping[str, Any],
    ) -> None:
        self.root = Path(root)
        self._events = tuple(events)
        self._manifest = json.loads(json.dumps(manifest))
        self._by_key = {
            (event.symbol, event.bar_open): event
            for event in self._events
        }

    @classmethod
    def load(cls, root: str | Path) -> "AbsorptionEventDataset":
        base = Path(root).expanduser().resolve()
        if not base.exists() or not base.is_dir():
            raise FileNotFoundError(f"Order-flow directory does not exist: {base}")
        manifest_path = base / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise OrderFlowDataValidationError(
                f"{manifest_path}: sealed sidecar manifest is required"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise OrderFlowDataValidationError(
                f"{manifest_path}: invalid UTF-8 JSON manifest"
            ) from exc
        if not isinstance(manifest, Mapping):
            raise OrderFlowDataValidationError("manifest must be a JSON object")
        missing = sorted(_MANIFEST_FIELDS - set(manifest))
        extra = sorted(set(manifest) - _MANIFEST_FIELDS)
        if missing:
            raise OrderFlowDataValidationError(
                f"manifest is missing field(s): {missing}"
            )
        if extra:
            raise OrderFlowDataValidationError(
                f"manifest has unknown field(s): {extra}"
            )
        if manifest["schema"] != SCHEMA_NAME:
            raise OrderFlowDataValidationError(
                f"Unsupported order-flow schema: {manifest['schema']!r}"
            )
        version = manifest["schema_version"]
        if isinstance(version, bool) or version != SCHEMA_VERSION:
            raise OrderFlowDataValidationError(
                f"Unsupported order-flow schema_version: {version!r}"
            )
        if manifest["timeframe"] != TIMEFRAME:
            raise OrderFlowDataValidationError(
                f"Order-flow timeframe must be {TIMEFRAME!r}"
            )
        source = _require_nonempty_text(manifest["source"], field="source")
        venue = _require_nonempty_text(manifest["venue"], field="venue")
        mapping = _normalize_mapping(manifest["symbol_mapping"])
        if manifest["symbol_mapping"] != mapping:
            raise OrderFlowDataValidationError(
                "symbol_mapping must use normalized, sorted uppercase symbols"
            )
        polarity = manifest["polarity"]
        if polarity != REQUIRED_POLARITY:
            raise OrderFlowDataValidationError(
                "manifest polarity is unsupported or ambiguous"
            )
        content_sha256 = _require_sha256(
            manifest["content_sha256"],
            field="content_sha256",
        )
        supplied_manifest_hash = _require_sha256(
            manifest["manifest_sha256"],
            field="manifest_sha256",
        )
        manifest_payload = dict(manifest)
        manifest_payload.pop("manifest_sha256")
        if _canonical_hash(manifest_payload) != supplied_manifest_hash:
            raise OrderFlowDataValidationError(
                "manifest_sha256 does not match manifest content"
            )

        raw_files = manifest["files"]
        if not isinstance(raw_files, list) or not raw_files:
            raise OrderFlowDataValidationError("manifest files must be a non-empty list")
        listed_paths: list[Path] = []
        canonical_file_records: list[Dict[str, Any]] = []
        seen_relative: set[str] = set()
        for index, record in enumerate(raw_files):
            if not isinstance(record, Mapping) or set(record) != _FILE_FIELDS:
                raise OrderFlowDataValidationError(
                    f"manifest files[{index}] must contain exactly "
                    f"{sorted(_FILE_FIELDS)}"
                )
            relative = record["path"]
            if not isinstance(relative, str):
                raise OrderFlowDataValidationError(
                    f"manifest files[{index}].path must be a string"
                )
            pure = PurePosixPath(relative)
            if (
                pure.is_absolute()
                or not pure.parts
                or ".." in pure.parts
                or "\\" in relative
            ):
                raise OrderFlowDataValidationError(
                    f"manifest files[{index}].path is unsafe: {relative!r}"
                )
            if relative in seen_relative:
                raise OrderFlowDataValidationError(
                    f"manifest lists file more than once: {relative!r}"
                )
            seen_relative.add(relative)
            candidate = (base / Path(*pure.parts)).resolve()
            try:
                candidate.relative_to(base)
            except ValueError as exc:
                raise OrderFlowDataValidationError(
                    f"manifest path escapes sidecar root: {relative!r}"
                ) from exc
            if (
                candidate == manifest_path.resolve()
                or candidate.suffix.lower() not in _DATA_SUFFIXES
            ):
                raise OrderFlowDataValidationError(
                    f"manifest lists unsupported data file: {relative!r}"
                )
            if not candidate.is_file():
                raise OrderFlowDataValidationError(
                    f"manifest data file is missing: {relative!r}"
                )
            size = record["size"]
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise OrderFlowDataValidationError(
                    f"manifest files[{index}].size must be a non-negative integer"
                )
            expected_file_hash = _require_sha256(
                record["sha256"],
                field=f"manifest files[{index}].sha256",
            )
            actual_size = int(candidate.stat().st_size)
            actual_hash = _sha256_file(candidate)
            if actual_size != size or actual_hash != expected_file_hash:
                raise OrderFlowDataValidationError(
                    f"Sealed file content mismatch: {relative!r}"
                )
            listed_paths.append(candidate)
            canonical_file_records.append(
                {
                    "path": relative,
                    "size": size,
                    "sha256": expected_file_hash,
                }
            )
        if canonical_file_records != sorted(
            canonical_file_records,
            key=lambda item: item["path"],
        ):
            raise OrderFlowDataValidationError(
                "manifest files must be sorted by relative path"
            )

        discovered = {
            path.relative_to(base).as_posix()
            for path in _discover_data_files(base)
        }
        if discovered != seen_relative:
            missing_from_manifest = sorted(discovered - seen_relative)
            missing_from_disk = sorted(seen_relative - discovered)
            raise OrderFlowDataValidationError(
                "manifest file set does not seal the sidecar directory; "
                f"unlisted={missing_from_manifest}, absent={missing_from_disk}"
            )

        events = _load_normalized_events(
            listed_paths,
            base=base,
            source=source,
            venue=venue,
            polarity=polarity,
            symbol_mapping=mapping,
        )
        # Re-hash after parsing to fail if a shard changed between validation
        # and read.  This is intentionally strict even on a local filesystem.
        for record, path in zip(canonical_file_records, listed_paths):
            if (
                int(path.stat().st_size) != record["size"]
                or _sha256_file(path) != record["sha256"]
            ):
                raise OrderFlowDataValidationError(
                    f"Sealed file changed while loading: {record['path']!r}"
                )
        if _content_hash(events) != content_sha256:
            raise OrderFlowDataValidationError(
                "content_sha256 does not match normalized absorption events"
            )
        return cls(base, events, manifest)

    @property
    def events(self) -> Tuple[AbsorptionEvent, ...]:
        return self._events

    @property
    def symbols(self) -> Tuple[str, ...]:
        return tuple(sorted({event.symbol for event in self._events}))

    @property
    def manifest(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self._manifest))

    @property
    def manifest_sha256(self) -> str:
        return str(self._manifest["manifest_sha256"])

    @property
    def content_sha256(self) -> str:
        return str(self._manifest["content_sha256"])

    def event_asof(
        self,
        symbol: str,
        bar_open: Any,
        decision_time: Any,
    ) -> AbsorptionEvent | None:
        """Return one event only after both its bar close and availability.

        Missing, unfinished, delayed or future events all resolve to ``None``;
        callers must treat that as "absorption unavailable", never synthesize
        an OHLCV proxy.
        """

        normalized_symbol = _normalized_symbol(symbol, field="symbol")
        normalized_open = _aligned_bar_open(bar_open)
        decision = _utc_timestamp(decision_time, field="decision_time")
        event = self._by_key.get((normalized_symbol, normalized_open))
        if event is None:
            return None
        if event.bar_close > decision or event.available_at > decision:
            return None
        return event


__all__ = [
    "AbsorptionEvent",
    "AbsorptionEventDataset",
    "EVENT_REVISION",
    "MANIFEST_FILENAME",
    "OrderFlowDataValidationError",
    "REQUIRED_POLARITY",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "TIMEFRAME",
    "build_manifest",
    "event_checksum",
    "write_manifest",
]
