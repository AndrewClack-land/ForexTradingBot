"""Immutable FxPro DOM quote-pressure rejection sidecars for causal WFO."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import pandas as pd

from core.fxpro_quote_pressure import (
    LIQUIDITY_DATA_KIND,
    LIQUIDITY_EVENT_SCHEMA_VERSION,
    LIQUIDITY_MARKET_TYPE,
    LIQUIDITY_SOURCE,
    LIQUIDITY_TIMEFRAME,
    LIQUIDITY_VENUE,
    validate_liquidity_event,
)


SIDECAR_SCHEMA = "forexbot.fxpro-quote-pressure-rejection-15m"
SIDECAR_SCHEMA_VERSION = 1


class LiquidityDataValidationError(ValueError):
    """Raised when a sidecar is incomplete, mutable, or non-causal."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: Any, *, field: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except Exception as exc:
        raise LiquidityDataValidationError(
            f"{field} must be a timezone-aware UTC timestamp"
        ) from exc
    if (
        pd.isna(result)
        or result.tzinfo is None
        or result.utcoffset() != pd.Timedelta(0)
    ):
        raise LiquidityDataValidationError(
            f"{field} must be a timezone-aware UTC timestamp"
        )
    return result.tz_convert("UTC")


def _event_sort_key(event: Mapping[str, Any]) -> tuple[str, pd.Timestamp]:
    return str(event["symbol"]), pd.Timestamp(event["bar_open"])


def _content_hash(events: Iterable[Mapping[str, Any]]) -> str:
    checksums = [
        str(event["checksum"])
        for event in sorted(events, key=_event_sort_key)
    ]
    return _sha256_bytes(_canonical_bytes(checksums))


def _load_event_lines(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise LiquidityDataValidationError(
            f"cannot read liquidity file: {path}"
        ) from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LiquidityDataValidationError(
                f"{path}:{line_number}: invalid JSON"
            ) from exc
        event = validate_liquidity_event(raw)
        if event is None:
            raise LiquidityDataValidationError(
                f"{path}:{line_number}: invalid or untrusted FxPro event"
            )
        events.append(event)
    return events


def _validate_unique(
    events: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ordered = [dict(event) for event in sorted(events, key=_event_sort_key)]
    seen: set[tuple[str, pd.Timestamp]] = set()
    for event in ordered:
        identity = _event_sort_key(event)
        if identity in seen:
            raise LiquidityDataValidationError(
                "duplicate FxPro liquidity event identity: "
                f"{identity[0]} {identity[1].isoformat()}"
            )
        seen.add(identity)
    if not ordered:
        raise LiquidityDataValidationError(
            "FxPro Quote Pressure sidecar contains no events"
        )
    return ordered


class FxProQuotePressureEventDataset:
    """Verified immutable events addressable only as-of a decision time."""

    def __init__(
        self,
        *,
        root: Path,
        manifest: Mapping[str, Any],
        events: Iterable[Mapping[str, Any]],
    ) -> None:
        self.root = Path(root)
        self.manifest = dict(manifest)
        self.events = tuple(_validate_unique(events))
        self.manifest_sha256 = _sha256_bytes(_canonical_bytes(self.manifest))
        self._by_identity = {
            _event_sort_key(event): event for event in self.events
        }

    @classmethod
    def load(cls, root: str | Path) -> "FxProQuotePressureEventDataset":
        directory = Path(root)
        manifest_path = directory / "manifest.json"
        if not directory.is_dir() or not manifest_path.is_file():
            raise LiquidityDataValidationError(
                f"FxPro Quote Pressure manifest is required: {manifest_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LiquidityDataValidationError(
                "FxPro Quote Pressure manifest is invalid"
            ) from exc
        expected = {
            "schema": SIDECAR_SCHEMA,
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "event_schema_version": LIQUIDITY_EVENT_SCHEMA_VERSION,
            "timeframe": LIQUIDITY_TIMEFRAME,
            "source": LIQUIDITY_SOURCE,
            "venue": LIQUIDITY_VENUE,
            "market_type": LIQUIDITY_MARKET_TYPE,
            "data_kind": LIQUIDITY_DATA_KIND,
        }
        if not isinstance(manifest, dict):
            raise LiquidityDataValidationError("manifest must be an object")
        for field, value in expected.items():
            if manifest.get(field) != value:
                raise LiquidityDataValidationError(
                    f"manifest {field} does not match FxPro contract"
                )
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise LiquidityDataValidationError(
                "manifest files must be a non-empty list"
            )

        events: list[dict[str, Any]] = []
        listed: set[Path] = set()
        resolved_root = directory.resolve()
        for item in files:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise LiquidityDataValidationError("invalid manifest file entry")
            relative = Path(item["path"])
            path = (directory / relative).resolve()
            if resolved_root not in path.parents or path.suffix != ".jsonl":
                raise LiquidityDataValidationError("unsafe sidecar file path")
            if not path.is_file():
                raise LiquidityDataValidationError(f"missing sidecar file: {relative}")
            if path.stat().st_size != item.get("bytes"):
                raise LiquidityDataValidationError(f"size mismatch: {relative}")
            if _sha256_file(path) != item.get("sha256"):
                raise LiquidityDataValidationError(f"SHA-256 mismatch: {relative}")
            file_events = _load_event_lines(path)
            if len(file_events) != item.get("rows"):
                raise LiquidityDataValidationError(f"row mismatch: {relative}")
            listed.add(path)
            events.extend(file_events)

        unlisted = {
            path.resolve() for path in directory.rglob("*.jsonl")
        } - listed
        if unlisted:
            raise LiquidityDataValidationError(
                "sidecar contains JSONL files not sealed by the manifest"
            )
        ordered = _validate_unique(events)
        if len(ordered) != manifest.get("event_count"):
            raise LiquidityDataValidationError("manifest event_count mismatch")
        if _content_hash(ordered) != manifest.get("content_sha256"):
            raise LiquidityDataValidationError("manifest content_sha256 mismatch")
        symbols = sorted({str(event["symbol"]) for event in ordered})
        if symbols != manifest.get("symbols"):
            raise LiquidityDataValidationError("manifest symbols mismatch")
        return cls(root=directory, manifest=manifest, events=ordered)

    def event_asof(
        self,
        symbol: str,
        candle_open: Any,
        decision_time: Any,
    ) -> Optional[dict[str, Any]]:
        expected_open = _utc(candle_open, field="candle_open")
        decision = _utc(decision_time, field="decision_time")
        if (
            expected_open.minute % 15
            or expected_open.second
            or expected_open.microsecond
            or expected_open.nanosecond
        ):
            raise LiquidityDataValidationError(
                "candle_open must be on a 15-minute UTC boundary"
            )
        key = (str(symbol or "").strip().upper(), expected_open)
        event = self._by_identity.get(key)
        if event is None:
            return None
        if (
            pd.Timestamp(event["bar_close"]) > decision
            or pd.Timestamp(event["available_at"]) > decision
        ):
            return None
        return dict(event)


def seal_recorded_events(
    recording_root: str | Path,
    output_root: str | Path,
) -> Path:
    """Seal recorder event JSONL into a new immutable WFO sidecar."""

    source = Path(recording_root)
    event_root = source / "events" if (source / "events").is_dir() else source
    paths = sorted(event_root.rglob("*.jsonl")) if event_root.is_dir() else []
    events: list[dict[str, Any]] = []
    for path in paths:
        events.extend(_load_event_lines(path))
    ordered = _validate_unique(events)

    destination = Path(output_root)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing sidecar: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{uuid.uuid4().hex}"
    )
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        event_path = temporary / "events.jsonl"
        with event_path.open("w", encoding="utf-8", newline="\n") as handle:
            for event in ordered:
                handle.write(_canonical_bytes(event).decode("utf-8") + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        manifest = {
            "schema": SIDECAR_SCHEMA,
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "event_schema_version": LIQUIDITY_EVENT_SCHEMA_VERSION,
            "timeframe": LIQUIDITY_TIMEFRAME,
            "source": LIQUIDITY_SOURCE,
            "venue": LIQUIDITY_VENUE,
            "market_type": LIQUIDITY_MARKET_TYPE,
            "data_kind": LIQUIDITY_DATA_KIND,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "symbols": sorted({str(event["symbol"]) for event in ordered}),
            "event_count": len(ordered),
            "content_sha256": _content_hash(ordered),
            "files": [
                {
                    "path": event_path.name,
                    "bytes": event_path.stat().st_size,
                    "rows": len(ordered),
                    "sha256": _sha256_file(event_path),
                }
            ],
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(_canonical_bytes(manifest) + b"\n")
        FxProQuotePressureEventDataset.load(temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


__all__ = [
    "FxProQuotePressureEventDataset",
    "LiquidityDataValidationError",
    "SIDECAR_SCHEMA",
    "seal_recorded_events",
]
