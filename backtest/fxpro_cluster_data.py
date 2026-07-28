"""Immutable Quantower/FxPro cluster-proxy sidecars for causal research."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd

from core.fxpro_cluster_rejection import (
    CLUSTER_CLASSIFICATION,
    CLUSTER_DATA_KIND,
    CLUSTER_EVENT_SCHEMA_VERSION,
    CLUSTER_MARKET_TYPE,
    CLUSTER_SOURCE,
    CLUSTER_TIMEFRAME,
    CLUSTER_VENUE,
    canonical_hash,
    seal_cluster_event,
    validate_cluster_event,
)
def _validate_diagnostic_export(
    *args: Any, **kwargs: Any
) -> Any:
    """Load the diagnostic-only validator only while sealing research data."""
    from tools.validate_quantower_cluster_diagnostic import validate_export

    return validate_export(*args, **kwargs)


SIDECAR_SCHEMA = "forexbot.fxpro-cluster-rejection-15m"
SIDECAR_SCHEMA_VERSION = 1


class ClusterDataValidationError(ValueError):
    """Raised when a cluster sidecar is incomplete, mutable, or non-causal."""


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
        raise ClusterDataValidationError(
            f"{field} must be a timezone-aware UTC timestamp"
        ) from exc
    if (
        pd.isna(result)
        or result.tzinfo is None
        or result.utcoffset() != pd.Timedelta(0)
    ):
        raise ClusterDataValidationError(
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
        raise ClusterDataValidationError(
            f"cannot read cluster file: {path}"
        ) from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ClusterDataValidationError(
                f"{path}:{line_number}: invalid JSON"
            ) from exc
        if validate_cluster_event(raw) is None:
            raise ClusterDataValidationError(
                f"{path}:{line_number}: invalid or untrusted cluster event"
            )
        # Preserve the canonical JSON representation. Returning normalized
        # pandas timestamps here would make a second checksum validation
        # non-serializable and diverge from the sealed bytes.
        events.append(dict(raw))
    return events


def _validate_unique(
    events: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ordered = [dict(event) for event in sorted(events, key=_event_sort_key)]
    seen: set[tuple[str, pd.Timestamp]] = set()
    for event in ordered:
        identity = _event_sort_key(event)
        if identity in seen:
            raise ClusterDataValidationError(
                "duplicate cluster event identity: "
                f"{identity[0]} {identity[1].isoformat()}"
            )
        seen.add(identity)
    if not ordered:
        raise ClusterDataValidationError(
            "FxPro Cluster Rejection sidecar contains no events"
        )
    return ordered


class FxProClusterEventDataset:
    """Verified immutable cluster events addressable only as-of decision time."""

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
        self.research_only = bool(self.manifest.get("research_only"))
        self._by_identity = {
            _event_sort_key(event): event for event in self.events
        }

    @classmethod
    def load(cls, root: str | Path) -> "FxProClusterEventDataset":
        directory = Path(root)
        manifest_path = directory / "manifest.json"
        if not directory.is_dir() or not manifest_path.is_file():
            raise ClusterDataValidationError(
                f"FxPro Cluster manifest is required: {manifest_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ClusterDataValidationError(
                "FxPro Cluster manifest is invalid"
            ) from exc
        expected = {
            "schema": SIDECAR_SCHEMA,
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "event_schema_version": CLUSTER_EVENT_SCHEMA_VERSION,
            "timeframe": CLUSTER_TIMEFRAME,
            "source": CLUSTER_SOURCE,
            "venue": CLUSTER_VENUE,
            "market_type": CLUSTER_MARKET_TYPE,
            "data_kind": CLUSTER_DATA_KIND,
            "classification": CLUSTER_CLASSIFICATION,
            "execution_proof": False,
            "aggressor_polarity_proven": False,
        }
        if not isinstance(manifest, dict):
            raise ClusterDataValidationError("manifest must be an object")
        for field, value in expected.items():
            if manifest.get(field) != value:
                raise ClusterDataValidationError(
                    f"manifest {field} does not match cluster-proxy contract"
                )
        if not isinstance(manifest.get("research_only"), bool):
            raise ClusterDataValidationError(
                "manifest research_only must be boolean"
            )
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise ClusterDataValidationError(
                "manifest files must be a non-empty list"
            )

        events: list[dict[str, Any]] = []
        listed: set[Path] = set()
        resolved_root = directory.resolve()
        for item in files:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise ClusterDataValidationError("invalid manifest file entry")
            relative = Path(item["path"])
            path = (directory / relative).resolve()
            if resolved_root not in path.parents or path.suffix != ".jsonl":
                raise ClusterDataValidationError("unsafe sidecar file path")
            if not path.is_file():
                raise ClusterDataValidationError(
                    f"missing sidecar file: {relative}"
                )
            if path.stat().st_size != item.get("bytes"):
                raise ClusterDataValidationError(f"size mismatch: {relative}")
            if _sha256_file(path) != item.get("sha256"):
                raise ClusterDataValidationError(
                    f"SHA-256 mismatch: {relative}"
                )
            file_events = _load_event_lines(path)
            if len(file_events) != item.get("rows"):
                raise ClusterDataValidationError(f"row mismatch: {relative}")
            listed.add(path)
            events.extend(file_events)

        unlisted = {
            path.resolve() for path in directory.rglob("*.jsonl")
        } - listed
        if unlisted:
            raise ClusterDataValidationError(
                "sidecar contains JSONL files not sealed by the manifest"
            )
        ordered = _validate_unique(events)
        if len(ordered) != manifest.get("event_count"):
            raise ClusterDataValidationError("manifest event_count mismatch")
        if _content_hash(ordered) != manifest.get("content_sha256"):
            raise ClusterDataValidationError(
                "manifest content_sha256 mismatch"
            )
        symbols = sorted({str(event["symbol"]) for event in ordered})
        if symbols != manifest.get("symbols"):
            raise ClusterDataValidationError("manifest symbols mismatch")
        modes = sorted({str(event["availability_mode"]) for event in ordered})
        if modes != manifest.get("availability_modes"):
            raise ClusterDataValidationError(
                "manifest availability_modes mismatch"
            )
        research_only = "research_assumption" in modes
        if research_only != manifest.get("research_only"):
            raise ClusterDataValidationError(
                "manifest research_only does not match event provenance"
            )
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
            raise ClusterDataValidationError(
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


def _read_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClusterDataValidationError(f"{label} is invalid") from exc
    if not isinstance(value, Mapping):
        raise ClusterDataValidationError(f"{label} must be an object")
    return value


def _event_from_diagnostic_bar(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    bar: Mapping[str, Any],
    delay: pd.Timedelta,
    availability_evidence: str,
) -> dict[str, Any]:
    instrument = manifest["instrument"]
    bar_close = _utc(bar["bar_close"], field="bar_close")
    total = {
        field: bar["total"][field]
        for field in (
            "volume",
            "trades",
            "buy_volume",
            "sell_volume",
            "buy_trades",
            "sell_trades",
            "delta",
        )
    }
    levels = [
        {
            field: raw_level[field]
            for field in (
                "price",
                "price_ticks",
                "volume",
                "trades",
                "buy_volume",
                "sell_volume",
                "buy_trades",
                "sell_trades",
                "delta",
            )
        }
        for raw_level in bar["price_levels"]
    ]
    source_bar_hash = canonical_hash(bar)
    payload = {
        "schema_version": CLUSTER_EVENT_SCHEMA_VERSION,
        "symbol": str(bar["symbol"]).strip().upper(),
        "timeframe": CLUSTER_TIMEFRAME,
        "bar_open": _utc(bar["bar_open"], field="bar_open").isoformat(),
        "bar_close": bar_close.isoformat(),
        "available_at": (bar_close + delay).isoformat(),
        "availability_mode": "research_assumption",
        "availability_evidence": availability_evidence,
        "source": CLUSTER_SOURCE,
        "venue": CLUSTER_VENUE,
        "market_type": CLUSTER_MARKET_TYPE,
        "data_kind": CLUSTER_DATA_KIND,
        "classification": CLUSTER_CLASSIFICATION,
        "execution_proof": False,
        "aggressor_polarity_proven": False,
        "finalized": True,
        "tick_size": instrument["tick_size"],
        "ohlc": {
            field: bar["ohlc"][field]
            for field in ("open", "high", "low", "close")
        },
        "total": total,
        "price_levels": levels,
        "source_bar_hash": source_bar_hash,
        "source_manifest_sha256": manifest_sha256,
    }
    event = seal_cluster_event(payload)
    if validate_cluster_event(event) is None:
        raise ClusterDataValidationError(
            f"converted cluster event is invalid: {payload['symbol']} "
            f"{payload['bar_open']}"
        )
    return event


def seal_quantower_diagnostics(
    exports: Sequence[str | Path],
    output_root: str | Path,
    *,
    availability_delay_ms: int,
    availability_evidence: str,
    exporter_dll: str | Path | None = None,
) -> Path:
    """Convert validated diagnostics into an explicit research-only sidecar.

    Diagnostic history does not prove when a historical cluster was originally
    available.  The caller must therefore provide a nonnegative assumed delay
    and a non-empty evidence/assumption note.  The resulting sidecar is marked
    ``research_only`` and live detection rejects it by default.
    """

    if (
        isinstance(availability_delay_ms, bool)
        or int(availability_delay_ms) < 0
    ):
        raise ClusterDataValidationError(
            "availability_delay_ms must be a nonnegative integer"
        )
    evidence = str(availability_evidence or "").strip()
    if len(evidence) < 12:
        raise ClusterDataValidationError(
            "availability_evidence must document the research assumption"
        )
    if not exports:
        raise ClusterDataValidationError(
            "at least one Quantower diagnostic export is required"
        )
    delay = pd.Timedelta(milliseconds=int(availability_delay_ms))

    events: list[dict[str, Any]] = []
    source_exports: list[dict[str, Any]] = []
    for raw_root in exports:
        root = Path(raw_root).resolve()
        summary = _validate_diagnostic_export(
            root, exporter_dll=exporter_dll
        )
        manifest_path = root / "manifest.json"
        bars_path = root / "bars.json"
        manifest_raw = manifest_path.read_bytes()
        manifest_sha256 = _sha256_bytes(manifest_raw)
        manifest = _read_mapping(manifest_path, label=str(manifest_path))
        bars_document = _read_mapping(bars_path, label=str(bars_path))
        if (
            manifest.get("semantics", {}).get("classification")
            != "quantower_tick_reconstructed_bidask_history_available"
        ):
            raise ClusterDataValidationError(
                "only reconstructed FxPro BidAsk tick clusters are accepted"
            )
        bars = bars_document.get("bars")
        if not isinstance(bars, Sequence) or isinstance(bars, (str, bytes)):
            raise ClusterDataValidationError("bars.json.bars must be an array")
        events.extend(
            _event_from_diagnostic_bar(
                manifest=manifest,
                manifest_sha256=manifest_sha256,
                bar=bar,
                delay=delay,
                availability_evidence=evidence,
            )
            for bar in bars
        )
        source_exports.append(
            {
                "path": str(root),
                "export_id": summary.export_id,
                "day_utc": summary.day_utc,
                "bars": summary.bars,
                "price_levels": summary.price_levels,
                "missing_slots": summary.missing_slots,
                "content_sha256": summary.content_sha256,
                "manifest_sha256": manifest_sha256,
                "exporter_binary_sha256": (
                    summary.exporter_binary_sha256
                ),
            }
        )
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
            "event_schema_version": CLUSTER_EVENT_SCHEMA_VERSION,
            "timeframe": CLUSTER_TIMEFRAME,
            "source": CLUSTER_SOURCE,
            "venue": CLUSTER_VENUE,
            "market_type": CLUSTER_MARKET_TYPE,
            "data_kind": CLUSTER_DATA_KIND,
            "classification": CLUSTER_CLASSIFICATION,
            "execution_proof": False,
            "aggressor_polarity_proven": False,
            "research_only": True,
            "availability_modes": ["research_assumption"],
            "availability_delay_ms": int(availability_delay_ms),
            "availability_evidence": evidence,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "symbols": sorted({str(event["symbol"]) for event in ordered}),
            "event_count": len(ordered),
            "content_sha256": _content_hash(ordered),
            "source_exports": source_exports,
            "files": [
                {
                    "path": event_path.name,
                    "bytes": event_path.stat().st_size,
                    "rows": len(ordered),
                    "sha256": _sha256_file(event_path),
                }
            ],
        }
        (temporary / "manifest.json").write_bytes(
            _canonical_bytes(manifest) + b"\n"
        )
        FxProClusterEventDataset.load(temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


__all__ = [
    "ClusterDataValidationError",
    "FxProClusterEventDataset",
    "SIDECAR_SCHEMA",
    "seal_quantower_diagnostics",
]
