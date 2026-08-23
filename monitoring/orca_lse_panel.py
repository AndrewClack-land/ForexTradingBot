"""Materialize a causal four-asset ORCA panel from sealed LSE snapshots.

The materializer is deliberately read-only with respect to every source
snapshot.  It verifies their stored manifests and every declared file before
reading only the D1 ``close`` and ``bar_close_time`` columns.  The resulting
wide panel is an exact, no-fill inner join in the fixed ORCA-4 symbol order.

History is assembled from a three-symbol FX snapshot plus a GOLD-only
snapshot, so neither has to be downloaded again.  An optional four-symbol
``tail`` snapshot then extends that history to the present.  The tail is
spliced only when it overlaps the history and every overlapping bar agrees
bit-for-bit on all four closes; any disagreement is a fail-closed error rather
than a silent preference for one source.  Nothing is ever forward-filled,
interpolated, or reconciled by tolerance.

Publication is a single directory rename containing ``prices.parquet`` and a
self-hashed lineage ``manifest.json``.  Existing outputs are never replaced.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from typing import Any, Optional

import numpy as np
import pandas as pd

from core.orca_spectral import OrcaValidationError, validate_d1_price_panel


ORCA_LSE_PANEL_SCHEMA = "orca-lse-orca4-panel-v1"
ORCA_LSE_UNIVERSE = ("EURUSD", "GBPUSD", "USDCAD", "GOLD")
ORCA_LSE_FX_UNIVERSE = ORCA_LSE_UNIVERSE[:3]
ORCA_LSE_GOLD_UNIVERSE = ORCA_LSE_UNIVERSE[3:]
DEFAULT_MINIMUM_ALIGNED_ROWS = 372
DEFAULT_MINIMUM_OVERLAP_ROWS = 5
ORCA_LSE_PANEL_FILENAME = "prices.parquet"
ORCA_LSE_MANIFEST_FILENAME = "manifest.json"

_LSE_PROVIDER = "london_strategic_edge"
_HASHED_METADATA_FILES = {"source_manifest.json", "environment.freeze.txt"}
_IGNORED_METADATA_FILES = {"manifest.json", *_HASHED_METADATA_FILES}
_CANDLE_SUFFIXES = {".json", ".parquet", ".pq"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class OrcaLsePanelError(ValueError):
    """Fail-closed source-validation or publication error."""


def _positive_int(value: Any, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise OrcaLsePanelError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class OrcaLsePanelConfig:
    fx_snapshot: Path
    gold_snapshot: Path
    output_dir: Path
    tail_snapshot: Optional[Path] = None
    minimum_aligned_rows: int = DEFAULT_MINIMUM_ALIGNED_ROWS
    minimum_overlap_rows: int = DEFAULT_MINIMUM_OVERLAP_ROWS

    def __post_init__(self) -> None:
        object.__setattr__(self, "fx_snapshot", Path(self.fx_snapshot))
        object.__setattr__(self, "gold_snapshot", Path(self.gold_snapshot))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.tail_snapshot is not None:
            object.__setattr__(self, "tail_snapshot", Path(self.tail_snapshot))
        object.__setattr__(
            self,
            "minimum_aligned_rows",
            _positive_int(
                self.minimum_aligned_rows,
                name="minimum_aligned_rows",
            ),
        )
        object.__setattr__(
            self,
            "minimum_overlap_rows",
            _positive_int(
                self.minimum_overlap_rows,
                name="minimum_overlap_rows",
            ),
        )


@dataclass(frozen=True)
class _VerifiedSnapshot:
    role: str
    root: Path
    manifest: Mapping[str, Any]
    source_manifest: Mapping[str, Any]
    manifest_file_sha256: str
    source_manifest_sha256: str
    d1_entries: Mapping[str, Mapping[str, Any]]


def _utc(value: Any, *, name: str) -> datetime:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise OrcaLsePanelError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    token = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(token):
        raise OrcaLsePanelError(f"{name} must be a lowercase SHA-256")
    return token


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OrcaLsePanelError(f"{name} is missing or is not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OrcaLsePanelError(f"{name} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise OrcaLsePanelError(f"{name} must contain a JSON object: {path}")
    return payload


def _declared_path(root: Path, raw: Any, *, name: str) -> tuple[str, Path]:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise OrcaLsePanelError(f"{name} must be a non-empty POSIX relative path")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise OrcaLsePanelError(f"{name} must remain inside its snapshot")
    if relative.parts and ":" in relative.parts[0]:
        raise OrcaLsePanelError(f"{name} must not contain a drive prefix")
    unresolved = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise OrcaLsePanelError(f"{name} must not traverse symlinks: {raw}")
    resolved = unresolved.resolve()
    if not resolved.is_relative_to(root):
        raise OrcaLsePanelError(f"{name} escapes its snapshot: {raw}")
    return relative.as_posix(), resolved


def _snapshot_actual_files(root: Path) -> set[str]:
    actual: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if (
            path.suffix.lower() in _CANDLE_SUFFIXES
            and lower_name not in _IGNORED_METADATA_FILES
        ) or lower_name in _HASHED_METADATA_FILES:
            if path.is_symlink():
                raise OrcaLsePanelError(
                    f"sealed snapshot must not contain symlinked inputs: {path}"
                )
            actual.add(path.relative_to(root).as_posix())
    return actual


def _verify_manifest_files(
    root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    stored_hash = _require_sha256(
        manifest.get("manifest_sha256"),
        name="manifest.manifest_sha256",
    )
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    if _canonical_hash(unhashed) != stored_hash:
        raise OrcaLsePanelError("stored manifest self-hash does not match its payload")

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise OrcaLsePanelError("stored manifest.files must be a non-empty list")
    declared: dict[str, Mapping[str, Any]] = {}
    for position, raw_entry in enumerate(raw_files):
        if not isinstance(raw_entry, dict):
            raise OrcaLsePanelError(f"manifest.files[{position}] must be an object")
        relative, path = _declared_path(
            root,
            raw_entry.get("path"),
            name=f"manifest.files[{position}].path",
        )
        if relative in declared:
            raise OrcaLsePanelError(f"duplicate manifest file path: {relative}")
        if not path.is_file():
            raise OrcaLsePanelError(f"manifest-declared file is missing: {relative}")
        try:
            expected_size = int(raw_entry.get("size"))
        except (TypeError, ValueError) as exc:
            raise OrcaLsePanelError(
                f"manifest file size is invalid: {relative}"
            ) from exc
        if expected_size < 0 or path.stat().st_size != expected_size:
            raise OrcaLsePanelError(f"manifest file size mismatch: {relative}")
        expected_hash = _require_sha256(
            raw_entry.get("sha256"),
            name=f"manifest sha256 for {relative}",
        )
        if _sha256_file(path) != expected_hash:
            raise OrcaLsePanelError(f"manifest file SHA-256 mismatch: {relative}")
        declared[relative] = raw_entry

    actual = _snapshot_actual_files(root)
    if set(declared) != actual:
        mismatch = sorted(set(declared) ^ actual)
        raise OrcaLsePanelError(
            f"stored manifest file inventory mismatch: {mismatch}"
        )
    return declared


def _source_file_entries(
    root: Path,
    source_manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    raw_files = source_manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise OrcaLsePanelError("source_manifest.files must be a non-empty list")
    entries: dict[str, Mapping[str, Any]] = {}
    for position, raw_entry in enumerate(raw_files):
        if not isinstance(raw_entry, dict):
            raise OrcaLsePanelError(
                f"source_manifest.files[{position}] must be an object"
            )
        relative, path = _declared_path(
            root,
            raw_entry.get("path"),
            name=f"source_manifest.files[{position}].path",
        )
        if relative in entries:
            raise OrcaLsePanelError(f"duplicate source manifest path: {relative}")
        if not path.is_file():
            raise OrcaLsePanelError(f"source-manifest file is missing: {relative}")
        try:
            expected_size = int(raw_entry.get("bytes"))
        except (TypeError, ValueError) as exc:
            raise OrcaLsePanelError(
                f"source manifest byte count is invalid: {relative}"
            ) from exc
        if expected_size < 0 or path.stat().st_size != expected_size:
            raise OrcaLsePanelError(
                f"source manifest file size mismatch: {relative}"
            )
        expected_hash = _require_sha256(
            raw_entry.get("sha256"),
            name=f"source manifest sha256 for {relative}",
        )
        if _sha256_file(path) != expected_hash:
            raise OrcaLsePanelError(
                f"source manifest file SHA-256 mismatch: {relative}"
            )
        entries[relative] = raw_entry
    return entries


def _exact_manifest_symbols(
    source_manifest: Mapping[str, Any],
    *,
    expected_symbols: Sequence[str],
) -> None:
    raw_symbols = source_manifest.get("symbols")
    if not isinstance(raw_symbols, list):
        raise OrcaLsePanelError("source_manifest.symbols must be a list")
    symbols: list[str] = []
    for position, item in enumerate(raw_symbols):
        if not isinstance(item, dict):
            raise OrcaLsePanelError(
                f"source_manifest.symbols[{position}] must be an object"
            )
        symbol = str(item.get("bot_symbol") or "").strip().upper()
        if not symbol:
            raise OrcaLsePanelError(
                f"source_manifest.symbols[{position}].bot_symbol is missing"
            )
        symbols.append(symbol)
    if tuple(symbols) != tuple(expected_symbols):
        raise OrcaLsePanelError(
            "source snapshot symbol order/universe mismatch: "
            f"expected={tuple(expected_symbols)}, received={tuple(symbols)}"
        )


def _d1_entries(
    entries: Mapping[str, Mapping[str, Any]],
    *,
    expected_symbols: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for entry in entries.values():
        timeframe = str(entry.get("timeframe") or "").strip().lower()
        if timeframe != "1d":
            continue
        symbol = str(entry.get("symbol") or "").strip().upper()
        if not symbol or symbol in result:
            raise OrcaLsePanelError("D1 source entries need one unique symbol each")
        result[symbol] = entry
    if tuple(symbol for symbol in expected_symbols if symbol in result) != tuple(
        expected_symbols
    ) or set(result) != set(expected_symbols):
        raise OrcaLsePanelError(
            "D1 source universe mismatch: "
            f"expected={tuple(expected_symbols)}, received={tuple(result)}"
        )
    return {symbol: result[symbol] for symbol in expected_symbols}


def _verify_snapshot(
    path: Path,
    *,
    role: str,
    expected_symbols: Sequence[str],
) -> _VerifiedSnapshot:
    raw_root = path.expanduser()
    if raw_root.is_symlink():
        raise OrcaLsePanelError(f"{role} snapshot root must not be a symlink")
    root = raw_root.resolve()
    if not root.is_dir():
        raise OrcaLsePanelError(f"{role} snapshot directory does not exist: {root}")

    manifest_path = root / ORCA_LSE_MANIFEST_FILENAME
    manifest = _load_json_object(manifest_path, name=f"{role} manifest")
    declared = _verify_manifest_files(root, manifest)
    source_relative = "source_manifest.json"
    if source_relative not in declared:
        raise OrcaLsePanelError(
            f"{role} manifest does not bind source_manifest.json"
        )
    source_path = root / source_relative
    source_manifest = _load_json_object(
        source_path,
        name=f"{role} source manifest",
    )
    if source_manifest.get("provider") != _LSE_PROVIDER:
        raise OrcaLsePanelError(
            f"{role} source provider must be {_LSE_PROVIDER!r}"
        )
    _exact_manifest_symbols(
        source_manifest,
        expected_symbols=expected_symbols,
    )
    source_entries = _source_file_entries(root, source_manifest)
    primary_paths = {
        relative
        for relative, entry in declared.items()
        if entry.get("kind") != "metadata"
    }
    if set(source_entries) != primary_paths:
        mismatch = sorted(set(source_entries) ^ primary_paths)
        raise OrcaLsePanelError(
            f"{role} source/stored manifest inventory mismatch: {mismatch}"
        )
    d1 = _d1_entries(source_entries, expected_symbols=expected_symbols)
    return _VerifiedSnapshot(
        role=role,
        root=root,
        manifest=manifest,
        source_manifest=source_manifest,
        manifest_file_sha256=_sha256_file(manifest_path),
        source_manifest_sha256=_sha256_file(source_path),
        d1_entries=d1,
    )


def _load_d1_close(
    snapshot: _VerifiedSnapshot,
    *,
    symbol: str,
) -> pd.Series:
    entry = snapshot.d1_entries[symbol]
    _, path = _declared_path(
        snapshot.root,
        entry.get("path"),
        name=f"{snapshot.role} {symbol} D1 path",
    )
    if path.suffix.lower() not in {".parquet", ".pq"}:
        raise OrcaLsePanelError(f"{symbol} D1 source must be Parquet")
    try:
        frame = pd.read_parquet(
            path,
            columns=["symbol", "tf", "bar_close_time", "close"],
        )
    except Exception as exc:
        raise OrcaLsePanelError(f"cannot read {symbol} D1 source: {path}") from exc
    if frame.empty:
        raise OrcaLsePanelError(f"{symbol} D1 source is empty")

    symbols = frame["symbol"].astype(str).str.strip().str.upper()
    if not symbols.eq(symbol).all():
        raise OrcaLsePanelError(f"{symbol} D1 file contains another symbol")
    timeframes = frame["tf"].astype(str).str.strip().str.lower()
    if not timeframes.eq("1d").all():
        raise OrcaLsePanelError(f"{symbol} D1 file contains another timeframe")

    raw_close_times = frame["bar_close_time"]
    try:
        close_times = pd.DatetimeIndex(raw_close_times)
    except (TypeError, ValueError) as exc:
        raise OrcaLsePanelError(f"{symbol} has invalid bar_close_time") from exc
    if close_times.tz is None:
        raise OrcaLsePanelError(f"{symbol} bar_close_time must be timezone-aware")
    close_times = close_times.tz_convert("UTC")
    if (
        close_times.hasnans
        or close_times.has_duplicates
        or not close_times.is_monotonic_increasing
    ):
        raise OrcaLsePanelError(
            f"{symbol} bar_close_time must be unique and strictly increasing"
        )
    if close_times.normalize().has_duplicates:
        raise OrcaLsePanelError(f"{symbol} has more than one D1 close per UTC date")

    try:
        values = frame["close"].to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise OrcaLsePanelError(f"{symbol} close values must be numeric") from exc
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise OrcaLsePanelError(f"{symbol} close values must be finite and positive")

    try:
        declared_rows = int(entry.get("rows"))
    except (TypeError, ValueError) as exc:
        raise OrcaLsePanelError(f"{symbol} source row count is invalid") from exc
    if declared_rows != len(frame):
        raise OrcaLsePanelError(f"{symbol} source row count does not match Parquet")

    result = pd.Series(values, index=close_times, name=symbol, dtype=float)
    result.index.name = "bar_close_time"
    return result


def _aligned_panel(
    series: Sequence[pd.Series],
    *,
    role: str,
) -> pd.DataFrame:
    """Inner-join one snapshot group into the fixed ORCA-4 column order."""

    panel = pd.concat(list(series), axis=1, join="inner").sort_index()
    panel = panel.loc[:, list(ORCA_LSE_UNIVERSE)]
    if panel.isna().any().any():
        raise OrcaLsePanelError(
            f"inner-joined {role} panel unexpectedly contains missing data"
        )
    if panel.empty:
        raise OrcaLsePanelError(f"{role} sources share no aligned D1 bar")
    return panel


def _first_overlap_disagreement(
    history: pd.DataFrame,
    tail: pd.DataFrame,
    overlap: pd.DatetimeIndex,
) -> str:
    """Name one concrete disagreeing bar instead of dumping the whole overlap."""

    for timestamp in overlap:
        for symbol in ORCA_LSE_UNIVERSE:
            left = float(history.at[timestamp, symbol])
            right = float(tail.at[timestamp, symbol])
            if left != right:
                return (
                    f"{symbol} at {pd.Timestamp(timestamp).isoformat()}: "
                    f"history={left!r}, tail={right!r}"
                )
    return "no disagreement found"  # pragma: no cover - guarded by caller


def _splice_tail(
    history: pd.DataFrame,
    tail: pd.DataFrame,
    *,
    minimum_overlap_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extend history with a tail only where the shared bars agree exactly.

    The tail is an independently downloaded snapshot of the same provider
    series, so an overlapping bar must reproduce the sealed history bit for
    bit.  Anything else means the two imports disagree about the past, which
    is a data-integrity failure rather than something to average or prefer
    away.  Only bars strictly after the sealed history are appended.
    """

    history_start = history.index[0]
    history_end = history.index[-1]
    tail_start = tail.index[0]
    tail_end = tail.index[-1]

    overlap = history.index.intersection(tail.index)
    if len(overlap) < minimum_overlap_rows:
        raise OrcaLsePanelError(
            f"tail must share at least {minimum_overlap_rows} exact D1 bars with "
            f"the sealed history; found {len(overlap)}. History covers "
            f"{pd.Timestamp(history_start).isoformat()}.."
            f"{pd.Timestamp(history_end).isoformat()} and the tail covers "
            f"{pd.Timestamp(tail_start).isoformat()}.."
            f"{pd.Timestamp(tail_end).isoformat()}"
        )

    left = history.loc[overlap].to_numpy(dtype=float, copy=False)
    right = tail.loc[overlap].to_numpy(dtype=float, copy=False)
    if not np.array_equal(left, right):
        raise OrcaLsePanelError(
            "tail disagrees with the sealed history on an overlapping D1 close; "
            "refusing to splice two inconsistent imports ("
            + _first_overlap_disagreement(history, tail, overlap)
            + ")"
        )

    extension = tail.loc[tail.index > history_end]
    if extension.empty:
        raise OrcaLsePanelError(
            "tail adds no D1 bar after "
            f"{pd.Timestamp(history_end).isoformat()}; re-import the tail "
            "through a later end date instead of republishing stale history"
        )

    combined = pd.concat([history, extension])
    if combined.index.has_duplicates or not combined.index.is_monotonic_increasing:
        raise OrcaLsePanelError("spliced panel index must stay unique and increasing")

    within_history = tail.index[
        (tail.index >= history_start) & (tail.index <= history_end)
    ]
    audit = {
        "tail_rows": int(len(tail)),
        "tail_start_utc": pd.Timestamp(tail_start).isoformat(),
        "tail_end_utc": pd.Timestamp(tail_end).isoformat(),
        "overlap_rows": int(len(overlap)),
        "overlap_start_utc": pd.Timestamp(overlap[0]).isoformat(),
        "overlap_end_utc": pd.Timestamp(overlap[-1]).isoformat(),
        "rows_added_after_history": int(len(extension)),
        "history_end_utc": pd.Timestamp(history_end).isoformat(),
        # A tail bar inside the sealed span that the history does not carry is
        # dropped rather than inserted: the sealed inner join already decided
        # that date lacks a complete four-asset observation.
        "tail_rows_inside_history_not_in_history": int(
            len(within_history) - len(overlap)
        ),
    }
    return combined, audit


def build_orca_lse_panel(
    fx_snapshot: Path | str,
    gold_snapshot: Path | str,
    *,
    tail_snapshot: Optional[Path | str] = None,
    as_of_utc: datetime,
    minimum_aligned_rows: int = DEFAULT_MINIMUM_ALIGNED_ROWS,
    minimum_overlap_rows: int = DEFAULT_MINIMUM_OVERLAP_ROWS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify the sealed snapshots and build an exact no-fill ORCA-4 panel."""

    as_of = _utc(as_of_utc, name="as_of_utc")
    minimum = _positive_int(minimum_aligned_rows, name="minimum_aligned_rows")
    minimum_overlap = _positive_int(
        minimum_overlap_rows,
        name="minimum_overlap_rows",
    )

    fx = _verify_snapshot(
        Path(fx_snapshot),
        role="fx",
        expected_symbols=ORCA_LSE_FX_UNIVERSE,
    )
    gold = _verify_snapshot(
        Path(gold_snapshot),
        role="gold",
        expected_symbols=ORCA_LSE_GOLD_UNIVERSE,
    )
    verified = [fx, gold]
    tail: Optional[_VerifiedSnapshot] = None
    if tail_snapshot is not None:
        tail = _verify_snapshot(
            Path(tail_snapshot),
            role="tail",
            expected_symbols=ORCA_LSE_UNIVERSE,
        )
        verified.append(tail)
    roots = [item.root for item in verified]
    if len(set(roots)) != len(roots):
        raise OrcaLsePanelError("every sealed snapshot must be a distinct root")

    history_snapshots = {"EURUSD": fx, "GBPUSD": fx, "USDCAD": fx, "GOLD": gold}
    history_series = [
        _load_d1_close(history_snapshots[symbol], symbol=symbol)
        for symbol in ORCA_LSE_UNIVERSE
    ]
    history = _aligned_panel(history_series, role="history")

    panel = history
    tail_audit: Optional[dict[str, Any]] = None
    if tail is not None:
        tail_series = [
            _load_d1_close(tail, symbol=symbol) for symbol in ORCA_LSE_UNIVERSE
        ]
        panel, tail_audit = _splice_tail(
            history,
            _aligned_panel(tail_series, role="tail"),
            minimum_overlap_rows=minimum_overlap,
        )

    if len(panel) < minimum:
        raise OrcaLsePanelError(
            f"ORCA-4 panel needs at least {minimum} aligned rows; received {len(panel)}"
        )
    try:
        panel = validate_d1_price_panel(panel, as_of_utc=as_of)
    except OrcaValidationError as exc:
        raise OrcaLsePanelError(f"ORCA rejected the aligned D1 panel: {exc}") from exc
    panel.index.name = "bar_close_time"

    source_rows = {item.name: int(len(item)) for item in history_series}
    lineage: dict[str, Any] = {
        "schema_version": ORCA_LSE_PANEL_SCHEMA,
        "as_of_utc": _iso(as_of),
        "universe": list(ORCA_LSE_UNIVERSE),
        "join_contract": {
            "method": "exact_bar_close_time_inner_join",
            "fill": "none",
            "price_field": "close",
            "time_field": "bar_close_time",
            "timezone": "UTC",
            "tail_splice": "exact_overlap_equality_then_append_after_history",
        },
        "minimum_aligned_rows": minimum,
        "minimum_overlap_rows": minimum_overlap,
        "aligned_rows": int(len(panel)),
        "aligned_start_utc": pd.Timestamp(panel.index[0]).isoformat(),
        "aligned_end_utc": pd.Timestamp(panel.index[-1]).isoformat(),
        "history_rows": int(len(history)),
        "history_start_utc": pd.Timestamp(history.index[0]).isoformat(),
        "history_end_utc": pd.Timestamp(history.index[-1]).isoformat(),
        "history_source_rows": source_rows,
        "history_rows_excluded_by_inner_join": {
            symbol: rows - len(history) for symbol, rows in source_rows.items()
        },
        "tail": tail_audit,
        "sources": [_snapshot_lineage(item) for item in verified],
    }
    return panel, lineage


def _snapshot_lineage(snapshot: _VerifiedSnapshot) -> dict[str, Any]:
    d1_files: list[dict[str, Any]] = []
    for symbol, entry in snapshot.d1_entries.items():
        d1_files.append(
            {
                "symbol": symbol,
                "path": str(entry["path"]),
                "rows": int(entry["rows"]),
                "sha256": _require_sha256(
                    entry.get("sha256"),
                    name=f"{snapshot.role} {symbol} sha256",
                ),
            }
        )
    return {
        "role": snapshot.role,
        "snapshot_path": str(snapshot.root),
        "stored_manifest_sha256": _require_sha256(
            snapshot.manifest.get("manifest_sha256"),
            name=f"{snapshot.role} manifest sha256",
        ),
        "manifest_file_sha256": snapshot.manifest_file_sha256,
        "source_manifest_sha256": snapshot.source_manifest_sha256,
        "provider": snapshot.source_manifest.get("provider"),
        "requested_start": snapshot.source_manifest.get("requested_start"),
        "requested_end_exclusive": snapshot.source_manifest.get(
            "requested_end_exclusive"
        ),
        "d1_files": d1_files,
    }


def _verify_written_panel(path: Path, expected: pd.DataFrame) -> None:
    try:
        actual = pd.read_parquet(path)
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=True,
            check_freq=False,
            check_names=True,
        )
    except Exception as exc:
        raise OrcaLsePanelError("written ORCA-4 Parquet failed verification") from exc


def _cleanup_staging(staging: Path, *, parent: Path, prefix: str) -> None:
    if not staging.exists():
        return
    resolved_parent = parent.resolve()
    resolved_staging = staging.resolve()
    if (
        resolved_staging.parent != resolved_parent
        or not resolved_staging.name.startswith(prefix)
    ):
        raise OrcaLsePanelError("refusing to clean an unexpected staging directory")
    shutil.rmtree(resolved_staging)


def materialize_orca_lse_panel(
    config: OrcaLsePanelConfig,
    *,
    generated_at_utc: Optional[datetime] = None,
) -> dict[str, Any]:
    """Atomically publish ``prices.parquet`` and its lineage manifest."""

    generated = _utc(
        generated_at_utc or datetime.now(timezone.utc),
        name="generated_at_utc",
    )
    output = config.output_dir.expanduser().resolve()
    source_roots = [
        config.fx_snapshot.expanduser().resolve(),
        config.gold_snapshot.expanduser().resolve(),
    ]
    if config.tail_snapshot is not None:
        source_roots.append(config.tail_snapshot.expanduser().resolve())
    for source in source_roots:
        if output == source or output.is_relative_to(source):
            raise OrcaLsePanelError("output_dir must not be inside a sealed snapshot")
    if output.exists():
        raise OrcaLsePanelError(f"output_dir already exists: {output}")
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir():
        raise OrcaLsePanelError(f"output parent is not a directory: {parent}")

    panel, lineage = build_orca_lse_panel(
        config.fx_snapshot,
        config.gold_snapshot,
        tail_snapshot=config.tail_snapshot,
        as_of_utc=generated,
        minimum_aligned_rows=config.minimum_aligned_rows,
        minimum_overlap_rows=config.minimum_overlap_rows,
    )

    prefix = f".{output.name}.partial-"
    staging = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    try:
        panel_path = staging / ORCA_LSE_PANEL_FILENAME
        panel.to_parquet(panel_path, index=True)
        _verify_written_panel(panel_path, panel)
        output_entry = {
            "path": ORCA_LSE_PANEL_FILENAME,
            "bytes": int(panel_path.stat().st_size),
            "sha256": _sha256_file(panel_path),
            "rows": int(len(panel)),
            "columns": list(panel.columns),
        }
        payload: dict[str, Any] = {
            **lineage,
            "generated_at_utc": _iso(generated),
            "publication": "atomic_new_directory_rename",
            "output": output_entry,
        }
        payload["manifest_sha256"] = _canonical_hash(payload)
        manifest_path = staging / ORCA_LSE_MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        reloaded_manifest = _load_json_object(
            manifest_path,
            name="output manifest",
        )
        stored_hash = reloaded_manifest.pop("manifest_sha256", None)
        if stored_hash != _canonical_hash(reloaded_manifest):
            raise OrcaLsePanelError("output manifest self-verification failed")
        if output.exists():
            raise OrcaLsePanelError(f"output_dir appeared during publication: {output}")
        os.rename(staging, output)
    except Exception:
        _cleanup_staging(staging, parent=parent, prefix=prefix)
        raise

    return {
        "status": "PUBLISHED",
        "output_dir": str(output),
        "panel_path": str(output / ORCA_LSE_PANEL_FILENAME),
        "manifest_path": str(output / ORCA_LSE_MANIFEST_FILENAME),
        "manifest_sha256": payload["manifest_sha256"],
        "rows": int(len(panel)),
        "columns": list(panel.columns),
        "aligned_end_utc": lineage["aligned_end_utc"],
        "tail_rows_added": (
            int(lineage["tail"]["rows_added_after_history"])
            if lineage["tail"] is not None
            else 0
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a sealed no-fill LSE ORCA-4 D1 panel"
    )
    parser.add_argument(
        "--fx-snapshot",
        required=True,
        help="sealed EURUSD/GBPUSD/USDCAD snapshot providing FX history",
    )
    parser.add_argument(
        "--gold-snapshot",
        required=True,
        help="sealed GOLD-only snapshot covering the same history",
    )
    parser.add_argument(
        "--tail-snapshot",
        default=None,
        help=(
            "optional sealed four-symbol snapshot extending the history to the "
            "present; spliced only when every overlapping D1 close matches "
            "exactly"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--minimum-aligned-rows",
        type=int,
        default=DEFAULT_MINIMUM_ALIGNED_ROWS,
    )
    parser.add_argument(
        "--minimum-overlap-rows",
        type=int,
        default=DEFAULT_MINIMUM_OVERLAP_ROWS,
        help="minimum exactly matching D1 bars the tail must share with history",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = materialize_orca_lse_panel(
            OrcaLsePanelConfig(
                fx_snapshot=Path(args.fx_snapshot),
                gold_snapshot=Path(args.gold_snapshot),
                output_dir=Path(args.output_dir),
                tail_snapshot=(
                    Path(args.tail_snapshot) if args.tail_snapshot else None
                ),
                minimum_aligned_rows=args.minimum_aligned_rows,
                minimum_overlap_rows=args.minimum_overlap_rows,
            )
        )
    except Exception as exc:
        print(f"ORCA LSE PANEL ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
