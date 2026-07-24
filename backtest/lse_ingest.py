"""London Strategic Edge candle ingestion for immutable backtest snapshots.

The live trading process deliberately does not import this module.  LSE is a
research/backfill source; broker execution quotes remain the source of truth
for live orders and for exact Bid/Ask execution modelling.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import pandas as pd

from .data import DataValidationError, HistoricalDataset, normalize_timeframe


SOURCE_TIMEFRAME = "1m"
DEFAULT_TARGET_TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h", "1d")
DEFAULT_BAR_TIMEZONE = "Europe/Athens"
DEFAULT_REST_MIN_INTERVAL = 0.35
DEFAULT_RETRY_AFTER_SECONDS = 60.0
MIN_REST_DAY_LIMIT = 1441
DEFAULT_MIN_BUCKET_COVERAGE = 0.95
DEFAULT_MAX_EDGE_GAP_DAYS = 4.0
DEFAULT_MAX_INTERNAL_GAP_DAYS = 4.0
DEFAULT_EXPORT_CHUNK_DAYS = 600
DEFAULT_MAX_EXPORT_JOBS = 5

_DEFAULT_PROVIDER_SYMBOLS = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDCAD": "USD/CAD",
    "XAUUSD": "XAU/USD",
    "GOLD": "XAU/USD",
}
_DEFAULT_BOT_SYMBOLS = {
    "EUR/USD": "EURUSD",
    "GBP/USD": "GBPUSD",
    "USD/CAD": "USDCAD",
    "XAU/USD": "GOLD",
}
_TARGET_DELTAS = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}
_UTC_RESAMPLE_RULES = {
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
}
_RETRIABLE_STATUSES = {0, 429, 502, 503, 504}
_BOT_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9_-]{0,31}$")
_PROVIDER_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9._/-]{0,31}$")
_DATASET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_EXPORT_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RELEASE_COMMIT_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_ENVIRONMENT_LOCK_SNAPSHOT = "environment.freeze.txt"


class LSEIngestError(DataValidationError):
    """Raised when an LSE request cannot produce a safe candle snapshot."""


@dataclass(frozen=True)
class LSESymbolSpec:
    bot_symbol: str
    provider_symbol: str

    def to_dict(self) -> dict[str, str]:
        return {
            "bot_symbol": self.bot_symbol,
            "provider_symbol": self.provider_symbol,
        }


@dataclass(frozen=True)
class LSEImportResult:
    output: Path
    manifest_sha256: str
    readiness: Mapping[str, Any]
    symbols: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": str(self.output),
            "manifest_sha256": self.manifest_sha256,
            "readiness": dict(self.readiness),
            "symbols": [dict(item) for item in self.symbols],
        }


def parse_symbol_spec(value: str) -> LSESymbolSpec:
    """Parse ``BOT`` or ``BOT=PROVIDER`` without allowing path characters."""

    raw = str(value or "").strip().upper()
    if not raw:
        raise LSEIngestError("LSE symbol cannot be empty")

    if "=" in raw:
        bot_symbol, provider_symbol = (part.strip() for part in raw.split("=", 1))
    elif raw in _DEFAULT_PROVIDER_SYMBOLS:
        bot_symbol = raw
        provider_symbol = _DEFAULT_PROVIDER_SYMBOLS[raw]
    elif raw in _DEFAULT_BOT_SYMBOLS:
        bot_symbol = _DEFAULT_BOT_SYMBOLS[raw]
        provider_symbol = raw
    elif "/" in raw:
        provider_symbol = raw
        bot_symbol = raw.replace("/", "")
    elif len(raw) == 6 and raw.isalpha():
        bot_symbol = raw
        provider_symbol = f"{raw[:3]}/{raw[3:]}"
    else:
        bot_symbol = raw
        provider_symbol = raw

    if not _BOT_SYMBOL_RE.fullmatch(bot_symbol):
        raise LSEIngestError(f"Unsafe/invalid bot symbol: {bot_symbol!r}")
    if ".." in provider_symbol or not _PROVIDER_SYMBOL_RE.fullmatch(provider_symbol):
        raise LSEIngestError(f"Unsafe/invalid LSE symbol: {provider_symbol!r}")
    return LSESymbolSpec(bot_symbol=bot_symbol, provider_symbol=provider_symbol)


def _utc_bound(value: Any, *, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise LSEIngestError(f"Invalid {name}: {value!r}") from exc
    if pd.isna(timestamp):
        raise LSEIngestError(f"Invalid {name}: {value!r}")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _sdk_version() -> str:
    try:
        return version("lse-data")
    except PackageNotFoundError:
        return "not-installed"


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_environment_metadata(
    *,
    release_commit_file: str | Path | None,
    environment_lock_file: str | Path | None,
) -> Optional[dict[str, Any]]:
    """Build path-free provenance for the deployed code and Python lock."""

    if (release_commit_file is None) != (environment_lock_file is None):
        raise LSEIngestError(
            "release_commit_file and environment_lock_file must be provided together"
        )
    if release_commit_file is None:
        return None

    release_path = Path(release_commit_file).expanduser()
    lock_path = Path(environment_lock_file).expanduser()
    if not release_path.is_file():
        raise LSEIngestError(f"Release commit file does not exist: {release_path.name}")
    if not lock_path.is_file():
        raise LSEIngestError(f"Environment lock file does not exist: {lock_path.name}")
    try:
        release_commit = release_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise LSEIngestError(
            f"Cannot read release commit file {release_path.name!r}: {exc}"
        ) from exc
    if not _RELEASE_COMMIT_RE.fullmatch(release_commit):
        raise LSEIngestError(
            "Release commit must contain exactly one 40- or 64-character hex digest"
        )
    try:
        lock_bytes = int(lock_path.stat().st_size)
        lock_sha256 = _sha256_file(lock_path)
    except OSError as exc:
        raise LSEIngestError(
            f"Cannot hash environment lock file {lock_path.name!r}: {exc}"
        ) from exc
    return {
        "release_commit": release_commit.lower(),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {
            "pandas": _package_version("pandas"),
            "pyarrow": _package_version("pyarrow"),
        },
        "lock": {
            "filename": _ENVIRONMENT_LOCK_SNAPSHOT,
            "bytes": lock_bytes,
            "sha256": lock_sha256,
        },
    }


def _redacted_error(exc: Exception) -> str:
    message = str(exc)
    key = os.environ.get("LSE_API_KEY", "").strip()
    if key:
        message = message.replace(key, "[REDACTED]")
    return message[:500]


def create_lse_client(*, timeout: float = 60.0):
    """Create the official SDK client using only ``LSE_API_KEY``."""

    key = os.environ.get("LSE_API_KEY", "").strip()
    if not key or "CHANGE_ME" in key.upper() or "XXXXXXXX" in key.upper():
        raise LSEIngestError(
            "LSE_API_KEY is missing or still contains a placeholder"
        )
    try:
        from lse import LSE
    except ImportError as exc:
        raise LSEIngestError(
            "lse-data is not installed; install requirements-backtest.txt"
        ) from exc
    return LSE(api_key=key, timeout=float(timeout))


def _lowercase_columns(frame: pd.DataFrame) -> pd.DataFrame:
    names = [str(column).strip().lower() for column in frame.columns]
    if len(names) != len(set(names)):
        raise LSEIngestError("LSE candle columns collide after case normalization")
    out = frame.copy()
    out.columns = names
    return out


def _timestamp_values(frame: pd.DataFrame) -> pd.Series:
    if "timestamp" in frame.columns:
        return frame["timestamp"]
    if "ts" in frame.columns:
        return frame["ts"]
    if "time" in frame.columns:
        return frame["time"]
    if isinstance(frame.index, pd.DatetimeIndex):
        return pd.Series(frame.index, index=frame.index)
    raise LSEIngestError("LSE candles have no timestamp/ts/time column")


def _validate_returned_identity(
    frame: pd.DataFrame,
    *,
    expected_provider_symbol: str,
    expected_timeframe: str,
) -> None:
    """Fail closed when LSE labels returned rows as another series."""

    expected_symbol = str(expected_provider_symbol).strip().upper()
    if "symbol" in frame.columns:
        normalized_symbols = frame["symbol"].map(
            lambda value: "" if pd.isna(value) else str(value).strip().upper()
        )
        if (normalized_symbols == "").any():
            raise LSEIngestError("LSE returned null/blank symbol label(s)")
        symbols = {
            value for value in normalized_symbols.unique()
        }
        if symbols != {expected_symbol}:
            raise LSEIngestError(
                "LSE returned unexpected symbol label(s): "
                f"expected {expected_symbol!r}, got {sorted(symbols)!r}"
            )

    timeframe_columns = [
        column for column in ("tf", "timeframe") if column in frame.columns
    ]
    for timeframe_column in timeframe_columns:
        raw_labels = frame[timeframe_column]
        blank_labels = raw_labels.map(
            lambda value: pd.isna(value) or str(value).strip() == ""
        )
        if blank_labels.any():
            raise LSEIngestError(
                f"LSE returned null/blank timeframe label(s) in {timeframe_column}"
            )
        labels: set[str] = set()
        for value in raw_labels.unique():
            try:
                labels.add(normalize_timeframe(value))
            except DataValidationError as exc:
                raise LSEIngestError(
                    f"LSE returned invalid timeframe label: {value!r}"
                ) from exc
        expected_tf = normalize_timeframe(expected_timeframe)
        if labels != {expected_tf}:
            raise LSEIngestError(
                "LSE returned unexpected timeframe label(s) "
                f"in {timeframe_column}: "
                f"expected {expected_tf!r}, got {sorted(labels)!r}"
            )


def normalize_lse_candles(
    frame: pd.DataFrame,
    *,
    expected_provider_symbol: str,
    expected_timeframe: str = SOURCE_TIMEFRAME,
    start: Any,
    end: Any,
    max_duplicate_ratio: float = 0.001,
    max_edge_gap_days: float = DEFAULT_MAX_EDGE_GAP_DAYS,
    max_internal_gap_days: float = DEFAULT_MAX_INTERNAL_GAP_DAYS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return sorted, unique UTC M1 candles and a quality report."""

    if frame is None or frame.empty:
        raise LSEIngestError("LSE returned an empty candle frame")
    if not 0.0 <= float(max_duplicate_ratio) <= 1.0:
        raise LSEIngestError("max_duplicate_ratio must be between 0 and 1")
    if not math.isfinite(float(max_edge_gap_days)) or float(max_edge_gap_days) < 0:
        raise LSEIngestError("max_edge_gap_days must be a finite non-negative value")
    if (
        not math.isfinite(float(max_internal_gap_days))
        or float(max_internal_gap_days) < 0
    ):
        raise LSEIngestError(
            "max_internal_gap_days must be a finite non-negative value"
        )

    start_utc = _utc_bound(start, name="start")
    end_utc = _utc_bound(end, name="end")
    if end_utc <= start_utc:
        raise LSEIngestError("end must be after start")

    out = _lowercase_columns(frame)
    _validate_returned_identity(
        out,
        expected_provider_symbol=expected_provider_symbol,
        expected_timeframe=expected_timeframe,
    )
    parsed = pd.to_datetime(_timestamp_values(out), utc=True, errors="coerce")
    if parsed.isna().any():
        raise LSEIngestError(
            f"LSE candles contain {int(parsed.isna().sum())} invalid timestamp(s)"
        )
    parsed_index = pd.DatetimeIndex(parsed, name="timestamp")
    misaligned = parsed_index != parsed_index.floor("min")
    if misaligned.any():
        raise LSEIngestError(
            f"LSE M1 candles contain {int(misaligned.sum())} "
            "timestamp(s) not aligned to an exact UTC minute"
        )
    out.index = parsed_index
    out = out.loc[(out.index >= start_utc) & (out.index < end_utc)].copy()
    if out.empty:
        raise LSEIngestError("LSE returned no candles inside the requested range")

    missing = [
        column for column in ("open", "high", "low", "close") if column not in out
    ]
    if missing:
        raise LSEIngestError(f"LSE candles are missing OHLC columns: {missing}")

    for column in ("open", "high", "low", "close"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
        invalid = out[column].isna() | ~out[column].map(math.isfinite)
        if invalid.any():
            raise LSEIngestError(
                f"LSE candles contain {int(invalid.sum())} invalid {column} value(s)"
            )
        if (out[column] <= 0).any():
            raise LSEIngestError("LSE candle prices must be positive")

    volume_was_missing = "volume" not in out.columns
    if volume_was_missing:
        out["volume"] = 0.0
    else:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
        invalid_volume = (
            out["volume"].isna()
            | ~out["volume"].map(math.isfinite)
            | (out["volume"] < 0)
        )
        if invalid_volume.any():
            raise LSEIngestError(
                f"LSE candles contain {int(invalid_volume.sum())} invalid volume value(s)"
            )

    impossible = (
        (out["high"] < out["low"])
        | (out["high"] < out[["open", "close"]].max(axis=1))
        | (out["low"] > out[["open", "close"]].min(axis=1))
    )
    if impossible.any():
        raise LSEIngestError(
            f"LSE candles contain {int(impossible.sum())} impossible OHLC row(s)"
        )

    out = out.sort_index(kind="stable")
    raw_rows = len(out)
    duplicate_rows = int(out.index.duplicated(keep="last").sum())
    if duplicate_rows:
        duplicate_group_rows = out.loc[
            out.index.duplicated(keep=False),
            ["open", "high", "low", "close", "volume"],
        ]
        conflicts = (
            duplicate_group_rows.groupby(level=0, sort=False)
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if conflicts.any():
            raise LSEIngestError(
                "LSE candles contain "
                f"{int(conflicts.sum())} conflicting duplicate timestamp(s)"
            )
    duplicate_ratio = duplicate_rows / raw_rows
    if duplicate_ratio > float(max_duplicate_ratio):
        raise LSEIngestError(
            "LSE duplicate timestamp ratio "
            f"{duplicate_ratio:.6f} exceeds {float(max_duplicate_ratio):.6f}"
        )
    out = out.loc[~out.index.duplicated(keep="last")]
    out = out[["open", "high", "low", "close", "volume"]].copy()

    gaps = out.index.to_series().diff().dropna()
    max_gap_seconds = float(gaps.max().total_seconds()) if not gaps.empty else 0.0
    internal_gap_tolerance = pd.Timedelta(days=float(max_internal_gap_days))
    max_missing_gap = (
        max(
            pd.Timedelta(0),
            gaps.max() - _TARGET_DELTAS[SOURCE_TIMEFRAME],
        )
        if not gaps.empty
        else pd.Timedelta(0)
    )
    if max_missing_gap > internal_gap_tolerance:
        raise LSEIngestError(
            "LSE history contains an internal gap larger than allowed: "
            f"max_missing_gap={max_missing_gap}, allowed={internal_gap_tolerance}"
        )
    first_open_gap = max(pd.Timedelta(0), out.index[0] - start_utc)
    last_close = out.index[-1] + _TARGET_DELTAS[SOURCE_TIMEFRAME]
    last_close_gap = max(pd.Timedelta(0), end_utc - last_close)
    edge_tolerance = pd.Timedelta(days=float(max_edge_gap_days))
    if first_open_gap > edge_tolerance or last_close_gap > edge_tolerance:
        raise LSEIngestError(
            "LSE history does not cover the requested range edges: "
            f"first_open_gap={first_open_gap}, last_close_gap={last_close_gap}, "
            f"allowed={edge_tolerance}"
        )
    quality = {
        "raw_rows": raw_rows,
        "rows": len(out),
        "duplicate_rows": duplicate_rows,
        "duplicate_ratio": duplicate_ratio,
        "volume_was_missing": volume_was_missing,
        "start": out.index[0].isoformat(),
        "end_open": out.index[-1].isoformat(),
        "end_close": last_close.isoformat(),
        "first_open_gap_seconds": float(first_open_gap.total_seconds()),
        "last_close_gap_seconds": float(last_close_gap.total_seconds()),
        "max_edge_gap_days": float(max_edge_gap_days),
        "max_internal_gap_days": float(max_internal_gap_days),
        "max_gap_seconds": max_gap_seconds,
        "max_missing_gap_seconds": float(max_missing_gap.total_seconds()),
    }
    return out, quality


def _request_with_retry(
    call: Callable[[], Any],
    *,
    retries: int,
    sleep: Callable[[float], None],
    retry_after_seconds: float = DEFAULT_RETRY_AFTER_SECONDS,
) -> Any:
    attempt = 0
    while True:
        try:
            return call()
        except Exception as exc:
            try:
                status = int(getattr(exc, "status", -1))
            except (TypeError, ValueError):
                status = -1
            if status not in _RETRIABLE_STATUSES or attempt >= retries:
                raise LSEIngestError(
                    f"LSE request failed (status={status}): {_redacted_error(exc)}"
                ) from exc
            if status == 429:
                delay = float(retry_after_seconds)
            else:
                delay = min(8.0, 0.5 * (2**attempt)) + random.uniform(0.0, 0.25)
            sleep(delay)
            attempt += 1


def _deduplicate_transport_boundaries(
    frames: Sequence[pd.DataFrame],
) -> tuple[pd.DataFrame, int]:
    """Remove only duplicates crossing page/chunk boundaries.

    Duplicate timestamps inside one provider response remain intact so normal
    provider duplicate-ratio QC can still detect them.
    """

    if not frames:
        raise LSEIngestError("LSE transport returned no candle frames")

    def canonical_timeframe(value: Any) -> Any:
        if pd.isna(value) or str(value).strip() == "":
            return None
        try:
            return normalize_timeframe(value)
        except DataValidationError:
            return f"invalid:{str(value).strip().lower()}"

    tagged: list[pd.DataFrame] = []
    for frame_number, frame in enumerate(frames):
        item = _lowercase_columns(frame)
        parsed = pd.to_datetime(_timestamp_values(item), utc=True, errors="coerce")
        if parsed.isna().any():
            raise LSEIngestError("LSE transport data contains invalid timestamps")
        # Canonicalize before concat so mixed ``ts``/``timestamp`` files and
        # Parquet files with a DatetimeIndex cannot lose their timestamps.
        item["timestamp"] = pd.DatetimeIndex(parsed)
        item["_transport_frame"] = frame_number
        item["_transport_symbol"] = (
            item["symbol"].map(
                lambda value: (
                    None if pd.isna(value) else str(value).strip().upper()
                )
            )
            if "symbol" in item
            else None
        )
        timeframe_columns = [
            column for column in ("tf", "timeframe") if column in item
        ]
        if timeframe_columns:
            canonical_columns = [
                item[column].map(canonical_timeframe)
                for column in timeframe_columns
            ]
            item["_transport_timeframe"] = canonical_columns[0]
            for canonical in canonical_columns[1:]:
                mismatch = item["_transport_timeframe"] != canonical
                item.loc[mismatch, "_transport_timeframe"] = "<conflict>"
        else:
            item["_transport_timeframe"] = None
        for column in ("open", "high", "low", "close"):
            item[f"_transport_{column}"] = pd.to_numeric(
                item[column] if column in item else None,
                errors="coerce",
            )
        item["_transport_volume"] = (
            pd.to_numeric(item["volume"], errors="coerce")
            if "volume" in item
            else 0.0
        )
        tagged.append(item)
    merged = pd.concat(tagged, ignore_index=True)
    merged["_transport_timestamp"] = merged["timestamp"]
    first_transport_frame = merged.groupby(
        "_transport_timestamp", sort=False
    )["_transport_frame"].transform("min")
    transport_overlap = merged["_transport_frame"] > first_transport_frame
    remove_overlap = pd.Series(False, index=merged.index)
    fingerprint_columns = [
        "_transport_symbol",
        "_transport_timeframe",
        "_transport_open",
        "_transport_high",
        "_transport_low",
        "_transport_close",
        "_transport_volume",
    ]
    overlap_timestamps = merged.loc[
        transport_overlap, "_transport_timestamp"
    ].unique()
    boundary_rows = merged.loc[
        merged["_transport_timestamp"].isin(overlap_timestamps)
    ]
    for timestamp, timestamp_rows in boundary_rows.groupby(
        "_transport_timestamp",
        sort=False,
    ):
        frame_numbers = sorted(timestamp_rows["_transport_frame"].unique())
        for frame_number in frame_numbers[1:]:
            earlier = timestamp_rows.loc[
                timestamp_rows["_transport_frame"] < frame_number
            ]
            later = timestamp_rows.loc[
                timestamp_rows["_transport_frame"] == frame_number
            ]
            earlier_fingerprints = {
                tuple(row)
                for row in earlier[fingerprint_columns].itertuples(
                    index=False,
                    name=None,
                )
            }
            later_fingerprints = [
                tuple(row)
                for row in later[fingerprint_columns].itertuples(
                    index=False,
                    name=None,
                )
            ]
            if any(
                fingerprint not in earlier_fingerprints
                for fingerprint in later_fingerprints
            ):
                raise LSEIngestError(
                    "Conflicting candle values at an LSE transport boundary: "
                    f"{pd.Timestamp(timestamp).isoformat()}"
                )
            # One matching row per fingerprint is the intentional page/chunk
            # overlap. Keep any extra duplicates so provider duplicate QC can
            # still see and reject them.
            removed_fingerprints: set[tuple[Any, ...]] = set()
            for row_index, fingerprint in zip(
                later.index,
                later_fingerprints,
            ):
                if fingerprint not in removed_fingerprints:
                    remove_overlap.loc[row_index] = True
                    removed_fingerprints.add(fingerprint)

    duplicate_rows = int(remove_overlap.sum())
    merged = merged.loc[~remove_overlap].copy()
    helper_columns = [
        column for column in merged if column.startswith("_transport_")
    ]
    return merged.drop(columns=helper_columns), duplicate_rows


def fetch_lse_rest(
    client: Any,
    *,
    provider_symbol: str,
    start: Any,
    end: Any,
    dataset: Optional[str] = None,
    page_limit: int = 5000,
    retries: int = 3,
    rest_min_interval: float = DEFAULT_REST_MIN_INTERVAL,
    retry_after_seconds: float = DEFAULT_RETRY_AFTER_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    now: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch M1 candles through the provider's UTC date-window contract.

    ``/vault/candles`` accepts ``YYYY-MM-DD`` bounds, with an inclusive start
    date and an exclusive end date.  Each request therefore covers one to three
    complete UTC calendar days.  A complete M1 day has at most 1,440 aligned
    bars; the window is sized so its theoretical maximum remains below the
    configured row limit.  Reaching the limit is treated as possible
    server-side truncation.  Partial requested days are filtered later by
    :func:`normalize_lse_candles`.
    """

    start_utc = _utc_bound(start, name="start")
    end_utc = _utc_bound(end, name="end")
    if end_utc <= start_utc:
        raise LSEIngestError("end must be after start")
    now_utc = (
        pd.Timestamp.now(tz="UTC")
        if now is None
        else _utc_bound(now, name="now")
    )
    last_closed_utc_day = now_utc.normalize()
    if end_utc > last_closed_utc_day:
        raise LSEIngestError(
            "REST snapshots cannot include the current, still-open UTC day; "
            f"end must be no later than {last_closed_utc_day.isoformat()}"
        )
    page_limit = min(max(int(page_limit), 1), 5000)
    if page_limit < MIN_REST_DAY_LIMIT:
        raise LSEIngestError(
            "page_limit must be at least "
            f"{MIN_REST_DAY_LIMIT} for a complete UTC M1 day"
        )
    window_days = min(3, max(1, (page_limit - 1) // 1440))
    if int(retries) < 0:
        raise LSEIngestError("retries must be non-negative")
    if (
        not math.isfinite(float(rest_min_interval))
        or float(rest_min_interval) < 0
    ):
        raise LSEIngestError("rest_min_interval must be finite and non-negative")
    if (
        not math.isfinite(float(retry_after_seconds))
        or float(retry_after_seconds) < 0
    ):
        raise LSEIngestError("retry_after_seconds must be finite and non-negative")

    first_day = start_utc.normalize()
    end_day = end_utc.normalize()
    if end_utc != end_day:
        end_day += pd.Timedelta(days=1)

    pages: list[pd.DataFrame] = []
    requested_windows = 0
    empty_windows = 0
    api_requests = 0
    window_start = first_day
    while window_start < end_day:
        requested_windows += 1
        window_end = min(
            window_start + pd.Timedelta(days=window_days),
            end_day,
        )
        request_start = window_start.strftime("%Y-%m-%d")
        request_end = window_end.strftime("%Y-%m-%d")

        def fetch_day(*, order: str, limit: int):
            def request():
                nonlocal api_requests
                if api_requests and float(rest_min_interval) > 0:
                    sleep(float(rest_min_interval))
                api_requests += 1
                return client.candles(
                    provider_symbol,
                    SOURCE_TIMEFRAME,
                    start=request_start,
                    end=request_end,
                    limit=limit,
                    order=order,
                    dataset=dataset,
                )

            return _request_with_retry(
                request,
                retries=retries,
                sleep=sleep,
                retry_after_seconds=retry_after_seconds,
            )

        rows = fetch_day(order="asc", limit=page_limit)
        if rows is None:
            raise LSEIngestError(
                "LSE REST returned a null ascending response for "
                f"{request_start}"
            )
        page = _lowercase_columns(
            pd.DataFrame(rows)
        )
        if len(page) >= page_limit:
            raise LSEIngestError(
                "LSE REST UTC-day response reached its row limit and may be "
                f"truncated: {request_start}, rows={len(page)}, "
                f"limit={page_limit}"
            )

        tail_rows = fetch_day(order="desc", limit=1)
        if tail_rows is None:
            raise LSEIngestError(
                f"LSE REST returned a null tail-probe response for {request_start}"
            )
        tail = _lowercase_columns(
            pd.DataFrame(tail_rows)
        )
        if len(tail) > 1:
            raise LSEIngestError(
                "LSE REST tail probe returned more than one candle for "
                f"{request_start}"
            )

        def validate_day_response(
            response: pd.DataFrame,
            *,
            label: str,
        ) -> pd.DatetimeIndex:
            if response.empty:
                return pd.DatetimeIndex([], tz="UTC")
            _validate_returned_identity(
                response,
                expected_provider_symbol=provider_symbol,
                expected_timeframe=SOURCE_TIMEFRAME,
            )
            timestamps = pd.to_datetime(
                _timestamp_values(response),
                utc=True,
                errors="coerce",
            )
            if timestamps.isna().any():
                raise LSEIngestError(
                    f"LSE REST {label} UTC-day response contains invalid "
                    f"timestamps: {request_start}"
                )
            parsed_index = pd.DatetimeIndex(timestamps)
            misaligned = parsed_index != parsed_index.floor("min")
            if misaligned.any():
                raise LSEIngestError(
                    f"LSE REST {label} response contains timestamp(s) not "
                    f"aligned to an exact UTC minute: {request_start}"
                )
            outside_window = (parsed_index < window_start) | (
                parsed_index >= window_end
            )
            if outside_window.any():
                raise LSEIngestError(
                    f"LSE REST {label} response returned candle(s) outside "
                    "the requested UTC-day window: "
                    f"{request_start}..{request_end}"
                )
            return parsed_index

        parsed = validate_day_response(page, label="ascending")
        tail_parsed = validate_day_response(tail, label="tail-probe")
        if page.empty != tail.empty:
            raise LSEIngestError(
                "LSE REST UTC-day response disagrees with its tail probe and "
                f"may be truncated: {request_start}; use bulk export"
            )
        if page.empty:
            empty_windows += 1
            window_start = window_end
            continue
        if parsed.max() != tail_parsed.max():
            raise LSEIngestError(
                "LSE REST UTC-day response is capped before the provider's "
                f"last candle: {request_start}; use bulk export"
            )
        latest_rows = page.loc[
            pd.DatetimeIndex(parsed) == parsed.max()
        ]
        try:
            _, matched_tail_rows = _deduplicate_transport_boundaries(
                [latest_rows, tail]
            )
        except LSEIngestError as exc:
            raise LSEIngestError(
                "LSE REST tail-probe candle conflicts with the ascending "
                f"response: {request_start}; use bulk export"
            ) from exc
        if matched_tail_rows != 1:
            raise LSEIngestError(
                "LSE REST tail probe did not match the ascending response: "
                f"{request_start}; use bulk export"
            )
        pages.append(page)
        window_start = window_end

    if not pages:
        raise LSEIngestError("LSE REST returned no candles in the UTC-day windows")
    merged = pd.concat(pages, ignore_index=True, sort=False)
    return merged, {
        "transport": "rest",
        "requests": api_requests,
        "pages": len(pages),
        "utc_date_windows": requested_windows,
        "window_days": window_days,
        "empty_utc_date_windows": empty_windows,
        "tail_probes": requested_windows,
        "pagination_overlap_rows": 0,
        "page_limit": page_limit,
        "rest_min_interval": float(rest_min_interval),
    }


def _export_date_chunks(
    start: str,
    end: str,
    *,
    chunk_days: int,
) -> tuple[tuple[str, str], ...]:
    if not _EXPORT_DATE_RE.fullmatch(str(start)) or not _EXPORT_DATE_RE.fullmatch(
        str(end)
    ):
        raise LSEIngestError(
            "LSE export transport requires start/end in YYYY-MM-DD format"
        )
    if int(chunk_days) < 1:
        raise LSEIngestError("export_chunk_days must be at least 1")
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    if end_date <= start_date:
        raise LSEIngestError("end must be after start")

    chunks: list[tuple[str, str]] = []
    cursor = start_date
    while cursor < end_date:
        chunk_end = min(cursor + pd.Timedelta(days=int(chunk_days)), end_date)
        chunks.append(
            (cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
        )
        # Adjacent exports deliberately share their boundary date. This avoids
        # gaps across either inclusive or exclusive server-side date semantics.
        cursor = chunk_end
    return tuple(chunks)


def _load_export_checkpoint(
    directory: Path,
    *,
    expected: Mapping[str, Any],
) -> Optional[tuple[pd.DataFrame, dict[str, Any]]]:
    marker = directory / "checkpoint.json"
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if any(payload.get(key) != value for key, value in expected.items()):
            return None
        raw_path = (directory / str(payload["file"])).resolve()
        raw_path.relative_to(directory.resolve())
        if not raw_path.is_file():
            return None
        if _sha256_file(raw_path) != payload.get("sha256"):
            return None
        frame = pd.read_parquet(raw_path)
    except Exception:
        return None
    return frame, {
        **dict(expected),
        "file": raw_path.name,
        "bytes": int(raw_path.stat().st_size),
        "sha256": str(payload["sha256"]),
        "checkpoint_reused": True,
    }


def _save_export_checkpoint(
    directory: Path,
    *,
    expected: Mapping[str, Any],
    raw_path: Path,
) -> dict[str, Any]:
    report = {
        **dict(expected),
        "file": raw_path.name,
        "bytes": int(raw_path.stat().st_size),
        "sha256": _sha256_file(raw_path),
        "checkpoint_reused": False,
    }
    marker = directory / "checkpoint.json"
    temporary = directory / "checkpoint.json.tmp"
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(marker)
    return report


def fetch_lse_export(
    client: Any,
    *,
    provider_symbol: str,
    start: str,
    end: str,
    destination: Path,
    dataset: Optional[str] = None,
    export_chunk_days: int = DEFAULT_EXPORT_CHUNK_DAYS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run chunked M1 exports, reusing completed deterministic checkpoints."""

    normalized_provider = str(provider_symbol).strip().upper()
    if (
        ".." in normalized_provider
        or not _PROVIDER_SYMBOL_RE.fullmatch(normalized_provider)
    ):
        raise LSEIngestError("LSE export requires a safe provider symbol")
    if dataset is None or not _DATASET_RE.fullmatch(str(dataset)):
        raise LSEIngestError("LSE export requires a safe resolved dataset name")
    chunks = _export_date_chunks(start, end, chunk_days=export_chunk_days)
    if destination.is_symlink():
        raise LSEIngestError("LSE export checkpoint cannot be a symbolic link")
    destination.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    chunk_reports: list[dict[str, Any]] = []
    jobs_started = 0
    for index, (chunk_start, chunk_end) in enumerate(chunks):
        chunk_dir = destination / f"{index:04d}_{chunk_start}_{chunk_end}"
        if chunk_dir.is_symlink():
            raise LSEIngestError("LSE export chunk cannot be a symbolic link")
        chunk_dir.mkdir(parents=True, exist_ok=True)
        expected = {
            "provider_symbol": provider_symbol,
            "dataset": dataset,
            "timeframe": SOURCE_TIMEFRAME,
            "start": chunk_start,
            "end": chunk_end,
        }
        checkpoint = _load_export_checkpoint(chunk_dir, expected=expected)
        if checkpoint is not None:
            frame, report = checkpoint
            frames.append(frame)
            chunk_reports.append(report)
            continue

        # SDK 0.14 resumes any existing ``.part`` by byte offset, but this call
        # creates a new export job. A partial artifact from an older job must
        # never be appended to the new job's bytes.
        stale_part = chunk_dir / (
            f"{dataset}_{provider_symbol.replace('/', '_')}_"
            f"{SOURCE_TIMEFRAME}.parquet.part"
        )
        try:
            stale_part.resolve().relative_to(chunk_dir.resolve())
            stale_part.unlink(missing_ok=True)
        except (OSError, ValueError) as exc:
            raise LSEIngestError(
                f"Cannot remove stale partial export for {provider_symbol}"
            ) from exc

        try:
            raw_path = Path(
                client.history(
                    provider_symbol,
                    dataset=dataset,
                    timeframe=SOURCE_TIMEFRAME,
                    start=chunk_start,
                    end=chunk_end,
                    dest=str(chunk_dir),
                    dataframe=False,
                )
            ).resolve()
            jobs_started += 1
        except Exception as exc:
            raise LSEIngestError(
                f"LSE export failed: {_redacted_error(exc)}"
            ) from exc
        try:
            raw_path.relative_to(chunk_dir.resolve())
        except ValueError as exc:
            raise LSEIngestError(
                f"LSE export created a file outside its checkpoint: {raw_path}"
            ) from exc
        if not raw_path.is_file():
            raise LSEIngestError(
                f"LSE export did not create a Parquet file: {raw_path}"
            )
        try:
            frame = pd.read_parquet(raw_path)
        except Exception as exc:
            raise LSEIngestError(f"Cannot read LSE export Parquet: {exc}") from exc
        frames.append(frame)
        chunk_reports.append(
            _save_export_checkpoint(
                chunk_dir,
                expected=expected,
                raw_path=raw_path,
            )
        )

    merged, overlap_rows = _deduplicate_transport_boundaries(frames)
    combined_hash = hashlib.sha256(
        json.dumps(
            [report["sha256"] for report in chunk_reports],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return merged, {
        "transport": "export",
        "jobs": len(chunks),
        "jobs_started": jobs_started,
        "checkpoint_reused": len(chunks) - jobs_started,
        "export_chunk_days": int(export_chunk_days),
        "raw_bytes": sum(int(report["bytes"]) for report in chunk_reports),
        "raw_sha256": combined_hash,
        "chunk_overlap_rows": overlap_rows,
        "chunks": chunk_reports,
    }


def _resolve_symbol_datasets(
    client: Any,
    specs: Sequence[LSESymbolSpec],
    *,
    dataset: Optional[str],
    retries: int,
    retry_after_seconds: float,
    sleep: Callable[[float], None],
) -> dict[str, str]:
    """Resolve every requested symbol against the official vault catalog."""

    try:
        catalog_raw = _request_with_retry(
            client.datasets,
            retries=retries,
            sleep=sleep,
            retry_after_seconds=retry_after_seconds,
        )
    except AttributeError as exc:
        raise LSEIngestError(
            "LSE client does not expose datasets(); cannot validate symbols"
        ) from exc
    if isinstance(catalog_raw, pd.DataFrame):
        catalog = catalog_raw.to_dict(orient="records")
    elif isinstance(catalog_raw, Sequence) and not isinstance(
        catalog_raw, (str, bytes)
    ):
        catalog = list(catalog_raw)
    else:
        raise LSEIngestError("LSE datasets() returned an invalid catalog")

    rows: list[tuple[str, str]] = []
    for item in catalog:
        if not isinstance(item, Mapping):
            continue
        catalog_symbol = str(item.get("symbol") or "").strip().upper()
        catalog_dataset = str(item.get("dataset") or "").strip()
        if catalog_symbol and catalog_dataset:
            if not _DATASET_RE.fullmatch(catalog_dataset):
                raise LSEIngestError(
                    f"LSE catalog contains an unsafe dataset label for "
                    f"{catalog_symbol!r}"
                )
            rows.append((catalog_symbol, catalog_dataset))
    if not rows:
        raise LSEIngestError("LSE catalog contains no usable symbol rows")

    requested_dataset = str(dataset).strip() if dataset is not None else None
    if requested_dataset == "":
        requested_dataset = None
    resolved: dict[str, str] = {}
    for spec in specs:
        matches = sorted(
            {
                catalog_dataset
                for catalog_symbol, catalog_dataset in rows
                if catalog_symbol == spec.provider_symbol.upper()
            }
        )
        if not matches:
            raise LSEIngestError(
                f"{spec.provider_symbol!r} is not present in the LSE catalog"
            )
        if requested_dataset is not None:
            selected = next(
                (
                    item
                    for item in matches
                    if item.lower() == requested_dataset.lower()
                ),
                None,
            )
            if selected is None:
                raise LSEIngestError(
                    f"{spec.provider_symbol!r} is catalogued in {matches}, "
                    f"not dataset {requested_dataset!r}"
                )
        else:
            non_options = [item for item in matches if item.lower() != "options"]
            candidates = non_options or matches
            if len(candidates) != 1:
                raise LSEIngestError(
                    f"{spec.provider_symbol!r} exists in multiple LSE datasets "
                    f"{candidates}; pass dataset explicitly"
                )
            selected = candidates[0]
        resolved[spec.bot_symbol] = selected
    return resolved


def _aggregate_ohlcv(frame: pd.DataFrame, labels: pd.DatetimeIndex) -> pd.DataFrame:
    work = frame.copy()
    work["_bucket"] = labels
    grouped = work.groupby("_bucket", sort=True)
    out = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        _source_rows=("open", "size"),
    )
    out.index = pd.DatetimeIndex(out.index, name="timestamp")
    return out


def _wall_clock_resample(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    bar_timezone: str,
) -> pd.DataFrame:
    """Bucket H4/D1 at broker wall-clock boundaries, including DST changes."""

    try:
        local = frame.index.tz_convert(bar_timezone)
    except Exception as exc:
        raise LSEIngestError(f"Invalid bar timezone {bar_timezone!r}: {exc}") from exc
    naive = local.tz_localize(None)
    if timeframe == "4h":
        offsets = pd.to_timedelta((naive.hour // 4) * 4, unit="h")
        labels_naive = naive.normalize() + offsets
        closes_naive = labels_naive + pd.Timedelta(hours=4)
    elif timeframe == "1d":
        labels_naive = naive.normalize()
        closes_naive = labels_naive + pd.Timedelta(days=1)
    else:  # pragma: no cover - guarded by the public dispatcher
        raise LSEIngestError(f"Unsupported wall-clock timeframe: {timeframe}")

    try:
        labels_local = labels_naive.tz_localize(
            bar_timezone, ambiguous="raise", nonexistent="raise"
        )
        closes_local = closes_naive.tz_localize(
            bar_timezone, ambiguous="raise", nonexistent="raise"
        )
    except Exception as exc:
        raise LSEIngestError(
            f"{bar_timezone} has an ambiguous/nonexistent {timeframe} boundary: {exc}"
        ) from exc

    labels_utc = labels_local.tz_convert("UTC")
    closes_utc = closes_local.tz_convert("UTC")
    out = _aggregate_ohlcv(frame, labels_utc)
    close_by_label = (
        pd.DataFrame({"label": labels_utc, "bar_close_time": closes_utc})
        .drop_duplicates("label", keep="last")
        .set_index("label")["bar_close_time"]
    )
    out["bar_close_time"] = pd.DatetimeIndex(
        [close_by_label.loc[label] for label in out.index]
    )
    return out


def resample_lse_candles(
    minute_frame: pd.DataFrame,
    *,
    timeframe: str,
    start: Any,
    end: Any,
    bar_timezone: str = DEFAULT_BAR_TIMEZONE,
    min_bucket_coverage: float = DEFAULT_MIN_BUCKET_COVERAGE,
) -> pd.DataFrame:
    """Build one canonical timeframe from UTC M1 candles."""

    tf = normalize_timeframe(timeframe)
    start_utc = _utc_bound(start, name="start")
    end_utc = _utc_bound(end, name="end")
    if minute_frame.empty:
        raise LSEIngestError("Cannot resample an empty M1 frame")
    if (
        not math.isfinite(float(min_bucket_coverage))
        or not 0.0 < float(min_bucket_coverage) <= 1.0
    ):
        raise LSEIngestError("min_bucket_coverage must be in the interval (0, 1]")

    if tf == "1m":
        out = minute_frame.copy()
        out["bar_close_time"] = out.index + _TARGET_DELTAS[tf]
        out["_source_rows"] = 1
    elif tf in _UTC_RESAMPLE_RULES:
        work = minute_frame.copy()
        work["_source_rows"] = 1
        out = (
            work.resample(
                _UTC_RESAMPLE_RULES[tf],
                label="left",
                closed="left",
                origin="start_day",
            )
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                    "_source_rows": "sum",
                }
            )
            .dropna(subset=["open", "high", "low", "close"])
        )
        out.index = pd.DatetimeIndex(out.index, name="timestamp")
        out["bar_close_time"] = out.index + _TARGET_DELTAS[tf]
    elif tf in {"4h", "1d"}:
        out = _wall_clock_resample(
            minute_frame,
            timeframe=tf,
            bar_timezone=bar_timezone,
        )
    else:
        raise LSEIngestError(f"Unsupported LSE target timeframe: {timeframe!r}")

    candidate_buckets = len(out)
    complete = (out.index >= start_utc) & (out["bar_close_time"] <= end_utc)
    out = out.loc[complete].copy()
    dropped_outside_range = candidate_buckets - len(out)
    if out.empty:
        raise LSEIngestError(
            f"No complete {tf} candles remain inside the requested range"
        )
    close_times = pd.DatetimeIndex(
        pd.to_datetime(out["bar_close_time"], utc=True, errors="coerce")
    )
    expected_minutes = (close_times - out.index).total_seconds() / 60.0
    if (
        pd.isna(expected_minutes).any()
        or (expected_minutes <= 0).any()
        or "_source_rows" not in out
    ):
        raise LSEIngestError(f"Cannot calculate {tf} source bucket coverage")
    coverage = out["_source_rows"].to_numpy(dtype=float) / expected_minutes
    low_coverage = coverage + 1e-12 < float(min_bucket_coverage)
    dropped_low_coverage = int(low_coverage.sum())
    out = out.loc[~low_coverage].copy()
    if out.empty:
        raise LSEIngestError(
            f"No {tf} candles meet min_bucket_coverage="
            f"{float(min_bucket_coverage):.6f}"
        )
    out = out.drop(columns=["_source_rows"])
    out.attrs["bucket_quality"] = {
        "candidate_buckets": candidate_buckets,
        "dropped_outside_requested_range": dropped_outside_range,
        "dropped_low_coverage": dropped_low_coverage,
        "min_bucket_coverage": float(min_bucket_coverage),
    }
    return out


def _snapshot_file_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    out = frame.reset_index()
    out.insert(0, "tf", timeframe)
    out.insert(0, "symbol", symbol)
    return out[
        [
            "symbol",
            "tf",
            "timestamp",
            "bar_close_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]


def import_lse_snapshot(
    *,
    output: str | Path,
    symbols: Iterable[str | LSESymbolSpec],
    start: str,
    end: str,
    timeframes: Sequence[str] = DEFAULT_TARGET_TIMEFRAMES,
    transport: str = "rest",
    dataset: Optional[str] = None,
    bar_timezone: str = DEFAULT_BAR_TIMEZONE,
    max_duplicate_ratio: float = 0.001,
    page_limit: int = 5000,
    timeout: float = 60.0,
    retries: int = 3,
    rest_min_interval: float = DEFAULT_REST_MIN_INTERVAL,
    retry_after_seconds: float = DEFAULT_RETRY_AFTER_SECONDS,
    min_bucket_coverage: float = DEFAULT_MIN_BUCKET_COVERAGE,
    max_edge_gap_days: float = DEFAULT_MAX_EDGE_GAP_DAYS,
    max_internal_gap_days: float = DEFAULT_MAX_INTERNAL_GAP_DAYS,
    export_chunk_days: int = DEFAULT_EXPORT_CHUNK_DAYS,
    max_export_jobs: int = DEFAULT_MAX_EXPORT_JOBS,
    release_commit_file: str | Path | None = None,
    environment_lock_file: str | Path | None = None,
    client: Any = None,
) -> LSEImportResult:
    """Create a new immutable snapshot directory from LSE M1 candles."""

    runtime_environment = _runtime_environment_metadata(
        release_commit_file=release_commit_file,
        environment_lock_file=environment_lock_file,
    )
    target = Path(output).expanduser().resolve()
    if target.exists() or target.is_symlink():
        raise LSEIngestError(
            f"Snapshot target already exists; use a new immutable path: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)

    specs = tuple(
        item if isinstance(item, LSESymbolSpec) else parse_symbol_spec(item)
        for item in symbols
    )
    if not specs:
        raise LSEIngestError("At least one LSE symbol is required")
    if len({item.bot_symbol for item in specs}) != len(specs):
        raise LSEIngestError("Duplicate bot symbols are not allowed")

    normalized_timeframes = tuple(
        dict.fromkeys(normalize_timeframe(item) for item in timeframes)
    )
    if SOURCE_TIMEFRAME not in normalized_timeframes:
        normalized_timeframes = (SOURCE_TIMEFRAME, *normalized_timeframes)
    unsupported = set(normalized_timeframes) - set(_TARGET_DELTAS)
    if unsupported:
        raise LSEIngestError(f"Unsupported target timeframe(s): {sorted(unsupported)}")

    start_utc = _utc_bound(start, name="start")
    end_utc = _utc_bound(end, name="end")
    if end_utc <= start_utc:
        raise LSEIngestError("end must be after start")
    transport = str(transport).strip().lower()
    if transport not in {"rest", "export"}:
        raise LSEIngestError("transport must be 'rest' or 'export'")
    if int(retries) < 0:
        raise LSEIngestError("retries must be non-negative")
    if (
        not math.isfinite(float(retry_after_seconds))
        or float(retry_after_seconds) < 0
    ):
        raise LSEIngestError("retry_after_seconds must be finite and non-negative")

    export_chunks: tuple[tuple[str, str], ...] = ()
    if transport == "export":
        export_chunks = _export_date_chunks(
            start,
            end,
            chunk_days=export_chunk_days,
        )
        planned_jobs = len(export_chunks) * len(specs)
        if int(max_export_jobs) < 1:
            raise LSEIngestError("max_export_jobs must be at least 1")
        if planned_jobs > int(max_export_jobs):
            raise LSEIngestError(
                f"LSE export requires {planned_jobs} jobs "
                f"({len(export_chunks)} chunk(s) x {len(specs)} symbol(s)), "
                f"exceeding max_export_jobs={int(max_export_jobs)}; "
                "use REST, narrow the range, or split the import"
            )

    lse_client = client if client is not None else create_lse_client(timeout=timeout)
    resolved_datasets = _resolve_symbol_datasets(
        lse_client,
        specs,
        dataset=dataset,
        retries=int(retries),
        retry_after_seconds=float(retry_after_seconds),
        sleep=time.sleep,
    )
    raw_root = target.parent / f".{target.name}.lse-checkpoint"
    if raw_root.is_symlink():
        raise LSEIngestError(f"LSE checkpoint cannot be a symbolic link: {raw_root}")
    if transport == "export" and raw_root.exists() and not raw_root.is_dir():
        raise LSEIngestError(f"LSE checkpoint path is not a directory: {raw_root}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=str(target.parent))
    )
    symbol_reports: list[dict[str, Any]] = []
    file_reports: list[dict[str, Any]] = []
    completed = False

    try:
        if runtime_environment is not None:
            lock_source = Path(environment_lock_file).expanduser()
            lock_destination = staging / _ENVIRONMENT_LOCK_SNAPSHOT
            try:
                shutil.copyfile(lock_source, lock_destination)
            except OSError as exc:
                raise LSEIngestError(
                    "Cannot embed the resolved Python environment in the snapshot"
                ) from exc
            expected_lock = runtime_environment["lock"]
            if (
                int(lock_destination.stat().st_size) != int(expected_lock["bytes"])
                or _sha256_file(lock_destination) != expected_lock["sha256"]
            ):
                raise LSEIngestError(
                    "Environment lock changed while the snapshot was being created"
                )

        for spec in specs:
            resolved_dataset = resolved_datasets[spec.bot_symbol]
            raw_dir = raw_root / spec.bot_symbol
            if transport == "export":
                raw, request_report = fetch_lse_export(
                    lse_client,
                    provider_symbol=spec.provider_symbol,
                    start=start,
                    end=end,
                    destination=raw_dir,
                    dataset=resolved_dataset,
                    export_chunk_days=export_chunk_days,
                )
            else:
                raw, request_report = fetch_lse_rest(
                    lse_client,
                    provider_symbol=spec.provider_symbol,
                    start=start_utc,
                    end=end_utc,
                    dataset=resolved_dataset,
                    page_limit=page_limit,
                    retries=int(retries),
                    rest_min_interval=rest_min_interval,
                    retry_after_seconds=retry_after_seconds,
                )

            minute_frame, quality = normalize_lse_candles(
                raw,
                expected_provider_symbol=spec.provider_symbol,
                start=start_utc,
                end=end_utc,
                max_duplicate_ratio=max_duplicate_ratio,
                max_edge_gap_days=max_edge_gap_days,
                max_internal_gap_days=max_internal_gap_days,
            )
            derived_rows: dict[str, int] = {}
            derived_quality: dict[str, Mapping[str, Any]] = {}
            for timeframe in normalized_timeframes:
                derived = resample_lse_candles(
                    minute_frame,
                    timeframe=timeframe,
                    start=start_utc,
                    end=end_utc,
                    bar_timezone=bar_timezone,
                    min_bucket_coverage=min_bucket_coverage,
                )
                bucket_quality = dict(derived.attrs.get("bucket_quality", {}))
                destination = staging / f"{spec.bot_symbol}_{timeframe}.parquet"
                _snapshot_file_frame(
                    derived,
                    symbol=spec.bot_symbol,
                    timeframe=timeframe,
                ).to_parquet(destination, index=False)
                derived_rows[timeframe] = len(derived)
                derived_quality[timeframe] = bucket_quality
                file_reports.append(
                    {
                        "path": destination.name,
                        "symbol": spec.bot_symbol,
                        "timeframe": timeframe,
                        "rows": len(derived),
                        "bytes": int(destination.stat().st_size),
                        "sha256": _sha256_file(destination),
                        "start": derived.index[0].isoformat(),
                        "end_open": derived.index[-1].isoformat(),
                        "end_close": pd.Timestamp(
                            derived["bar_close_time"].iloc[-1]
                        ).isoformat(),
                        "bucket_quality": bucket_quality,
                    }
                )

            symbol_reports.append(
                {
                    **spec.to_dict(),
                    "dataset": resolved_dataset,
                    **request_report,
                    "quality": quality,
                    "derived_rows": derived_rows,
                    "derived_quality": derived_quality,
                }
            )

        source_manifest = {
            "schema_version": 1,
            "provider": "london_strategic_edge",
            "sdk": "lse-data",
            "sdk_version": _sdk_version(),
            "requested_dataset": dataset,
            "transport": transport,
            "source_timeframe": SOURCE_TIMEFRAME,
            "requested_start": start_utc.isoformat(),
            "requested_end_exclusive": end_utc.isoformat(),
            "bar_timezone": bar_timezone,
            "timestamp_semantics": "bar_open_utc",
            "price_basis": "provider_ohlc_unsuitable_for_exact_broker_execution",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": sorted(file_reports, key=lambda item: item["path"]),
            "symbols": symbol_reports,
        }
        if runtime_environment is not None:
            source_manifest["runtime_environment"] = runtime_environment
        (staging / "source_manifest.json").write_text(
            json.dumps(source_manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        dataset_snapshot = HistoricalDataset.load(staging)
        readiness = dataset_snapshot.audit_readiness()
        dataset_snapshot.write_manifest(staging / "manifest.json")
        if target.exists() or target.is_symlink():
            raise LSEIngestError(
                f"Snapshot target appeared during import; refusing to replace it: {target}"
            )
        staging.replace(target)
        completed = True
        if transport == "export":
            shutil.rmtree(raw_root, ignore_errors=True)
        return LSEImportResult(
            output=target,
            manifest_sha256=dataset_snapshot.manifest_sha256,
            readiness=readiness,
            symbols=tuple(symbol_reports),
        )
    finally:
        if not completed:
            shutil.rmtree(staging, ignore_errors=True)
