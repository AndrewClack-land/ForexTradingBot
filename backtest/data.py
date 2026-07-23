"""Causal OHLC snapshot loading and audit support.

MT5 timestamps identify the *opening* time of a candle.  Consequently an
as-of lookup must compare ``open_time + timeframe`` with the decision time;
using ``df.loc[:decision_time]`` would leak the still-forming candle.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import pandas as pd


LIVE_CLOSED_BAR_LIMIT = 299

_TF_ALIASES: Dict[str, str] = {
    "m1": "1m",
    "1m": "1m",
    "m5": "5m",
    "5m": "5m",
    "m15": "15m",
    "15m": "15m",
    "h1": "1h",
    "1h": "1h",
    "h4": "4h",
    "4h": "4h",
    "d": "1d",
    "d1": "1d",
    "1d": "1d",
}

_TF_DURATIONS: Dict[str, pd.Timedelta] = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}

_STRATEGY_KEYS: Dict[str, str] = {
    "1d": "D",
    "4h": "4H",
    "1h": "1H",
    "15m": "15M",
    "5m": "5M",
    "1m": "1M",
}

_REQUIRED_OHLC = ("open", "high", "low", "close")


class DataValidationError(ValueError):
    """Raised when a snapshot cannot be interpreted as valid causal OHLC."""


def normalize_timeframe(value: Any) -> str:
    key = str(value or "").strip().lower()
    try:
        return _TF_ALIASES[key]
    except KeyError as exc:
        raise DataValidationError(f"Unsupported timeframe: {value!r}") from exc


def _utc_timestamp(value: Any) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except Exception as exc:  # pragma: no cover - pandas controls exact errors
        raise DataValidationError(f"Invalid timestamp: {value!r}") from exc
    if pd.isna(ts):
        raise DataValidationError(f"Invalid timestamp: {value!r}")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _infer_key_from_name(path: Path) -> Optional[Tuple[str, str]]:
    stem = path.stem
    if "_" not in stem:
        return None
    symbol, raw_tf = stem.rsplit("_", 1)
    try:
        tf = normalize_timeframe(raw_tf)
    except DataValidationError:
        return None
    symbol = symbol.strip().upper()
    return (symbol, tf) if symbol else None


def _parse_timestamp_column(values: pd.Series, *, source: Path) -> pd.DatetimeIndex:
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        finite = numeric.dropna()
        if finite.empty:
            raise DataValidationError(f"{source}: timestamp column is empty")
        magnitude = abs(float(finite.median()))
        unit = "ms" if magnitude >= 100_000_000_000 else "s"
        parsed = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    else:
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        bad = int(parsed.isna().sum())
        raise DataValidationError(f"{source}: {bad} invalid timestamp value(s)")
    return pd.DatetimeIndex(parsed, name="timestamp")


def _validate_ohlc(frame: pd.DataFrame, *, source: Path) -> None:
    missing = [column for column in _REQUIRED_OHLC if column not in frame.columns]
    if missing:
        raise DataValidationError(f"{source}: missing OHLC columns: {missing}")

    for column in _REQUIRED_OHLC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].isna() | ~frame[column].map(math.isfinite)
        if invalid.any():
            raise DataValidationError(
                f"{source}: {int(invalid.sum())} non-numeric/non-finite {column} value(s)"
            )
        if (frame[column] <= 0).any():
            raise DataValidationError(f"{source}: OHLC prices must be positive")

    high = frame["high"]
    low = frame["low"]
    body_max = frame[["open", "close"]].max(axis=1)
    body_min = frame[["open", "close"]].min(axis=1)
    invalid_range = (high < low) | (high < body_max) | (low > body_min)
    if invalid_range.any():
        raise DataValidationError(
            f"{source}: {int(invalid_range.sum())} impossible OHLC candle(s)"
        )

    if "volume" in frame.columns:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
        if frame["volume"].isna().any() or (frame["volume"] < 0).any():
            raise DataValidationError(f"{source}: invalid volume value(s)")


def _normalize_frame(frame: pd.DataFrame, *, source: Path) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise DataValidationError(f"{source}: empty candle file")
    out = frame.copy()

    if "timestamp" in out.columns:
        index = _parse_timestamp_column(out["timestamp"], source=source)
    elif "time" in out.columns:
        index = _parse_timestamp_column(out["time"], source=source)
    elif isinstance(out.index, pd.DatetimeIndex):
        index = _parse_timestamp_column(pd.Series(out.index), source=source)
    else:
        raise DataValidationError(f"{source}: no timestamp/time column or datetime index")

    _validate_ohlc(out, source=source)
    columns = list(_REQUIRED_OHLC)
    if "volume" in out.columns:
        columns.append("volume")
    out = out[columns].copy()
    out.index = index
    return out


def _read_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            return pd.read_json(path, orient="records")
        except ValueError as exc:
            raise DataValidationError(f"{path}: invalid records JSON") from exc
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            raise DataValidationError(f"{path}: cannot read Parquet: {exc}") from exc
    raise DataValidationError(f"Unsupported candle file: {path}")


@dataclass(frozen=True)
class SeriesCoverage:
    symbol: str
    timeframe: str
    rows: int
    raw_rows: int
    duplicate_rows: int
    start: pd.Timestamp
    end_open: pd.Timestamp
    end_close: pd.Timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "rows": self.rows,
            "raw_rows": self.raw_rows,
            "duplicate_rows": self.duplicate_rows,
            "start": self.start.isoformat(),
            "end_open": self.end_open.isoformat(),
            "end_close": self.end_close.isoformat(),
        }


class HistoricalDataset:
    """Immutable in-memory view of a broker-native multi-timeframe snapshot."""

    def __init__(
        self,
        root: Path,
        frames: Mapping[Tuple[str, str], pd.DataFrame],
        *,
        source_files: Iterable[Mapping[str, Any]],
        raw_counts: Mapping[Tuple[str, str], int],
    ) -> None:
        self.root = Path(root)
        self._frames = {
            (symbol.upper(), normalize_timeframe(tf)): frame.copy()
            for (symbol, tf), frame in frames.items()
        }
        self._raw_counts = dict(raw_counts)
        self._source_files = tuple(dict(item) for item in source_files)
        self._coverage = self._build_coverage()
        manifest_payload: Dict[str, Any] = {
            "schema_version": 1,
            "files": list(self._source_files),
            "series": [item.to_dict() for item in self._coverage],
        }
        self._manifest = dict(manifest_payload)
        self._manifest["manifest_sha256"] = _canonical_hash(manifest_payload)

    @classmethod
    def load(cls, root: str | Path) -> "HistoricalDataset":
        base = Path(root).expanduser().resolve()
        if not base.exists() or not base.is_dir():
            raise FileNotFoundError(f"Snapshot directory does not exist: {base}")

        paths = sorted(
            path
            for path in base.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".json", ".parquet", ".pq"}
            and path.name.lower() != "manifest.json"
        )
        if not paths:
            raise FileNotFoundError(f"No JSON/Parquet candle files found under {base}")

        grouped: Dict[Tuple[str, str], list[pd.DataFrame]] = {}
        raw_counts: Dict[Tuple[str, str], int] = {}
        source_files: list[Dict[str, Any]] = []

        for path in paths:
            raw = _read_file(path)
            fallback_key = _infer_key_from_name(path)

            has_symbol = "symbol" in raw.columns
            has_tf = "tf" in raw.columns or "timeframe" in raw.columns
            if has_symbol and has_tf:
                tf_column = "tf" if "tf" in raw.columns else "timeframe"
                groups = raw.groupby(["symbol", tf_column], sort=True, dropna=False)
                keyed_frames = [
                    ((str(symbol).upper(), normalize_timeframe(tf)), part)
                    for (symbol, tf), part in groups
                ]
            elif fallback_key is not None:
                keyed_frames = [(fallback_key, raw)]
            else:
                raise DataValidationError(
                    f"{path}: cannot infer symbol/timeframe from columns or filename"
                )

            for key, part in keyed_frames:
                normalized = _normalize_frame(part, source=path)
                grouped.setdefault(key, []).append(normalized)
                raw_counts[key] = raw_counts.get(key, 0) + len(normalized)

            source_files.append(
                {
                    "path": path.relative_to(base).as_posix(),
                    "size": int(path.stat().st_size),
                    "sha256": _sha256_file(path),
                }
            )

        frames: Dict[Tuple[str, str], pd.DataFrame] = {}
        for key, parts in grouped.items():
            combined = pd.concat(parts, axis=0)
            combined = combined.sort_index(kind="stable")
            combined = combined[~combined.index.duplicated(keep="last")]
            if combined.empty:
                raise DataValidationError(f"No candles remain after dedupe for {key}")
            if not combined.index.is_monotonic_increasing or not combined.index.is_unique:
                raise DataValidationError(f"Timestamp normalization failed for {key}")
            frames[key] = combined

        return cls(
            base,
            frames,
            source_files=source_files,
            raw_counts=raw_counts,
        )

    @property
    def symbols(self) -> Tuple[str, ...]:
        return tuple(sorted({symbol for symbol, _ in self._frames}))

    @property
    def keys(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(sorted(self._frames))

    @property
    def manifest(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self._manifest))

    @property
    def manifest_sha256(self) -> str:
        return str(self._manifest["manifest_sha256"])

    def write_manifest(self, destination: str | Path) -> Path:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def _build_coverage(self) -> Tuple[SeriesCoverage, ...]:
        items: list[SeriesCoverage] = []
        for (symbol, tf), frame in sorted(self._frames.items()):
            raw_rows = int(self._raw_counts.get((symbol, tf), len(frame)))
            items.append(
                SeriesCoverage(
                    symbol=symbol,
                    timeframe=tf,
                    rows=len(frame),
                    raw_rows=raw_rows,
                    duplicate_rows=max(0, raw_rows - len(frame)),
                    start=frame.index[0],
                    end_open=frame.index[-1],
                    end_close=frame.index[-1] + _TF_DURATIONS[tf],
                )
            )
        return tuple(items)

    def coverage(self) -> Tuple[SeriesCoverage, ...]:
        return self._coverage

    def get_frame(self, symbol: str, timeframe: str) -> pd.DataFrame:
        key = (str(symbol).upper(), normalize_timeframe(timeframe))
        try:
            return self._frames[key].copy()
        except KeyError as exc:
            raise KeyError(f"No candle series for {key[0]} {key[1]}") from exc

    def frame_asof(
        self,
        symbol: str,
        timeframe: str,
        at: Any,
        *,
        limit: int = LIVE_CLOSED_BAR_LIMIT,
    ) -> pd.DataFrame:
        if limit <= 0:
            raise ValueError("limit must be positive")
        tf = normalize_timeframe(timeframe)
        frame = self._frames.get((str(symbol).upper(), tf))
        if frame is None:
            return pd.DataFrame(columns=list(_REQUIRED_OHLC))
        decision_time = _utc_timestamp(at)
        close_times = frame.index + _TF_DURATIONS[tf]
        causal = frame.loc[close_times <= decision_time]
        return causal.tail(int(limit)).copy()

    def frames_asof(
        self,
        symbol: str,
        at: Any,
        *,
        limit: int = LIVE_CLOSED_BAR_LIMIT,
        strategy_keys: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        available = {
            tf for candidate, tf in self._frames if candidate == str(symbol).upper()
        }
        out: Dict[str, pd.DataFrame] = {}
        for tf in sorted(available, key=lambda value: _TF_DURATIONS[value], reverse=True):
            key = _STRATEGY_KEYS[tf] if strategy_keys else tf
            out[key] = self.frame_asof(symbol, tf, at, limit=limit)
        return out

    def audit_readiness(
        self,
        *,
        min_d1_bars: int = 365,
        min_months: int = 12,
        symbols: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        if min_d1_bars <= 0 or min_months <= 0:
            raise ValueError("readiness thresholds must be positive")
        selected = tuple(sorted({s.upper() for s in symbols})) if symbols else self.symbols
        reasons: list[str] = []
        per_symbol: Dict[str, Any] = {}

        for symbol in selected:
            frame = self._frames.get((symbol, "1d"))
            if frame is None or frame.empty:
                reasons.append(f"{symbol}: D1 series is missing")
                per_symbol[symbol] = {"d1_bars": 0, "months": 0.0, "ready": False}
                continue
            start = frame.index[0]
            end = frame.index[-1] + _TF_DURATIONS["1d"]
            days = max(0.0, (end - start).total_seconds() / 86400.0)
            months = days / (365.2425 / 12.0)
            symbol_reasons: list[str] = []
            if len(frame) < min_d1_bars:
                symbol_reasons.append(
                    f"{symbol}: only {len(frame)} D1 bars; at least {min_d1_bars} required"
                )
            required_end = start + pd.DateOffset(months=int(min_months))
            if end < required_end:
                symbol_reasons.append(
                    f"{symbol}: only {months:.2f} months; at least {min_months} required"
                )
            reasons.extend(symbol_reasons)
            per_symbol[symbol] = {
                "d1_bars": int(len(frame)),
                "months": round(months, 3),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "ready": not symbol_reasons,
            }

        return {
            "wfo_ready": not reasons,
            "min_d1_bars": int(min_d1_bars),
            "min_months": int(min_months),
            "symbols": per_symbol,
            "reasons": reasons,
        }
