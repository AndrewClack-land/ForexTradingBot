"""Causal EODHD daily-price ingest for the ORCA research universe.

The job is intentionally separate from the monitor and the MT5 process.  It
downloads the paper's exact 24 US-listed instruments, maps each exchange date
to a conservative *availability* timestamp, builds one fully aligned panel,
and atomically replaces the last-known-good file only after the monitor's own
loader accepts the candidate.

The API token is never accepted on the command line and request failures are
re-raised without their token-bearing URL.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Optional
from urllib.parse import quote, quote_plus, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from monitoring.orca_monitor import load_price_panel


EODHD_ENDPOINT = "https://eodhd.com/api/eod"
ORCA_EODHD_SCHEMA = "orca-eodhd-panel-v1"
ORCA_PAPER_UNIVERSE = (
    "SPY",
    "QQQ",
    "IWM",
    "XLF",
    "XLE",
    "XLK",
    "XLV",
    "XLU",
    "XLP",
    "XLY",
    "XLI",
    "XLB",
    "XLRE",
    "EFA",
    "EEM",
    "VGK",
    "EWJ",
    "TLT",
    "IEF",
    "LQD",
    "HYG",
    "GLD",
    "USO",
    "UUP",
)


class OrcaEodhdIngestError(ValueError):
    """A sanitized, fail-closed ingest error."""


def _utc(value: Any, *, name: str) -> datetime:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise OrcaEodhdIngestError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise OrcaEodhdIngestError(f"{name} must be an integer")
    parsed = int(value)
    if parsed <= 0:
        raise OrcaEodhdIngestError(f"{name} must be positive")
    return parsed


def _redact_secret(message: Any, secret: str) -> str:
    text = str(message)
    if not secret:
        return text
    for token in {secret, quote_plus(secret), quote(secret, safe="")}:
        if token:
            text = text.replace(token, "[REDACTED]")
    return text


@dataclass(frozen=True)
class OrcaEodhdConfig:
    output_path: Path
    minimum_aligned_rows: int = 372
    ffill_limit: int = 5
    availability_hour_utc: int = 12
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_path", Path(self.output_path))
        object.__setattr__(
            self,
            "minimum_aligned_rows",
            _positive_int(self.minimum_aligned_rows, name="minimum_aligned_rows"),
        )
        object.__setattr__(
            self,
            "ffill_limit",
            _positive_int(self.ffill_limit, name="ffill_limit"),
        )
        if (
            isinstance(self.availability_hour_utc, bool)
            or not isinstance(self.availability_hour_utc, int)
            or not 0 <= self.availability_hour_utc <= 23
        ):
            raise OrcaEodhdIngestError("availability_hour_utc must be in [0, 23]")
        timeout = float(self.timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise OrcaEodhdIngestError("timeout_seconds must be positive")
        object.__setattr__(self, "timeout_seconds", timeout)
        if self.output_path.suffix.lower() not in {".csv", ".parquet", ".pq"}:
            raise OrcaEodhdIngestError("output_path must end in .csv, .parquet, or .pq")


class EodhdClient:
    """Small token-redacting client for the official EOD endpoint."""

    def __init__(
        self,
        api_token: str,
        *,
        timeout_seconds: float = 30.0,
        opener: Optional[Callable[..., Any]] = None,
    ) -> None:
        token = str(api_token or "").strip()
        if not token:
            raise OrcaEodhdIngestError("EODHD_API_TOKEN is required")
        self._api_token = token
        self._timeout_seconds = float(timeout_seconds)
        self._opener = opener or urlopen

    def fetch_daily(self, ticker: str) -> list[Mapping[str, Any]]:
        if ticker not in ORCA_PAPER_UNIVERSE:
            raise OrcaEodhdIngestError(f"unsupported ORCA ticker {ticker!r}")
        query = urlencode(
            {
                "api_token": self._api_token,
                "fmt": "json",
                "period": "d",
                "order": "a",
            }
        )
        url = f"{EODHD_ENDPOINT}/{ticker}.US?{query}"
        request = Request(
            url,
            method="GET",
            headers={
                "Accept": "application/json",
                "User-Agent": "forexbot-orca-eodhd/1",
            },
        )
        try:
            with self._opener(request, timeout=self._timeout_seconds) as response:
                status = int(getattr(response, "status", 200))
                if status != 200:
                    raise OrcaEodhdIngestError(
                        f"EODHD returned HTTP {status} for {ticker}"
                    )
                payload = json.loads(response.read())
        except OrcaEodhdIngestError:
            raise
        except Exception:
            # urllib exceptions commonly contain the full token-bearing URL.
            raise OrcaEodhdIngestError(f"EODHD request failed for {ticker}") from None
        if not isinstance(payload, list):
            raise OrcaEodhdIngestError(
                f"EODHD response for {ticker} must be a JSON array"
            )
        if any(not isinstance(row, Mapping) for row in payload):
            raise OrcaEodhdIngestError(
                f"EODHD response for {ticker} contains a non-object row"
            )
        return payload


def fetch_orca_universe(client: EodhdClient) -> dict[str, list[Mapping[str, Any]]]:
    """Fetch every paper-universe member or return nothing."""

    return {ticker: client.fetch_daily(ticker) for ticker in ORCA_PAPER_UNIVERSE}


def _exchange_date(value: Any, *, ticker: str, position: int) -> date:
    raw = str(value or "").strip()
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise OrcaEodhdIngestError(
            f"{ticker} row {position} has an invalid exchange date"
        ) from exc
    if parsed.isoformat() != raw:
        raise OrcaEodhdIngestError(f"{ticker} row {position} date must be YYYY-MM-DD")
    return parsed


def _adjusted_close(value: Any, *, ticker: str, position: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise OrcaEodhdIngestError(
            f"{ticker} row {position} adjusted_close must be numeric"
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise OrcaEodhdIngestError(
            f"{ticker} row {position} adjusted_close must be finite and positive"
        )
    return parsed


def _ticker_series(
    ticker: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    as_of_utc: datetime,
    availability_hour_utc: int,
) -> pd.Series:
    timestamps: list[datetime] = []
    values: list[float] = []
    seen_dates: set[date] = set()
    for position, row in enumerate(rows):
        exchange_date = _exchange_date(
            row.get("date"), ticker=ticker, position=position
        )
        if exchange_date in seen_dates:
            raise OrcaEodhdIngestError(
                f"{ticker} contains duplicate date {exchange_date.isoformat()}"
            )
        seen_dates.add(exchange_date)
        available = datetime.combine(
            exchange_date + timedelta(days=1),
            time(hour=availability_hour_utc),
            tzinfo=timezone.utc,
        )
        if available > as_of_utc:
            continue
        timestamps.append(available)
        values.append(
            _adjusted_close(row.get("adjusted_close"), ticker=ticker, position=position)
        )
    if not timestamps:
        raise OrcaEodhdIngestError(f"{ticker} has no causally available rows")
    series = pd.Series(
        values,
        index=pd.DatetimeIndex(timestamps, tz="UTC"),
        name=ticker,
        dtype=float,
    ).sort_index()
    if series.index.has_duplicates:
        raise OrcaEodhdIngestError(f"{ticker} availability timestamps are duplicated")
    return series


def build_aligned_eodhd_panel(
    responses: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    as_of_utc: datetime,
    minimum_aligned_rows: int = 372,
    ffill_limit: int = 5,
    availability_hour_utc: int = 12,
) -> pd.DataFrame:
    """Build a point-in-time 24-asset panel from adjusted closes."""

    as_of = _utc(as_of_utc, name="as_of_utc")
    minimum = _positive_int(minimum_aligned_rows, name="minimum_aligned_rows")
    limit = _positive_int(ffill_limit, name="ffill_limit")
    if not 0 <= availability_hour_utc <= 23:
        raise OrcaEodhdIngestError("availability_hour_utc must be in [0, 23]")
    expected = set(ORCA_PAPER_UNIVERSE)
    actual = set(responses)
    if actual != expected:
        raise OrcaEodhdIngestError(
            "EODHD universe mismatch; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    series = [
        _ticker_series(
            ticker,
            responses[ticker],
            as_of_utc=as_of,
            availability_hour_utc=availability_hour_utc,
        )
        for ticker in ORCA_PAPER_UNIVERSE
    ]
    raw = pd.concat(series, axis=1).sort_index()
    panel = raw.ffill(limit=limit).dropna(axis=0, how="any")
    panel = panel.loc[:, list(ORCA_PAPER_UNIVERSE)].astype(float)
    if len(panel) < minimum:
        raise OrcaEodhdIngestError(
            f"aligned ORCA panel needs at least {minimum} rows; received {len(panel)}"
        )
    if panel.index[-1].to_pydatetime() > as_of:
        raise OrcaEodhdIngestError("aligned panel contains a future availability row")
    if not np.isfinite(panel.to_numpy()).all() or (panel.to_numpy() <= 0.0).any():
        raise OrcaEodhdIngestError("aligned panel contains invalid prices")
    panel.index.name = "timestamp"
    return panel


def _write_candidate(panel: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        panel.to_csv(path, index=True)
    elif suffix in {".parquet", ".pq"}:
        panel.to_parquet(path, index=True)
    else:  # guarded by OrcaEodhdConfig, retained for direct testing
        raise OrcaEodhdIngestError("unsupported output format")
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _validate_candidate(path: Path, expected: pd.DataFrame) -> None:
    try:
        loaded = load_price_panel(path)
    except Exception as exc:
        raise OrcaEodhdIngestError(
            f"monitor rejected candidate panel: {type(exc).__name__}"
        ) from None
    if list(loaded.columns) != list(ORCA_PAPER_UNIVERSE):
        raise OrcaEodhdIngestError("monitor loader changed the ORCA symbol contract")
    if not loaded.index.equals(expected.index):
        raise OrcaEodhdIngestError("monitor loader changed availability timestamps")
    if loaded.shape != expected.shape:
        raise OrcaEodhdIngestError("monitor loader changed panel dimensions")
    if not np.allclose(
        loaded.to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise OrcaEodhdIngestError("monitor loader changed adjusted-close values")


def publish_panel_atomically(panel: pd.DataFrame, config: OrcaEodhdConfig) -> None:
    """Validate a same-directory temporary file, then replace last-good."""

    target = config.output_path
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.",
        suffix=target.suffix,
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_candidate(panel, temporary)
        _validate_candidate(temporary, panel)
        os.replace(temporary, target)
        if os.name != "nt":
            directory_fd = os.open(str(target.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def refresh_eodhd_panel(
    *,
    api_token: str,
    config: OrcaEodhdConfig,
    as_of_utc: Optional[datetime] = None,
    client: Optional[EodhdClient] = None,
) -> dict[str, Any]:
    """Fetch, causally align, validate, and atomically publish one panel."""

    as_of = _utc(
        as_of_utc or datetime.now(timezone.utc),
        name="as_of_utc",
    )
    source = client or EodhdClient(
        api_token,
        timeout_seconds=config.timeout_seconds,
    )
    responses = fetch_orca_universe(source)
    panel = build_aligned_eodhd_panel(
        responses,
        as_of_utc=as_of,
        minimum_aligned_rows=config.minimum_aligned_rows,
        ffill_limit=config.ffill_limit,
        availability_hour_utc=config.availability_hour_utc,
    )
    publish_panel_atomically(panel, config)
    return {
        "schema": ORCA_EODHD_SCHEMA,
        "status": "PUBLISHED",
        "output_path": str(config.output_path),
        "asset_count": len(panel.columns),
        "aligned_rows": len(panel),
        "available_from_utc": panel.index[0].isoformat(),
        "available_through_utc": panel.index[-1].isoformat(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish a causal adjusted-close panel for ORCA"
    )
    parser.add_argument(
        "--output",
        default=os.getenv(
            "ORCA_EODHD_OUTPUT_PATH",
            "/srv/forexbot-backtest/orca/prices.parquet",
        ),
    )
    parser.add_argument(
        "--minimum-aligned-rows",
        type=int,
        default=int(os.getenv("ORCA_EODHD_MIN_ALIGNED_ROWS", "372")),
    )
    parser.add_argument(
        "--ffill-limit",
        type=int,
        default=int(os.getenv("ORCA_EODHD_FFILL_LIMIT", "5")),
    )
    parser.add_argument(
        "--availability-hour-utc",
        type=int,
        default=int(os.getenv("ORCA_EODHD_AVAILABILITY_HOUR_UTC", "12")),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.getenv("ORCA_EODHD_TIMEOUT_SECONDS", "30")),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    token = str(os.getenv("EODHD_API_TOKEN", "")).strip()
    arguments = _parser().parse_args(argv)
    try:
        config = OrcaEodhdConfig(
            output_path=Path(arguments.output),
            minimum_aligned_rows=arguments.minimum_aligned_rows,
            ffill_limit=arguments.ffill_limit,
            availability_hour_utc=arguments.availability_hour_utc,
            timeout_seconds=arguments.timeout_seconds,
        )
        result = refresh_eodhd_panel(api_token=token, config=config)
    except Exception as exc:
        message = _redact_secret(str(exc), token)
        print(
            f"ORCA EODHD ingest failed: {type(exc).__name__}: {message}",
            file=os.sys.stderr,
        )
        return 1
    print(
        "ORCA EODHD panel published: "
        f"assets={result['asset_count']} rows={result['aligned_rows']} "
        f"through={result['available_through_utc']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())


__all__ = [
    "EODHD_ENDPOINT",
    "ORCA_EODHD_SCHEMA",
    "ORCA_PAPER_UNIVERSE",
    "EodhdClient",
    "OrcaEodhdConfig",
    "OrcaEodhdIngestError",
    "build_aligned_eodhd_panel",
    "fetch_orca_universe",
    "main",
    "publish_panel_atomically",
    "refresh_eodhd_panel",
]
