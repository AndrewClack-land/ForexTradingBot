"""Read-only preflight profiling of a candidate executed-trade tape.

The sealed converter in :mod:`backtest.orderflow_ingest` is deliberately
intolerant: a wrong ``--tick-size``, an undeclared third aggressor label or a
silently truncated file aborts the whole build.  This module answers those
questions *before* a long build, without writing or sealing anything.

It is lenient on purpose.  Unparseable rows and unexpected aggressor labels are
counted and sampled rather than raised, because "this vendor emits a third
``N`` bucket for 4% of prints" is exactly the finding that decides whether a
tape is usable at all.  Nothing here may ever produce sidecar events.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from math import gcd
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from .orderflow_ingest import (
    TIMESTAMP_FORMATS,
    TapeColumns,
    TapeIngestError,
    _parse_timestamp,
)


BAR_LENGTH = pd.Timedelta(minutes=15)
_MAX_BAD_ROW_SAMPLES = 20
_MAX_DISTINCT_PRICES = 2_000_000


@dataclass
class _Accumulator:
    rows: int = 0
    bad_rows: int = 0
    unsorted_pairs: int = 0


def _percentiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def pick(fraction: float) -> float:
        index = int(round(fraction * (len(ordered) - 1)))
        return float(ordered[index])

    return {
        "min": float(ordered[0]),
        "p05": pick(0.05),
        "p50": pick(0.50),
        "p95": pick(0.95),
        "max": float(ordered[-1]),
    }


def _recommended_tick_size(prices: set[str]) -> tuple[str | None, str | None]:
    """Return the coarsest and the finest tick grid consistent with the prices.

    The coarsest grid is the greatest common divisor of every observed price.
    On a small sample it can overstate the real instrument tick, so the finest
    grid (the smallest quotation step actually written by the vendor) is
    reported alongside it and the caller warns when they disagree.
    """

    decimals: List[Decimal] = []
    for text in prices:
        try:
            value = Decimal(text)
        except InvalidOperation:
            return (None, None)
        if not value.is_finite() or value <= 0:
            return (None, None)
        decimals.append(value)
    if not decimals:
        return (None, None)
    places = max(max(0, -value.as_tuple().exponent) for value in decimals)
    scale = Decimal(10) ** places
    finest = f"{(Decimal(1) / scale).normalize():f}"
    scaled = [int(value * scale) for value in decimals]
    common = 0
    for item in scaled:
        common = gcd(common, abs(item))
        if common == 1:
            break
    if common == 0:
        return (None, finest)
    return (f"{(Decimal(common) / scale).normalize():f}", finest)


def inspect_csv_tape(
    paths: Sequence[Path],
    *,
    columns: TapeColumns,
    timestamp_format: str = "iso8601",
    naive_is_utc: bool = False,
    limit_rows: int | None = None,
) -> Dict[str, Any]:
    """Profile one or more CSV tapes and report what the converter will need."""

    if timestamp_format not in TIMESTAMP_FORMATS:
        raise TapeIngestError(f"unsupported timestamp format {timestamp_format!r}")
    if limit_rows is not None and limit_rows < 1:
        raise TapeIngestError("limit_rows must be at least 1")

    state = _Accumulator()
    aggressor_labels: Counter[str] = Counter()
    bad_samples: List[Dict[str, str]] = []
    distinct_prices: set[str] = set()
    prices_overflowed = False
    bar_trades: Counter[pd.Timestamp] = Counter()
    bar_max_gap: Dict[pd.Timestamp, float] = {}
    last_timestamp: pd.Timestamp | None = None
    last_in_bar: Dict[pd.Timestamp, pd.Timestamp] = {}
    naive_seen = 0
    sizes_seen = 0
    size_min: Decimal | None = None
    size_max: Decimal | None = None
    headers: Dict[str, List[str]] = {}

    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers[path.name] = list(reader.fieldnames or [])
            for index, row in enumerate(reader, start=2):
                if limit_rows is not None and state.rows >= limit_rows:
                    break
                state.rows += 1
                location = f"{path.name}:{index}"
                raw_timestamp = row.get(columns.timestamp)
                if timestamp_format == "iso8601":
                    try:
                        probe = pd.Timestamp(str(raw_timestamp or "").strip())
                    except Exception:
                        probe = None
                    if probe is not None and not pd.isna(probe) and probe.tzinfo is None:
                        naive_seen += 1
                try:
                    timestamp = _parse_timestamp(
                        raw_timestamp,
                        field="timestamp",
                        location=location,
                        timestamp_format=timestamp_format,
                        naive_is_utc=True,
                    )
                except TapeIngestError as exc:
                    state.bad_rows += 1
                    if len(bad_samples) < _MAX_BAD_ROW_SAMPLES:
                        bad_samples.append({"location": location, "error": str(exc)})
                    continue

                label = str(row.get(columns.aggressor) or "").strip().upper()
                aggressor_labels[label or "<empty>"] += 1

                row_errors: List[str] = []
                price_text = str(row.get(columns.price) or "").strip()
                try:
                    price = Decimal(price_text)
                except InvalidOperation:
                    row_errors.append("unparseable price")
                else:
                    if not price.is_finite() or price <= 0:
                        row_errors.append("price must be finite and positive")

                size_text = str(row.get(columns.size) or "").strip()
                try:
                    size = Decimal(size_text)
                except InvalidOperation:
                    row_errors.append("unparseable size")
                else:
                    if not size.is_finite() or size <= 0:
                        row_errors.append("size must be finite and positive")

                if row_errors:
                    state.bad_rows += 1
                    if len(bad_samples) < _MAX_BAD_ROW_SAMPLES:
                        bad_samples.append(
                            {"location": location, "error": "; ".join(row_errors)}
                        )
                    continue

                if len(distinct_prices) < _MAX_DISTINCT_PRICES:
                    distinct_prices.add(price_text)
                else:
                    prices_overflowed = True
                sizes_seen += 1
                size_min = size if size_min is None else min(size_min, size)
                size_max = size if size_max is None else max(size_max, size)
                if last_timestamp is not None and timestamp < last_timestamp:
                    state.unsorted_pairs += 1
                last_timestamp = timestamp

                bar_open = timestamp.floor("15min")
                bar_trades[bar_open] += 1
                previous = last_in_bar.get(bar_open)
                if previous is not None:
                    gap = (timestamp - previous) / pd.Timedelta(seconds=1)
                    if gap > bar_max_gap.get(bar_open, 0.0):
                        bar_max_gap[bar_open] = gap
                else:
                    bar_max_gap.setdefault(bar_open, 0.0)
                last_in_bar[bar_open] = timestamp

    bars = sorted(bar_trades)
    first_bar = bars[0] if bars else None
    last_bar = bars[-1] if bars else None
    expected_bars = (
        int((last_bar - first_bar) / BAR_LENGTH) + 1 if first_bar is not None else 0
    )
    recommended_tick, finest_tick = (
        (None, None)
        if prices_overflowed
        else _recommended_tick_size(distinct_prices)
    )

    suggested: Dict[str, Any] = {
        "tick_size": recommended_tick,
        "timestamp_format": timestamp_format,
        "naive_timestamps_are_utc": bool(naive_is_utc or naive_seen),
    }
    if len(aggressor_labels) == 2 and "<empty>" not in aggressor_labels:
        first, second = sorted(aggressor_labels)
        suggested["aggressor_labels"] = {"candidates": [first, second]}
    if first_bar is not None and last_bar is not None:
        suggested["covers_from"] = first_bar.strftime("%Y-%m-%dT%H:%M:%SZ")
        suggested["covers_through"] = (last_bar + BAR_LENGTH).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    return {
        "inputs": [str(path) for path in paths],
        "headers": headers,
        "columns": {
            "timestamp": columns.timestamp,
            "price": columns.price,
            "size": columns.size,
            "aggressor": columns.aggressor,
            "arrival": columns.arrival,
        },
        "rows_read": state.rows,
        "unparseable_rows": state.bad_rows,
        "unparseable_samples": bad_samples,
        "out_of_order_rows": state.unsorted_pairs,
        "tz_naive_timestamps": naive_seen,
        "aggressor_labels": dict(sorted(aggressor_labels.items())),
        "distinct_prices": (
            None if prices_overflowed else len(distinct_prices)
        ),
        "recommended_tick_size": recommended_tick,
        "finest_observed_tick_size": finest_tick,
        "size": {
            "rows": sizes_seen,
            "min": None if size_min is None else f"{size_min:f}",
            "max": None if size_max is None else f"{size_max:f}",
        },
        "coverage": {
            "first_bar_open": None if first_bar is None else first_bar.isoformat(),
            "last_bar_open": None if last_bar is None else last_bar.isoformat(),
            "bars_with_trades": len(bars),
            "bars_in_span": expected_bars,
            "bars_missing": max(0, expected_bars - len(bars)),
        },
        "trades_per_bar": _percentiles([float(v) for v in bar_trades.values()]),
        "intrabar_gap_seconds": _percentiles(list(bar_max_gap.values())),
        "suggested_converter_flags": suggested,
        "warnings": _warnings(
            aggressor_labels=aggressor_labels,
            bad_rows=state.bad_rows,
            unsorted=state.unsorted_pairs,
            missing_bars=max(0, expected_bars - len(bars)),
            recommended_tick=recommended_tick,
            finest_tick=finest_tick,
        ),
    }


def _warnings(
    *,
    aggressor_labels: Counter[str],
    bad_rows: int,
    unsorted: int,
    missing_bars: int,
    recommended_tick: str | None,
    finest_tick: str | None,
) -> List[str]:
    messages: List[str] = []
    if len(aggressor_labels) > 2:
        messages.append(
            "more than two aggressor labels: every label must be explicitly "
            "declared buy or sell, and an 'unknown' bucket means the venue did "
            "not report the aggressor for those prints"
        )
    if "<empty>" in aggressor_labels:
        messages.append("some prints carry no aggressor label at all")
    if bad_rows:
        messages.append(f"{bad_rows} row(s) could not be parsed")
    if unsorted:
        messages.append(
            f"{unsorted} row(s) are out of chronological order; the converter "
            "requires a sorted tape"
        )
    if missing_bars:
        messages.append(
            f"{missing_bars} M15 bar(s) inside the observed span carry no "
            "prints; confirm these are genuine market closures, not feed gaps"
        )
    if recommended_tick is None:
        messages.append("could not derive a tick size from the observed prices")
    elif finest_tick is not None and recommended_tick != finest_tick:
        messages.append(
            f"observed prices all divide by {recommended_tick} but the vendor "
            f"quotes to {finest_tick}; this sample may simply not contain every "
            "level, so declare the instrument's real tick rather than the "
            "derived one"
        )
    return messages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inspect-trade-tape",
        description=(
            "Read-only profiling of a candidate spot-ECN trade tape. Never "
            "writes or seals an absorption sidecar."
        ),
    )
    parser.add_argument("--tape", action="append", required=True)
    parser.add_argument("--column-timestamp", default="timestamp")
    parser.add_argument("--column-price", default="price")
    parser.add_argument("--column-size", default="size")
    parser.add_argument("--column-aggressor", default="aggressor")
    parser.add_argument("--column-arrival")
    parser.add_argument(
        "--timestamp-format",
        choices=TIMESTAMP_FORMATS,
        default="iso8601",
    )
    parser.add_argument("--limit-rows", type=int)
    parser.add_argument("--json", help="write the full report to this path")
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    paths = [Path(item).expanduser().resolve() for item in args.tape]
    for path in paths:
        if not path.is_file():
            raise TapeIngestError(f"tape file does not exist: {path}")
    return inspect_csv_tape(
        paths,
        columns=TapeColumns(
            timestamp=args.column_timestamp,
            price=args.column_price,
            size=args.column_size,
            aggressor=args.column_aggressor,
            arrival=args.column_arrival,
        ),
        timestamp_format=args.timestamp_format,
        limit_rows=args.limit_rows,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (TapeIngestError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
        return 2
    if args.json:
        Path(args.json).expanduser().write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    coverage = report["coverage"]
    print(f"rows={report['rows_read']} unparseable={report['unparseable_rows']}")
    print(f"aggressor labels: {report['aggressor_labels']}")
    print(f"recommended --tick-size {report['recommended_tick_size']}")
    print(
        "coverage: {first_bar_open} .. {last_bar_open} "
        "bars={bars_with_trades}/{bars_in_span} missing={bars_missing}".format(
            **coverage
        )
    )
    print(f"trades per bar: {report['trades_per_bar']}")
    print(f"max intrabar gap (s): {report['intrabar_gap_seconds']}")
    for message in report["warnings"]:
        print(f"WARNING: {message}")
    return 0


__all__ = [
    "build_parser",
    "inspect_csv_tape",
    "main",
    "run",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
