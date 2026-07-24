"""Command-line entry point for offline snapshot tooling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .data import DataValidationError, HistoricalDataset
from .lse_ingest import (
    DEFAULT_BAR_TIMEZONE,
    DEFAULT_TARGET_TIMEFRAMES,
    LSEIngestError,
    import_lse_snapshot,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backtest",
        description="Pure offline backtest snapshot utilities",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser(
        "audit",
        help="validate candle files and report walk-forward readiness",
    )
    audit.add_argument("--data", required=True, help="JSON/Parquet snapshot directory")
    audit.add_argument("--symbols", nargs="*", help="optional symbol subset")
    audit.add_argument("--min-d1-bars", type=int, default=365)
    audit.add_argument("--min-months", type=int, default=12)
    audit.add_argument("--json", action="store_true", help="machine-readable output")
    audit.add_argument(
        "--manifest-out",
        help="optionally write the deterministic content manifest to this path",
    )
    verify = subparsers.add_parser(
        "verify",
        help="verify a stored snapshot manifest without WFO readiness semantics",
    )
    verify.add_argument("--data", required=True, help="sealed snapshot directory")
    verify.add_argument(
        "--json",
        action="store_true",
        help="machine-readable verification result",
    )

    lse_import = subparsers.add_parser(
        "lse-import",
        help="download LSE M1 candles into a new immutable Parquet snapshot",
    )
    lse_import.add_argument(
        "--output",
        required=True,
        help="new snapshot directory; must not already exist",
    )
    lse_import.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="bot symbols or BOT=LSE mappings, e.g. EURUSD GOLD=XAU/USD",
    )
    lse_import.add_argument(
        "--start",
        required=True,
        help="inclusive UTC start (export requires YYYY-MM-DD)",
    )
    lse_import.add_argument(
        "--end",
        required=True,
        help="exclusive UTC end (export requires YYYY-MM-DD)",
    )
    lse_import.add_argument(
        "--transport",
        choices=("export", "rest"),
        default="rest",
        help=(
            "paced UTC date-window REST with tail truncation probes "
            "(default), or chunked bulk Parquet export"
        ),
    )
    lse_import.add_argument(
        "--dataset",
        default=None,
        help="optional LSE dataset override; default resolves each symbol from catalog",
    )
    lse_import.add_argument(
        "--timeframes",
        nargs="+",
        default=list(DEFAULT_TARGET_TIMEFRAMES),
        help="canonical outputs built from M1 (default: 1m 5m 15m 1h 4h 1d)",
    )
    lse_import.add_argument(
        "--bar-timezone",
        default=DEFAULT_BAR_TIMEZONE,
        help="broker wall-clock timezone used for H4/D1 boundaries",
    )
    lse_import.add_argument(
        "--max-duplicate-ratio",
        type=float,
        default=0.001,
        help="fail when duplicate M1 timestamps exceed this fraction",
    )
    lse_import.add_argument(
        "--page-limit",
        type=int,
        default=5000,
        help=(
            "REST row limit; must exceed 1440 and reaching it fails closed "
            "(default: 5000)"
        ),
    )
    lse_import.add_argument(
        "--rest-min-interval",
        type=float,
        default=0.35,
        help=(
            "minimum seconds between every REST request, including tail "
            "probes and retries (default stays below 200/min)"
        ),
    )
    lse_import.add_argument(
        "--retry-after-seconds",
        type=float,
        default=60.0,
        help="minimum delay after HTTP 429",
    )
    lse_import.add_argument(
        "--min-bucket-coverage",
        type=float,
        default=0.95,
        help="drop derived candles below this fraction of expected M1 rows",
    )
    lse_import.add_argument(
        "--max-edge-gap-days",
        type=float,
        default=4.0,
        help="maximum allowed missing coverage at either requested range edge",
    )
    lse_import.add_argument(
        "--max-internal-gap-days",
        type=float,
        default=4.0,
        help="fail when an internal M1 history gap exceeds this many days",
    )
    lse_import.add_argument(
        "--export-chunk-days",
        type=int,
        default=600,
        help="date-only export chunk size; start/end must be YYYY-MM-DD",
    )
    lse_import.add_argument(
        "--max-export-jobs",
        type=int,
        default=5,
        help="fail before export when all symbol/chunk jobs exceed this budget",
    )
    lse_import.add_argument("--timeout", type=float, default=60.0)
    lse_import.add_argument(
        "--require-wfo-ready",
        action="store_true",
        help="return exit code 3 when the completed snapshot lacks WFO history",
    )
    lse_import.add_argument(
        "--release-commit-file",
        help="optional file containing the exact deployed 40/64-hex Git commit",
    )
    lse_import.add_argument(
        "--environment-lock-file",
        help="optional full pip freeze file; must accompany --release-commit-file",
    )
    lse_import.add_argument(
        "--json",
        action="store_true",
        help="machine-readable result without secrets",
    )
    return parser


def _audit(args: argparse.Namespace) -> int:
    dataset = HistoricalDataset.load(Path(args.data))
    if args.manifest_out:
        dataset.write_manifest(args.manifest_out)
    readiness = dataset.audit_readiness(
        min_d1_bars=args.min_d1_bars,
        min_months=args.min_months,
        symbols=args.symbols,
    )
    payload = {
        "data": str(dataset.root),
        "manifest_sha256": dataset.manifest_sha256,
        "coverage": [item.to_dict() for item in dataset.coverage()],
        **readiness,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(f"Snapshot: {dataset.root}")
        print(f"Manifest SHA-256: {dataset.manifest_sha256}")
        print("Coverage:")
        for item in dataset.coverage():
            print(
                f"  {item.symbol:10s} {item.timeframe:3s} "
                f"rows={item.rows:7d} duplicates={item.duplicate_rows:5d} "
                f"{item.start.isoformat()} -> {item.end_close.isoformat()}"
            )
        if readiness["wfo_ready"]:
            print("WFO READY")
        else:
            print("WFO BLOCKED")
            for reason in readiness["reasons"]:
                print(f"  - {reason}")

    # A blocked audit must fail closed so automation cannot accidentally start
    # walk-forward analysis on an inadequate snapshot.
    return 0 if readiness["wfo_ready"] else 3


def _verify(args: argparse.Namespace) -> int:
    root = Path(args.data).expanduser().resolve()
    if not (root / "manifest.json").is_file():
        raise DataValidationError(f"{root}: stored manifest.json is missing")
    dataset = HistoricalDataset.load(root)
    payload = {
        "verified": True,
        "data": str(dataset.root),
        "manifest_sha256": dataset.manifest_sha256,
        "files": len(dataset.manifest["files"]),
        "series": len(dataset.coverage()),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(f"SNAPSHOT VERIFIED: {dataset.root}")
        print(f"Manifest SHA-256: {dataset.manifest_sha256}")
    return 0


def _lse_import(args: argparse.Namespace) -> int:
    result = import_lse_snapshot(
        output=args.output,
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        timeframes=args.timeframes,
        transport=args.transport,
        dataset=args.dataset or None,
        bar_timezone=args.bar_timezone,
        max_duplicate_ratio=args.max_duplicate_ratio,
        page_limit=args.page_limit,
        rest_min_interval=args.rest_min_interval,
        retry_after_seconds=args.retry_after_seconds,
        min_bucket_coverage=args.min_bucket_coverage,
        max_edge_gap_days=args.max_edge_gap_days,
        max_internal_gap_days=args.max_internal_gap_days,
        export_chunk_days=args.export_chunk_days,
        max_export_jobs=args.max_export_jobs,
        release_commit_file=args.release_commit_file,
        environment_lock_file=args.environment_lock_file,
        timeout=args.timeout,
    )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(f"LSE snapshot: {result.output}")
        print(f"Manifest SHA-256: {result.manifest_sha256}")
        for item in result.symbols:
            rows = ", ".join(
                f"{timeframe}={count}"
                for timeframe, count in item["derived_rows"].items()
            )
            print(
                f"  {item['provider_symbol']} -> {item['bot_symbol']}: "
                f"{rows}"
            )
        print("WFO READY" if result.readiness["wfo_ready"] else "WFO BLOCKED")
    if args.require_wfo_ready and not result.readiness["wfo_ready"]:
        return 3
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            return _audit(args)
        if args.command == "verify":
            return _verify(args)
        if args.command == "lse-import":
            return _lse_import(args)
    except (DataValidationError, LSEIngestError, FileNotFoundError, ValueError) as exc:
        print(f"BACKTEST ERROR [{args.command}]: {exc}", file=sys.stderr)
        return 2
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
