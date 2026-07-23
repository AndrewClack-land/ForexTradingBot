"""Command-line entry point for offline snapshot tooling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .data import DataValidationError, HistoricalDataset


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            return _audit(args)
    except (DataValidationError, FileNotFoundError, ValueError) as exc:
        print(f"AUDIT ERROR: {exc}", file=sys.stderr)
        return 2
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
