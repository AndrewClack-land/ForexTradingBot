#!/usr/bin/env python3
"""Seal validated Quantower/FxPro diagnostics as a research cluster sidecar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.fxpro_cluster_data import (  # noqa: E402
    FxProClusterEventDataset,
    seal_quantower_diagnostics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export",
        action="append",
        required=True,
        type=Path,
        help="validated diagnostic export; repeat for each UTC day/symbol",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--availability-delay-ms",
        required=True,
        type=int,
        help=(
            "research assumption: delay after M15 close before the cluster "
            "would have been available"
        ),
    )
    parser.add_argument(
        "--availability-evidence",
        required=True,
        help=(
            "audit note describing the assumed/measured delay; this does not "
            "turn reconstructed ticks into exchange executions"
        ),
    )
    parser.add_argument(
        "--exporter-dll",
        type=Path,
        help="installed exporter DLL for SHA-256/MVID verification",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    destination = seal_quantower_diagnostics(
        args.export,
        args.output,
        availability_delay_ms=args.availability_delay_ms,
        availability_evidence=args.availability_evidence,
        exporter_dll=args.exporter_dll,
    )
    dataset = FxProClusterEventDataset.load(destination)
    print(
        json.dumps(
            {
                "sidecar": str(destination.resolve()),
                "manifest_sha256": dataset.manifest_sha256,
                "events": len(dataset.events),
                "symbols": dataset.manifest["symbols"],
                "research_only": dataset.research_only,
                "execution_proof_claimed": False,
                "aggressor_polarity_claimed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
