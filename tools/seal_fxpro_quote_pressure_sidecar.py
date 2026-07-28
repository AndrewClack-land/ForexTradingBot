"""Seal FxPro DOM M15 quote-pressure events into an immutable WFO sidecar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.fxpro_quote_pressure_data import (  # noqa: E402
    FxProQuotePressureEventDataset,
    seal_recorded_events,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Seal FxPro MT5 DOM recorder events for causal Quote Pressure "
            "Rejection walk-forward optimization."
        )
    )
    parser.add_argument(
        "recording",
        type=Path,
        help="ai_data/fxpro_dom directory, or its events directory",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="new sidecar directory; existing paths are never overwritten",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    destination = seal_recorded_events(args.recording, args.output)
    dataset = FxProQuotePressureEventDataset.load(destination)
    print(
        json.dumps(
            {
                "sidecar": str(destination.resolve()),
                "manifest_sha256": dataset.manifest_sha256,
                "events": len(dataset.events),
                "symbols": dataset.manifest["symbols"],
                "data_kind": dataset.manifest["data_kind"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
