#!/usr/bin/env python3
"""CLI entry point for the executed-trade-tape absorption-sidecar converter.

Usage example (spot ECN tape, identity mapping, measured vendor delay)::

    python tools/build_absorption_sidecar.py \
      --tape /data/eurusd-2024.csv \
      --out /srv/orderflow/eurusd-2024-v1 \
      --symbol EURUSD --venue LMAX --source-id lmax-ecn-tape \
      --tick-size 0.00001 --volume-measure executed_size \
      --aggressor-provenance venue_reported_aggressor \
      --buy-label BUY --sell-label SELL \
      --timestamp-format iso8601 \
      --covers-from 2024-01-01T00:00:00Z \
      --covers-through 2025-01-01T00:00:00Z \
      --available-at-rule bar_close_plus_measured_vendor_delay.v1 \
      --vendor-delay-ms 2500 \
      --delay-measurement "p99 arrival-vs-bar_close over 2024-06, n=..."

All validation lives in :mod:`backtest.orderflow_ingest`; this file only wires
the repository root onto ``sys.path`` so the tool runs from a checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.orderflow_ingest import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
