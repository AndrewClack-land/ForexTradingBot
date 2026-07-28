#!/usr/bin/env python3
"""CLI entry point for read-only trade-tape profiling.

Run this before :mod:`tools.build_absorption_sidecar` to learn the tick grid,
the exact set of aggressor labels, the timestamp format and the bar coverage of
a candidate spot-ECN tape::

    python tools/inspect_trade_tape.py \
      --tape /data/eurusd-2024.csv \
      --column-timestamp exec_time --column-price px \
      --column-size qty --column-aggressor taker_side \
      --json /tmp/eurusd-2024.tape-profile.json

It never writes or seals an absorption sidecar.
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.orderflow_inspect import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
