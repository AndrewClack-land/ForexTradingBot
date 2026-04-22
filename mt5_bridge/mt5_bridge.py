"""MT5 → bot bridge.

Listens for ZeroMQ payloads produced by the FXProBridge.mq5 EA and materializes
shared JSON files so the strategy can consume them as it previously did with
TradingView (via `core.data_feed`).

Usage:
    python mt5_bridge.py --host 127.0.0.1 --port 7777

Outputs per (symbol, timeframe) a JSON file in `ai_data/mt5_cache/` with the
latest `limit` bars, normalized to the format expected by `core/data_feed.py`.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import zmq

LOG = logging.getLogger("mt5_bridge")

# directory that the main bot already uses for data/logs
PROJECT_ROOT = Path(__file__).resolve().parents[1]
AI_DATA_DIR = PROJECT_ROOT / "ai_data"
CACHE_DIR = AI_DATA_DIR / "mt5_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Candle:
    symbol: str
    tf: str
    timestamp: str  # ISO8601 (mt5 EA should send in UTC)
    open: float
    high: float
    low: float
    close: float
    volume: float

    @staticmethod
    def from_dict(d: Dict) -> "Candle":
        required = ["symbol", "tf", "timestamp", "open", "high", "low", "close", "volume"]
        for key in required:
            if key not in d:
                raise ValueError(f"Missing key '{key}' in candle payload: {d}")
        return Candle(
            symbol=str(d["symbol"]).upper(),
            tf=str(d["tf"]).lower(),
            timestamp=str(d["timestamp"]),
            open=float(d["open"]),
            high=float(d["high"]),
            low=float(d["low"]),
            close=float(d["close"]),
            volume=float(d.get("volume", 0.0)),
        )


class MT5Bridge:
    def __init__(self, host: str = "127.0.0.1", port: int = 7777, limit: int = 500):
        self.host = host
        self.port = port
        self.limit = limit

        self.ctx = zmq.Context.instance()
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")  # subscribe to all topics
        self.socket.connect(f"tcp://{host}:{port}")

        LOG.info("Connected to MT5 bridge publisher tcp://%s:%s", host, port)

        self.buffers: Dict[tuple[str, str], List[Candle]] = {}

    def _write_cache(self, key: tuple[str, str]) -> None:
        candles = self.buffers.get(key, [])
        if not candles:
            return

        df = pd.DataFrame([c.__dict__ for c in candles])
        df = df.sort_values("timestamp")
        path = CACHE_DIR / f"{key[0]}_{key[1]}.json"
        path.write_text(df.to_json(orient="records"))
        LOG.debug("Wrote %s (%d candles)", path.name, len(df))

    def run(self) -> None:
        LOG.info("Listening...")
        try:
            while True:
                raw = self.socket.recv()
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    LOG.warning("Bad JSON: %s", exc)
                    continue

                try:
                    candle = Candle.from_dict(payload)
                except Exception as exc:
                    LOG.warning("Bad payload %s: %s", payload, exc)
                    continue

                key = (candle.symbol, candle.tf)
                bucket = self.buffers.setdefault(key, [])
                bucket.append(candle)
                if len(bucket) > self.limit:
                    del bucket[:-self.limit]

                self._write_cache(key)
        except KeyboardInterrupt:
            LOG.info("Interrupted, shutting down...")
        finally:
            self.socket.close(0)
            self.ctx.term()


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(AI_DATA_DIR / "mt5_bridge.log"),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MT5 bridge listener")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)
    bridge = MT5Bridge(host=args.host, port=args.port, limit=args.limit)

    def handle_sig(signum, frame):
        LOG.info("Signal %s received, exiting", signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_sig)
    signal.signal(signal.SIGINT, handle_sig)

    bridge.run()


if __name__ == "__main__":
    main()
