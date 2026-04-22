"""Native MT5 → bot bridge using MetaTrader5 Python API.

Pulls candles directly from a locally running MT5 terminal (same machine) and
materializes JSON caches under ``ai_data/mt5_cache`` so the strategy can read
high‑TF data without relying on the ZeroMQ EA bridge.

Usage (example):
    python mt5_native_bridge.py --server "FxPro-MT5 Demo" \
        --login 591216595 --password "***" \
        --symbols "EURUSD,GBPUSD,USDCAD,GOLD:XAUUSD" \
        --timeframes "1m,5m,15m,1h,4h,1d" --lookback-days 15 --interval 60

Credentials can also be provided via env vars MT5_LOGIN / MT5_PASSWORD / MT5_SERVER.
The MT5 terminal must be running on the same host and logged into the specified
account (FxPro demo/live, etc.).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

import MetaTrader5 as mt5
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AI_DATA_DIR = PROJECT_ROOT / "ai_data"
CACHE_DIR = AI_DATA_DIR / "mt5_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SYMBOL_SPEC = "EURUSD,GBPUSD,USDCAD,GOLD:XAUUSD"
DEFAULT_TF_SPEC = "1m,5m,15m,1h,4h,1d"

TF_MAP = {
    "1m": mt5.TIMEFRAME_M1,
    "m1": mt5.TIMEFRAME_M1,
    "5m": mt5.TIMEFRAME_M5,
    "m5": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15,
    "m15": mt5.TIMEFRAME_M15,
    "1h": mt5.TIMEFRAME_H1,
    "h1": mt5.TIMEFRAME_H1,
    "4h": mt5.TIMEFRAME_H4,
    "h4": mt5.TIMEFRAME_H4,
    "1d": mt5.TIMEFRAME_D1,
    "d1": mt5.TIMEFRAME_D1,
}

TF_NORMALIZE = {
    "1m": "1m",
    "m1": "1m",
    "5m": "5m",
    "m5": "5m",
    "15m": "15m",
    "m15": "15m",
    "1h": "1h",
    "h1": "1h",
    "4h": "4h",
    "h4": "4h",
    "1d": "1d",
    "d1": "1d",
}


@dataclass(frozen=True)
class SymbolMapping:
    mt5_symbol: str
    bot_symbol: str


def parse_symbol_spec(raw: str) -> Tuple[SymbolMapping, ...]:
    mappings = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            mt5_symbol, bot_symbol = part.split(":", 1)
        else:
            mt5_symbol = bot_symbol = part
        mappings.append(SymbolMapping(mt5_symbol.strip().upper(), bot_symbol.strip().upper()))
    if not mappings:
        raise ValueError("No symbols parsed from specification")
    return tuple(mappings)


def parse_timeframes(raw: str) -> Tuple[str, ...]:
    out = []
    for part in (raw or "").split(","):
        part = part.strip().lower()
        if not part:
            continue
        norm = TF_NORMALIZE.get(part)
        if not norm:
            raise ValueError(f"Unsupported timeframe '{part}'. Use 1m,5m,15m,1h,4h,1d")
        if norm not in out:
            out.append(norm)
    if not out:
        raise ValueError("No timeframes parsed from specification")
    return tuple(out)


def isoformat(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class MT5NativeBridge:
    def __init__(
        self,
        mappings: Iterable[SymbolMapping],
        timeframes: Iterable[str],
        lookback_days: int = 15,
        poll_interval: int = 60,
        cache_dir: Path | None = None,
    ) -> None:
        self.mappings = tuple(mappings)
        self.timeframes = tuple(timeframes)
        self.lookback_days = int(max(1, lookback_days))
        self.poll_interval = max(15, int(poll_interval))
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("mt5_native_bridge")

    def _fetch_rates(self, mt5_symbol: str, tf_name: str) -> pd.DataFrame | None:
        tf_const = TF_MAP[tf_name]
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self.lookback_days)
        data = mt5.copy_rates_range(mt5_symbol, tf_const, start, end)
        if data is None:
            err = mt5.last_error()
            self.logger.warning("copy_rates_range failed for %s %s: %s", mt5_symbol, tf_name, err)
            return None
        if len(data) == 0:
            self.logger.warning("No data returned for %s %s", mt5_symbol, tf_name)
            return None
        df = pd.DataFrame(data)
        if df.empty:
            return None
        df = df.sort_values("time")
        df["timestamp"] = df["time"].apply(isoformat)
        if "real_volume" in df.columns and df["real_volume"].sum() > 0:
            df["volume"] = df["real_volume"]
        else:
            df["volume"] = df.get("tick_volume", 0)
        return df

    def _write_cache(self, bot_symbol: str, tf_name: str, df: pd.DataFrame) -> None:
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            self.logger.warning("Missing columns %s for %s %s", missing, bot_symbol, tf_name)
            return
        payload = df[cols].copy()
        payload.insert(0, "tf", tf_name)
        payload.insert(0, "symbol", bot_symbol)
        path = self.cache_dir / f"{bot_symbol}_{tf_name}.json"
        path.write_text(payload.to_json(orient="records"))
        self.logger.info("Wrote %s (%d rows)", path.name, len(payload))

    def sync_once(self) -> None:
        for mapping in self.mappings:
            for tf in self.timeframes:
                df = self._fetch_rates(mapping.mt5_symbol, tf)
                if df is None:
                    continue
                self._write_cache(mapping.bot_symbol, tf, df)

    def run(
        self,
        once: bool = False,
        stop_event: "threading.Event | None" = None,
        manage_connection: bool = True,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
    ) -> None:
        if manage_connection:
            init_mt5(login, password, server)
        self.logger.info(
            "Starting MT5 native bridge for symbols=%s tfs=%s (lookback=%d days, interval=%ds)",
            [f"{m.mt5_symbol}->{m.bot_symbol}" for m in self.mappings],
            list(self.timeframes),
            self.lookback_days,
            self.poll_interval,
        )
        try:
            while True:
                self.sync_once()
                if once:
                    break
                if stop_event is not None and stop_event.wait(self.poll_interval):
                    break
                if stop_event is None:
                    time.sleep(self.poll_interval)
        finally:
            if manage_connection:
                mt5.shutdown()
            self.logger.info("Bridge loop finished")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(AI_DATA_DIR / "mt5_native_bridge.log"),
        ],
    )


def init_mt5(login: int | None, password: str | None, server: str | None) -> None:
    kwargs = {}
    if login:
        kwargs["login"] = int(login)
    if password:
        kwargs["password"] = password
    if server:
        kwargs["server"] = server
    if not mt5.initialize(**kwargs):
        raise RuntimeError(f"mt5.initialize failed: {mt5.last_error()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native MT5 bridge (MetaTrader5 API)")
    parser.add_argument("--login", type=int, default=os.getenv("MT5_LOGIN"))
    parser.add_argument("--password", default=os.getenv("MT5_PASSWORD"))
    parser.add_argument("--server", default=os.getenv("MT5_SERVER"))
    parser.add_argument("--symbols", default=DEFAULT_SYMBOL_SPEC, help="Comma list, MT5_SYMBOL or MT5_SYMBOL:BOT_SYMBOL")
    parser.add_argument("--timeframes", default=DEFAULT_TF_SPEC, help="Comma list e.g. 15m,1h,4h,1d")
    parser.add_argument("--lookback-days", type=int, default=15)
    parser.add_argument("--interval", type=int, default=60, help="Seconds between refresh runs")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--once", action="store_true", help="Fetch once and exit")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    mappings = parse_symbol_spec(args.symbols)
    timeframes = parse_timeframes(args.timeframes)

    init_mt5(args.login, args.password, args.server)

    bridge = MT5NativeBridge(
        mappings=mappings,
        timeframes=timeframes,
        lookback_days=args.lookback_days,
        poll_interval=args.interval,
        cache_dir=Path(args.cache_dir),
    )
    bridge.run(
        once=args.once,
        stop_event=None,
        manage_connection=False,
    )
    mt5.shutdown()


if __name__ == "__main__":
    main()
