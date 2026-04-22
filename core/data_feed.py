# core/data_feed.py
#
# MT5-only data feed.
#
# Primary path : mt5.copy_rates_range() called in-process — same MT5 connection
#                that the executor already owns, always current to the tick.
# Fallback path: JSON cache files written by MT5NativeBridge thread
#                (ai_data/mt5_cache/<SYMBOL>_<tf>.json).
#
# TradingView has been removed entirely.  Price data now comes from the same
# broker that executes the trades, eliminating the TV↔broker price divergence
# that caused premature TP/SL closures.
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple, Optional

import MetaTrader5 as mt5
import pandas as pd


# MT5 timeframe constants
_TF_MT5: Dict[str, int] = {
    "1d":  mt5.TIMEFRAME_D1,
    "4h":  mt5.TIMEFRAME_H4,
    "1h":  mt5.TIMEFRAME_H1,
    "15m": mt5.TIMEFRAME_M15,
    "5m":  mt5.TIMEFRAME_M5,
    "1m":  mt5.TIMEFRAME_M1,
}

# How many calendar days to request per timeframe so we always get ≥300 bars
_TF_LOOKBACK_DAYS: Dict[str, int] = {
    "1d":  400,
    "4h":  60,
    "1h":  16,
    "15m": 4,
    "5m":  2,
    "1m":  1,
}


class DataFeed:
    """MT5-only market data provider.

    ``universe`` maps bot symbol names to MT5 terminal symbol names, e.g.::

        {"GOLD": "GOLD", "GBPUSD": "GBPUSD", "USDCAD": "USDCAD"}

    The old ``tv`` and ``data_source`` parameters have been removed.
    """

    def __init__(
        self,
        universe: Dict[str, str],
        mt5_cache_dir: Optional[Path] = None,
        # legacy keyword kept so callers that still pass tv= get a clear error
        **_ignored,
    ):
        if _ignored:
            import warnings
            warnings.warn(
                f"DataFeed: unknown keyword arguments ignored: {list(_ignored)}. "
                "The 'tv' and 'data_source' parameters have been removed.",
                stacklevel=2,
            )

        # bot_symbol → mt5_symbol
        self.universe = universe
        self.mt5_cache_dir = Path(mt5_cache_dir) if mt5_cache_dir else None
        if self.mt5_cache_dir:
            self.mt5_cache_dir.mkdir(parents=True, exist_ok=True)

        # in-process candle cache: (symbol, tf) → (timestamp, DataFrame)
        self._cache: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}

        # minimum seconds before re-fetching each timeframe
        self.refresh_sec: Dict[str, float] = {
            "1d":  6 * 3600,
            "4h":  30 * 60,
            "1h":  10 * 60,
            "15m": 2 * 60,
            "5m":  60,
            "1m":  20,
        }

        # per-symbol error tracking
        self._fail_count: Dict[str, int] = {}
        self._cooldown_until: Dict[str, float] = {}
        self.fail_threshold = 5
        self.cooldown_sec = 10 * 60

        # suppress repeated log noise
        self._last_fail_print: Dict[Tuple[str, str], float] = {}
        self.fail_print_cooldown_sec = 120
        self._cache_fallback_warned: set = set()

    # ------------------------------------------------------------------
    # public API (same signature as the old TV-based DataFeed)
    # ------------------------------------------------------------------

    def get_klines(
        self,
        symbol_key: str,
        tf: str,
        limit: int = 300,
        retries: int = 2,
        force: bool = False,
    ) -> pd.DataFrame:
        if symbol_key not in self.universe:
            return pd.DataFrame()

        if self._is_on_cooldown(symbol_key):
            cached = self._cache.get((symbol_key, tf))
            return cached[1] if cached else pd.DataFrame()

        key = (symbol_key, tf)
        now = time.time()

        # return in-process cache if still fresh
        if not force and key in self._cache:
            ts, df = self._cache[key]
            if now - ts < self.refresh_sec.get(tf, 60):
                return df

        mt5_symbol = self.universe[symbol_key]

        # ── Primary: direct MT5 API ──────────────────────────────────
        for attempt in range(retries):
            df = self._fetch_direct(mt5_symbol, tf, limit)
            if df is not None and not df.empty:
                self._cache[key] = (time.time(), df)
                self._note_success(symbol_key)
                return df
            if attempt < retries - 1:
                time.sleep(0.05)

        # ── Fallback: JSON cache written by MT5NativeBridge ──────────
        df = self._fetch_cache(symbol_key, tf, limit)
        if df is not None and not df.empty:
            self._cache[key] = (time.time(), df)
            self._note_success(symbol_key)
            if key not in self._cache_fallback_warned:
                print(f"[DataFeed] {symbol_key} {tf}: MT5 API failed — using bridge cache file")
                self._cache_fallback_warned.add(key)
            return df

        # ── Both failed ──────────────────────────────────────────────
        self._note_fail(symbol_key)
        nowp = time.time()
        lp = self._last_fail_print.get(key, 0.0)
        if nowp - lp > self.fail_print_cooldown_sec:
            try:
                err = mt5.last_error()
            except Exception:
                err = "MT5 not initialized"
            print(
                f"[DataFeed] FAILED {symbol_key} {tf} "
                f"(MT5={err}, fails={self._fail_count.get(symbol_key, 0)})"
            )
            self._last_fail_print[key] = nowp

        # return stale cache rather than empty if available
        if key in self._cache:
            return self._cache[key][1]
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _fetch_direct(self, mt5_symbol: str, tf: str, limit: int) -> Optional[pd.DataFrame]:
        """Call mt5.copy_rates_range() in-process."""
        tf_const = _TF_MT5.get(tf)
        if tf_const is None:
            return None

        lookback = _TF_LOOKBACK_DAYS.get(tf, 7)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback)

        try:
            data = mt5.copy_rates_range(mt5_symbol, tf_const, start, end)
        except Exception:
            return None

        if data is None or len(data) == 0:
            return None

        df = pd.DataFrame(data)
        if df.empty:
            return None

        df = df.sort_values("time")
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("timestamp")

        # prefer real volume over tick volume
        if "real_volume" in df.columns and df["real_volume"].sum() > 0:
            df["volume"] = df["real_volume"]
        elif "tick_volume" in df.columns:
            df["volume"] = df["tick_volume"]
        else:
            df["volume"] = 0

        df = df[["open", "high", "low", "close", "volume"]].copy()
        if len(df) > limit:
            df = df.tail(limit)
        return df

    def _fetch_cache(self, symbol_key: str, tf: str, limit: int) -> Optional[pd.DataFrame]:
        """Read JSON file written by MT5NativeBridge."""
        if not self.mt5_cache_dir:
            return None
        path = self.mt5_cache_dir / f"{symbol_key}_{tf.lower()}.json"
        if not path.exists():
            return None
        try:
            df = pd.read_json(path, orient="records")
            if df.empty:
                return None
            cols = ["open", "high", "low", "close", "volume"]
            if not all(c in df.columns for c in cols):
                return None
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df = df.dropna(subset=["timestamp"]).set_index("timestamp")
            df = df[cols].copy()
            if len(df) > limit:
                df = df.tail(limit)
            return df
        except Exception:
            return None

    def _is_on_cooldown(self, symbol_key: str) -> bool:
        return time.time() < self._cooldown_until.get(symbol_key, 0.0)

    def _note_fail(self, symbol_key: str) -> None:
        c = self._fail_count.get(symbol_key, 0) + 1
        self._fail_count[symbol_key] = c
        if c >= self.fail_threshold:
            self._cooldown_until[symbol_key] = time.time() + self.cooldown_sec

    def _note_success(self, symbol_key: str) -> None:
        self._fail_count[symbol_key] = 0
        self._cooldown_until.pop(symbol_key, None)
