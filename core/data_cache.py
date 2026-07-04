from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict, Tuple, Optional

import pandas as pd


class DataCache:
    """Threaded cache that periodically refreshes klines for requested symbols/tfs."""

    def __init__(self, data_feed, *, default_limit: int = 200, refresh_sec: Optional[Dict[str, float]] = None):
        self.feed = data_feed
        self.default_limit = default_limit
        self.refresh_sec = refresh_sec or {
            "4h": 30 * 60,
            "1h": 10 * 60,
            "15m": 120,
            "5m": 45,
            "1m": 15,
        }
        self._cache: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._subscriptions: Dict[Tuple[str, str], int] = defaultdict(int)
        # Largest limit ever requested per key — the background refresher must
        # not overwrite a 300-bar frame with a default_limit-sized one.
        self._limits: Dict[Tuple[str, str], int] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="DataCacheLoop", daemon=True)
        self._thread.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout)

    def request(self, symbol: str, tf: str, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        key = (symbol, tf)
        lim = int(limit or self.default_limit)
        with self._lock:
            self._subscriptions[key] += 1
            self._limits[key] = max(self._limits.get(key, 0), lim)
            cached = self._cache.get(key)
        if cached is None:
            df = self.feed.get_klines(symbol, tf, lim)
            if df is not None and not df.empty:
                with self._lock:
                    self._cache[key] = (time.time(), df)
                return df
            return None
        ts, df = cached
        refresh = float(self.refresh_sec.get(tf, 30))
        if time.time() - ts > refresh:
            self._schedule_refresh(key, lim)
            with self._lock:
                refreshed = self._cache.get(key)
            if refreshed is not None:
                return refreshed[1]
        return df

    def _schedule_refresh(self, key: Tuple[str, str], limit: int) -> None:
        symbol, tf = key
        try:
            df = self.feed.get_klines(symbol, tf, limit, force=True)
            if df is not None and not df.empty:
                with self._lock:
                    self._cache[key] = (time.time(), df)
        except Exception:
            pass

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(0.5)
            now = time.time()
            keys = list(self._subscriptions.keys())
            for key in keys:
                symbol, tf = key
                refresh = float(self.refresh_sec.get(tf, 30))
                with self._lock:
                    cached = self._cache.get(key)
                if cached is None:
                    continue
                ts, _ = cached
                if now - ts < refresh:
                    continue
                with self._lock:
                    limit = self._limits.get(key, self.default_limit)
                self._schedule_refresh(key, limit)
