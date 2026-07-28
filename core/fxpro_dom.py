"""FxPro MT5 OTC depth recorder and causal M15 feature aggregation.

The recorder is read-only.  It subscribes to ``market_book_get`` for explicitly
configured FxPro symbols, writes append-only raw snapshots, and exposes only
fully closed M15 summaries to the strategy.

Quote removal is never labelled as an execution.  Replenishment is counted
conservatively only when volume previously removed from an already-observed
price level later returns at that same price.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd

from core.liquidity_rejection import (
    LIQUIDITY_DATA_KIND,
    LIQUIDITY_EVENT_SCHEMA_VERSION,
    LIQUIDITY_MARKET_TYPE,
    LIQUIDITY_SOURCE,
    LIQUIDITY_TIMEFRAME,
    LIQUIDITY_VENUE,
    seal_liquidity_event,
    validate_liquidity_event,
)


RAW_SNAPSHOT_SCHEMA = "forexbot.fxpro-dom-snapshot"
RAW_SNAPSHOT_SCHEMA_VERSION = 1
AGGREGATOR_VERSION = "fxpro-dom-m15-aggregator/1"
SOURCE_NAME = LIQUIDITY_SOURCE
VENUE_NAME = LIQUIDITY_VENUE


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def _finite_nonnegative(value: Any) -> Optional[float]:
    if isinstance(value, (bool, str, bytes)) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


@dataclass(frozen=True)
class DomLevel:
    price: float
    volume: float

    def as_pair(self) -> list[float]:
        return [float(self.price), float(self.volume)]


@dataclass(frozen=True)
class DomSnapshot:
    captured_at: pd.Timestamp
    tick_time: pd.Timestamp
    source_symbol: str
    symbol: str
    bid: float
    ask: float
    bids: tuple[DomLevel, ...]
    asks: tuple[DomLevel, ...]

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "bid": float(self.bid),
            "ask": float(self.ask),
            "bids": [level.as_pair() for level in self.bids],
            "asks": [level.as_pair() for level in self.asks],
        }

    def fingerprint(self) -> str:
        return _canonical_hash(self.fingerprint_payload())

    def raw_record(self) -> dict[str, Any]:
        payload = {
            "schema": RAW_SNAPSHOT_SCHEMA,
            "schema_version": RAW_SNAPSHOT_SCHEMA_VERSION,
            "aggregator_version": AGGREGATOR_VERSION,
            "source": SOURCE_NAME,
            "venue": VENUE_NAME,
            "market_type": LIQUIDITY_MARKET_TYPE,
            "data_kind": LIQUIDITY_DATA_KIND,
            "source_symbol": self.source_symbol,
            "symbol": self.symbol,
            "captured_at": self.captured_at.isoformat(),
            "tick_time": self.tick_time.isoformat(),
            **self.fingerprint_payload(),
        }
        return {**payload, "checksum": _canonical_hash(payload)}


class M15LiquidityAccumulator:
    """State for one symbol and one UTC M15 interval."""

    def __init__(self, snapshot: DomSnapshot) -> None:
        self.bar_open = snapshot.captured_at.floor("15min")
        self.bar_close = self.bar_open + pd.Timedelta(minutes=15)
        self.first_observed_at = snapshot.captured_at
        self.last_observed_at = snapshot.captured_at
        self.last_snapshot = snapshot
        self.last_fingerprint = snapshot.fingerprint()
        self.sample_count = 1
        self.changed_snapshots = 1
        self.max_gap_ms = 0.0
        self.up_ticks = 0
        self.down_ticks = 0
        self.up_distance = 0.0
        self.down_distance = 0.0
        self.start_mid = snapshot.mid
        self.high_mid = snapshot.mid
        self.low_mid = snapshot.mid
        self.end_mid = snapshot.mid
        self.spread_sum = snapshot.spread
        self.max_spread = snapshot.spread
        self.min_bid_levels = len(snapshot.bids)
        self.min_ask_levels = len(snapshot.asks)
        self.bid_depleted_volume = 0.0
        self.bid_replenished_volume = 0.0
        self.ask_depleted_volume = 0.0
        self.ask_replenished_volume = 0.0
        self._bid_deficits: dict[float, float] = {}
        self._ask_deficits: dict[float, float] = {}

    @staticmethod
    def _volume_map(levels: Sequence[DomLevel]) -> dict[float, float]:
        return {float(level.price): float(level.volume) for level in levels}

    @staticmethod
    def _update_side(
        previous: Mapping[float, float],
        current: Mapping[float, float],
        deficits: dict[float, float],
    ) -> tuple[float, float]:
        """Return conservative (depleted, replenished) common-level volume."""

        depleted = 0.0
        replenished = 0.0
        for price in previous.keys() & current.keys():
            before = float(previous[price])
            after = float(current[price])
            delta = after - before
            if delta < 0.0:
                removed = -delta
                depleted += removed
                deficits[price] = deficits.get(price, 0.0) + removed
            elif delta > 0.0:
                outstanding = deficits.get(price, 0.0)
                restored = min(delta, outstanding)
                if restored > 0.0:
                    replenished += restored
                    outstanding -= restored
                    if outstanding <= 1e-12:
                        deficits.pop(price, None)
                    else:
                        deficits[price] = outstanding
        return depleted, replenished

    def observe(self, snapshot: DomSnapshot) -> bool:
        if snapshot.captured_at < self.bar_open:
            return False
        self.sample_count += 1
        gap_ms = max(
            0.0,
            (snapshot.captured_at - self.last_observed_at).total_seconds()
            * 1000.0,
        )
        self.max_gap_ms = max(self.max_gap_ms, gap_ms)
        self.last_observed_at = snapshot.captured_at

        fingerprint = snapshot.fingerprint()
        changed = fingerprint != self.last_fingerprint
        if changed:
            self.changed_snapshots += 1
            previous_mid = self.last_snapshot.mid
            delta_mid = snapshot.mid - previous_mid
            if delta_mid > 0.0:
                self.up_ticks += 1
                self.up_distance += delta_mid
            elif delta_mid < 0.0:
                self.down_ticks += 1
                self.down_distance += -delta_mid

            bid_depleted, bid_replenished = self._update_side(
                self._volume_map(self.last_snapshot.bids),
                self._volume_map(snapshot.bids),
                self._bid_deficits,
            )
            ask_depleted, ask_replenished = self._update_side(
                self._volume_map(self.last_snapshot.asks),
                self._volume_map(snapshot.asks),
                self._ask_deficits,
            )
            self.bid_depleted_volume += bid_depleted
            self.bid_replenished_volume += bid_replenished
            self.ask_depleted_volume += ask_depleted
            self.ask_replenished_volume += ask_replenished
            self.last_fingerprint = fingerprint
            self.last_snapshot = snapshot

        self.high_mid = max(self.high_mid, snapshot.mid)
        self.low_mid = min(self.low_mid, snapshot.mid)
        self.end_mid = snapshot.mid
        self.spread_sum += snapshot.spread
        self.max_spread = max(self.max_spread, snapshot.spread)
        self.min_bid_levels = min(self.min_bid_levels, len(snapshot.bids))
        self.min_ask_levels = min(self.min_ask_levels, len(snapshot.asks))
        return changed

    def finalize(self, *, available_at: pd.Timestamp) -> dict[str, Any]:
        observed_seconds = max(
            0.0,
            (self.last_observed_at - self.first_observed_at).total_seconds(),
        )
        coverage_ratio = min(1.0, observed_seconds / (15.0 * 60.0))
        payload = {
            "schema_version": LIQUIDITY_EVENT_SCHEMA_VERSION,
            "revision": 0,
            "timeframe": LIQUIDITY_TIMEFRAME,
            "source": SOURCE_NAME,
            "venue": VENUE_NAME,
            "market_type": LIQUIDITY_MARKET_TYPE,
            "data_kind": LIQUIDITY_DATA_KIND,
            "source_symbol": self.last_snapshot.source_symbol,
            "symbol": self.last_snapshot.symbol,
            "bar_open": self.bar_open.isoformat(),
            "bar_close": self.bar_close.isoformat(),
            "available_at": max(available_at, self.bar_close).isoformat(),
            "finalized": True,
            "coverage_ratio": float(coverage_ratio),
            "sample_count": int(self.sample_count),
            "changed_snapshots": int(self.changed_snapshots),
            "max_gap_ms": float(self.max_gap_ms),
            "up_ticks": int(self.up_ticks),
            "down_ticks": int(self.down_ticks),
            "up_distance": float(self.up_distance),
            "down_distance": float(self.down_distance),
            "bid_depleted_volume": float(self.bid_depleted_volume),
            "bid_replenished_volume": float(self.bid_replenished_volume),
            "ask_depleted_volume": float(self.ask_depleted_volume),
            "ask_replenished_volume": float(self.ask_replenished_volume),
            "start_mid": float(self.start_mid),
            "high_mid": float(self.high_mid),
            "low_mid": float(self.low_mid),
            "end_mid": float(self.end_mid),
            "mean_spread": float(self.spread_sum / self.sample_count),
            "max_spread": float(self.max_spread),
            "min_bid_levels": int(self.min_bid_levels),
            "min_ask_levels": int(self.min_ask_levels),
        }
        return seal_liquidity_event(payload)


class FxProDomRecorder:
    """Threaded MT5 DOM recorder with an in-memory closed-event cache."""

    def __init__(
        self,
        *,
        mt5_module: Any,
        symbols: Iterable[str],
        output_dir: Path,
        poll_interval_ms: int = 500,
        max_levels: int = 20,
        heartbeat_seconds: float = 1.0,
    ) -> None:
        normalized = tuple(
            dict.fromkeys(
                str(symbol).strip().upper()
                for symbol in symbols
                if str(symbol).strip()
            )
        )
        if not normalized:
            raise ValueError("FxPro DOM recorder requires at least one symbol")
        self.mt5 = mt5_module
        self.symbols = normalized
        self.output_dir = Path(output_dir)
        self.poll_interval = max(0.1, int(poll_interval_ms) / 1000.0)
        self.max_levels = max(1, int(max_levels))
        self.heartbeat_seconds = max(
            self.poll_interval,
            float(heartbeat_seconds),
        )
        self.logger = logging.getLogger("fxpro_dom")
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._subscribed: set[str] = set()
        self._unsupported_logged: set[str] = set()
        self._accumulators: dict[str, M15LiquidityAccumulator] = {}
        self._latest_events: dict[str, dict[str, Any]] = {}
        self._last_raw_write: dict[str, pd.Timestamp] = {}
        self._last_raw_fingerprint: dict[str, str] = {}
        self._raw_handles: dict[tuple[str, str], Any] = {}
        (self.output_dir / "snapshots").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "events").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "latest").mkdir(parents=True, exist_ok=True)
        self._load_latest_events()

    def _load_latest_events(self) -> None:
        for symbol in self.symbols:
            path = self.output_dir / "latest" / f"{symbol}.json"
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            validated = validate_liquidity_event(payload)
            if validated is not None:
                self._latest_events[symbol] = validated

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="FxProDomRecorder",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, float(timeout)))
        with self._lock:
            for handle in self._raw_handles.values():
                try:
                    handle.flush()
                    handle.close()
                except OSError:
                    pass
            self._raw_handles.clear()
            for symbol in tuple(self._subscribed):
                try:
                    self.mt5.market_book_release(symbol)
                except Exception:
                    pass
            self._subscribed.clear()

    def _run(self) -> None:
        self.logger.info(
            "Starting FxPro DOM recorder symbols=%s interval=%.3fs",
            list(self.symbols),
            self.poll_interval,
        )
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.poll_once()
            except Exception:
                self.logger.exception("FxPro DOM poll failed")
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, self.poll_interval - elapsed))
        self.logger.info("FxPro DOM recorder stopped")

    def _ensure_subscription(self, symbol: str) -> bool:
        if symbol in self._subscribed:
            return True
        try:
            ok = bool(self.mt5.market_book_add(symbol))
        except Exception as exc:
            if symbol not in self._unsupported_logged:
                self.logger.warning(
                    "market_book_add failed for %s: %s",
                    symbol,
                    exc,
                )
                self._unsupported_logged.add(symbol)
            return False
        if not ok:
            if symbol not in self._unsupported_logged:
                last_error = getattr(self.mt5, "last_error", lambda: None)()
                self.logger.warning(
                    "FxPro DOM unavailable for %s: %s",
                    symbol,
                    last_error,
                )
                self._unsupported_logged.add(symbol)
            return False
        self._subscribed.add(symbol)
        self._unsupported_logged.discard(symbol)
        self.logger.info("Subscribed to FxPro DOM for %s", symbol)
        return True

    def _book_type_sets(self) -> tuple[set[int], set[int]]:
        buy = {
            value
            for value in (
                getattr(self.mt5, "BOOK_TYPE_BUY", None),
                getattr(self.mt5, "BOOK_TYPE_BUY_MARKET", None),
            )
            if isinstance(value, int)
        }
        sell = {
            value
            for value in (
                getattr(self.mt5, "BOOK_TYPE_SELL", None),
                getattr(self.mt5, "BOOK_TYPE_SELL_MARKET", None),
            )
            if isinstance(value, int)
        }
        return buy, sell

    def _snapshot(self, symbol: str) -> Optional[DomSnapshot]:
        if not self._ensure_subscription(symbol):
            return None
        book = self.mt5.market_book_get(symbol)
        tick = self.mt5.symbol_info_tick(symbol)
        if not book or tick is None:
            return None
        bid = _finite_nonnegative(getattr(tick, "bid", None))
        ask = _finite_nonnegative(getattr(tick, "ask", None))
        if bid is None or ask is None or bid <= 0.0 or ask <= bid:
            return None

        buy_types, sell_types = self._book_type_sets()
        bids: dict[float, float] = {}
        asks: dict[float, float] = {}
        for item in book:
            price = _finite_nonnegative(getattr(item, "price", None))
            volume = _finite_nonnegative(
                getattr(item, "volume_dbl", None)
            )
            if volume is None or volume <= 0.0:
                volume = _finite_nonnegative(getattr(item, "volume", None))
            item_type = getattr(item, "type", None)
            if (
                price is None
                or price <= 0.0
                or volume is None
                or volume <= 0.0
            ):
                continue
            target = bids if item_type in buy_types else asks if item_type in sell_types else None
            if target is not None:
                target[price] = target.get(price, 0.0) + volume
        if not bids or not asks:
            return None

        captured_at = _utc_now()
        tick_msc = getattr(tick, "time_msc", None)
        try:
            tick_time = pd.Timestamp(int(tick_msc), unit="ms", tz="UTC")
        except (TypeError, ValueError, OverflowError):
            tick_time = captured_at
        bid_levels = tuple(
            DomLevel(price=price, volume=bids[price])
            for price in sorted(bids, reverse=True)[: self.max_levels]
        )
        ask_levels = tuple(
            DomLevel(price=price, volume=asks[price])
            for price in sorted(asks)[: self.max_levels]
        )
        return DomSnapshot(
            captured_at=captured_at,
            tick_time=tick_time,
            source_symbol=symbol,
            symbol=symbol,
            bid=bid,
            ask=ask,
            bids=bid_levels,
            asks=ask_levels,
        )

    def poll_once(self) -> None:
        for symbol in self.symbols:
            try:
                snapshot = self._snapshot(symbol)
            except Exception as exc:
                self.logger.warning("DOM snapshot failed for %s: %s", symbol, exc)
                continue
            if snapshot is None:
                continue
            self._observe(snapshot)

    def _observe(self, snapshot: DomSnapshot) -> None:
        symbol = snapshot.symbol
        with self._lock:
            accumulator = self._accumulators.get(symbol)
            snapshot_bar = snapshot.captured_at.floor("15min")
            if accumulator is None:
                accumulator = M15LiquidityAccumulator(snapshot)
                self._accumulators[symbol] = accumulator
                changed = True
            elif snapshot_bar != accumulator.bar_open:
                if snapshot_bar > accumulator.bar_open:
                    event = accumulator.finalize(
                        available_at=snapshot.captured_at,
                    )
                    self._latest_events[symbol] = event
                    self._write_event(event)
                accumulator = M15LiquidityAccumulator(snapshot)
                self._accumulators[symbol] = accumulator
                changed = True
            else:
                changed = accumulator.observe(snapshot)
            self._write_raw_if_due(snapshot, changed=changed)

    def _raw_handle(self, symbol: str, date_key: str) -> Any:
        key = (symbol, date_key)
        handle = self._raw_handles.get(key)
        if handle is not None:
            return handle
        for old_key in [
            item for item in self._raw_handles if item[0] == symbol
        ]:
            old = self._raw_handles.pop(old_key)
            old.flush()
            old.close()
        directory = self.output_dir / "snapshots" / date_key
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{symbol}.jsonl"
        handle = path.open("a", encoding="utf-8", newline="\n")
        self._raw_handles[key] = handle
        return handle

    def _write_raw_if_due(
        self,
        snapshot: DomSnapshot,
        *,
        changed: bool,
    ) -> None:
        symbol = snapshot.symbol
        previous_write = self._last_raw_write.get(symbol)
        due = (
            previous_write is None
            or (
                snapshot.captured_at - previous_write
            ).total_seconds()
            >= self.heartbeat_seconds
        )
        fingerprint = snapshot.fingerprint()
        if not changed and not due:
            return
        if (
            not due
            and self._last_raw_fingerprint.get(symbol) == fingerprint
        ):
            return
        record = snapshot.raw_record()
        date_key = snapshot.captured_at.strftime("%Y-%m-%d")
        handle = self._raw_handle(symbol, date_key)
        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        self._last_raw_write[symbol] = snapshot.captured_at
        self._last_raw_fingerprint[symbol] = fingerprint

    def _write_event(self, event: Mapping[str, Any]) -> None:
        symbol = str(event["symbol"])
        bar_open = pd.Timestamp(event["bar_open"])
        date_key = bar_open.strftime("%Y-%m-%d")
        event_path = self.output_dir / "events" / f"{date_key}.jsonl"
        line = json.dumps(
            dict(event),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with event_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        latest_path = self.output_dir / "latest" / f"{symbol}.json"
        temp_path = latest_path.with_suffix(".json.tmp")
        temp_path.write_text(line + "\n", encoding="utf-8")
        os.replace(temp_path, latest_path)

    def event_asof(
        self,
        symbol: str,
        candle_open: Any,
        decision_time: Any,
    ) -> Optional[dict[str, Any]]:
        try:
            expected_open = pd.Timestamp(candle_open)
            decision = pd.Timestamp(decision_time)
        except Exception:
            return None
        if (
            expected_open.tzinfo is None
            or decision.tzinfo is None
            or expected_open.utcoffset() != pd.Timedelta(0)
            or decision.utcoffset() != pd.Timedelta(0)
        ):
            return None
        key = str(symbol or "").strip().upper()
        with self._lock:
            event = self._latest_events.get(key)
            if event is None:
                return None
            candidate = dict(event)
        validated = validate_liquidity_event(candidate)
        if validated is None:
            return None
        if (
            pd.Timestamp(validated["bar_open"]) != expected_open
            or pd.Timestamp(validated["bar_close"]) > decision
            or pd.Timestamp(validated["available_at"]) > decision
        ):
            return None
        return validated
