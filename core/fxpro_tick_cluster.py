"""FxPro MT5 bid-tick cluster capture and forward-observed M15 publication.

This module reconstructs the same broker feed Quantower reconstructs, but
in-process from ``MetaTrader5.copy_ticks_range`` against the terminal the bot
already trades through.  That removes the cross-venue candle-identity problem:
MT5 builds FX candles from bid quotes, so an event whose OHLC is derived from
the very same bid ticks agrees with the execution candle by construction
rather than by luck.

What this is *not*:

* It is not True Absorption.  FxPro publishes no exchange tape, so a tick is a
  quote update, never a proven execution and never a proven aggressor.
* ``volume`` is a **bid-tick count**, not traded volume.  Spot FX ticks from
  this broker carry no usable ``volume``/``volume_real``, so every classified
  tick contributes exactly one unit.  The measure is sealed into each event as
  ``volume_measure`` so no downstream reader can mistake it for lots.

Classification rule (``mt5_bid_tickdirection_up_down``): a tick whose bid rose
against the previous observed bid is an up-tick, a tick whose bid fell is a
down-tick, and a tick that leaves the bid unchanged (an ask-only update, or a
repeat) stays unclassified.  Unclassified ticks still count toward ``volume``
and ``trades``, so ``classification_ratio`` stays an honest measurement of how
much of the bar could be read directionally instead of being inflated to 1.0
by definition.

Availability is measured, never inferred: ``available_at`` is the UTC instant
at which this process finished pulling the ticks and sealed the bar.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd

from core.fxpro_cluster_rejection import (
    CLUSTER_CLASSIFICATION_MT5_BIDASK,
    CLUSTER_DATA_KIND,
    CLUSTER_EVENT_SCHEMA_VERSION,
    CLUSTER_MARKET_TYPE,
    CLUSTER_SOURCE_MT5_BIDASK,
    CLUSTER_TIMEFRAME,
    CLUSTER_VENUE,
    canonical_hash,
    seal_cluster_event,
    validate_cluster_event,
)


RECORDER_ID = "fxpro-mt5-bidtick-cluster"
RECORDER_VERSION = 1
RAW_TICK_SCHEMA = "forexbot.fxpro-mt5-tick-bar"
RAW_TICK_SCHEMA_VERSION = 1

#: Sealed into every event so a reader can never read the counts as lots.
VOLUME_MEASURE = "bid_tick_count_no_traded_volume"
TICK_SOURCE = "mt5_copy_ticks_range_info"

SIDECAR_SCHEMA = "forexbot.fxpro-cluster-rejection-15m"
SIDECAR_SCHEMA_VERSION = 1

BAR = pd.Timedelta(minutes=15)
#: Ticks pulled before ``bar_open`` only seed the previous bid; they never
#: contribute volume to the bar itself.
SEED_LOOKBEHIND = pd.Timedelta(minutes=2)

AVAILABILITY_EVIDENCE = (
    "available_at is the UTC instant this recorder finished pulling "
    "MetaTrader5.copy_ticks_range for the closed bar and sealed the event; "
    "it is measured locally and never inferred from the bar close"
)


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_line(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _tick_value(tick: Any, name: str) -> Any:
    """Read one field from a numpy row, a mapping, or a plain object."""

    try:
        return tick[name]
    except (TypeError, KeyError, IndexError, ValueError):
        return getattr(tick, name, None)


def _finite_positive(value: Any) -> Optional[float]:
    if isinstance(value, (bool, str, bytes)) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _tick_millis(tick: Any) -> Optional[int]:
    raw = _tick_value(tick, "time_msc")
    if raw is None:
        seconds = _tick_value(tick, "time")
        if seconds is None:
            return None
        try:
            return int(float(seconds) * 1000.0)
        except (TypeError, ValueError, OverflowError):
            return None
    try:
        return int(raw)
    except (TypeError, ValueError, OverflowError):
        return None


def capture_descriptor(
    *,
    symbol: str,
    source_symbol: str,
    tick_size: float,
) -> dict[str, Any]:
    """Deterministic provenance anchor replacing an upstream export manifest."""

    return {
        "recorder": RECORDER_ID,
        "recorder_version": RECORDER_VERSION,
        "source": CLUSTER_SOURCE_MT5_BIDASK,
        "classification": CLUSTER_CLASSIFICATION_MT5_BIDASK,
        "volume_measure": VOLUME_MEASURE,
        "tick_source": TICK_SOURCE,
        "price_basis": "bid",
        "symbol": str(symbol),
        "source_symbol": str(source_symbol),
        "tick_size": repr(float(tick_size)),
    }


@dataclass(frozen=True)
class TickClusterBar:
    """Result of aggregating one closed M15 of bid ticks."""

    event: Optional[dict[str, Any]]
    reason: str
    tick_count: int = 0
    classified_ticks: int = 0
    level_count: int = 0

    @property
    def ok(self) -> bool:
        return self.event is not None


@dataclass
class _LevelCounts:
    volume: int = 0
    trades: int = 0
    buy: int = 0
    sell: int = 0


def aggregate_bid_tick_cluster(
    *,
    symbol: str,
    source_symbol: str,
    ticks: Iterable[Any],
    bar_open: Any,
    tick_size: float,
    available_at: Any,
    seed_bid: Optional[float] = None,
) -> TickClusterBar:
    """Build one sealed cluster event from raw MT5 bid ticks.

    ``ticks`` may span a wider window than the bar.  Ticks before ``bar_open``
    are used only to establish the bid the first in-bar tick is compared
    against; they never contribute volume.  Ticks at or after ``bar_close`` are
    discarded, so a sloppy caller range can never leak the next bar into this
    one.
    """

    size = _finite_positive(tick_size)
    if size is None:
        return TickClusterBar(None, "invalid tick_size")
    open_ts = pd.Timestamp(bar_open)
    if open_ts.tzinfo is None:
        return TickClusterBar(None, "bar_open must be tz-aware UTC")
    open_ts = open_ts.tz_convert("UTC")
    if (
        open_ts.minute % 15
        or open_ts.second
        or open_ts.microsecond
        or open_ts.nanosecond
    ):
        return TickClusterBar(None, "bar_open is not on a 15-minute boundary")
    close_ts = open_ts + BAR

    open_ms = int(open_ts.value // 1_000_000)
    close_ms = int(close_ts.value // 1_000_000)
    step = Decimal(repr(size))

    previous_bid = _finite_positive(seed_bid)
    levels: dict[int, _LevelCounts] = {}
    raw_rows: list[list[Any]] = []
    first_bid: Optional[float] = None
    last_bid: Optional[float] = None
    high_bid: Optional[float] = None
    low_bid: Optional[float] = None
    tick_count = 0
    classified = 0

    for tick in ticks:
        millis = _tick_millis(tick)
        bid = _finite_positive(_tick_value(tick, "bid"))
        if millis is None or bid is None:
            continue
        if millis < open_ms:
            # Pre-bar tick: seed the direction reference only.
            previous_bid = bid
            continue
        if millis >= close_ms:
            continue

        ask = _finite_positive(_tick_value(tick, "ask"))
        try:
            flags = int(_tick_value(tick, "flags") or 0)
        except (TypeError, ValueError):
            flags = 0
        raw_rows.append(
            [millis, repr(bid), repr(ask) if ask is not None else None, flags]
        )

        price_ticks = int(
            (Decimal(repr(bid)) / step).to_integral_value(
                rounding=ROUND_HALF_UP
            )
        )
        bucket = levels.get(price_ticks)
        if bucket is None:
            bucket = _LevelCounts()
            levels[price_ticks] = bucket
        bucket.volume += 1
        bucket.trades += 1

        if previous_bid is not None:
            if bid > previous_bid:
                bucket.buy += 1
                classified += 1
            elif bid < previous_bid:
                bucket.sell += 1
                classified += 1
        previous_bid = bid

        tick_count += 1
        if first_bid is None:
            first_bid = bid
        last_bid = bid
        high_bid = bid if high_bid is None else max(high_bid, bid)
        low_bid = bid if low_bid is None else min(low_bid, bid)

    if not tick_count or first_bid is None or last_bid is None:
        return TickClusterBar(None, "no bid ticks inside the bar")
    if high_bid is None or low_bid is None or high_bid <= low_bid:
        return TickClusterBar(
            None,
            "bar has no bid range",
            tick_count=tick_count,
            classified_ticks=classified,
            level_count=len(levels),
        )

    price_levels: list[dict[str, Any]] = []
    totals = _LevelCounts()
    for price_ticks in sorted(levels):
        counts = levels[price_ticks]
        price = float(step * Decimal(price_ticks))
        price_levels.append(
            {
                "price": price,
                "price_ticks": price_ticks,
                "volume": float(counts.volume),
                "trades": int(counts.trades),
                "buy_volume": float(counts.buy),
                "sell_volume": float(counts.sell),
                "buy_trades": int(counts.buy),
                "sell_trades": int(counts.sell),
                "delta": float(counts.buy - counts.sell),
            }
        )
        totals.volume += counts.volume
        totals.trades += counts.trades
        totals.buy += counts.buy
        totals.sell += counts.sell

    available_ts = pd.Timestamp(available_at)
    if available_ts.tzinfo is None:
        return TickClusterBar(None, "available_at must be tz-aware UTC")
    available_ts = max(available_ts.tz_convert("UTC"), close_ts)

    descriptor = capture_descriptor(
        symbol=symbol,
        source_symbol=source_symbol,
        tick_size=size,
    )
    source_bar_hash = canonical_hash(
        {
            "descriptor": descriptor,
            "bar_open": open_ts.isoformat(),
            "bar_close": close_ts.isoformat(),
            "ticks": raw_rows,
        }
    )
    payload = {
        "schema_version": CLUSTER_EVENT_SCHEMA_VERSION,
        "symbol": str(symbol).strip().upper(),
        "source_symbol": str(source_symbol),
        "timeframe": CLUSTER_TIMEFRAME,
        "bar_open": open_ts.isoformat(),
        "bar_close": close_ts.isoformat(),
        "available_at": available_ts.isoformat(),
        "availability_mode": "forward_observed",
        "availability_evidence": AVAILABILITY_EVIDENCE,
        "capture_lag_ms": int(
            (available_ts - close_ts).total_seconds() * 1000.0
        ),
        "source": CLUSTER_SOURCE_MT5_BIDASK,
        "venue": CLUSTER_VENUE,
        "market_type": CLUSTER_MARKET_TYPE,
        "data_kind": CLUSTER_DATA_KIND,
        "classification": CLUSTER_CLASSIFICATION_MT5_BIDASK,
        "volume_measure": VOLUME_MEASURE,
        "tick_source": TICK_SOURCE,
        "price_basis": "bid",
        "execution_proof": False,
        "aggressor_polarity_proven": False,
        "finalized": True,
        "tick_size": float(size),
        "ohlc": {
            "open": float(first_bid),
            "high": float(high_bid),
            "low": float(low_bid),
            "close": float(last_bid),
        },
        "total": {
            "volume": float(totals.volume),
            "trades": int(totals.trades),
            "buy_volume": float(totals.buy),
            "sell_volume": float(totals.sell),
            "buy_trades": int(totals.buy),
            "sell_trades": int(totals.sell),
            "delta": float(totals.buy - totals.sell),
        },
        "price_levels": price_levels,
        "source_bar_hash": source_bar_hash,
        "source_manifest_sha256": canonical_hash(descriptor),
    }
    event = seal_cluster_event(payload)
    if validate_cluster_event(event) is None:
        return TickClusterBar(
            None,
            "aggregated bar failed the cluster event contract",
            tick_count=tick_count,
            classified_ticks=classified,
            level_count=len(price_levels),
        )
    return TickClusterBar(
        event,
        "ok",
        tick_count=tick_count,
        classified_ticks=classified,
        level_count=len(price_levels),
    )


@dataclass(frozen=True)
class TickClusterSymbol:
    """One captured symbol and its MT5 counterpart."""

    symbol: str
    source_symbol: str


class FxProTickClusterRecorder:
    """Pull closed-M15 bid ticks from MT5 and publish a rolling live sidecar.

    The recorder is read-only against MT5 and independent of the entry flag:
    capture may run for weeks of forward shadow collection while
    ``FXPRO_CLUSTER_REJECTION_ENTRY_ENABLED`` stays off.
    """

    def __init__(
        self,
        *,
        mt5_module: Any,
        symbols: Iterable[TickClusterSymbol | Mapping[str, str] | str],
        output_dir: Path,
        sidecar_dir: Optional[Path] = None,
        retention_days: int = 2,
        settle_seconds: float = 2.0,
        poll_seconds: float = 20.0,
        max_catchup_bars: int = 4,
        archive_raw_ticks: bool = True,
    ) -> None:
        resolved: list[TickClusterSymbol] = []
        for item in symbols:
            if isinstance(item, TickClusterSymbol):
                resolved.append(item)
            elif isinstance(item, Mapping):
                name = str(item.get("symbol") or "").strip().upper()
                source = str(item.get("source_symbol") or name).strip()
                if name:
                    resolved.append(TickClusterSymbol(name, source or name))
            else:
                name = str(item).strip().upper()
                if name:
                    resolved.append(TickClusterSymbol(name, name))
        if not resolved:
            raise ValueError("tick cluster recorder requires at least one symbol")

        self.mt5 = mt5_module
        self.symbols = tuple(dict((s.symbol, s) for s in resolved).values())
        self.output_dir = Path(output_dir)
        self.sidecar_dir = Path(sidecar_dir) if sidecar_dir else None
        self.retention_days = max(1, int(retention_days))
        self.settle = pd.Timedelta(seconds=max(0.0, float(settle_seconds)))
        self.poll_seconds = max(1.0, float(poll_seconds))
        self.max_catchup_bars = max(1, int(max_catchup_bars))
        self.archive_raw_ticks = bool(archive_raw_ticks)

        self.logger = logging.getLogger("fxpro_tick_cluster")
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_bar: dict[str, pd.Timestamp] = {}
        self._window: dict[tuple[str, str], dict[str, Any]] = {}
        self._tick_size: dict[str, float] = {}

        (self.output_dir / "events").mkdir(parents=True, exist_ok=True)
        if self.archive_raw_ticks:
            (self.output_dir / "ticks").mkdir(parents=True, exist_ok=True)
        self._load_recent_events()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="FxProTickClusterRecorder",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, float(timeout)))

    def _run(self) -> None:
        self.logger.info(
            "Starting FxPro tick cluster recorder symbols=%s poll=%.1fs",
            [item.symbol for item in self.symbols],
            self.poll_seconds,
        )
        self._compress_completed_tick_days()
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.poll_once()
            except Exception:
                self.logger.exception("FxPro tick cluster poll failed")
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, self.poll_seconds - elapsed))
        self.logger.info("FxPro tick cluster recorder stopped")

    # -- capture -----------------------------------------------------------

    def _resolve_tick_size(self, source_symbol: str) -> Optional[float]:
        cached = self._tick_size.get(source_symbol)
        if cached is not None:
            return cached
        # copy_ticks_range can return nothing for a symbol that is not in
        # Market Watch, and this recorder may start before the executor has
        # ever touched the symbol.  Selecting is idempotent and read-only.
        try:
            self.mt5.symbol_select(source_symbol, True)
        except Exception as exc:
            self.logger.warning(
                "symbol_select failed for %s: %s", source_symbol, exc
            )
        try:
            info = self.mt5.symbol_info(source_symbol)
        except Exception as exc:
            self.logger.warning(
                "symbol_info failed for %s: %s", source_symbol, exc
            )
            return None
        if info is None:
            return None
        size = _finite_positive(getattr(info, "point", None))
        if size is None:
            digits = getattr(info, "digits", None)
            try:
                size = float(10.0 ** -int(digits))
            except (TypeError, ValueError):
                return None
        self._tick_size[source_symbol] = size
        return size

    def _copy_ticks(
        self,
        source_symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> Optional[Sequence[Any]]:
        flags = getattr(self.mt5, "COPY_TICKS_INFO", 1)
        # The MT5 Python API interprets naive datetimes as UTC.
        date_from = start.tz_convert("UTC").tz_localize(None).to_pydatetime()
        date_to = end.tz_convert("UTC").tz_localize(None).to_pydatetime()
        try:
            ticks = self.mt5.copy_ticks_range(
                source_symbol, date_from, date_to, flags
            )
        except Exception as exc:
            self.logger.warning(
                "copy_ticks_range failed for %s: %s", source_symbol, exc
            )
            return None
        if ticks is None:
            last_error = getattr(self.mt5, "last_error", lambda: None)()
            self.logger.warning(
                "copy_ticks_range returned no data for %s: %s",
                source_symbol,
                last_error,
            )
            return None
        return ticks

    def poll_once(self) -> None:
        now = _utc_now()
        newest_closed = (now - self.settle).floor("15min") - BAR
        for item in self.symbols:
            try:
                self._capture_symbol(item, newest_closed)
            except Exception as exc:
                self.logger.warning(
                    "tick cluster capture failed for %s: %s", item.symbol, exc
                )

    def _capture_symbol(
        self,
        item: TickClusterSymbol,
        newest_closed: pd.Timestamp,
    ) -> None:
        previous = self._last_bar.get(item.symbol)
        if previous is None:
            start = newest_closed
        else:
            start = previous + BAR
        if start > newest_closed:
            return
        # Bound catch-up so a long outage cannot mass-publish stale bars.
        earliest = newest_closed - (self.max_catchup_bars - 1) * BAR
        if start < earliest:
            start = earliest

        tick_size = self._resolve_tick_size(item.source_symbol)
        if tick_size is None:
            self.logger.warning(
                "tick size unavailable for %s; cluster capture skipped",
                item.source_symbol,
            )
            return

        published = False
        bar_open = start
        while bar_open <= newest_closed:
            if self._capture_bar(item, bar_open, tick_size):
                published = True
            self._last_bar[item.symbol] = bar_open
            bar_open = bar_open + BAR
        if published:
            self._republish_sidecar()

    def _capture_bar(
        self,
        item: TickClusterSymbol,
        bar_open: pd.Timestamp,
        tick_size: float,
    ) -> bool:
        ticks = self._copy_ticks(
            item.source_symbol,
            bar_open - SEED_LOOKBEHIND,
            bar_open + BAR,
        )
        if ticks is None or len(ticks) == 0:
            return False
        result = aggregate_bid_tick_cluster(
            symbol=item.symbol,
            source_symbol=item.source_symbol,
            ticks=ticks,
            bar_open=bar_open,
            tick_size=tick_size,
            available_at=_utc_now(),
        )
        if not result.ok or result.event is None:
            self.logger.info(
                "cluster bar %s %s not published: %s (ticks=%d)",
                item.symbol,
                bar_open.isoformat(),
                result.reason,
                result.tick_count,
            )
            return False

        event = result.event
        with self._lock:
            self._window[(item.symbol, str(event["bar_open"]))] = event
            self._prune_window()
        self._append_event(event)
        if self.archive_raw_ticks:
            self._archive_ticks(item, bar_open, tick_size, ticks, event)
        self.logger.info(
            "cluster bar %s %s published ticks=%d classified=%d levels=%d",
            item.symbol,
            bar_open.isoformat(),
            result.tick_count,
            result.classified_ticks,
            result.level_count,
        )
        return True

    # -- persistence -------------------------------------------------------

    def _append_event(self, event: Mapping[str, Any]) -> None:
        bar_open = pd.Timestamp(event["bar_open"])
        path = (
            self.output_dir / "events" / f"{bar_open.strftime('%Y-%m-%d')}.jsonl"
        )
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical_line(event) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _archive_ticks(
        self,
        item: TickClusterSymbol,
        bar_open: pd.Timestamp,
        tick_size: float,
        ticks: Sequence[Any],
        event: Mapping[str, Any],
    ) -> None:
        rows: list[list[Any]] = []
        for tick in ticks:
            millis = _tick_millis(tick)
            bid = _finite_positive(_tick_value(tick, "bid"))
            if millis is None or bid is None:
                continue
            ask = _finite_positive(_tick_value(tick, "ask"))
            rows.append([millis, repr(bid), repr(ask) if ask else None])
        payload = {
            "schema": RAW_TICK_SCHEMA,
            "schema_version": RAW_TICK_SCHEMA_VERSION,
            "symbol": item.symbol,
            "source_symbol": item.source_symbol,
            "bar_open": bar_open.isoformat(),
            "bar_close": (bar_open + BAR).isoformat(),
            "tick_size": float(tick_size),
            "source_bar_hash": str(event["source_bar_hash"]),
            "checksum": str(event["checksum"]),
            "ticks": rows,
        }
        directory = self.output_dir / "ticks" / bar_open.strftime("%Y-%m-%d")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{item.symbol}.jsonl"
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical_line(payload) + "\n")
            handle.flush()

    def _compress_completed_tick_days(self) -> None:
        if not self.archive_raw_ticks:
            return
        today = _utc_now().strftime("%Y-%m-%d")
        for path in sorted((self.output_dir / "ticks").glob("*/*.jsonl")):
            if path.parent.name >= today:
                continue
            destination = path.with_suffix(path.suffix + ".gz")
            if destination.exists():
                continue
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            try:
                with path.open("rb") as source, gzip.open(
                    temporary, "wb", compresslevel=6
                ) as target:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(chunk)
                os.replace(temporary, destination)
                path.unlink()
            except Exception as exc:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
                self.logger.warning(
                    "tick archive compression failed for %s: %s", path, exc
                )

    def _load_recent_events(self) -> None:
        """Repopulate the rolling window so a restart keeps the sidecar valid."""

        cutoff = _utc_now() - pd.Timedelta(days=self.retention_days)
        for path in sorted((self.output_dir / "events").glob("*.jsonl")):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if validate_cluster_event(raw) is None:
                    continue
                if pd.Timestamp(raw["bar_open"]) < cutoff:
                    continue
                self._window[(str(raw["symbol"]), str(raw["bar_open"]))] = raw
        for (symbol, bar_open) in self._window:
            stamp = pd.Timestamp(bar_open)
            current = self._last_bar.get(symbol)
            if current is None or stamp > current:
                self._last_bar[symbol] = stamp

    def _prune_window(self) -> None:
        cutoff = _utc_now() - pd.Timedelta(days=self.retention_days)
        for key in [
            key
            for key, event in self._window.items()
            if pd.Timestamp(event["bar_open"]) < cutoff
        ]:
            self._window.pop(key, None)

    # -- rolling sidecar publication ---------------------------------------

    def _republish_sidecar(self) -> None:
        """Atomically refresh the sidecar the live Core reads.

        ``events.jsonl`` is replaced before ``manifest.json`` on purpose: the
        reader keys its reload on the manifest mtime, so it can only ever
        observe a manifest that already has its events on disk.
        """

        if self.sidecar_dir is None:
            return
        with self._lock:
            self._prune_window()
            events = [
                dict(event)
                for _, event in sorted(
                    self._window.items(),
                    key=lambda pair: (pair[0][0], pair[0][1]),
                )
            ]
        if not events:
            return

        root = self.sidecar_dir
        root.mkdir(parents=True, exist_ok=True)
        events_path = root / "events.jsonl"
        events_tmp = root / "events.jsonl.tmp"
        body = "".join(_canonical_line(event) + "\n" for event in events)
        raw = body.encode("utf-8")

        try:
            with events_tmp.open("wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(events_tmp, events_path)

            # FxProClusterEventDataset hashes the checksums in (symbol,
            # bar_open) event order, not in lexicographic checksum order.
            # `events` is already sorted on exactly that key, so preserve it;
            # re-sorting here silently breaks content_sha256 for every sidecar
            # holding more than one event.
            checksums = [str(event["checksum"]) for event in events]
            manifest = {
                "schema": SIDECAR_SCHEMA,
                "schema_version": SIDECAR_SCHEMA_VERSION,
                "event_schema_version": CLUSTER_EVENT_SCHEMA_VERSION,
                "timeframe": CLUSTER_TIMEFRAME,
                "source": CLUSTER_SOURCE_MT5_BIDASK,
                "venue": CLUSTER_VENUE,
                "market_type": CLUSTER_MARKET_TYPE,
                "data_kind": CLUSTER_DATA_KIND,
                "classification": CLUSTER_CLASSIFICATION_MT5_BIDASK,
                "volume_measure": VOLUME_MEASURE,
                "execution_proof": False,
                "aggressor_polarity_proven": False,
                "research_only": False,
                "availability_modes": ["forward_observed"],
                "availability_evidence": AVAILABILITY_EVIDENCE,
                "recorder": RECORDER_ID,
                "recorder_version": RECORDER_VERSION,
                "retention_days": self.retention_days,
                "generated_at": _utc_now().isoformat(),
                "symbols": sorted({str(event["symbol"]) for event in events}),
                "event_count": len(events),
                "content_sha256": canonical_hash(checksums),
                "files": [
                    {
                        "path": events_path.name,
                        "bytes": len(raw),
                        "rows": len(events),
                        "sha256": _sha256_bytes(raw),
                    }
                ],
            }
            manifest_tmp = root / "manifest.json.tmp"
            with manifest_tmp.open("wb") as handle:
                handle.write(_canonical_line(manifest).encode("utf-8") + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(manifest_tmp, root / "manifest.json")
        except Exception as exc:
            for leftover in (events_tmp, root / "manifest.json.tmp"):
                try:
                    leftover.unlink(missing_ok=True)
                except OSError:
                    pass
            self.logger.warning("cluster sidecar publication failed: %s", exc)


__all__ = [
    "AVAILABILITY_EVIDENCE",
    "FxProTickClusterRecorder",
    "RECORDER_ID",
    "RECORDER_VERSION",
    "TICK_SOURCE",
    "TickClusterBar",
    "TickClusterSymbol",
    "VOLUME_MEASURE",
    "aggregate_bid_tick_cluster",
    "capture_descriptor",
]
