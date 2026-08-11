"""Non-executing tick evidence for entry-window latency experiments.

The watcher deliberately has no dependency on :class:`MT5Executor` and its
MT5 facade exposes only quote/tick-history reads.  It can therefore observe a
validated entry plan, but cannot place, modify, or cancel an order.

The unit of analysis is a *plan*, not a tick.  Tick history is consumed in
causal time order from the exact MT5 ``time_msc`` watermark observed when the
plan was activated.  The resulting SQLite database is intentionally separate
from ``trades.db`` so diagnostic rows can never affect live win-rate or risk
statistics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence


UTC = timezone.utc
_TERMINAL_STATES = {"FILLED", "CANCELLED", "EXPIRED", "INVALIDATED"}
_TERMINAL_PRECEDENCE = {
    "EXPIRED": 0,
    "CANCELLED": 1,
    "INVALIDATED": 2,
    "FILLED": 3,
}


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _epoch_msc(value: datetime) -> int:
    return int(_utc(value).timestamp() * 1000)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat()


def _mt5_datetime(epoch_msc: int) -> datetime:
    # Match the existing FxPro tick recorder: MT5's Python bridge interprets a
    # naive datetime as UTC.
    return datetime.fromtimestamp(epoch_msc / 1000.0, tz=UTC).replace(tzinfo=None)


def _finite_positive(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    try:
        return item[name]
    except (KeyError, IndexError, TypeError, ValueError):
        return getattr(item, name, default)


def stable_plan_id(
    *,
    deployment_id: str,
    symbol: str,
    candidate_signature: str,
    decision_bar_close: datetime,
) -> str:
    """Return an idempotent identity for repeated scans of one candidate."""

    raw = "|".join(
        (
            str(deployment_id).strip(),
            str(symbol).strip().upper(),
            str(candidate_signature).strip(),
            _iso(decision_bar_close),
        )
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class QuoteTick:
    time_msc: int
    bid: float
    ask: float
    flags: int = 0

    @property
    def key(self) -> tuple[int, float, float, int]:
        return (int(self.time_msc), float(self.bid), float(self.ask), int(self.flags))


@dataclass(frozen=True)
class ShadowPlanSpec:
    plan_id: str
    symbol: str
    side: str
    entry_min: float
    entry_max: float
    planned_entry: float
    stop_price: float
    activated_at_utc: datetime
    activation_tick_msc: int
    expires_at_utc: datetime
    activation_source_tick_msc: Optional[int] = None
    source_symbol: Optional[str] = None
    point: float = 0.0
    candidate_signature: str = ""
    trigger_kind: str = ""
    trigger_event_id: Optional[str] = None
    setup_tf: str = ""
    strategy_version: str = ""
    deployment_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    auto_invalidate: bool = True

    def validate(self) -> None:
        if not str(self.plan_id).strip():
            raise ValueError("plan_id is required")
        if not str(self.symbol).strip():
            raise ValueError("symbol is required")
        if str(self.side).upper() not in {"LONG", "SHORT"}:
            raise ValueError("side must be LONG or SHORT")
        values = {
            "entry_min": self.entry_min,
            "entry_max": self.entry_max,
            "planned_entry": self.planned_entry,
            "stop_price": self.stop_price,
            "point": self.point,
        }
        for name, value in values.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if float(self.entry_min) > float(self.entry_max):
            raise ValueError("entry_min cannot exceed entry_max")
        if int(self.activation_tick_msc) < 0:
            raise ValueError("activation_tick_msc must be non-negative")
        if (
            self.activation_source_tick_msc is not None
            and int(self.activation_source_tick_msc) < 0
        ):
            raise ValueError("activation_source_tick_msc must be non-negative")
        activated = _utc(self.activated_at_utc)
        expires = _utc(self.expires_at_utc)
        if expires <= activated:
            raise ValueError("expires_at_utc must be after activated_at_utc")
        causal_msc = _epoch_msc(activated)
        if abs(int(self.activation_tick_msc) - causal_msc) > 5_000:
            raise ValueError(
                "activation_tick_msc must use the UTC observation clock"
            )
        if int(self.activation_tick_msc) >= _epoch_msc(expires):
            raise ValueError("activation_tick_msc must precede expiry")


@dataclass(frozen=True)
class TouchClassification:
    valid: bool
    executable_price: Optional[float]
    spread: Optional[float]
    tolerance: Optional[float]
    strict_touch: bool = False
    tolerant_touch: bool = False
    limit_threshold: bool = False
    limit_cross: bool = False
    gap_beyond_zone: bool = False
    zone_gap_cross: bool = False
    broker_invalidated: bool = False
    legacy_invalidated: bool = False
    ambiguous_touch_and_stop: bool = False


def classify_quote(
    plan: ShadowPlanSpec | Mapping[str, Any],
    quote: QuoteTick,
    *,
    previous_executable_price: Optional[float] = None,
) -> TouchClassification:
    """Classify one quote without making any trading decision.

    Entry semantics match market/pending execution: ASK for LONG and BID for
    SHORT.  Stop invalidation follows the broker instead: BID for a long and
    ASK for a short.  ``legacy_invalidated`` records the current executor's
    pre-entry convention so reports can quantify that difference explicitly.
    """

    getter = plan.get if isinstance(plan, Mapping) else lambda key, default=None: getattr(plan, key, default)
    side = str(getter("side", "")).upper()
    bid = _finite_positive(quote.bid)
    ask = _finite_positive(quote.ask)
    if side not in {"LONG", "SHORT"} or bid is None or ask is None or ask < bid:
        return TouchClassification(False, None, None, None)

    lower = float(getter("entry_min"))
    upper = float(getter("entry_max"))
    planned = float(getter("planned_entry"))
    stop = float(getter("stop_price"))
    point = max(0.0, float(getter("point", 0.0) or 0.0))
    executable = ask if side == "LONG" else bid
    spread = ask - bid
    tolerance = max(spread, point)
    strict = lower <= executable <= upper
    tolerant = lower - tolerance <= executable <= upper + tolerance

    if side == "LONG":
        limit_threshold = executable <= planned
        limit_cross = (
            previous_executable_price is not None
            and float(previous_executable_price) > planned
            and executable <= planned
        )
        gap_beyond = limit_threshold and executable < lower
        zone_gap_cross = (
            previous_executable_price is not None
            and float(previous_executable_price) > upper
            and executable < lower
        )
        broker_invalidated = bid <= stop
        legacy_invalidated = ask <= stop
    else:
        limit_threshold = executable >= planned
        limit_cross = (
            previous_executable_price is not None
            and float(previous_executable_price) < planned
            and executable >= planned
        )
        gap_beyond = limit_threshold and executable > upper
        zone_gap_cross = (
            previous_executable_price is not None
            and float(previous_executable_price) < lower
            and executable > upper
        )
        broker_invalidated = ask >= stop
        legacy_invalidated = bid >= stop

    return TouchClassification(
        valid=True,
        executable_price=executable,
        spread=spread,
        tolerance=tolerance,
        strict_touch=strict,
        tolerant_touch=tolerant,
        limit_threshold=limit_threshold,
        limit_cross=limit_cross,
        gap_beyond_zone=gap_beyond,
        zone_gap_cross=zone_gap_cross,
        broker_invalidated=broker_invalidated,
        legacy_invalidated=legacy_invalidated,
        ambiguous_touch_and_stop=broker_invalidated and (strict or limit_threshold),
    )


class _ReadTickCallable(Protocol):
    def __call__(self, symbol: str, start: datetime, end: datetime, flags: int) -> Any: ...


class MT5ReadOnlyFacade:
    """Capability-limited MT5 adapter: it has no order-related methods."""

    __slots__ = ("_symbol_info_tick", "_copy_ticks_range", "copy_ticks_info")

    def __init__(
        self,
        *,
        symbol_info_tick: Callable[[str], Any],
        copy_ticks_range: _ReadTickCallable,
        copy_ticks_info: int = 1,
    ) -> None:
        self._symbol_info_tick = symbol_info_tick
        self._copy_ticks_range = copy_ticks_range
        self.copy_ticks_info = int(copy_ticks_info)

    @classmethod
    def from_module(cls, module: Any) -> "MT5ReadOnlyFacade":
        # Store bound read callables only; do not retain the module itself.
        return cls(
            symbol_info_tick=getattr(module, "symbol_info_tick"),
            copy_ticks_range=getattr(module, "copy_ticks_range"),
            copy_ticks_info=int(getattr(module, "COPY_TICKS_INFO", 1)),
        )

    def symbol_info_tick(self, symbol: str) -> Any:
        return self._symbol_info_tick(symbol)

    def copy_ticks_range(self, symbol: str, start: datetime, end: datetime) -> Any:
        return self._copy_ticks_range(symbol, start, end, self.copy_ticks_info)


class ShadowTickWatcher:
    """Persist and label entry touches while remaining structurally inert."""

    SCHEMA_VERSION = 2

    def __init__(
        self,
        *,
        mt5: MT5ReadOnlyFacade,
        db_path: Path | str,
        poll_seconds: float = 5.0,
        max_query_span_seconds: float = 60.0,
        max_chunks_per_poll: int = 12,
        logger: Optional[logging.Logger] = None,
        connection_factory: Callable[..., sqlite3.Connection] = sqlite3.connect,
    ) -> None:
        self._mt5 = mt5
        self.db_path = Path(db_path)
        self.poll_seconds = max(1.0, float(poll_seconds))
        self.max_query_span_msc = max(1_000, int(float(max_query_span_seconds) * 1000))
        self.max_chunks_per_poll = max(1, int(max_chunks_per_poll))
        self.logger = logger or logging.getLogger("shadow_tick_watcher")
        self._connection_factory = connection_factory
        self._conn: Optional[sqlite3.Connection] = None
        self._db_lock = threading.RLock()
        self._poll_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_error_log = 0.0
        self.disabled_reason: Optional[str] = None
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = self._connection_factory(
                str(self.db_path), timeout=5.0, check_same_thread=False
            )
            self._conn.row_factory = sqlite3.Row
            self._initialize_schema()
        except Exception as exc:  # fail open: diagnostics never stop trading
            self.disabled_reason = f"{type(exc).__name__}: {exc}"
            self._conn = None
            self._log_failure("initialization", exc)

    @property
    def enabled(self) -> bool:
        return self._conn is not None and self.disabled_reason is None

    def active_plan_ids(self, symbol: Optional[str] = None) -> list[str]:
        """Return non-finalized plans for minute-scan instrumentation.

        This is a diagnostic read only. It lets Core record the quote visible
        to the real 60-second loop even when the original candidate is no
        longer emitted, including after a process restart.
        """
        if not self.enabled:
            return []
        try:
            query = "SELECT plan_id FROM plans WHERE finalized = 0"
            params: tuple[Any, ...] = ()
            if symbol is not None:
                query += " AND symbol = ?"
                params = (str(symbol).strip().upper(),)
            query += " ORDER BY activated_at_utc, plan_id"
            with self._db_lock:
                assert self._conn is not None
                rows = self._conn.execute(query, params).fetchall()
            return [str(row["plan_id"]) for row in rows]
        except Exception as exc:
            self._log_failure("active_plan_ids", exc)
            return []

    def _log_failure(self, operation: str, exc: BaseException) -> None:
        now = time.monotonic()
        if now - self._last_error_log >= 60.0:
            self._last_error_log = now
            self.logger.error("Shadow tick watcher %s failed open: %s", operation, exc)

    def _initialize_schema(self) -> None:
        assert self._conn is not None
        with self._db_lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS shadow_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS plans (
                    plan_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    source_symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_min REAL NOT NULL,
                    entry_max REAL NOT NULL,
                    planned_entry REAL NOT NULL,
                    stop_price REAL NOT NULL,
                    point REAL NOT NULL,
                    activated_at_utc TEXT NOT NULL,
                    activation_tick_msc INTEGER NOT NULL,
                    activation_source_tick_msc INTEGER,
                    expires_at_utc TEXT NOT NULL,
                    expires_tick_msc INTEGER NOT NULL,
                    candidate_signature TEXT,
                    trigger_kind TEXT,
                    trigger_event_id TEXT,
                    setup_tf TEXT,
                    strategy_version TEXT,
                    deployment_id TEXT,
                    metadata_json TEXT,
                    auto_invalidate INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    terminal_reason TEXT,
                    terminal_at_utc TEXT,
                    terminal_tick_msc INTEGER,
                    terminal_inclusive INTEGER NOT NULL DEFAULT 0,
                    finalized INTEGER NOT NULL DEFAULT 0,
                    covered_through_msc INTEGER NOT NULL,
                    last_tick_msc INTEGER,
                    last_exec_price REAL,
                    tick_count INTEGER NOT NULL DEFAULT 0,
                    invalid_tick_count INTEGER NOT NULL DEFAULT 0,
                    gap_count INTEGER NOT NULL DEFAULT 0,
                    data_complete INTEGER NOT NULL DEFAULT 1,
                    strict_touch_ticks INTEGER NOT NULL DEFAULT 0,
                    tolerant_touch_ticks INTEGER NOT NULL DEFAULT 0,
                    limit_touch_ticks INTEGER NOT NULL DEFAULT 0,
                    first_strict_touch_msc INTEGER,
                    first_tolerant_touch_msc INTEGER,
                    first_limit_touch_msc INTEGER,
                    first_gap_msc INTEGER,
                    first_broker_invalidated_msc INTEGER,
                    first_legacy_invalidated_msc INTEGER,
                    first_scan_touch_msc INTEGER,
                    first_policy_touch_msc INTEGER,
                    missed_by_cadence INTEGER,
                    missed_by_policy INTEGER,
                    recoverable_by_limit INTEGER,
                    result_class TEXT,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_shadow_plans_work
                    ON plans(finalized, source_symbol, covered_through_msc);
                CREATE INDEX IF NOT EXISTS idx_shadow_plans_symbol
                    ON plans(symbol, activated_at_utc);
                CREATE TABLE IF NOT EXISTS scan_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_id TEXT NOT NULL REFERENCES plans(plan_id),
                    observed_at_utc TEXT NOT NULL,
                    tick_msc INTEGER NOT NULL,
                    source_tick_msc INTEGER,
                    bid REAL,
                    ask REAL,
                    executable_price REAL,
                    strict_touch INTEGER NOT NULL,
                    tolerant_touch INTEGER NOT NULL,
                    candidate_matches INTEGER NOT NULL,
                    policy_executable INTEGER NOT NULL,
                    disposition TEXT,
                    UNIQUE(plan_id, observed_at_utc, tick_msc)
                );
                CREATE INDEX IF NOT EXISTS idx_shadow_scan_plan
                    ON scan_observations(plan_id, tick_msc);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_key TEXT NOT NULL UNIQUE,
                    plan_id TEXT NOT NULL REFERENCES plans(plan_id),
                    event_type TEXT NOT NULL,
                    event_at_utc TEXT NOT NULL,
                    tick_msc INTEGER,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_shadow_events_plan
                    ON events(plan_id, tick_msc);
                CREATE TABLE IF NOT EXISTS tick_batches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_symbol TEXT NOT NULL,
                    range_start_msc INTEGER NOT NULL,
                    range_end_msc INTEGER NOT NULL,
                    tick_count INTEGER NOT NULL,
                    valid_tick_count INTEGER NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    captured_at_utc TEXT NOT NULL
                );
                """
            )
            previous_version_row = self._conn.execute(
                "SELECT value FROM shadow_meta WHERE key = 'schema_version'"
            ).fetchone()
            try:
                previous_version = int(previous_version_row[0])
            except (TypeError, ValueError, IndexError):
                previous_version = 1

            plan_columns = {
                str(row[1])
                for row in self._conn.execute("PRAGMA table_info(plans)").fetchall()
            }
            if "activation_source_tick_msc" not in plan_columns:
                self._conn.execute(
                    "ALTER TABLE plans ADD COLUMN activation_source_tick_msc INTEGER"
                )
            scan_columns = {
                str(row[1])
                for row in self._conn.execute(
                    "PRAGMA table_info(scan_observations)"
                ).fetchall()
            }
            if "source_tick_msc" not in scan_columns:
                self._conn.execute(
                    "ALTER TABLE scan_observations ADD COLUMN source_tick_msc INTEGER"
                )

            if previous_version < 2:
                self._conn.execute(
                    """
                    UPDATE plans
                    SET activation_source_tick_msc = COALESCE(
                        activation_source_tick_msc,
                        activation_tick_msc
                    )
                    """
                )
                # Schema v1 used raw ``symbol_info_tick.time_msc`` as the causal
                # watermark. Some MT5 terminals expose that field in broker
                # server time while copy_ticks_range is UTC. Such rows cannot
                # support a binary no-touch conclusion and must remain UNKNOWN.
                self._conn.execute(
                    """
                    UPDATE plans
                    SET status = CASE
                            WHEN finalized = 0 THEN 'INVALIDATED'
                            ELSE status
                        END,
                        terminal_reason = 'legacy watcher clock-domain mismatch',
                        terminal_at_utc = COALESCE(
                            terminal_at_utc,
                            expires_at_utc
                        ),
                        terminal_tick_msc = COALESCE(
                            terminal_tick_msc,
                            expires_tick_msc
                        ),
                        finalized = 1,
                        data_complete = 0,
                        first_strict_touch_msc = NULL,
                        first_tolerant_touch_msc = NULL,
                        first_limit_touch_msc = NULL,
                        first_gap_msc = NULL,
                        first_broker_invalidated_msc = NULL,
                        first_legacy_invalidated_msc = NULL,
                        first_scan_touch_msc = NULL,
                        first_policy_touch_msc = NULL,
                        missed_by_cadence = NULL,
                        missed_by_policy = NULL,
                        recoverable_by_limit = NULL,
                        result_class = 'UNKNOWN_CLOCK_DOMAIN',
                        updated_at_utc = ?
                    WHERE activation_tick_msc >= expires_tick_msc
                       OR ABS(
                            activation_tick_msc
                            - CAST(strftime('%s', activated_at_utc) AS INTEGER)
                              * 1000
                       ) > 60000
                    """,
                    (_iso(datetime.now(UTC)),),
                )
            self._conn.execute(
                "INSERT OR REPLACE INTO shadow_meta(key, value) VALUES('schema_version', ?)",
                (str(self.SCHEMA_VERSION),),
            )
            self._conn.commit()

    def start(self) -> bool:
        if not self.enabled:
            return False
        if self._thread is not None and self._thread.is_alive():
            return True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="shadow-tick-watcher", daemon=True
        )
        self._thread.start()
        return True

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout)))

    def close(self) -> None:
        self.stop()
        conn = self._conn
        if conn is None:
            return
        try:
            with self._db_lock:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                conn.close()
        except Exception as exc:
            self._log_failure("close", exc)
        finally:
            self._conn = None

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            self.poll_once()
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, self.poll_seconds - elapsed))

    def register_plan(
        self, spec: ShadowPlanSpec, *, activation_quote: Optional[QuoteTick] = None
    ) -> bool:
        """Persist a plan once; repeated scans never extend its original TTL."""

        if not self.enabled:
            return False
        try:
            spec.validate()
            now = datetime.now(UTC)
            metadata = json.dumps(dict(spec.metadata), sort_keys=True, default=str)
            source_symbol = (spec.source_symbol or spec.symbol).strip()
            expires_msc = _epoch_msc(spec.expires_at_utc)
            with self._db_lock:
                assert self._conn is not None
                cursor = self._conn.execute(
                    """
                    INSERT OR IGNORE INTO plans (
                        plan_id, symbol, source_symbol, side,
                        entry_min, entry_max, planned_entry, stop_price, point,
                        activated_at_utc, activation_tick_msc,
                        activation_source_tick_msc,
                        expires_at_utc, expires_tick_msc,
                        candidate_signature, trigger_kind, trigger_event_id,
                        setup_tf, strategy_version, deployment_id, metadata_json,
                        auto_invalidate, covered_through_msc,
                        created_at_utc, updated_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        spec.plan_id,
                        spec.symbol.strip().upper(),
                        source_symbol,
                        spec.side.upper(),
                        float(spec.entry_min),
                        float(spec.entry_max),
                        float(spec.planned_entry),
                        float(spec.stop_price),
                        max(0.0, float(spec.point)),
                        _iso(spec.activated_at_utc),
                        int(spec.activation_tick_msc),
                        (
                            None
                            if spec.activation_source_tick_msc is None
                            else int(spec.activation_source_tick_msc)
                        ),
                        _iso(spec.expires_at_utc),
                        expires_msc,
                        spec.candidate_signature,
                        spec.trigger_kind,
                        spec.trigger_event_id,
                        spec.setup_tf,
                        spec.strategy_version,
                        spec.deployment_id,
                        metadata,
                        int(bool(spec.auto_invalidate)),
                        int(spec.activation_tick_msc) - 1,
                        _iso(now),
                        _iso(now),
                    ),
                )
                created = cursor.rowcount == 1
                if created:
                    self._insert_event_locked(
                        spec.plan_id,
                        "PLAN_CREATED",
                        event_at=spec.activated_at_utc,
                        tick_msc=int(spec.activation_tick_msc),
                        payload={
                            "expires_at_utc": _iso(spec.expires_at_utc),
                            "activation_source_tick_msc": (
                                spec.activation_source_tick_msc
                            ),
                        },
                    )
                self._conn.commit()
            if created and activation_quote is not None:
                causal_quote = QuoteTick(
                    time_msc=int(spec.activation_tick_msc),
                    bid=float(activation_quote.bid),
                    ask=float(activation_quote.ask),
                    flags=int(activation_quote.flags),
                )
                self._apply_ticks_to_plan(
                    spec.plan_id,
                    [causal_quote],
                    coverage_end_msc=int(spec.activation_tick_msc),
                )
            return created
        except Exception as exc:
            self._log_failure("register_plan", exc)
            return False

    def capture_activation_quote(self, source_symbol: str) -> Optional[QuoteTick]:
        """Read one activation watermark through the read-only facade."""

        if not self.enabled:
            return None
        try:
            return self._normalize_tick(self._mt5.symbol_info_tick(source_symbol))
        except Exception as exc:
            self._log_failure("activation_quote", exc)
            return None

    def record_scan(
        self,
        plan_id: str,
        quote: QuoteTick,
        *,
        observed_at_utc: datetime,
        candidate_matches: bool,
        policy_executable: bool,
        disposition: Optional[str] = None,
    ) -> bool:
        """Record what the real 60-second loop could see at this scan."""

        if not self.enabled:
            return False
        try:
            observed_msc = _epoch_msc(observed_at_utc)
            with self._db_lock:
                assert self._conn is not None
                row = self._conn.execute(
                    "SELECT * FROM plans WHERE plan_id = ?", (plan_id,)
                ).fetchone()
                if row is None or int(row["finalized"]):
                    return False
                if not self._tick_inside_plan(row, observed_msc):
                    return False
                classification = classify_quote(dict(row), quote)
                policy_touch = bool(
                    classification.strict_touch
                    and candidate_matches
                    and policy_executable
                )
                self._conn.execute(
                    """
                    INSERT INTO scan_observations (
                        plan_id, observed_at_utc, tick_msc, source_tick_msc,
                        bid, ask,
                        executable_price, strict_touch, tolerant_touch,
                        candidate_matches, policy_executable, disposition
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(plan_id, observed_at_utc, tick_msc)
                    DO UPDATE SET
                        candidate_matches = MAX(
                            scan_observations.candidate_matches,
                            excluded.candidate_matches
                        ),
                        policy_executable = MAX(
                            scan_observations.policy_executable,
                            excluded.policy_executable
                        ),
                        source_tick_msc = excluded.source_tick_msc,
                        disposition = CASE
                            WHEN excluded.candidate_matches = 1
                            THEN excluded.disposition
                            ELSE scan_observations.disposition
                        END
                    """,
                    (
                        plan_id,
                        _iso(observed_at_utc),
                        observed_msc,
                        int(quote.time_msc),
                        float(quote.bid),
                        float(quote.ask),
                        classification.executable_price,
                        int(classification.strict_touch),
                        int(classification.tolerant_touch),
                        int(bool(candidate_matches)),
                        int(bool(policy_executable)),
                        disposition,
                    ),
                )
                updates: list[str] = ["updated_at_utc = ?"]
                values: list[Any] = [_iso(datetime.now(UTC))]
                if classification.strict_touch and row["first_scan_touch_msc"] is None:
                    updates.append("first_scan_touch_msc = ?")
                    values.append(observed_msc)
                    self._insert_event_locked(
                        plan_id,
                        "SCAN_TOUCH",
                        event_at=observed_at_utc,
                        tick_msc=observed_msc,
                        payload={
                            "candidate_matches": bool(candidate_matches),
                            "source_tick_msc": int(quote.time_msc),
                        },
                    )
                if policy_touch and row["first_policy_touch_msc"] is None:
                    updates.append("first_policy_touch_msc = ?")
                    values.append(observed_msc)
                    self._insert_event_locked(
                        plan_id,
                        "POLICY_TOUCH",
                        event_at=observed_at_utc,
                        tick_msc=observed_msc,
                        payload={
                            "disposition": disposition,
                            "source_tick_msc": int(quote.time_msc),
                        },
                    )
                values.append(plan_id)
                self._conn.execute(
                    f"UPDATE plans SET {', '.join(updates)} WHERE plan_id = ?",
                    values,
                )
                self._conn.commit()
            return True
        except Exception as exc:
            self._log_failure("record_scan", exc)
            return False

    def record_terminal(
        self,
        plan_id: str,
        status: str,
        *,
        at_utc: datetime,
        tick_msc: Optional[int] = None,
        reason: Optional[str] = None,
        inclusive: Optional[bool] = None,
    ) -> bool:
        """Observe (never cause) a fill/cancel/expiry/invalidation fact."""

        if not self.enabled:
            return False
        normalized = str(status).upper()
        if normalized not in _TERMINAL_STATES:
            return False
        try:
            event_msc = int(tick_msc) if tick_msc is not None else _epoch_msc(at_utc)
            if inclusive is None:
                inclusive = normalized in {"FILLED", "INVALIDATED"}
            with self._db_lock:
                assert self._conn is not None
                row = self._conn.execute(
                    "SELECT * FROM plans WHERE plan_id = ?",
                    (plan_id,),
                ).fetchone()
                if row is None:
                    return False
                if event_msc < int(row["activation_tick_msc"]):
                    return False
                existing_status = str(row["status"] or "").upper()
                existing_msc = row["terminal_tick_msc"]
                was_finalized = bool(row["finalized"])
                late_authoritative_fill = bool(
                    was_finalized
                    and normalized == "FILLED"
                    and existing_status
                    in {"EXPIRED", "INVALIDATED", "CANCELLED"}
                    and existing_msc is not None
                    and event_msc < int(existing_msc)
                )
                if was_finalized and not late_authoritative_fill:
                    return False
                if not was_finalized:
                    if (
                        existing_status in _TERMINAL_STATES
                        and existing_msc is not None
                        and (
                            int(existing_msc) < event_msc
                            or (
                                int(existing_msc) == event_msc
                                and _TERMINAL_PRECEDENCE[existing_status]
                                >= _TERMINAL_PRECEDENCE[normalized]
                            )
                        )
                    ):
                        # Detection/reconciliation may run after an exact expiry
                        # or broker event was already observed. Never move the
                        # causal horizon forward; on an exact tie keep the
                        # stronger fact.
                        return False
                if late_authoritative_fill:
                    self._reopen_for_earlier_fill_locked(
                        row,
                        fill_tick_msc=event_msc,
                    )
                self._conn.execute(
                    """
                    UPDATE plans
                    SET status = ?, terminal_reason = ?, terminal_at_utc = ?,
                        terminal_tick_msc = ?, terminal_inclusive = ?, updated_at_utc = ?
                    WHERE plan_id = ?
                    """,
                    (
                        normalized,
                        reason,
                        _iso(at_utc),
                        event_msc,
                        int(bool(inclusive)),
                        _iso(datetime.now(UTC)),
                        plan_id,
                    ),
                )
                self._insert_event_locked(
                    plan_id,
                    f"PLAN_{normalized}",
                    event_at=at_utc,
                    tick_msc=event_msc,
                    payload={"reason": reason, "inclusive": bool(inclusive)},
                )
                self._conn.commit()
            self._finalize_ready_plans()
            return True
        except Exception as exc:
            self._log_failure("record_terminal", exc)
            return False

    def _reopen_for_earlier_fill_locked(
        self,
        row: sqlite3.Row,
        *,
        fill_tick_msc: int,
    ) -> None:
        """Recompute horizon-dependent state for a late broker fill fact.

        MT5 deal history can become visible after an inferred expiry or stop
        already finalized.  The broker timestamp is authoritative when it is
        strictly earlier.  We retain captured batches and tick evidence, but
        remove the old derived finalization and clamp every plan-level first
        timestamp to the corrected inclusive fill horizon.
        """

        assert self._conn is not None
        plan_id = str(row["plan_id"])
        activation_msc = int(row["activation_tick_msc"])
        horizon = int(fill_tick_msc)

        scan = self._conn.execute(
            """
            SELECT
                MIN(CASE WHEN strict_touch = 1 THEN tick_msc END)
                    AS first_scan,
                MIN(
                    CASE
                        WHEN strict_touch = 1
                            AND candidate_matches = 1
                            AND policy_executable = 1
                        THEN tick_msc
                    END
                ) AS first_policy
            FROM scan_observations
            WHERE plan_id = ? AND tick_msc BETWEEN ? AND ?
            """,
            (plan_id, activation_msc, horizon),
        ).fetchone()
        gap_count = self._coverage_gap_count_locked(
            plan_id,
            activation_msc=activation_msc,
            horizon_msc=horizon,
        )
        timestamp_fields = (
            "first_strict_touch_msc",
            "first_tolerant_touch_msc",
            "first_limit_touch_msc",
            "first_gap_msc",
            "first_broker_invalidated_msc",
            "first_legacy_invalidated_msc",
        )
        assignments = [
            (
                f"{name} = CASE WHEN {name} <= ? "
                f"THEN {name} ELSE NULL END"
            )
            for name in timestamp_fields
        ]
        self._conn.execute(
            f"""
            UPDATE plans SET
                finalized = 0,
                missed_by_cadence = NULL,
                missed_by_policy = NULL,
                recoverable_by_limit = NULL,
                result_class = NULL,
                first_scan_touch_msc = ?,
                first_policy_touch_msc = ?,
                data_complete = ?,
                gap_count = ?,
                last_tick_msc = CASE
                    WHEN last_tick_msc <= ? THEN last_tick_msc
                    ELSE NULL
                END,
                last_exec_price = CASE
                    WHEN last_tick_msc <= ? THEN last_exec_price
                    ELSE NULL
                END,
                {", ".join(assignments)}
            WHERE plan_id = ?
            """,
            (
                scan["first_scan"] if scan is not None else None,
                scan["first_policy"] if scan is not None else None,
                int(gap_count == 0),
                gap_count,
                horizon,
                horizon,
                *(horizon for _ in timestamp_fields),
                plan_id,
            ),
        )
        # PLAN_FINALIZED is derived state, not immutable source evidence. Keep
        # the original terminal/touch facts and replace exactly one final row.
        self._conn.execute(
            "DELETE FROM events WHERE plan_id = ? AND event_type = 'PLAN_FINALIZED'",
            (plan_id,),
        )
        self._insert_event_locked(
            plan_id,
            "PLAN_CORRECTED_EARLIER_FILL",
            event_at=datetime.fromtimestamp(horizon / 1000.0, tz=UTC),
            tick_msc=horizon,
            payload={
                "previous_status": str(row["status"]),
                "previous_terminal_tick_msc": row["terminal_tick_msc"],
                "previous_result_class": row["result_class"],
            },
        )

    def _coverage_gap_count_locked(
        self,
        plan_id: str,
        *,
        activation_msc: int,
        horizon_msc: int,
    ) -> int:
        """Count unique captured data gaps overlapping a causal horizon."""

        assert self._conn is not None
        rows = self._conn.execute(
            """
            SELECT payload_json FROM events
            WHERE plan_id = ? AND event_type = 'DATA_GAP'
            """,
            (plan_id,),
        ).fetchall()
        count = 0
        for event in rows:
            try:
                payload = json.loads(str(event["payload_json"]))
                start = int(payload["range_start_msc"])
                end = int(payload["range_end_msc"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                # An unreadable gap event cannot establish complete coverage.
                count += 1
                continue
            if start <= horizon_msc and end >= activation_msc:
                count += 1
        return count

    def poll_once(self, *, now_utc: Optional[datetime] = None) -> None:
        if not self.enabled or not self._poll_lock.acquire(blocking=False):
            return
        try:
            now = _utc(now_utc or datetime.now(UTC))
            now_msc = _epoch_msc(now)
            with self._db_lock:
                assert self._conn is not None
                rows = self._conn.execute(
                    "SELECT * FROM plans WHERE finalized = 0 ORDER BY source_symbol, activated_at_utc"
                ).fetchall()
            groups: dict[str, list[sqlite3.Row]] = {}
            for row in rows:
                groups.setdefault(str(row["source_symbol"]), []).append(row)

            for source_symbol, group in groups.items():
                self._poll_symbol(source_symbol, group, now_msc)

            self._expire_due_plans(now)
            self._finalize_ready_plans()
        except Exception as exc:
            self._log_failure("poll_once", exc)
        finally:
            self._poll_lock.release()

    def _poll_symbol(
        self, source_symbol: str, rows: Sequence[sqlite3.Row], now_msc: int
    ) -> None:
        work: list[sqlite3.Row] = []
        for row in rows:
            horizon = self._horizon_msc(row)
            if horizon >= int(row["activation_tick_msc"]) and int(row["covered_through_msc"]) < horizon:
                work.append(row)
        if not work:
            return
        cursor = min(int(row["covered_through_msc"]) for row in work)
        furthest = min(
            now_msc,
            max(self._horizon_msc(row) for row in work),
        )
        chunks = 0
        while cursor < furthest and chunks < self.max_chunks_per_poll:
            chunk_end = min(cursor + self.max_query_span_msc, furthest)
            chunks += 1
            try:
                raw_ticks = self._mt5.copy_ticks_range(
                    source_symbol,
                    _mt5_datetime(cursor + 1),
                    _mt5_datetime(chunk_end),
                )
                if raw_ticks is None:
                    raise RuntimeError("copy_ticks_range returned None")
                ticks = self._normalize_ticks(raw_ticks, cursor + 1, chunk_end)
            except Exception as exc:
                self._mark_coverage_failure(
                    [str(row["plan_id"]) for row in work],
                    cursor + 1,
                    chunk_end,
                    exc,
                )
                self._log_failure("copy_ticks_range", exc)
                return

            payload = json.dumps([tick.key for tick in ticks], separators=(",", ":"))
            with self._db_lock:
                assert self._conn is not None
                self._conn.execute(
                    """
                    INSERT INTO tick_batches (
                        source_symbol, range_start_msc, range_end_msc,
                        tick_count, valid_tick_count, payload_sha256, captured_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_symbol,
                        cursor + 1,
                        chunk_end,
                        len(raw_ticks),
                        len(ticks),
                        hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                        _iso(datetime.now(UTC)),
                    ),
                )
                self._conn.commit()
            for row in work:
                plan_id = str(row["plan_id"])
                self._apply_ticks_to_plan(
                    plan_id, ticks, coverage_end_msc=chunk_end
                )
            cursor = chunk_end

    @staticmethod
    def _normalize_tick(raw: Any) -> Optional[QuoteTick]:
        if raw is None:
            return None
        millis = _field(raw, "time_msc")
        if millis is None:
            seconds = _field(raw, "time")
            millis = int(seconds) * 1000 if seconds is not None else None
        try:
            time_msc = int(millis)
            bid = float(_field(raw, "bid"))
            ask = float(_field(raw, "ask"))
            flags = int(_field(raw, "flags", 0) or 0)
        except (TypeError, ValueError):
            return None
        return QuoteTick(time_msc=time_msc, bid=bid, ask=ask, flags=flags)

    @classmethod
    def _normalize_ticks(
        cls, raw_ticks: Iterable[Any], start_msc: int, end_msc: int
    ) -> list[QuoteTick]:
        unique: dict[tuple[int, float, float, int], QuoteTick] = {}
        for raw in raw_ticks:
            tick = cls._normalize_tick(raw)
            if tick is None or tick.time_msc < start_msc or tick.time_msc > end_msc:
                continue
            unique.setdefault(tick.key, tick)
        return sorted(unique.values(), key=lambda tick: tick.key)

    def _apply_ticks_to_plan(
        self,
        plan_id: str,
        ticks: Sequence[QuoteTick],
        *,
        coverage_end_msc: int,
    ) -> None:
        with self._db_lock:
            assert self._conn is not None
            row = self._conn.execute(
                "SELECT * FROM plans WHERE plan_id = ?", (plan_id,)
            ).fetchone()
            if row is None or int(row["finalized"]):
                return
            state = dict(row)
            covered = int(state["covered_through_msc"])
            horizon = self._horizon_msc(state)
            upper = min(int(coverage_end_msc), horizon)
            if upper <= covered:
                return
            events: list[tuple[str, QuoteTick, dict[str, Any]]] = []
            previous = state.get("last_exec_price")
            terminalized_here = False
            for tick in ticks:
                if tick.time_msc <= covered or tick.time_msc > upper:
                    continue
                classification = classify_quote(
                    state, tick, previous_executable_price=previous
                )
                state["last_tick_msc"] = tick.time_msc
                if not classification.valid:
                    state["invalid_tick_count"] = int(state["invalid_tick_count"]) + 1
                    continue
                state["tick_count"] = int(state["tick_count"]) + 1
                state["last_exec_price"] = classification.executable_price
                previous = classification.executable_price

                for flag, count_name, first_name, event_type in (
                    (
                        classification.strict_touch,
                        "strict_touch_ticks",
                        "first_strict_touch_msc",
                        "TOUCH_STRICT",
                    ),
                    (
                        classification.tolerant_touch,
                        "tolerant_touch_ticks",
                        "first_tolerant_touch_msc",
                        "TOUCH_TOLERANT",
                    ),
                    (
                        classification.limit_threshold,
                        "limit_touch_ticks",
                        "first_limit_touch_msc",
                        "TOUCH_LIMIT_THRESHOLD",
                    ),
                ):
                    if flag:
                        state[count_name] = int(state[count_name]) + 1
                        if state.get(first_name) is None:
                            state[first_name] = tick.time_msc
                            events.append(
                                (
                                    event_type,
                                    tick,
                                    self._classification_payload(classification),
                                )
                            )
                if (
                    (classification.gap_beyond_zone or classification.zone_gap_cross)
                    and state.get("first_gap_msc") is None
                ):
                    state["first_gap_msc"] = tick.time_msc
                    events.append(
                        (
                            "GAP_BEYOND_ZONE",
                            tick,
                            self._classification_payload(classification),
                        )
                    )
                if (
                    classification.legacy_invalidated
                    and state.get("first_legacy_invalidated_msc") is None
                ):
                    state["first_legacy_invalidated_msc"] = tick.time_msc
                if (
                    classification.broker_invalidated
                    and state.get("first_broker_invalidated_msc") is None
                ):
                    state["first_broker_invalidated_msc"] = tick.time_msc
                    events.append(
                        (
                            "BROKER_INVALIDATION",
                            tick,
                            self._classification_payload(classification),
                        )
                    )
                    if int(state.get("auto_invalidate") or 0) and state["status"] == "ACTIVE":
                        state["status"] = "INVALIDATED"
                        state["terminal_reason"] = "broker-side stop crossed before entry"
                        state["terminal_at_utc"] = datetime.fromtimestamp(
                            tick.time_msc / 1000.0, tz=UTC
                        ).isoformat()
                        state["terminal_tick_msc"] = tick.time_msc
                        state["terminal_inclusive"] = 1
                        upper = tick.time_msc
                        terminalized_here = True
                if terminalized_here:
                    break

            state["covered_through_msc"] = upper
            state["updated_at_utc"] = _iso(datetime.now(UTC))
            self._conn.execute(
                """
                UPDATE plans SET
                    status = ?, terminal_reason = ?, terminal_at_utc = ?,
                    terminal_tick_msc = ?, terminal_inclusive = ?,
                    covered_through_msc = ?, last_tick_msc = ?, last_exec_price = ?,
                    tick_count = ?, invalid_tick_count = ?,
                    strict_touch_ticks = ?, tolerant_touch_ticks = ?, limit_touch_ticks = ?,
                    first_strict_touch_msc = ?, first_tolerant_touch_msc = ?,
                    first_limit_touch_msc = ?, first_gap_msc = ?,
                    first_broker_invalidated_msc = ?, first_legacy_invalidated_msc = ?,
                    updated_at_utc = ?
                WHERE plan_id = ?
                """,
                (
                    state["status"],
                    state.get("terminal_reason"),
                    state.get("terminal_at_utc"),
                    state.get("terminal_tick_msc"),
                    int(state.get("terminal_inclusive") or 0),
                    state["covered_through_msc"],
                    state.get("last_tick_msc"),
                    state.get("last_exec_price"),
                    state["tick_count"],
                    state["invalid_tick_count"],
                    state["strict_touch_ticks"],
                    state["tolerant_touch_ticks"],
                    state["limit_touch_ticks"],
                    state.get("first_strict_touch_msc"),
                    state.get("first_tolerant_touch_msc"),
                    state.get("first_limit_touch_msc"),
                    state.get("first_gap_msc"),
                    state.get("first_broker_invalidated_msc"),
                    state.get("first_legacy_invalidated_msc"),
                    state["updated_at_utc"],
                    plan_id,
                ),
            )
            for event_type, tick, payload in events:
                self._insert_event_locked(
                    plan_id,
                    event_type,
                    event_at=datetime.fromtimestamp(tick.time_msc / 1000.0, tz=UTC),
                    tick_msc=tick.time_msc,
                    payload=payload,
                )
            self._conn.commit()

    @staticmethod
    def _classification_payload(classification: TouchClassification) -> dict[str, Any]:
        return {
            "executable_price": classification.executable_price,
            "spread": classification.spread,
            "tolerance": classification.tolerance,
            "strict_touch": classification.strict_touch,
            "tolerant_touch": classification.tolerant_touch,
            "limit_threshold": classification.limit_threshold,
            "limit_cross": classification.limit_cross,
            "gap_beyond_zone": classification.gap_beyond_zone,
            "zone_gap_cross": classification.zone_gap_cross,
            "broker_invalidated": classification.broker_invalidated,
            "legacy_invalidated": classification.legacy_invalidated,
            "ambiguous_touch_and_stop": classification.ambiguous_touch_and_stop,
        }

    def _mark_coverage_failure(
        self,
        plan_ids: Sequence[str],
        start_msc: int,
        end_msc: int,
        exc: BaseException,
    ) -> None:
        with self._db_lock:
            assert self._conn is not None
            for plan_id in plan_ids:
                self._conn.execute(
                    """
                    UPDATE plans SET data_complete = 0, gap_count = gap_count + 1,
                        updated_at_utc = ? WHERE plan_id = ? AND finalized = 0
                    """,
                    (_iso(datetime.now(UTC)), plan_id),
                )
                self._insert_event_locked(
                    plan_id,
                    "DATA_GAP",
                    event_at=datetime.now(UTC),
                    tick_msc=start_msc,
                    payload={
                        "range_start_msc": start_msc,
                        "range_end_msc": end_msc,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            self._conn.commit()

    def _expire_due_plans(self, now: datetime) -> None:
        now_msc = _epoch_msc(now)
        with self._db_lock:
            assert self._conn is not None
            rows = self._conn.execute(
                """
                SELECT plan_id, expires_at_utc, expires_tick_msc
                FROM plans
                WHERE finalized = 0 AND status = 'ACTIVE' AND expires_tick_msc <= ?
                """,
                (now_msc,),
            ).fetchall()
            for row in rows:
                self._conn.execute(
                    """
                    UPDATE plans SET status = 'EXPIRED', terminal_reason = 'entry TTL expired',
                        terminal_at_utc = expires_at_utc,
                        terminal_tick_msc = expires_tick_msc,
                        terminal_inclusive = 0, updated_at_utc = ?
                    WHERE plan_id = ?
                    """,
                    (_iso(datetime.now(UTC)), row["plan_id"]),
                )
                self._insert_event_locked(
                    str(row["plan_id"]),
                    "PLAN_EXPIRED",
                    event_at=datetime.fromtimestamp(
                        int(row["expires_tick_msc"]) / 1000.0, tz=UTC
                    ),
                    tick_msc=int(row["expires_tick_msc"]),
                    payload={"reason": "entry TTL expired", "inclusive": False},
                )
            self._conn.commit()

    def _finalize_ready_plans(self) -> None:
        if not self.enabled:
            return
        with self._db_lock:
            assert self._conn is not None
            rows = self._conn.execute(
                "SELECT * FROM plans WHERE finalized = 0 AND status != 'ACTIVE'"
            ).fetchall()
            for row in rows:
                horizon = self._horizon_msc(row)
                # A successful history stream must reach the complete causal horizon.
                # If the provider failed, however, close the observation explicitly as
                # UNKNOWN_COVERAGE instead of leaving an expired plan pending forever.
                if int(row["covered_through_msc"]) < horizon and bool(row["data_complete"]):
                    continue
                strict = self._effective_touch(row, "first_strict_touch_msc", horizon)
                scan = self._effective_touch(row, "first_scan_touch_msc", horizon)
                policy = self._effective_touch(row, "first_policy_touch_msc", horizon)
                limit_touch = self._effective_touch(row, "first_limit_touch_msc", horizon)
                invalidated = self._effective_touch(
                    row, "first_broker_invalidated_msc", horizon
                )
                complete = bool(row["data_complete"])
                if not complete:
                    missed_cadence: Optional[int] = None
                    missed_policy: Optional[int] = None
                    recoverable: Optional[int] = None
                    result_class = "UNKNOWN_COVERAGE"
                elif strict is None:
                    missed_cadence = 0
                    missed_policy = 0
                    recoverable = int(
                        limit_touch is not None
                        and (invalidated is None or limit_touch < invalidated)
                    )
                    result_class = "NO_STRICT_TICK_TOUCH"
                else:
                    missed_cadence = int(scan is None)
                    missed_policy = int(policy is None)
                    recoverable = int(
                        limit_touch is not None
                        and (invalidated is None or limit_touch < invalidated)
                    )
                    result_class = (
                        "MISSED_BY_CADENCE" if missed_cadence else "SEEN_BY_60S"
                    )
                self._conn.execute(
                    """
                    UPDATE plans SET finalized = 1,
                        missed_by_cadence = ?, missed_by_policy = ?,
                        recoverable_by_limit = ?, result_class = ?, updated_at_utc = ?
                    WHERE plan_id = ?
                    """,
                    (
                        missed_cadence,
                        missed_policy,
                        recoverable,
                        result_class,
                        _iso(datetime.now(UTC)),
                        row["plan_id"],
                    ),
                )
                self._insert_event_locked(
                    str(row["plan_id"]),
                    "PLAN_FINALIZED",
                    event_at=datetime.now(UTC),
                    tick_msc=horizon,
                    payload={
                        "result_class": result_class,
                        "missed_by_cadence": missed_cadence,
                        "missed_by_policy": missed_policy,
                        "recoverable_by_limit": recoverable,
                    },
                )
            self._conn.commit()

    @staticmethod
    def _effective_touch(
        row: Mapping[str, Any] | sqlite3.Row, name: str, horizon: int
    ) -> Optional[int]:
        value = row[name]
        if value is None:
            return None
        tick_msc = int(value)
        return tick_msc if tick_msc <= horizon else None

    @staticmethod
    def _horizon_msc(row: Mapping[str, Any] | sqlite3.Row) -> int:
        status = str(row["status"])
        if status == "ACTIVE" or status == "EXPIRED":
            return int(row["expires_tick_msc"]) - 1
        terminal = row["terminal_tick_msc"]
        if terminal is None:
            return int(row["expires_tick_msc"]) - 1
        return int(terminal) if int(row["terminal_inclusive"]) else int(terminal) - 1

    @classmethod
    def _tick_inside_plan(
        cls, row: Mapping[str, Any] | sqlite3.Row, tick_msc: int
    ) -> bool:
        return int(row["activation_tick_msc"]) <= tick_msc <= cls._horizon_msc(row)

    def _insert_event_locked(
        self,
        plan_id: str,
        event_type: str,
        *,
        event_at: datetime,
        tick_msc: Optional[int],
        payload: Mapping[str, Any],
    ) -> None:
        assert self._conn is not None
        payload_json = json.dumps(dict(payload), sort_keys=True, default=str)
        raw_key = f"{plan_id}|{event_type}|{tick_msc}|{payload_json}".encode("utf-8")
        event_key = hashlib.sha256(raw_key).hexdigest()
        self._conn.execute(
            """
            INSERT OR IGNORE INTO events (
                event_key, plan_id, event_type, event_at_utc, tick_msc, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (event_key, plan_id, event_type, _iso(event_at), tick_msc, payload_json),
        )

    def get_plan(self, plan_id: str) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            with self._db_lock:
                assert self._conn is not None
                row = self._conn.execute(
                    "SELECT * FROM plans WHERE plan_id = ?", (plan_id,)
                ).fetchone()
            return dict(row) if row is not None else None
        except Exception as exc:
            self._log_failure("get_plan", exc)
            return None

    def summary(self, *, finalized_only: bool = True) -> dict[str, Any]:
        """Return compact plan-level metrics; unknown coverage stays explicit."""

        empty = {
            "plans": 0,
            "complete_plans": 0,
            "tick_touched": 0,
            "scan_touched": 0,
            "policy_touched": 0,
            "missed_by_cadence": 0,
            "missed_by_policy": 0,
            "limit_recoverable": 0,
            "unknown_coverage": 0,
            "unknown_clock_domain": 0,
            "miss_rate": None,
        }
        if not self.enabled:
            return empty
        try:
            where = "WHERE finalized = 1" if finalized_only else ""
            with self._db_lock:
                assert self._conn is not None
                row = self._conn.execute(
                    f"""
                    SELECT
                        COUNT(*) AS plans,
                        SUM(CASE WHEN data_complete = 1 THEN 1 ELSE 0 END) AS complete_plans,
                        SUM(CASE WHEN first_strict_touch_msc IS NOT NULL THEN 1 ELSE 0 END) AS tick_touched,
                        SUM(CASE WHEN first_scan_touch_msc IS NOT NULL THEN 1 ELSE 0 END) AS scan_touched,
                        SUM(CASE WHEN first_policy_touch_msc IS NOT NULL THEN 1 ELSE 0 END) AS policy_touched,
                        SUM(CASE WHEN missed_by_cadence = 1 THEN 1 ELSE 0 END) AS missed_by_cadence,
                        SUM(CASE WHEN missed_by_policy = 1 THEN 1 ELSE 0 END) AS missed_by_policy,
                        SUM(CASE WHEN recoverable_by_limit = 1 THEN 1 ELSE 0 END) AS limit_recoverable,
                        SUM(CASE WHEN data_complete = 0 THEN 1 ELSE 0 END) AS unknown_coverage,
                        SUM(
                            CASE WHEN result_class = 'UNKNOWN_CLOCK_DOMAIN'
                            THEN 1 ELSE 0 END
                        ) AS unknown_clock_domain
                    FROM plans {where}
                    """
                ).fetchone()
            result = {key: int(row[key] or 0) for key in empty if key != "miss_rate"}
            touched_complete = 0
            with self._db_lock:
                assert self._conn is not None
                touched_complete = int(
                    self._conn.execute(
                        f"SELECT COUNT(*) FROM plans {where} "
                        + ("AND " if where else "WHERE ")
                        + "data_complete = 1 AND first_strict_touch_msc IS NOT NULL"
                    ).fetchone()[0]
                )
            result["miss_rate"] = (
                result["missed_by_cadence"] / touched_complete
                if touched_complete
                else None
            )
            return result
        except Exception as exc:
            self._log_failure("summary", exc)
            return empty


__all__ = [
    "MT5ReadOnlyFacade",
    "QuoteTick",
    "ShadowPlanSpec",
    "ShadowTickWatcher",
    "TouchClassification",
    "classify_quote",
    "stable_plan_id",
]
