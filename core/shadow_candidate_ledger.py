"""Fail-open telemetry for simultaneous entry candidates.

The production strategy intentionally returns one trigger from a fixed
waterfall. This module records the full, shadow-only candidate population so
that another arbitration policy can be evaluated without changing an entry
decision. A scan is a scheduler observation; an opportunity is the same
executable trigger signature within one UTC trading day. Keeping those units
separate prevents minute polling from inflating the number of ideas.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Optional


SCHEMA_VERSION = 1


def _digest(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _utc_iso(value: datetime) -> str:
    return _aware_utc(value).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _decision_day(
    decision_bar_close: Optional[datetime],
    observed_at_utc: datetime,
) -> date:
    source = decision_bar_close or observed_at_utc
    return _aware_utc(source).date()


def stable_candidate_event_id(symbol: str, trigger_signature: str) -> str:
    """Identity of the structural event, independent of polling/deployment."""

    return _digest("candidate-event-v1", symbol.upper(), trigger_signature)


def stable_candidate_opportunity_id(
    symbol: str,
    trigger_signature: str,
    trading_day: date,
) -> str:
    """Identity of one independently eligible idea under the daily guard."""

    return _digest(
        "candidate-opportunity-v1",
        symbol.upper(),
        trigger_signature,
        trading_day.isoformat(),
    )


def stable_candidate_scan_id(
    deployment_id: str,
    symbol: str,
    observed_at_utc: datetime,
) -> str:
    """Idempotent identity for one symbol/minute scheduler observation."""

    observed = _aware_utc(observed_at_utc)
    minute = observed.replace(second=0, microsecond=0)
    return _digest(
        "candidate-scan-v1",
        deployment_id,
        symbol.upper(),
        _utc_iso(minute),
    )


def trigger_signature(signal: Mapping[str, Any]) -> str:
    """Match the live/backtest trigger identity without importing Core."""

    trigger_kind = str(signal.get("trigger_kind") or "").strip().lower()
    if not trigger_kind:
        trigger_kind = (
            str(signal.get("trigger_reason") or "")
            .split("|")[0]
            .strip()
            .split(" ")[0]
            .lower()
        )
    event_id = str(signal.get("trigger_event_id") or "").strip()
    zone_low = signal.get("zone_low")
    zone_high = signal.get("zone_high")
    if event_id:
        anchor = f"e{event_id}"
    elif zone_low is not None and zone_high is not None:
        anchor = f"z{float(zone_low):.5f}-{float(zone_high):.5f}"
    else:
        anchor = f"s{float(signal.get('stop_price') or 0.0):.5f}"
    return f"{signal.get('side')}|{trigger_kind}|{anchor}"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime):
        return _utc_iso(value) if value.tzinfo is not None else value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return str(isoformat())
        except Exception:
            pass
    return str(value)


def _json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


class ShadowCandidateLedger:
    """Separate SQLite ledger that can never return an execution decision."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=5.0,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self.enabled = True
        self.disabled_reason: Optional[str] = None
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            row = self._conn.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
            if row is not None and int(row[0]) != SCHEMA_VERSION:
                raise RuntimeError(
                    f"unsupported candidate-ledger schema {row[0]}"
                )
            self._conn.execute(
                "INSERT OR IGNORE INTO metadata(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scans (
                    scan_id TEXT PRIMARY KEY,
                    deployment_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    observed_minute_utc TEXT NOT NULL,
                    observed_at_utc TEXT NOT NULL,
                    decision_bar_close_utc TEXT,
                    production_signal TEXT,
                    production_trigger_signature TEXT,
                    production_opportunity_id TEXT,
                    candidate_count INTEGER NOT NULL,
                    detector_errors_json TEXT NOT NULL,
                    downstream_disposition TEXT,
                    downstream_details_json TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS candidates (
                    observation_id TEXT PRIMARY KEY,
                    scan_id TEXT NOT NULL REFERENCES scans(scan_id)
                        ON DELETE CASCADE,
                    event_id TEXT NOT NULL,
                    opportunity_id TEXT NOT NULL,
                    trading_day_utc TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT,
                    trigger_kind TEXT,
                    trigger_reason TEXT,
                    trigger_signature TEXT NOT NULL,
                    candidate_rank INTEGER,
                    production_priority INTEGER,
                    candidate_source TEXT,
                    selected_by_production INTEGER NOT NULL,
                    entry_price REAL,
                    entry_min REAL,
                    entry_max REAL,
                    stop_price REAL,
                    tp1 REAL,
                    tp2 REAL,
                    tp3 REAL,
                    fvg_regime TEXT,
                    fvg_age_bars REAL,
                    factor_vector_json TEXT,
                    trigger_meta_json TEXT,
                    payload_json TEXT NOT NULL,
                    model_id TEXT,
                    expected_gross_r REAL,
                    estimated_cost_r REAL,
                    expected_net_r REAL,
                    correlation_penalty_r REAL,
                    ranking_score REAL,
                    shadow_rank_position INTEGER,
                    shadow_selected INTEGER,
                    created_at REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_candidates_opportunity "
                "ON candidates(opportunity_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_candidates_event "
                "ON candidates(event_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scans_symbol_time "
                "ON scans(symbol, observed_at_utc)"
            )

    def _disable(self, exc: BaseException) -> None:
        self.enabled = False
        self.disabled_reason = f"{type(exc).__name__}: {exc}"

    def record_scan(
        self,
        *,
        deployment_id: str,
        symbol: str,
        observed_at_utc: datetime,
        decision_bar_close: Optional[datetime],
        production_signal: Optional[Mapping[str, Any]],
        candidates: Sequence[Mapping[str, Any]],
        detector_errors: Sequence[Mapping[str, Any]] = (),
        downstream_disposition: Optional[str] = None,
        downstream_details: Optional[Mapping[str, Any]] = None,
    ) -> Optional[str]:
        """Atomically persist the first causal observation in a UTC minute.

        A retry in the same minute returns the same scan id and does not mutate
        the original candidate set. This prevents later information from
        rewriting the past.
        """

        if not self.enabled:
            return None
        try:
            observed = _aware_utc(observed_at_utc)
            decision = (
                _aware_utc(decision_bar_close)
                if decision_bar_close is not None
                else None
            )
            symbol_key = str(symbol).strip().upper()
            if not symbol_key:
                raise ValueError("symbol is required")
            deployment = str(deployment_id).strip() or "unversioned"
            scan_id = stable_candidate_scan_id(
                deployment, symbol_key, observed
            )
            trading_day = _decision_day(decision, observed)
            raw_production = dict(production_signal or {})
            production_kind = str(raw_production.get("signal") or "")
            production_signature = (
                trigger_signature(raw_production)
                if production_kind == "ENTER"
                else None
            )
            production_opportunity = (
                stable_candidate_opportunity_id(
                    symbol_key,
                    production_signature,
                    trading_day,
                )
                if production_signature
                else None
            )
            observed_minute = observed.replace(second=0, microsecond=0)
            created_at = time.time()

            prepared: list[tuple[Any, ...]] = []
            for ordinal, source_candidate in enumerate(candidates, start=1):
                candidate = dict(source_candidate)
                signature = str(
                    candidate.get("shadow_trigger_signature") or ""
                ).strip() or trigger_signature(candidate)
                event_id = stable_candidate_event_id(
                    symbol_key, signature
                )
                opportunity_id = stable_candidate_opportunity_id(
                    symbol_key, signature, trading_day
                )
                source = str(
                    candidate.get("shadow_candidate_source") or ""
                )
                observation_id = _digest(
                    "candidate-observation-v1",
                    scan_id,
                    opportunity_id,
                    source,
                    candidate.get("production_priority") or ordinal,
                )
                tp_prices = list(candidate.get("tp_prices") or [])
                tp_values = [
                    _optional_float(tp_prices[index])
                    if index < len(tp_prices)
                    else None
                    for index in range(3)
                ]
                prepared.append(
                    (
                        observation_id,
                        scan_id,
                        event_id,
                        opportunity_id,
                        trading_day.isoformat(),
                        symbol_key,
                        candidate.get("side"),
                        candidate.get("trigger_kind"),
                        candidate.get("trigger_reason"),
                        signature,
                        int(candidate.get("shadow_candidate_rank") or ordinal),
                        int(candidate.get("production_priority") or ordinal),
                        source or None,
                        int(signature == production_signature),
                        _optional_float(candidate.get("entry_price")),
                        _optional_float(candidate.get("entry_min")),
                        _optional_float(candidate.get("entry_max")),
                        _optional_float(candidate.get("stop_price")),
                        tp_values[0],
                        tp_values[1],
                        tp_values[2],
                        candidate.get("fvg_regime"),
                        _optional_float(candidate.get("fvg_age_bars")),
                        _json(candidate.get("factor_vector")),
                        _json(candidate.get("trigger_meta")),
                        _json(candidate),
                        created_at,
                    )
                )

            with self._lock, self._conn:
                inserted = self._conn.execute(
                    """
                    INSERT OR IGNORE INTO scans(
                        scan_id,deployment_id,symbol,observed_minute_utc,
                        observed_at_utc,decision_bar_close_utc,
                        production_signal,production_trigger_signature,
                        production_opportunity_id,candidate_count,
                        detector_errors_json,downstream_disposition,
                        downstream_details_json,created_at,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        scan_id,
                        deployment,
                        symbol_key,
                        _utc_iso(observed_minute),
                        _utc_iso(observed),
                        _utc_iso(decision) if decision is not None else None,
                        production_kind or None,
                        production_signature,
                        production_opportunity,
                        len(prepared),
                        _json(detector_errors),
                        downstream_disposition,
                        _json(downstream_details)
                        if downstream_details is not None
                        else None,
                        created_at,
                        created_at,
                    ),
                ).rowcount
                if inserted:
                    self._conn.executemany(
                        """
                        INSERT INTO candidates(
                            observation_id,scan_id,event_id,opportunity_id,
                            trading_day_utc,symbol,side,trigger_kind,
                            trigger_reason,trigger_signature,candidate_rank,
                            production_priority,candidate_source,
                            selected_by_production,entry_price,entry_min,
                            entry_max,stop_price,tp1,tp2,tp3,fvg_regime,
                            fvg_age_bars,factor_vector_json,trigger_meta_json,
                            payload_json,created_at
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        prepared,
                    )
            return scan_id
        except Exception as exc:
            self._disable(exc)
            return None

    def record_outcome(
        self,
        scan_id: str,
        disposition: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        if not self.enabled:
            return False
        try:
            with self._lock, self._conn:
                changed = self._conn.execute(
                    """
                    UPDATE scans
                    SET downstream_disposition=?, downstream_details_json=?,
                        updated_at=?
                    WHERE scan_id=?
                    """,
                    (
                        str(disposition),
                        _json(details) if details is not None else None,
                        time.time(),
                        str(scan_id),
                    ),
                ).rowcount
            return bool(changed)
        except Exception as exc:
            self._disable(exc)
            return False

    def record_ranking(
        self,
        scan_id: str,
        rankings: Sequence[Mapping[str, Any]],
        *,
        model_id: str,
    ) -> bool:
        """Attach shadow scores; no return value can select a live signal."""

        if not self.enabled:
            return False
        try:
            updates = []
            for item in rankings:
                updates.append(
                    (
                        str(model_id),
                        _optional_float(item.get("expected_gross_r")),
                        _optional_float(item.get("estimated_cost_r")),
                        _optional_float(item.get("expected_net_r")),
                        _optional_float(item.get("correlation_penalty_r")),
                        _optional_float(item.get("ranking_score")),
                        int(item["rank_position"])
                        if item.get("rank_position") is not None
                        else None,
                        int(bool(item.get("selected"))),
                        str(scan_id),
                        str(item.get("opportunity_id") or ""),
                    )
                )
            with self._lock, self._conn:
                self._conn.executemany(
                    """
                    UPDATE candidates SET
                        model_id=?, expected_gross_r=?, estimated_cost_r=?,
                        expected_net_r=?, correlation_penalty_r=?,
                        ranking_score=?, shadow_rank_position=?,
                        shadow_selected=?
                    WHERE scan_id=? AND opportunity_id=?
                    """,
                    updates,
                )
            return True
        except Exception as exc:
            self._disable(exc)
            return False

    def summary(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "disabled_reason": self.disabled_reason,
            }
        with self._lock:
            scans = self._conn.execute(
                "SELECT COUNT(*) FROM scans"
            ).fetchone()[0]
            observations = self._conn.execute(
                "SELECT COUNT(*) FROM candidates"
            ).fetchone()[0]
            opportunities = self._conn.execute(
                "SELECT COUNT(DISTINCT opportunity_id) FROM candidates"
            ).fetchone()[0]
            events = self._conn.execute(
                "SELECT COUNT(DISTINCT event_id) FROM candidates"
            ).fetchone()[0]
            simultaneous = self._conn.execute(
                "SELECT COUNT(*) FROM scans WHERE candidate_count > 1"
            ).fetchone()[0]
            selected = self._conn.execute(
                "SELECT COUNT(*) FROM candidates "
                "WHERE selected_by_production=1"
            ).fetchone()[0]
        return {
            "enabled": True,
            "scans": int(scans),
            "candidate_observations": int(observations),
            "independent_opportunities": int(opportunities),
            "structural_events": int(events),
            "simultaneous_candidate_scans": int(simultaneous),
            "production_selected_observations": int(selected),
        }

    def close(self) -> None:
        with self._lock:
            if self.enabled:
                self._conn.commit()
            self._conn.close()
            self.enabled = False


__all__ = [
    "SCHEMA_VERSION",
    "ShadowCandidateLedger",
    "stable_candidate_event_id",
    "stable_candidate_opportunity_id",
    "stable_candidate_scan_id",
    "trigger_signature",
]
