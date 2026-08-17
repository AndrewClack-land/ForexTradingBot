"""Append-only diagnostic event ledger for quality-score research.

The ledger deliberately has no API that can return an execution decision.  A
write failure disables further writes and returns ``None`` so diagnostics can
never block or alter the production trading path.

Use a dedicated SQLite file for this writer. SQLite permits only one writer
per database, so sharing its file with another fail-open diagnostic ledger can
couple their availability through SQLITE_BUSY timeouts.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional


SCHEMA_VERSION = 1
EVENT_TYPES = frozenset(
    {
        "CANDIDATE_DECISION",
        "QUALITY_SCORE",
        "EXECUTION",
        "BROKER_OUTCOME",
        "OUTCOME_CORRECTION",
    }
)

_ROW_COLUMNS = (
    "event_id",
    "schema_version",
    "event_type",
    "occurred_at_utc",
    "known_at_utc",
    "deployment_id",
    "strategy_version",
    "scan_id",
    "observation_id",
    "opportunity_id",
    "setup_id",
    "idea_id",
    "symbol",
    "side",
    "trigger_kind",
    "payload_json",
    "payload_sha256",
    "supersedes_event_id",
)

_CORRECTION_IDENTITY_COLUMNS = (
    "deployment_id",
    "strategy_version",
    "symbol",
    "side",
    "trigger_kind",
    "opportunity_id",
    "setup_id",
    "idea_id",
)

_INDEX_DEFINITIONS = {
    "idx_quality_events_setup": (
        "CREATE INDEX IF NOT EXISTS idx_quality_events_setup "
        "ON quality_events(setup_id, occurred_at_utc)"
    ),
    "idx_quality_events_opportunity": (
        "CREATE INDEX IF NOT EXISTS idx_quality_events_opportunity "
        "ON quality_events(opportunity_id, occurred_at_utc)"
    ),
    "idx_quality_events_type_time": (
        "CREATE INDEX IF NOT EXISTS idx_quality_events_type_time "
        "ON quality_events(event_type, occurred_at_utc)"
    ),
    "uq_quality_events_supersedes": (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_quality_events_supersedes "
        "ON quality_events(supersedes_event_id) "
        "WHERE supersedes_event_id IS NOT NULL"
    ),
}

_TRIGGER_DEFINITIONS = {
    "quality_events_no_update": """
        CREATE TRIGGER IF NOT EXISTS quality_events_no_update
        BEFORE UPDATE ON quality_events
        BEGIN
            SELECT RAISE(ABORT, 'quality_events is append-only');
        END
    """,
    "quality_events_no_delete": """
        CREATE TRIGGER IF NOT EXISTS quality_events_no_delete
        BEFORE DELETE ON quality_events
        BEGIN
            SELECT RAISE(ABORT, 'quality_events is append-only');
        END
    """,
    "quality_events_no_replace": """
        CREATE TRIGGER IF NOT EXISTS quality_events_no_replace
        BEFORE INSERT ON quality_events
        WHEN EXISTS (
            SELECT 1 FROM quality_events
            WHERE event_id = NEW.event_id
        )
        BEGIN
            SELECT RAISE(
                ABORT,
                'quality_events event_id is append-only'
            );
        END
    """,
}


def _aware_utc(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _utc_iso(value: datetime, *, name: str) -> str:
    return (
        _aware_utc(value, name=name)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("payload contains a non-finite float")
        return 0.0 if value == 0.0 else value
    if isinstance(value, datetime):
        return _utc_iso(value, name="payload datetime")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("payload object keys must be strings")
            result[key] = _canonical_value(item)
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_canonical_value(item) for item in value]
    scalar = getattr(value, "item", None)
    if callable(scalar):
        return _canonical_value(scalar())
    raise TypeError(f"payload value is not JSON-safe: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the one canonical UTF-8 JSON representation used for hashing."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_text(value: Any, *, name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{name} is required")
    return text


def _normalized_schema_sql(value: str) -> str:
    normalized = " ".join(str(value).split()).strip().lower()
    return normalized.replace(" if not exists ", " ")


def _verify_schema_objects(
    connection: sqlite3.Connection,
    *,
    object_type: str,
    definitions: Mapping[str, str],
) -> None:
    for name, expected_sql in definitions.items():
        row = connection.execute(
            "SELECT tbl_name,sql FROM sqlite_master "
            "WHERE type=? AND name=?",
            (object_type, name),
        ).fetchone()
        if (
            row is None
            or str(row["tbl_name"]) != "quality_events"
            or row["sql"] is None
            or _normalized_schema_sql(str(row["sql"]))
            != _normalized_schema_sql(expected_sql)
        ):
            raise RuntimeError(
                f"unexpected {object_type} definition for {name}"
            )


class QualityEventLedger:
    """Fail-open SQLite sink with immutable, idempotent diagnostic events."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        connection_factory: Callable[..., sqlite3.Connection] = sqlite3.connect,
    ) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self.enabled = False
        self.disabled_reason: Optional[str] = None
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = connection_factory(
                str(self.db_path),
                timeout=5.0,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._initialize_schema()
            self.enabled = True
        except Exception as exc:
            self._disable(exc)

    def _disable(self, exc: BaseException) -> None:
        self.enabled = False
        self.disabled_reason = f"{type(exc).__name__}: {exc}"

    def _initialize_schema(self) -> None:
        if self._conn is None:
            raise RuntimeError("database connection is unavailable")
        allowed = ",".join(f"'{value}'" for value in sorted(EVENT_TYPES))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA recursive_triggers=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        recursive_triggers = self._conn.execute(
            "PRAGMA recursive_triggers"
        ).fetchone()
        if recursive_triggers is None or int(recursive_triggers[0]) != 1:
            raise RuntimeError("recursive SQLite triggers are required")
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS quality_events (
                    event_id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    event_type TEXT NOT NULL CHECK(event_type IN ({allowed})),
                    occurred_at_utc TEXT NOT NULL,
                    known_at_utc TEXT NOT NULL,
                    deployment_id TEXT NOT NULL,
                    strategy_version TEXT NOT NULL,
                    scan_id TEXT,
                    observation_id TEXT,
                    opportunity_id TEXT,
                    setup_id TEXT,
                    idea_id TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT,
                    trigger_kind TEXT,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    supersedes_event_id TEXT REFERENCES quality_events(event_id),
                    CHECK(length(payload_sha256) = 64),
                    CHECK(
                        (event_type = 'OUTCOME_CORRECTION'
                         AND supersedes_event_id IS NOT NULL)
                        OR
                        (event_type != 'OUTCOME_CORRECTION'
                         AND supersedes_event_id IS NULL)
                    )
                )
                """
            )
            columns = tuple(
                str(row[1])
                for row in self._conn.execute(
                    "PRAGMA table_info(quality_events)"
                ).fetchall()
            )
            if columns != _ROW_COLUMNS:
                raise RuntimeError(
                    "unsupported quality-event schema: "
                    f"expected {_ROW_COLUMNS!r}, got {columns!r}"
                )
            for statement in _INDEX_DEFINITIONS.values():
                self._conn.execute(statement)
            for statement in _TRIGGER_DEFINITIONS.values():
                self._conn.execute(statement)
            _verify_schema_objects(
                self._conn,
                object_type="index",
                definitions=_INDEX_DEFINITIONS,
            )
            _verify_schema_objects(
                self._conn,
                object_type="trigger",
                definitions=_TRIGGER_DEFINITIONS,
            )

    @staticmethod
    def _stable_event_id(values_without_id: Mapping[str, Any]) -> str:
        identity = canonical_json(
            {
                "domain": "quality-event-v1",
                **dict(values_without_id),
            }
        )
        return _sha256_text(identity)

    def record_event(
        self,
        *,
        event_type: str,
        occurred_at_utc: datetime,
        known_at_utc: datetime,
        deployment_id: str,
        strategy_version: str,
        symbol: str,
        payload: Mapping[str, Any],
        scan_id: Optional[str] = None,
        observation_id: Optional[str] = None,
        opportunity_id: Optional[str] = None,
        setup_id: Optional[str] = None,
        idea_id: Optional[str] = None,
        side: Optional[str] = None,
        trigger_kind: Optional[str] = None,
        supersedes_event_id: Optional[str] = None,
        event_id: Optional[str] = None,
    ) -> Optional[str]:
        """Append one event, returning its identity or ``None`` on any failure.

        No exception escapes this method.  An identical retry is a no-op.  A
        caller-supplied identity that already names different bytes disables
        the diagnostic sink and leaves the original row untouched.
        """

        if not self.enabled or self._conn is None:
            return None
        try:
            normalized_type = _required_text(
                event_type, name="event_type"
            ).upper()
            if normalized_type not in EVENT_TYPES:
                raise ValueError(f"unsupported event_type {normalized_type!r}")
            occurred = _aware_utc(occurred_at_utc, name="occurred_at_utc")
            known = _aware_utc(known_at_utc, name="known_at_utc")
            if known < occurred:
                raise ValueError("known_at_utc cannot precede occurred_at_utc")
            occurred_iso = _utc_iso(occurred, name="occurred_at_utc")
            known_iso = _utc_iso(known, name="known_at_utc")
            deployment = _required_text(deployment_id, name="deployment_id")
            strategy = _required_text(
                strategy_version, name="strategy_version"
            )
            symbol_key = _required_text(symbol, name="symbol").upper()
            side_key = _optional_text(side)
            if side_key is not None:
                side_key = side_key.upper()
                if side_key not in {"LONG", "SHORT"}:
                    raise ValueError("side must be LONG or SHORT")
            if not isinstance(payload, Mapping):
                raise TypeError("payload must be a mapping")
            payload_json = canonical_json(payload)
            payload_sha256 = _sha256_text(payload_json)
            supersedes = _optional_text(supersedes_event_id)
            if normalized_type == "OUTCOME_CORRECTION":
                if supersedes is None:
                    raise ValueError(
                        "OUTCOME_CORRECTION requires supersedes_event_id"
                    )
            elif supersedes is not None:
                raise ValueError(
                    "supersedes_event_id is only valid for OUTCOME_CORRECTION"
                )

            values_without_id = {
                "schema_version": SCHEMA_VERSION,
                "event_type": normalized_type,
                "occurred_at_utc": occurred_iso,
                "known_at_utc": known_iso,
                "deployment_id": deployment,
                "strategy_version": strategy,
                "scan_id": _optional_text(scan_id),
                "observation_id": _optional_text(observation_id),
                "opportunity_id": _optional_text(opportunity_id),
                "setup_id": _optional_text(setup_id),
                "idea_id": _optional_text(idea_id),
                "symbol": symbol_key,
                "side": side_key,
                "trigger_kind": _optional_text(trigger_kind),
                "payload_sha256": payload_sha256,
                "supersedes_event_id": supersedes,
            }
            normalized_id = _optional_text(event_id)
            if normalized_id is None:
                normalized_id = self._stable_event_id(values_without_id)
            if len(normalized_id) > 256:
                raise ValueError("event_id exceeds 256 characters")
            if supersedes == normalized_id:
                raise ValueError("an event cannot supersede itself")

            row_values = (
                normalized_id,
                SCHEMA_VERSION,
                normalized_type,
                occurred_iso,
                known_iso,
                deployment,
                strategy,
                values_without_id["scan_id"],
                values_without_id["observation_id"],
                values_without_id["opportunity_id"],
                values_without_id["setup_id"],
                values_without_id["idea_id"],
                symbol_key,
                side_key,
                values_without_id["trigger_kind"],
                payload_json,
                payload_sha256,
                supersedes,
            )

            with self._lock, self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                existing = self._conn.execute(
                    "SELECT * FROM quality_events WHERE event_id = ?",
                    (normalized_id,),
                ).fetchone()
                if existing is not None:
                    existing_values = tuple(existing[column] for column in _ROW_COLUMNS)
                    if existing_values == row_values:
                        return normalized_id
                    raise ValueError(
                        "event_id already exists with different canonical bytes"
                    )
                if supersedes is not None:
                    prior = self._conn.execute(
                        "SELECT * FROM quality_events WHERE event_id = ?",
                        (supersedes,),
                    ).fetchone()
                    if prior is None:
                        raise ValueError("superseded event does not exist")
                    if str(prior["event_type"]) not in {
                        "BROKER_OUTCOME",
                        "OUTCOME_CORRECTION",
                    }:
                        raise ValueError(
                            "OUTCOME_CORRECTION must supersede an outcome event"
                        )
                    for column in _CORRECTION_IDENTITY_COLUMNS:
                        parent_value = prior[column]
                        if (
                            parent_value is not None
                            and values_without_id[column] != parent_value
                        ):
                            raise ValueError(
                                "OUTCOME_CORRECTION identity mismatch: "
                                f"{column}"
                            )
                    parent_known = _aware_utc(
                        datetime.fromisoformat(
                            str(prior["known_at_utc"]).replace(
                                "Z",
                                "+00:00",
                            )
                        ),
                        name="superseded known_at_utc",
                    )
                    if known < parent_known:
                        raise ValueError(
                            "OUTCOME_CORRECTION known_at_utc cannot precede "
                            "the superseded event"
                        )
                    existing_child = self._conn.execute(
                        "SELECT event_id FROM quality_events "
                        "WHERE supersedes_event_id = ? LIMIT 1",
                        (supersedes,),
                    ).fetchone()
                    if existing_child is not None:
                        raise ValueError(
                            "superseded event already has a correction"
                        )
                placeholders = ",".join("?" for _ in _ROW_COLUMNS)
                self._conn.execute(
                    f"INSERT INTO quality_events "
                    f"({','.join(_ROW_COLUMNS)}) VALUES ({placeholders})",
                    row_values,
                )
            return normalized_id
        except Exception as exc:
            self._disable(exc)
            return None

    def get_event(self, event_id: str) -> Optional[dict[str, Any]]:
        """Return one stored row without mutating ledger state."""

        if self._conn is None:
            return None
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT * FROM quality_events WHERE event_id = ?",
                    (str(event_id),),
                ).fetchone()
            return dict(row) if row is not None else None
        except Exception as exc:
            self._disable(exc)
            return None

    def query_events(
        self,
        *,
        event_type: Optional[str] = None,
        setup_id: Optional[str] = None,
        opportunity_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Read events through a bounded, parameterized diagnostic query."""

        if self._conn is None:
            return []
        try:
            if isinstance(limit, bool) or not 1 <= int(limit) <= 10_000:
                raise ValueError("limit must be between 1 and 10000")
            clauses: list[str] = []
            parameters: list[Any] = []
            if event_type is not None:
                normalized_type = str(event_type).strip().upper()
                if normalized_type not in EVENT_TYPES:
                    raise ValueError("unsupported event_type")
                clauses.append("event_type = ?")
                parameters.append(normalized_type)
            if setup_id is not None:
                clauses.append("setup_id = ?")
                parameters.append(str(setup_id))
            if opportunity_id is not None:
                clauses.append("opportunity_id = ?")
                parameters.append(str(opportunity_id))
            where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
            parameters.append(int(limit))
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM quality_events"
                    + where
                    + " ORDER BY known_at_utc, event_id LIMIT ?",
                    parameters,
                ).fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:
            self._disable(exc)
            return []

    def summary(self) -> dict[str, Any]:
        """Return read-only event counts and current sink health."""

        base: dict[str, Any] = {
            "enabled": bool(self.enabled),
            "disabled_reason": self.disabled_reason,
        }
        if self._conn is None:
            return {**base, "events": 0, "by_type": {}}
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT event_type, COUNT(*) AS count "
                    "FROM quality_events GROUP BY event_type"
                ).fetchall()
            by_type = {str(row["event_type"]): int(row["count"]) for row in rows}
            return {
                **base,
                "events": int(sum(by_type.values())),
                "by_type": by_type,
            }
        except Exception as exc:
            self._disable(exc)
            return {
                "enabled": False,
                "disabled_reason": self.disabled_reason,
                "events": 0,
                "by_type": {},
            }

    def close(self) -> None:
        with self._lock:
            connection = self._conn
            self._conn = None
            self.enabled = False
            if connection is not None:
                connection.close()


__all__ = [
    "EVENT_TYPES",
    "SCHEMA_VERSION",
    "QualityEventLedger",
    "canonical_json",
]
