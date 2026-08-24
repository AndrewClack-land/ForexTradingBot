"""Durable one-shot event ledger for exact H1 rejection-block entries.

The ledger deliberately has no pruning API: an exact first-touch event remains
consumed indefinitely.  Missing state is the only condition that produces an
empty ledger; an existing malformed snapshot fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Optional


RB_CONSUMED_EVENTS_SCHEMA_VERSION = "rb-consumed-events-v1"
_EVENT_FIELDS = {
    "event_id",
    "symbol",
    "trigger_signature",
    "consumed_at_utc",
    "reason",
}


class RBConsumedEventsValidationError(ValueError):
    """Raised when consumed-event state cannot be trusted."""


def _strict_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise RBConsumedEventsValidationError(f"{name} must be a string")
    if not value or value != value.strip():
        raise RBConsumedEventsValidationError(
            f"{name} must be non-empty and have no surrounding whitespace"
        )
    return value


def _canonical_symbol(value: Any) -> str:
    symbol = _strict_text(value, name="symbol")
    if symbol != symbol.upper():
        raise RBConsumedEventsValidationError(
            "symbol must use canonical uppercase form"
        )
    return symbol


def _normalize_utc(value: Any) -> datetime:
    if not isinstance(value, datetime):
        raise RBConsumedEventsValidationError(
            "consumed_at_utc must be a timezone-aware datetime"
        )
    if value.tzinfo is None or value.utcoffset() is None:
        raise RBConsumedEventsValidationError(
            "consumed_at_utc must be timezone-aware"
        )
    try:
        timestamp = float(value.timestamp())
    except (OSError, OverflowError, ValueError) as exc:
        raise RBConsumedEventsValidationError(
            "consumed_at_utc is outside the supported timestamp range"
        ) from exc
    if not math.isfinite(timestamp):
        raise RBConsumedEventsValidationError(
            "consumed_at_utc timestamp must be finite"
        )
    return value.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    normalized = _normalize_utc(value)
    return normalized.isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _parse_utc(value: Any) -> datetime:
    raw = _strict_text(value, name="consumed_at_utc")
    if not raw.endswith("Z"):
        raise RBConsumedEventsValidationError(
            "consumed_at_utc must use canonical UTC 'Z' notation"
        )
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00")
    except ValueError as exc:
        raise RBConsumedEventsValidationError(
            "consumed_at_utc is not a valid ISO-8601 timestamp"
        ) from exc
    normalized = _normalize_utc(parsed)
    if _format_utc(normalized) != raw:
        raise RBConsumedEventsValidationError(
            "consumed_at_utc must use canonical microsecond precision"
        )
    return normalized


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RBConsumedEventsValidationError(
                f"duplicate JSON object key {key!r}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise RBConsumedEventsValidationError(
        f"non-finite JSON constant {value!r} is forbidden"
    )


@dataclass(frozen=True)
class RBConsumedEvent:
    event_id: str
    symbol: str
    trigger_signature: str
    consumed_at_utc: datetime
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "event_id",
            _strict_text(self.event_id, name="event_id"),
        )
        object.__setattr__(self, "symbol", _canonical_symbol(self.symbol))
        object.__setattr__(
            self,
            "trigger_signature",
            _strict_text(
                self.trigger_signature,
                name="trigger_signature",
            ),
        )
        object.__setattr__(
            self,
            "consumed_at_utc",
            _normalize_utc(self.consumed_at_utc),
        )
        object.__setattr__(
            self,
            "reason",
            _strict_text(self.reason, name="reason"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "symbol": self.symbol,
            "trigger_signature": self.trigger_signature,
            "consumed_at_utc": _format_utc(self.consumed_at_utc),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "RBConsumedEvent":
        if not isinstance(raw, dict) or set(raw) != _EVENT_FIELDS:
            raise RBConsumedEventsValidationError(
                "consumed RB event object is malformed"
            )
        return cls(
            event_id=raw["event_id"],
            symbol=raw["symbol"],
            trigger_signature=raw["trigger_signature"],
            consumed_at_utc=_parse_utc(raw["consumed_at_utc"]),
            reason=raw["reason"],
        )


@dataclass(frozen=True)
class RBConsumedEventStore:
    """Immutable snapshot of every consumed exact-RB event."""

    events: Mapping[str, RBConsumedEvent] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.events, Mapping):
            raise RBConsumedEventsValidationError(
                "consumed RB events must be a mapping"
            )
        copied: dict[str, RBConsumedEvent] = {}
        for key, event in self.events.items():
            event_id = _strict_text(key, name="event map key")
            if not isinstance(event, RBConsumedEvent):
                raise RBConsumedEventsValidationError(
                    f"event {event_id!r} is not RBConsumedEvent"
                )
            if event_id != event.event_id:
                raise RBConsumedEventsValidationError(
                    f"event map key {event_id!r} does not match "
                    f"event_id {event.event_id!r}"
                )
            if event_id in copied:
                raise RBConsumedEventsValidationError(
                    f"duplicate consumed event_id {event_id!r}"
                )
            copied[event_id] = event
        object.__setattr__(self, "events", MappingProxyType(copied))

    def __len__(self) -> int:
        return len(self.events)

    def is_consumed(self, event_id: str) -> bool:
        normalized = _strict_text(event_id, name="event_id")
        return normalized in self.events

    def consume(
        self,
        *,
        event_id: str,
        symbol: str,
        trigger_signature: str,
        reason: str,
        consumed_at_utc: Optional[datetime] = None,
    ) -> "RBConsumedEventStore":
        event = RBConsumedEvent(
            event_id=event_id,
            symbol=symbol,
            trigger_signature=trigger_signature,
            consumed_at_utc=(
                datetime.now(timezone.utc)
                if consumed_at_utc is None
                else consumed_at_utc
            ),
            reason=reason,
        )
        existing = self.events.get(event.event_id)
        if existing is not None:
            if (
                existing.symbol != event.symbol
                or existing.trigger_signature != event.trigger_signature
            ):
                raise RBConsumedEventsValidationError(
                    f"consumed event_id {event.event_id!r} already belongs "
                    "to different symbol or trigger_signature"
                )
            # First consumption is authoritative. Repeated delivery is
            # idempotent and cannot rewrite its time or reason.
            return self
        updated = dict(self.events)
        updated[event.event_id] = event
        return RBConsumedEventStore(updated)

    def save(self, path: Path) -> None:
        save(self, path)

    @classmethod
    def load(cls, path: Path) -> "RBConsumedEventStore":
        return load(path)


def load(path: Path) -> RBConsumedEventStore:
    """Load a validated ledger; only a missing file means no events."""

    source = Path(path)
    try:
        encoded = source.read_text(encoding="utf-8")
    except FileNotFoundError:
        return RBConsumedEventStore()
    except (OSError, UnicodeError) as exc:
        raise RBConsumedEventsValidationError(
            f"cannot read consumed RB event state {source}: {exc}"
        ) from exc
    try:
        raw = json.loads(
            encoded,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except RBConsumedEventsValidationError:
        raise
    except json.JSONDecodeError as exc:
        raise RBConsumedEventsValidationError(
            f"cannot read consumed RB event state {source}: {exc}"
        ) from exc
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "events",
    }:
        raise RBConsumedEventsValidationError(
            "consumed RB event store envelope is malformed"
        )
    if raw["schema_version"] != RB_CONSUMED_EVENTS_SCHEMA_VERSION:
        raise RBConsumedEventsValidationError(
            "unsupported consumed RB event schema_version "
            f"{raw['schema_version']!r}"
        )
    raw_events = raw["events"]
    if not isinstance(raw_events, dict):
        raise RBConsumedEventsValidationError(
            "consumed RB events must be an object"
        )
    restored: dict[str, RBConsumedEvent] = {}
    for key, value in raw_events.items():
        event_id = _strict_text(key, name="event map key")
        event = RBConsumedEvent.from_dict(value)
        if event_id != event.event_id:
            raise RBConsumedEventsValidationError(
                f"event map key {event_id!r} does not match "
                f"event_id {event.event_id!r}"
            )
        restored[event_id] = event
    return RBConsumedEventStore(restored)


def save(store: RBConsumedEventStore, path: Path) -> None:
    """Atomically persist one validated immutable snapshot."""

    if not isinstance(store, RBConsumedEventStore):
        raise RBConsumedEventsValidationError(
            "store must be RBConsumedEventStore"
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": RB_CONSUMED_EVENTS_SCHEMA_VERSION,
        "events": {
            event_id: event.to_dict()
            for event_id, event in sorted(store.events.items())
        },
    }
    encoded = json.dumps(
        envelope,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def consume(
    store: RBConsumedEventStore,
    *,
    event_id: str,
    symbol: str,
    trigger_signature: str,
    reason: str,
    consumed_at_utc: Optional[datetime] = None,
) -> RBConsumedEventStore:
    """Return a snapshot containing the event without mutating ``store``."""

    if not isinstance(store, RBConsumedEventStore):
        raise RBConsumedEventsValidationError(
            "store must be RBConsumedEventStore"
        )
    return store.consume(
        event_id=event_id,
        symbol=symbol,
        trigger_signature=trigger_signature,
        reason=reason,
        consumed_at_utc=consumed_at_utc,
    )


def is_consumed(store: RBConsumedEventStore, event_id: str) -> bool:
    """Return whether ``event_id`` is permanently present in the snapshot."""

    if not isinstance(store, RBConsumedEventStore):
        raise RBConsumedEventsValidationError(
            "store must be RBConsumedEventStore"
        )
    return store.is_consumed(event_id)


__all__ = [
    "RB_CONSUMED_EVENTS_SCHEMA_VERSION",
    "RBConsumedEvent",
    "RBConsumedEventStore",
    "RBConsumedEventsValidationError",
    "consume",
    "is_consumed",
    "load",
    "save",
]
