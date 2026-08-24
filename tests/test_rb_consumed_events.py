from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

import core.rb_consumed_events as consumed_events
from core.rb_consumed_events import (
    RB_CONSUMED_EVENTS_SCHEMA_VERSION,
    RBConsumedEventStore,
    RBConsumedEventsValidationError,
)


EVENT_ID = "rb:h1:touch:v1:long:2026-08-24T08:00:00+00:00"
SIGNATURE = f"long|rejection_block_1h|e{EVENT_ID}"
CONSUMED_AT = datetime(2026, 8, 24, 10, 15, tzinfo=timezone.utc)


def _consume(
    store: RBConsumedEventStore | None = None,
    **overrides,
) -> RBConsumedEventStore:
    values = {
        "event_id": EVENT_ID,
        "symbol": "EURUSD",
        "trigger_signature": SIGNATURE,
        "reason": "native-limit-armed",
        "consumed_at_utc": CONSUMED_AT,
    }
    values.update(overrides)
    return consumed_events.consume(
        store or RBConsumedEventStore(),
        **values,
    )


def _raw_event(**overrides) -> dict:
    event = {
        "event_id": EVENT_ID,
        "symbol": "EURUSD",
        "trigger_signature": SIGNATURE,
        "consumed_at_utc": "2026-08-24T10:15:00.000000Z",
        "reason": "native-limit-armed",
    }
    event.update(overrides)
    return event


def _write_envelope(path: Path, events) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": RB_CONSUMED_EVENTS_SCHEMA_VERSION,
                "events": events,
            }
        ),
        encoding="utf-8",
    )


def test_missing_file_is_the_only_empty_store_case(tmp_path):
    path = tmp_path / "missing.json"

    store = consumed_events.load(path)

    assert len(store) == 0
    assert not consumed_events.is_consumed(store, EVENT_ID)

    path.write_text("", encoding="utf-8")
    with pytest.raises(RBConsumedEventsValidationError):
        consumed_events.load(path)


def test_existing_unreadable_store_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / "consumed.json"
    _write_envelope(path, {})
    original_read_text = Path.read_text

    def fail_selected_path(self, *args, **kwargs):
        if self == path:
            raise PermissionError("simulated denied read")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_selected_path)

    with pytest.raises(
        RBConsumedEventsValidationError,
        match="simulated denied read",
    ):
        consumed_events.load(path)


def test_consume_round_trip_is_immutable_versioned_and_utc(tmp_path):
    path = tmp_path / "state" / "consumed.json"
    original = RBConsumedEventStore()
    local_time = datetime(
        2026,
        8,
        24,
        12,
        15,
        3,
        25,
        tzinfo=timezone(timedelta(hours=2)),
    )

    updated = _consume(original, consumed_at_utc=local_time)
    consumed_events.save(updated, path)
    restored = RBConsumedEventStore.load(path)

    assert len(original) == 0
    assert updated.is_consumed(EVENT_ID)
    assert restored == updated
    event = restored.events[EVENT_ID]
    assert event.consumed_at_utc == datetime(
        2026,
        8,
        24,
        10,
        15,
        3,
        25,
        tzinfo=timezone.utc,
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == RB_CONSUMED_EVENTS_SCHEMA_VERSION
    assert raw["events"][EVENT_ID]["consumed_at_utc"] == (
        "2026-08-24T10:15:03.000025Z"
    )
    with pytest.raises(TypeError):
        restored.events["another"] = event


def test_duplicate_consume_is_idempotent_and_never_rewrites_first_fact():
    first = _consume()

    duplicate = _consume(
        first,
        reason="later-reason-must-not-win",
        consumed_at_utc=CONSUMED_AT + timedelta(hours=1),
    )

    assert duplicate is first
    assert first.events[EVENT_ID].reason == "native-limit-armed"
    assert first.events[EVENT_ID].consumed_at_utc == CONSUMED_AT


@pytest.mark.parametrize(
    "overrides",
    [
        {"symbol": "GBPUSD"},
        {"trigger_signature": "different-signature"},
    ],
)
def test_duplicate_event_id_with_conflicting_identity_fails_closed(overrides):
    first = _consume()

    with pytest.raises(
        RBConsumedEventsValidationError,
        match="already belongs",
    ):
        _consume(first, **overrides)


@pytest.mark.parametrize(
    "raw",
    [
        [],
        {},
        {"schema_version": "unknown", "events": {}},
        {
            "schema_version": RB_CONSUMED_EVENTS_SCHEMA_VERSION,
            "events": {},
            "unexpected": True,
        },
        {
            "schema_version": RB_CONSUMED_EVENTS_SCHEMA_VERSION,
            "events": [],
        },
    ],
)
def test_malformed_or_unknown_envelope_fails_closed(tmp_path, raw):
    path = tmp_path / "consumed.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(RBConsumedEventsValidationError):
        consumed_events.load(path)


@pytest.mark.parametrize(
    "key,event",
    [
        ("different-event", _raw_event()),
        (EVENT_ID, _raw_event(symbol="eurusd")),
        (EVENT_ID, _raw_event(reason="")),
        (
            EVENT_ID,
            _raw_event(consumed_at_utc="2026-08-24T10:15:00+00:00"),
        ),
        (
            EVENT_ID,
            {**_raw_event(), "unexpected": "field"},
        ),
    ],
)
def test_malformed_event_fails_closed(tmp_path, key, event):
    path = tmp_path / "consumed.json"
    _write_envelope(path, {key: event})

    with pytest.raises(RBConsumedEventsValidationError):
        consumed_events.load(path)


def test_duplicate_json_event_id_is_rejected_instead_of_overwritten(tmp_path):
    path = tmp_path / "consumed.json"
    event = json.dumps(_raw_event(), separators=(",", ":"))
    path.write_text(
        "{"
        f'"schema_version":"{RB_CONSUMED_EVENTS_SCHEMA_VERSION}",'
        f'"events":{{"{EVENT_ID}":{event},"{EVENT_ID}":{event}}}'
        "}",
        encoding="utf-8",
    )

    with pytest.raises(
        RBConsumedEventsValidationError,
        match="duplicate JSON object key",
    ):
        consumed_events.load(path)


def test_nonfinite_and_naive_consumption_times_are_rejected():
    class NonFiniteDateTime(datetime):
        def timestamp(self):
            return float("nan")

    with pytest.raises(
        RBConsumedEventsValidationError,
        match="timezone-aware",
    ):
        _consume(consumed_at_utc=datetime(2026, 8, 24, 10, 15))

    with pytest.raises(
        RBConsumedEventsValidationError,
        match="finite",
    ):
        _consume(
            consumed_at_utc=NonFiniteDateTime(
                2026,
                8,
                24,
                10,
                15,
                tzinfo=timezone.utc,
            )
        )


def test_atomic_replace_failure_preserves_last_valid_snapshot(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "consumed.json"
    first = _consume()
    consumed_events.save(first, path)
    original_bytes = path.read_bytes()
    second = _consume(
        first,
        event_id="rb:h1:touch:v1:short:2026-08-24T09:00:00+00:00",
        symbol="GBPUSD",
        trigger_signature="short|rejection_block_h1|event-2",
    )

    def fail_replace(_source, _destination):
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(consumed_events.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated atomic replace failure"):
        second.save(path)

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.glob(".*.tmp")) == []


def test_save_rejects_non_store_without_touching_existing_file(tmp_path):
    path = tmp_path / "consumed.json"
    path.write_text("last-valid-snapshot", encoding="utf-8")

    with pytest.raises(RBConsumedEventsValidationError):
        consumed_events.save({}, path)

    assert path.read_text(encoding="utf-8") == "last-valid-snapshot"
