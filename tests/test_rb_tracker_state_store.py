from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import rb_tracker_state_store as store
from core.rb_tracker_state_store import RBTrackerStateValidationError


def test_missing_file_is_the_only_empty_state_case(tmp_path: Path) -> None:
    assert store.load(tmp_path / "missing.json") is None


def test_round_trip_uses_versioned_envelope_and_preserves_all_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "rb_h1.json"
    state = {
        "watermark": "2026-08-24T12:00:00+00:00",
        "symbols": {
            "EURUSD": {
                "blocks": [
                    {"event_id": f"rb-{index}", "age": index}
                    for index in range(25)
                ]
            }
        },
    }

    store.save(state, path)

    assert store.load(path) == state
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw == {
        "schema_version": store.RB_TRACKER_STATE_SCHEMA_VERSION,
        "state": state,
    }
    assert path.read_bytes().endswith(b"\n")
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "not json",
        "[]",
        "null",
        '{"schema_version": 1}',
        '{"schema_version": 1, "state": {}, "extra": true}',
        '{"schema_version": 2, "state": {}}',
        '{"schema_version": true, "state": {}}',
        '{"schema_version": 1, "state": []}',
        '{"schema_version": 1, "state": {}, "state": {}}',
        '{"schema_version": 1, "state": {"x": 1, "x": 2}}',
        '{"schema_version": 1, "state": {"x": NaN}}',
        '{"schema_version": 1, "state": {"x": Infinity}}',
        '{"schema_version": 1, "state": {"x": -Infinity}}',
    ],
)
def test_existing_invalid_file_fails_closed(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "rb_h1.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(RBTrackerStateValidationError):
        store.load(path)


def test_unreadable_existing_file_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rb_h1.json"
    path.write_text("{}", encoding="utf-8")

    def denied(_self: Path, **_kwargs: object) -> str:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_text", denied)

    with pytest.raises(RBTrackerStateValidationError, match="cannot read"):
        store.load(path)


@pytest.mark.parametrize(
    "state",
    [
        [],
        {"x": float("nan")},
        {"x": float("inf")},
        {"x": float("-inf")},
        {"x": object()},
    ],
)
def test_save_rejects_non_mapping_or_non_strict_json(
    tmp_path: Path, state: object
) -> None:
    path = tmp_path / "rb_h1.json"

    with pytest.raises(RBTrackerStateValidationError):
        store.save(state, path)  # type: ignore[arg-type]

    assert not path.exists()


def test_save_rejects_cycles_before_touching_destination(tmp_path: Path) -> None:
    path = tmp_path / "rb_h1.json"
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic

    with pytest.raises(RBTrackerStateValidationError, match="strict JSON"):
        store.save(cyclic, path)

    assert not path.exists()


def test_atomic_replace_failure_preserves_last_good_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rb_h1.json"
    store.save({"generation": 1}, path)
    original = path.read_bytes()

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(store.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        store.save({"generation": 2}, path)

    assert path.read_bytes() == original
    assert store.load(path) == {"generation": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_save_flushes_file_to_disk_before_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rb_h1.json"
    real_fsync = store.os.fsync
    fsynced: list[int] = []

    def record_fsync(fd: int) -> None:
        fsynced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(store.os, "fsync", record_fsync)

    store.save({"generation": 1}, path)

    assert len(fsynced) == 1
    assert store.load(path) == {"generation": 1}
