from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import UTC, datetime, timedelta, timezone

import pytest

from core.quality_event_ledger import (
    EVENT_TYPES,
    SCHEMA_VERSION,
    QualityEventLedger,
)


OCCURRED = datetime(2026, 8, 17, 10, 0, tzinfo=UTC)
KNOWN = OCCURRED + timedelta(seconds=2)


def record(
    ledger: QualityEventLedger,
    *,
    event_type: str = "CANDIDATE_DECISION",
    payload: dict | None = None,
    event_id: str | None = None,
    supersedes_event_id: str | None = None,
    occurred_at_utc: datetime = OCCURRED,
    known_at_utc: datetime = KNOWN,
    deployment_id: str = "deploy-a",
    strategy_version: str = "release-a",
    scan_id: str | None = "scan-a",
    observation_id: str | None = "observation-a",
    opportunity_id: str | None = "opportunity-a",
    setup_id: str | None = "setup-a",
    idea_id: str | None = "idea-a",
    symbol: str = "eurusd",
    side: str | None = "long",
    trigger_kind: str | None = "pivot_reclaim_1h",
) -> str | None:
    return ledger.record_event(
        event_type=event_type,
        occurred_at_utc=occurred_at_utc,
        known_at_utc=known_at_utc,
        deployment_id=deployment_id,
        strategy_version=strategy_version,
        scan_id=scan_id,
        observation_id=observation_id,
        opportunity_id=opportunity_id,
        setup_id=setup_id,
        idea_id=idea_id,
        symbol=symbol,
        side=side,
        trigger_kind=trigger_kind,
        payload=payload or {"score": 0.25},
        supersedes_event_id=supersedes_event_id,
        event_id=event_id,
    )


def test_schema_is_exact_append_only_and_uses_wal(tmp_path) -> None:
    db_path = tmp_path / "quality.db"
    ledger = QualityEventLedger(db_path)
    try:
        assert ledger.enabled
        assert (
            ledger._conn.execute(
                "PRAGMA recursive_triggers"
            ).fetchone()[0]
            == 1
        )
        event_id = record(ledger)
        assert event_id is not None

        with sqlite3.connect(db_path) as connection:
            columns = tuple(
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(quality_events)"
                ).fetchall()
            )
            assert columns == (
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
            assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(
                    "UPDATE quality_events SET symbol='GBPUSD' WHERE event_id=?",
                    (event_id,),
                )
            connection.rollback()
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(
                    "DELETE FROM quality_events WHERE event_id=?",
                    (event_id,),
                )
            connection.rollback()
            stored = list(
                connection.execute(
                    "SELECT * FROM quality_events WHERE event_id=?",
                    (event_id,),
                ).fetchone()
            )
            stored[columns.index("symbol")] = "GBPUSD"
            placeholders = ",".join("?" for _ in columns)
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(
                    "INSERT OR REPLACE INTO quality_events "
                    f"({','.join(columns)}) VALUES ({placeholders})",
                    stored,
                )
            connection.rollback()
            assert connection.execute(
                "SELECT symbol FROM quality_events WHERE event_id=?",
                (event_id,),
            ).fetchone()[0] == "EURUSD"
    finally:
        ledger.close()


def test_precreated_trigger_name_poisoning_disables_initialization(
    tmp_path,
) -> None:
    db_path = tmp_path / "poisoned.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE unrelated (value INTEGER);
            CREATE TRIGGER quality_events_no_update
            BEFORE UPDATE ON unrelated
            BEGIN
                SELECT 1;
            END;
            """
        )

    ledger = QualityEventLedger(db_path)
    try:
        assert ledger.enabled is False
        assert "unexpected trigger definition" in str(
            ledger.disabled_reason
        )
    finally:
        ledger.close()


def test_canonical_payload_hash_and_identical_retry_are_idempotent(tmp_path) -> None:
    ledger = QualityEventLedger(tmp_path / "quality.db")
    try:
        first = record(
            ledger,
            payload={"z": [3, 2, 1], "a": {"price": 1.25}},
        )
        second = record(
            ledger,
            payload={"a": {"price": 1.25}, "z": [3, 2, 1]},
        )
        assert first == second
        assert first is not None
        row = ledger.get_event(first)
        assert row is not None
        assert row["payload_json"] == '{"a":{"price":1.25},"z":[3,2,1]}'
        assert row["payload_sha256"] == hashlib.sha256(
            row["payload_json"].encode("utf-8")
        ).hexdigest()
        assert row["schema_version"] == SCHEMA_VERSION
        assert ledger.summary()["events"] == 1
    finally:
        ledger.close()


def test_generated_identity_is_stable_across_equivalent_utc_times(tmp_path) -> None:
    eastern = timezone(timedelta(hours=3))
    first = QualityEventLedger(tmp_path / "first.db")
    second = QualityEventLedger(tmp_path / "second.db")
    try:
        first_id = record(first, payload={"x": -0.0})
        second_id = record(
            second,
            payload={"x": 0.0},
            occurred_at_utc=OCCURRED.astimezone(eastern),
            known_at_utc=KNOWN.astimezone(eastern),
        )
        assert first_id == second_id
    finally:
        first.close()
        second.close()


def test_same_explicit_identity_with_different_bytes_fails_open(tmp_path) -> None:
    ledger = QualityEventLedger(tmp_path / "quality.db")
    explicit = "outcome-identity"
    assert record(ledger, event_id=explicit, payload={"net": 1.0}) == explicit

    assert record(ledger, event_id=explicit, payload={"net": -1.0}) is None
    assert ledger.enabled is False
    assert "different canonical bytes" in str(ledger.disabled_reason)
    row = ledger.get_event(explicit)
    assert row is not None
    assert json.loads(row["payload_json"]) == {"net": 1.0}
    assert ledger.summary()["events"] == 1


def test_correction_is_a_new_immutable_event_referencing_an_outcome(tmp_path) -> None:
    ledger = QualityEventLedger(tmp_path / "quality.db")
    try:
        outcome = record(
            ledger,
            event_type="BROKER_OUTCOME",
            payload={"pnl_complete": True, "realized_net": 10.0},
        )
        assert outcome is not None
        correction = record(
            ledger,
            event_type="OUTCOME_CORRECTION",
            payload={"pnl_complete": True, "realized_net": 9.5},
            supersedes_event_id=outcome,
            known_at_utc=KNOWN + timedelta(minutes=1),
        )
        assert correction is not None
        assert correction != outcome
        correction_row = ledger.get_event(correction)
        assert correction_row is not None
        assert correction_row["supersedes_event_id"] == outcome
        original_row = ledger.get_event(outcome)
        assert original_row is not None
        assert json.loads(original_row["payload_json"])["realized_net"] == 10.0
        assert ledger.summary()["events"] == 2
    finally:
        ledger.close()


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"event_type": "UNKNOWN"}, "unsupported event_type"),
        (
            {"occurred_at_utc": datetime(2026, 8, 17, 10, 0)},
            "timezone-aware",
        ),
        (
            {"known_at_utc": OCCURRED - timedelta(seconds=1)},
            "cannot precede",
        ),
        ({"payload": {"value": float("nan")}}, "non-finite"),
        (
            {
                "event_type": "OUTCOME_CORRECTION",
                "supersedes_event_id": "missing",
            },
            "superseded event does not exist",
        ),
    ],
)
def test_invalid_event_is_rejected_fail_open(tmp_path, overrides, error) -> None:
    ledger = QualityEventLedger(tmp_path / f"{error[:5]}.db")
    assert record(ledger, **overrides) is None
    assert ledger.enabled is False
    assert error in str(ledger.disabled_reason)
    assert ledger.summary()["events"] == 0


def test_correction_cannot_rewrite_a_non_outcome_event(tmp_path) -> None:
    ledger = QualityEventLedger(tmp_path / "quality.db")
    decision = record(ledger)
    assert decision is not None
    assert record(
        ledger,
        event_type="OUTCOME_CORRECTION",
        supersedes_event_id=decision,
    ) is None
    assert ledger.enabled is False
    assert "must supersede an outcome" in str(ledger.disabled_reason)


def test_read_only_queries_are_bounded_and_filterable(tmp_path) -> None:
    ledger = QualityEventLedger(tmp_path / "quality.db")
    try:
        decision = record(ledger)
        score = ledger.record_event(
            event_type="QUALITY_SCORE",
            occurred_at_utc=OCCURRED,
            known_at_utc=KNOWN,
            deployment_id="deploy-a",
            strategy_version="release-a",
            opportunity_id="opportunity-b",
            setup_id="setup-b",
            symbol="GBPUSD",
            side="SHORT",
            trigger_kind="order_block_1h",
            payload={"conservative_net_r": 0.12},
        )
        assert decision is not None and score is not None
        assert [row["event_id"] for row in ledger.query_events(setup_id="setup-a")] == [
            decision
        ]
        assert [
            row["event_id"]
            for row in ledger.query_events(event_type="quality_score")
        ] == [score]
        summary = ledger.summary()
        assert summary == {
            "enabled": True,
            "disabled_reason": None,
            "events": 2,
            "by_type": {"CANDIDATE_DECISION": 1, "QUALITY_SCORE": 1},
        }
        assert set(summary["by_type"]) <= EVENT_TYPES
    finally:
        ledger.close()


def test_concurrent_distinct_writes_are_all_recorded(tmp_path) -> None:
    ledger = QualityEventLedger(tmp_path / "quality.db")
    thread_count, per_thread = 8, 5
    results: list[list[str | None]] = [[] for _ in range(thread_count)]
    barrier = threading.Barrier(thread_count)

    def worker(index: int) -> None:
        barrier.wait()
        for item in range(per_thread):
            results[index].append(
                record(
                    ledger,
                    scan_id=f"scan-{index}",
                    payload={"cell": [index, item]},
                )
            )

    threads = [
        threading.Thread(target=worker, args=(index,))
        for index in range(thread_count)
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        event_ids = [event_id for chunk in results for event_id in chunk]
        assert all(event_id is not None for event_id in event_ids)
        assert len(set(event_ids)) == thread_count * per_thread
        assert ledger.enabled, ledger.disabled_reason
        assert ledger.summary()["events"] == thread_count * per_thread
    finally:
        ledger.close()


def test_concurrent_identical_retries_stay_idempotent(tmp_path) -> None:
    ledger = QualityEventLedger(tmp_path / "quality.db")
    thread_count = 8
    results: list[str | None] = [None] * thread_count
    barrier = threading.Barrier(thread_count)

    def worker(index: int) -> None:
        barrier.wait()
        results[index] = record(ledger, payload={"net": 1.0})

    threads = [
        threading.Thread(target=worker, args=(index,))
        for index in range(thread_count)
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert all(event_id is not None for event_id in results)
        assert len(set(results)) == 1
        assert ledger.enabled, ledger.disabled_reason
        assert ledger.summary()["events"] == 1
    finally:
        ledger.close()


def test_two_connections_on_one_file_serialize_without_failing(
    tmp_path,
) -> None:
    db_path = tmp_path / "quality.db"
    first = QualityEventLedger(db_path)
    second = QualityEventLedger(db_path)
    barrier = threading.Barrier(2)
    outcomes: dict[str, list[str | None]] = {"first": [], "second": []}

    def worker(name: str, ledger: QualityEventLedger) -> None:
        barrier.wait()
        for item in range(10):
            outcomes[name].append(
                record(
                    ledger,
                    scan_id=f"{name}-{item}",
                    payload={"writer": name, "item": item},
                )
            )

    threads = [
        threading.Thread(target=worker, args=("first", first)),
        threading.Thread(target=worker, args=("second", second)),
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        written = outcomes["first"] + outcomes["second"]
        assert all(event_id is not None for event_id in written)
        assert len(set(written)) == 20
        assert first.enabled, first.disabled_reason
        assert second.enabled, second.disabled_reason
        assert first.summary()["events"] == 20
        assert second.summary()["events"] == 20
    finally:
        first.close()
        second.close()


def test_database_open_failure_is_fail_open(tmp_path) -> None:
    def broken_connection(*args, **kwargs):
        del args, kwargs
        raise OSError("disk unavailable")

    ledger = QualityEventLedger(
        tmp_path / "cannot-open.db",
        connection_factory=broken_connection,
    )
    assert ledger.enabled is False
    assert "disk unavailable" in str(ledger.disabled_reason)
    assert record(ledger) is None
    assert ledger.get_event("missing") is None
    assert ledger.query_events() == []
    assert ledger.summary() == {
        "enabled": False,
        "disabled_reason": "OSError: disk unavailable",
        "events": 0,
        "by_type": {},
    }
