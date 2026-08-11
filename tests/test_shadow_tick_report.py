from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from tools.shadow_tick_report import build_report


def test_report_keeps_unknown_coverage_out_of_miss_rate(tmp_path):
    path = tmp_path / "shadow.db"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE plans (
            plan_id TEXT PRIMARY KEY,
            symbol TEXT,
            trigger_kind TEXT,
            setup_tf TEXT,
            activated_at_utc TEXT,
            activation_tick_msc INTEGER,
            finalized INTEGER,
            data_complete INTEGER,
            first_strict_touch_msc INTEGER,
            first_scan_touch_msc INTEGER,
            first_policy_touch_msc INTEGER,
            missed_by_cadence INTEGER,
            missed_by_policy INTEGER,
            recoverable_by_limit INTEGER,
            status TEXT,
            terminal_reason TEXT,
            first_gap_msc INTEGER,
            result_class TEXT
        )
        """
    )
    rows = [
        (
            "complete",
            "EURUSD",
            "RB H1",
            "1H",
            "2026-08-10T10:00:00+00:00",
            1_000,
            1,
            1,
            6_000,
            None,
            None,
            1,
            1,
            1,
            "FILLED",
            "broker pending LIMIT filled",
            None,
            "MISSED_BY_CADENCE",
        ),
        (
            "unknown",
            "EURUSD",
            "RB H1",
            "1H",
            "2026-08-10T10:15:00+00:00",
            10_000,
            1,
            0,
            11_000,
            None,
            None,
            1,
            1,
            1,
            "EXPIRED",
            "coverage gap",
            10_500,
            "UNKNOWN_CLOCK_DOMAIN",
        ),
    ]
    connection.executemany(
        "INSERT INTO plans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    connection.commit()
    connection.close()

    report = build_report(
        path,
        since=datetime(2026, 8, 10, tzinfo=timezone.utc),
    )

    assert report["total"]["plans"] == 2
    assert report["total"]["complete"] == 1
    assert report["total"]["unknown_coverage"] == 1
    assert report["total"]["unknown_clock_domain"] == 1
    assert report["total"]["missed_by_cadence"] == 1
    assert report["total"]["cadence_miss_rate"] == 1.0
    assert report["total"]["actual_pending_fills"] == 1
    assert report["groups"]["EURUSD"]["limit_recoverable"] == 1
