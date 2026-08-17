from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import sqlite3

from core.shadow_candidate_ledger import (
    ShadowCandidateLedger,
    stable_candidate_opportunity_id,
    stable_candidate_scan_id,
    trigger_signature,
)


UTC = timezone.utc


def _candidate(kind: str, event: str, rank: int) -> dict:
    return {
        "signal": "ENTER",
        "side": "LONG",
        "trigger_kind": kind,
        "trigger_event_id": event,
        "trigger_reason": f"{kind} test",
        "entry_price": 1.1,
        "entry_min": 1.099,
        "entry_max": 1.101,
        "stop_price": 1.09,
        "tp_prices": [1.11, 1.12, 1.13],
        "shadow_candidate_rank": rank,
        "production_priority": rank,
        "shadow_candidate_source": kind,
        "factor_vector": {"finite": 1.0, "nan": float("nan")},
    }


def test_scan_identity_is_minute_idempotent_but_opportunity_is_daily():
    first = datetime(2026, 8, 11, 10, 3, 5, tzinfo=UTC)
    same_minute = first + timedelta(seconds=40)
    next_minute = first + timedelta(minutes=1)
    assert stable_candidate_scan_id("d1", "eurusd", first) == (
        stable_candidate_scan_id("d1", "EURUSD", same_minute)
    )
    assert stable_candidate_scan_id("d1", "EURUSD", first) != (
        stable_candidate_scan_id("d1", "EURUSD", next_minute)
    )

    signature = trigger_signature(_candidate("pivot_reclaim", "p1", 1))
    day_one = stable_candidate_opportunity_id("EURUSD", signature, first.date())
    day_two = stable_candidate_opportunity_id(
        "EURUSD", signature, (first + timedelta(days=1)).date()
    )
    assert day_one != day_two


def test_ledger_separates_scans_observations_and_independent_ideas(tmp_path):
    ledger = ShadowCandidateLedger(tmp_path / "candidates.db")
    observed = datetime(2026, 8, 11, 10, 3, 5, tzinfo=UTC)
    candidates = [
        _candidate("rejection_block_h1", "rb1", 1),
        _candidate("pivot_reclaim", "p1", 4),
    ]
    production = dict(candidates[0])

    first_scan = ledger.record_scan(
        deployment_id="release-a",
        symbol="EURUSD",
        observed_at_utc=observed,
        decision_bar_close=observed.replace(minute=0, second=0),
        production_signal=production,
        candidates=candidates,
        downstream_disposition="WAIT_SESSION",
        downstream_details={"signal": "WAIT_SESSION"},
    )
    retry = ledger.record_scan(
        deployment_id="release-a",
        symbol="EURUSD",
        observed_at_utc=observed + timedelta(seconds=30),
        decision_bar_close=observed.replace(minute=0, second=0),
        production_signal=production,
        candidates=[candidates[0]],
    )
    second_scan = ledger.record_scan(
        deployment_id="release-a",
        symbol="EURUSD",
        observed_at_utc=observed + timedelta(minutes=1),
        decision_bar_close=observed.replace(minute=0, second=0),
        production_signal=production,
        candidates=candidates,
    )

    assert first_scan == retry
    assert second_scan != first_scan
    assert ledger.summary() == {
        "enabled": True,
        "scans": 2,
        "candidate_observations": 4,
        "independent_opportunities": 2,
        "structural_events": 2,
        "simultaneous_candidate_scans": 2,
        "production_selected_observations": 2,
    }

    with sqlite3.connect(tmp_path / "candidates.db") as conn:
        row = conn.execute(
            "SELECT candidate_count,downstream_disposition FROM scans WHERE scan_id=?",
            (first_scan,),
        ).fetchone()
        payload = conn.execute(
            "SELECT payload_json FROM candidates "
            "WHERE scan_id=? ORDER BY candidate_rank LIMIT 1",
            (first_scan,),
        ).fetchone()[0]
    assert row == (2, "WAIT_SESSION")
    assert '"nan":null' in payload
    ledger.close()


def test_production_selection_uses_exact_trigger_identity(tmp_path):
    ledger = ShadowCandidateLedger(tmp_path / "selection.db")
    observed = datetime(2026, 8, 11, 10, 3, tzinfo=UTC)
    first = _candidate("pivot_reclaim", "event-1", 1)
    second = _candidate("pivot_reclaim", "event-2", 2)
    scan_id = ledger.record_scan(
        deployment_id="release-a",
        symbol="EURUSD",
        observed_at_utc=observed,
        decision_bar_close=observed,
        production_signal=second,
        candidates=[first, second],
    )
    with sqlite3.connect(tmp_path / "selection.db") as conn:
        rows = conn.execute(
            "SELECT trigger_signature,selected_by_production "
            "FROM candidates WHERE scan_id=? ORDER BY candidate_rank",
            (scan_id,),
        ).fetchall()
    assert rows[0][1] == 0
    assert rows[1][1] == 1
    ledger.close()


def test_naive_time_is_rejected_fail_open(tmp_path):
    ledger = ShadowCandidateLedger(tmp_path / "invalid.db")
    assert (
        ledger.record_scan(
            deployment_id="release-a",
            symbol="EURUSD",
            observed_at_utc=datetime(2026, 8, 11, 10, 3),
            decision_bar_close=None,
            production_signal=None,
            candidates=[],
        )
        is None
    )
    assert ledger.enabled is False
    assert "timezone-aware" in str(ledger.disabled_reason)


def test_ranking_is_shadow_annotation_only(tmp_path):
    ledger = ShadowCandidateLedger(tmp_path / "ranking.db")
    observed = datetime(2026, 8, 11, 10, 3, tzinfo=UTC)
    candidate = _candidate("pivot_reclaim", "event-1", 1)
    scan_id = ledger.record_scan(
        deployment_id="release-a",
        symbol="EURUSD",
        observed_at_utc=observed,
        decision_bar_close=observed,
        production_signal=candidate,
        candidates=[candidate],
    )
    with sqlite3.connect(tmp_path / "ranking.db") as conn:
        opportunity_id = conn.execute(
            "SELECT opportunity_id FROM candidates WHERE scan_id=?",
            (scan_id,),
        ).fetchone()[0]
    assert ledger.record_ranking(
        scan_id,
        [
            {
                "opportunity_id": opportunity_id,
                "expected_gross_r": 0.2,
                "estimated_cost_r": 0.08,
                "expected_net_r": 0.12,
                "correlation_penalty_r": 0.02,
                "ranking_score": 0.10,
                "rank_position": 1,
                "selected": True,
            }
        ],
        model_id="train-fold-1",
    )
    with sqlite3.connect(tmp_path / "ranking.db") as conn:
        row = conn.execute(
            "SELECT model_id,expected_net_r,shadow_selected,"
            "selected_by_production FROM candidates"
        ).fetchone()
    assert row == ("train-fold-1", 0.12, 1, 1)
    ledger.close()


def test_v1_schema_migrates_additively_and_preserves_rows(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT INTO metadata VALUES ('schema_version', '1');
            CREATE TABLE scans (
                scan_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                observed_at_utc TEXT NOT NULL
            );
            INSERT INTO scans VALUES (
                'scan-1', 'EURUSD', '2026-08-11T10:03:05Z'
            );
            CREATE TABLE candidates (
                observation_id TEXT PRIMARY KEY,
                scan_id TEXT NOT NULL,
                opportunity_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                trigger_kind TEXT,
                side TEXT
            );
            INSERT INTO candidates VALUES (
                'observation-1', 'scan-1', 'opportunity-1',
                'event-1', 'EURUSD', 'pivot_reclaim', 'LONG'
            );
            """
        )

    ledger = ShadowCandidateLedger(path)
    with sqlite3.connect(path) as conn:
        version = conn.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()[0]
        candidate = conn.execute(
            "SELECT observation_id,opportunity_id,quality_status,"
            "quality_conservative_net_r FROM candidates"
        ).fetchone()
        scan = conn.execute("SELECT scan_id,watcher_plan_id FROM scans").fetchone()
    assert version == "2"
    assert candidate == ("observation-1", "opportunity-1", None, None)
    assert scan == ("scan-1", None)
    ledger.close()

    reopened = ShadowCandidateLedger(path)
    assert reopened.enabled is True
    reopened.close()


def test_quality_ranking_is_separate_whitelisted_annotation(tmp_path):
    path = tmp_path / "quality.db"
    ledger = ShadowCandidateLedger(path)
    observed = datetime(2026, 8, 11, 10, 3, tzinfo=UTC)
    candidate = _candidate("pivot_reclaim", "event-1", 1)
    scan_id = ledger.record_scan(
        deployment_id="release-a",
        symbol="GBPUSD",
        observed_at_utc=observed,
        decision_bar_close=observed,
        production_signal=candidate,
        candidates=[candidate],
        downstream_details={"setup_id": "watcher-plan-1"},
    )
    with sqlite3.connect(path) as conn:
        opportunity_id = conn.execute(
            "SELECT opportunity_id FROM candidates WHERE scan_id=?",
            (scan_id,),
        ).fetchone()[0]
    assert ledger.record_ranking(
        scan_id,
        [
            {
                "opportunity_id": opportunity_id,
                "expected_net_r": 0.12,
                "ranking_score": 0.10,
                "rank_position": 1,
                "selected": True,
            }
        ],
        model_id="legacy-model",
    )
    assert ledger.record_quality_ranking(
        scan_id,
        [
            {
                "opportunity_id": opportunity_id,
                "quality_status": "SCORED",
                "quality_tp_probability": 0.62,
                "quality_tp_lcb": 0.51,
                "quality_tp_ucb": 0.72,
                "quality_expected_net_r": 0.18,
                "quality_conservative_net_r": 0.07,
                "quality_net_uncertainty_r": 0.11,
                "quality_effective_n": 13.5,
                "quality_backoff_path": [
                    "symbol_trigger_side",
                    "trigger_side",
                    "global",
                ],
                "quality_stop_atr_h1": 1.1,
                "quality_feature_missing": ["spread"],
                "quality_rank_score_r": 0.06,
                "quality_rank_position": 1,
                "quality_selected": True,
                "side": "SHORT",
            }
        ],
        "quality-model-1",
    )

    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT model_id,ranking_score,shadow_selected,"
            "quality_model_id,quality_status,quality_tp_probability,"
            "quality_conservative_net_r,quality_feature_missing_json,"
            "quality_payload_json,quality_selected FROM candidates"
        ).fetchone()
        watcher_plan_id = conn.execute(
            "SELECT watcher_plan_id FROM scans WHERE scan_id=?",
            (scan_id,),
        ).fetchone()[0]
    payload = json.loads(row[8])
    assert row[:3] == ("legacy-model", 0.10, 1)
    assert row[3:8] == (
        "quality-model-1",
        "SCORED",
        0.62,
        0.07,
        '["spread"]',
    )
    assert row[9] == 1
    assert watcher_plan_id == "watcher-plan-1"
    assert payload["quality_model_id"] == "quality-model-1"
    assert payload["quality_feature_missing"] == ["spread"]
    assert "side" not in payload
    ledger.close()
