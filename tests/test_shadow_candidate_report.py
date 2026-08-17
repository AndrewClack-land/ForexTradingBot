from datetime import datetime, timezone
import sqlite3

import pytest

from core.shadow_candidate_ledger import ShadowCandidateLedger
from tools.shadow_candidate_report import build_report


def _candidate(
    kind: str,
    event: str,
    priority: int,
    *,
    side: str = "LONG",
) -> dict:
    return {
        "signal": "ENTER",
        "side": side,
        "trigger_kind": kind,
        "trigger_event_id": event,
        "trigger_reason": kind,
        "entry_price": 1.1,
        "stop_price": 1.09,
        "tp_prices": [1.11, 1.12, 1.13],
        "shadow_candidate_rank": priority,
        "production_priority": priority,
        "shadow_candidate_source": kind,
    }


def test_report_distinguishes_population_units_and_combinations(tmp_path):
    path = tmp_path / "candidates.db"
    ledger = ShadowCandidateLedger(path)
    observed = datetime(2026, 8, 11, 10, 3, tzinfo=timezone.utc)
    rb = _candidate("rejection_block_h1", "rb", 1)
    pivot = _candidate("pivot_reclaim", "pivot", 4)
    ledger.record_scan(
        deployment_id="release",
        symbol="EURUSD",
        observed_at_utc=observed,
        decision_bar_close=observed,
        production_signal=rb,
        candidates=[rb, pivot],
        downstream_disposition="ENTER",
    )
    ledger.close()

    report = build_report(path)
    assert report["total"]["decision_scans"] == 1
    assert report["total"]["candidate_observations"] == 2
    assert report["total"]["independent_opportunities"] == 2
    assert report["total"]["simultaneous_candidate_scans"] == 1
    assert report["total"]["unselected_independent_opportunities"] == 1
    assert report["downstream_dispositions"] == {"ENTER": 1}
    assert report["simultaneous_combinations"] == {
        "pivot_reclaim + rejection_block_h1": 1
    }


def test_report_exposes_quality_diagnostics_and_filters(tmp_path):
    path = tmp_path / "quality-candidates.db"
    ledger = ShadowCandidateLedger(path)
    observed = datetime(2026, 8, 11, 10, 3, tzinfo=timezone.utc)
    pivot = _candidate("pivot_reclaim", "pivot", 1)
    order_block = _candidate(
        "order_block_1h",
        "ob",
        2,
        side="SHORT",
    )
    scan_id = ledger.record_scan(
        deployment_id="release",
        symbol="GBPUSD",
        observed_at_utc=observed,
        decision_bar_close=observed,
        production_signal=pivot,
        candidates=[pivot, order_block],
        downstream_disposition="ENTER",
        downstream_details={"setup_id": "watcher-plan-1"},
    )
    with sqlite3.connect(path) as conn:
        opportunities = dict(
            conn.execute(
                "SELECT trigger_kind,opportunity_id FROM candidates WHERE scan_id=?",
                (scan_id,),
            )
        )
    assert ledger.record_quality_ranking(
        scan_id,
        [
            {
                "opportunity_id": opportunities["pivot_reclaim"],
                "quality_status": "SCORED",
                "quality_tp_probability": 0.55,
                "quality_tp_lcb": 0.45,
                "quality_tp_ucb": 0.65,
                "quality_expected_net_r": 0.10,
                "quality_conservative_net_r": 0.01,
                "quality_net_uncertainty_r": 0.09,
                "quality_effective_n": 20,
                "quality_backoff_path": "trigger_side>global",
                "quality_stop_atr_h1": 1.0,
                "quality_feature_missing": [],
                "quality_rank_score_r": 0.01,
                "quality_rank_position": 2,
                "quality_selected": False,
            },
            {
                "opportunity_id": opportunities["order_block_1h"],
                "quality_status": "SCORED",
                "quality_tp_probability": 0.64,
                "quality_tp_lcb": 0.53,
                "quality_tp_ucb": 0.74,
                "quality_expected_net_r": 0.22,
                "quality_conservative_net_r": 0.09,
                "quality_net_uncertainty_r": 0.13,
                "quality_effective_n": 12,
                "quality_backoff_path": "symbol_trigger_side>global",
                "quality_stop_atr_h1": 1.8,
                "quality_feature_missing": ["spread"],
                "quality_rank_score_r": 0.08,
                "quality_rank_position": 1,
                "quality_selected": True,
            },
        ],
        "quality-model",
    )
    ledger.close()

    report = build_report(path)
    assert (
        report["candidate_groups"]["GBPUSD x pivot_reclaim x LONG"][
            "mean_quality_tp_probability"
        ]
        == 0.55
    )
    assert report["atr_stop_buckets"]["inside_0.75_1.50"]["candidate_observations"] == 1
    assert (
        report["atr_stop_buckets"]["outside_0.75_1.50"]["candidate_observations"] == 1
    )
    assert report["gbpusd_short"]["candidate_observations"] == 1
    assert (
        report["pivot_ob_side_comparisons"]["pivot_reclaim"]["by_symbol"]["GBPUSD"][
            "LONG"
        ]["candidate_observations"]
        == 1
    )
    assert (
        report["pivot_ob_side_comparisons"]["order_block"]["by_symbol"]["GBPUSD"][
            "SHORT"
        ]["candidate_observations"]
        == 1
    )
    assert report["production_vs_quality_winner"]["disagreement_scans"] == 1
    assert report["watcher_plan_coverage"]["coverage_rate"] == 1.0

    filtered = build_report(path, side="short", status="scored")
    assert filtered["total"]["candidate_observations"] == 1
    assert filtered["filters"]["side"] == "short"
    assert filtered["filters"]["status"] == "scored"


def test_report_requires_migrated_v2_schema(tmp_path):
    path = tmp_path / "v1.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO metadata VALUES ('schema_version', '1')")
    with pytest.raises(RuntimeError, match="migrate to v2"):
        build_report(path)
