from datetime import datetime, timezone

from core.shadow_candidate_ledger import ShadowCandidateLedger
from tools.shadow_candidate_report import build_report


def _candidate(kind: str, event: str, priority: int) -> dict:
    return {
        "signal": "ENTER",
        "side": "LONG",
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
