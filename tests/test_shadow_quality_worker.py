"""Prove the shadow quality diagnostics cannot alter production signals.

The batch writer runs in the diagnostic worker after the live decision is
final.  These tests execute the real ``Core._write_shadow_candidate_batch``
with a real ``ShadowCandidateLedger`` and a real fitted quality scorer, then
assert byte-identical production inputs and a ledger that only gained
``quality_*`` telemetry.
"""

from __future__ import annotations

import copy
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd

import main
from core.hierarchical_quality_score import (
    LiveHierarchicalQualityScorer,
    fit_quality_profile,
)
from core.shadow_candidate_ledger import ShadowCandidateLedger


DECISION = datetime(2026, 6, 10, 9, 30, tzinfo=timezone.utc)
OBSERVED = DECISION + timedelta(seconds=5)


class StubStrategy:
    """Deterministic candidate source with the production attribute contract."""

    def __init__(self, candidates):
        self._candidates = candidates
        self._last_candidate_errors = []

    def generate_candidate_signals(self, data, symbol=""):
        del data, symbol
        self._last_candidate_errors = []
        return [dict(candidate) for candidate in self._candidates]


class BrokenScorer:
    """A quality scorer whose every access fails."""

    @property
    def profile(self):
        raise RuntimeError("profile unavailable")

    def score(self, *args, **kwargs):
        raise RuntimeError("scoring unavailable")


def _scorer() -> LiveHierarchicalQualityScorer:
    train_start = datetime(2026, 5, 4, 9, 30, tzinfo=timezone.utc)
    rows = []
    for index in range(12):
        decision = train_start + timedelta(hours=6 * index)
        rows.append({
            "decision_time_utc": decision.isoformat().replace("+00:00", "Z"),
            "symbol": "EURUSD",
            "side": "LONG" if index % 2 == 0 else "SHORT",
            "trigger_kind": (
                "pivot_reclaim" if index % 3 else "orderblock_1h"
            ),
            "stop_atr_h1": 1.0 + 0.05 * (index % 4),
            "spread_r": 0.05,
            "fvg_relative_state": "NEUTRAL",
            "intended_execution_mode": "MARKET",
            "label_complete": True,
            "label_tp1_hit": int(index % 2 == 0),
            "label_realized_net_r": 0.8 if index % 2 == 0 else -1.0,
        })
    return LiveHierarchicalQualityScorer(fit_quality_profile(rows))


def _hourly_frame(rows: int = 24) -> pd.DataFrame:
    index = pd.date_range(
        DECISION - timedelta(hours=rows),
        periods=rows,
        freq="1h",
        tz="UTC",
    )
    base = 1.10
    return pd.DataFrame(
        {
            "open": [base + 0.001 * (i % 5) for i in range(rows)],
            "high": [base + 0.004 + 0.001 * (i % 5) for i in range(rows)],
            "low": [base - 0.004 + 0.001 * (i % 5) for i in range(rows)],
            "close": [base + 0.002 + 0.001 * (i % 5) for i in range(rows)],
        },
        index=index,
    )


def _candidates():
    return [
        {
            "signal": "ENTER",
            "side": "LONG",
            "entry_price": 1.1000,
            "stop_price": 1.0950,
            "trigger_kind": "pivot_reclaim",
            "trigger_event_id": "pivot-1",
            "trigger_reason": "pivot_reclaim | test",
            "shadow_candidate_source": "pivot_reclaim",
        },
        {
            "signal": "ENTER",
            "side": "LONG",
            "entry_price": 1.1005,
            "stop_price": 1.0940,
            "trigger_kind": "orderblock_1h",
            "trigger_event_id": "ob-1",
            "trigger_reason": "orderblock_1h | test",
            "shadow_candidate_source": "orderblock_1h",
        },
    ]


def _job():
    return {
        "deployment_id": "test-deploy",
        "symbol": "EURUSD",
        "observed_at_utc": OBSERVED,
        "decision_bar_close": DECISION,
        "production_signal": {
            "signal": "ENTER",
            "side": "LONG",
            "entry_price": 1.1000,
            "stop_price": 1.0950,
            "trigger_kind": "pivot_reclaim",
            "trigger_event_id": "pivot-1",
        },
        "strategy_data": {"1H": _hourly_frame()},
        "active_exposures": {},
    }


def _run_batch(tmp_path, quality_scorer):
    ledger = ShadowCandidateLedger(tmp_path / "shadow.db")
    strategy = StubStrategy(_candidates())
    job = _job()
    outcomes = {"EURUSD": {"signal": "ENTER", "executed": True}}
    production_frozen = copy.deepcopy(job["production_signal"])
    outcomes_frozen = copy.deepcopy(outcomes)
    frame_frozen = job["strategy_data"]["1H"].copy(deep=True)

    result = main.Core._write_shadow_candidate_batch(
        ledger,
        strategy,
        (job,),
        outcomes,
        None,
        None,
        None,
        quality_scorer,
    )

    assert result is None
    assert job["production_signal"] == production_frozen
    assert outcomes == outcomes_frozen
    assert job["strategy_data"]["1H"].equals(frame_frozen)
    assert not any(
        str(key).startswith("quality_")
        for key in job["production_signal"]
    )
    assert ledger.enabled, ledger.disabled_reason
    ledger.close()
    return production_frozen


def test_quality_scoring_writes_telemetry_without_touching_production(
    tmp_path,
) -> None:
    scorer = _scorer()
    production_frozen = _run_batch(tmp_path, scorer)

    with sqlite3.connect(tmp_path / "shadow.db") as connection:
        connection.row_factory = sqlite3.Row
        scans = connection.execute("SELECT * FROM scans").fetchall()
        assert len(scans) == 1
        assert scans[0]["production_signal"] == production_frozen["signal"]
        rows = connection.execute(
            "SELECT * FROM candidates ORDER BY quality_rank_position"
        ).fetchall()
        assert len(rows) == 2
        assert [row["quality_rank_position"] for row in rows] == [1, 2]
        assert sum(row["quality_selected"] for row in rows) == 1
        for row in rows:
            assert row["quality_model_id"] == scorer.profile.profile_id
            assert row["quality_status"] == "SCORED"
            assert row["quality_tp_probability"] is not None
            assert row["quality_tp_lcb"] <= row["quality_tp_probability"]
            assert row["quality_tp_probability"] <= row["quality_tp_ucb"]
            assert row["quality_conservative_net_r"] is not None
            assert row["quality_net_uncertainty_r"] > 0.0
            # The H1 ATR injected from the closed-bars view makes the
            # stop-distance feature real instead of MISSING.
            assert row["quality_stop_atr_h1"] is not None
            missing = json.loads(row["quality_feature_missing_json"])
            assert "stop_atr_h1" not in missing


def test_broken_quality_scorer_cannot_break_the_batch(tmp_path) -> None:
    _run_batch(tmp_path, BrokenScorer())

    with sqlite3.connect(tmp_path / "shadow.db") as connection:
        connection.row_factory = sqlite3.Row
        assert (
            connection.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
            == 1
        )
        rows = connection.execute("SELECT * FROM candidates").fetchall()
        assert len(rows) == 2
        for row in rows:
            assert row["quality_model_id"] is None
            assert row["quality_status"] is None
            assert row["quality_rank_position"] is None


def test_missing_quality_scorer_keeps_v1_behaviour(tmp_path) -> None:
    _run_batch(tmp_path, None)

    with sqlite3.connect(tmp_path / "shadow.db") as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM candidates").fetchall()
        assert len(rows) == 2
        for row in rows:
            assert row["quality_model_id"] is None
            assert row["quality_rank_position"] is None
