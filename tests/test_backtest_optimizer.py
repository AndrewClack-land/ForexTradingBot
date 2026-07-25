from __future__ import annotations

from copy import deepcopy
import json

import pandas as pd

from backtest.optimizer import build_shadow_scores
from core.narrative_scoring import build_factor_vector


def _vector(side: str):
    return build_factor_vector(
        {
            "h1_premium_discount": {
                "present": True,
                "side": side,
            },
            "true_breakout_15m": {
                "present": True,
                "side": side,
            },
        },
        base_margin=1,
    )


def _setup(
    setup_id: str,
    *,
    fold: int,
    decision: str,
    exit_time: str,
    net_r: float,
    policy: str = "stop-first",
):
    return {
        "setup_id": setup_id,
        "candidate_id": f"c-{setup_id}",
        "fold_index": fold,
        "symbol": "EURUSD",
        "decision_time": decision,
        "entry_time": decision,
        "exit_time": exit_time,
        "side": "LONG",
        "trigger_kind": "h1_pivot_reclaim_15m",
        "policy": policy,
        "status": "CLOSED",
        "net_r": net_r,
        "vol_r": 20.0,
        "factor_vector": _vector("LONG"),
    }


def _periods():
    return [
        {
            "fold_index": 0,
            "train_start": "2025-11-01T00:00:00Z",
            "train_end": "2026-01-01T00:00:00Z",
            "test_start": "2026-01-01T00:00:00Z",
            "test_end": "2026-02-01T00:00:00Z",
        },
        {
            "fold_index": 1,
            "train_start": "2025-12-01T00:00:00Z",
            "train_end": "2026-02-01T00:00:00Z",
            "test_start": "2026-02-01T00:00:00Z",
            "test_end": "2026-03-01T00:00:00Z",
        },
        {
            "fold_index": 2,
            "train_start": "2026-01-01T00:00:00Z",
            "train_end": "2026-03-01T00:00:00Z",
            "test_start": "2026-03-01T00:00:00Z",
            "test_end": "2026-04-01T00:00:00Z",
        },
    ]


def _run(setups):
    return build_shadow_scores(
        setups=setups,
        periods=_periods(),
        symbols=["EURUSD"],
        entry_ttl=pd.Timedelta(minutes=15),
        max_holding=pd.Timedelta(days=1),
        min_train_setups=1,
    )


def test_shadow_score_never_changes_execution_or_risk():
    setups = [
        _setup(
            "a",
            fold=0,
            decision="2026-01-05T00:00:00Z",
            exit_time="2026-01-06T00:00:00Z",
            net_r=1.0,
        ),
        _setup(
            "b",
            fold=1,
            decision="2026-02-05T00:00:00Z",
            exit_time="2026-02-06T00:00:00Z",
            net_r=-1.0,
        ),
        _setup(
            "c",
            fold=2,
            decision="2026-03-05T00:00:00Z",
            exit_time="2026-03-06T00:00:00Z",
            net_r=0.5,
        ),
    ]
    before = deepcopy(setups)

    result = _run(setups)

    assert setups == before
    assert result == _run(deepcopy(before))
    json.dumps(result, allow_nan=False)
    assert result["summary"]["execution_changed_by_score"] is False
    assert result["summary"]["no_hard_rejection"] is True
    assert result["summary"]["risk_changed"] is False
    assert result["summary"]["target"] == "net_r_given_production_fill"
    by_fold = {
        row["fold_index"]: row
        for row in result["predictions"]
    }
    assert by_fold[0]["model_status"] == "NOT_FIT"
    assert by_fold[1]["model_status"] == "FIT"
    assert by_fold[2]["model_status"] == "FIT"


def test_shadow_score_is_canonical_under_input_permutation():
    setups = [
        _setup(
            "a",
            fold=0,
            decision="2026-01-05T00:00:00Z",
            exit_time="2026-01-06T00:00:00Z",
            net_r=1.0,
        ),
        _setup(
            "b",
            fold=0,
            decision="2026-01-07T00:00:00Z",
            exit_time="2026-01-08T00:00:00Z",
            net_r=-0.5,
        ),
        _setup(
            "c",
            fold=1,
            decision="2026-02-05T00:00:00Z",
            exit_time="2026-02-06T00:00:00Z",
            net_r=0.25,
        ),
    ]

    assert _run(setups) == _run(list(reversed(setups)))


def test_future_outcome_cannot_change_earlier_models_or_scores():
    setups = [
        _setup(
            "a",
            fold=0,
            decision="2026-01-05T00:00:00Z",
            exit_time="2026-01-06T00:00:00Z",
            net_r=1.0,
        ),
        _setup(
            "b",
            fold=1,
            decision="2026-02-05T00:00:00Z",
            exit_time="2026-02-06T00:00:00Z",
            net_r=-1.0,
        ),
        _setup(
            "c",
            fold=2,
            decision="2026-03-05T00:00:00Z",
            exit_time="2026-03-06T00:00:00Z",
            net_r=0.5,
        ),
    ]
    before = _run(setups)
    mutated = deepcopy(setups)
    mutated[-1]["net_r"] = -1.0
    after = _run(mutated)

    before_models = [
        (model["fold_index"], model["model_id"])
        for model in before["models"]
    ]
    after_models = [
        (model["fold_index"], model["model_id"])
        for model in after["models"]
    ]
    assert before_models == after_models
    before_scores = [
        (row["fold_index"], row["predicted_expected_r"])
        for row in before["predictions"]
    ]
    after_scores = [
        (row["fold_index"], row["predicted_expected_r"])
        for row in after["predictions"]
    ]
    assert before_scores == after_scores


def test_purge_and_primary_policy_prevent_label_leakage_and_duplication():
    setups = [
        _setup(
            "mature",
            fold=0,
            decision="2026-01-05T00:00:00Z",
            exit_time="2026-01-06T00:00:00Z",
            net_r=1.0,
        ),
        _setup(
            "purged",
            fold=0,
            decision="2026-01-31T12:00:00Z",
            exit_time="2026-01-31T13:00:00Z",
            net_r=1.0,
        ),
        _setup(
            "duplicate-policy",
            fold=0,
            decision="2026-01-05T00:00:00Z",
            exit_time="2026-01-06T00:00:00Z",
            net_r=1.7,
            policy="tp-first",
        ),
        _setup(
            "test",
            fold=1,
            decision="2026-02-05T00:00:00Z",
            exit_time="2026-02-06T00:00:00Z",
            net_r=-1.0,
        ),
    ]

    result = _run(setups)
    fold_one = next(
        model for model in result["models"] if model["fold_index"] == 1
    )

    assert fold_one["train_setups"] == 1
    assert fold_one["status"] == "FIT"
