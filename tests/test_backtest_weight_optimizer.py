from __future__ import annotations

from copy import deepcopy
import json
import math

import pandas as pd
import pytest

from backtest.weight_optimizer import (
    REFERENCE_WEIGHTS,
    WEIGHT_MAX,
    WEIGHT_MIN,
    WEIGHT_SUM,
    WeightOptimizerError,
    build_counterfactual_weight_scores,
)
from core.narrative_scoring import build_factor_vector


def _vector(votes: dict[str, str]):
    return build_factor_vector(
        {
            key: {
                "present": True,
                "side": side,
            }
            for key, side in votes.items()
        },
        base_margin=2,
    )


def _row(
    opportunity_id: str,
    *,
    decision_time: str,
    side: str = "LONG",
    votes: dict[str, str] | None = None,
    label_status: str = "FILLED",
    opportunity_r: float | None = 1.0,
    exit_time: str | None = None,
    symbol: str = "EURUSD",
    trigger_kind: str = "h1_pivot_reclaim_15m",
):
    if exit_time is None and label_status in {"FILLED", "NO_FILL"}:
        exit_time = (
            pd.Timestamp(decision_time) + pd.Timedelta(hours=1)
        ).isoformat()
    return {
        "opportunity_id": opportunity_id,
        "decision_event_id": f"event-{opportunity_id}",
        "symbol": symbol,
        "decision_time": decision_time,
        "exit_time": exit_time,
        "side": side,
        "trigger_kind": trigger_kind,
        "label_status": label_status,
        "opportunity_r": opportunity_r,
        "factor_vector": _vector(
            votes or {"h1_premium_discount": side}
        ),
    }


def _periods():
    return [
        {
            "fold_index": 0,
            "train_start": "2026-01-01T00:00:00Z",
            "train_end": "2026-02-01T00:00:00Z",
            "test_start": "2026-02-01T00:00:00Z",
            "test_end": "2026-03-01T00:00:00Z",
        }
    ]


def _run(rows, **overrides):
    return build_counterfactual_weight_scores(
        opportunities=rows,
        periods=_periods(),
        entry_ttl=overrides.pop("entry_ttl", "1D"),
        max_holding=overrides.pop("max_holding", "1D"),
        ridge_alpha=overrides.pop("ridge_alpha", 0.1),
        min_train_opportunities=overrides.pop(
            "min_train_opportunities",
            2,
        ),
        **overrides,
    )


def test_uses_only_mature_purged_train_rows_and_scores_every_oos_row():
    rows = [
        _row(
            "train-filled",
            decision_time="2026-01-05T00:00:00Z",
            votes={"h1_premium_discount": "LONG"},
            opportunity_r=1.0,
            trigger_kind="absorption_15m",
        ),
        _row(
            "train-no-fill",
            decision_time="2026-01-10T00:00:00Z",
            side="SHORT",
            votes={"true_breakout_15m": "SHORT"},
            label_status="NO_FILL",
            opportunity_r=0.0,
            trigger_kind="order_block_1h",
        ),
        _row(
            "purged",
            decision_time="2026-01-30T00:00:00Z",
            opportunity_r=1.5,
        ),
        _row(
            "label-matures-late",
            decision_time="2026-01-20T00:00:00Z",
            exit_time="2026-02-01T00:00:00Z",
            opportunity_r=-1.0,
        ),
        _row(
            "oos-aligned",
            decision_time="2026-02-05T00:00:00Z",
            votes={"h1_premium_discount": "LONG"},
            opportunity_r=0.5,
        ),
        _row(
            "oos-opposed",
            decision_time="2026-02-06T00:00:00Z",
            side="SHORT",
            votes={"h1_premium_discount": "LONG"},
            label_status="NO_FILL",
            opportunity_r=0.0,
            trigger_kind="absorption_15m",
        ),
        _row(
            "oos-pending",
            decision_time="2026-02-07T00:00:00Z",
            label_status="PENDING",
            opportunity_r=None,
            exit_time=None,
        ),
    ]

    result = _run(rows)

    model = result["models"][0]
    assert model["status"] == "FIT"
    assert model["train_opportunities"] == 2
    assert model["test_opportunities"] == 3
    assert model["purge_cutoff"] == "2026-01-30T00:00:00+00:00"
    assert {
        row["opportunity_id"] for row in result["predictions"]
    } == {"oos-aligned", "oos-opposed", "oos-pending"}
    assert all(
        row["predicted_opportunity_r"] is not None
        for row in result["predictions"]
    )
    assert all(
        row["score_threshold_applied"] is False
        for row in result["predictions"]
    )
    by_id = {
        row["opportunity_id"]: row
        for row in result["predictions"]
    }
    assert by_id["oos-aligned"]["weighted_factor_score"] > 0
    assert by_id["oos-opposed"]["weighted_factor_score"] < 0
    assert by_id["oos-pending"]["actual_opportunity_r"] is None
    assert result["summary"]["no_hard_threshold"] is True
    assert (
        result["summary"]["execution_side_effects_in_this_module"]
        is False
    )
    assert result["summary"]["risk_changed_by_score"] is False


def test_weights_are_constrained_regularized_and_canonical():
    rows = []
    start = pd.Timestamp("2026-01-02T00:00:00Z")
    factor_keys = tuple(REFERENCE_WEIGHTS)
    for index in range(25):
        factor = factor_keys[index % len(factor_keys)]
        rows.append(
            _row(
                f"train-{index:02d}",
                decision_time=(
                    start + pd.Timedelta(hours=24 * index)
                ).isoformat(),
                votes={factor: "LONG"},
                opportunity_r=(
                    1.5
                    if factor == "h1_premium_discount"
                    else -1.0
                ),
                trigger_kind=(
                    "absorption_15m"
                    if index % 2
                    else "order_block_1h"
                ),
            )
        )
    rows.extend(
        [
            _row(
                "oos-long",
                decision_time="2026-02-05T00:00:00Z",
                votes={"h1_premium_discount": "LONG"},
            ),
            _row(
                "oos-short",
                decision_time="2026-02-06T00:00:00Z",
                side="SHORT",
                votes={"h1_premium_discount": "LONG"},
            ),
        ]
    )

    result = _run(
        rows,
        ridge_alpha=0.001,
        min_train_opportunities=10,
    )
    permuted = _run(
        list(reversed(deepcopy(rows))),
        ridge_alpha=0.001,
        min_train_opportunities=10,
    )

    assert result == permuted
    json.dumps(result, allow_nan=False)
    model = result["models"][0]
    weights = list(model["weights"].values())
    assert model["status"] == "FIT"
    assert model["converged"] is True
    assert all(math.isfinite(value) for value in weights)
    assert all(
        WEIGHT_MIN <= value <= WEIGHT_MAX
        for value in weights
    )
    assert sum(weights) == pytest.approx(WEIGHT_SUM, abs=1e-10)
    assert any(
        not math.isclose(
            model["weights"][key],
            REFERENCE_WEIGHTS[key],
            abs_tol=1e-6,
        )
        for key in REFERENCE_WEIGHTS
    )
    assert len(result["coefficients"]) == 6


def test_oos_outcomes_cannot_change_frozen_model_or_forecast():
    rows = [
        _row(
            "train-a",
            decision_time="2026-01-05T00:00:00Z",
            votes={"h1_premium_discount": "LONG"},
            opportunity_r=1.0,
        ),
        _row(
            "train-b",
            decision_time="2026-01-10T00:00:00Z",
            side="SHORT",
            votes={"false_breakout_4h": "LONG"},
            opportunity_r=-1.0,
        ),
        _row(
            "test",
            decision_time="2026-02-05T00:00:00Z",
            votes={"true_breakout_15m": "LONG"},
            opportunity_r=1.5,
        ),
    ]
    before = _run(rows)
    mutated = deepcopy(rows)
    mutated[-1]["opportunity_r"] = -1.0
    after = _run(mutated)

    assert before["models"] == after["models"]
    assert before["coefficients"] == after["coefficients"]

    def forecasts(result):
        return [
            {
                "opportunity_id": row["opportunity_id"],
                "model_id": row["model_id"],
                "weighted_factor_score": row[
                    "weighted_factor_score"
                ],
                "predicted_opportunity_r": row[
                    "predicted_opportunity_r"
                ],
            }
            for row in result["predictions"]
        ]

    assert forecasts(before) == forecasts(after)
    assert (
        before["predictions"][0]["actual_opportunity_r"]
        != after["predictions"][0]["actual_opportunity_r"]
    )


def test_each_outer_fold_uses_its_own_train_interval_and_stays_frozen():
    periods = [
        *_periods(),
        {
            "fold_index": 1,
            "train_start": "2026-02-01T00:00:00Z",
            "train_end": "2026-03-01T00:00:00Z",
            "test_start": "2026-03-01T00:00:00Z",
            "test_end": "2026-04-01T00:00:00Z",
        },
    ]
    rows = [
        _row(
            "january-a",
            decision_time="2026-01-05T00:00:00Z",
            votes={"h1_premium_discount": "LONG"},
            opportunity_r=1.0,
        ),
        _row(
            "january-b",
            decision_time="2026-01-10T00:00:00Z",
            votes={"false_breakout_4h": "LONG"},
            opportunity_r=-1.0,
        ),
        _row(
            "february-a",
            decision_time="2026-02-05T00:00:00Z",
            votes={"true_breakout_15m": "LONG"},
            opportunity_r=1.0,
        ),
        _row(
            "february-b",
            decision_time="2026-02-10T00:00:00Z",
            votes={"order_block_1h": "LONG"},
            opportunity_r=-1.0,
        ),
        _row(
            "march-oos",
            decision_time="2026-03-05T00:00:00Z",
            votes={"rejection_block_1h": "LONG"},
            opportunity_r=0.5,
        ),
    ]

    before = build_counterfactual_weight_scores(
        opportunities=rows,
        periods=periods,
        entry_ttl="1D",
        max_holding="1D",
        ridge_alpha=0.01,
        min_train_opportunities=2,
    )
    mutated = deepcopy(rows)
    mutated[2]["opportunity_r"] = -1.0
    after = build_counterfactual_weight_scores(
        opportunities=mutated,
        periods=periods,
        entry_ttl="1D",
        max_holding="1D",
        ridge_alpha=0.01,
        min_train_opportunities=2,
    )
    before_models = {
        model["fold_index"]: model
        for model in before["models"]
    }
    after_models = {
        model["fold_index"]: model
        for model in after["models"]
    }

    assert before_models[0] == after_models[0]
    assert (
        before_models[1]["model_id"]
        != after_models[1]["model_id"]
    )
    fold_zero_before = [
        row["predicted_opportunity_r"]
        for row in before["predictions"]
        if row["fold_index"] == 0
    ]
    fold_zero_after = [
        row["predicted_opportunity_r"]
        for row in after["predictions"]
        if row["fold_index"] == 0
    ]
    assert fold_zero_before == fold_zero_after


def test_sparse_fold_freezes_reference_but_does_not_emit_fake_prediction():
    rows = [
        _row(
            "only-train",
            decision_time="2026-01-05T00:00:00Z",
        ),
        _row(
            "test-aligned",
            decision_time="2026-02-05T00:00:00Z",
            votes={"h1_premium_discount": "LONG"},
        ),
        _row(
            "test-opposed",
            decision_time="2026-02-06T00:00:00Z",
            side="SHORT",
            votes={"h1_premium_discount": "LONG"},
        ),
    ]

    result = _run(rows, min_train_opportunities=2)

    model = result["models"][0]
    assert model["status"] == "NOT_FIT"
    assert model["weights"] == REFERENCE_WEIGHTS
    assert len(result["predictions"]) == 2
    assert all(
        row["predicted_opportunity_r"] is None
        for row in result["predictions"]
    )
    scores = {
        row["opportunity_id"]: row["weighted_factor_score"]
        for row in result["predictions"]
    }
    assert scores["test-aligned"] == 2.0
    assert scores["test-opposed"] == -2.0


def test_fit_readiness_counts_unique_decision_side_clusters():
    rows = [
        _row(
            f"same-decision-trigger-{index}",
            decision_time="2026-01-05T00:00:00Z",
            trigger_kind=trigger,
        )
        for index, trigger in enumerate(
            (
                "rejection_block_15m",
                "absorption_15m",
                "h1_pivot_reclaim_15m",
                "order_block_1h",
            )
        )
    ]
    for row in rows:
        row["decision_event_id"] = "one-m15-decision"
    rows.append(
        _row(
            "test",
            decision_time="2026-02-05T00:00:00Z",
        )
    )

    result = _run(rows, min_train_opportunities=2)

    model = result["models"][0]
    assert model["status"] == "NOT_FIT"
    assert model["train_opportunities"] == 4
    assert model["train_decision_side_clusters"] == 1
    assert model["train_effective_sample_size"] == pytest.approx(4.0)


def test_oos_actual_is_hidden_when_exit_is_after_fold_boundary():
    rows = [
        _row(
            "train-a",
            decision_time="2026-01-05T00:00:00Z",
        ),
        _row(
            "train-b",
            decision_time="2026-01-10T00:00:00Z",
            side="SHORT",
        ),
        _row(
            "test-late-exit",
            decision_time="2026-02-28T23:00:00Z",
            exit_time="2026-03-01T00:30:00Z",
            opportunity_r=2.0,
        ),
    ]

    prediction = _run(rows)["predictions"][0]

    assert prediction["actual_label_status"] == "FILLED"
    assert prediction["actual_label_known_before_test_end"] is False
    assert prediction["actual_opportunity_r"] is None


def test_row_contract_rejects_bad_no_fill_and_duplicate_identity():
    bad_no_fill = _row(
        "bad",
        decision_time="2026-01-05T00:00:00Z",
        label_status="NO_FILL",
        opportunity_r=-1.0,
    )
    with pytest.raises(
        WeightOptimizerError,
        match="NO_FILL opportunity_r must be zero",
    ):
        _run([bad_no_fill], min_train_opportunities=1)

    duplicate = _row(
        "same",
        decision_time="2026-01-05T00:00:00Z",
    )
    with pytest.raises(
        WeightOptimizerError,
        match="duplicate opportunity_id",
    ):
        _run(
            [duplicate, deepcopy(duplicate)],
            min_train_opportunities=1,
        )
