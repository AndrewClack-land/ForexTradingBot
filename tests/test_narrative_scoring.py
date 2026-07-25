from __future__ import annotations

import json

import pytest

from core.narrative_scoring import (
    FACTOR_VECTOR_SCHEMA,
    build_factor_vector,
    rescore_factor_vector,
)


def _votes():
    return {
        "h1_premium_discount": {
            "present": True,
            "side": "LONG",
            "evidence": {"position": "DISCOUNT"},
        },
        "false_breakout_4h": {
            "present": True,
            "side": "SHORT",
            "evidence": {"bars_ago": 2},
        },
        "true_breakout_15m": {
            "present": True,
            "side": "LONG",
            "evidence": {"bars_ago": 0},
        },
    }


def test_factor_vector_reproduces_configured_scores_and_absence():
    vector = build_factor_vector(_votes(), base_margin=1)

    assert vector["schema"] == FACTOR_VECTOR_SCHEMA
    assert vector["score_long"] == 3
    assert vector["score_short"] == 2
    assert vector["bias"] == "LONG"
    factors = {row["key"]: row for row in vector["factors"]}
    assert factors["h1_premium_discount"]["long_contribution"] == 2
    assert factors["false_breakout_4h"]["short_contribution"] == 2
    assert factors["order_block_1h"]["present"] is False
    assert factors["order_block_1h"]["vote_side"] == "NEUTRAL"


def test_pivotal_flag_uses_the_same_margin_as_production_bias():
    vector = build_factor_vector(_votes(), base_margin=1)
    factors = {row["key"]: row for row in vector["factors"]}

    assert factors["true_breakout_15m"]["pivotal_without_factor"] is True
    assert factors["true_breakout_15m"]["bias_without_factor"] == "NEUTRAL"
    assert factors["false_breakout_4h"]["pivotal_without_factor"] is False
    assert factors["false_breakout_4h"]["bias_without_factor"] == "LONG"


def test_opposing_fvg_changes_margin_not_factor_votes():
    neutral = build_factor_vector(_votes(), base_margin=1)
    opposed = build_factor_vector(
        _votes(),
        base_margin=1,
        fvg_side="SHORT",
    )

    assert neutral["bias"] == "LONG"
    assert opposed["bias"] == "NEUTRAL"
    assert opposed["margin_long"] == 2
    assert [
        (row["key"], row["vote_side"])
        for row in opposed["factors"]
    ] == [
        (row["key"], row["vote_side"])
        for row in neutral["factors"]
    ]


def test_frozen_vector_can_be_rescored_without_market_data():
    vector = build_factor_vector(_votes(), base_margin=1)
    rescored = rescore_factor_vector(
        vector,
        weights={
            "h1_premium_discount": 1,
            "false_breakout_4h": 0,
            "true_breakout_15m": 1,
            "order_block_1h": 1,
            "rejection_block_1h": 1,
        },
    )

    assert rescored["score_long"] == 2
    assert rescored["score_short"] == 0
    assert rescored["bias"] == "LONG"


def test_counterfactual_weights_are_constrained_non_negative():
    vector = build_factor_vector(_votes(), base_margin=1)

    with pytest.raises(ValueError, match="non-negative"):
        rescore_factor_vector(
            vector,
            weights={"h1_premium_discount": -1},
        )


def test_short_and_fvg_tightened_pivotal_paths_are_explicit():
    short_vector = build_factor_vector(
        {
            "h1_premium_discount": {
                "present": True,
                "side": "SHORT",
            },
            "true_breakout_15m": {
                "present": True,
                "side": "LONG",
            },
        },
        base_margin=1,
    )
    short_factors = {
        row["key"]: row
        for row in short_vector["factors"]
    }
    assert short_vector["bias"] == "SHORT"
    assert short_factors["h1_premium_discount"][
        "pivotal_without_factor"
    ] is True
    assert short_factors["h1_premium_discount"][
        "bias_without_factor"
    ] == "LONG"

    fvg_vector = build_factor_vector(
        {
            "h1_premium_discount": {
                "present": True,
                "side": "LONG",
            },
            "true_breakout_15m": {
                "present": True,
                "side": "LONG",
            },
            "order_block_1h": {
                "present": True,
                "side": "SHORT",
            },
        },
        base_margin=1,
        fvg_side="SHORT",
    )
    fvg_factors = {
        row["key"]: row
        for row in fvg_vector["factors"]
    }
    assert fvg_vector["bias"] == "LONG"
    assert fvg_vector["margin_long"] == 2
    assert fvg_factors["true_breakout_15m"][
        "pivotal_without_factor"
    ] is True


def test_neutral_tie_and_strict_json_serialization():
    vector = build_factor_vector(
        {
            "h1_premium_discount": {
                "present": True,
                "side": "LONG",
            },
            "false_breakout_4h": {
                "present": True,
                "side": "SHORT",
            },
        },
        base_margin=1,
    )

    assert vector["score_long"] == vector["score_short"] == 2
    assert vector["bias"] == "NEUTRAL"
    assert all(
        row["selected_side_contribution"] == 0
        for row in vector["factors"]
    )
    json.dumps(vector, allow_nan=False)
