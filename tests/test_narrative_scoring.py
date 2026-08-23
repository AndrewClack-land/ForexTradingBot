from __future__ import annotations

import json

import pytest

from core.narrative_scoring import (
    FACTOR_CONTRACT_FVG_VOTE,
    FACTOR_CONTRACT_LEGACY,
    FACTOR_VECTOR_SCHEMA,
    build_factor_vector,
    rescore_factor_vector,
    resolve_factor_contract,
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


def _contract_votes(fvg_side: str):
    """H1 P/D and the FVG regime disagree; nothing else votes."""

    return {
        "h1_premium_discount": {"present": True, "side": "LONG"},
        "fvg_regime_1h": {
            "present": fvg_side in {"LONG", "SHORT"},
            "side": fvg_side,
        },
    }


def test_legacy_contract_keeps_the_fvg_regime_out_of_the_score():
    contract = resolve_factor_contract(FACTOR_CONTRACT_LEGACY)
    vector = build_factor_vector(
        _contract_votes("SHORT"),
        base_margin=2,
        fvg_side="SHORT",
        weights=contract["weights"],
        fvg_margin_enabled=contract["fvg_margin_enabled"],
    )

    rows = {row["key"]: row for row in vector["factors"]}
    # The regime is recorded but contributes nothing to either score.
    assert rows["fvg_regime_1h"]["vote_side"] == "SHORT"
    assert rows["fvg_regime_1h"]["effective_weight"] == 0
    assert vector["score_long"] == 2
    assert vector["score_short"] == 0
    # It brakes the opposing side by raising that margin instead.
    assert vector["margin_long"] == 3
    assert vector["margin_short"] == 2
    assert vector["bias"] == "NEUTRAL"


def test_fvg_vote_contract_scores_the_regime_and_keeps_margins_symmetric():
    contract = resolve_factor_contract(FACTOR_CONTRACT_FVG_VOTE)
    vector = build_factor_vector(
        _contract_votes("SHORT"),
        base_margin=2,
        fvg_side="SHORT",
        weights=contract["weights"],
        fvg_margin_enabled=contract["fvg_margin_enabled"],
    )

    rows = {row["key"]: row for row in vector["factors"]}
    assert rows["fvg_regime_1h"]["effective_weight"] == 1
    # H1 P/D drops to 1, so a disagreeing regime now cancels it exactly.
    assert rows["h1_premium_discount"]["effective_weight"] == 1
    assert vector["score_long"] == 1
    assert vector["score_short"] == 1
    assert vector["margin_long"] == vector["margin_short"] == 2
    assert vector["fvg_margin_enabled"] is False
    assert vector["bias"] == "NEUTRAL"


def test_challenger_weights_are_the_agreed_contract():
    contract = resolve_factor_contract(FACTOR_CONTRACT_FVG_VOTE)

    assert contract["weights"] == {
        "h1_premium_discount": 1,
        "false_breakout_4h": 2,
        "true_breakout_15m": 1,
        "false_breakout_1h": 1,
        "true_breakout_1h": 1,
        "order_block_1h": 2,
        "rejection_block_1h": 1,
        "fvg_regime_1h": 1,
    }
    legacy = resolve_factor_contract(FACTOR_CONTRACT_LEGACY)["weights"]
    assert legacy == {
        "h1_premium_discount": 2,
        "false_breakout_4h": 2,
        "true_breakout_15m": 1,
        "false_breakout_1h": 1,
        "true_breakout_1h": 1,
        "order_block_1h": 1,
        "rejection_block_1h": 1,
        "fvg_regime_1h": 0,
    }
    # The 1H breakout rows are live votes, so both arms carry them at +1 and
    # the arms still differ only by the FVG row's promotion to a vote.
    assert sum(legacy.values()) == 9
    assert sum(contract["weights"].values()) == 10


def test_unknown_factor_contract_fails_closed():
    with pytest.raises(ValueError, match="unknown factor contract"):
        resolve_factor_contract("v3-guesswork")


def test_rescore_preserves_the_margin_rule_of_the_frozen_vector():
    contract = resolve_factor_contract(FACTOR_CONTRACT_FVG_VOTE)
    vector = build_factor_vector(
        _contract_votes("SHORT"),
        base_margin=2,
        fvg_side="SHORT",
        weights=contract["weights"],
        fvg_margin_enabled=contract["fvg_margin_enabled"],
    )

    rescored = rescore_factor_vector(vector, weights=contract["weights"])

    # A frozen challenger vector must not silently regain the margin penalty.
    assert rescored["fvg_margin_enabled"] is False
    assert rescored["margin_long"] == rescored["margin_short"] == 2
    assert rescored["score_long"] == vector["score_long"]
    assert rescored["score_short"] == vector["score_short"]


def _veto_stub(bias: str, fvg_side: str):
    """Real generate_signal, stubbed collaborators, so only the veto is tested."""
    import pandas as pd

    from core.strategy_narrative import NarrativeStrategy

    strat = NarrativeStrategy.__new__(NarrativeStrategy)
    strat.fvg_veto_enabled = True
    strat._last_factor_vector = None
    strat._last_htf_context = None
    strat.rejection_block_h1_entry_enabled = False
    strat.calc_narrative = lambda *a: (bias, "narrative")
    strat.calc_fvg_regime_1h = lambda _df: (fvg_side, f"FVG {fvg_side}")
    for name in (
        "trigger_15m_cluster_rejection",
        "trigger_15m_quote_pressure_rejection",
        "trigger_h1_pivot_reclaim_on_15m",
        "trigger_orderblock_touch",
    ):
        setattr(strat, name, lambda *a, **k: None)
    frame = pd.DataFrame({"close": [1.0] * 5})
    return strat.generate_signal({"4H": frame, "1H": frame, "15M": frame})


def test_fvg_veto_blocks_only_the_opposing_direction():
    opposed = _veto_stub(bias="LONG", fvg_side="SHORT")
    assert opposed["signal"] == "NO_TREND"
    assert "FVG-вето" in opposed["info"]

    # Aligned: the veto must not fire, so the chain reaches the triggers and
    # falls through to NO_TRIGGER instead of being refused up front.
    aligned = _veto_stub(bias="SHORT", fvg_side="SHORT")
    assert aligned["signal"] == "NO_TRIGGER"

    # A neutral regime never vetoes either.
    neutral = _veto_stub(bias="LONG", fvg_side="NEUTRAL")
    assert neutral["signal"] == "NO_TRIGGER"


def test_fvg_veto_is_off_by_default():
    import config

    assert config.FVG_VETO_ENABLED is False
