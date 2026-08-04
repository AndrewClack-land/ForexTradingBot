from __future__ import annotations

import json
import types

import numpy as np
import pytest

import main
from backtest.optimizer import (
    SHADOW_SCORE_SCHEMA,
    _feature_names,
    _feature_vector,
)
from backtest.strategy_runner import _trigger_kind as offline_trigger_kind
from core.shadow_score import (
    LiveShadowScorer,
    normalize_trigger_kind,
    tp1_em_ratio,
)


_SCALERS = {
    "vol_mean": 30.0,
    "vol_scale": 10.0,
    "ratio_mean": 0.5,
    "ratio_scale": 0.25,
}


def _write_models(tmp_path, *, status="FIT", symbols=("EURUSD", "GBPUSD")):
    names = _feature_names(symbols)
    coefficients = {
        name: round(0.013 * (index + 1) - 0.2, 5)
        for index, name in enumerate(names)
    }
    model = {
        "schema": SHADOW_SCORE_SCHEMA,
        "fold_index": 2,
        "status": status,
        "test_end": "2026-01-01T00:00:00+00:00",
        "feature_names": list(names),
        "model_id": "shadowmodel0001" if status == "FIT" else None,
        "intercept": 0.25,
        "coefficients": coefficients,
        **_SCALERS,
    }
    path = tmp_path / "shadow_models.json"
    path.write_text(json.dumps({"models": [model]}), encoding="utf-8")
    return path, names, model


def _signal(**overrides):
    factor_vector = {
        "fvg_side": "SHORT",
        "factors": [
            {"key": "h1_premium_discount", "present": True, "vote_side": "LONG"},
            {"key": "false_breakout_4h", "present": True, "vote_side": "LONG"},
            {"key": "true_breakout_15m", "present": False, "vote_side": "NEUTRAL"},
            {"key": "order_block_1h", "present": True, "vote_side": "SHORT"},
            {"key": "rejection_block_1h", "present": False, "vote_side": "NEUTRAL"},
        ],
    }
    signal = {
        "signal": "ENTER",
        "side": "LONG",
        "entry_price": 1.1000,
        "stop_price": 1.0950,
        "tp_prices": [1.1050, 1.1100, 1.1150],
        "factor_vector": factor_vector,
        "trigger_reason": "OrderBlock touch LONG (H1) zone=1.09500-1.10000",
        "vol_R": 42.0,
        "vol_em_1d": 0.01,
    }
    signal.update(overrides)
    return signal


# --------------------------------------------------------------------------
# Trigger vocabulary parity with the offline runner
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "signal",
    [
        {"trigger_reason": "RejectionBlock 15M BEAR | intrusion=7.0%"},
        {"trigger_reason": "TurtleSoup 15M reclaim lvl=1.10000 sweep=0.050%"},
        {"trigger_reason": "H1 PivotHigh reclaim on 15M | lvl=1.10000"},
        {"trigger_reason": "H1 PivotLow reclaim on 15M | lvl=1.10000"},
        {"trigger_reason": "OrderBlock touch LONG (H1) zone=1.09-1.10"},
        {"trigger_reason": "Absorption 15M something"},
        {"trigger_reason": "totally unrecognised reason"},
        {"trigger_reason": ""},
        {},
        {"trigger_kind": "fxpro_cluster_rejection_15m"},
        {"trigger_kind": "fxpro_quote_pressure_rejection_15m"},
        {"trigger_kind": "order_block_1h"},
        {"trigger_kind": "not_a_real_kind", "trigger_reason": "OrderBlock touch X"},
    ],
)
def test_normalize_trigger_kind_matches_offline_runner(signal):
    """The live copy must never drift from backtest.strategy_runner."""

    assert normalize_trigger_kind(signal) == offline_trigger_kind(signal)


# --------------------------------------------------------------------------
# Feature parity
# --------------------------------------------------------------------------

def test_tp1_em_ratio_matches_offline_definition():
    # Offline: min |target - entry| / em_1d
    assert tp1_em_ratio(1.1000, [1.1050, 1.1100, 1.1150], 0.01) == pytest.approx(0.5)
    assert tp1_em_ratio(1.1000, [], 0.01) is None
    assert tp1_em_ratio(1.1000, [1.1050], 0.0) is None
    assert tp1_em_ratio(None, [1.1050], 0.01) is None
    assert tp1_em_ratio(1.1000, ["nonsense"], 0.01) is None


def test_live_score_reproduces_offline_prediction(tmp_path):
    path, names, model = _write_models(tmp_path)
    scorer = LiveShadowScorer.load(path)
    assert scorer is not None

    signal = _signal()
    result = scorer.score(signal, symbol="EURUSD")

    row = {
        "side": "LONG",
        "symbol": "EURUSD",
        "trigger_kind": "order_block_1h",
        "factor_vector": signal["factor_vector"],
        "vol_r": 42.0,
        "vol_tp1_em_ratio": tp1_em_ratio(1.1000, signal["tp_prices"], 0.01),
    }
    vector = _feature_vector(row, names=names, **_SCALERS)
    expected = model["intercept"] + float(
        vector @ np.asarray([model["coefficients"][name] for name in names])
    )

    assert result["shadow_score"] == pytest.approx(round(expected, 4))
    assert result["shadow_model_id"] == "shadowmodel0001"
    assert result["shadow_executing"] is False


def test_missing_vol_context_uses_model_missing_indicators(tmp_path):
    """No volatility context must not fabricate a zero observation."""

    path, names, model = _write_models(tmp_path)
    scorer = LiveShadowScorer.load(path)

    signal = _signal()
    signal.pop("vol_R")
    signal.pop("vol_em_1d")
    result = scorer.score(signal, symbol="EURUSD")

    row = {
        "side": "LONG",
        "symbol": "EURUSD",
        "trigger_kind": "order_block_1h",
        "factor_vector": signal["factor_vector"],
        "vol_r": None,
        "vol_tp1_em_ratio": None,
    }
    vector = _feature_vector(row, names=names, **_SCALERS)
    expected = model["intercept"] + float(
        vector @ np.asarray([model["coefficients"][name] for name in names])
    )
    assert result["shadow_score"] == pytest.approx(round(expected, 4))


def test_fxpro_triggers_are_flagged_out_of_vocabulary(tmp_path):
    """The fitted vocabulary predates the FxPro proxies."""

    path, _, _ = _write_models(tmp_path)
    scorer = LiveShadowScorer.load(path)

    cluster = scorer.score(
        _signal(trigger_kind="fxpro_cluster_rejection_15m"),
        symbol="EURUSD",
    )
    assert cluster["shadow_trigger_kind"] == "fxpro_cluster_rejection_15m"
    assert cluster["shadow_trigger_in_vocab"] is False

    quote = scorer.score(
        _signal(trigger_kind="fxpro_quote_pressure_rejection_15m"),
        symbol="EURUSD",
    )
    assert quote["shadow_trigger_in_vocab"] is False

    order_block = scorer.score(_signal(), symbol="EURUSD")
    assert order_block["shadow_trigger_kind"] == "order_block_1h"
    assert order_block["shadow_trigger_in_vocab"] is True


def test_baseline_trigger_level_counts_as_in_vocabulary(tmp_path):
    """The dropped reference level has no column but was still trained on."""

    path, names, _ = _write_models(tmp_path)
    scorer = LiveShadowScorer.load(path)
    assert "trigger:h1_pivot_reclaim_15m" not in names

    result = scorer.score(
        _signal(trigger_reason="H1 PivotLow reclaim on 15M | lvl=1.10000"),
        symbol="EURUSD",
    )
    assert result["shadow_trigger_kind"] == "h1_pivot_reclaim_15m"
    assert result["shadow_trigger_in_vocab"] is True


# --------------------------------------------------------------------------
# Fail-open loading
# --------------------------------------------------------------------------

def test_loader_fails_open(tmp_path):
    assert LiveShadowScorer.load(None) is None
    assert LiveShadowScorer.load(tmp_path / "absent.json") is None

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert LiveShadowScorer.load(broken) is None

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"models": []}), encoding="utf-8")
    assert LiveShadowScorer.load(empty) is None

    not_fit, _, _ = _write_models(tmp_path, status="NOT_FIT")
    assert LiveShadowScorer.load(not_fit) is None


def test_loader_rejects_foreign_schema(tmp_path):
    path, _, _ = _write_models(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["models"][0]["schema"] = "some-other-model/v9"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert LiveShadowScorer.load(path) is None


def test_loader_rejects_non_finite_coefficients(tmp_path):
    path, names, _ = _write_models(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["models"][0]["coefficients"][names[0]] = "not-a-number"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert LiveShadowScorer.load(path) is None


def test_loader_picks_newest_fitted_fold(tmp_path):
    path, names, model = _write_models(tmp_path)
    older = dict(model)
    older["fold_index"] = 1
    older["test_end"] = "2025-01-01T00:00:00+00:00"
    older["model_id"] = "olderfold000001"
    payload = {"models": [model, older]}
    path.write_text(json.dumps(payload), encoding="utf-8")

    scorer = LiveShadowScorer.load(path)
    assert scorer.model_id == "shadowmodel0001"
    assert scorer.fold_index == 2


# --------------------------------------------------------------------------
# Non-executing guarantee
# --------------------------------------------------------------------------

def _stub_core(scorer):
    return types.SimpleNamespace(shadow_scorer=scorer)


def test_attach_shadow_score_only_adds_shadow_keys(tmp_path):
    path, _, _ = _write_models(tmp_path)
    core = _stub_core(LiveShadowScorer.load(path))

    signal = _signal()
    before = dict(signal)
    result = main.Core._attach_shadow_score(core, "EURUSD", signal)

    for key, value in before.items():
        assert result[key] == value, f"{key} was mutated by the shadow score"
    added = set(result) - set(before)
    assert added
    assert all(key.startswith("shadow_") for key in added)


def test_attach_shadow_score_cannot_overwrite_execution_fields(tmp_path):
    path, _, _ = _write_models(tmp_path)
    scorer = LiveShadowScorer.load(path)

    def hostile(signal, *, symbol):
        return {
            "signal": "AI_REJECT",
            "side": "SHORT",
            "entry_price": 9.9,
            "stop_price": 9.8,
            "tp_prices": [1.0],
            "shadow_score": 1.23,
        }

    scorer.score = hostile
    core = _stub_core(scorer)

    signal = _signal()
    result = main.Core._attach_shadow_score(core, "EURUSD", signal)

    assert result["signal"] == "ENTER"
    assert result["side"] == "LONG"
    assert result["entry_price"] == 1.1000
    assert result["stop_price"] == 1.0950
    assert result["tp_prices"] == [1.1050, 1.1100, 1.1150]
    assert result["shadow_score"] == 1.23


def test_attach_shadow_score_survives_a_broken_model(tmp_path):
    path, _, _ = _write_models(tmp_path)
    scorer = LiveShadowScorer.load(path)

    def explode(signal, *, symbol):
        raise RuntimeError("model blew up")

    scorer.score = explode
    core = _stub_core(scorer)

    signal = _signal()
    result = main.Core._attach_shadow_score(core, "EURUSD", signal)
    assert result == signal


def test_attach_shadow_score_is_a_noop_without_a_model():
    core = _stub_core(None)
    signal = _signal()
    assert main.Core._attach_shadow_score(core, "EURUSD", signal) is signal


@pytest.mark.parametrize(
    "state",
    ["NO_TRIGGER", "NO_TREND", "AI_REJECT", "WAIT_SESSION", "SKIP_VOL_REGIME"],
)
def test_only_enter_signals_are_annotated(tmp_path, state):
    path, _, _ = _write_models(tmp_path)
    core = _stub_core(LiveShadowScorer.load(path))

    signal = _signal(signal=state)
    result = main.Core._attach_shadow_score(core, "EURUSD", signal)
    assert result is signal
    assert not any(key.startswith("shadow_") for key in result)
