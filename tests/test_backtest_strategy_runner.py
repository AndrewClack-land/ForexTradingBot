from __future__ import annotations

import json
from datetime import time
from pathlib import Path

import pandas as pd
import pytest

from backtest.data import HistoricalDataset
from backtest.strategy_runner import (
    NarrativeBacktestConfig,
    StrategyBacktestError,
    _EMPTY_CSV_FIELDS,
    _REQUIRED_RELEASE_FILES,
    _configure_strategy,
    _default_strategy_factory,
    _force_close_outcome,
    _metrics_for,
    _sha256_file,
    _time_in_window,
    _trigger_kind,
    _trigger_signature,
    _validate_factor_vector,
    run_narrative_backtest,
    verify_release_manifest,
)
from backtest.simulator import simulate_split_outcome
from core.narrative_scoring import build_factor_vector


def _frame(index, rows):
    return pd.DataFrame(
        rows,
        columns=["open", "high", "low", "close"],
        index=pd.DatetimeIndex(index, name="timestamp"),
    )


def _dataset(tmp_path, *, collision=False, future_close=999.0):
    first_low = 89.0 if collision else 99.0
    frames = {
        ("EURUSD", "1d"): _frame(
            ["2025-12-31T00:00:00Z"],
            [(100.0, 101.0, 99.0, 100.0)],
        ),
        ("EURUSD", "4h"): _frame(
            ["2025-12-31T20:00:00Z"],
            [(100.0, 101.0, 99.0, 100.0)],
        ),
        ("EURUSD", "1h"): _frame(
            ["2025-12-31T23:00:00Z"],
            [(100.0, 101.0, 99.0, 100.0)],
        ),
        ("EURUSD", "15m"): _frame(
            [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:15:00Z",
            ],
            [
                (100.0, 101.0, 99.0, 100.0),
                (future_close, future_close, future_close, future_close),
            ],
        ),
        ("EURUSD", "1m"): _frame(
            [
                "2026-01-01T00:14:00Z",
                "2026-01-01T00:15:00Z",
                "2026-01-01T00:16:00Z",
            ],
            [
                (100.0, 101.0, 80.0, 90.0),
                (100.0, 999.0, 1.0, 108.0),
                (100.0, 111.0, first_low, 108.0),
            ],
        ),
    }
    return HistoricalDataset(
        tmp_path,
        frames,
        source_files=[],
        raw_counts={key: len(frame) for key, frame in frames.items()},
    )


class _AlwaysEnter:
    def generate_signal(self, data, symbol=""):
        assert symbol == "EURUSD"
        assert data["15M"].index[-1] == pd.Timestamp(
            "2026-01-01T00:00:00Z"
        )
        factor_vector = build_factor_vector(
            {
                "h1_premium_discount": {
                    "present": True,
                    "side": "LONG",
                    "evidence": {"position": "DISCOUNT"},
                },
                "false_breakout_4h": {
                    "present": True,
                    "side": "SHORT",
                    "evidence": {"bars_ago": 1},
                },
                "true_breakout_15m": {
                    "present": True,
                    "side": "LONG",
                    "evidence": {"bars_ago": 0},
                },
                "rejection_block_1h": {
                    "present": True,
                    "side": "LONG",
                    "evidence": {"age_bars": 2},
                },
            },
            base_margin=1,
        )
        return {
            "signal": "ENTER",
            "side": "LONG",
            "entry_min": 99.0,
            "entry_max": 101.0,
            "entry_price": 100.0,
            "stop_price": 90.0,
            "tp_prices": [110.0, 120.0, 130.0],
            "trigger_reason": "TEST",
            "narrative": "causal fixture",
            "fvg_regime": "NEUTRAL",
            "factor_vector": factor_vector,
        }


class _NeverEnter:
    def generate_signal(self, data, symbol=""):
        return {
            "signal": "NO_TRIGGER",
            "narrative": f"no trigger for {symbol}",
        }


def _config(*, policy="stop-first"):
    return NarrativeBacktestConfig.build(
        symbols=["EURUSD"],
        start="2026-01-01T00:15:00Z",
        end="2026-01-01T00:30:00Z",
        initial_capital=10_000,
        risk_fraction=0.01,
        history_limit=1,
        min_context_bars=1,
        min_daily_bars=1,
        entry_ttl="15min",
        max_holding="15min",
        intrabar_policy=policy,
        profile="signal-quality",
    )


def test_runner_is_causal_and_uses_fixed_initial_capital_risk(tmp_path):
    result = run_narrative_backtest(
        _dataset(tmp_path),
        _config(),
        strategy_factory=_AlwaysEnter,
    )

    assert len(result.setups) == 1
    setup = result.setups[0]
    assert setup["decision_time"] == "2026-01-01T00:15:00+00:00"
    assert setup["entry_time"] == "2026-01-01T00:16:00+00:00"
    assert setup["net_r"] == pytest.approx(0.9)
    assert setup["risk_amount"] == pytest.approx(100.0)
    assert setup["pnl_amount"] == pytest.approx(90.0)
    # The damaging 00:14 and already-open 00:15 bars must not affect outcome.
    assert setup["status"] == "CLOSED"


def test_runner_emits_structured_factor_attribution_without_changing_trade(
    tmp_path,
):
    result = run_narrative_backtest(
        _dataset(tmp_path),
        _config(),
        strategy_factory=_AlwaysEnter,
    )

    assert len(result.candidate_factors) == 6
    assert len(result.setup_factors) == 6
    by_factor = {
        row["factor_key"]: row
        for row in result.setup_factors
    }
    assert by_factor["h1_premium_discount"]["relation"] == "ALIGNED"
    assert by_factor["false_breakout_4h"]["relation"] == "OPPOSED"
    assert by_factor["order_block_1h"]["relation"] == "ABSENT"
    # The FVG regime is carried as an explicit row under both contracts; the
    # live contract scores it at weight 0 and moves the margin instead.
    assert by_factor["fvg_regime_1h"]["configured_weight"] == 0
    assert by_factor["true_breakout_15m"]["net_r"] == pytest.approx(0.9)
    coverage = result.summary["factor_attribution"]
    assert coverage["complete_candidate_vectors"] is True
    assert coverage["complete_setup_vectors"] is True
    assert coverage["mode"] == "conditional-production-selection"
    assert (
        coverage["candidate_population"]
        == "raw_enter_bias_plus_detected_trigger"
    )
    assert coverage["outcome_population"] == "executed_and_filled_setups"
    assert result.setups[0]["net_r"] == pytest.approx(0.9)


def test_future_candle_mutation_cannot_change_pre_cutoff_result(tmp_path):
    before = run_narrative_backtest(
        _dataset(tmp_path / "before", future_close=999.0),
        _config(),
        strategy_factory=_AlwaysEnter,
    )
    after = run_narrative_backtest(
        _dataset(tmp_path / "after", future_close=0.01),
        _config(),
        strategy_factory=_AlwaysEnter,
    )

    before_setup = dict(before.setups[0])
    after_setup = dict(after.setups[0])
    for field in ("setup_id", "candidate_id"):
        before_setup.pop(field)
        after_setup.pop(field)
    assert before_setup == after_setup


def test_both_intrabar_policies_are_independent(tmp_path):
    result = run_narrative_backtest(
        _dataset(tmp_path, collision=True),
        _config(policy="both"),
        strategy_factory=_AlwaysEnter,
    )
    by_policy = {setup["policy"]: setup for setup in result.setups}

    assert by_policy["stop-first"]["net_r"] == pytest.approx(-1.0)
    assert by_policy["tp-first"]["net_r"] == pytest.approx(0.5)
    assert by_policy["stop-first"]["ambiguous_bars"] == 1
    assert by_policy["tp-first"]["ambiguous_bars"] == 1
    assert len(result.executions) == 2
    assert {row["disposition"] for row in result.executions} == {"FILLED"}
    sensitivity = result.summary["intrabar_sensitivity"]
    assert sensitivity["matched_filled_candidates"] == 1
    assert sensitivity["matched_outcome_changed"] == 1


def test_risk_fraction_cannot_exceed_one_percent():
    with pytest.raises(StrategyBacktestError, match="cannot exceed"):
        NarrativeBacktestConfig.build(
            symbols=["EURUSD"],
            start="2026-01-01",
            end="2026-02-01",
            initial_capital=10_000,
            risk_fraction=0.0101,
        )


def test_required_factor_vector_fails_closed_and_score_mismatch_is_rejected():
    with pytest.raises(StrategyBacktestError, match="missing"):
        _validate_factor_vector(
            {"signal": "ENTER", "side": "LONG"},
            status="ENTER",
            required=True,
        )
    vector = build_factor_vector(
        {
            "h1_premium_discount": {
                "present": True,
                "side": "LONG",
            },
        },
        base_margin=1,
    )
    vector["score_long"] = 99
    with pytest.raises(StrategyBacktestError, match="do not reproduce"):
        _validate_factor_vector(
            {
                "signal": "ENTER",
                "side": "LONG",
                "factor_vector": vector,
            },
            status="ENTER",
            required=True,
        )


@pytest.mark.parametrize(
    "tamper",
    [
        lambda vector: vector.update({"margin_long": 99}),
        lambda vector: vector["factors"][0].update(
            {"pivotal_without_factor": False}
        ),
        lambda vector: vector["factors"].append(
            dict(vector["factors"][-1])
        ),
    ],
)
def test_required_factor_vector_rejects_tampered_derived_fields(tamper):
    vector = build_factor_vector(
        {
            "h1_premium_discount": {
                "present": True,
                "side": "LONG",
            },
        },
        base_margin=1,
    )
    tamper(vector)

    with pytest.raises(StrategyBacktestError, match="factor.vector|five"):
        _validate_factor_vector(
            {
                "signal": "ENTER",
                "side": "LONG",
                "factor_vector": vector,
            },
            status="ENTER",
            required=True,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"entry_ttl": "NaT"}, "entry_ttl must be positive"),
        ({"vol_max_r": float("nan")}, "must be finite"),
        (
            {"history_limit": 10, "min_context_bars": 10, "min_daily_bars": 22},
            "min_daily_bars cannot exceed",
        ),
    ],
)
def test_invalid_runtime_settings_fail_closed(overrides, message):
    kwargs = {
        "symbols": ["EURUSD"],
        "start": "2026-01-01",
        "end": "2026-02-01",
        "initial_capital": 10_000,
        **overrides,
    }
    with pytest.raises(StrategyBacktestError, match=message):
        NarrativeBacktestConfig.build(**kwargs)


def test_requested_range_and_unknown_symbol_fail_cleanly(tmp_path):
    dataset = _dataset(tmp_path)
    outside = NarrativeBacktestConfig.build(
        symbols=["EURUSD"],
        start="2025-12-31",
        end="2026-01-01T00:30:00Z",
        initial_capital=10_000,
        history_limit=1,
        min_context_bars=1,
        min_daily_bars=1,
        profile="signal-quality",
    )
    with pytest.raises(StrategyBacktestError, match="outside common M15"):
        run_narrative_backtest(
            dataset,
            outside,
            strategy_factory=_AlwaysEnter,
        )

    unknown = NarrativeBacktestConfig.build(
        symbols=["GBPUSD"],
        start="2026-01-01T00:15:00Z",
        end="2026-01-01T00:30:00Z",
        initial_capital=10_000,
        profile="signal-quality",
    )
    with pytest.raises(StrategyBacktestError, match="absent from snapshot"):
        run_narrative_backtest(dataset, unknown)


def test_session_window_end_is_exclusive():
    assert _time_in_window(time(12), time(12), time(21))
    assert not _time_in_window(time(21), time(12), time(21))


def test_portfolio_path_metrics_are_sorted_by_realized_time():
    rows = [
        {
            "setup_id": "a",
            "status": "CLOSED",
            "net_r": 2.0,
            "exit_time": "2026-01-01T00:01:00+00:00",
        },
        {
            "setup_id": "c",
            "status": "CLOSED",
            "net_r": 2.0,
            "exit_time": "2026-01-01T00:03:00+00:00",
        },
        {
            "setup_id": "b",
            "status": "CLOSED",
            "net_r": -1.0,
            "exit_time": "2026-01-01T00:02:00+00:00",
        },
        {
            "setup_id": "d",
            "status": "CLOSED",
            "net_r": -3.0,
            "exit_time": "2026-01-01T00:04:00+00:00",
        },
    ]
    metrics = _metrics_for(rows)
    assert metrics["max_drawdown_r"] == pytest.approx(3.0)
    assert metrics["longest_loss_streak"] == 1


def test_report_is_atomic_manifested_and_never_overwritten(tmp_path):
    result = run_narrative_backtest(
        _dataset(tmp_path / "data"),
        _config(),
        strategy_factory=_AlwaysEnter,
    )
    output = tmp_path / "report"

    assert result.write(output) == output.resolve()
    assert {
        "candidates.csv",
        "candidate_factors.csv",
        "config.json",
        "executions.csv",
        "factor_summary.csv",
        "folds.csv",
        "legs.csv",
        "manifest.json",
        "shadow_coefficients.csv",
        "shadow_models.json",
        "shadow_score_metrics.csv",
        "shadow_scores.csv",
        "setup_factor_attribution.csv",
        "setups.csv",
        "summary.json",
    } == {path.name for path in output.iterdir()}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "narrative-backtest/v2"
    assert len(manifest["files"]) == 14
    with pytest.raises(StrategyBacktestError, match="already exists"):
        result.write(output)


def test_empty_reports_keep_stable_csv_headers(tmp_path):
    result = run_narrative_backtest(
        _dataset(tmp_path / "data"),
        _config(),
        strategy_factory=_NeverEnter,
    )
    output = result.write(tmp_path / "empty-report")

    report_schemas = {
        "candidates.csv": "candidates",
        "executions.csv": "executions",
        "setups.csv": "setups",
        "legs.csv": "legs",
        "candidate_factors.csv": "candidate_factors",
        "setup_factor_attribution.csv": "setup_factors",
        "factor_summary.csv": "factor_summary",
        "shadow_scores.csv": "shadow_predictions",
        "shadow_coefficients.csv": "shadow_coefficients",
        "shadow_score_metrics.csv": "shadow_metrics",
    }
    for filename, schema_key in report_schemas.items():
        assert tuple(pd.read_csv(output / filename).columns) == (
            _EMPTY_CSV_FIELDS[schema_key]
        )


def test_live_strategy_settings_are_explicit_and_rejection_entry_is_off():
    config = NarrativeBacktestConfig.build(
        symbols=["EURUSD"],
        start="2026-01-01",
        end="2026-02-01",
        initial_capital=10_000,
    )
    assert config.rejection_block_entry_enabled is False
    assert config.orderblock_entry_enabled is True
    assert config.orderblock_max_age_bars == 80
    assert config.htf_score_margin == 2

    strategy = _configure_strategy(_default_strategy_factory(), config)
    assert strategy.rejection_block_entry_enabled is False
    assert strategy.orderblock_max_age_bars == 80
    assert strategy.htf_score_margin == 2


def test_absorption_trigger_kind_prefers_structured_value_and_reason_fallback():
    assert (
        _trigger_kind({"trigger_kind": "absorption_15m"})
        == "absorption_15m"
    )
    assert (
        _trigger_kind({"trigger_kind": "h1_pivot_reclaim_15m"})
        == "h1_pivot_reclaim_15m"
    )
    assert (
        _trigger_kind(
            {"trigger_reason": "Absorption 15M LONG footprint | ratio=4"}
        )
        == "absorption_15m"
    )


def test_trigger_signature_keeps_structured_families_and_events_separate():
    base = {
        "side": "LONG",
        "zone_low": 99.0,
        "zone_high": 100.0,
        "stop_price": 98.0,
        "trigger_reason": "shared reason",
    }
    pivot = _trigger_signature(
        {**base, "trigger_kind": "h1_pivot_reclaim_15m"}
    )
    order_block = _trigger_signature(
        {**base, "trigger_kind": "order_block_1h"}
    )
    first_event = _trigger_signature(
        {
            **base,
            "trigger_kind": "absorption_15m",
            "trigger_event_id": "event-a",
        }
    )
    second_event = _trigger_signature(
        {
            **base,
            "trigger_kind": "absorption_15m",
            "trigger_event_id": "event-b",
        }
    )

    assert pivot != order_block
    assert first_event != second_event


def test_forced_close_values_every_remaining_leg_at_the_same_price():
    bars = _frame(
        ["2026-01-02T17:58:00Z"],
        [(100.0, 105.0, 95.0, 104.0)],
    )
    open_outcome = simulate_split_outcome(
        side="LONG",
        entry=100.0,
        stop=90.0,
        tp_prices=[110.0, 120.0, 130.0],
        bars=bars,
    )
    forced = _force_close_outcome(
        open_outcome,
        price=105.0,
        timestamp=pd.Timestamp("2026-01-02T18:00:00Z"),
    )

    assert forced.status == "CLOSED"
    assert forced.net_r == pytest.approx(0.5)
    assert {leg.exit_reason for leg in forced.legs} == {"TIME"}


def test_release_manifest_binds_the_running_strategy_files(tmp_path):
    root = Path(__file__).resolve().parents[1]
    files = {
        name: _sha256_file(root / name)
        for name in sorted(_REQUIRED_RELEASE_FILES)
    }
    payload = {
        "schema": "forexbot-backtest-release/v1",
        "release_commit": "a" * 40,
        "files": files,
    }
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    digest = verify_release_manifest(
        manifest,
        expected_commit="a" * 40,
    )
    assert len(digest) == 64

    payload["files"]["core/strategy_narrative.py"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StrategyBacktestError, match="hash mismatch"):
        verify_release_manifest(manifest, expected_commit="a" * 40)
