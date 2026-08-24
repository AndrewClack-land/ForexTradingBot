from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from core.strategy_narrative import CandidateEntry, NarrativeStrategy


def _frames():
    frame = pd.DataFrame(
        [{"open": 1.0, "high": 1.01, "low": 0.99, "close": 1.0, "volume": 1.0}]
    )
    return {"D": frame, "4H": frame, "1H": frame, "15M": frame}


def test_retired_m15_and_disabled_h1_fall_through_to_quote_pressure():
    strategy = NarrativeStrategy()
    strategy._last_htf_context = None
    strategy.calc_narrative = lambda *args, **kwargs: ("LONG", "test bias")
    strategy.calc_fvg_regime_1h = lambda *args, **kwargs: ("LONG", "fvg")
    strategy.trigger_15m_rejection_block = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("retired RB M15 must never run")
    )
    strategy.trigger_h1_rejection_block = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("disabled RB H1 must not run")
    )
    strategy.trigger_15m_turtle_soup = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("retired Turtle Soup must not run")
    )
    strategy.trigger_15m_quote_pressure_rejection = lambda *args, **kwargs: CandidateEntry(
        side="LONG",
        entry_price=1.0,
        entry_min=0.999,
        entry_max=1.001,
        tf="15M",
        reason="FxPro Quote Pressure Rejection 15M test",
        trigger_kind="fxpro_quote_pressure_rejection_15m",
        lock_entry_range=True,
    )
    strategy.calc_stop_and_tps = lambda *args, **kwargs: (0.99, [1.01, 1.02, 1.03])
    strategy.rejection_block_entry_enabled = True
    strategy.rejection_block_h1_entry_enabled = False

    signal = strategy.generate_signal(_frames(), symbol="EURUSD")

    assert signal["signal"] == "ENTER"
    assert signal["trigger_reason"] == "FxPro Quote Pressure Rejection 15M test"
    assert signal["trigger_kind"] == "fxpro_quote_pressure_rejection_15m"


def test_enabled_h1_rejection_block_has_priority_without_calling_m15():
    strategy = NarrativeStrategy()
    strategy._last_htf_context = None
    strategy.calc_narrative = lambda *args, **kwargs: ("LONG", "test bias")
    strategy.calc_fvg_regime_1h = lambda *args, **kwargs: ("LONG", "fvg")
    strategy.trigger_15m_rejection_block = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("retired RB M15 must never run")
    )
    strategy.trigger_h1_rejection_block = lambda *args, **kwargs: CandidateEntry(
        side="LONG",
        entry_price=1.0,
        entry_min=0.999,
        entry_max=1.001,
        tf="1H",
        reason="RejectionBlock 1H test",
        trigger_kind="rejection_block_1h",
        lock_entry_range=True,
    )
    strategy.trigger_15m_turtle_soup = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("retired Turtle Soup must not run")
    )
    strategy.trigger_15m_quote_pressure_rejection = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("RB H1 must keep priority when enabled")
    )
    strategy.calc_stop_and_tps = lambda *args, **kwargs: (0.99, [1.01, 1.02, 1.03])
    strategy.rejection_block_h1_entry_enabled = True

    signal = strategy.generate_signal(_frames(), symbol="EURUSD")

    assert signal["trigger_reason"] == "RejectionBlock 1H test"
    assert signal["trigger_kind"] == "rejection_block_1h"
    assert signal["setup_tf"] == "1H"
    assert signal["tf"] == "15M"


def test_liquidity_miss_falls_through_to_pivot_without_turtle():
    strategy = NarrativeStrategy()
    strategy._last_htf_context = None
    strategy.calc_narrative = lambda *args, **kwargs: ("LONG", "test bias")
    strategy.calc_fvg_regime_1h = lambda *args, **kwargs: ("LONG", "fvg")
    strategy.trigger_15m_rejection_block = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("retired RB M15 must never run")
    )
    strategy.trigger_h1_rejection_block = lambda *args, **kwargs: None
    strategy.trigger_15m_quote_pressure_rejection = lambda *args, **kwargs: None
    strategy.trigger_15m_turtle_soup = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("retired Turtle Soup must not run")
    )
    strategy.trigger_h1_pivot_reclaim_on_15m = (
        lambda *args, **kwargs: CandidateEntry(
            side="LONG",
            entry_price=1.0,
            entry_min=0.999,
            entry_max=1.001,
            tf="15M",
            reason="H1 PivotLow reclaim test",
            lock_entry_range=True,
        )
    )
    strategy.calc_stop_and_tps = lambda *args, **kwargs: (
        0.99,
        [1.01, 1.02, 1.03],
    )
    strategy.rejection_block_h1_entry_enabled = True

    signal = strategy.generate_signal(_frames(), symbol="EURUSD")

    assert signal["signal"] == "ENTER"
    assert signal["trigger_reason"] == "H1 PivotLow reclaim test"


def test_shadow_population_lists_all_triggers_without_changing_live_priority():
    strategy = NarrativeStrategy()
    strategy._last_htf_context = None
    strategy.calc_narrative = lambda *args, **kwargs: ("LONG", "test bias")
    strategy.calc_fvg_regime_1h = lambda *args, **kwargs: ("LONG", "fvg")
    strategy.calc_stop_and_tps = lambda *args, **kwargs: (
        0.99,
        [1.01, 1.02, 1.03],
    )
    strategy.rejection_block_h1_entry_enabled = True

    def entry(kind: str, tf: str = "15M") -> CandidateEntry:
        return CandidateEntry(
            side="LONG",
            entry_price=1.0,
            entry_min=0.999,
            entry_max=1.001,
            tf=tf,
            reason=f"{kind} test",
            trigger_kind=kind,
            lock_entry_range=True,
        )

    strategy.trigger_h1_rejection_block = (
        lambda *args, **kwargs: entry("rejection_block_1h", "1H")
    )
    strategy.trigger_15m_cluster_rejection = (
        lambda *args, **kwargs: entry("fxpro_cluster_rejection_15m")
    )
    strategy.trigger_15m_quote_pressure_rejection = (
        lambda *args, **kwargs: entry(
            "fxpro_quote_pressure_rejection_15m"
        )
    )
    strategy.trigger_h1_pivot_reclaim_on_15m = (
        lambda *args, **kwargs: entry("pivot_reclaim")
    )
    strategy.trigger_orderblock_touch = (
        lambda *args, **kwargs: entry("orderblock_1h", "1H")
    )

    live = strategy.generate_signal(_frames(), symbol="EURUSD")
    population = strategy.generate_candidate_signals(
        _frames(),
        symbol="EURUSD",
    )

    assert live["trigger_kind"] == "rejection_block_1h"
    assert [row["trigger_kind"] for row in population] == [
        "rejection_block_1h",
        "fxpro_cluster_rejection_15m",
        "fxpro_quote_pressure_rejection_15m",
        "pivot_reclaim",
        "orderblock_1h",
    ]
    assert [row["production_priority"] for row in population] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert [row["shadow_candidate_rank"] for row in population] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert all(row["shadow_only"] is True for row in population)
    assert population[0]["entry_price"] == live["entry_price"]
    assert strategy._last_candidate_errors == []


def test_shadow_population_failure_is_diagnostic_and_does_not_hide_others():
    strategy = NarrativeStrategy()
    strategy._last_htf_context = None
    strategy.calc_narrative = lambda *args, **kwargs: ("LONG", "test bias")
    strategy.calc_fvg_regime_1h = lambda *args, **kwargs: ("LONG", "fvg")
    strategy.calc_stop_and_tps = lambda *args, **kwargs: (
        0.99,
        [1.01, 1.02, 1.03],
    )
    strategy.rejection_block_h1_entry_enabled = False
    strategy.trigger_15m_cluster_rejection = lambda *args, **kwargs: None
    strategy.trigger_15m_quote_pressure_rejection = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("diagnostic detector failed")
        )
    )
    pivot = CandidateEntry(
        side="LONG",
        entry_price=1.0,
        entry_min=0.999,
        entry_max=1.001,
        tf="15M",
        reason="pivot survives",
        trigger_kind="pivot_reclaim",
        lock_entry_range=True,
    )
    strategy.trigger_h1_pivot_reclaim_on_15m = (
        lambda *args, **kwargs: pivot
    )
    strategy.trigger_orderblock_touch = lambda *args, **kwargs: None

    population = strategy.generate_candidate_signals(
        _frames(),
        symbol="EURUSD",
    )

    assert [row["trigger_kind"] for row in population] == ["pivot_reclaim"]
    assert strategy._last_candidate_errors == [
        {
            "stage": "detector",
            "priority": 3,
            "detector": "fxpro_quote_pressure_rejection_15m",
            "error": "RuntimeError: diagnostic detector failed",
        }
    ]

def test_h1_detector_tags_timeframe_and_stable_event():
    index = pd.date_range("2026-01-01", periods=9, freq="1h", tz="UTC")
    rows = [
        {"open": 1.10, "high": 1.12, "low": 1.04, "close": 1.08, "volume": 1.0}
        for _ in range(6)
    ]
    rows.extend(
        [
            {"open": 1.05, "high": 1.08, "low": 1.00, "close": 1.06, "volume": 1.0},
            {"open": 1.02, "high": 1.04, "low": 0.90, "close": 1.03, "volume": 1.0},
            {"open": 1.025, "high": 1.05, "low": 0.95, "close": 1.04, "volume": 1.0},
        ]
    )
    frame = pd.DataFrame(rows, index=index)
    strategy = NarrativeStrategy()
    ctx = SimpleNamespace(
        rejection_blocks=[
            SimpleNamespace(
                side="LONG",
                zone_low=0.90,
                zone_high=1.02,
                created_idx=7,
                available_idx=8,
                pivot_time=str(index[7]),
                available_time=str(index[8]),
                wick_ratio=2.5,
                atr_at_pivot=0.03,
                detector_version="pine-v1-causal",
                valid=True,
                broken=False,
                retested=False,
            )
        ]
    )

    entry = strategy.trigger_h1_rejection_block(
        frame,
        "LONG",
        ctx=ctx,
        symbol="EURUSD",
    )

    assert entry is not None
    assert entry.tf == "1H"
    assert entry.trigger_kind == "rejection_block_1h"
    assert entry.trigger_event_id == (
        f"rb:h1:touch:v1:EURUSD:LONG:"
        f"{pd.Timestamp(index[7]).isoformat()}"
    )
    assert entry.entry_price == 1.02
    assert entry.entry_min == entry.entry_max == 1.02
    assert entry.stop_override == 0.90
    assert entry.lock_entry_range is True
    assert entry.lock_stop_override is True
    assert entry.entry_order_type == "LIMIT_RETEST"
    assert entry.trigger_meta["schema"] == "rb-h1-exact-retest/v1"
    assert entry.trigger_meta["pivot_index"] == 7

def test_h4_detector_tags_timeframe_and_stable_event():
    index = pd.date_range("2026-01-01", periods=9, freq="4h", tz="UTC")
    rows = [
        {"open": 1.10, "high": 1.12, "low": 1.04, "close": 1.08, "volume": 1.0}
        for _ in range(6)
    ]
    rows.extend(
        [
            {"open": 1.05, "high": 1.08, "low": 1.00, "close": 1.06, "volume": 1.0},
            {"open": 1.02, "high": 1.04, "low": 0.90, "close": 1.03, "volume": 1.0},
            {"open": 1.025, "high": 1.05, "low": 0.95, "close": 1.04, "volume": 1.0},
        ]
    )
    frame = pd.DataFrame(rows, index=index)
    strategy = NarrativeStrategy()

    entry = strategy.trigger_h4_rejection_block(frame, "LONG")

    assert entry is not None
    assert entry.tf == "4H"
    assert entry.trigger_kind == "rejection_block_4h"
    assert entry.trigger_event_id == f"rb:4h:long:{index[-2]}"
    assert entry.trigger_meta == {
        "setup_timeframe": "4H",
        "pivot_index": str(index[-2]),
    }
