from __future__ import annotations

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

    entry = strategy.trigger_h1_rejection_block(frame, "LONG")

    assert entry is not None
    assert entry.tf == "1H"
    assert entry.trigger_kind == "rejection_block_1h"
    assert entry.trigger_event_id == f"rb:1h:long:{index[-2]}"
    assert entry.trigger_meta == {
        "setup_timeframe": "1H",
        "pivot_index": str(index[-2]),
    }

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
