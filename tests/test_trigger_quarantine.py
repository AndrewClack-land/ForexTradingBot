from __future__ import annotations

import pandas as pd

from core.strategy_narrative import CandidateEntry, NarrativeStrategy


def _frames():
    frame = pd.DataFrame(
        [{"open": 1.0, "high": 1.01, "low": 0.99, "close": 1.0, "volume": 1.0}]
    )
    return {"D": frame, "4H": frame, "1H": frame, "15M": frame}


def test_rejection_block_quarantine_falls_through_to_next_trigger():
    strategy = NarrativeStrategy()
    strategy._last_htf_context = None
    strategy.calc_narrative = lambda *args, **kwargs: ("LONG", "test bias")
    strategy.calc_fvg_regime_1h = lambda *args, **kwargs: ("LONG", "fvg")
    strategy.trigger_15m_rejection_block = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("quarantined trigger must not run")
    )
    strategy.trigger_15m_turtle_soup = lambda *args, **kwargs: CandidateEntry(
        side="LONG",
        entry_price=1.0,
        entry_min=0.999,
        entry_max=1.001,
        tf="15M",
        reason="TurtleSoup test",
        lock_entry_range=True,
    )
    strategy.calc_stop_and_tps = lambda *args, **kwargs: (0.99, [1.01, 1.02, 1.03])
    strategy.rejection_block_entry_enabled = False

    signal = strategy.generate_signal(_frames(), symbol="EURUSD")

    assert signal["signal"] == "ENTER"
    assert signal["trigger_reason"] == "TurtleSoup test"


def test_enabled_rejection_block_keeps_trigger_priority():
    strategy = NarrativeStrategy()
    strategy._last_htf_context = None
    strategy.calc_narrative = lambda *args, **kwargs: ("LONG", "test bias")
    strategy.calc_fvg_regime_1h = lambda *args, **kwargs: ("LONG", "fvg")
    strategy.trigger_15m_rejection_block = lambda *args, **kwargs: CandidateEntry(
        side="LONG",
        entry_price=1.0,
        entry_min=0.999,
        entry_max=1.001,
        tf="15M",
        reason="RejectionBlock test",
        lock_entry_range=True,
    )
    strategy.trigger_15m_turtle_soup = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("RB must keep priority when enabled")
    )
    strategy.calc_stop_and_tps = lambda *args, **kwargs: (0.99, [1.01, 1.02, 1.03])
    strategy.rejection_block_entry_enabled = True

    signal = strategy.generate_signal(_frames(), symbol="EURUSD")

    assert signal["trigger_reason"] == "RejectionBlock test"
