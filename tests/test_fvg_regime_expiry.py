from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from core.htf_context import HourlyRange
from core.strategy_narrative import NarrativeStrategy

FLAT = (1.0, 1.1, 0.9, 1.0)


def _fvg_frame(pad_after: int) -> pd.DataFrame:
    """H1 bars whose newest IMFVG signal is a bull one ``pad_after`` bars ago.

    The three pattern bars satisfy the LuxAlgo instantaneous-mitigation bull
    rule (``low[i-3] > high[i-1]``, ``close[i-2] < low[i-3]``,
    ``close[i] > low[i-3]``); the flat padding on both sides is inert, so the
    signal age is exactly ``pad_after``.
    """

    rows = [FLAT] * 5
    rows.append((2.1, 2.2, 2.0, 2.1))      # i-3: gap origin
    rows.append((1.6, 1.7, 1.4, 1.5))      # i-2: closes below low[i-3]
    rows.append((0.9, 1.0, 0.8, 0.95))     # i-1: high below low[i-3]
    rows.append((2.4, 2.6, 2.3, 2.5))      # i  : reclaims above low[i-3]
    rows.extend([FLAT] * pad_after)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


def test_fresh_signal_keeps_the_regime():
    strategy = NarrativeStrategy()
    strategy.fvg_regime_max_age_bars = 24

    side, text = strategy.calc_fvg_regime_1h(_fvg_frame(2))

    assert side == "LONG"
    assert "2 барами ранее" in text


def test_signal_older_than_the_ceiling_expires_to_neutral():
    strategy = NarrativeStrategy()
    strategy.fvg_regime_max_age_bars = 24

    side, text = strategy.calc_fvg_regime_1h(_fvg_frame(30))

    assert side == "NEUTRAL"
    assert "истёк" in text
    assert "лимита 24" in text


def test_ceiling_is_inclusive_at_its_exact_age():
    strategy = NarrativeStrategy()
    strategy.fvg_regime_max_age_bars = 24

    assert strategy.calc_fvg_regime_1h(_fvg_frame(24))[0] == "LONG"
    assert strategy.calc_fvg_regime_1h(_fvg_frame(25))[0] == "NEUTRAL"


def test_zero_restores_the_unbounded_latch():
    strategy = NarrativeStrategy()
    strategy.fvg_regime_max_age_bars = 0

    side, _ = strategy.calc_fvg_regime_1h(_fvg_frame(120))

    assert side == "LONG"


def test_default_regime_is_time_unbounded():
    assert NarrativeStrategy().fvg_regime_max_age_bars == 0


def test_expired_regime_restores_the_symmetric_base_margin():
    """An expired regime must stop taxing the opposing side's margin."""

    strategy = NarrativeStrategy()
    strategy.htf_score_margin = 2
    strategy._build_htf_context = lambda *args, **kwargs: SimpleNamespace(
        hourly_range=HourlyRange(
            high=1.2, low=1.0, close=1.05, position="DISCOUNT"
        ),
        false_breakout_4h=None,
        true_breakout_15m=None,
        false_breakout_1h=None,
        true_breakout_1h=None,
        order_blocks=[],
        rejection_blocks=[],
    )
    frame_15m = pd.DataFrame([FLAT], columns=["open", "high", "low", "close"])

    strategy.fvg_regime_max_age_bars = 0
    strategy.calc_narrative(frame_15m, _fvg_frame(120), frame_15m)
    latched = strategy._last_factor_vector

    strategy.fvg_regime_max_age_bars = 24
    strategy.calc_narrative(frame_15m, _fvg_frame(120), frame_15m)
    expired = strategy._last_factor_vector

    # The stale LONG regime taxed SHORT; once expired both sides pay the base.
    assert latched["fvg_side"] == "LONG"
    assert latched["margin_short"] == 3
    assert latched["margin_long"] == 2

    assert expired["fvg_side"] == "NEUTRAL"
    assert expired["margin_short"] == 2
    assert expired["margin_long"] == 2


def test_full_zone_break_is_causal_event_invalidation_first_copy():
    strategy = NarrativeStrategy()
    strategy.fvg_regime_max_age_bars = 0
    strategy.fvg_event_invalidation_mode = "zone_break"
    frame = _fvg_frame(0)
    frame.loc[len(frame)] = (1.0, 1.2, 0.7, 0.8)

    side, text = strategy.calc_fvg_regime_1h(frame)

    assert side == "NEUTRAL"
    assert "полное пробитие зоны" in text
    assert strategy._last_fvg_regime_meta["invalidation_reason"] == (
        "full_zone_break"
    )
    assert strategy._last_fvg_regime_meta["zone_low"] == 1.0


def test_structure_change_requires_confirmed_post_signal_swing_first_copy():
    strategy = NarrativeStrategy()
    strategy.fvg_regime_max_age_bars = 0
    strategy.fvg_event_invalidation_mode = "structure_change"
    frame = _fvg_frame(0)
    frame.loc[len(frame)] = (2.3, 2.8, 2.0, 2.4)
    frame.loc[len(frame)] = (2.0, 2.7, 1.8, 2.1)
    frame.loc[len(frame)] = (2.2, 2.9, 2.0, 2.5)
    frame.loc[len(frame)] = (1.8, 2.6, 1.6, 1.7)

    side, text = strategy.calc_fvg_regime_1h(frame)

    assert side == "NEUTRAL"
    assert "смена структуры" in text
    assert strategy._last_fvg_regime_meta["invalidation_level"] == 1.8
    # The close never traversed the original gap's distal edge at 1.0.
    assert frame["close"].iloc[-1] > 1.0


def test_event_breaks_do_not_affect_production_when_mode_is_none_first_copy():
    strategy = NarrativeStrategy()
    strategy.fvg_regime_max_age_bars = 0
    strategy.fvg_event_invalidation_mode = "none"
    frame = _fvg_frame(0)
    frame.loc[len(frame)] = (1.0, 1.2, 0.7, 0.8)

    assert strategy.calc_fvg_regime_1h(frame)[0] == "LONG"
    assert strategy._last_fvg_regime_meta["invalidated"] is False


def test_opposite_fvg_supersedes_the_latch_without_time_expiry_first_copy():
    strategy = NarrativeStrategy()
    strategy.fvg_regime_max_age_bars = 0
    frame = _fvg_frame(0)
    frame.loc[len(frame)] = (0.9, 1.0, 0.8, 0.9)
    frame.loc[len(frame)] = (1.4, 1.6, 1.2, 1.5)
    frame.loc[len(frame)] = (2.1, 2.3, 2.0, 2.2)
    frame.loc[len(frame)] = (0.6, 0.9, 0.5, 0.6)

    side, _ = strategy.calc_fvg_regime_1h(frame)

    assert side == "SHORT"
    assert strategy._last_fvg_regime_meta[
        "opposite_transition_seen"
    ] is True
    assert strategy._last_fvg_regime_meta["signal_age_bars"] == 0


def test_full_zone_break_is_causal_event_invalidation():
    strategy = NarrativeStrategy()
    strategy.fvg_regime_max_age_bars = 0
    strategy.fvg_event_invalidation_mode = "zone_break"
    frame = _fvg_frame(0)
    frame.loc[len(frame)] = (1.0, 1.2, 0.7, 0.8)

    side, text = strategy.calc_fvg_regime_1h(frame)

    assert side == "NEUTRAL"
    assert "полное пробитие зоны" in text
    assert strategy._last_fvg_regime_meta["invalidation_reason"] == (
        "full_zone_break"
    )
    assert strategy._last_fvg_regime_meta["zone_low"] == 1.0


def test_structure_change_requires_confirmed_post_signal_swing():
    strategy = NarrativeStrategy()
    strategy.fvg_regime_max_age_bars = 0
    strategy.fvg_event_invalidation_mode = "structure_change"
    frame = _fvg_frame(0)
    frame.loc[len(frame)] = (2.3, 2.8, 2.0, 2.4)
    frame.loc[len(frame)] = (2.0, 2.7, 1.8, 2.1)
    frame.loc[len(frame)] = (2.2, 2.9, 2.0, 2.5)
    frame.loc[len(frame)] = (1.8, 2.6, 1.6, 1.7)

    side, text = strategy.calc_fvg_regime_1h(frame)

    assert side == "NEUTRAL"
    assert "смена структуры" in text
    assert strategy._last_fvg_regime_meta["invalidation_level"] == 1.8
    # The close never traversed the original gap's distal edge at 1.0.
    assert frame["close"].iloc[-1] > 1.0


def test_event_breaks_do_not_affect_production_when_mode_is_none():
    strategy = NarrativeStrategy()
    strategy.fvg_regime_max_age_bars = 0
    strategy.fvg_event_invalidation_mode = "none"
    frame = _fvg_frame(0)
    frame.loc[len(frame)] = (1.0, 1.2, 0.7, 0.8)

    assert strategy.calc_fvg_regime_1h(frame)[0] == "LONG"
    assert strategy._last_fvg_regime_meta["invalidated"] is False


def test_opposite_fvg_supersedes_the_latch_without_time_expiry():
    strategy = NarrativeStrategy()
    strategy.fvg_regime_max_age_bars = 0
    frame = _fvg_frame(0)
    frame.loc[len(frame)] = (0.9, 1.0, 0.8, 0.9)
    frame.loc[len(frame)] = (1.4, 1.6, 1.2, 1.5)
    frame.loc[len(frame)] = (2.1, 2.3, 2.0, 2.2)
    frame.loc[len(frame)] = (0.6, 0.9, 0.5, 0.6)

    side, _ = strategy.calc_fvg_regime_1h(frame)

    assert side == "SHORT"
    assert strategy._last_fvg_regime_meta[
        "opposite_transition_seen"
    ] is True
    assert strategy._last_fvg_regime_meta["signal_age_bars"] == 0
