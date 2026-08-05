"""Tests for the per-symbol entry-window floor.

The floor exists because the trigger and the executor disagree on how far
price may sit from the planned entry: a trigger accepts a setup with pips of
tolerance while the executor demands the live tick inside a window that can be
narrower than two pips. Widening is a research lever — it is OFF by default and
it moves the risk edge, so the stop, targets and lot size move with it.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from backtest.__main__ import _parse_entry_range_min_width
from backtest.strategy_runner import (
    NarrativeBacktestConfig,
    StrategyBacktestError,
    _configure_strategy,
)
from config import _env_price_map
from core.strategy_narrative import CandidateEntry, NarrativeStrategy


def _frames():
    frame = pd.DataFrame(
        [{"open": 1.0, "high": 1.01, "low": 0.99, "close": 1.0, "volume": 1.0}]
    )
    return {"D": frame, "4H": frame, "1H": frame, "15M": frame}


def _strategy(
    *,
    width=None,
    side="LONG",
    entry_min=0.9990,
    entry_max=1.0010,
    stop_and_tps=(0.99, [1.01, 1.02, 1.03]),
    lock_entry_range=True,
):
    """A strategy whose only live part is the entry-window plumbing."""
    strategy = NarrativeStrategy()
    strategy._last_htf_context = None
    strategy.calc_narrative = lambda *a, **k: (side, "test bias")
    strategy.calc_fvg_regime_1h = lambda *a, **k: (side, "fvg")
    strategy.rejection_block_h1_entry_enabled = True
    strategy.trigger_h1_rejection_block = lambda *a, **k: CandidateEntry(
        side=side,
        entry_price=1.0,
        entry_min=entry_min,
        entry_max=entry_max,
        tf="1H",
        reason="RejectionBlock 1H test",
        trigger_kind="rejection_block_1h",
        lock_entry_range=lock_entry_range,
    )
    strategy.calc_stop_and_tps = lambda *a, **k: stop_and_tps
    if width is not None:
        strategy.entry_range_min_width = dict(width)
    return strategy


# --------------------------------------------------------------------------
# Strategy geometry
# --------------------------------------------------------------------------


def test_floor_is_off_by_default_and_leaves_production_geometry_alone():
    signal = _strategy().generate_signal(_frames(), symbol="EURUSD")

    assert signal["signal"] == "ENTER"
    assert signal["entry_min"] == pytest.approx(0.9990)
    assert signal["entry_max"] == pytest.approx(1.0010)
    assert signal["entry_range_widened"] is False


def test_narrow_window_is_widened_symmetrically_about_its_midpoint():
    strategy = _strategy(width={"EURUSD": 0.0060})

    signal = strategy.generate_signal(_frames(), symbol="EURUSD")

    low, high = signal["entry_min"], signal["entry_max"]
    assert high - low == pytest.approx(0.0060)
    # midpoint preserved: the trigger's own geometry still centres the window
    assert (low + high) / 2 == pytest.approx(1.0)
    assert signal["entry_range_widened"] is True


def test_window_already_wider_than_the_floor_is_untouched():
    strategy = _strategy(
        width={"EURUSD": 0.0010}, entry_min=0.9950, entry_max=1.0050
    )

    signal = strategy.generate_signal(_frames(), symbol="EURUSD")

    assert signal["entry_min"] == pytest.approx(0.9950)
    assert signal["entry_max"] == pytest.approx(1.0050)
    assert signal["entry_range_widened"] is False


def test_floor_applies_only_to_its_own_symbol():
    strategy = _strategy(width={"GBPUSD": 0.0060})

    signal = strategy.generate_signal(_frames(), symbol="EURUSD")

    assert signal["entry_max"] - signal["entry_min"] == pytest.approx(0.0020)
    assert signal["entry_range_widened"] is False


@pytest.mark.parametrize("width", [{}, {"EURUSD": 0.0}, {"EURUSD": -1.0}])
def test_absent_zero_or_negative_width_is_a_no_op(width):
    strategy = _strategy(width=width)

    signal = strategy.generate_signal(_frames(), symbol="EURUSD")

    assert signal["entry_min"] == pytest.approx(0.9990)
    assert signal["entry_range_widened"] is False


def test_widening_moves_the_long_risk_edge_so_stop_and_targets_follow():
    """LONG prices risk off entry_max: widening must make R larger, not free."""
    seen = []
    strategy = _strategy(width={"EURUSD": 0.0060})
    strategy.calc_stop_and_tps = lambda entry_price, *a, **k: (
        seen.append(entry_price) or (0.99, [1.01, 1.02, 1.03])
    )

    strategy.generate_signal(_frames(), symbol="EURUSD")

    # unwidened risk edge would have been 1.0010
    assert seen == [pytest.approx(1.0030)]


def test_widening_moves_the_short_risk_edge_the_other_way():
    seen = []
    strategy = _strategy(width={"EURUSD": 0.0060}, side="SHORT")
    strategy.calc_stop_and_tps = lambda entry_price, *a, **k: (
        seen.append(entry_price) or (1.01, [0.99, 0.98, 0.97])
    )

    strategy.generate_signal(_frames(), symbol="EURUSD")

    assert seen == [pytest.approx(0.9970)]


def test_widened_long_window_never_reaches_past_its_own_stop():
    """A fill at the favourable edge must not open already beyond the stop."""
    strategy = _strategy(
        width={"EURUSD": 0.0200}, stop_and_tps=(0.9950, [1.02, 1.03, 1.04])
    )

    signal = strategy.generate_signal(_frames(), symbol="EURUSD")

    # window would have spanned [0.9900, 1.0100]; the stop sits inside it
    assert signal["entry_min"] > 0.9950
    assert signal["entry_max"] == pytest.approx(1.0100)
    assert signal["stop_price"] == pytest.approx(0.9950)


def test_widened_short_window_never_reaches_past_its_own_stop():
    strategy = _strategy(
        width={"EURUSD": 0.0200},
        side="SHORT",
        stop_and_tps=(1.0050, [0.98, 0.97, 0.96]),
    )

    signal = strategy.generate_signal(_frames(), symbol="EURUSD")

    assert signal["entry_max"] < 1.0050
    assert signal["entry_min"] == pytest.approx(0.9900)


def test_floor_reaches_ranges_built_from_m15_not_only_locked_ones():
    """OB touch locks its range; pivot reclaim does not. Both must be floored."""
    strategy = _strategy(width={"EURUSD": 0.0060}, lock_entry_range=False)
    strategy._build_entry_range = lambda entry, *a, **k: entry

    signal = strategy.generate_signal(_frames(), symbol="EURUSD")

    assert signal["entry_max"] - signal["entry_min"] == pytest.approx(0.0060)
    assert signal["entry_range_widened"] is True


def test_widening_is_idempotent_across_repeated_decisions():
    strategy = _strategy(width={"EURUSD": 0.0060})

    first = strategy.generate_signal(_frames(), symbol="EURUSD")
    second = strategy.generate_signal(_frames(), symbol="EURUSD")

    assert first["entry_min"] == pytest.approx(second["entry_min"])
    assert first["entry_max"] == pytest.approx(second["entry_max"])


# --------------------------------------------------------------------------
# config.py env parsing
# --------------------------------------------------------------------------


def test_env_map_parses_a_multi_symbol_spec(monkeypatch):
    monkeypatch.setenv(
        "ENTRY_RANGE_MIN_WIDTH",
        "eurusd=0.0006, GBPUSD=0.0007 ,USDCAD=0.0012,GOLD=11",
    )

    assert _env_price_map("ENTRY_RANGE_MIN_WIDTH") == {
        "EURUSD": 0.0006,
        "GBPUSD": 0.0007,
        "USDCAD": 0.0012,
        "GOLD": 11.0,
    }


def test_env_map_is_empty_when_unset(monkeypatch):
    monkeypatch.delenv("ENTRY_RANGE_MIN_WIDTH", raising=False)

    assert _env_price_map("ENTRY_RANGE_MIN_WIDTH") == {}


@pytest.mark.parametrize(
    "raw",
    ["EURUSD", "EURUSD=abc", "EURUSD=-1", "EURUSD=nan", "=0.0006",
     "EURUSD=0.0006,EURUSD=0.0007"],
)
def test_env_map_refuses_malformed_specs(monkeypatch, raw):
    monkeypatch.setenv("ENTRY_RANGE_MIN_WIDTH", raw)

    with pytest.raises(ValueError):
        _env_price_map("ENTRY_RANGE_MIN_WIDTH")


# --------------------------------------------------------------------------
# Backtest config plumbing
# --------------------------------------------------------------------------


def _config(**kwargs):
    return NarrativeBacktestConfig.build(
        symbols=("EURUSD", "GBPUSD"),
        start="2026-01-01",
        end="2026-02-01",
        initial_capital=78652.24,
        profile="signal-quality",
        **kwargs,
    )


def test_backtest_config_defaults_to_no_floor():
    config = _config()

    assert config.entry_range_min_width == ()
    assert config.to_dict()["strategy_settings"]["entry_range_min_width"] == {}


def test_backtest_config_normalizes_sorts_and_drops_zero_widths():
    config = _config(
        entry_range_min_width={
            "gbpusd": 0.0007,
            "EURUSD": 0.0006,
            "USDCAD": 0.0,  # explicit zero means "no floor", not a setting
        }
    )

    assert config.entry_range_min_width == (
        ("EURUSD", 0.0006),
        ("GBPUSD", 0.0007),
    )


def test_backtest_config_rejects_an_empty_symbol_even_at_zero_width():
    with pytest.raises(StrategyBacktestError, match="empty symbol"):
        _config(entry_range_min_width={"   ": 0.0})


def test_backtest_config_is_permutation_independent():
    first = _config(entry_range_min_width={"EURUSD": 0.0006, "GBPUSD": 0.0007})
    second = _config(entry_range_min_width={"GBPUSD": 0.0007, "EURUSD": 0.0006})

    assert first.entry_range_min_width == second.entry_range_min_width
    assert first.to_dict() == second.to_dict()


def test_backtest_config_serializes_the_floor_for_attestation():
    config = _config(entry_range_min_width={"EURUSD": 0.0006})

    settings = config.to_dict()["strategy_settings"]
    assert settings["entry_range_min_width"] == {"EURUSD": 0.0006}


def test_backtest_config_rejects_a_symbol_outside_the_run():
    with pytest.raises(StrategyBacktestError, match="outside the run"):
        _config(entry_range_min_width={"USDCAD": 0.0012})


@pytest.mark.parametrize("width", [-0.001, float("nan"), float("inf")])
def test_backtest_config_rejects_non_finite_or_negative_widths(width):
    with pytest.raises(StrategyBacktestError, match="finite"):
        _config(entry_range_min_width={"EURUSD": width})


def test_configure_strategy_pushes_the_floor_onto_the_strategy():
    config = _config(entry_range_min_width={"EURUSD": 0.0006})
    strategy = NarrativeStrategy()

    _configure_strategy(strategy, config)

    assert strategy.entry_range_min_width == {"EURUSD": 0.0006}


def test_configure_strategy_fails_closed_when_the_floor_cannot_be_set():
    class Stubborn(NarrativeStrategy):
        @property
        def entry_range_min_width(self):
            return {}

        @entry_range_min_width.setter
        def entry_range_min_width(self, value):
            pass  # silently ignores the setting

    config = _config(entry_range_min_width={"EURUSD": 0.0006})

    with pytest.raises(StrategyBacktestError, match="failed closed"):
        _configure_strategy(Stubborn(), config)


# --------------------------------------------------------------------------
# CLI parsing
# --------------------------------------------------------------------------


def test_cli_parses_repeated_symbol_width_options():
    parsed = _parse_entry_range_min_width(
        ["eurusd=0.0006", "GBPUSD=0.0007", "GOLD=11"]
    )

    assert parsed == {"EURUSD": 0.0006, "GBPUSD": 0.0007, "GOLD": 11.0}


def test_cli_returns_empty_for_no_options():
    assert _parse_entry_range_min_width(None) == {}


@pytest.mark.parametrize(
    "item",
    ["EURUSD", "EURUSD=", "=0.0006", "EURUSD=abc", "EURUSD=-1",
     "EURUSD=inf"],
)
def test_cli_refuses_malformed_options(item):
    with pytest.raises(SystemExit):
        _parse_entry_range_min_width([item])


def test_cli_refuses_a_repeated_symbol():
    with pytest.raises(SystemExit, match="repeats symbol"):
        _parse_entry_range_min_width(["EURUSD=0.0006", "eurusd=0.0007"])


def test_researched_widths_are_expressible_end_to_end():
    """The values measured from the live journal must survive every hop."""
    parsed = _parse_entry_range_min_width(
        ["EURUSD=0.0006", "GBPUSD=0.0007", "USDCAD=0.0012"]
    )
    config = NarrativeBacktestConfig.build(
        symbols=("EURUSD", "GBPUSD", "USDCAD"),
        start="2026-01-01",
        end="2026-02-01",
        initial_capital=78652.24,
        profile="signal-quality",
        entry_range_min_width=parsed,
    )
    strategy = NarrativeStrategy()
    _configure_strategy(strategy, config)

    assert strategy.entry_range_min_width == parsed
    assert all(math.isfinite(width) for width in parsed.values())
