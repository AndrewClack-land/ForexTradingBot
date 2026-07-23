from __future__ import annotations

import pandas as pd
import pytest

from backtest.metrics import aggregate_setup_metrics
from backtest.simulator import simulate_split_outcome


def _bars(*rows):
    index = pd.date_range("2026-01-01T00:00:00Z", periods=len(rows), freq="min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)


def _simulate(bars, *, policy="stop-first", side="LONG"):
    if side == "LONG":
        return simulate_split_outcome(
            side="LONG",
            entry=100,
            stop=90,
            tp_prices=[110, 120, 130],
            bars=bars,
            intrabar_policy=policy,
        )
    return simulate_split_outcome(
        side="SHORT",
        entry=100,
        stop=110,
        tp_prices=[90, 80, 70],
        bars=bars,
        intrabar_policy=policy,
    )


def test_full_stop_is_one_setup_loss_not_three_trades():
    outcome = _simulate(_bars((100, 105, 89, 95)))
    assert outcome.status == "CLOSED"
    assert outcome.net_r == pytest.approx(-1.0)
    assert [leg.exit_reason for leg in outcome.legs] == ["STOP", "STOP", "STOP"]


@pytest.mark.parametrize(
    ("bars", "expected_r", "expected_hits"),
    [
        (_bars((101, 111, 101, 108), (104, 106, 99, 100)), 0.5, (1,)),
        (_bars((101, 121, 101, 118), (104, 106, 99, 100)), 1.1, (1, 2)),
        (_bars((101, 131, 101, 128)), 1.7, (1, 2, 3)),
    ],
)
def test_tp1_moves_remaining_legs_to_be(bars, expected_r, expected_hits):
    outcome = _simulate(bars)
    assert outcome.status == "CLOSED"
    assert outcome.net_r == pytest.approx(expected_r)
    assert outcome.tp_hits == expected_hits
    assert outcome.moved_to_be is (len(expected_hits) < 3)


def test_ambiguous_bar_is_counted_and_policy_is_deterministic():
    collision = _bars((100, 111, 89, 100))
    stop_first = _simulate(collision, policy="stop-first")
    tp_first = _simulate(collision, policy="tp-first")
    assert stop_first.ambiguous_bars == 1
    assert tp_first.ambiguous_bars == 1
    assert stop_first.net_r == pytest.approx(-1.0)
    assert tp_first.net_r == pytest.approx(0.5)


def test_short_setup_is_symmetric():
    outcome = _simulate(
        _bars((99, 99, 79, 82), (99, 101, 95, 100)),
        side="SHORT",
    )
    assert outcome.net_r == pytest.approx(1.1)
    assert outcome.tp_hits == (1, 2)


def test_metrics_aggregate_setups_not_legs():
    outcomes = [
        _simulate(_bars((100, 105, 89, 95))),
        _simulate(_bars((101, 111, 101, 108), (104, 106, 99, 100))),
        _simulate(_bars((101, 121, 101, 118), (104, 106, 99, 100))),
        _simulate(_bars((101, 131, 101, 128))),
    ]
    metrics = aggregate_setup_metrics(outcomes)
    assert metrics["setups_total"] == 4
    assert metrics["setups_closed"] == 4
    assert metrics["wins"] == 3
    assert metrics["losses"] == 1
    assert metrics["win_rate"] == pytest.approx(0.75)
    assert metrics["net_r"] == pytest.approx(2.3)
    assert metrics["expectancy_r"] == pytest.approx(0.575)
    assert metrics["profit_factor"] == pytest.approx(3.3)
    assert metrics["max_drawdown_r"] == pytest.approx(1.0)
    assert metrics["tp1_reach_rate"] == pytest.approx(0.75)
