from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import main
from core.htf_context import OrderBlock, OrderBlockTracker
from core.strategy_narrative import NarrativeStrategy


def _bars(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


def test_bullish_order_block_breaks_below_its_own_bottom():
    df = _bars(
        [
            (10.0, 10.8, 9.6, 10.1),
            (10.1, 10.9, 9.5, 10.0),
            (10.0, 10.7, 9.4, 10.2),
            (10.2, 10.5, 9.0, 9.6),
            (9.8, 12.2, 9.8, 12.0),
            (11.8, 11.9, 8.5, 8.8),
            (8.8, 9.2, 8.6, 8.9),
            (8.9, 9.3, 8.7, 9.0),
        ]
    )

    blocks = OrderBlockTracker(swing_lookback=3, show_last=3).build(df)
    bullish = next(ob for ob in blocks if ob.side == "LONG")

    assert bullish.bottom == 9.0
    assert bullish.breaker is True
    assert bullish.breaker_idx == 5


def test_bearish_order_block_breaks_above_its_own_top():
    df = _bars(
        [
            (10.0, 10.5, 9.4, 9.9),
            (9.9, 10.6, 9.5, 10.0),
            (10.0, 10.7, 9.6, 9.8),
            (10.2, 11.2, 10.0, 10.8),
            (10.0, 10.2, 8.5, 8.8),
            (9.0, 11.8, 8.9, 11.5),
            (11.5, 11.7, 11.0, 11.4),
            (11.4, 11.6, 10.9, 11.3),
        ]
    )

    blocks = OrderBlockTracker(swing_lookback=3, show_last=3).build(df)
    bearish = next(ob for ob in blocks if ob.side == "SHORT")

    assert bearish.top == 11.2
    assert bearish.breaker is True
    assert bearish.breaker_idx == 5


def test_order_blocks_are_returned_by_recency_not_side():
    blocks = OrderBlockTracker(swing_lookback=3, show_last=3).build(
        _bars(
            [
                (10.0, 10.8, 9.6, 10.1),
                (10.1, 10.9, 9.5, 10.0),
                (10.0, 10.7, 9.4, 10.2),
                (10.2, 10.5, 9.0, 9.6),
                (9.8, 12.2, 9.8, 12.0),
                (11.8, 11.9, 8.5, 8.8),
                (8.8, 9.2, 8.6, 8.9),
                (8.9, 9.3, 8.7, 9.0),
            ]
        )
    )

    assert [ob.created_idx for ob in blocks] == sorted(
        (ob.created_idx for ob in blocks), reverse=True
    )


def test_narrative_votes_for_most_recent_active_order_block():
    strategy = NarrativeStrategy()
    strategy.htf_score_margin = 1
    ctx = SimpleNamespace(
        daily_range=None,
        dealing_range=None,
        daily_breakout=None,
        order_blocks_4h=[
            OrderBlock(side="LONG", top=1.2, bottom=1.1, created_idx=10),
            OrderBlock(side="SHORT", top=1.4, bottom=1.3, created_idx=20),
        ],
        rejection_blocks_4h=[],
    )
    strategy._build_htf_context = lambda *args, **kwargs: ctx
    strategy.calc_fvg_regime_1h = lambda df: ("NEUTRAL", "FVG neutral")
    df = pd.DataFrame(
        {"open": [1.0], "high": [1.1], "low": [0.9], "close": [1.0]}
    )

    side, text = strategy.calc_narrative(None, df, df)

    assert side == "SHORT"
    assert "OB4H SHORT" in text


def test_build_tf_data_requests_daily_candles():
    requested = []
    frame = pd.DataFrame(
        {"open": [1.0], "high": [1.1], "low": [0.9], "close": [1.0]}
    )

    class Cache:
        def request(self, symbol, tf, limit):
            requested.append((symbol, tf, limit))
            return frame

    core = main.Core.__new__(main.Core)
    core.N_BARS = 300
    core.data_cache = Cache()
    core.feed = SimpleNamespace(
        get_klines=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cache result should be used")
        )
    )

    data = core._build_tf_data("EURUSD")

    assert data["D"] is frame
    assert ("EURUSD", "1d", 300) in requested
