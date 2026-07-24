from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import main
from core.htf_context import (
    FractalBreakout,
    HourlyRange,
    HtfContext,
    OrderBlock,
    OrderBlockTracker,
    RejectionBlock,
    RejectionBlockTracker,
)
from core.strategy_narrative import NarrativeStrategy


def _bars(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


def _bias_context(**overrides):
    values = {
        "hourly_range": None,
        "false_breakout_4h": None,
        "true_breakout_15m": None,
        "order_blocks": [],
        "rejection_blocks": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _calc_with_context(ctx, *, margin=1):
    strategy = NarrativeStrategy()
    strategy.htf_score_margin = margin
    strategy._build_htf_context = lambda *args, **kwargs: ctx
    strategy.calc_fvg_regime_1h = lambda df: ("NEUTRAL", "FVG neutral")
    frame = _bars([(1.0, 1.1, 0.9, 1.0)])
    return strategy.calc_narrative(frame, frame, frame)


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


def test_narrative_votes_for_most_recent_active_h1_order_block():
    ctx = _bias_context(
        order_blocks=[
            OrderBlock(side="LONG", top=1.2, bottom=1.1, created_idx=10),
            OrderBlock(side="SHORT", top=1.4, bottom=1.3, created_idx=20),
            OrderBlock(
                side="LONG",
                top=1.6,
                bottom=1.5,
                created_idx=30,
                breaker=True,
            ),
        ],
    )
    side, text = _calc_with_context(ctx)

    assert side == "SHORT"
    assert "OB1H SHORT" in text


def test_h1_premium_discount_uses_latest_m15_close():
    df_1h = _bars([(100.0, 110.0, 90.0, 100.0)])
    df_4h = _bars([(100.0, 111.0, 89.0, 100.0)])

    discount = HtfContext(
        df_1h=df_1h,
        df_4h=df_4h,
        df_15m=_bars([(100.0, 101.0, 94.0, 95.0)]),
    )
    premium = HtfContext(
        df_1h=df_1h,
        df_4h=df_4h,
        df_15m=_bars([(100.0, 106.0, 99.0, 105.0)]),
    )

    assert discount.hourly_range is not None
    assert discount.hourly_range.position == "DISCOUNT"
    assert discount.hourly_range.bias == "LONG"
    assert discount.hourly_range.close == 95.0
    assert premium.hourly_range is not None
    assert premium.hourly_range.position == "PREMIUM"
    assert premium.hourly_range.bias == "SHORT"


def test_h1_premium_discount_votes_with_weight_two():
    ctx = _bias_context(
        hourly_range=HourlyRange(
            high=110.0,
            low=90.0,
            close=95.0,
            position="DISCOUNT",
        )
    )

    side, text = _calc_with_context(ctx, margin=2)

    assert side == "LONG"
    assert "scores L/S=2/0" in text
    assert "H1PD" in text


def test_4h_false_fractal_breakout_votes_with_weight_two():
    ctx = _bias_context(
        false_breakout_4h=FractalBreakout(
            kind="FALSE_BREAK",
            side="SHORT",
            level=1.25,
            level_kind="HIGH",
            bar_index=10,
            bars_ago=0,
            timeframe="4H",
        )
    )

    side, text = _calc_with_context(ctx, margin=2)

    assert side == "SHORT"
    assert "scores L/S=0/2" in text
    assert "4H FALSE_BREAK" in text


def test_15m_true_fractal_breakout_votes_with_weight_one():
    ctx = _bias_context(
        true_breakout_15m=FractalBreakout(
            kind="TRUE_BREAK",
            side="LONG",
            level=1.25,
            level_kind="HIGH",
            bar_index=10,
            bars_ago=0,
            timeframe="15M",
        )
    )

    side, text = _calc_with_context(ctx, margin=1)

    assert side == "LONG"
    assert "scores L/S=1/0" in text
    assert "15M TRUE_BREAK" in text


def test_timeframe_breakout_detection_returns_latest_interaction():
    context = HtfContext.__new__(HtfContext)
    context.pivot_lookback = 1
    false_high = _bars(
        [
            (9.0, 10.0, 8.0, 9.0),
            (10.0, 12.0, 9.0, 11.0),
            (10.0, 11.0, 9.0, 10.0),
            (12.0, 13.0, 10.0, 11.5),
            (11.0, 11.5, 9.5, 10.5),
        ]
    )
    true_high = false_high.copy()
    true_high.loc[3, "close"] = 12.5

    event_4h = context._calc_latest_fractal_breakout(
        false_high,
        timeframe="4H",
    )
    event_15m = context._calc_latest_fractal_breakout(
        true_high,
        timeframe="15M",
    )

    assert event_4h is not None
    assert event_4h.kind == "FALSE_BREAK"
    assert event_4h.side == "SHORT"
    assert event_4h.timeframe == "4H"
    assert event_15m is not None
    assert event_15m.kind == "TRUE_BREAK"
    assert event_15m.side == "LONG"
    assert event_15m.timeframe == "15M"
    assert event_15m.bar_index == 3
    assert event_15m.bars_ago == 1


def test_timeframe_breakout_detection_is_symmetric_for_fractal_lows():
    context = HtfContext.__new__(HtfContext)
    context.pivot_lookback = 1
    false_low = _bars(
        [
            (10.0, 11.0, 9.0, 10.0),
            (9.0, 10.0, 8.0, 9.0),
            (10.0, 11.0, 9.0, 10.0),
            (8.0, 10.0, 7.0, 8.5),
            (9.0, 10.0, 8.5, 9.0),
        ]
    )
    true_low = false_low.copy()
    true_low.loc[3, "close"] = 7.5

    event_4h = context._calc_latest_fractal_breakout(
        false_low,
        timeframe="4H",
    )
    event_15m = context._calc_latest_fractal_breakout(
        true_low,
        timeframe="15M",
    )

    assert event_4h is not None
    assert event_4h.side == "LONG"
    assert event_4h.level_kind == "LOW"
    assert event_15m is not None
    assert event_15m.side == "SHORT"
    assert event_15m.level_kind == "LOW"


def test_newer_opposite_fractal_interaction_invalidates_stale_vote():
    false_then_true = _bars(
        [
            (9.0, 10.0, 8.0, 9.0),
            (10.0, 12.0, 9.0, 11.0),
            (10.0, 11.0, 9.0, 10.0),
            (12.0, 13.0, 10.0, 11.5),
            (12.0, 13.5, 10.5, 12.5),
        ]
    )
    true_then_false = false_then_true.copy()
    true_then_false.loc[3, "close"] = 12.5
    true_then_false.loc[4, "close"] = 11.5
    context = HtfContext(
        df_1h=_bars([(10.0, 11.0, 9.0, 10.0)]),
        df_4h=false_then_true,
        df_15m=true_then_false,
        pivot_lookback=1,
    )

    assert context.false_breakout_4h is None
    assert context.true_breakout_15m is None


def test_rejection_block_stays_broken_after_price_returns_to_zone():
    tracker = RejectionBlockTracker(
        pivot_left=1,
        box_length=20,
        min_intrusion_pct=20.0,
        body_rule="HARD_RIGHT",
    )
    blocks = tracker.build(
        _bars(
            [
                (9.0, 10.0, 8.0, 9.0),
                (9.0, 10.5, 8.5, 9.5),
                (10.0, 12.0, 9.0, 10.0),
                (10.5, 11.0, 9.0, 9.5),
                (11.0, 13.0, 10.5, 12.5),
                (12.0, 12.5, 10.5, 11.0),
            ]
        )
    )

    block = next(rb for rb in blocks if rb.side == "SHORT" and rb.created_idx == 2)

    assert block.zone_high == 12.0
    assert block.broken is True
    assert block.valid is True


def test_narrative_uses_most_recent_valid_h1_rejection_block():
    valid = RejectionBlock(
        side="LONG",
        zone_high=1.2,
        zone_low=1.1,
        created_idx=10,
        midline=1.15,
        wick_ratio=2.0,
        intrusion_pct=30.0,
    )
    broken_newer = RejectionBlock(
        side="SHORT",
        zone_high=1.4,
        zone_low=1.3,
        created_idx=20,
        midline=1.35,
        wick_ratio=2.0,
        intrusion_pct=30.0,
        broken=True,
    )

    side, text = _calc_with_context(
        _bias_context(rejection_blocks=[valid, broken_newer])
    )

    assert side == "LONG"
    assert "RB1H LONG" in text


def test_removed_daily_and_dealing_models_are_absent_from_payload():
    ctx = HtfContext(
        df_1h=_bars([(1.0, 1.1, 0.9, 1.0)]),
        df_4h=_bars([(1.0, 1.1, 0.9, 1.0)]),
        df_15m=_bars([(1.0, 1.1, 0.9, 1.0)]),
    )

    payload = ctx.to_payload()

    assert "daily_range" not in payload
    assert "daily_breakout" not in payload
    assert "dealing_range" not in payload


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
