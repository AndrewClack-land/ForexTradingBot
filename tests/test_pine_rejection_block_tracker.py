from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from core.htf_context import (
    HtfContext,
    PineRejectionBlockTracker,
    RejectionBlockTracker,
)


def _frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["open", "high", "low", "close"],
        index=pd.date_range(
            "2026-01-01T00:00:00Z",
            periods=len(rows),
            freq="h",
        ),
    )


def _bullish_rows() -> list[tuple[float, float, float, float]]:
    return [
        (10.0, 10.8, 9.4, 10.0),
        (10.0, 10.8, 9.3, 10.0),
        (10.0, 10.8, 9.2, 10.0),
        (10.0, 10.4, 8.0, 10.0),
        (10.1, 10.7, 9.1, 10.1),
        (10.1, 10.7, 9.2, 10.1),
        # Registration bar overlaps the zone, but is not an eligible retest.
        (10.2, 10.5, 9.5, 10.2),
    ]


def _bearish_rows() -> list[tuple[float, float, float, float]]:
    return [
        (10.0, 10.5, 9.2, 10.0),
        (10.0, 10.6, 9.2, 10.0),
        (10.0, 10.7, 9.2, 10.0),
        (10.0, 12.0, 9.6, 10.0),
        (9.9, 10.8, 9.2, 9.9),
        (9.9, 10.7, 9.2, 9.9),
        # Registration bar overlaps the zone, but is not an eligible retest.
        (9.8, 10.2, 9.3, 9.8),
    ]


def _block_at(blocks, created_idx: int, side: str):
    return next(
        block
        for block in blocks
        if block.created_idx == created_idx and block.side == side
    )


def test_pivot_is_available_only_after_three_right_bars_and_registration_is_not_retest():
    tracker = PineRejectionBlockTracker(min_tick=0.01)
    rows = _bullish_rows()

    assert tracker.build(_frame(rows[:-1])) == []

    registered = tracker.build(_frame(rows))
    block = _block_at(registered, 3, "LONG")
    assert block.available_idx == 6
    assert block.pivot_time == "2026-01-01T03:00:00+00:00"
    # H1 indices are bar-open timestamps; causal availability is its close.
    assert block.available_time == "2026-01-01T07:00:00+00:00"
    assert block.zone_low == pytest.approx(8.0)
    assert block.zone_high == pytest.approx(10.0)
    assert block.valid is True
    assert block.baseline_unverified is False
    assert block.retested is False
    assert block.retest_idx is None

    rows.append((10.2, 10.3, 9.8, 10.1))
    retested = _block_at(tracker.build(_frame(rows)), 3, "LONG")
    assert retested.retested is True
    assert retested.retest_idx == 7
    assert retested.to_dict()["detector_version"] == "pine-v1-causal"


def test_doji_wick_ratio_uses_min_tick_floor_and_opposite_wick_dominance():
    doji = [
        (10.0, 10.1, 9.9, 10.0),
        (10.0, 10.1, 9.9, 10.0),
        (10.0, 10.1, 9.9, 10.0),
        (10.0, 10.05, 9.85, 10.0),
        (10.0, 10.1, 9.9, 10.0),
        (10.0, 10.1, 9.9, 10.0),
        (10.0, 10.1, 9.9, 10.0),
    ]
    assert PineRejectionBlockTracker(min_tick=0.1).build(_frame(doji)) == []
    accepted = PineRejectionBlockTracker(min_tick=0.05).build(_frame(doji))
    assert _block_at(accepted, 3, "LONG").wick_ratio == pytest.approx(3.0)

    dominant_opposite = list(doji)
    dominant_opposite[3] = (10.0, 10.4, 9.7, 10.0)
    blocks = PineRejectionBlockTracker(min_tick=0.05).build(
        _frame(dominant_opposite)
    )
    assert not any(block.side == "LONG" for block in blocks)


def test_same_side_separation_uses_wilder_atr_at_pivot_and_never_expires():
    rows = [(10.0, 11.0, 9.0, 10.0)] * 36
    rows[15] = (10.0, 11.0, 8.0, 10.0)
    rows[23] = (10.0, 11.0, 8.5, 10.0)
    rows[31] = (10.0, 11.0, 6.0, 10.0)

    tracker = PineRejectionBlockTracker(min_tick=0.01)
    tracker.build(_frame(rows[:14]))
    blocks = tracker.build(_frame(rows))
    long_pivots = [
        block.created_idx
        for block in blocks
        if block.side == "LONG"
    ]
    assert long_pivots == [15, 31]

    first = _block_at(blocks, 15, "LONG")
    last = _block_at(blocks, 31, "LONG")
    assert first.atr_at_pivot == pytest.approx(29.0 / 14.0)
    assert first.separation_price is None
    assert last.separation_price == pytest.approx(2.0)
    assert last.separation_atr == pytest.approx(
        2.0 / float(last.atr_at_pivot)
    )
    assert first.valid is True
    assert first.broken is False


def test_strict_close_break_has_precedence_over_same_bar_touch():
    tracker = PineRejectionBlockTracker(min_tick=0.01)

    broken_rows = _bullish_rows() + [(10.0, 10.2, 7.5, 7.9)]
    tracker.build(_frame(broken_rows[:4]))
    tracker.build(_frame(broken_rows[:7]))
    broken = _block_at(tracker.build(_frame(broken_rows)), 3, "LONG")
    assert broken.broken is True
    assert broken.broken_idx == 7
    assert broken.retested is False

    wick_only_rows = _bullish_rows() + [(10.0, 10.2, 7.5, 8.5)]
    tracker = PineRejectionBlockTracker(min_tick=0.01)
    tracker.build(_frame(wick_only_rows[:4]))
    tracker.build(_frame(wick_only_rows[:7]))
    wick_only = _block_at(
        tracker.build(_frame(wick_only_rows)),
        3,
        "LONG",
    )
    assert wick_only.broken is False
    assert wick_only.retested is True
    assert wick_only.retest_idx == 7


def test_bearish_zone_is_symmetric_and_only_first_post_available_retest_is_kept():
    tracker = PineRejectionBlockTracker(min_tick=0.01)
    rows = _bearish_rows()
    tracker.build(_frame(rows[:4]))
    registered = tracker.build(_frame(rows))
    assert _block_at(registered, 3, "SHORT").retested is False
    rows.extend(
        [
            (9.9, 10.1, 9.4, 9.9),
            (9.9, 10.2, 9.4, 9.9),
        ]
    )

    block = _block_at(tracker.build(_frame(rows)), 3, "SHORT")
    assert block.zone_low == pytest.approx(10.0)
    assert block.zone_high == pytest.approx(12.0)
    assert block.available_idx == 6
    assert block.retested is True
    assert block.retest_idx == 7
    assert block.retest_time == "2026-01-01T08:00:00+00:00"
    assert block.broken is False


def test_bootstrap_blocks_are_separation_only_and_fail_closed():
    tracker = PineRejectionBlockTracker(
        min_tick=0.01,
        symbol="TEST",
    )

    block = _block_at(
        tracker.build(_frame(_bullish_rows())),
        3,
        "LONG",
    )

    assert block.baseline_unverified is True
    assert block.valid is False
    assert block.retested is True
    state = tracker.export_state()
    assert state["bootstrap_watermark"] == (
        "2026-01-01T07:00:00+00:00"
    )


def test_stateful_block_survives_when_pivot_leaves_rolling_window():
    rows = _bullish_rows() + [
        (10.3, 10.6, 10.2, 10.4)
        for _ in range(12)
    ]
    tracker = PineRejectionBlockTracker(
        min_tick=0.01,
        symbol="TEST",
    )
    tracker.build(_frame(rows[:4]))
    formed = tracker.build(_frame(rows[:9]))
    block = _block_at(formed, 3, "LONG")
    assert block.valid is True
    assert block.retested is False

    shifted = tracker.build(_frame(rows).iloc[9:])

    persisted = next(
        item
        for item in shifted
        if item.pivot_time == "2026-01-01T03:00:00+00:00"
        and item.side == "LONG"
    )
    assert persisted.valid is True
    assert persisted.retested is False


def test_closed_m1_consumes_first_touch_after_available_time():
    rows = _bullish_rows()
    tracker = PineRejectionBlockTracker(
        min_tick=0.01,
        symbol="TEST",
    )
    tracker.build(_frame(rows[:4]))
    tracker.build(_frame(rows))
    m1 = pd.DataFrame(
        {
            "open": [10.2, 10.1],
            "high": [10.3, 10.2],
            "low": [10.1, 9.9],
            "close": [10.2, 10.05],
        },
        index=pd.date_range(
            "2026-01-01T07:00:00Z",
            periods=2,
            freq="min",
        ),
    )

    consumed = _block_at(
        tracker.build(_frame(rows), df_1m=m1),
        3,
        "LONG",
    )

    assert consumed.retested is True
    assert consumed.retest_idx is None
    assert consumed.retest_time == "2026-01-01T07:02:00+00:00"


# Touch first, strict distal break two bars later. A catch-up batch must not
# let the later break cancel the earlier touch.
_CATCH_UP_TOUCH = (10.0, 10.2, 9.0, 9.5)
_CATCH_UP_BREAK = (9.5, 9.6, 7.5, 7.9)


def _catch_up_tracker() -> PineRejectionBlockTracker:
    tracker = PineRejectionBlockTracker(min_tick=0.01, symbol="TEST")
    base = _bullish_rows()
    tracker.build(_frame(base[:4]))
    tracker.build(_frame(base))
    return tracker


def test_h1_catch_up_batch_keeps_earlier_retest_before_later_break():
    """Several new H1 bars in one build must replay in chronological order."""

    sequential = _catch_up_tracker()
    rows = _bullish_rows() + [_CATCH_UP_TOUCH]
    sequential.build(_frame(rows))
    rows.append(_CATCH_UP_BREAK)
    one_at_a_time = _block_at(sequential.build(_frame(rows)), 3, "LONG")

    batched_tracker = _catch_up_tracker()
    batched = _block_at(
        batched_tracker.build(
            _frame(_bullish_rows() + [_CATCH_UP_TOUCH, _CATCH_UP_BREAK])
        ),
        3,
        "LONG",
    )

    # The earlier touch survives, and the later break is still recorded.
    assert one_at_a_time.retested is True
    assert one_at_a_time.retest_idx == 7
    assert one_at_a_time.retest_time == "2026-01-01T08:00:00+00:00"
    assert one_at_a_time.broken is True
    assert one_at_a_time.broken_idx == 8

    # Batch size must not change the outcome at all.
    assert batched.retested == one_at_a_time.retested
    assert batched.retest_idx == one_at_a_time.retest_idx
    assert batched.retest_time == one_at_a_time.retest_time
    assert batched.broken == one_at_a_time.broken
    assert batched.broken_idx == one_at_a_time.broken_idx


def test_h1_catch_up_batch_keeps_earlier_m1_consumption_before_later_break():
    """An M1 touch inside an earlier H1 bar outranks a later H1 break."""

    m1 = pd.DataFrame(
        {
            "open": [10.0, 9.8],
            "high": [10.1, 9.9],
            "low": [9.6, 9.0],
            "close": [9.8, 9.5],
        },
        index=pd.date_range("2026-01-01T07:00:00Z", periods=2, freq="min"),
    )

    batched = _block_at(
        _catch_up_tracker().build(
            _frame(_bullish_rows() + [_CATCH_UP_TOUCH, _CATCH_UP_BREAK]),
            df_1m=m1,
        ),
        3,
        "LONG",
    )

    # Consumed by the closed M1 inside bar 7, not erased by the bar 8 break.
    assert batched.retested is True
    assert batched.retest_idx is None
    assert batched.retest_time == "2026-01-01T07:01:00+00:00"
    assert batched.broken is True
    assert batched.broken_idx == 8


def test_same_bar_break_still_outranks_its_own_m1_touch_in_a_batch():
    """The intra-bar rule is unchanged: a bar's break beats its own M1 touch."""

    m1 = pd.DataFrame(
        {
            "open": [9.5, 9.4],
            "high": [9.6, 9.5],
            "low": [9.0, 7.5],
            "close": [9.4, 7.9],
        },
        index=pd.date_range("2026-01-01T08:00:00Z", periods=2, freq="min"),
    )

    block = _block_at(
        _catch_up_tracker().build(
            _frame(_bullish_rows() + [(10.2, 10.4, 10.1, 10.3), _CATCH_UP_BREAK]),
            df_1m=m1,
        ),
        3,
        "LONG",
    )

    assert block.broken is True
    assert block.broken_idx == 8
    assert block.retested is False
    assert block.retest_time is None


def test_state_roundtrip_is_json_safe_atomic_and_keeps_watermarks():
    rows = _bullish_rows()
    tracker = PineRejectionBlockTracker(
        min_tick=0.01,
        symbol="TEST",
    )
    tracker.build(_frame(rows[:4]))
    tracker.build(_frame(rows))
    payload = json.loads(json.dumps(tracker.export_state()))

    restored = PineRejectionBlockTracker(
        min_tick=0.01,
        symbol="TEST",
    )
    restored.import_state(payload)
    assert restored.export_state() == payload
    assert restored.dirty is False

    corrupt = json.loads(json.dumps(payload))
    corrupt["blocks"][0]["zone_high"] = float("nan")
    before = restored.export_state()
    with pytest.raises(ValueError, match="non-finite"):
        restored.import_state(corrupt)
    assert restored.export_state() == before


def test_htf_context_selects_version_explicitly_and_validates_min_tick():
    frame = _frame(_bullish_rows())

    legacy = HtfContext(frame, None, None)
    assert legacy.rb_detector_version == "legacy"
    assert isinstance(legacy.rb_tracker, RejectionBlockTracker)

    pine = HtfContext(
        frame,
        None,
        None,
        rb_detector_version="pine-v1-causal",
        min_tick=0.01,
    )
    assert isinstance(pine.rb_tracker, PineRejectionBlockTracker)
    assert _block_at(pine.rejection_blocks, 3, "LONG").available_idx == 6

    with pytest.raises(ValueError, match="finite positive"):
        HtfContext(frame, None, None, min_tick=0.0)
    with pytest.raises(ValueError, match="rb_detector_version"):
        HtfContext(frame, None, None, rb_detector_version="unknown")


def test_wilder_atr_seed_and_recurrence_are_not_simple_rolling_mean():
    high = np.asarray([2.0] * 14 + [4.0], dtype=float)
    low = np.asarray([0.0] * 15, dtype=float)
    close = np.asarray([1.0] * 15, dtype=float)

    atr = PineRejectionBlockTracker._wilder_atr(
        high,
        low,
        close,
        period=14,
    )
    assert np.all(np.isnan(atr[:13]))
    assert atr[13] == pytest.approx(2.0)
    assert atr[14] == pytest.approx((2.0 * 13.0 + 4.0) / 14.0)
