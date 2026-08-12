from __future__ import annotations

import pandas as pd

from backtest.h1_m1_trigger import (
    H1Context,
    M1ReclaimConfig,
    detect_m1_reclaim,
)


def _frame(rows):
    return pd.DataFrame(
        [
            {
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
            }
            for row in rows
        ],
        index=pd.DatetimeIndex(
            [row[0] for row in rows],
            name="timestamp",
        ),
    )


def _long_context(*, expiry="2026-01-01T00:10:00Z"):
    return H1Context.from_signal(
        symbol="EURUSD",
        signal={
            "signal": "ENTER",
            "side": "LONG",
            "trigger_kind": "rejection_block_1h",
            "entry_price": 1.0995,
            "entry_min": 1.0990,
            "entry_max": 1.1000,
            "stop_price": 1.0950,
            "tp_prices": [1.1050, 1.1100, 1.1150],
        },
        decision_time="2026-01-01T00:01:00Z",
        expires_at=expiry,
    )


def _short_context():
    return H1Context.from_signal(
        symbol="EURUSD",
        signal={
            "signal": "ENTER",
            "side": "SHORT",
            "trigger_kind": "order_block_1h",
            "entry_price": 1.1005,
            "entry_min": 1.1000,
            "entry_max": 1.1010,
            "stop_price": 1.1050,
            "tp_prices": [1.0950, 1.0900, 1.0850],
        },
        decision_time="2026-01-01T00:01:00Z",
        expires_at="2026-01-01T00:10:00Z",
    )


def test_completed_m1_reclaim_enters_only_on_next_bar_open():
    frame = _frame([
        ("2026-01-01T00:00:00Z", 1.1010, 1.1020, 1.1005, 1.1010),
        ("2026-01-01T00:01:00Z", 1.1005, 1.1020, 1.0995, 1.1015),
        ("2026-01-01T00:02:00Z", 1.1005, 1.1010, 1.1000, 1.1007),
    ])

    result = detect_m1_reclaim(_long_context(), frame)

    assert result["status"] == "TRIGGERED"
    assert result["trigger_time"] == "2026-01-01T00:02:00+00:00"
    assert result["entry_time"] == "2026-01-01T00:02:00+00:00"
    assert result["entry_price"] == 1.1005
    assert result["signal"]["parent_trigger_kind"] == (
        "rejection_block_1h"
    )
    assert result["signal"]["trigger_kind"].endswith("__m1_reclaim")


def test_bar_opening_before_context_decision_is_never_a_trigger():
    frame = _frame([
        ("2026-01-01T00:00:00Z", 1.1005, 1.1020, 1.0990, 1.1015),
        ("2026-01-01T00:01:00Z", 1.1015, 1.1020, 1.1010, 1.1012),
        ("2026-01-01T00:02:00Z", 1.1012, 1.1020, 1.1005, 1.1010),
    ])

    result = detect_m1_reclaim(_long_context(), frame)

    assert result["status"] == "NO_TRIGGER"


def test_pre_entry_stop_touch_invalidates_before_reclaim():
    frame = _frame([
        ("2026-01-01T00:00:00Z", 1.1010, 1.1020, 1.1005, 1.1010),
        ("2026-01-01T00:01:00Z", 1.1005, 1.1020, 1.0940, 1.1015),
        ("2026-01-01T00:02:00Z", 1.1005, 1.1010, 1.1000, 1.1007),
    ])

    result = detect_m1_reclaim(_long_context(), frame)

    assert result["status"] == "INVALIDATED_STOP"


def test_trigger_close_at_exclusive_expiry_is_rejected():
    frame = _frame([
        ("2026-01-01T00:00:00Z", 1.1010, 1.1020, 1.1005, 1.1010),
        ("2026-01-01T00:01:00Z", 1.1005, 1.1020, 1.0995, 1.1015),
        ("2026-01-01T00:02:00Z", 1.1005, 1.1010, 1.1000, 1.1007),
    ])

    result = detect_m1_reclaim(
        _long_context(expiry="2026-01-01T00:02:00Z"),
        frame,
    )

    assert result["status"] == "NO_TRIGGER"


def test_next_open_chasing_beyond_extension_cap_is_rejected():
    frame = _frame([
        ("2026-01-01T00:00:00Z", 1.1010, 1.1020, 1.1005, 1.1010),
        ("2026-01-01T00:01:00Z", 1.1005, 1.1020, 1.0995, 1.1015),
        ("2026-01-01T00:02:00Z", 1.1030, 1.1040, 1.1025, 1.1035),
    ])

    result = detect_m1_reclaim(
        _long_context(),
        frame,
        config=M1ReclaimConfig(max_entry_extension_r=0.25),
    )

    assert result["status"] == "ENTRY_EXTENSION_REJECT"


def test_short_reclaim_is_directionally_symmetric():
    frame = _frame([
        ("2026-01-01T00:00:00Z", 1.0990, 1.0995, 1.0980, 1.0990),
        ("2026-01-01T00:01:00Z", 1.1005, 1.1005, 1.0985, 1.0990),
        ("2026-01-01T00:02:00Z", 1.0995, 1.1000, 1.0990, 1.0992),
    ])

    result = detect_m1_reclaim(_short_context(), frame)

    assert result["status"] == "TRIGGERED"
    assert result["entry_price"] == 1.0995
    assert result["signal"]["side"] == "SHORT"
