from __future__ import annotations

from datetime import datetime, timezone

from core.execution_quality import (
    ExecutionQualityProfile,
    PointInTimeNewsCalendar,
    assess_execution_quality,
)


def _profile(
    *,
    news=True,
    rollover_hour=22,
    before=15,
    after=15,
) -> ExecutionQualityProfile:
    return ExecutionQualityProfile.from_mapping({
        "schema": "execution-quality-profile-v1",
        "profile_id": "fxpro-forward-w32",
        "measured_from": "forward broker quotes",
        "created_at_utc": "2026-08-10T00:00:00Z",
        "rollover_timezone": "UTC",
        "rollover_hour": rollover_hour,
        "rollover_minute": 0,
        "rollover_before_minutes": before,
        "rollover_after_minutes": after,
        "news_filter_enabled": news,
        "news_before_minutes": 15,
        "news_after_minutes": 15,
        "news_min_impact": 3,
        "symbols": {
            "EURUSD": {
                "currencies": ["EUR", "USD"],
                "max_spread_price": 0.00020,
                "max_spread_r": 0.03,
            },
        },
    })


def _calendar() -> PointInTimeNewsCalendar:
    return PointInTimeNewsCalendar.from_mapping({
        "schema": "point-in-time-news-calendar-v1",
        "calendar_id": "calendar-history",
        "source": "archived provider revisions",
        "captured_from": "immutable event revision log",
        "revisions": [
            {
                "event_id": "us-cpi",
                "revision_id": "r1",
                "known_at_utc": "2026-08-01T00:00:00Z",
                "scheduled_at_utc": "2026-08-11T14:00:00Z",
                "currencies": ["USD"],
                "impact": 3,
                "title": "CPI",
            },
            {
                "event_id": "us-cpi",
                "revision_id": "r2",
                "known_at_utc": "2026-08-11T15:00:00Z",
                "scheduled_at_utc": "2026-08-11T16:00:00Z",
                "currencies": ["USD"],
                "impact": 3,
                "title": "CPI rescheduled",
            },
            {
                "event_id": "cad-event",
                "revision_id": "r1",
                "known_at_utc": "2026-08-01T00:00:00Z",
                "scheduled_at_utc": "2026-08-11T14:00:00Z",
                "currencies": ["CAD"],
                "impact": 3,
                "title": "Canada event",
            },
        ],
    })


def _signal():
    return {
        "side": "LONG",
        "entry_price": 1.10,
        "stop_price": 1.09,
    }


def test_spread_gate_uses_broker_bid_ask_and_risk_distance():
    result = assess_execution_quality(
        _profile(news=False),
        symbol="EURUSD",
        signal=_signal(),
        bid=1.1000,
        ask=1.1003,
        observed_at_utc=datetime(
            2026,
            8,
            11,
            12,
            tzinfo=timezone.utc,
        ),
        news_calendar=None,
    )

    assert result["allowed"] is False
    assert result["reason"] == "SPREAD"
    assert result["spread_price"] > result["max_spread_price"]


def test_rollover_blackout_handles_window_around_cutoff():
    result = assess_execution_quality(
        _profile(news=False),
        symbol="EURUSD",
        signal=_signal(),
        bid=1.1000,
        ask=1.1001,
        observed_at_utc=datetime(
            2026,
            8,
            11,
            21,
            50,
            tzinfo=timezone.utc,
        ),
        news_calendar=None,
    )

    assert result["allowed"] is False
    assert result["reason"] == "ROLLOVER"


def test_news_filter_uses_only_revision_known_at_decision_time():
    calendar = _calendar()
    early = calendar.blocking_events(
        as_of_utc=datetime(
            2026,
            8,
            11,
            13,
            55,
            tzinfo=timezone.utc,
        ),
        currencies=["EUR", "USD"],
        before_minutes=15,
        after_minutes=15,
        min_impact=3,
    )
    late = calendar.blocking_events(
        as_of_utc=datetime(
            2026,
            8,
            11,
            15,
            55,
            tzinfo=timezone.utc,
        ),
        currencies=["EUR", "USD"],
        before_minutes=15,
        after_minutes=15,
        min_impact=3,
    )

    assert [event.revision_id for event in early] == ["r1"]
    assert [event.revision_id for event in late] == ["r2"]


def test_enabled_news_filter_fails_closed_without_calendar():
    result = assess_execution_quality(
        _profile(news=True),
        symbol="EURUSD",
        signal=_signal(),
        bid=1.1000,
        ask=1.1001,
        observed_at_utc=datetime(
            2026,
            8,
            11,
            12,
            tzinfo=timezone.utc,
        ),
        news_calendar=None,
    )

    assert result["allowed"] is False
    assert result["reason"] == "NEWS_DATA_UNAVAILABLE"


def test_known_high_impact_event_blocks_only_related_symbol():
    result = assess_execution_quality(
        _profile(news=True),
        symbol="EURUSD",
        signal=_signal(),
        bid=1.1000,
        ask=1.1001,
        observed_at_utc=datetime(
            2026,
            8,
            11,
            13,
            55,
            tzinfo=timezone.utc,
        ),
        news_calendar=_calendar(),
    )

    assert result["allowed"] is False
    assert result["reason"] == "NEWS"
    assert [event["event_id"] for event in result["blocking_events"]] == [
        "us-cpi"
    ]
