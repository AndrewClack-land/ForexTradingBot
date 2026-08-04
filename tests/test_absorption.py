from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from core.absorption import (
    ABSORPTION_EVENT_SCHEMA_VERSION,
    REQUIRED_POLARITY,
    AbsorptionThresholds,
    detect_footprint_absorption,
)
from core.strategy_narrative import (
    FXPRO_QUOTE_PRESSURE_REJECTION_ENTRY_ENABLED,
    NarrativeStrategy,
)


OPEN_TIME = pd.Timestamp("2026-07-25T10:00:00Z")
DECISION_TIME = OPEN_TIME + pd.Timedelta(minutes=15)


def _event(**overrides):
    event = {
        "schema_version": ABSORPTION_EVENT_SCHEMA_VERSION,
        "source_symbol": "EURUSD",
        "symbol": "EURUSD",
        "timeframe": "15m",
        "bar_open": OPEN_TIME.isoformat(),
        "bar_close": (OPEN_TIME + pd.Timedelta(minutes=15)).isoformat(),
        "available_at": (
            OPEN_TIME + pd.Timedelta(minutes=15)
        ).isoformat(),
        "finalized": True,
        "revision": 0,
        "source": "sealed-test-footprint",
        "venue": "test-venue",
        "polarity": REQUIRED_POLARITY,
        "low_buy_volume": 10.0,
        "low_sell_volume": 150.0,
        "high_buy_volume": 20.0,
        "high_sell_volume": 20.0,
    }
    checksum_override = overrides.pop("checksum", None)
    event.update(overrides)
    checksum_payload = {
        "schema_version": event["schema_version"],
        "revision": event["revision"],
        "timeframe": event["timeframe"],
        "source_symbol": str(event["source_symbol"]).strip().upper(),
        "symbol": str(event["symbol"]).strip().upper(),
        "bar_open": pd.Timestamp(event["bar_open"]).isoformat(),
        "bar_close": pd.Timestamp(event["bar_close"]).isoformat(),
        "available_at": pd.Timestamp(event["available_at"]).isoformat(),
        "finalized": event["finalized"],
        "high_buy_volume": float(event["high_buy_volume"]),
        "high_sell_volume": float(event["high_sell_volume"]),
        "low_buy_volume": float(event["low_buy_volume"]),
        "low_sell_volume": float(event["low_sell_volume"]),
    }
    event["checksum"] = (
        checksum_override
        if checksum_override is not None
        else hashlib.sha256(
            json.dumps(
                checksum_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=True,
            ).encode("utf-8")
        ).hexdigest()
    )
    return event


def _candle(*, close=106.0):
    return {
        "open": 100.0,
        "high": 110.0,
        "low": 90.0,
        "close": close,
    }


def test_long_absorption_requires_lower_edge_selling_and_low_rejection():
    signal = detect_footprint_absorption(
        candle=_candle(),
        candle_open_time=OPEN_TIME,
        event=_event(),
        symbol="EURUSD",
        side="LONG",
        decision_time=DECISION_TIME,
    )

    assert signal is not None
    assert signal.side == "LONG"
    assert signal.imbalance_ratio == pytest.approx(15.0)
    assert signal.rejection_fraction == pytest.approx(0.8)
    json.dumps(signal.to_dict(), allow_nan=False)


def test_short_absorption_is_symmetric_at_upper_edge():
    signal = detect_footprint_absorption(
        candle=_candle(close=94.0),
        candle_open_time=OPEN_TIME,
        event=_event(
            low_buy_volume=20.0,
            low_sell_volume=20.0,
            high_buy_volume=150.0,
            high_sell_volume=10.0,
        ),
        symbol="EURUSD",
        side="SHORT",
        decision_time=DECISION_TIME,
    )

    assert signal is not None
    assert signal.side == "SHORT"
    assert signal.imbalance_ratio == pytest.approx(15.0)
    assert signal.rejection_fraction == pytest.approx(0.8)


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": 2},
        {"source_symbol": "BTCUSDT"},
        {"symbol": "GBPUSD"},
        {"timeframe": "1m"},
        {"finalized": False},
        {"revision": 1},
        {"revision": False},
        {"source": ""},
        {"venue": ""},
        {"checksum": "not-a-sha256"},
        {"checksum": "A" * 64},
        {"polarity": "unknown"},
        {"bar_open": "2026-07-25T10:15:00Z"},
        {"bar_open": "2026-07-25T10:00:00"},
        {"bar_close": "2026-07-25T10:14:00Z"},
        {"available_at": "2026-07-25T10:14:00Z"},
        {"available_at": "2026-07-25T10:15:01Z"},
        {"high_buy_volume": float("nan")},
        {"low_sell_volume": -1.0},
    ],
)
def test_malformed_or_unverified_event_fails_closed(overrides):
    assert (
        detect_footprint_absorption(
            candle=_candle(),
            candle_open_time=OPEN_TIME,
            event=_event(**overrides),
            symbol="EURUSD",
            side="LONG",
            decision_time=DECISION_TIME,
        )
        is None
    )


@pytest.mark.parametrize(
    "field",
    (
        "schema_version",
        "source_symbol",
        "finalized",
        "revision",
        "source",
        "venue",
        "polarity",
        "bar_open",
        "bar_close",
        "available_at",
        "checksum",
        "high_buy_volume",
        "high_sell_volume",
        "low_buy_volume",
        "low_sell_volume",
    ),
)
def test_missing_contract_field_fails_closed(field):
    event = _event()
    event.pop(field)

    assert (
        detect_footprint_absorption(
            candle=_candle(),
            candle_open_time=OPEN_TIME,
            event=event,
            symbol="EURUSD",
            side="LONG",
            decision_time=DECISION_TIME,
        )
        is None
    )


def test_event_close_and_availability_cannot_exceed_decision_time():
    assert (
        detect_footprint_absorption(
            candle=_candle(),
            candle_open_time=OPEN_TIME,
            event=_event(),
            symbol="EURUSD",
            side="LONG",
            decision_time=DECISION_TIME - pd.Timedelta(seconds=1),
        )
        is None
    )
    assert (
        detect_footprint_absorption(
            candle=_candle(),
            candle_open_time=OPEN_TIME,
            event=_event(
                available_at=(
                    DECISION_TIME + pd.Timedelta(seconds=1)
                ).isoformat()
            ),
            symbol="EURUSD",
            side="LONG",
            decision_time=DECISION_TIME,
        )
        is None
    )


def test_formula_fails_closed_when_any_component_is_too_weak():
    cases = (
        _event(low_buy_volume=80.0),
        _event(low_sell_volume=79.0),
        _event(low_sell_volume=0.0),
    )
    for event in cases:
        assert (
            detect_footprint_absorption(
                candle=_candle(),
                candle_open_time=OPEN_TIME,
                event=event,
                symbol="EURUSD",
                side="LONG",
                decision_time=DECISION_TIME,
            )
            is None
        )

    assert (
        detect_footprint_absorption(
            candle=_candle(close=99.0),
            candle_open_time=OPEN_TIME,
            event=_event(),
            symbol="EURUSD",
            side="LONG",
            decision_time=DECISION_TIME,
        )
        is None
    )


def test_audited_formula_boundaries_are_inclusive_except_close_midpoint():
    boundary = _event(
        low_buy_volume=40.0,
        low_sell_volume=80.0,
    )
    assert (
        detect_footprint_absorption(
            candle=_candle(close=100.0001),
            candle_open_time=OPEN_TIME,
            event=boundary,
            symbol="EURUSD",
            side="LONG",
            decision_time=DECISION_TIME,
        )
        is not None
    )
    assert (
        detect_footprint_absorption(
            candle=_candle(close=100.0),
            candle_open_time=OPEN_TIME,
            event=boundary,
            symbol="EURUSD",
            side="LONG",
            decision_time=DECISION_TIME,
        )
        is None
    )


def test_zero_opposite_edge_stays_finite_and_json_safe():
    signal = detect_footprint_absorption(
        candle=_candle(),
        candle_open_time=OPEN_TIME,
        event=_event(low_buy_volume=0.0),
        symbol="EURUSD",
        side="LONG",
        decision_time=DECISION_TIME,
    )

    assert signal is not None
    assert signal.imbalance_ratio == 1_000_000_000.0
    json.dumps(signal.to_dict(), allow_nan=False)


def test_detector_rejects_legacy_schema_and_edge_aliases():
    canonical = _event()
    legacy = {
        "schema": "footprint-15m/v1",
        "symbol": canonical["symbol"],
        "timeframe": canonical["timeframe"],
        "bar_open_time": canonical["bar_open"],
        "bar_close_time": canonical["bar_close"],
        "finalized": True,
        "revision": 0,
        "source": canonical["source"],
        "source_bar_hash": canonical["checksum"],
        "lower_buy_volume": canonical["low_buy_volume"],
        "lower_sell_volume": canonical["low_sell_volume"],
        "upper_buy_volume": canonical["high_buy_volume"],
        "upper_sell_volume": canonical["high_sell_volume"],
    }

    signal = detect_footprint_absorption(
        candle=_candle(),
        candle_open_time=OPEN_TIME,
        event=legacy,
        symbol="EURUSD",
        side="LONG",
        decision_time=DECISION_TIME,
    )

    assert signal is None


def test_invalid_thresholds_are_rejected():
    with pytest.raises(ValueError, match="exceed 1"):
        AbsorptionThresholds(min_imbalance_ratio=1.0)
    with pytest.raises(ValueError, match="edge_volume"):
        AbsorptionThresholds(min_edge_volume=-1.0)


def test_absorption_remains_archived_outside_production_strategy():
    strategy = NarrativeStrategy()

    assert not hasattr(strategy, "absorption_15m_entry_enabled")
    assert not hasattr(strategy, "trigger_15m_absorption")

    # Quote Pressure is a separate broker-DOM challenger, never an Absorption
    # alias or fallback. Whether its entries are live is an operator decision,
    # so pin the wiring — the flag is its own and nothing else drives it —
    # instead of asserting the ambient deployment's value.
    assert (
        strategy.quote_pressure_rejection_15m_entry_enabled
        is FXPRO_QUOTE_PRESSURE_REJECTION_ENTRY_ENABLED
    )
    assert not hasattr(strategy, "absorption_event_dataset")


def test_checksum_must_match_normalized_event_content():
    event = _event()
    event["low_sell_volume"] += 1.0

    assert (
        detect_footprint_absorption(
            candle=_candle(),
            candle_open_time=OPEN_TIME,
            event=event,
            symbol="EURUSD",
            side="LONG",
            decision_time=DECISION_TIME,
        )
        is None
    )
