from __future__ import annotations

import json

import pandas as pd
import pytest

from backtest.liquidity_data import (
    FxProLiquidityEventDataset,
    LiquidityDataValidationError,
    seal_recorded_events,
)
from core.fxpro_dom import (
    DomLevel,
    DomSnapshot,
    FxProDomRecorder,
    M15LiquidityAccumulator,
)
from core.liquidity_rejection import (
    LIQUIDITY_DATA_KIND,
    LIQUIDITY_MARKET_TYPE,
    LIQUIDITY_SOURCE,
    LIQUIDITY_VENUE,
    LiquidityRejectionThresholds,
    detect_fxpro_liquidity_rejection,
    seal_liquidity_event,
    validate_liquidity_event,
)


def _event(**overrides):
    payload = {
        "schema_version": 1,
        "revision": 0,
        "timeframe": "15m",
        "source": LIQUIDITY_SOURCE,
        "venue": LIQUIDITY_VENUE,
        "market_type": LIQUIDITY_MARKET_TYPE,
        "data_kind": LIQUIDITY_DATA_KIND,
        "source_symbol": "EURUSD",
        "symbol": "EURUSD",
        "bar_open": "2026-07-27T10:00:00+00:00",
        "bar_close": "2026-07-27T10:15:00+00:00",
        "available_at": "2026-07-27T10:15:01+00:00",
        "finalized": True,
        "coverage_ratio": 0.999,
        "sample_count": 900,
        "changed_snapshots": 101,
        "max_gap_ms": 1500.0,
        "up_ticks": 30,
        "down_ticks": 70,
        "up_distance": 0.0012,
        "down_distance": 0.0014,
        "bid_depleted_volume": 100.0,
        "bid_replenished_volume": 80.0,
        "ask_depleted_volume": 70.0,
        "ask_replenished_volume": 20.0,
        "start_mid": 1.1000,
        "high_mid": 1.1008,
        "low_mid": 1.0990,
        "end_mid": 1.1001,
        "mean_spread": 0.00010,
        "max_spread": 0.00015,
        "min_bid_levels": 5,
        "min_ask_levels": 5,
    }
    payload.update(overrides)
    return seal_liquidity_event(payload)


def _snapshot(
    captured_at: str,
    *,
    bid: float,
    ask: float,
    bid_volume: float,
    ask_volume: float = 100.0,
) -> DomSnapshot:
    timestamp = pd.Timestamp(captured_at)
    return DomSnapshot(
        captured_at=timestamp,
        tick_time=timestamp,
        source_symbol="EURUSD",
        symbol="EURUSD",
        bid=bid,
        ask=ask,
        bids=(
            DomLevel(price=1.0999, volume=bid_volume),
            DomLevel(price=1.0998, volume=50.0),
        ),
        asks=(
            DomLevel(price=1.1001, volume=ask_volume),
            DomLevel(price=1.1002, volume=50.0),
        ),
    )


def test_long_liquidity_rejection_uses_fxpro_depth_without_execution_claim():
    result = detect_fxpro_liquidity_rejection(
        candle={"high": 1.1010, "low": 1.0990, "close": 1.1005},
        candle_open_time="2026-07-27T10:00:00Z",
        event=_event(),
        symbol="EURUSD",
        side="LONG",
        decision_time="2026-07-27T10:15:01Z",
    )

    assert result is not None
    assert result.side == "LONG"
    assert result.quote_pressure == pytest.approx(-0.4)
    assert result.replenishment_ratio == pytest.approx(0.8)
    assert result.replenishment_share == pytest.approx(0.8)
    assert result.source == LIQUIDITY_SOURCE
    assert LIQUIDITY_DATA_KIND == "depth_quotes_no_execution_proof"


def test_liquidity_event_is_fail_closed_for_delay_tamper_and_wrong_source():
    event = _event()
    assert (
        detect_fxpro_liquidity_rejection(
            candle={"high": 1.1010, "low": 1.0990, "close": 1.1005},
            candle_open_time="2026-07-27T10:00:00Z",
            event=event,
            symbol="EURUSD",
            side="LONG",
            decision_time="2026-07-27T10:15:00Z",
        )
        is None
    )

    tampered = dict(event)
    tampered["bid_replenished_volume"] = 999.0
    assert validate_liquidity_event(tampered) is None

    wrong_source = _event(source="quantower_cluster")
    assert validate_liquidity_event(wrong_source) is None


def test_accumulator_counts_only_same_level_deficit_replenishment():
    first = _snapshot(
        "2026-07-27T10:00:00Z",
        bid=1.0999,
        ask=1.1001,
        bid_volume=100.0,
    )
    accumulator = M15LiquidityAccumulator(first)
    accumulator.observe(
        _snapshot(
            "2026-07-27T10:05:00Z",
            bid=1.0998,
            ask=1.1000,
            bid_volume=60.0,
        )
    )
    accumulator.observe(
        _snapshot(
            "2026-07-27T10:14:50Z",
            bid=1.0999,
            ask=1.1001,
            bid_volume=90.0,
        )
    )

    event = accumulator.finalize(
        available_at=pd.Timestamp("2026-07-27T10:15:01Z")
    )
    assert event["bid_depleted_volume"] == pytest.approx(40.0)
    assert event["bid_replenished_volume"] == pytest.approx(30.0)
    assert event["coverage_ratio"] == pytest.approx(890.0 / 900.0)
    assert validate_liquidity_event(event) is not None


class _NoopMt5:
    def market_book_release(self, _symbol):
        return True


def test_recorder_publishes_only_closed_m15_event_asof(tmp_path):
    recorder = FxProDomRecorder(
        mt5_module=_NoopMt5(),
        symbols=["EURUSD"],
        output_dir=tmp_path,
    )
    recorder._observe(
        _snapshot(
            "2026-07-27T10:00:00Z",
            bid=1.0999,
            ask=1.1001,
            bid_volume=100.0,
        )
    )
    recorder._observe(
        _snapshot(
            "2026-07-27T10:14:59Z",
            bid=1.0998,
            ask=1.1000,
            bid_volume=60.0,
        )
    )
    assert (
        recorder.event_asof(
            "EURUSD",
            "2026-07-27T10:00:00Z",
            "2026-07-27T10:15:00Z",
        )
        is None
    )

    recorder._observe(
        _snapshot(
            "2026-07-27T10:15:01Z",
            bid=1.0999,
            ask=1.1001,
            bid_volume=90.0,
        )
    )
    visible = recorder.event_asof(
        "EURUSD",
        "2026-07-27T10:00:00Z",
        "2026-07-27T10:15:01Z",
    )
    assert visible is not None
    assert visible["available_at"] == "2026-07-27T10:15:01+00:00"
    assert (tmp_path / "latest" / "EURUSD.json").is_file()


def test_sealed_sidecar_round_trip_and_tamper_detection(tmp_path):
    recording = tmp_path / "recording"
    events_dir = recording / "events"
    events_dir.mkdir(parents=True)
    event = _event()
    (events_dir / "2026-07-27.jsonl").write_text(
        json.dumps(event, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidecar = seal_recorded_events(recording, tmp_path / "sidecar")
    dataset = FxProLiquidityEventDataset.load(sidecar)

    assert dataset.event_asof(
        "EURUSD",
        "2026-07-27T10:00:00Z",
        "2026-07-27T10:15:00Z",
    ) is None
    assert dataset.event_asof(
        "EURUSD",
        "2026-07-27T10:00:00Z",
        "2026-07-27T10:15:01Z",
    ) == event

    event_path = sidecar / "events.jsonl"
    event_path.write_text(
        event_path.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(LiquidityDataValidationError, match="size mismatch"):
        FxProLiquidityEventDataset.load(sidecar)


def test_thresholds_reject_partial_or_gappy_recordings():
    strict = LiquidityRejectionThresholds()
    for changes in (
        {"coverage_ratio": 0.50},
        {"max_gap_ms": 6000.0},
        {"changed_snapshots": 5},
    ):
        assert (
            detect_fxpro_liquidity_rejection(
                candle={"high": 1.1010, "low": 1.0990, "close": 1.1005},
                candle_open_time="2026-07-27T10:00:00Z",
                event=_event(**changes),
                symbol="EURUSD",
                side="LONG",
                decision_time="2026-07-27T10:15:01Z",
                thresholds=strict,
            )
            is None
        )
