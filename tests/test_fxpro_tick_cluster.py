"""Contract tests for the in-process FxPro MT5 bid-tick cluster capture."""

from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest

import core.fxpro_tick_cluster as tick_cluster
from backtest.fxpro_cluster_data import (
    ClusterDataValidationError,
    FxProClusterEventDataset,
)
from core.fxpro_cluster_rejection import (
    CLUSTER_CLASSIFICATION_MT5_BIDASK,
    CLUSTER_CLASSIFICATION_QUANTOWER,
    CLUSTER_SOURCE_MT5_BIDASK,
    CLUSTER_SOURCE_QUANTOWER,
    ClusterRejectionThresholds,
    detect_fxpro_cluster_rejection,
    seal_cluster_event,
    validate_cluster_event,
)
from core.fxpro_tick_cluster import (
    FxProTickClusterRecorder,
    TickClusterSymbol,
    aggregate_bid_tick_cluster,
)


TICK_SIZE = 0.00001
BAR_OPEN = pd.Timestamp("2026-07-24T10:00:00Z")
BAR_CLOSE = BAR_OPEN + pd.Timedelta(minutes=15)
BASE_MS = int(BAR_OPEN.value // 1_000_000)


def _px(ticks: int) -> float:
    """Exact price on the tick grid, free of accumulated float drift."""

    return float(Decimal(ticks) * Decimal(repr(TICK_SIZE)))


def _tick(offset_ms: int, price_ticks: int) -> dict:
    bid = _px(price_ticks)
    return {
        "time_msc": BASE_MS + offset_ms,
        "bid": bid,
        "ask": _px(price_ticks + 2),
        "flags": 2,
    }


def _long_rejection_ticks() -> list[dict]:
    """Deep sell-side wick into the low, then a fast recovery to the close.

    The descent walks the grid one tick at a time so the bottom edge collects
    many down-ticks, while the recovery jumps in wide steps so almost no
    up-tick lands back inside that edge.
    """

    ticks = [_tick(-1_000, 108_051)]  # pre-bar seed only
    offset = 0
    ticks.append(_tick(offset, 108_050))
    for level in range(108_049, 107_999, -1):
        offset += 1_000
        ticks.append(_tick(offset, level))
    for level in (108_030, 108_060, 108_100, 108_080):
        offset += 1_000
        ticks.append(_tick(offset, level))
    return ticks


def _aggregate(ticks, *, available_at=BAR_CLOSE, seed_bid=None):
    return aggregate_bid_tick_cluster(
        symbol="EURUSD",
        source_symbol="EURUSD",
        ticks=ticks,
        bar_open=BAR_OPEN,
        tick_size=TICK_SIZE,
        available_at=available_at,
        seed_bid=seed_bid,
    )


def _candle(event) -> dict:
    return dict(event["ohlc"])


# ---------------------------------------------------------------------------
# Aggregation semantics
# ---------------------------------------------------------------------------


def test_bid_direction_classifies_up_and_down_ticks():
    ticks = [
        _tick(0, 108_000),
        _tick(1_000, 108_010),  # up   -> buy
        _tick(2_000, 108_010),  # flat -> unclassified
        _tick(3_000, 108_005),  # down -> sell
    ]
    result = _aggregate(ticks)

    assert result.ok, result.reason
    total = result.event["total"]
    assert result.tick_count == 4
    # The first in-bar tick has no reference bid, so it stays unclassified
    # together with the repeated quote.
    assert total["volume"] == 4.0
    assert total["buy_volume"] == 1.0
    assert total["sell_volume"] == 1.0
    assert total["delta"] == 0.0
    assert result.classified_ticks == 2


def test_seed_bid_classifies_the_first_in_bar_tick():
    ticks = [_tick(0, 108_010), _tick(1_000, 108_000)]
    seeded = _aggregate(ticks, seed_bid=_px(108_000))

    assert seeded.ok
    # 108000 -> 108010 is now an up-tick because the seed supplied a reference.
    assert seeded.event["total"]["buy_volume"] == 1.0
    assert seeded.event["total"]["sell_volume"] == 1.0


def test_pre_bar_ticks_seed_direction_without_contributing_volume():
    ticks = [
        {"time_msc": BASE_MS - 5_000, "bid": _px(108_000), "ask": _px(108_002)},
        _tick(0, 108_010),
        _tick(1_000, 108_020),
    ]
    result = _aggregate(ticks)

    assert result.ok
    assert result.tick_count == 2
    assert result.event["total"]["volume"] == 2.0
    # Both in-bar ticks are up-ticks once the pre-bar quote seeds the reference.
    assert result.event["total"]["buy_volume"] == 2.0


def test_ticks_at_or_after_bar_close_are_discarded():
    close_ms = int(pd.Timedelta(minutes=15).total_seconds() * 1000)
    ticks = [
        _tick(0, 108_000),
        _tick(1_000, 108_010),
        _tick(close_ms, 108_500),
        _tick(close_ms + 1_000, 108_900),
    ]
    result = _aggregate(ticks)

    assert result.ok
    assert result.tick_count == 2
    assert result.event["ohlc"]["high"] == pytest.approx(_px(108_010))


def test_flat_bar_without_range_is_data_unavailable():
    ticks = [_tick(offset, 108_000) for offset in (0, 1_000, 2_000)]
    result = _aggregate(ticks)

    assert not result.ok
    assert result.event is None
    assert "range" in result.reason


def test_empty_bar_is_data_unavailable():
    result = _aggregate([])

    assert not result.ok
    assert result.event is None


def test_ohlc_is_derived_from_bid_prices():
    result = _aggregate(_long_rejection_ticks())

    assert result.ok
    ohlc = result.event["ohlc"]
    assert ohlc["open"] == pytest.approx(_px(108_050))
    assert ohlc["high"] == pytest.approx(_px(108_100))
    assert ohlc["low"] == pytest.approx(_px(108_000))
    assert ohlc["close"] == pytest.approx(_px(108_080))


def test_price_levels_are_a_strictly_increasing_grid():
    result = _aggregate(_long_rejection_ticks())
    levels = result.event["price_levels"]

    ticks_seen = [level["price_ticks"] for level in levels]
    assert ticks_seen == sorted(set(ticks_seen))
    for level in levels:
        assert level["price"] == pytest.approx(
            level["price_ticks"] * TICK_SIZE, abs=TICK_SIZE * 1e-6
        )


# ---------------------------------------------------------------------------
# Provenance contract
# ---------------------------------------------------------------------------


def test_event_declares_the_mt5_reconstruction_and_no_execution_proof():
    event = _aggregate(_long_rejection_ticks()).event
    normalized = validate_cluster_event(event)

    assert normalized is not None
    assert normalized["source"] == CLUSTER_SOURCE_MT5_BIDASK
    assert normalized["classification"] == CLUSTER_CLASSIFICATION_MT5_BIDASK
    assert normalized["venue"] == "FxPro"
    assert normalized["execution_proof"] is False
    assert normalized["aggressor_polarity_proven"] is False
    assert normalized["availability_mode"] == "forward_observed"
    # The counts must never be readable as traded lots.
    assert normalized["volume_measure"] == "bid_tick_count_no_traded_volume"
    assert normalized["price_basis"] == "bid"


def test_source_and_classification_must_stay_paired():
    event = _aggregate(_long_rejection_ticks()).event
    swapped = dict(event)
    swapped["classification"] = CLUSTER_CLASSIFICATION_QUANTOWER
    assert validate_cluster_event(seal_cluster_event(swapped)) is None

    relabelled = dict(event)
    relabelled["source"] = CLUSTER_SOURCE_QUANTOWER
    assert validate_cluster_event(seal_cluster_event(relabelled)) is None

    unknown = dict(event)
    unknown["source"] = "dxfeed_cme_globex_tape"
    unknown["classification"] = "exchange_reported_aggressor"
    assert validate_cluster_event(seal_cluster_event(unknown)) is None


def test_available_at_is_never_earlier_than_the_bar_close():
    result = _aggregate(
        _long_rejection_ticks(),
        available_at=BAR_OPEN + pd.Timedelta(minutes=1),
    )

    assert result.ok
    assert pd.Timestamp(result.event["available_at"]) == BAR_CLOSE
    assert result.event["capture_lag_ms"] == 0


def test_capture_lag_records_a_late_publication():
    late = BAR_CLOSE + pd.Timedelta(seconds=3)
    result = _aggregate(_long_rejection_ticks(), available_at=late)

    assert result.event["capture_lag_ms"] == 3_000


def test_tampering_with_a_sealed_event_breaks_the_checksum():
    event = dict(_aggregate(_long_rejection_ticks()).event)
    event["total"] = {**event["total"], "buy_volume": 999.0}

    assert validate_cluster_event(event) is None


# ---------------------------------------------------------------------------
# Detector integration
# ---------------------------------------------------------------------------


def test_detector_accepts_a_captured_long_rejection():
    event = _aggregate(_long_rejection_ticks()).event
    signal = detect_fxpro_cluster_rejection(
        candle=_candle(event),
        candle_open_time=BAR_OPEN,
        event=event,
        symbol="EURUSD",
        side="LONG",
        decision_time=BAR_CLOSE + pd.Timedelta(seconds=30),
    )

    assert signal is not None
    assert signal.side == "LONG"
    assert signal.source == CLUSTER_SOURCE_MT5_BIDASK
    assert signal.availability_mode == "forward_observed"
    assert signal.classification_ratio == pytest.approx(1.0)
    payload = signal.to_dict()
    assert payload["execution_proof_claimed"] is False
    assert payload["aggressor_polarity_claimed"] is False


def test_forward_observed_capture_needs_no_research_override():
    event = _aggregate(_long_rejection_ticks()).event
    signal = detect_fxpro_cluster_rejection(
        candle=_candle(event),
        candle_open_time=BAR_OPEN,
        event=event,
        symbol="EURUSD",
        side="LONG",
        decision_time=BAR_CLOSE + pd.Timedelta(seconds=30),
        allow_research_assumption=False,
    )

    assert signal is not None


def test_detector_still_fails_closed_before_availability():
    event = _aggregate(
        _long_rejection_ticks(),
        available_at=BAR_CLOSE + pd.Timedelta(seconds=30),
    ).event
    signal = detect_fxpro_cluster_rejection(
        candle=_candle(event),
        candle_open_time=BAR_OPEN,
        event=event,
        symbol="EURUSD",
        side="LONG",
        decision_time=BAR_CLOSE + pd.Timedelta(seconds=5),
    )

    assert signal is None


def test_same_venue_candle_matches_without_relaxing_the_tolerance():
    """The whole point of in-process capture: strict tolerance still passes."""

    event = _aggregate(_long_rejection_ticks()).event
    strict = ClusterRejectionThresholds(candle_match_tolerance_ticks=0)
    signal = detect_fxpro_cluster_rejection(
        candle=_candle(event),
        candle_open_time=BAR_OPEN,
        event=event,
        symbol="EURUSD",
        side="LONG",
        decision_time=BAR_CLOSE + pd.Timedelta(seconds=30),
        thresholds=strict,
    )

    assert signal is not None


# ---------------------------------------------------------------------------
# Sidecar publication
# ---------------------------------------------------------------------------


class _FakeMt5:
    COPY_TICKS_INFO = 1

    def __init__(self, ticks):
        self._ticks = ticks
        self.calls: list[tuple] = []
        self.selected: list[str] = []

    def symbol_select(self, symbol, enable):
        self.selected.append(symbol)
        return True

    def symbol_info(self, symbol):
        return SimpleNamespace(point=TICK_SIZE, digits=5)

    def copy_ticks_range(self, symbol, date_from, date_to, flags):
        self.calls.append((symbol, date_from, date_to, flags))
        return [SimpleNamespace(**tick) for tick in self._ticks]

    def last_error(self):
        return (0, "ok")


@pytest.fixture()
def frozen_now(monkeypatch):
    now = BAR_CLOSE + pd.Timedelta(minutes=1)
    monkeypatch.setattr(tick_cluster, "_utc_now", lambda: now)
    return now


def _recorder(tmp_path, mt5_module, **kwargs):
    return FxProTickClusterRecorder(
        mt5_module=mt5_module,
        symbols=[TickClusterSymbol("EURUSD", "EURUSD")],
        output_dir=tmp_path / "capture",
        sidecar_dir=tmp_path / "sidecar",
        **kwargs,
    )


def test_recorder_publishes_a_loadable_forward_observed_sidecar(
    tmp_path, frozen_now
):
    mt5_module = _FakeMt5(_long_rejection_ticks())
    recorder = _recorder(tmp_path, mt5_module)

    recorder.poll_once()

    dataset = FxProClusterEventDataset.load(tmp_path / "sidecar")
    assert dataset.research_only is False
    assert len(dataset.events) == 1
    assert dataset.manifest["source"] == CLUSTER_SOURCE_MT5_BIDASK
    assert dataset.manifest["availability_modes"] == ["forward_observed"]

    event = dataset.event_asof("EURUSD", BAR_OPEN, frozen_now)
    assert event is not None
    assert event["symbol"] == "EURUSD"


def test_recorder_pulls_the_closed_bar_with_a_seed_lookbehind(
    tmp_path, frozen_now
):
    mt5_module = _FakeMt5(_long_rejection_ticks())
    recorder = _recorder(tmp_path, mt5_module)

    recorder.poll_once()

    # The symbol must be pulled into Market Watch before ticks are requested,
    # because the recorder can start before the executor ever touches it.
    assert mt5_module.selected == ["EURUSD"]
    assert len(mt5_module.calls) == 1
    _, date_from, date_to, flags = mt5_module.calls[0]
    assert flags == _FakeMt5.COPY_TICKS_INFO
    assert pd.Timestamp(date_to).tz_localize("UTC") == BAR_CLOSE
    assert pd.Timestamp(date_from).tz_localize("UTC") < BAR_OPEN


def test_recorder_does_not_republish_the_same_bar(tmp_path, frozen_now):
    mt5_module = _FakeMt5(_long_rejection_ticks())
    recorder = _recorder(tmp_path, mt5_module)

    recorder.poll_once()
    recorder.poll_once()

    assert len(mt5_module.calls) == 1
    dataset = FxProClusterEventDataset.load(tmp_path / "sidecar")
    assert len(dataset.events) == 1


def test_recorder_reloads_its_window_after_a_restart(tmp_path, frozen_now):
    mt5_module = _FakeMt5(_long_rejection_ticks())
    _recorder(tmp_path, mt5_module).poll_once()

    revived = _recorder(tmp_path, _FakeMt5(_long_rejection_ticks()))

    assert revived._last_bar["EURUSD"] == BAR_OPEN
    assert len(revived._window) == 1


def test_sidecar_publication_leaves_no_unsealed_jsonl(tmp_path, frozen_now):
    recorder = _recorder(tmp_path, _FakeMt5(_long_rejection_ticks()))
    recorder.poll_once()

    sidecar = tmp_path / "sidecar"
    assert sorted(path.name for path in sidecar.glob("*.jsonl")) == [
        "events.jsonl"
    ]
    # A stale temporary file must never be picked up as a sealed shard.
    assert not list(sidecar.glob("*.tmp"))


def test_raw_ticks_are_archived_against_the_event_hash(tmp_path, frozen_now):
    recorder = _recorder(tmp_path, _FakeMt5(_long_rejection_ticks()))
    recorder.poll_once()

    archive = (
        tmp_path
        / "capture"
        / "ticks"
        / BAR_OPEN.strftime("%Y-%m-%d")
        / "EURUSD.jsonl"
    )
    record = json.loads(archive.read_text(encoding="utf-8").splitlines()[0])
    dataset = FxProClusterEventDataset.load(tmp_path / "sidecar")

    assert record["source_bar_hash"] == dataset.events[0]["source_bar_hash"]
    assert record["checksum"] == dataset.events[0]["checksum"]


def test_capture_survives_a_bar_mt5_cannot_serve(tmp_path, frozen_now):
    class _EmptyMt5(_FakeMt5):
        def copy_ticks_range(self, symbol, date_from, date_to, flags):
            self.calls.append((symbol, date_from, date_to, flags))
            return None

    recorder = _recorder(tmp_path, _EmptyMt5([]))
    recorder.poll_once()

    # Missing data is DATA_UNAVAILABLE: nothing is published and nothing raises.
    assert not (tmp_path / "sidecar" / "manifest.json").exists()


# ---------------------------------------------------------------------------
# Dataset-level source pinning
# ---------------------------------------------------------------------------


def test_dataset_refuses_a_manifest_whose_source_is_unknown(
    tmp_path, frozen_now
):
    recorder = _recorder(tmp_path, _FakeMt5(_long_rejection_ticks()))
    recorder.poll_once()

    manifest_path = tmp_path / "sidecar" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"] = "dxfeed_cme_globex_tape"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ClusterDataValidationError, match="reconstruction"):
        FxProClusterEventDataset.load(tmp_path / "sidecar")


def test_dataset_refuses_events_that_disagree_with_the_manifest_source(
    tmp_path, frozen_now
):
    recorder = _recorder(tmp_path, _FakeMt5(_long_rejection_ticks()))
    recorder.poll_once()

    manifest_path = tmp_path / "sidecar" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"] = CLUSTER_SOURCE_QUANTOWER
    manifest["classification"] = CLUSTER_CLASSIFICATION_QUANTOWER
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ClusterDataValidationError):
        FxProClusterEventDataset.load(tmp_path / "sidecar")
