from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from tools.validate_quantower_cluster_diagnostic import (
    DiagnosticValidationError,
    validate_export,
)


UTC = timezone.utc


def _stamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _write_export(tmp_path, *, mutate_manifest=None, mutate_bars=None):
    root = tmp_path / "diagnostic"
    root.mkdir()
    day_start = datetime(2026, 7, 24, tzinfo=UTC)
    bar_open = day_start + timedelta(hours=10)
    bar_close = bar_open + timedelta(minutes=15)
    levels = [
        {
            "price": "1.10000",
            "price_ticks": 110000,
            "volume": "5",
            "trades": 5,
            "buy_volume": "2",
            "sell_volume": "3",
            "buy_trades": 2,
            "sell_trades": 3,
            "delta": "-1",
        },
        {
            "price": "1.10001",
            "price_ticks": 110001,
            "volume": "7",
            "trades": 7,
            "buy_volume": "3",
            "sell_volume": "4",
            "buy_trades": 3,
            "sell_trades": 4,
            "delta": "-1",
        },
    ]
    bar = {
        "schema_version": 1,
        "revision": 0,
        "source_symbol": "EURUSD",
        "symbol": "EURUSD",
        "timeframe": "15m",
        "bar_open": _stamp(bar_open),
        "bar_close": _stamp(bar_close),
        "captured_at": _stamp(day_start + timedelta(days=1, hours=1)),
        "historical_available_at": None,
        "availability_semantics": "unknown_historical_latency",
        "finalized": True,
        "finalization_basis": "bar_close_before_closed_export_day",
        "ohlc": {
            "open": "1.10000",
            "high": "1.10100",
            "low": "1.09900",
            "close": "1.10050",
        },
        "bar_volume": "12",
        "bar_ticks": 12,
        "total": {
            "volume": "12",
            "trades": 12,
            "buy_volume": "5",
            "sell_volume": "7",
            "buy_trades": 5,
            "sell_trades": 7,
            "delta": "-2",
        },
        "price_levels": levels,
        "validation": {
            "price_levels_status": "AVAILABLE",
            "level_count": 2,
            "sum_levels_matches_total": True,
            "unclassified_volume": "0",
            "unclassified_trades": 0,
            "level_sums": {
                "volume": "12",
                "trades": 12,
                "buy_volume": "5",
                "sell_volume": "7",
                "buy_trades": 5,
                "sell_trades": 7,
                "delta": "-2",
            },
        },
    }
    bars_document = {
        "schema": "forexbot.quantower-cluster-diagnostic-bars",
        "schema_version": 1,
        "export_id": "11111111-2222-3333-4444-555555555555",
        "bars": [bar],
    }
    if mutate_bars is not None:
        mutate_bars(bars_document)
    bars_raw = json.dumps(bars_document, indent=2).encode("utf-8")
    digest = hashlib.sha256(bars_raw).hexdigest()
    (root / "bars.json").write_bytes(bars_raw)

    expected_slots = {day_start + timedelta(minutes=15 * slot) for slot in range(96)}
    actual_slots = {
        datetime.fromisoformat(item["bar_open"].replace("Z", "+00:00"))
        for item in bars_document["bars"]
    }
    missing = [_stamp(value) for value in sorted(expected_slots - actual_slots)]
    bars_with_levels = sum(bool(item["price_levels"]) for item in bars_document["bars"])
    manifest = {
        "schema": "forexbot.quantower-cluster-diagnostic",
        "schema_version": 1,
        "export_id": bars_document["export_id"],
        "created_at": _stamp(day_start + timedelta(days=1, hours=1)),
        "day_utc": "2026-07-24",
        "interval": {
            "from_inclusive": _stamp(day_start),
            "to_exclusive": _stamp(day_start + timedelta(days=1)),
        },
        "timeframe": "15m",
        "source": {
            "platform": "Quantower",
            "platform_version": "1.146.14.0",
            "connector_vendor": "FxPro",
            "connector_vendor_semantics": "operator_label_not_connection_identity",
            "exporter_version": "0.1.0",
            "exporter_binary_sha256": "a" * 64,
        },
        "instrument": {
            "source_symbol": "EURUSD",
            "strategy_symbol": "EURUSD",
            "tick_size": "0.00001",
            "min_volume_analysis_tick_size": "0.00001",
            "symbol_history_type": {"name": "BidAsk", "numeric_value": 4},
            "symbol_volume_type": {"name": "Ticks", "numeric_value": 1},
            "symbol_delta_calculation_type": {
                "name": "TickDirection",
                "numeric_value": 1,
            },
            "allow_calculate_realtime_ticks": True,
            "allow_calculate_realtime_volume": False,
            "allow_calculate_realtime_trades": False,
        },
        "calculation": {
            "period": "15m",
            "chart_aggregation": "M15 BidAsk",
            "price_levels_requested": True,
            "volume_basis": "native_quantower_volume_analysis",
            "timezone": "UTC",
            "unspecified_times_assumed_utc": False,
        },
        "semantics": {
            "factor_name": "FxPro Tick Absorption 15M",
            "classification": "quantower_bidask_tick_reconstructed",
            "classification_status": "UNVERIFIED",
            "buy_volume": "native Quantower VolumeAnalysisItem.BuyVolume",
            "sell_volume": "native Quantower VolumeAnalysisItem.SellVolume",
            "trades": "classified tick-events, not exchange executions",
            "exchange_executions_proven": False,
            "aggressor_polarity_proven": False,
            "evidence": {
                "last_probe": {
                    "status": "OK",
                    "requested_history_type": {"name": "Last", "numeric_value": 3},
                    "items": 0,
                    "typed_items": 0,
                    "positive_size_items": 0,
                    "aggressor_buy": 0,
                    "aggressor_sell": 0,
                    "aggressor_unknown": 0,
                    "error_type": None,
                },
                "bid_ask_probe": {
                    "status": "OK",
                    "requested_history_type": {"name": "BidAsk", "numeric_value": 4},
                    "items": 100,
                    "typed_items": 100,
                    "positive_size_items": 0,
                    "aggressor_buy": 0,
                    "aggressor_sell": 0,
                    "aggressor_unknown": 0,
                    "error_type": None,
                },
            },
        },
        "causality": {
            "history_request_started_at": _stamp(
                day_start + timedelta(days=1, minutes=59)
            ),
            "history_request_start_semantics": "indicator_on_init_lower_bound",
            "volume_analysis_finished_at": _stamp(
                day_start + timedelta(days=1, hours=1)
            ),
            "snapshot_finished_at": _stamp(
                day_start + timedelta(days=1, hours=1, seconds=1)
            ),
            "historical_availability_semantics": (
                "snapshot_observed_at_not_historical_feed_latency"
            )
        },
        "coverage": {
            "expected_utc_slots": 96,
            "bars_returned": len(bars_document["bars"]),
            "bars_with_price_levels": bars_with_levels,
            "missing_bar_opens": missing,
            "null_price_level_bars": 0,
            "empty_price_level_bars": 0,
            "validation_status": "WARN",
            "eligible_for_sidecar_conversion": False,
        },
        "files": [
            {
                "path": "bars.json",
                "size": len(bars_raw),
                "sha256": digest,
                "records": len(bars_document["bars"]),
            }
        ],
        "content_sha256": digest,
    }
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return root


def test_valid_diagnostic_export(tmp_path):
    summary = validate_export(_write_export(tmp_path))
    assert summary.bars == 1
    assert summary.price_levels == 2
    assert summary.missing_slots == 95


def test_rejects_payload_tampering_after_manifest(tmp_path):
    root = _write_export(tmp_path)
    with (root / "bars.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(DiagnosticValidationError, match="size.*does not match"):
        validate_export(root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("classification_status", "VERIFIED"),
        ("exchange_executions_proven", True),
        ("aggressor_polarity_proven", True),
    ],
)
def test_rejects_unsafe_provenance_claims(tmp_path, field, value):
    root = _write_export(
        tmp_path,
        mutate_manifest=lambda manifest: manifest["semantics"].__setitem__(field, value),
    )
    with pytest.raises(DiagnosticValidationError, match=field):
        validate_export(root)


def test_rejects_unsorted_price_levels_even_when_resealed(tmp_path):
    def reverse_levels(document):
        document["bars"][0]["price_levels"].reverse()

    root = _write_export(tmp_path, mutate_bars=reverse_levels)
    with pytest.raises(DiagnosticValidationError, match="strictly increasing"):
        validate_export(root)


def test_rejects_non_m15_bar_even_when_resealed(tmp_path):
    def move_close(document):
        document["bars"][0]["bar_close"] = "2026-07-24T10:16:00.000Z"

    root = _write_export(tmp_path, mutate_bars=move_close)
    with pytest.raises(DiagnosticValidationError, match="15 minutes"):
        validate_export(root)


def test_rejects_buy_sell_split_larger_than_total(tmp_path):
    def overclassify(document):
        total = document["bars"][0]["total"]
        total["buy_volume"] = "13"
        total["sell_volume"] = "7"
        total["delta"] = "6"

    root = _write_export(tmp_path, mutate_bars=overclassify)
    with pytest.raises(DiagnosticValidationError, match="exceeds Total volume"):
        validate_export(root)


def test_rejects_bar_symbol_not_bound_to_manifest(tmp_path):
    root = _write_export(
        tmp_path,
        mutate_bars=lambda document: document["bars"][0].__setitem__(
            "source_symbol", "GBPUSD"
        ),
    )
    with pytest.raises(DiagnosticValidationError, match="does not match manifest"):
        validate_export(root)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda document: document["bars"][0]["validation"].__setitem__(
                "price_levels_status", "DATA_UNAVAILABLE"
            ),
            "requires AVAILABLE",
        ),
        (
            lambda document: document["bars"][0]["validation"][
                "level_sums"
            ].__setitem__("volume", "999"),
            "does not match computed levels",
        ),
        (
            lambda document: document["bars"][0]["total"].__setitem__(
                "delta", "999"
            ),
            "buy_volume - sell_volume",
        ),
    ],
)
def test_rejects_false_internal_validation_claims(tmp_path, mutate, match):
    root = _write_export(tmp_path, mutate_bars=mutate)
    with pytest.raises(DiagnosticValidationError, match=match):
        validate_export(root)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda manifest: manifest.__setitem__("schema_version", True),
            "must be an integer",
        ),
        (
            lambda manifest: manifest["files"][0].__setitem__("records", 1.0),
            "must be an integer",
        ),
        (
            lambda manifest: manifest["source"].pop("exporter_binary_sha256"),
            "exporter_binary_sha256",
        ),
    ],
)
def test_rejects_json_type_and_provenance_bypasses(tmp_path, mutate, match):
    root = _write_export(tmp_path, mutate_manifest=mutate)
    with pytest.raises(DiagnosticValidationError, match=match):
        validate_export(root)


def test_rejects_skipped_origin_probe(tmp_path):
    def skip_probe(manifest):
        manifest["semantics"]["classification"] = "unknown"
        manifest["semantics"]["evidence"]["last_probe"]["status"] = "SKIPPED"

    root = _write_export(tmp_path, mutate_manifest=skip_probe)
    with pytest.raises(DiagnosticValidationError, match="successful Last and BidAsk"):
        validate_export(root)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda manifest: manifest["instrument"]["symbol_history_type"].__setitem__(
                "name", "Ask"
            ),
            "does not match Last/BidAsk evidence",
        ),
        (
            lambda manifest: manifest["semantics"]["evidence"][
                "bid_ask_probe"
            ].__setitem__("typed_items", 0),
            "does not match Last/BidAsk evidence",
        ),
        (
            lambda manifest: manifest["calculation"].__setitem__(
                "unspecified_times_assumed_utc", True
            ),
            "requires explicit UTC",
        ),
    ],
)
def test_rejects_incomplete_bidask_and_utc_evidence(tmp_path, mutate, match):
    root = _write_export(tmp_path, mutate_manifest=mutate)
    with pytest.raises(DiagnosticValidationError, match=match):
        validate_export(root)


def test_rejects_current_partial_utc_day(tmp_path):
    def make_current_day(manifest):
        manifest["created_at"] = "2026-07-24T12:00:00.000Z"
        manifest["causality"]["history_request_started_at"] = (
            "2026-07-24T11:00:00.000Z"
        )
        manifest["causality"]["volume_analysis_finished_at"] = (
            "2026-07-24T12:00:00.000Z"
        )
        manifest["causality"]["snapshot_finished_at"] = (
            "2026-07-24T12:00:01.000Z"
        )

    root = _write_export(tmp_path, mutate_manifest=make_current_day)
    with pytest.raises(DiagnosticValidationError, match="day must be closed"):
        validate_export(root)


def test_rejects_production_sidecar_schema(tmp_path):
    root = _write_export(
        tmp_path,
        mutate_manifest=lambda manifest: manifest.__setitem__(
            "schema", "forexbot.absorption-15m"
        ),
    )
    with pytest.raises(DiagnosticValidationError, match="production sidecars are rejected"):
        validate_export(root)
