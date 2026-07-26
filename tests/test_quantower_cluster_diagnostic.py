from __future__ import annotations

import hashlib
import json
import struct
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from tools.validate_quantower_cluster_diagnostic import (
    DiagnosticValidationError,
    _read_dotnet_mvid_bytes,
    validate_export,
)


UTC = timezone.utc
EXPORTER_MVID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _managed_pe_bytes(
    mvid: str = EXPORTER_MVID,
    *,
    cli_in_virtual_tail: bool = False,
    metadata_size: int = 0xB0,
) -> bytes:
    """Build the smallest PE32/CLR image needed to carry a Module MVID."""
    pe_offset = 0x80
    optional_size = 0xE0
    section_header_offset = pe_offset + 4 + 20 + optional_size
    section_rva = 0x2000
    section_raw_pointer = 0x200

    if cli_in_virtual_tail:
        # The apparent CLR payload is deliberately placed in the file overlay.
        # Its RVAs fall inside VirtualSize but outside SizeOfRawData.
        section_virtual_size = 0x600
        section_raw_size = 0x200
        cli_delta = 0x300
        metadata_delta = 0x380
    else:
        section_virtual_size = 0x400
        section_raw_size = 0x400
        cli_delta = 0
        metadata_delta = 0x80

    cli_rva = section_rva + cli_delta
    metadata_rva = section_rva + metadata_delta
    cli_offset = section_raw_pointer + cli_delta
    metadata_offset = section_raw_pointer + metadata_delta
    file_size = max(
        section_raw_pointer + section_raw_size,
        metadata_offset + 0xB0,
    )
    image = bytearray(file_size)

    image[:2] = b"MZ"
    struct.pack_into("<I", image, 0x3C, pe_offset)
    image[pe_offset : pe_offset + 4] = b"PE\x00\x00"
    struct.pack_into(
        "<HHIIIHH",
        image,
        pe_offset + 4,
        0x14C,
        1,
        0,
        0,
        0,
        optional_size,
        0x2102,
    )

    optional_offset = pe_offset + 24
    struct.pack_into("<H", image, optional_offset, 0x10B)
    struct.pack_into("<I", image, optional_offset + 20, section_rva)
    struct.pack_into("<I", image, optional_offset + 24, section_rva)
    struct.pack_into("<I", image, optional_offset + 28, 0x00400000)
    struct.pack_into("<I", image, optional_offset + 32, 0x1000)
    struct.pack_into("<I", image, optional_offset + 36, 0x200)
    struct.pack_into("<I", image, optional_offset + 56, 0x3000)
    struct.pack_into("<I", image, optional_offset + 60, 0x200)
    struct.pack_into("<I", image, optional_offset + 92, 16)
    struct.pack_into(
        "<II",
        image,
        optional_offset + 96 + 14 * 8,
        cli_rva,
        0x48,
    )

    image[section_header_offset : section_header_offset + 8] = b".text\x00\x00\x00"
    struct.pack_into(
        "<IIII",
        image,
        section_header_offset + 8,
        section_virtual_size,
        section_rva,
        section_raw_size,
        section_raw_pointer,
    )
    struct.pack_into("<I", image, section_header_offset + 36, 0x60000020)

    struct.pack_into(
        "<IHHII", image, cli_offset, 0x48, 2, 5, metadata_rva, metadata_size
    )
    struct.pack_into("<I", image, cli_offset + 16, 1)

    version = b"v4.0.30319\x00\x00"
    struct.pack_into(
        "<IHHII", image, metadata_offset, 0x424A5342, 1, 1, 0, len(version)
    )
    image[metadata_offset + 16 : metadata_offset + 16 + len(version)] = version
    stream_header_offset = metadata_offset + 28
    struct.pack_into("<HH", image, stream_header_offset, 0, 2)

    first_stream_header = stream_header_offset + 4
    struct.pack_into("<II", image, first_stream_header, 0x60, 0x26)
    image[first_stream_header + 8 : first_stream_header + 12] = b"#~\x00\x00"

    second_stream_header = first_stream_header + 12
    struct.pack_into("<II", image, second_stream_header, 0xA0, 16)
    image[second_stream_header + 8 : second_stream_header + 16] = b"#GUID\x00\x00\x00"

    tables_offset = metadata_offset + 0x60
    struct.pack_into("<IBBBBQQI", image, tables_offset, 0, 2, 0, 0, 1, 1, 0, 1)
    struct.pack_into("<HHHHH", image, tables_offset + 28, 0, 0, 1, 0, 0)
    image[metadata_offset + 0xA0 : metadata_offset + 0xB0] = UUID(mvid).bytes_le
    return bytes(image)


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
        "schema_version": 2,
        "revision": 0,
        "source_symbol": "EURUSD",
        "symbol": "EURUSD",
        "timeframe": "15m",
        "bar_open": _stamp(bar_open),
        "bar_close": _stamp(bar_close),
        "source_bar_span_ticks": 8_999_999_999,
        "captured_at": _stamp(day_start + timedelta(days=1, hours=1)),
        "historical_available_at": None,
        "availability_semantics": "unknown_historical_latency",
        "finalized": True,
        "finalization_basis": "canonical_bar_close_at_or_before_closed_export_day_end",
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
        "schema_version": 2,
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
        "schema_version": 2,
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
            "exporter_version": "0.2.3",
            "exporter_binary_sha256": "a" * 64,
            "exporter_binary_sha256_status": "AVAILABLE",
            "exporter_module_version_id": EXPORTER_MVID,
        },
        "instrument": {
            "source_symbol": "EURUSD",
            "strategy_symbol": "EURUSD",
            "tick_size": "0.00001",
            "min_volume_analysis_tick_size": "0.00001",
            "symbol_history_type": {"name": "Bid", "numeric_value": 0},
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
            "bar_right_boundary_semantics": (
                "quantower_time_right_equals_open_plus_period_minus_100ns_normalized_to_exclusive_period_end"
            ),
            "price_levels_requested": True,
            "volume_basis": "native_quantower_volume_analysis",
            "timezone": "UTC",
            "unspecified_times_assumed_utc": False,
        },
        "semantics": {
            "factor_name": "FxPro Tick Absorption 15M",
            "classification": "quantower_tick_reconstructed_bidask_history_available",
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
            ),
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


def test_reads_mvid_from_synthetic_managed_pe():
    assert _read_dotnet_mvid_bytes(_managed_pe_bytes(), "managed PE fixture") == (
        EXPORTER_MVID
    )


@pytest.mark.parametrize(
    "data",
    [
        b"not a PE image",
        _managed_pe_bytes()[:0x210],
    ],
    ids=["malformed", "truncated"],
)
def test_rejects_malformed_or_truncated_managed_pe(data):
    with pytest.raises(DiagnosticValidationError):
        _read_dotnet_mvid_bytes(data, "managed PE fixture")


def test_rejects_clr_rva_in_virtual_tail_backed_only_by_overlay():
    with pytest.raises(DiagnosticValidationError):
        _read_dotnet_mvid_bytes(
            _managed_pe_bytes(cli_in_virtual_tail=True),
            "managed PE fixture",
        )


def test_rejects_clr_stream_outside_declared_metadata():
    with pytest.raises(DiagnosticValidationError):
        _read_dotnet_mvid_bytes(
            _managed_pe_bytes(metadata_size=0xA0),
            "managed PE fixture",
        )


def test_valid_diagnostic_export(tmp_path):
    summary = validate_export(_write_export(tmp_path))
    assert summary.bars == 1
    assert summary.price_levels == 2
    assert summary.missing_slots == 95


def test_accepts_unavailable_min_volume_analysis_tick_size(tmp_path):
    root = _write_export(
        tmp_path,
        mutate_manifest=lambda manifest: manifest["instrument"].__setitem__(
            "min_volume_analysis_tick_size", None
        ),
    )
    assert validate_export(root).bars == 1


def test_accepts_exact_exclusive_quantower_bar_right(tmp_path):
    def use_exclusive_boundary(manifest):
        manifest["calculation"]["bar_right_boundary_semantics"] = (
            "quantower_time_right_equals_open_plus_period_used_as_exclusive_period_end"
        )

    def use_exclusive_span(document):
        document["bars"][0]["source_bar_span_ticks"] = 9_000_000_000

    root = _write_export(
        tmp_path,
        mutate_manifest=use_exclusive_boundary,
        mutate_bars=use_exclusive_span,
    )
    assert validate_export(root).bars == 1


def test_requires_external_dll_for_in_memory_assembly(tmp_path):
    def mark_runtime_path_unavailable(manifest):
        manifest["source"]["exporter_binary_sha256"] = None
        manifest["source"]["exporter_binary_sha256_status"] = (
            "UNAVAILABLE_FROM_RUNTIME_ASSEMBLY_PATH"
        )

    root = _write_export(tmp_path, mutate_manifest=mark_runtime_path_unavailable)
    with pytest.raises(DiagnosticValidationError, match="--exporter-dll"):
        validate_export(root)


def test_external_dll_attests_in_memory_assembly(tmp_path):
    def mark_runtime_path_unavailable(manifest):
        manifest["source"]["exporter_binary_sha256"] = None
        manifest["source"]["exporter_binary_sha256_status"] = (
            "UNAVAILABLE_FROM_RUNTIME_ASSEMBLY_PATH"
        )

    root = _write_export(tmp_path, mutate_manifest=mark_runtime_path_unavailable)
    exporter_dll = tmp_path / "FxProTickClusterExporter.dll"
    exporter_bytes = _managed_pe_bytes()
    exporter_dll.write_bytes(exporter_bytes)
    summary = validate_export(root, exporter_dll=exporter_dll)
    assert summary.exporter_binary_sha256 == hashlib.sha256(exporter_bytes).hexdigest()


def test_rejects_external_dll_mvid_mismatch(tmp_path):
    def mark_runtime_path_unavailable(manifest):
        manifest["source"]["exporter_binary_sha256"] = None
        manifest["source"]["exporter_binary_sha256_status"] = (
            "UNAVAILABLE_FROM_RUNTIME_ASSEMBLY_PATH"
        )

    root = _write_export(tmp_path, mutate_manifest=mark_runtime_path_unavailable)
    exporter_dll = tmp_path / "FxProTickClusterExporter.dll"
    exporter_dll.write_bytes(_managed_pe_bytes("11111111-2222-3333-4444-555555555555"))
    with pytest.raises(DiagnosticValidationError, match="MVID does not match"):
        validate_export(root, exporter_dll=exporter_dll)


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
        mutate_manifest=lambda manifest: manifest["semantics"].__setitem__(
            field, value
        ),
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


def test_rejects_sub_millisecond_timestamp_that_would_be_truncated(tmp_path):
    def shift_both_boundaries_by_one_dotnet_tick(document):
        document["bars"][0]["bar_open"] = "2026-07-24T10:00:00.0000001Z"
        document["bars"][0]["bar_close"] = "2026-07-24T10:15:00.0000001Z"

    root = _write_export(tmp_path, mutate_bars=shift_both_boundaries_by_one_dotnet_tick)
    with pytest.raises(DiagnosticValidationError, match="canonical UTC format"):
        validate_export(root)


def test_rejects_unsupported_bar_right_boundary_semantics(tmp_path):
    root = _write_export(
        tmp_path,
        mutate_manifest=lambda manifest: manifest["calculation"].__setitem__(
            "bar_right_boundary_semantics", "approximate_tolerance"
        ),
    )
    with pytest.raises(DiagnosticValidationError, match="right-boundary normalization"):
        validate_export(root)


def test_rejects_unrecognized_source_bar_span(tmp_path):
    root = _write_export(
        tmp_path,
        mutate_bars=lambda document: document["bars"][0].__setitem__(
            "source_bar_span_ticks", 8_999_999_998
        ),
    )
    with pytest.raises(DiagnosticValidationError, match="source_bar_span_ticks"):
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
            lambda document: document["bars"][0]["total"].__setitem__("delta", "999"),
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
            lambda manifest: manifest["semantics"]["evidence"][
                "last_probe"
            ].__setitem__("items", 1),
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
        manifest["causality"]["history_request_started_at"] = "2026-07-24T11:00:00.000Z"
        manifest["causality"]["volume_analysis_finished_at"] = (
            "2026-07-24T12:00:00.000Z"
        )
        manifest["causality"]["snapshot_finished_at"] = "2026-07-24T12:00:01.000Z"

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
    with pytest.raises(
        DiagnosticValidationError, match="production sidecars are rejected"
    ):
        validate_export(root)
