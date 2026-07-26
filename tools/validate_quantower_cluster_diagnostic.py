"""Fail-closed validator for raw Quantower cluster diagnostic exports.

This schema is intentionally distinct from ``forexbot.absorption-15m``.
Passing this validator proves structural integrity and conservative provenance;
it does not prove exchange executions, aggressor polarity, or historical feed
latency, and therefore does not make the export a production WFO sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import UUID


SCHEMA = "forexbot.quantower-cluster-diagnostic"
BARS_SCHEMA = "forexbot.quantower-cluster-diagnostic-bars"
SCHEMA_VERSION = 2
M15 = timedelta(minutes=15)
UTC = timezone.utc
EXCLUSIVE_BAR_RIGHT_SEMANTICS = (
    "quantower_time_right_equals_open_plus_period_used_as_exclusive_period_end"
)
INCLUSIVE_BAR_RIGHT_SEMANTICS = (
    "quantower_time_right_equals_open_plus_period_minus_100ns_normalized_to_exclusive_period_end"
)
M15_TICKS = 9_000_000_000
SOURCE_SPAN_TICKS_BY_SEMANTICS = {
    EXCLUSIVE_BAR_RIGHT_SEMANTICS: M15_TICKS,
    INCLUSIVE_BAR_RIGHT_SEMANTICS: M15_TICKS - 1,
}
UTC_MILLISECOND_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z"
)


class DiagnosticValidationError(ValueError):
    """Raised when a diagnostic export is incomplete or internally unsafe."""


@dataclass(frozen=True)
class ValidationSummary:
    export_id: str
    day_utc: str
    bars: int
    price_levels: int
    missing_slots: int
    content_sha256: str
    exporter_binary_sha256: str


def _fail(location: str, message: str) -> "None":
    raise DiagnosticValidationError(f"{location}: {message}")


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DiagnosticValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(raw: bytes, location: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(location, f"invalid UTF-8 JSON ({type(exc).__name__})")
    if not isinstance(value, dict):
        _fail(location, "top-level value must be an object")
    return value


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(location, "must be an object")
    return value


def _sequence(value: Any, location: str) -> Sequence[Any]:
    if not isinstance(value, list):
        _fail(location, "must be an array")
    return value


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(location, "must be a non-empty string")
    return value


def _integer(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(location, f"must be an integer >= {minimum}")
    return value


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        _fail(location, "must be a boolean")
    return value


def _enum_snapshot(value: Any, location: str) -> tuple[str, int]:
    enum_value = _mapping(value, location)
    name = _string(_require(enum_value, "name", location), f"{location}.name")
    numeric = _integer(
        _require(enum_value, "numeric_value", location),
        f"{location}.numeric_value",
        minimum=-(2**31),
    )
    if numeric > 2**31 - 1:
        _fail(f"{location}.numeric_value", "outside Int32 range")
    return name, numeric


def _sha256(value: Any, location: str) -> str:
    digest = _string(value, location)
    if len(digest) != 64 or digest != digest.lower() or any(
        character not in "0123456789abcdef" for character in digest
    ):
        _fail(location, "must be a lowercase SHA-256 hex digest")
    return digest


def _unpack_from(data: bytes, format_string: str, offset: int, location: str) -> tuple[Any, ...]:
    size = struct.calcsize(format_string)
    if offset < 0 or offset + size > len(data):
        _fail(location, "truncated binary structure")
    return struct.unpack_from(format_string, data, offset)


def _require_binary_range(
    offset: int,
    size: int,
    container_start: int,
    container_end: int,
    location: str,
    description: str,
) -> int:
    if (
        size < 0
        or offset < container_start
        or offset > container_end
        or size > container_end - offset
    ):
        _fail(location, f"{description} is outside its declared container")
    return offset + size


def _rva_to_file_range(
    rva: int,
    size: int,
    sections: Sequence[tuple[int, int, int, int]],
    location: str,
) -> tuple[int, int]:
    if size <= 0:
        _fail(location, "PE directory has an empty range")

    matches: list[tuple[int, int]] = []
    for virtual_address, _virtual_size, raw_pointer, raw_size in sections:
        delta = rva - virtual_address
        if delta < 0 or delta > raw_size or size > raw_size - delta:
            continue
        offset = raw_pointer + delta
        matches.append((offset, offset + size))

    if not matches:
        _fail(location, f"RVA range 0x{rva:x}+0x{size:x} is outside section raw data")
    if len(matches) != 1:
        _fail(location, f"RVA range 0x{rva:x}+0x{size:x} maps ambiguously")
    return matches[0]


def _read_dotnet_mvid_bytes(data: bytes, location: str) -> str:
    if len(data) < 0x40 or data[:2] != b"MZ":
        _fail(location, "is not a PE image")
    (pe_offset,) = _unpack_from(data, "<I", 0x3C, location)
    if pe_offset < 0x40:
        _fail(location, "has an invalid PE header offset")
    if data[pe_offset : pe_offset + 4] != b"PE\x00\x00":
        _fail(location, "has no PE signature")

    coff_offset = pe_offset + 4
    _require_binary_range(coff_offset, 20, 0, len(data), location, "COFF header")
    (_, section_count, _, _, _, optional_size, _) = _unpack_from(
        data, "<HHIIIHH", coff_offset, location
    )
    if section_count < 1:
        _fail(location, "has no PE sections")

    optional_offset = coff_offset + 20
    optional_end = _require_binary_range(
        optional_offset,
        optional_size,
        0,
        len(data),
        location,
        "optional header",
    )
    (optional_magic,) = _unpack_from(data, "<H", optional_offset, location)
    if optional_magic == 0x10B:
        number_of_directories_offset = optional_offset + 92
        data_directories_offset = optional_offset + 96
    elif optional_magic == 0x20B:
        number_of_directories_offset = optional_offset + 108
        data_directories_offset = optional_offset + 112
    else:
        _fail(location, "has an unsupported PE optional-header magic")

    _require_binary_range(
        number_of_directories_offset,
        4,
        optional_offset,
        optional_end,
        location,
        "NumberOfRvaAndSizes",
    )
    (number_of_directories,) = _unpack_from(
        data, "<I", number_of_directories_offset, location
    )
    if number_of_directories <= 14:
        _fail(location, "optional header has no CLR data directory")

    cli_directory_offset = data_directories_offset + 14 * 8
    _require_binary_range(
        cli_directory_offset,
        8,
        optional_offset,
        optional_end,
        location,
        "CLR data-directory entry",
    )
    cli_rva, cli_size = _unpack_from(data, "<II", cli_directory_offset, location)
    if cli_rva == 0 or cli_size < 0x48:
        _fail(location, "has no CLR header")

    section_offset = optional_offset + optional_size
    _require_binary_range(
        section_offset,
        section_count * 40,
        0,
        len(data),
        location,
        "PE section table",
    )
    sections: list[tuple[int, int, int, int]] = []
    for index in range(section_count):
        current = section_offset + index * 40
        virtual_size, virtual_address, raw_size, raw_pointer = _unpack_from(
            data, "<IIII", current + 8, location
        )
        if raw_size:
            _require_binary_range(
                raw_pointer,
                raw_size,
                0,
                len(data),
                location,
                f"PE section {index} raw data",
            )
        sections.append((virtual_address, virtual_size, raw_pointer, raw_size))

    cli_offset, cli_end = _rva_to_file_range(cli_rva, cli_size, sections, location)
    (cli_header_size,) = _unpack_from(data, "<I", cli_offset, location)
    if cli_header_size < 0x48 or cli_header_size > cli_size:
        _fail(location, "CLR header size is outside its data-directory range")
    _require_binary_range(
        cli_offset, cli_header_size, cli_offset, cli_end, location, "CLR header"
    )
    metadata_rva, metadata_size = _unpack_from(data, "<II", cli_offset + 8, location)
    if metadata_rva == 0 or metadata_size < 20:
        _fail(location, "CLR header has no metadata root")
    metadata_offset, metadata_end = _rva_to_file_range(
        metadata_rva, metadata_size, sections, location
    )

    _require_binary_range(
        metadata_offset,
        16,
        metadata_offset,
        metadata_end,
        location,
        "CLR metadata root header",
    )
    (metadata_signature,) = _unpack_from(data, "<I", metadata_offset, location)
    if metadata_signature != 0x424A5342:
        _fail(location, "has no CLR metadata signature")

    (version_length,) = _unpack_from(data, "<I", metadata_offset + 12, location)
    if version_length < 1:
        _fail(location, "CLR metadata has an empty version string")
    version_offset = metadata_offset + 16
    _require_binary_range(
        version_offset,
        version_length,
        metadata_offset,
        metadata_end,
        location,
        "CLR metadata version string",
    )
    if b"\x00" not in data[version_offset : version_offset + version_length]:
        _fail(location, "CLR metadata version string is not null-terminated")

    stream_header = metadata_offset + ((16 + version_length + 3) & ~3)
    _require_binary_range(
        stream_header,
        4,
        metadata_offset,
        metadata_end,
        location,
        "CLR storage header",
    )
    (_, stream_count) = _unpack_from(data, "<HH", stream_header, location)
    if stream_count < 1:
        _fail(location, "CLR metadata has no streams")
    stream_header += 4
    streams: dict[str, tuple[int, int]] = {}
    for _ in range(stream_count):
        _require_binary_range(
            stream_header,
            8,
            metadata_offset,
            metadata_end,
            location,
            "CLR stream header",
        )
        stream_relative_offset, stream_size = _unpack_from(
            data, "<II", stream_header, location
        )
        name_start = stream_header + 8
        name_end = data.find(
            b"\x00", name_start, min(name_start + 32, metadata_end)
        )
        if name_end < 0:
            _fail(location, "has an unterminated CLR stream name")
        try:
            name = data[name_start:name_end].decode("ascii")
        except UnicodeDecodeError:
            _fail(location, "has a non-ASCII CLR stream name")
        if not name:
            _fail(location, "has an empty CLR stream name")
        if name in streams:
            _fail(location, f"has duplicate CLR stream {name!r}")
        stream_offset = metadata_offset + stream_relative_offset
        _require_binary_range(
            stream_offset,
            stream_size,
            metadata_offset,
            metadata_end,
            location,
            f"CLR stream {name!r}",
        )
        streams[name] = (stream_offset, stream_size)
        stream_header += (8 + (name_end - name_start) + 1 + 3) & ~3

    stream_headers_end = stream_header
    _require_binary_range(
        stream_headers_end,
        0,
        metadata_offset,
        metadata_end,
        location,
        "CLR stream-header table",
    )
    stream_ranges = sorted(
        (offset, offset + size, name)
        for name, (offset, size) in streams.items()
        if size
    )
    previous_end = stream_headers_end
    previous_name = "metadata headers"
    for stream_start, stream_end, stream_name in stream_ranges:
        if stream_start < stream_headers_end:
            _fail(location, f"CLR stream {stream_name!r} overlaps metadata headers")
        if stream_start < previous_end:
            _fail(
                location,
                f"CLR streams {previous_name!r} and {stream_name!r} overlap",
            )
        previous_end = stream_end
        previous_name = stream_name

    if "#~" in streams and "#-" in streams:
        _fail(location, "has both compressed and uncompressed CLR table streams")
    tables = streams.get("#~") or streams.get("#-")
    guid_stream = streams.get("#GUID")
    if tables is None or guid_stream is None:
        _fail(location, "is missing CLR tables or GUID metadata")
    tables_offset, tables_size = tables
    guid_offset, guid_size = guid_stream
    if tables_size < 24:
        _fail(location, "CLR tables stream is truncated")
    if guid_size < 16 or guid_size % 16:
        _fail(location, "CLR GUID stream has an invalid size")
    tables_end = tables_offset + tables_size

    (heap_sizes,) = _unpack_from(data, "<B", tables_offset + 6, location)
    (valid_mask,) = _unpack_from(data, "<Q", tables_offset + 8, location)
    row_counts_offset = tables_offset + 24
    row_counts: dict[int, int] = {}
    for table_index in range(64):
        if valid_mask & (1 << table_index):
            _require_binary_range(
                row_counts_offset,
                4,
                tables_offset,
                tables_end,
                location,
                "CLR table row counts",
            )
            (row_count,) = _unpack_from(data, "<I", row_counts_offset, location)
            row_counts[table_index] = row_count
            row_counts_offset += 4
    if row_counts.get(0, 0) != 1:
        _fail(location, "must contain exactly one CLR Module row")

    string_index_size = 4 if heap_sizes & 0x01 else 2
    guid_index_size = 4 if heap_sizes & 0x02 else 2
    module_row_size = 2 + string_index_size + 3 * guid_index_size
    _require_binary_range(
        row_counts_offset,
        module_row_size,
        tables_offset,
        tables_end,
        location,
        "CLR Module row",
    )
    mvid_index_offset = row_counts_offset + 2 + string_index_size
    mvid_format = "<I" if guid_index_size == 4 else "<H"
    (mvid_index,) = _unpack_from(data, mvid_format, mvid_index_offset, location)
    if mvid_index < 1:
        _fail(location, "has an empty CLR Module MVID")
    mvid_offset = guid_offset + (mvid_index - 1) * 16
    _require_binary_range(
        mvid_offset,
        16,
        guid_offset,
        guid_offset + guid_size,
        location,
        "CLR Module MVID",
    )
    mvid = UUID(bytes_le=data[mvid_offset : mvid_offset + 16])
    if mvid.int == 0:
        _fail(location, "has a zero CLR Module MVID")
    return str(mvid)


def _read_dotnet_mvid(path: Path) -> str:
    return _read_dotnet_mvid_bytes(path.read_bytes(), f"exporter DLL {path}")


def _history_probe(value: Any, location: str, *, expected_history_type: str) -> Mapping[str, Any]:
    probe = _mapping(value, location)
    status = _string(_require(probe, "status", location), f"{location}.status")
    if status not in {"OK", "ERROR", "SKIPPED"}:
        _fail(f"{location}.status", "must be OK, ERROR, or SKIPPED")
    history_name, _ = _enum_snapshot(
        _require(probe, "requested_history_type", location),
        f"{location}.requested_history_type",
    )
    if history_name != expected_history_type:
        _fail(f"{location}.requested_history_type.name", f"must equal {expected_history_type!r}")
    items = _integer(_require(probe, "items", location), f"{location}.items")
    typed_items = _integer(
        _require(probe, "typed_items", location), f"{location}.typed_items"
    )
    positive_size_items = _integer(
        _require(probe, "positive_size_items", location),
        f"{location}.positive_size_items",
    )
    aggressor_buy = _integer(
        _require(probe, "aggressor_buy", location), f"{location}.aggressor_buy"
    )
    aggressor_sell = _integer(
        _require(probe, "aggressor_sell", location), f"{location}.aggressor_sell"
    )
    aggressor_unknown = _integer(
        _require(probe, "aggressor_unknown", location),
        f"{location}.aggressor_unknown",
    )
    if typed_items > items or positive_size_items > typed_items:
        _fail(location, "typed/positive counters exceed item counters")
    if aggressor_buy + aggressor_sell + aggressor_unknown > typed_items:
        _fail(location, "aggressor counters exceed typed items")
    if expected_history_type == "Last" and status == "OK" and (
        aggressor_buy + aggressor_sell + aggressor_unknown != typed_items
    ):
        _fail(location, "Last aggressor counters must classify every typed item")
    error_type = _require(probe, "error_type", location)
    if status == "ERROR":
        _string(error_type, f"{location}.error_type")
    elif error_type is not None:
        _fail(f"{location}.error_type", "must be null unless status is ERROR")
    return {
        "status": status,
        "items": items,
        "typed_items": typed_items,
        "positive_size_items": positive_size_items,
        "aggressor_buy": aggressor_buy,
        "aggressor_sell": aggressor_sell,
        "aggressor_unknown": aggressor_unknown,
    }


def _decimal(value: Any, location: str, *, nonnegative: bool = False) -> Decimal:
    if not isinstance(value, str) or not value:
        _fail(location, "must be an invariant decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        _fail(location, "must be an invariant decimal string")
    if not parsed.is_finite():
        _fail(location, "must be finite")
    if nonnegative and parsed < 0:
        _fail(location, "must be nonnegative")
    return parsed


def _utc(value: Any, location: str) -> datetime:
    text = _string(value, location)
    if UTC_MILLISECOND_TIMESTAMP.fullmatch(text) is None:
        _fail(location, "must use canonical UTC format YYYY-MM-DDTHH:MM:SS.fffZ")
    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError:
        _fail(location, "must be a valid canonical UTC timestamp")
    return parsed


def _require(value: Mapping[str, Any], key: str, location: str) -> Any:
    if key not in value:
        _fail(location, f"missing field {key!r}")
    return value[key]


def _volume_fields(value: Any, location: str) -> dict[str, Decimal | int]:
    obj = _mapping(value, location)
    fields: dict[str, Decimal | int] = {
        "volume": _decimal(_require(obj, "volume", location), f"{location}.volume", nonnegative=True),
        "trades": _integer(_require(obj, "trades", location), f"{location}.trades"),
        "buy_volume": _decimal(_require(obj, "buy_volume", location), f"{location}.buy_volume", nonnegative=True),
        "sell_volume": _decimal(_require(obj, "sell_volume", location), f"{location}.sell_volume", nonnegative=True),
        "buy_trades": _integer(_require(obj, "buy_trades", location), f"{location}.buy_trades"),
        "sell_trades": _integer(_require(obj, "sell_trades", location), f"{location}.sell_trades"),
        "delta": _decimal(_require(obj, "delta", location), f"{location}.delta"),
    }
    if fields["buy_volume"] + fields["sell_volume"] > fields["volume"] and not _close_decimal(
        fields["buy_volume"] + fields["sell_volume"], fields["volume"]
    ):
        _fail(location, "Buy/Sell volume exceeds Total volume")
    if fields["buy_trades"] + fields["sell_trades"] > fields["trades"]:
        _fail(location, "Buy/Sell trades exceeds Total trades")
    expected_delta = fields["buy_volume"] - fields["sell_volume"]
    if not _close_decimal(fields["delta"], expected_delta):
        _fail(f"{location}.delta", "must equal buy_volume - sell_volume")
    return fields


def _sum_levels(levels: Iterable[dict[str, Decimal | int]]) -> dict[str, Decimal | int]:
    sums: dict[str, Decimal | int] = {
        "volume": Decimal(0),
        "trades": 0,
        "buy_volume": Decimal(0),
        "sell_volume": Decimal(0),
        "buy_trades": 0,
        "sell_trades": 0,
        "delta": Decimal(0),
    }
    for level in levels:
        for key, value in level.items():
            sums[key] += value
    return sums


def _close_decimal(left: Decimal, right: Decimal) -> bool:
    tolerance = max(Decimal("1e-8"), max(abs(left), abs(right)) * Decimal("1e-9"))
    return abs(left - right) <= tolerance


def _volume_sums_match(left: Mapping[str, Decimal | int], right: Mapping[str, Decimal | int]) -> bool:
    decimal_keys = ("volume", "buy_volume", "sell_volume", "delta")
    integer_keys = ("trades", "buy_trades", "sell_trades")
    return all(_close_decimal(left[key], right[key]) for key in decimal_keys) and all(
        left[key] == right[key] for key in integer_keys
    )


def _validate_bar(
    raw: Any,
    index: int,
    *,
    day_start: datetime,
    day_end: datetime,
    tick_size: Decimal,
    source_symbol: str,
    strategy_symbol: str,
    captured_at_expected: datetime,
    snapshot_finished_at: datetime,
    bar_right_boundary_semantics: str,
) -> tuple[datetime, int, str]:
    location = f"bars.json.bars[{index}]"
    bar = _mapping(raw, location)
    if _integer(_require(bar, "schema_version", location), f"{location}.schema_version") != SCHEMA_VERSION:
        _fail(f"{location}.schema_version", f"must equal {SCHEMA_VERSION}")
    if _integer(_require(bar, "revision", location), f"{location}.revision") != 0:
        _fail(f"{location}.revision", "must equal 0")
    if _string(_require(bar, "source_symbol", location), f"{location}.source_symbol") != source_symbol:
        _fail(f"{location}.source_symbol", "does not match manifest instrument")
    if _string(_require(bar, "symbol", location), f"{location}.symbol") != strategy_symbol:
        _fail(f"{location}.symbol", "does not match manifest instrument")
    if _require(bar, "timeframe", location) != "15m":
        _fail(f"{location}.timeframe", "must equal '15m'")

    bar_open = _utc(_require(bar, "bar_open", location), f"{location}.bar_open")
    bar_close = _utc(_require(bar, "bar_close", location), f"{location}.bar_close")
    captured_at = _utc(_require(bar, "captured_at", location), f"{location}.captured_at")
    if not day_start <= bar_open < day_end:
        _fail(f"{location}.bar_open", "outside manifest UTC day")
    if bar_open.minute % 15 or bar_open.second or bar_open.microsecond:
        _fail(f"{location}.bar_open", "must be on an exact M15 boundary")
    if bar_close - bar_open != M15:
        _fail(f"{location}.bar_close", "must equal bar_open + 15 minutes")
    source_bar_span_ticks = _integer(
        _require(bar, "source_bar_span_ticks", location),
        f"{location}.source_bar_span_ticks",
        minimum=0,
    )
    expected_source_span_ticks = SOURCE_SPAN_TICKS_BY_SEMANTICS[
        bar_right_boundary_semantics
    ]
    if source_bar_span_ticks != expected_source_span_ticks:
        _fail(
            f"{location}.source_bar_span_ticks",
            "does not match manifest bar_right_boundary_semantics",
        )
    if captured_at < bar_close:
        _fail(f"{location}.captured_at", "must not precede bar_close")
    if captured_at != captured_at_expected or captured_at > snapshot_finished_at:
        _fail(f"{location}.captured_at", "does not match manifest snapshot observation")
    if _require(bar, "historical_available_at", location) is not None:
        _fail(f"{location}.historical_available_at", "must remain null in diagnostic v2")
    if _require(bar, "availability_semantics", location) != "unknown_historical_latency":
        _fail(f"{location}.availability_semantics", "must remain unknown_historical_latency")
    if _require(bar, "finalized", location) is not True:
        _fail(f"{location}.finalized", "must be true")
    if (
        _require(bar, "finalization_basis", location)
        != "canonical_bar_close_at_or_before_closed_export_day_end"
    ):
        _fail(f"{location}.finalization_basis", "unexpected finalization rule")

    ohlc = _mapping(_require(bar, "ohlc", location), f"{location}.ohlc")
    open_price = _decimal(_require(ohlc, "open", f"{location}.ohlc"), f"{location}.ohlc.open")
    high = _decimal(_require(ohlc, "high", f"{location}.ohlc"), f"{location}.ohlc.high")
    low = _decimal(_require(ohlc, "low", f"{location}.ohlc"), f"{location}.ohlc.low")
    close = _decimal(_require(ohlc, "close", f"{location}.ohlc"), f"{location}.ohlc.close")
    if low > high or not (low <= open_price <= high) or not (low <= close <= high):
        _fail(f"{location}.ohlc", "inconsistent OHLC range")
    _decimal(_require(bar, "bar_volume", location), f"{location}.bar_volume", nonnegative=True)
    _integer(_require(bar, "bar_ticks", location), f"{location}.bar_ticks")

    total = _volume_fields(_require(bar, "total", location), f"{location}.total")
    levels_raw = _sequence(_require(bar, "price_levels", location), f"{location}.price_levels")
    captured_levels: list[dict[str, Decimal | int]] = []
    previous_ticks: int | None = None
    for level_index, raw_level in enumerate(levels_raw):
        level_location = f"{location}.price_levels[{level_index}]"
        level = _mapping(raw_level, level_location)
        price = _decimal(_require(level, "price", level_location), f"{level_location}.price")
        price_ticks = _integer(
            _require(level, "price_ticks", level_location),
            f"{level_location}.price_ticks",
            minimum=-(2**63),
        )
        if price_ticks > 2**63 - 1:
            _fail(f"{level_location}.price_ticks", "outside Int64 range")
        if previous_ticks is not None and price_ticks <= previous_ticks:
            _fail(f"{level_location}.price_ticks", "must be strictly increasing and unique")
        previous_ticks = price_ticks
        tolerance = max(Decimal("1e-12"), abs(tick_size) * Decimal("1e-6"))
        if abs(price - tick_size * price_ticks) > tolerance:
            _fail(f"{level_location}.price", "does not match price_ticks * tick_size")
        if price < low - tick_size or price > high + tick_size:
            _fail(f"{level_location}.price", "outside OHLC range by more than one tick")
        captured_levels.append(_volume_fields(level, level_location))

    validation = _mapping(_require(bar, "validation", location), f"{location}.validation")
    level_status = _string(
        _require(validation, "price_levels_status", f"{location}.validation"),
        f"{location}.validation.price_levels_status",
    )
    if not levels_raw or level_status != "AVAILABLE":
        _fail(f"{location}.price_levels", "complete diagnostic requires AVAILABLE PriceLevels")
    if _integer(
        _require(validation, "level_count", f"{location}.validation"),
        f"{location}.validation.level_count",
    ) != len(levels_raw):
        _fail(f"{location}.validation.level_count", "does not match price_levels")
    computed_sums = _sum_levels(captured_levels)
    computed_match = _volume_sums_match(computed_sums, total)
    if _require(validation, "sum_levels_matches_total", f"{location}.validation") is not computed_match:
        _fail(f"{location}.validation.sum_levels_matches_total", "does not match computed sums")
    if not computed_match:
        _fail(f"{location}.price_levels", "level sums do not match bar Total")
    declared_sums = _volume_fields(
        _require(validation, "level_sums", f"{location}.validation"),
        f"{location}.validation.level_sums",
    )
    if not _volume_sums_match(declared_sums, computed_sums):
        _fail(f"{location}.validation.level_sums", "does not match computed levels")

    unclassified_volume = _decimal(
        _require(validation, "unclassified_volume", f"{location}.validation"),
        f"{location}.validation.unclassified_volume",
        nonnegative=True,
    )
    expected_unclassified_volume = total["volume"] - total["buy_volume"] - total["sell_volume"]
    if not _close_decimal(unclassified_volume, expected_unclassified_volume):
        _fail(f"{location}.validation.unclassified_volume", "does not match Total polarity remainder")
    unclassified_trades = _require(validation, "unclassified_trades", f"{location}.validation")
    if isinstance(unclassified_trades, bool) or not isinstance(unclassified_trades, int):
        _fail(f"{location}.validation.unclassified_trades", "must be an integer")
    if unclassified_trades < 0:
        _fail(f"{location}.validation.unclassified_trades", "must be nonnegative")
    expected_unclassified_trades = total["trades"] - total["buy_trades"] - total["sell_trades"]
    if unclassified_trades != expected_unclassified_trades:
        _fail(f"{location}.validation.unclassified_trades", "does not match Total polarity remainder")

    return bar_open, len(levels_raw), level_status


def validate_export(
    root: str | Path,
    *,
    exporter_dll: str | Path | None = None,
) -> ValidationSummary:
    base = Path(root).resolve(strict=True)
    if not base.is_dir():
        _fail(str(base), "must be an export directory")

    manifest_path = base / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        _fail("manifest.json", "is required and must be written last")
    manifest = _load_json(manifest_path.read_bytes(), "manifest.json")
    if _require(manifest, "schema", "manifest.json") != SCHEMA:
        _fail("manifest.json.schema", f"must equal {SCHEMA!r}; production sidecars are rejected")
    if _integer(
        _require(manifest, "schema_version", "manifest.json"),
        "manifest.json.schema_version",
    ) != SCHEMA_VERSION:
        _fail("manifest.json.schema_version", f"must equal {SCHEMA_VERSION}")
    export_id = _string(_require(manifest, "export_id", "manifest.json"), "manifest.json.export_id")
    try:
        if str(UUID(export_id)) != export_id:
            _fail("manifest.json.export_id", "must be a canonical lowercase UUID")
    except ValueError:
        _fail("manifest.json.export_id", "must be a canonical lowercase UUID")
    created_at = _utc(
        _require(manifest, "created_at", "manifest.json"), "manifest.json.created_at"
    )
    if _require(manifest, "timeframe", "manifest.json") != "15m":
        _fail("manifest.json.timeframe", "must equal '15m'")

    source = _mapping(_require(manifest, "source", "manifest.json"), "manifest.json.source")
    if _require(source, "platform", "manifest.json.source") != "Quantower":
        _fail("manifest.json.source.platform", "must equal 'Quantower'")
    platform_version = _string(
        _require(source, "platform_version", "manifest.json.source"),
        "manifest.json.source.platform_version",
    )
    if platform_version == "UNKNOWN":
        _fail("manifest.json.source.platform_version", "must be known")
    if _require(source, "connector_vendor", "manifest.json.source") != "FxPro":
        _fail("manifest.json.source.connector_vendor", "must equal 'FxPro'")
    if _require(source, "connector_vendor_semantics", "manifest.json.source") != "operator_label_not_connection_identity":
        _fail("manifest.json.source.connector_vendor_semantics", "must disclose operator-label provenance")
    _string(
        _require(source, "exporter_version", "manifest.json.source"),
        "manifest.json.source.exporter_version",
    )
    exporter_binary_sha256_raw = _require(
        source, "exporter_binary_sha256", "manifest.json.source"
    )
    exporter_binary_sha256_status = _string(
        _require(source, "exporter_binary_sha256_status", "manifest.json.source"),
        "manifest.json.source.exporter_binary_sha256_status",
    )
    exporter_module_version_id = _string(
        _require(source, "exporter_module_version_id", "manifest.json.source"),
        "manifest.json.source.exporter_module_version_id",
    )
    try:
        if str(UUID(exporter_module_version_id)) != exporter_module_version_id:
            _fail(
                "manifest.json.source.exporter_module_version_id",
                "must be a canonical lowercase UUID",
            )
    except ValueError:
        _fail(
            "manifest.json.source.exporter_module_version_id",
            "must be a canonical lowercase UUID",
        )

    manifest_exporter_sha256: str | None
    if exporter_binary_sha256_status == "AVAILABLE":
        manifest_exporter_sha256 = _sha256(
            exporter_binary_sha256_raw,
            "manifest.json.source.exporter_binary_sha256",
        )
    elif exporter_binary_sha256_status == "UNAVAILABLE_FROM_RUNTIME_ASSEMBLY_PATH":
        if exporter_binary_sha256_raw is not None:
            _fail(
                "manifest.json.source.exporter_binary_sha256",
                "must be null when runtime assembly path is unavailable",
            )
        manifest_exporter_sha256 = None
    else:
        _fail(
            "manifest.json.source.exporter_binary_sha256_status",
            "unsupported exporter binary SHA-256 status",
        )

    validated_exporter_sha256 = manifest_exporter_sha256
    if exporter_dll is not None:
        exporter_dll_path = Path(exporter_dll).expanduser().resolve(strict=True)
        if not exporter_dll_path.is_file():
            _fail(str(exporter_dll_path), "exporter DLL must be a file")
        exporter_dll_bytes = exporter_dll_path.read_bytes()
        validated_exporter_sha256 = hashlib.sha256(exporter_dll_bytes).hexdigest()
        if (
            _read_dotnet_mvid_bytes(
                exporter_dll_bytes,
                f"exporter DLL {exporter_dll_path}",
            )
            != exporter_module_version_id
        ):
            _fail(
                str(exporter_dll_path),
                "CLR Module MVID does not match manifest exporter_module_version_id",
            )
        if (
            manifest_exporter_sha256 is not None
            and validated_exporter_sha256 != manifest_exporter_sha256
        ):
            _fail(
                str(exporter_dll_path),
                "SHA-256 does not match manifest exporter_binary_sha256",
            )
    elif manifest_exporter_sha256 is None:
        _fail(
            "manifest.json.source.exporter_binary_sha256",
            "runtime path unavailable; validate with --exporter-dll",
        )

    instrument = _mapping(_require(manifest, "instrument", "manifest.json"), "manifest.json.instrument")
    source_symbol = _string(
        _require(instrument, "source_symbol", "manifest.json.instrument"),
        "manifest.json.instrument.source_symbol",
    )
    strategy_symbol = _string(
        _require(instrument, "strategy_symbol", "manifest.json.instrument"),
        "manifest.json.instrument.strategy_symbol",
    )
    tick_size = _decimal(
        _require(instrument, "tick_size", "manifest.json.instrument"),
        "manifest.json.instrument.tick_size",
    )
    if tick_size <= 0:
        _fail("manifest.json.instrument.tick_size", "must be positive")
    min_volume_analysis_tick_size = _require(
        instrument,
        "min_volume_analysis_tick_size",
        "manifest.json.instrument",
    )
    if min_volume_analysis_tick_size is not None:
        _decimal(
            min_volume_analysis_tick_size,
            "manifest.json.instrument.min_volume_analysis_tick_size",
            nonnegative=True,
        )
    history_type, _ = _enum_snapshot(
        _require(instrument, "symbol_history_type", "manifest.json.instrument"),
        "manifest.json.instrument.symbol_history_type",
    )
    if history_type not in {"Ask", "Bid", "BidAsk", "Last", "Mark", "Midpoint"}:
        _fail("manifest.json.instrument.symbol_history_type.name", "unknown HistoryType")
    volume_type, _ = _enum_snapshot(
        _require(instrument, "symbol_volume_type", "manifest.json.instrument"),
        "manifest.json.instrument.symbol_volume_type",
    )
    if volume_type != "Ticks":
        _fail("manifest.json.instrument.symbol_volume_type.name", "FxPro diagnostic requires Ticks")
    delta_type, _ = _enum_snapshot(
        _require(instrument, "symbol_delta_calculation_type", "manifest.json.instrument"),
        "manifest.json.instrument.symbol_delta_calculation_type",
    )
    if delta_type != "TickDirection":
        _fail("manifest.json.instrument.symbol_delta_calculation_type.name", "FxPro diagnostic requires TickDirection")
    if _boolean(
        _require(instrument, "allow_calculate_realtime_ticks", "manifest.json.instrument"),
        "manifest.json.instrument.allow_calculate_realtime_ticks",
    ) is not True:
        _fail("manifest.json.instrument.allow_calculate_realtime_ticks", "must be true")
    for disabled_capability in (
        "allow_calculate_realtime_volume",
        "allow_calculate_realtime_trades",
    ):
        if _boolean(
            _require(instrument, disabled_capability, "manifest.json.instrument"),
            f"manifest.json.instrument.{disabled_capability}",
        ) is not False:
            _fail(f"manifest.json.instrument.{disabled_capability}", "must be false")

    calculation = _mapping(
        _require(manifest, "calculation", "manifest.json"), "manifest.json.calculation"
    )
    if _require(calculation, "period", "manifest.json.calculation") != "15m":
        _fail("manifest.json.calculation.period", "must equal '15m'")
    _string(
        _require(calculation, "chart_aggregation", "manifest.json.calculation"),
        "manifest.json.calculation.chart_aggregation",
    )
    bar_right_boundary_semantics = _string(
        _require(
            calculation,
            "bar_right_boundary_semantics",
            "manifest.json.calculation",
        ),
        "manifest.json.calculation.bar_right_boundary_semantics",
    )
    if bar_right_boundary_semantics not in SOURCE_SPAN_TICKS_BY_SEMANTICS:
        _fail(
            "manifest.json.calculation.bar_right_boundary_semantics",
            "unsupported M15 right-boundary normalization",
        )
    if _boolean(
        _require(calculation, "price_levels_requested", "manifest.json.calculation"),
        "manifest.json.calculation.price_levels_requested",
    ) is not True:
        _fail("manifest.json.calculation.price_levels_requested", "must be true")
    if _require(calculation, "volume_basis", "manifest.json.calculation") != "native_quantower_volume_analysis":
        _fail("manifest.json.calculation.volume_basis", "unexpected volume basis")
    if _require(calculation, "timezone", "manifest.json.calculation") != "UTC":
        _fail("manifest.json.calculation.timezone", "must equal UTC")
    if _boolean(
        _require(calculation, "unspecified_times_assumed_utc", "manifest.json.calculation"),
        "manifest.json.calculation.unspecified_times_assumed_utc",
    ) is not False:
        _fail(
            "manifest.json.calculation.unspecified_times_assumed_utc",
            "VALID_DIAGNOSTIC requires explicit UTC timestamps",
        )

    semantics = _mapping(_require(manifest, "semantics", "manifest.json"), "manifest.json.semantics")
    if _require(semantics, "factor_name", "manifest.json.semantics") != "FxPro Tick Absorption 15M":
        _fail("manifest.json.semantics.factor_name", "unexpected factor")
    classification = _string(
        _require(semantics, "classification", "manifest.json.semantics"),
        "manifest.json.semantics.classification",
    )
    if classification not in {
        "quantower_tick_reconstructed_bidask_history_available",
        "trade_capable_source_unverified_calculation",
        "unknown",
    }:
        _fail("manifest.json.semantics.classification", "unknown classification")
    if _require(semantics, "classification_status", "manifest.json.semantics") != "UNVERIFIED":
        _fail("manifest.json.semantics.classification_status", "diagnostic v2 must remain UNVERIFIED")
    for description in ("buy_volume", "sell_volume", "trades"):
        _string(
            _require(semantics, description, "manifest.json.semantics"),
            f"manifest.json.semantics.{description}",
        )
    if _require(semantics, "exchange_executions_proven", "manifest.json.semantics") is not False:
        _fail("manifest.json.semantics.exchange_executions_proven", "must be false")
    if _require(semantics, "aggressor_polarity_proven", "manifest.json.semantics") is not False:
        _fail("manifest.json.semantics.aggressor_polarity_proven", "must be false")
    evidence = _mapping(
        _require(semantics, "evidence", "manifest.json.semantics"),
        "manifest.json.semantics.evidence",
    )
    last_probe = _history_probe(
        _require(evidence, "last_probe", "manifest.json.semantics.evidence"),
        "manifest.json.semantics.evidence.last_probe",
        expected_history_type="Last",
    )
    bid_ask_probe = _history_probe(
        _require(evidence, "bid_ask_probe", "manifest.json.semantics.evidence"),
        "manifest.json.semantics.evidence.bid_ask_probe",
        expected_history_type="BidAsk",
    )
    if last_probe["status"] != "OK" or bid_ask_probe["status"] != "OK":
        _fail("manifest.json.semantics.evidence", "complete diagnostic requires successful Last and BidAsk probes")
    if classification == "unknown":
        _fail("manifest.json.semantics.classification", "complete diagnostic requires a probe-supported classification")
    if classification == "quantower_tick_reconstructed_bidask_history_available" and not (
        last_probe["status"] == "OK"
        and last_probe["items"] == 0
        and bid_ask_probe["status"] == "OK"
        and bid_ask_probe["items"] > 0
        and bid_ask_probe["typed_items"] > 0
    ):
        _fail("manifest.json.semantics.classification", "does not match Last/BidAsk evidence")
    if classification == "trade_capable_source_unverified_calculation" and not (
        last_probe["status"] == "OK"
        and last_probe["positive_size_items"] > 0
        and last_probe["aggressor_buy"] + last_probe["aggressor_sell"] > 0
    ):
        _fail("manifest.json.semantics.classification", "does not match Last evidence")

    causality = _mapping(_require(manifest, "causality", "manifest.json"), "manifest.json.causality")
    history_started_at = _utc(
        _require(causality, "history_request_started_at", "manifest.json.causality"),
        "manifest.json.causality.history_request_started_at",
    )
    if _require(causality, "history_request_start_semantics", "manifest.json.causality") != "indicator_on_init_lower_bound":
        _fail("manifest.json.causality.history_request_start_semantics", "unexpected request-time claim")
    volume_finished_at = _utc(
        _require(causality, "volume_analysis_finished_at", "manifest.json.causality"),
        "manifest.json.causality.volume_analysis_finished_at",
    )
    snapshot_finished_at = _utc(
        _require(causality, "snapshot_finished_at", "manifest.json.causality"),
        "manifest.json.causality.snapshot_finished_at",
    )
    if not history_started_at <= volume_finished_at <= snapshot_finished_at:
        _fail("manifest.json.causality", "timestamps are not monotonic")
    if created_at != volume_finished_at:
        _fail("manifest.json.created_at", "must equal the captured Volume Analysis observation")
    if _require(causality, "historical_availability_semantics", "manifest.json.causality") != "snapshot_observed_at_not_historical_feed_latency":
        _fail("manifest.json.causality.historical_availability_semantics", "unsafe causality claim")
    coverage = _mapping(_require(manifest, "coverage", "manifest.json"), "manifest.json.coverage")
    if _require(coverage, "eligible_for_sidecar_conversion", "manifest.json.coverage") is not False:
        _fail("manifest.json.coverage.eligible_for_sidecar_conversion", "must remain false")

    day_text = _string(_require(manifest, "day_utc", "manifest.json"), "manifest.json.day_utc")
    try:
        day_start = datetime.strptime(day_text, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        _fail("manifest.json.day_utc", "must use YYYY-MM-DD")
    day_end = day_start + timedelta(days=1)
    if day_end > created_at:
        _fail("manifest.json.day_utc", "export day must be closed before capture")
    interval = _mapping(_require(manifest, "interval", "manifest.json"), "manifest.json.interval")
    if _utc(_require(interval, "from_inclusive", "manifest.json.interval"), "manifest.json.interval.from_inclusive") != day_start:
        _fail("manifest.json.interval.from_inclusive", "must equal day_utc midnight")
    if _utc(_require(interval, "to_exclusive", "manifest.json.interval"), "manifest.json.interval.to_exclusive") != day_end:
        _fail("manifest.json.interval.to_exclusive", "must equal next UTC midnight")

    files = _sequence(_require(manifest, "files", "manifest.json"), "manifest.json.files")
    if len(files) != 1:
        _fail("manifest.json.files", "diagnostic v2 must contain only bars.json")
    matching = [item for item in files if isinstance(item, dict) and item.get("path") == "bars.json"]
    if len(matching) != 1:
        _fail("manifest.json.files", "must contain exactly one bars.json entry")
    file_meta = _mapping(matching[0], "manifest.json.files[bars.json]")
    unresolved_bars_path = base / "bars.json"
    if unresolved_bars_path.is_symlink():
        _fail("bars.json", "symlinks are not allowed")
    bars_path = unresolved_bars_path.resolve(strict=True)
    if bars_path.parent != base or not bars_path.is_file():
        _fail("bars.json", "must be a regular file directly under export root")
    bars_raw = bars_path.read_bytes()
    digest = hashlib.sha256(bars_raw).hexdigest()
    if _integer(
        _require(file_meta, "size", "manifest.json.files[bars.json]"),
        "manifest.json.files[bars.json].size",
    ) != len(bars_raw):
        _fail("manifest.json.files[bars.json].size", "does not match file")
    if _sha256(
        _require(file_meta, "sha256", "manifest.json.files[bars.json]"),
        "manifest.json.files[bars.json].sha256",
    ) != digest:
        _fail("manifest.json.files[bars.json].sha256", "does not match file")
    if _sha256(
        _require(manifest, "content_sha256", "manifest.json"),
        "manifest.json.content_sha256",
    ) != digest:
        _fail("manifest.json.content_sha256", "does not match raw UTF-8 bars.json")

    bars_document = _load_json(bars_raw, "bars.json")
    if _require(bars_document, "schema", "bars.json") != BARS_SCHEMA:
        _fail("bars.json.schema", f"must equal {BARS_SCHEMA!r}")
    if _integer(
        _require(bars_document, "schema_version", "bars.json"),
        "bars.json.schema_version",
    ) != SCHEMA_VERSION:
        _fail("bars.json.schema_version", f"must equal {SCHEMA_VERSION}")
    if _require(bars_document, "export_id", "bars.json") != export_id:
        _fail("bars.json.export_id", "does not match manifest")
    bars = _sequence(_require(bars_document, "bars", "bars.json"), "bars.json.bars")
    if not bars:
        _fail("bars.json.bars", "must not be empty")
    if _integer(
        _require(file_meta, "records", "manifest.json.files[bars.json]"),
        "manifest.json.files[bars.json].records",
    ) != len(bars):
        _fail("manifest.json.files[bars.json].records", "does not match bars array")

    opens: list[datetime] = []
    level_count = 0
    bars_with_levels = 0
    for index, raw_bar in enumerate(bars):
        bar_open, levels, level_status = _validate_bar(
            raw_bar,
            index,
            day_start=day_start,
            day_end=day_end,
            tick_size=tick_size,
            source_symbol=source_symbol,
            strategy_symbol=strategy_symbol,
            captured_at_expected=created_at,
            snapshot_finished_at=snapshot_finished_at,
            bar_right_boundary_semantics=bar_right_boundary_semantics,
        )
        opens.append(bar_open)
        level_count += levels
        bars_with_levels += int(level_status == "AVAILABLE")
    if opens != sorted(opens) or len(opens) != len(set(opens)):
        _fail("bars.json.bars", "bars must be strictly chronological and unique")

    expected_opens = {day_start + slot * M15 for slot in range(96)}
    actual_opens = set(opens)
    expected_missing = sorted(expected_opens - actual_opens)
    manifest_missing = [
        _utc(value, f"manifest.json.coverage.missing_bar_opens[{index}]")
        for index, value in enumerate(
            _sequence(_require(coverage, "missing_bar_opens", "manifest.json.coverage"), "manifest.json.coverage.missing_bar_opens")
        )
    ]
    if manifest_missing != expected_missing:
        _fail("manifest.json.coverage.missing_bar_opens", "does not match actual M15 coverage")
    if _integer(
        _require(coverage, "expected_utc_slots", "manifest.json.coverage"),
        "manifest.json.coverage.expected_utc_slots",
    ) != 96:
        _fail("manifest.json.coverage.expected_utc_slots", "must equal 96")
    if _integer(
        _require(coverage, "bars_returned", "manifest.json.coverage"),
        "manifest.json.coverage.bars_returned",
    ) != len(bars):
        _fail("manifest.json.coverage.bars_returned", "does not match bars array")
    if _integer(
        _require(coverage, "bars_with_price_levels", "manifest.json.coverage"),
        "manifest.json.coverage.bars_with_price_levels",
    ) != bars_with_levels:
        _fail("manifest.json.coverage.bars_with_price_levels", "does not match bars array")
    null_level_bars = _integer(
        _require(coverage, "null_price_level_bars", "manifest.json.coverage"),
        "manifest.json.coverage.null_price_level_bars",
    )
    empty_level_bars = _integer(
        _require(coverage, "empty_price_level_bars", "manifest.json.coverage"),
        "manifest.json.coverage.empty_price_level_bars",
    )
    if null_level_bars != 0 or empty_level_bars != 0 or bars_with_levels != len(bars):
        _fail("manifest.json.coverage", "complete diagnostic requires PriceLevels for every bar")
    expected_validation_status = "PASS" if not expected_missing else "WARN"
    if _require(coverage, "validation_status", "manifest.json.coverage") != expected_validation_status:
        _fail("manifest.json.coverage.validation_status", "does not match validated coverage")

    return ValidationSummary(
        export_id=export_id,
        day_utc=day_text,
        bars=len(bars),
        price_levels=level_count,
        missing_slots=len(expected_missing),
        content_sha256=digest,
        exporter_binary_sha256=validated_exporter_sha256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path, help="Quantower diagnostic export directory")
    parser.add_argument(
        "--exporter-dll",
        type=Path,
        help="installed exporter DLL used to verify SHA-256 and CLR Module MVID",
    )
    args = parser.parse_args(argv)
    try:
        summary = validate_export(args.export, exporter_dll=args.exporter_dll)
    except (DiagnosticValidationError, FileNotFoundError, OSError) as exc:
        print(f"INVALID: {exc}")
        return 2
    print(json.dumps({"status": "VALID_DIAGNOSTIC", **summary.__dict__}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
