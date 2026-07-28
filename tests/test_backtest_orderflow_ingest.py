from __future__ import annotations

import json
from decimal import Decimal

import pandas as pd
import pytest

import backtest.orderflow_ingest as orderflow_ingest
from backtest.orderflow_data import AbsorptionEventDataset
from backtest.orderflow_ingest import (
    TapeColumns,
    TapeIngestError,
    TapeTrade,
    aggregate_trades,
    build_parser,
    normalize_csv_tape,
    require_aggressor_provenance,
    require_identity_symbol,
    resolve_available_at_rule,
    run,
)
from core.absorption import detect_footprint_absorption


BAR = "2024-03-04T10:"

_HEADER = "timestamp,price,size,aggressor\n"

# One closed M15 bar with aggressive selling absorbed at the low edge.
_ROWS = (
    f"{BAR}00:05Z,1.08000,60,SELL\n"
    f"{BAR}01:00Z,1.08000,40,SELL\n"
    f"{BAR}02:00Z,1.08000,20,BUY\n"
    f"{BAR}03:00Z,1.08025,15,BUY\n"
    f"{BAR}04:00Z,1.08050,10,BUY\n"
    f"{BAR}05:00Z,1.08050,5,SELL\n"
)


def _tape(tmp_path, body=_ROWS, header=_HEADER, name="tape.csv"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    path.write_text(header + body, encoding="utf-8")
    return path


def _args(tmp_path, tape, **overrides):
    argv = [
        "--tape",
        str(tape),
        "--out",
        str(tmp_path / "sidecar"),
        "--symbol",
        "EURUSD",
        "--venue",
        "LMAX",
        "--source-id",
        "lmax-ecn-tape",
        "--tick-size",
        "0.00001",
        "--volume-measure",
        "executed_size",
        "--aggressor-provenance",
        "venue_reported_aggressor",
        "--buy-label",
        "BUY",
        "--sell-label",
        "SELL",
        "--timestamp-format",
        "iso8601",
        "--covers-from",
        "2024-03-04T00:00:00Z",
        "--covers-through",
        "2024-03-05T00:00:00Z",
        "--available-at-rule",
        "bar_close_plus_measured_vendor_delay.v1",
        "--vendor-delay-ms",
        "2500",
        "--delay-measurement",
        "p99 arrival minus bar_close over 2024-02, n=41231",
    ]
    for flag, value in overrides.items():
        option = "--" + flag.replace("_", "-")
        if value is None:
            # Drop the option and its value entirely.
            index = argv.index(option)
            del argv[index : index + 2]
            continue
        if value is True:
            argv.append(option)
            continue
        if option in argv:
            argv[argv.index(option) + 1] = str(value)
        else:
            argv.extend([option, str(value)])
    return build_parser().parse_args(argv)


def _receipt(tmp_path, **overrides):
    tape = overrides.pop("tape", None) or _tape(tmp_path)
    return run(_args(tmp_path, tape, **overrides))


def test_tape_seals_into_a_loadable_causal_sidecar(tmp_path):
    receipt = _receipt(tmp_path)

    dataset = AbsorptionEventDataset.load(tmp_path / "sidecar")
    assert dataset.symbols == ("EURUSD",)
    assert dataset.manifest_sha256 == receipt["manifest_sha256"]
    assert dataset.content_sha256 == receipt["content_sha256"]

    (event,) = dataset.events
    assert event.symbol == "EURUSD"
    assert event.source_symbol == "EURUSD"
    assert event.finalized is True
    assert event.revision == 0
    assert event.bar_open == pd.Timestamp("2024-03-04T10:00:00Z")
    assert event.bar_close == pd.Timestamp("2024-03-04T10:15:00Z")
    # bar_close + measured vendor delay, not bar_close.
    assert event.available_at == pd.Timestamp("2024-03-04T10:15:02.500Z")
    # Only the extreme traded levels contribute.
    assert event.low_sell_volume == 100.0
    assert event.low_buy_volume == 20.0
    assert event.high_buy_volume == 10.0
    assert event.high_sell_volume == 5.0

    assert dataset.event_asof("EURUSD", event.bar_open, "2024-03-04T10:15:02Z") is None
    assert (
        dataset.event_asof("EURUSD", event.bar_open, "2024-03-04T10:15:03Z") is event
    )


def test_sealed_event_drives_the_production_absorption_detector(tmp_path):
    _receipt(tmp_path)
    dataset = AbsorptionEventDataset.load(tmp_path / "sidecar")
    (event,) = dataset.events
    candle = {"high": 1.08050, "low": 1.08000, "close": 1.08045}

    signal = detect_footprint_absorption(
        candle=candle,
        candle_open_time=event.bar_open,
        event=event,
        symbol="EURUSD",
        side="LONG",
        decision_time="2024-03-04T10:15:03Z",
    )
    assert signal is not None
    assert signal.side == "LONG"
    assert signal.aggressive_edge_volume == 100.0
    assert signal.imbalance_ratio == pytest.approx(5.0)

    # The same event never fires the opposite side.
    assert (
        detect_footprint_absorption(
            candle=candle,
            candle_open_time=event.bar_open,
            event=event,
            symbol="EURUSD",
            side="SHORT",
            decision_time="2024-03-04T10:15:03Z",
        )
        is None
    )


def test_receipt_records_provenance_and_the_available_at_rule(tmp_path):
    receipt = _receipt(tmp_path)

    assert receipt["available_at_rule"]["rule_id"] == (
        "bar_close_plus_measured_vendor_delay.v1"
    )
    assert receipt["available_at_rule"]["offset_ms"] == 2500
    assert receipt["available_at_rule"]["evidence"].startswith("p99 arrival")
    assert receipt["aggressor_provenance"] == "venue_reported_aggressor"
    assert receipt["symbol_mapping"] == {"EURUSD": "EURUSD"}
    assert receipt["counts"]["events_sealed"] == 1
    assert receipt["inputs"][0]["sha256"]

    manifest = json.loads(
        (tmp_path / "sidecar" / "manifest.json").read_text(encoding="utf-8")
    )
    source = manifest["source"]
    assert source.startswith("lmax-ecn-tape;")
    assert "converter=forexbot.absorption-tape-converter/1" in source
    assert "volume=executed_size" in source
    assert "aggressor=venue_reported_aggressor" in source
    assert "available_at=bar_close_plus_measured_vendor_delay.v1(delay_ms=2500" in source

    # The receipt is deliberately outside the sealed root.
    assert (tmp_path / "sidecar.build_receipt.json").is_file()


def test_conversion_is_deterministic(tmp_path):
    first = _receipt(tmp_path / "a")
    second = _receipt(tmp_path / "b")
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["content_sha256"] == second["content_sha256"]


def test_reconstructed_aggressor_provenance_is_refused():
    with pytest.raises(TapeIngestError, match="local reconstruction"):
        require_aggressor_provenance("tick_rule_reconstructed")
    with pytest.raises(TapeIngestError, match="local reconstruction"):
        require_aggressor_provenance("quote_inferred")
    with pytest.raises(TapeIngestError, match="unknown aggressor provenance"):
        require_aggressor_provenance("fxpro_cluster")


def test_proxy_symbol_mapping_is_refused():
    assert require_identity_symbol("eurusd", "EURUSD") == "EURUSD"
    with pytest.raises(TapeIngestError, match="is not identity"):
        require_identity_symbol("EURUSD", "6E")
    with pytest.raises(TapeIngestError, match="versioned proxy schema"):
        require_identity_symbol("6E", "6E")
    with pytest.raises(TapeIngestError, match="six-letter spot FX symbol"):
        require_identity_symbol("ЕВРУСД", "ЕВРУСД")


def test_available_at_rule_has_no_default_and_requires_evidence():
    with pytest.raises(TapeIngestError, match="unknown available_at rule"):
        resolve_available_at_rule("bar_close")
    with pytest.raises(TapeIngestError, match="non-empty delay_measurement"):
        resolve_available_at_rule(
            "bar_close_plus_measured_vendor_delay.v1",
            vendor_delay_ms=1000,
            delay_measurement="   ",
        )
    with pytest.raises(TapeIngestError, match="vendor_delay_ms"):
        resolve_available_at_rule(
            "bar_close_plus_measured_vendor_delay.v1",
            delay_measurement="measured",
        )
    identical = [
        resolve_available_at_rule(
            "bar_close_plus_measured_vendor_delay.v1",
            vendor_delay_ms=1000,
            delay_measurement="measured",
        ).descriptor
        for _ in range(2)
    ]
    assert identical[0] == identical[1]


def test_arrival_rule_uses_observed_arrival_and_never_precedes_bar_close(tmp_path):
    header = "timestamp,price,size,aggressor,arrival\n"
    body = (
        f"{BAR}00:05Z,1.08000,60,SELL,{BAR}00:06Z\n"
        f"{BAR}01:00Z,1.08000,40,SELL,{BAR}01:01Z\n"
        f"{BAR}02:00Z,1.08000,20,BUY,{BAR}02:01Z\n"
        f"{BAR}14:00Z,1.08050,10,BUY,{BAR}14:04Z\n"
    )
    tape = _tape(tmp_path, body=body, header=header)
    receipt = _receipt(
        tmp_path,
        tape=tape,
        column_arrival="arrival",
        available_at_rule="max_observed_arrival_plus_margin.v1",
        arrival_margin_ms=500,
        vendor_delay_ms=None,
        delay_measurement=None,
    )
    assert receipt["counts"]["events_sealed"] == 1
    (event,) = AbsorptionEventDataset.load(tmp_path / "sidecar").events
    # max(arrival)=10:14:04.000 +500ms is still inside the bar, so bar_close wins.
    assert event.available_at == event.bar_close


def test_arrival_rule_requires_the_arrival_column(tmp_path):
    tape = _tape(tmp_path)
    with pytest.raises(TapeIngestError, match="requires --column-arrival"):
        _receipt(
            tmp_path,
            tape=tape,
            available_at_rule="max_observed_arrival_plus_margin.v1",
            arrival_margin_ms=0,
            vendor_delay_ms=None,
            delay_measurement=None,
        )


def test_unmapped_aggressor_label_aborts_the_build(tmp_path):
    tape = _tape(tmp_path, body=_ROWS + f"{BAR}06:00Z,1.08010,5,UNKNOWN\n")
    with pytest.raises(TapeIngestError, match="unmapped aggressor label"):
        _receipt(tmp_path, tape=tape)
    assert not (tmp_path / "sidecar").exists()


def test_unsorted_tape_aborts_the_build(tmp_path):
    body = f"{BAR}05:00Z,1.08000,60,SELL\n" f"{BAR}01:00Z,1.08000,40,SELL\n"
    tape = _tape(tmp_path, body=body)
    with pytest.raises(TapeIngestError, match="non-decreasing execution timestamp"):
        _receipt(tmp_path, tape=tape)


def test_price_off_the_declared_tick_grid_aborts_the_build(tmp_path):
    tape = _tape(tmp_path, body=f"{BAR}00:05Z,1.080005,60,SELL\n")
    with pytest.raises(TapeIngestError, match="not an exact multiple of tick size"):
        _receipt(tmp_path, tape=tape)


def test_non_positive_size_aborts_the_build(tmp_path):
    tape = _tape(tmp_path, body=f"{BAR}00:05Z,1.08000,0,SELL\n")
    with pytest.raises(TapeIngestError, match="size must be strictly positive"):
        _receipt(tmp_path, tape=tape)


def test_empty_declared_aggressor_label_is_refused(tmp_path):
    tape = _tape(tmp_path)
    with pytest.raises(TapeIngestError, match="labels must be non-empty"):
        _receipt(tmp_path, tape=tape, buy_label="")


def test_low_level_aggregation_refuses_unknown_aggressor():
    rule = resolve_available_at_rule(
        "bar_close_plus_measured_vendor_delay.v1",
        vendor_delay_ms=1,
        delay_measurement="measured",
    )
    trade = TapeTrade(
        timestamp=pd.Timestamp("2024-03-04T10:00:05Z"),
        level=108000,
        volume=Decimal("1"),
        aggressor="UNKNOWN",
        arrival=None,
    )
    with pytest.raises(TapeIngestError, match="must be BUY or SELL"):
        aggregate_trades(
            [trade],
            symbol="EURUSD",
            rule=rule,
            covers_from=pd.Timestamp("2024-03-04T10:00:00Z"),
            covers_through=pd.Timestamp("2024-03-04T10:15:00Z"),
        )


def test_normalizer_refuses_unknown_volume_measure(tmp_path):
    tape = _tape(tmp_path)
    with pytest.raises(TapeIngestError, match="unsupported volume_measure"):
        list(
            normalize_csv_tape(
                tape,
                columns=TapeColumns(),
                tick_size=Decimal("0.00001"),
                volume_measure="lots_or_ticks",
                buy_labels=["BUY"],
                sell_labels=["SELL"],
                timestamp_format="iso8601",
                naive_is_utc=False,
            )
        )


def test_non_finite_max_gap_is_refused_before_publication(tmp_path):
    tape = _tape(tmp_path)
    with pytest.raises(TapeIngestError, match="finite and non-negative"):
        _receipt(tmp_path, tape=tape, max_intrabar_gap_seconds="nan")
    assert not (tmp_path / "sidecar").exists()


def test_naive_timestamps_require_an_explicit_opt_in(tmp_path):
    tape = _tape(tmp_path, body="2024-03-04T10:00:05,1.08000,60,SELL\n")
    with pytest.raises(TapeIngestError, match="has no timezone"):
        _receipt(tmp_path, tape=tape)
    receipt = _receipt(
        tmp_path / "opted-in",
        tape=tape,
        naive_timestamps_are_utc=True,
    )
    assert receipt["naive_timestamps_are_utc"] is True


def test_bars_outside_the_attested_coverage_are_never_finalized(tmp_path):
    body = _ROWS + f"{BAR}20:00Z,1.08000,90,SELL\n"
    tape = _tape(tmp_path, body=body)
    receipt = _receipt(
        tmp_path,
        tape=tape,
        covers_through="2024-03-04T10:15:00Z",
    )
    assert receipt["counts"]["events_sealed"] == 1
    assert receipt["exclusion_reasons"] == {"outside_attested_coverage": 1}
    assert receipt["excluded_sample"][0]["bar_open"] == "2024-03-04T10:15:00.000Z"


def test_sparse_and_gappy_bars_are_excluded_with_an_audit_trail(tmp_path):
    body = (
        _ROWS
        + f"{BAR}15:00Z,1.08000,90,SELL\n"
        + f"{BAR}45:00Z,1.08000,90,SELL\n"
        + f"{BAR}59:00Z,1.08010,10,BUY\n"
    )
    tape = _tape(tmp_path, body=body)
    receipt = _receipt(
        tmp_path,
        tape=tape,
        min_trades_per_bar=2,
        max_intrabar_gap_seconds=600,
    )
    assert receipt["counts"]["events_sealed"] == 1
    assert receipt["exclusion_reasons"] == {
        "below_min_trades_per_bar": 1,
        "intrabar_gap_exceeds_limit": 1,
    }


def test_empty_result_is_never_sealed(tmp_path):
    tape = _tape(tmp_path)
    with pytest.raises(TapeIngestError, match="nothing can be sealed"):
        _receipt(
            tmp_path,
            tape=tape,
            covers_from="2025-01-01T00:00:00Z",
            covers_through="2025-01-02T00:00:00Z",
        )


def test_existing_sidecar_directory_is_never_overwritten(tmp_path):
    _receipt(tmp_path)
    tape = tmp_path / "tape.csv"
    with pytest.raises(TapeIngestError, match="immutable"):
        run(_args(tmp_path, tape))


def test_sidecar_staging_is_removed_when_manifest_sealing_fails(
    tmp_path,
    monkeypatch,
):
    tape = _tape(tmp_path)

    def fail_manifest(*args, **kwargs):
        raise RuntimeError("manifest sealing failed")

    monkeypatch.setattr(orderflow_ingest, "write_manifest", fail_manifest)
    with pytest.raises(RuntimeError, match="manifest sealing failed"):
        _receipt(tmp_path, tape=tape)
    assert not (tmp_path / "sidecar").exists()
    assert list(tmp_path.glob(".sidecar.partial-*")) == []


def test_existing_receipt_is_immutable_and_prevents_publication(tmp_path):
    tape = _tape(tmp_path)
    receipt_path = tmp_path / "sidecar.build_receipt.json"
    receipt_path.write_text("keep", encoding="utf-8")
    with pytest.raises(TapeIngestError, match="receipt already exists"):
        _receipt(tmp_path, tape=tape)
    assert receipt_path.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "sidecar").exists()


def test_receipt_cannot_overwrite_the_input_tape(tmp_path):
    tape = _tape(tmp_path)
    original = tape.read_text(encoding="utf-8")
    with pytest.raises(TapeIngestError, match="must not overwrite an input tape"):
        _receipt(tmp_path, tape=tape, receipt=str(tape))
    assert tape.read_text(encoding="utf-8") == original
    assert not (tmp_path / "sidecar").exists()


def test_input_tape_change_during_conversion_prevents_sealing(
    tmp_path,
    monkeypatch,
):
    tape = _tape(tmp_path)
    real_snapshot = orderflow_ingest._snapshot_inputs
    calls = 0

    def changing_snapshot(paths):
        nonlocal calls
        calls += 1
        snapshot = real_snapshot(paths)
        if calls == 2:
            snapshot[0] = {**snapshot[0], "sha256": "0" * 64}
        return snapshot

    monkeypatch.setattr(orderflow_ingest, "_snapshot_inputs", changing_snapshot)
    with pytest.raises(TapeIngestError, match="changed during conversion"):
        _receipt(tmp_path, tape=tape)
    assert not (tmp_path / "sidecar").exists()


def test_receipt_inside_the_sealed_root_is_refused(tmp_path):
    tape = _tape(tmp_path)
    with pytest.raises(TapeIngestError, match="outside the sealed sidecar root"):
        _receipt(
            tmp_path,
            tape=tape,
            receipt=str(tmp_path / "sidecar" / "receipt.json"),
        )


def test_month_sharding_is_sealed_by_the_manifest(tmp_path):
    _receipt(tmp_path)
    manifest = json.loads(
        (tmp_path / "sidecar" / "manifest.json").read_text(encoding="utf-8")
    )
    assert [item["path"] for item in manifest["files"]] == ["events/2024-03.json"]
    # Loading re-derives every hash from disk.
    assert AbsorptionEventDataset.load(tmp_path / "sidecar").events
