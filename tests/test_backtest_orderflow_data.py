from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from backtest.orderflow_data import (
    AbsorptionEventDataset,
    OrderFlowDataValidationError,
    REQUIRED_POLARITY,
    build_manifest,
    event_checksum,
    write_manifest,
)


def _event(**updates):
    record = {
        "schema_version": 1,
        "revision": 0,
        "timeframe": "15m",
        "source_symbol": "EURUSD",
        "symbol": "EURUSD",
        "bar_open": "2026-07-25T10:00:00Z",
        "bar_close": "2026-07-25T10:15:00Z",
        "available_at": "2026-07-25T10:15:03Z",
        "finalized": True,
        "high_buy_volume": 120.0,
        "high_sell_volume": 40.0,
        "low_buy_volume": 30.0,
        "low_sell_volume": 90.0,
    }
    record.update(updates)
    record["checksum"] = event_checksum(record)
    return record


def _write_json(path, records):
    path.write_text(
        json.dumps(records, ensure_ascii=False),
        encoding="utf-8",
    )


def _seal(root):
    return write_manifest(
        root,
        source="Quantower",
        venue="FxPro",
        symbol_mapping={"EURUSD": "EURUSD"},
    )


def _canonical_hash(payload):
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rewrite_manifest(path, update):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    update(manifest)
    payload = dict(manifest)
    payload.pop("manifest_sha256")
    manifest["manifest_sha256"] = _canonical_hash(payload)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def test_json_sidecar_is_sealed_and_event_visibility_is_causal(tmp_path):
    _write_json(tmp_path / "events.json", [_event()])
    manifest_path = _seal(tmp_path)

    dataset = AbsorptionEventDataset.load(tmp_path)

    assert manifest_path.name == "manifest.json"
    assert dataset.symbols == ("EURUSD",)
    assert len(dataset.events) == 1
    assert len(dataset.manifest_sha256) == 64
    assert len(dataset.content_sha256) == 64
    assert dataset.events[0].source == "Quantower"
    assert dataset.events[0].venue == "FxPro"
    assert dataset.events[0].polarity == REQUIRED_POLARITY

    # The candle is closed at 10:15, but the finalized footprint is only
    # available three seconds later.  Both conditions are enforced.
    assert dataset.event_asof(
        "EURUSD",
        "2026-07-25T10:00:00Z",
        "2026-07-25T10:15:00Z",
    ) is None
    visible = dataset.event_asof(
        "eurusd",
        "2026-07-25T10:00:00+00:00",
        "2026-07-25T10:15:03+00:00",
    )
    assert visible is dataset.events[0]
    assert visible.revision == 0
    assert visible.to_record()["revision"] == 0
    assert visible.high_buy_volume == pytest.approx(120.0)
    assert dataset.event_asof(
        "GBPUSD",
        "2026-07-25T10:00:00Z",
        "2026-07-25T10:15:03Z",
    ) is None


def test_parquet_sidecar_loads_with_the_same_contract(tmp_path):
    frame = pd.DataFrame([_event()])
    frame["bar_open"] = pd.to_datetime(frame["bar_open"], utc=True)
    frame["bar_close"] = pd.to_datetime(frame["bar_close"], utc=True)
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    frame.to_parquet(tmp_path / "events.parquet", index=False)
    _seal(tmp_path)

    dataset = AbsorptionEventDataset.load(tmp_path)
    assert len(dataset.events) == 1
    assert dataset.events[0].bar_close == pd.Timestamp(
        "2026-07-25T10:15:00Z"
    )


def test_manifest_and_content_hashes_are_deterministic(tmp_path):
    later = _event(
        bar_open="2026-07-25T10:15:00Z",
        bar_close="2026-07-25T10:30:00Z",
        available_at="2026-07-25T10:30:01Z",
    )
    _write_json(tmp_path / "events.json", [later, _event()])

    first = build_manifest(
        tmp_path,
        source="Quantower",
        venue="FxPro",
        symbol_mapping={"EURUSD": "EURUSD"},
    )
    second = build_manifest(
        tmp_path,
        source="Quantower",
        venue="FxPro",
        symbol_mapping={"eurusd": "eurusd"},
    )

    assert first == second
    assert first["content_sha256"] == second["content_sha256"]
    assert first["manifest_sha256"] == second["manifest_sha256"]


def test_loader_requires_a_manifest_and_rejects_unlisted_shards(tmp_path):
    _write_json(tmp_path / "events.json", [_event()])
    with pytest.raises(OrderFlowDataValidationError, match="manifest is required"):
        AbsorptionEventDataset.load(tmp_path)

    _seal(tmp_path)
    _write_json(
        tmp_path / "late-extra.json",
        [
            _event(
                bar_open="2026-07-25T10:15:00Z",
                bar_close="2026-07-25T10:30:00Z",
                available_at="2026-07-25T10:30:00Z",
            )
        ],
    )
    with pytest.raises(
        OrderFlowDataValidationError,
        match="does not seal the sidecar directory",
    ):
        AbsorptionEventDataset.load(tmp_path)


def test_loader_rejects_file_mutation_after_sealing(tmp_path):
    path = tmp_path / "events.json"
    _write_json(path, [_event()])
    _seal(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(OrderFlowDataValidationError, match="content mismatch"):
        AbsorptionEventDataset.load(tmp_path)


def test_event_checksum_detects_semantic_mutation(tmp_path):
    row = _event()
    row["high_buy_volume"] = 121.0
    _write_json(tmp_path / "events.json", [row])

    with pytest.raises(
        OrderFlowDataValidationError,
        match="event checksum does not match",
    ):
        build_manifest(
            tmp_path,
            source="Quantower",
            venue="FxPro",
            symbol_mapping={"EURUSD": "EURUSD"},
        )


def test_revision_is_mandatory(tmp_path):
    row = _event()
    row.pop("revision")
    _write_json(tmp_path / "events.json", [row])

    with pytest.raises(OrderFlowDataValidationError, match="missing event field"):
        build_manifest(
            tmp_path,
            source="Quantower",
            venue="FxPro",
            symbol_mapping={"EURUSD": "EURUSD"},
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {
                "bar_open": "2026-07-25T10:01:00Z",
                "bar_close": "2026-07-25T10:16:00Z",
                "available_at": "2026-07-25T10:16:00Z",
            },
            "15-minute UTC boundary",
        ),
        ({"bar_open": "2026-07-25T10:00:00"}, "timezone-aware UTC"),
        ({"bar_close": "2026-07-25T10:14:59Z"}, "bar_open \\+ 15 minutes"),
        ({"available_at": "2026-07-25T10:14:59Z"}, "cannot precede"),
        ({"finalized": False}, "boolean true"),
        ({"low_sell_volume": -1}, "finite non-negative"),
        ({"high_buy_volume": float("inf")}, "finite non-negative"),
        ({"schema_version": 2}, "schema_version"),
        ({"revision": 1}, "revision must be the integer 0"),
        ({"revision": 0.0}, "revision must be the integer 0"),
        ({"timeframe": "5m"}, "timeframe"),
    ],
)
def test_invalid_event_contract_fails_closed(tmp_path, updates, message):
    # Invalid rows cannot receive a valid checksum through the public helper.
    row = {
        **_event(),
        **updates,
        "checksum": "0" * 64,
    }
    _write_json(tmp_path / "events.json", [row])

    with pytest.raises(OrderFlowDataValidationError, match=message):
        build_manifest(
            tmp_path,
            source="Quantower",
            venue="FxPro",
            symbol_mapping={"EURUSD": "EURUSD"},
        )


def test_duplicate_or_conflicting_symbol_open_is_rejected(tmp_path):
    first = _event()
    second = _event(high_buy_volume=999.0)
    _write_json(tmp_path / "a.json", [first])
    _write_json(tmp_path / "b.json", [second])

    with pytest.raises(
        OrderFlowDataValidationError,
        match="Duplicate conflicting event",
    ):
        build_manifest(
            tmp_path,
            source="Quantower",
            venue="FxPro",
            symbol_mapping={"EURUSD": "EURUSD"},
        )


def test_provenance_mapping_and_polarity_are_pinned(tmp_path):
    _write_json(tmp_path / "events.json", [_event()])
    manifest_path = _seal(tmp_path)

    _rewrite_manifest(
        manifest_path,
        lambda manifest: manifest.update(
            {"symbol_mapping": {"EURUSD": "GBPUSD"}}
        ),
    )
    with pytest.raises(OrderFlowDataValidationError, match="identity-only"):
        AbsorptionEventDataset.load(tmp_path)

    _seal(tmp_path)
    _rewrite_manifest(
        manifest_path,
        lambda manifest: manifest.update({"polarity": "unknown"}),
    )
    with pytest.raises(OrderFlowDataValidationError, match="polarity"):
        AbsorptionEventDataset.load(tmp_path)

    with pytest.raises(OrderFlowDataValidationError, match="source"):
        build_manifest(
            tmp_path,
            source=" ",
            venue="FxPro",
            symbol_mapping={"EURUSD": "EURUSD"},
        )


def test_proxy_and_many_to_one_symbol_mappings_fail_closed(tmp_path):
    _write_json(tmp_path / "events.json", [_event()])

    with pytest.raises(OrderFlowDataValidationError, match="identity-only"):
        build_manifest(
            tmp_path,
            source="Quantower",
            venue="FxPro",
            symbol_mapping={"BTCUSDT": "EURUSD"},
        )

    with pytest.raises(
        OrderFlowDataValidationError,
        match="duplicate target.*many-to-one",
    ):
        build_manifest(
            tmp_path,
            source="Quantower",
            venue="FxPro",
            symbol_mapping={
                "EURUSD": "EURUSD",
                "EURUSD.FXPRO": "EURUSD",
            },
        )


def test_content_hash_is_checked_even_when_manifest_hash_is_valid(tmp_path):
    _write_json(tmp_path / "events.json", [_event()])
    manifest_path = _seal(tmp_path)
    _rewrite_manifest(
        manifest_path,
        lambda manifest: manifest.update({"content_sha256": "0" * 64}),
    )

    with pytest.raises(OrderFlowDataValidationError, match="content_sha256"):
        AbsorptionEventDataset.load(tmp_path)


def test_query_timestamps_must_be_utc_and_bar_aligned(tmp_path):
    _write_json(tmp_path / "events.json", [_event()])
    _seal(tmp_path)
    dataset = AbsorptionEventDataset.load(tmp_path)

    with pytest.raises(OrderFlowDataValidationError, match="timezone-aware UTC"):
        dataset.event_asof(
            "EURUSD",
            "2026-07-25T10:00:00",
            "2026-07-25T10:16:00Z",
        )
    with pytest.raises(OrderFlowDataValidationError, match="15-minute UTC boundary"):
        dataset.event_asof(
            "EURUSD",
            "2026-07-25T10:01:00Z",
            "2026-07-25T10:16:00Z",
        )
    with pytest.raises(OrderFlowDataValidationError, match="must use UTC"):
        dataset.event_asof(
            "EURUSD",
            "2026-07-25T10:00:00Z",
            "2026-07-25T12:16:00+02:00",
        )
