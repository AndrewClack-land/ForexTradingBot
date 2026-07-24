from __future__ import annotations

import json

import pandas as pd
import pytest

from backtest.__main__ import main as cli_main
from backtest.data import DataValidationError, HistoricalDataset


def _record(timestamp: str, price: float, **extra):
    return {
        "symbol": "EURUSD",
        "tf": "1h",
        "timestamp": timestamp,
        "open": price,
        "high": price + 0.002,
        "low": price - 0.002,
        "close": price + 0.001,
        "volume": 10,
        **extra,
    }


def _write_json(path, records):
    path.write_text(json.dumps(records), encoding="utf-8")


def test_json_loader_sorts_dedupes_and_frames_asof_are_causal(tmp_path):
    records = [
        _record("2026-01-01T10:00:00Z", 1.10),
        _record("2026-01-01T09:00:00Z", 1.09),
        _record("2026-01-01T09:00:00Z", 1.095),  # last duplicate wins
        _record("2026-01-01T11:00:00Z", 9.99),  # future sentinel
    ]
    _write_json(tmp_path / "EURUSD_1h.json", records)

    dataset = HistoricalDataset.load(tmp_path)
    full = dataset.get_frame("eurusd", "H1")
    assert full.index.is_monotonic_increasing
    assert full.index.is_unique
    assert len(full) == 3
    assert full.iloc[0]["open"] == pytest.approx(1.095)

    # At 10:30 the 09:00 H1 candle is closed; 10:00 and 11:00 are not.
    causal = dataset.frame_asof("EURUSD", "1h", "2026-01-01T10:30:00Z")
    assert list(causal.index) == [pd.Timestamp("2026-01-01T09:00:00Z")]
    assert causal.iloc[-1]["open"] == pytest.approx(1.095)
    assert dataset.coverage()[0].duplicate_rows == 1

    # File and manifest hashes are content-deterministic.
    again = HistoricalDataset.load(tmp_path)
    assert dataset.manifest_sha256 == again.manifest_sha256
    assert len(dataset.manifest["files"][0]["sha256"]) == 64


def test_future_mutation_cannot_change_asof_view(tmp_path):
    path = tmp_path / "EURUSD_15m.json"
    base = [
        {**_record(f"2026-01-01T09:{minute:02d}:00Z", 1.0 + minute / 10000), "tf": "15m"}
        for minute in (0, 15, 30, 45)
    ]
    _write_json(path, base)
    before = HistoricalDataset.load(tmp_path).frame_asof(
        "EURUSD", "15m", "2026-01-01T09:30:00Z"
    )

    base[-1] = {**base[-1], "open": 8.0, "high": 9.0, "low": 7.0, "close": 8.5}
    base.append({**_record("2026-01-01T10:00:00Z", 9.0), "tf": "15m"})
    _write_json(path, base)
    after = HistoricalDataset.load(tmp_path).frame_asof(
        "EURUSD", "15m", "2026-01-01T09:30:00Z"
    )
    pd.testing.assert_frame_equal(before, after)


def test_frames_asof_caps_each_timeframe_at_299_closed_bars(tmp_path):
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    records = []
    for index in range(305):
        records.append(
            {
                **_record((start + pd.Timedelta(minutes=15 * index)).isoformat(), 1.0),
                "tf": "15m",
            }
        )
    _write_json(tmp_path / "EURUSD_15m.json", records)

    frames = HistoricalDataset.load(tmp_path).frames_asof(
        "EURUSD", start + pd.Timedelta(minutes=15 * 305)
    )
    assert set(frames) == {"15M"}
    assert len(frames["15M"]) == 299
    assert frames["15M"].index[-1] == start + pd.Timedelta(minutes=15 * 304)


def test_parquet_loader_supports_timestamp_index_and_filename_key(tmp_path):
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1, 2],
        },
        index=pd.DatetimeIndex(
            ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
            name="timestamp",
        ),
    )
    frame.to_parquet(tmp_path / "GOLD_1d.parquet")
    dataset = HistoricalDataset.load(tmp_path)
    causal = dataset.frame_asof("GOLD", "D1", "2026-01-02T12:00:00Z")
    assert len(causal) == 1
    assert causal.iloc[0]["open"] == 100.0


def test_explicit_bar_close_time_controls_causal_visibility(tmp_path):
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-03-28T22:00:00Z")],
            "bar_close_time": [pd.Timestamp("2026-03-29T21:00:00Z")],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
        }
    )
    frame.to_parquet(tmp_path / "EURUSD_1d.parquet", index=False)

    dataset = HistoricalDataset.load(tmp_path)
    assert dataset.frame_asof(
        "EURUSD", "1d", "2026-03-29T20:59:59Z"
    ).empty
    visible = dataset.frame_asof(
        "EURUSD", "1d", "2026-03-29T21:00:00Z"
    )
    assert len(visible) == 1
    assert "bar_close_time" not in visible.columns


def test_mixed_legacy_and_explicit_close_shards_fill_legacy_close(tmp_path):
    legacy_dir = tmp_path / "legacy"
    explicit_dir = tmp_path / "explicit"
    legacy_dir.mkdir()
    explicit_dir.mkdir()
    _write_json(
        legacy_dir / "EURUSD_1d.json",
        [
            {
                **_record("2026-03-27T22:00:00Z", 1.0),
                "tf": "1d",
            }
        ],
    )
    _write_json(
        explicit_dir / "EURUSD_1d.json",
        [
            {
                **_record("2026-03-28T22:00:00Z", 1.1),
                "tf": "1d",
                "bar_close_time": "2026-03-29T21:00:00Z",
            }
        ],
    )

    dataset = HistoricalDataset.load(tmp_path)
    frame = dataset.get_frame("EURUSD", "1d")
    assert frame["bar_close_time"].isna().sum() == 0
    assert list(frame["bar_close_time"]) == [
        pd.Timestamp("2026-03-28T22:00:00Z"),
        pd.Timestamp("2026-03-29T21:00:00Z"),
    ]


def test_duplicate_legacy_row_cannot_replace_explicit_dst_close(tmp_path):
    explicit_dir = tmp_path / "a_explicit"
    legacy_dir = tmp_path / "z_legacy"
    explicit_dir.mkdir()
    legacy_dir.mkdir()
    open_time = "2026-10-24T21:00:00Z"
    _write_json(
        explicit_dir / "EURUSD_1d.json",
        [
            {
                **_record(open_time, 1.0),
                "tf": "1d",
                "bar_close_time": "2026-10-25T22:00:00Z",
            }
        ],
    )
    _write_json(
        legacy_dir / "EURUSD_1d.json",
        [{**_record(open_time, 1.0), "tf": "1d"}],
    )

    dataset = HistoricalDataset.load(tmp_path)
    frame = dataset.get_frame("EURUSD", "1d")
    assert frame["bar_close_time"].iloc[0] == pd.Timestamp(
        "2026-10-25T22:00:00Z"
    )
    assert dataset.frame_asof(
        "EURUSD", "1d", "2026-10-25T21:30:00Z"
    ).empty


def test_conflicting_explicit_closes_for_duplicate_open_are_rejected(tmp_path):
    for folder, close_time in (
        ("first", "2026-10-25T21:00:00Z"),
        ("second", "2026-10-25T22:00:00Z"),
    ):
        directory = tmp_path / folder
        directory.mkdir()
        _write_json(
            directory / "EURUSD_1d.json",
            [
                {
                    **_record("2026-10-24T21:00:00Z", 1.0),
                    "tf": "1d",
                    "bar_close_time": close_time,
                }
            ],
        )

    with pytest.raises(DataValidationError, match="Conflicting explicit"):
        HistoricalDataset.load(tmp_path)


def test_explicit_close_with_wrong_timeframe_duration_is_rejected(tmp_path):
    _write_json(
        tmp_path / "EURUSD_1h.json",
        [
            {
                **_record("2026-01-01T00:00:00Z", 1.0),
                "bar_close_time": "2026-01-01T00:01:00Z",
            }
        ],
    )

    with pytest.raises(DataValidationError, match="invalid 1h bar close duration"):
        HistoricalDataset.load(tmp_path)


def test_explicit_close_cannot_overlap_next_candle_open(tmp_path):
    records = [
        {
            **_record("2026-01-01T00:00:00Z", 1.0),
            "tf": "4h",
            "bar_close_time": "2026-01-01T05:00:00Z",
        },
        {
            **_record("2026-01-01T04:00:00Z", 1.1),
            "tf": "4h",
            "bar_close_time": "2026-01-01T08:00:00Z",
        },
    ]
    _write_json(tmp_path / "EURUSD_4h.json", records)

    with pytest.raises(DataValidationError, match="overlapping candle"):
        HistoricalDataset.load(tmp_path)


def test_invalid_ohlc_is_rejected(tmp_path):
    bad = _record("2026-01-01T00:00:00Z", 1.0)
    bad["high"] = 0.5
    _write_json(tmp_path / "EURUSD_1h.json", [bad])
    with pytest.raises(DataValidationError, match="impossible OHLC"):
        HistoricalDataset.load(tmp_path)


def test_audit_fails_closed_when_d1_history_is_too_short(tmp_path, capsys):
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    records = [
        {
            **_record((start + pd.Timedelta(days=index)).isoformat(), 1.0),
            "tf": "1d",
        }
        for index in range(20)
    ]
    _write_json(tmp_path / "EURUSD_1d.json", records)
    code = cli_main(["audit", "--data", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 3
    assert "WFO BLOCKED" in output
    assert "only 20 D1 bars" in output
    assert "at least 12 required" in output


def test_readiness_requires_both_365_d1_bars_and_twelve_months(tmp_path):
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    records = [
        {
            **_record((start + pd.Timedelta(hours=23 * index)).isoformat(), 1.0),
            "tf": "1d",
            "bar_close_time": (
                start + pd.Timedelta(hours=23 * (index + 1))
            ).isoformat(),
        }
        for index in range(365)
    ]
    _write_json(tmp_path / "EURUSD_1d.json", records)
    readiness = HistoricalDataset.load(tmp_path).audit_readiness()
    assert readiness["wfo_ready"] is False
    assert readiness["symbols"]["EURUSD"]["d1_bars"] == 365
    assert any("at least 12 required" in reason for reason in readiness["reasons"])


def test_readiness_accepts_full_year_with_365_d1_bars(tmp_path):
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    records = [
        {
            **_record((start + pd.Timedelta(days=index)).isoformat(), 1.0),
            "tf": "1d",
        }
        for index in range(365)
    ]
    _write_json(tmp_path / "EURUSD_1d.json", records)
    readiness = HistoricalDataset.load(tmp_path).audit_readiness()
    assert readiness["wfo_ready"] is True
    assert readiness["reasons"] == []


def test_readiness_uses_explicit_final_d1_close(tmp_path):
    records = [
        {
            **_record("2025-01-01T01:00:00Z", 1.0),
            "tf": "1d",
            "bar_close_time": "2025-01-02T01:00:00Z",
        },
        {
            **_record("2025-01-31T00:00:00Z", 1.1),
            "tf": "1d",
            "bar_close_time": "2025-02-01T01:00:00Z",
        },
    ]
    _write_json(tmp_path / "EURUSD_1d.json", records)

    readiness = HistoricalDataset.load(tmp_path).audit_readiness(
        min_d1_bars=2,
        min_months=1,
    )
    assert readiness["wfo_ready"] is True
    assert readiness["symbols"]["EURUSD"]["end"] == "2025-02-01T01:00:00+00:00"


def test_source_manifest_is_hashed_as_provenance_without_being_parsed(tmp_path):
    _write_json(
        tmp_path / "EURUSD_1h.json",
        [_record("2026-01-01T00:00:00Z", 1.0)],
    )
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(
        json.dumps({"provider": "lse", "snapshot": "first"}),
        encoding="utf-8",
    )

    first = HistoricalDataset.load(tmp_path)
    provenance = [
        item
        for item in first.manifest["files"]
        if item["path"] == "source_manifest.json"
    ]
    assert len(provenance) == 1
    assert provenance[0]["kind"] == "metadata"

    source_manifest.write_text(
        json.dumps({"provider": "lse", "snapshot": "changed"}),
        encoding="utf-8",
    )
    second = HistoricalDataset.load(tmp_path)
    assert first.manifest_sha256 != second.manifest_sha256


def test_stored_manifest_rejects_snapshot_tampering(tmp_path):
    candle_path = tmp_path / "EURUSD_1h.json"
    records = [_record("2026-01-01T00:00:00Z", 1.0)]
    _write_json(candle_path, records)
    dataset = HistoricalDataset.load(tmp_path)
    dataset.write_manifest(tmp_path / "manifest.json")

    records[0]["open"] = 1.01
    records[0]["high"] = 1.02
    records[0]["close"] = 1.015
    _write_json(candle_path, records)

    with pytest.raises(DataValidationError, match="stored manifest does not match"):
        HistoricalDataset.load(tmp_path)


def test_verify_command_requires_and_checks_stored_manifest(tmp_path, capsys):
    candle_path = tmp_path / "EURUSD_1h.json"
    records = [_record("2026-01-01T00:00:00Z", 1.0)]
    _write_json(candle_path, records)

    assert cli_main(["verify", "--data", str(tmp_path)]) == 2
    assert "stored manifest.json is missing" in capsys.readouterr().err

    dataset = HistoricalDataset.load(tmp_path)
    dataset.write_manifest(tmp_path / "manifest.json")
    assert cli_main(["verify", "--data", str(tmp_path)]) == 0
    assert "SNAPSHOT VERIFIED" in capsys.readouterr().out
