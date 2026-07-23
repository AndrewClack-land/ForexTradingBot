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
            **_record((start + pd.Timedelta(hours=12 * index)).isoformat(), 1.0),
            "tf": "1d",
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
