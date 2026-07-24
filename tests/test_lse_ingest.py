from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from backtest.data import HistoricalDataset
from backtest.lse_ingest import (
    DEFAULT_TARGET_TIMEFRAMES,
    LSEIngestError,
    create_lse_client,
    fetch_lse_rest,
    import_lse_snapshot,
    normalize_lse_candles,
    parse_symbol_spec,
    resample_lse_candles,
)


def _minute_rows(
    start: str,
    periods: int,
    *,
    symbol: str = "EUR/USD",
    timeframe: str = "1m",
):
    timestamps = pd.date_range(start, periods=periods, freq="1min", tz="UTC")
    rows = []
    for index, timestamp in enumerate(timestamps):
        price = 1.10 + index / 10_000_000
        rows.append(
            {
                "symbol": symbol,
                "tf": timeframe,
                "timestamp": timestamp.isoformat(),
                "open": price,
                "high": price + 0.0002,
                "low": price - 0.0002,
                "close": price + 0.0001,
                "volume": 0,
            }
        )
    return rows


class FakeRestClient:
    def __init__(
        self,
        rows,
        *,
        server_cap=None,
        catalog=None,
    ):
        self.rows = list(rows)
        self.calls = []
        self.server_cap = server_cap
        self.catalog_calls = 0
        symbols = {str(row["symbol"]) for row in self.rows}
        self.catalog = catalog or [
            {
                "symbol": symbol,
                "dataset": "commodity" if symbol == "XAU/USD" else "fx",
            }
            for symbol in sorted(symbols)
        ]

    def datasets(self):
        self.catalog_calls += 1
        return list(self.catalog)

    def candles(
        self,
        symbol,
        timeframe,
        *,
        start,
        end,
        limit,
        order,
        dataset,
    ):
        assert len(start) == 10 and start[4] == "-" and start[7] == "-"
        assert len(end) == 10 and end[4] == "-" and end[7] == "-"
        self.calls.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "start": start,
                "end": end,
                "limit": limit,
                "order": order,
                "dataset": dataset,
            }
        )
        lower = pd.Timestamp(start, tz="UTC")
        upper = pd.Timestamp(end, tz="UTC")
        selected = [
            row
            for row in self.rows
            if str(row["symbol"]) == symbol
            and lower <= pd.Timestamp(row["timestamp"])
            and pd.Timestamp(row["timestamp"]) < upper
        ]
        selected.sort(
            key=lambda row: pd.Timestamp(row["timestamp"]),
            reverse=order == "desc",
        )
        effective_limit = min(limit, self.server_cap or limit)
        return selected[:effective_limit]


class FakeExportClient:
    def __init__(self, rows, *, catalog=None):
        self.rows = list(rows)
        self.calls = []
        symbols = {str(row["symbol"]) for row in self.rows}
        self.catalog = catalog or [
            {
                "symbol": symbol,
                "dataset": "commodity" if symbol == "XAU/USD" else "fx",
            }
            for symbol in sorted(symbols)
        ]
        self.catalog_calls = 0

    def datasets(self):
        self.catalog_calls += 1
        return list(self.catalog)

    def history(self, symbol, **kwargs):
        self.calls.append((symbol, kwargs))
        path = (
            Path(kwargs["dest"])
            / f"{kwargs['dataset']}_{symbol.replace('/', '_')}_1m.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        lower = pd.Timestamp(kwargs["start"], tz="UTC")
        upper = pd.Timestamp(kwargs["end"], tz="UTC")
        selected = [
            row
            for row in self.rows
            if str(row["symbol"]) == symbol
            and lower <= pd.Timestamp(row["timestamp"]) < upper
        ]
        pd.DataFrame(selected).rename(columns={"timestamp": "ts"}).to_parquet(
            path, index=False
        )
        return str(path)


@pytest.mark.parametrize(
    ("raw", "bot", "provider"),
    [
        ("EURUSD", "EURUSD", "EUR/USD"),
        ("EUR/USD", "EURUSD", "EUR/USD"),
        ("GOLD", "GOLD", "XAU/USD"),
        ("XAUUSD=XAU/USD", "XAUUSD", "XAU/USD"),
    ],
)
def test_symbol_mapping(raw, bot, provider):
    spec = parse_symbol_spec(raw)
    assert spec.bot_symbol == bot
    assert spec.provider_symbol == provider


def test_symbol_mapping_rejects_path_traversal():
    with pytest.raises(LSEIngestError, match="Unsafe"):
        parse_symbol_spec("../SECRET=EUR/USD")


def test_normalization_fails_closed_on_excess_duplicate_timestamps():
    rows = _minute_rows("2026-01-01T00:00:00Z", 2)
    rows.extend([dict(rows[0]), dict(rows[0])])
    with pytest.raises(LSEIngestError, match="duplicate timestamp ratio"):
        normalize_lse_candles(
            pd.DataFrame(rows),
            expected_provider_symbol="EUR/USD",
            start="2026-01-01",
            end="2026-01-02",
            max_duplicate_ratio=0.1,
        )


def test_normalization_rejects_conflicting_duplicate_below_ratio_limit():
    rows = _minute_rows("2026-01-01T00:00:00Z", 2000)
    conflicting = dict(rows[0])
    conflicting["close"] += 0.00005
    conflicting["volume"] = 1
    rows.append(conflicting)
    with pytest.raises(LSEIngestError, match="conflicting duplicate timestamp"):
        normalize_lse_candles(
            pd.DataFrame(rows),
            expected_provider_symbol="EUR/USD",
            start="2026-01-01",
            end="2026-01-03",
            max_duplicate_ratio=0.001,
        )


def test_h4_and_d1_use_broker_wall_clock_across_dst():
    rows = _minute_rows("2026-03-28T22:00:00Z", 47 * 60)
    minute, _ = normalize_lse_candles(
        pd.DataFrame(rows),
        expected_provider_symbol="EUR/USD",
        start="2026-03-28T22:00:00Z",
        end="2026-03-30T21:00:00Z",
    )

    h4 = resample_lse_candles(
        minute,
        timeframe="4h",
        start="2026-03-28T22:00:00Z",
        end="2026-03-30T21:00:00Z",
        bar_timezone="Europe/Athens",
    )
    assert set(h4.index.tz_convert("Europe/Athens").hour).issubset(
        {0, 4, 8, 12, 16, 20}
    )

    daily = resample_lse_candles(
        minute,
        timeframe="1d",
        start="2026-03-28T22:00:00Z",
        end="2026-03-30T21:00:00Z",
        bar_timezone="Europe/Athens",
    )
    spring_day = daily.loc[pd.Timestamp("2026-03-28T22:00:00Z")]
    assert spring_day["bar_close_time"] == pd.Timestamp("2026-03-29T21:00:00Z")
    assert (
        spring_day["bar_close_time"] - pd.Timestamp("2026-03-28T22:00:00Z")
        == pd.Timedelta(hours=23)
    )


def test_rest_import_builds_immutable_snapshot_and_pages(tmp_path, monkeypatch):
    rows = _minute_rows("2026-07-01T00:00:00Z", 3 * 24 * 60)
    client = FakeRestClient(rows)
    target = tmp_path / "snapshot"
    monkeypatch.setenv("LSE_API_KEY", "test-secret-must-not-be-written")

    result = import_lse_snapshot(
        output=target,
        symbols=["EURUSD"],
        start="2026-07-01",
        end="2026-07-04",
        transport="rest",
        page_limit=5000,
        rest_min_interval=0,
        client=client,
    )

    assert result.output == target.resolve()
    assert len(client.calls) == 2
    assert all(call["dataset"] == "fx" for call in client.calls)
    assert {path.name for path in target.glob("*.parquet")} == {
        "EURUSD_1m.parquet",
        "EURUSD_5m.parquet",
        "EURUSD_15m.parquet",
        "EURUSD_1h.parquet",
        "EURUSD_4h.parquet",
        "EURUSD_1d.parquet",
    }

    dataset = HistoricalDataset.load(target)
    assert set(dataset.keys) == {
        ("EURUSD", "1m"),
        ("EURUSD", "5m"),
        ("EURUSD", "15m"),
        ("EURUSD", "1h"),
        ("EURUSD", "4h"),
        ("EURUSD", "1d"),
    }
    daily = dataset.get_frame("EURUSD", "1d")
    assert daily.index[0] == pd.Timestamp("2026-07-01T21:00:00Z")

    # The explicit DST-aware close is used for causality, then hidden from the
    # strategy-facing frame.
    before_close = dataset.frame_asof(
        "EURUSD", "1d", pd.Timestamp("2026-07-02T20:59:59Z")
    )
    after_close = dataset.frame_asof(
        "EURUSD", "1d", pd.Timestamp("2026-07-02T21:00:00Z")
    )
    assert before_close.empty
    assert len(after_close) == 1
    assert "bar_close_time" not in after_close.columns

    provenance = (target / "source_manifest.json").read_text(encoding="utf-8")
    assert "test-secret-must-not-be-written" not in provenance
    source_manifest = json.loads(provenance)
    assert source_manifest["bar_timezone"] == "Europe/Athens"
    assert "snapshot_manifest_sha256" not in source_manifest
    assert source_manifest["symbols"][0]["dataset"] == "fx"
    assert source_manifest["files"][0]["sha256"]
    assert "bucket_quality" in source_manifest["files"][0]
    assert (target / "manifest.json").is_file()
    assert result.manifest_sha256 == dataset.manifest_sha256

    with pytest.raises(LSEIngestError, match="already exists"):
        import_lse_snapshot(
            output=target,
            symbols=["EURUSD"],
            start="2026-07-01",
            end="2026-07-04",
            transport="rest",
            rest_min_interval=0,
            client=client,
        )


def test_export_import_uses_m1_not_default_tick(tmp_path):
    rows = _minute_rows("2026-07-01T00:00:00Z", 3 * 24 * 60)
    client = FakeExportClient(rows)
    target = tmp_path / "export-snapshot"

    result = import_lse_snapshot(
        output=target,
        symbols=["EURUSD"],
        start="2026-07-01",
        end="2026-07-04",
        transport="export",
        client=client,
    )

    assert result.output == target.resolve()
    assert len(client.calls) == 1
    _, kwargs = client.calls[0]
    assert kwargs["timeframe"] == "1m"
    assert kwargs["dataframe"] is False
    assert kwargs["dataset"] == "fx"
    assert not (tmp_path / ".export-snapshot.lse-checkpoint").exists()


def test_missing_key_fails_before_importing_sdk(monkeypatch):
    monkeypatch.delenv("LSE_API_KEY", raising=False)
    with pytest.raises(LSEIngestError, match="LSE_API_KEY"):
        create_lse_client()


def test_default_timeframes_include_five_minutes():
    assert DEFAULT_TARGET_TIMEFRAMES == ("1m", "5m", "15m", "1h", "4h", "1d")


def test_direct_import_defaults_to_quota_safe_rest_transport():
    assert (
        inspect.signature(import_lse_snapshot).parameters["transport"].default
        == "rest"
    )


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("symbol", "GBP/USD", "unexpected symbol"),
        ("tf", "15m", "unexpected timeframe"),
    ],
)
def test_normalization_rejects_mislabeled_series(column, bad_value, message):
    rows = _minute_rows("2026-01-01T00:00:00Z", 60)
    rows[0][column] = bad_value
    with pytest.raises(LSEIngestError, match=message):
        normalize_lse_candles(
            pd.DataFrame(rows),
            expected_provider_symbol="EUR/USD",
            start="2026-01-01T00:00:00Z",
            end="2026-01-01T01:00:00Z",
        )


def test_rest_uses_date_windows_and_filters_partial_days_later():
    rows = _minute_rows("2026-01-01T00:00:00Z", 2 * 24 * 60)
    client = FakeRestClient(rows)

    raw, report = fetch_lse_rest(
        client,
        provider_symbol="EUR/USD",
        dataset="fx",
        start="2026-01-01T12:00:00Z",
        end="2026-01-02T06:00:00Z",
        page_limit=5000,
        rest_min_interval=0,
    )
    minute, _ = normalize_lse_candles(
        raw,
        expected_provider_symbol="EUR/USD",
        start="2026-01-01T12:00:00Z",
        end="2026-01-02T06:00:00Z",
    )

    assert len(raw) == 2 * 24 * 60
    assert len(minute) == 18 * 60
    assert [(call["start"], call["end"], call["order"], call["limit"]) for call in client.calls] == [
        ("2026-01-01", "2026-01-03", "asc", 5000),
        ("2026-01-01", "2026-01-03", "desc", 1),
    ]
    assert report["utc_date_windows"] == 1
    assert report["window_days"] == 3
    assert report["tail_probes"] == 1
    assert report["pagination_overlap_rows"] == 0


def test_rest_skips_empty_weekend_date_windows_and_continues():
    rows = _minute_rows("2026-01-02T00:00:00Z", 2)
    rows += _minute_rows("2026-01-05T00:00:00Z", 2)
    client = FakeRestClient(rows)
    raw, report = fetch_lse_rest(
        client,
        provider_symbol="EUR/USD",
        dataset="fx",
        start="2026-01-02",
        end="2026-01-06",
        page_limit=1441,
        rest_min_interval=0,
    )

    assert len(raw) == 4
    assert report["utc_date_windows"] == 4
    assert report["empty_utc_date_windows"] == 2
    assert report["pages"] == 2
    assert report["requests"] == 8
    assert all(len(call["start"]) == 10 for call in client.calls)


def test_rest_tail_probe_rejects_hidden_plan_cap():
    client = FakeRestClient(
        _minute_rows("2026-01-01T00:00:00Z", 10),
        server_cap=3,
    )
    with pytest.raises(LSEIngestError, match="capped.*bulk export"):
        fetch_lse_rest(
            client,
            provider_symbol="EUR/USD",
            dataset="fx",
            start="2026-01-01T00:00:00Z",
            end="2026-01-01T00:10:00Z",
            page_limit=5000,
            rest_min_interval=0,
        )


def test_rest_tail_probe_rejects_conflicting_latest_candle():
    class ConflictingTailClient(FakeRestClient):
        def candles(self, *args, **kwargs):
            rows = super().candles(*args, **kwargs)
            if kwargs["order"] == "desc" and rows:
                rows = [dict(row) for row in rows]
                rows[0]["close"] += 0.001
                rows[0]["high"] += 0.001
            return rows

    with pytest.raises(LSEIngestError, match="tail-probe candle conflicts"):
        fetch_lse_rest(
            ConflictingTailClient(
                _minute_rows("2026-01-01T00:00:00Z", 10)
            ),
            provider_symbol="EUR/USD",
            dataset="fx",
            start="2026-01-01T00:00:00Z",
            end="2026-01-01T00:10:00Z",
            rest_min_interval=0,
        )


def test_rest_rejects_response_reaching_configured_window_limit():
    rows = _minute_rows("2026-01-01T00:00:00Z", 1)
    rows *= 1441
    client = FakeRestClient(rows)
    with pytest.raises(LSEIngestError, match="reached its row limit"):
        fetch_lse_rest(
            client,
            provider_symbol="EUR/USD",
            dataset="fx",
            start="2026-01-01",
            end="2026-01-02",
            page_limit=1441,
            rest_min_interval=0,
        )
    assert len(client.calls) == 1


def test_rest_rejects_limit_that_cannot_hold_a_complete_m1_day():
    client = FakeRestClient(_minute_rows("2026-01-01T00:00:00Z", 1))
    with pytest.raises(LSEIngestError, match="at least 1441"):
        fetch_lse_rest(
            client,
            provider_symbol="EUR/USD",
            dataset="fx",
            start="2026-01-01",
            end="2026-01-02",
            page_limit=1440,
            rest_min_interval=0,
        )
    assert client.calls == []


def test_rest_rejects_rows_outside_requested_date_window():
    class DateIgnoringClient(FakeRestClient):
        def candles(self, *args, **kwargs):
            self.calls.append(kwargs)
            return self.rows[: kwargs["limit"]]

    client = DateIgnoringClient(
        _minute_rows("2026-01-02T00:00:00Z", 1)
    )
    with pytest.raises(LSEIngestError, match="outside.*UTC-day window"):
        fetch_lse_rest(
            client,
            provider_symbol="EUR/USD",
            dataset="fx",
            start="2026-01-01",
            end="2026-01-02",
            rest_min_interval=0,
        )


def test_rest_rejects_null_sdk_response():
    class NullClient(FakeRestClient):
        def candles(self, *args, **kwargs):
            self.calls.append(kwargs)
            return None

    client = NullClient(_minute_rows("2026-01-01T00:00:00Z", 1))
    with pytest.raises(LSEIngestError, match="null ascending response"):
        fetch_lse_rest(
            client,
            provider_symbol="EUR/USD",
            dataset="fx",
            start="2026-01-01",
            end="2026-01-02",
            rest_min_interval=0,
        )
    assert len(client.calls) == 1


def test_rest_rejects_current_open_utc_day():
    client = FakeRestClient(_minute_rows("2026-07-25T00:00:00Z", 1))
    with pytest.raises(LSEIngestError, match="still-open UTC day"):
        fetch_lse_rest(
            client,
            provider_symbol="EUR/USD",
            dataset="fx",
            start="2026-07-24",
            end="2026-07-25T00:01:00Z",
            rest_min_interval=0,
            now="2026-07-25T12:00:00Z",
        )
    assert client.calls == []


def test_rest_rejects_misaligned_tail_probe_timestamp():
    class MisalignedTailClient(FakeRestClient):
        def candles(self, *args, **kwargs):
            rows = super().candles(*args, **kwargs)
            if kwargs["order"] == "desc" and rows:
                rows = [dict(row) for row in rows]
                rows[0]["timestamp"] = (
                    pd.Timestamp(rows[0]["timestamp"])
                    + pd.Timedelta(seconds=30)
                ).isoformat()
            return rows

    with pytest.raises(LSEIngestError, match="tail-probe.*exact UTC minute"):
        fetch_lse_rest(
            MisalignedTailClient(
                _minute_rows("2026-01-01T00:00:00Z", 10)
            ),
            provider_symbol="EUR/USD",
            dataset="fx",
            start="2026-01-01T00:00:00Z",
            end="2026-01-01T00:10:00Z",
            rest_min_interval=0,
        )


def test_rest_date_windows_do_not_hide_provider_duplicates():
    rows = _minute_rows("2026-01-01T00:00:00Z", 3)
    rows.extend([dict(rows[0]), dict(rows[0])])
    raw, _ = fetch_lse_rest(
        FakeRestClient(rows),
        provider_symbol="EUR/USD",
        dataset="fx",
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T00:03:00Z",
        rest_min_interval=0,
    )
    with pytest.raises(LSEIngestError, match="duplicate timestamp ratio"):
        normalize_lse_candles(
            raw,
            expected_provider_symbol="EUR/USD",
            start="2026-01-01T00:00:00Z",
            end="2026-01-01T00:03:00Z",
            max_duplicate_ratio=0.1,
        )


def test_rest_429_uses_configured_retry_delay():
    class RateLimitedClient(FakeRestClient):
        attempts = 0

        def candles(self, *args, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                error = RuntimeError("rate limited")
                error.status = 429
                raise error
            return super().candles(*args, **kwargs)

    sleeps = []
    client = RateLimitedClient(_minute_rows("2026-01-01T00:00:00Z", 2))
    fetch_lse_rest(
        client,
        provider_symbol="EUR/USD",
        dataset="fx",
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T00:02:00Z",
        rest_min_interval=0.35,
        retry_after_seconds=61,
        sleep=sleeps.append,
    )
    assert 61.0 in sleeps
    assert sleeps.count(0.35) == 2


def test_catalog_resolves_dataset_per_symbol_before_requests(tmp_path):
    rows = _minute_rows("2026-01-01T00:00:00Z", 24 * 60)
    rows += _minute_rows(
        "2026-01-01T00:00:00Z",
        24 * 60,
        symbol="XAU/USD",
    )
    client = FakeRestClient(rows)
    target = tmp_path / "mixed-assets"

    result = import_lse_snapshot(
        output=target,
        symbols=["EURUSD", "GOLD"],
        start="2026-01-01",
        end="2026-01-02",
        timeframes=["1m"],
        transport="rest",
        rest_min_interval=0,
        client=client,
    )

    datasets_by_symbol = {
        call["symbol"]: call["dataset"] for call in client.calls
    }
    assert datasets_by_symbol["EUR/USD"] == "fx"
    assert datasets_by_symbol["XAU/USD"] == "commodity"
    assert {item["dataset"] for item in result.symbols} == {"fx", "commodity"}
    assert client.catalog_calls == 1


def test_catalog_mismatch_and_missing_symbol_fail_before_data_request(tmp_path):
    rows = _minute_rows("2026-01-01T00:00:00Z", 60)
    client = FakeRestClient(rows)
    with pytest.raises(LSEIngestError, match="not dataset"):
        import_lse_snapshot(
            output=tmp_path / "wrong-dataset",
            symbols=["EURUSD"],
            start="2026-01-01",
            end="2026-01-02",
            timeframes=["1m"],
            transport="rest",
            dataset="commodity",
            rest_min_interval=0,
            client=client,
        )
    assert client.calls == []

    with pytest.raises(LSEIngestError, match="not present"):
        import_lse_snapshot(
            output=tmp_path / "unknown-symbol",
            symbols=["ZZZZZZ"],
            start="2026-01-01",
            end="2026-01-02",
            timeframes=["1m"],
            transport="rest",
            rest_min_interval=0,
            client=client,
        )
    assert client.calls == []


def test_catalog_rejects_unsafe_dataset_label_before_requests(tmp_path):
    rows = _minute_rows("2026-01-01T00:00:00Z", 60)
    client = FakeRestClient(
        rows,
        catalog=[{"symbol": "EUR/USD", "dataset": "../../incoming"}],
    )
    with pytest.raises(LSEIngestError, match="unsafe dataset label"):
        import_lse_snapshot(
            output=tmp_path / "unsafe-dataset",
            symbols=["EURUSD"],
            start="2026-01-01",
            end="2026-01-02",
            timeframes=["1m"],
            transport="export",
            client=client,
        )
    assert client.calls == []


def test_export_chunks_merge_and_preflight_job_cap(tmp_path):
    rows = _minute_rows("2026-01-01T00:00:00Z", 5 * 24 * 60)
    client = FakeExportClient(rows)
    result = import_lse_snapshot(
        output=tmp_path / "chunked",
        symbols=["EURUSD"],
        start="2026-01-01",
        end="2026-01-06",
        timeframes=["1m"],
        transport="export",
        export_chunk_days=2,
        client=client,
    )

    assert len(client.calls) == 3
    assert result.symbols[0]["jobs"] == 3
    assert result.symbols[0]["chunk_overlap_rows"] == 0

    capped = FakeExportClient(
        rows
        + _minute_rows(
            "2026-01-01T00:00:00Z",
            5 * 24 * 60,
            symbol="GBP/USD",
        )
    )
    with pytest.raises(LSEIngestError, match="exceeding max_export_jobs=5"):
        import_lse_snapshot(
            output=tmp_path / "too-many-jobs",
            symbols=["EURUSD", "GBPUSD"],
            start="2026-01-01",
            end="2026-01-06",
            timeframes=["1m"],
            transport="export",
            export_chunk_days=2,
            max_export_jobs=5,
            client=capped,
        )
    assert capped.calls == []
    assert capped.catalog_calls == 0


def test_export_checkpoint_is_reused_after_failed_validation(tmp_path):
    rows = _minute_rows("2026-01-01T00:00:00Z", 24 * 60)
    client = FakeExportClient(rows)
    target = tmp_path / "resume"

    with pytest.raises(LSEIngestError, match="requested range edges"):
        import_lse_snapshot(
            output=target,
            symbols=["EURUSD"],
            start="2026-01-01",
            end="2026-01-10",
            timeframes=["1m"],
            transport="export",
            client=client,
        )
    checkpoint = tmp_path / ".resume.lse-checkpoint"
    assert checkpoint.is_dir()
    assert len(client.calls) == 1

    result = import_lse_snapshot(
        output=target,
        symbols=["EURUSD"],
        start="2026-01-01",
        end="2026-01-10",
        timeframes=["1m"],
        transport="export",
        max_edge_gap_days=10,
        client=client,
    )
    assert result.symbols[0]["checkpoint_reused"] == 1
    assert len(client.calls) == 1
    assert not checkpoint.exists()


def test_new_export_job_removes_stale_sdk_partial_file(tmp_path):
    rows = _minute_rows("2026-01-01T00:00:00Z", 24 * 60)
    stale_part = (
        tmp_path
        / ".stale-part.lse-checkpoint"
        / "EURUSD"
        / "0000_2026-01-01_2026-01-02"
        / "fx_EUR_USD_1m.parquet.part"
    )
    stale_part.parent.mkdir(parents=True)
    stale_part.write_bytes(b"bytes-from-an-old-export-job")

    class RejectStalePartClient(FakeExportClient):
        def history(self, symbol, **kwargs):
            assert not stale_part.exists()
            return super().history(symbol, **kwargs)

    result = import_lse_snapshot(
        output=tmp_path / "stale-part",
        symbols=["EURUSD"],
        start="2026-01-01",
        end="2026-01-02",
        timeframes=["1m"],
        transport="export",
        client=RejectStalePartClient(rows),
    )
    assert result.output.is_dir()


def test_runtime_environment_is_path_free_and_bound_to_final_manifest(tmp_path):
    release = tmp_path / "release-commit.txt"
    lock = tmp_path / "requirements.freeze.txt"
    release.write_text("A" * 40 + "\n", encoding="utf-8")
    lock.write_text("pandas==2.3.2\n", encoding="utf-8")
    rows = _minute_rows("2026-01-01T00:00:00Z", 24 * 60)
    client = FakeRestClient(rows)

    first = import_lse_snapshot(
        output=tmp_path / "first",
        symbols=["EURUSD"],
        start="2026-01-01",
        end="2026-01-02",
        timeframes=["1m"],
        rest_min_interval=0,
        release_commit_file=release,
        environment_lock_file=lock,
        client=client,
    )
    first_source = json.loads(
        (first.output / "source_manifest.json").read_text(encoding="utf-8")
    )
    runtime = first_source["runtime_environment"]
    assert runtime["release_commit"] == "a" * 40
    assert runtime["lock"] == {
        "filename": "environment.freeze.txt",
        "bytes": lock.stat().st_size,
        "sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
    }
    assert str(tmp_path) not in json.dumps(runtime)
    embedded_lock = first.output / "environment.freeze.txt"
    assert embedded_lock.read_text(encoding="utf-8") == "pandas==2.3.2\n"
    assert any(
        item["path"] == "environment.freeze.txt"
        and item["kind"] == "metadata"
        for item in HistoricalDataset.load(first.output).manifest["files"]
    )

    lock.write_text("pandas==2.3.3\n", encoding="utf-8")
    second = import_lse_snapshot(
        output=tmp_path / "second",
        symbols=["EURUSD"],
        start="2026-01-01",
        end="2026-01-02",
        timeframes=["1m"],
        rest_min_interval=0,
        release_commit_file=release,
        environment_lock_file=lock,
        client=client,
    )
    second_source = json.loads(
        (second.output / "source_manifest.json").read_text(encoding="utf-8")
    )
    assert (
        second_source["runtime_environment"]["lock"]["sha256"]
        != runtime["lock"]["sha256"]
    )
    assert second.manifest_sha256 != first.manifest_sha256


def test_runtime_environment_rejects_invalid_or_partial_provenance(tmp_path):
    release = tmp_path / "release.txt"
    lock = tmp_path / "freeze.txt"
    release.write_text("not-a-commit\n", encoding="utf-8")
    lock.write_text("pandas==2.3.2\n", encoding="utf-8")
    client = FakeRestClient(_minute_rows("2026-01-01T00:00:00Z", 60))

    with pytest.raises(LSEIngestError, match="40- or 64-character"):
        import_lse_snapshot(
            output=tmp_path / "invalid",
            symbols=["EURUSD"],
            start="2026-01-01",
            end="2026-01-02",
            timeframes=["1m"],
            release_commit_file=release,
            environment_lock_file=lock,
            client=client,
        )
    with pytest.raises(LSEIngestError, match="must be provided together"):
        import_lse_snapshot(
            output=tmp_path / "partial",
            symbols=["EURUSD"],
            start="2026-01-01",
            end="2026-01-02",
            timeframes=["1m"],
            release_commit_file=release,
            client=client,
        )
    assert client.catalog_calls == 0
    assert client.calls == []


def test_edge_coverage_rejects_truncated_history():
    rows = _minute_rows("2026-01-01T00:00:00Z", 24 * 60)
    with pytest.raises(LSEIngestError, match="requested range edges"):
        normalize_lse_candles(
            pd.DataFrame(rows),
            expected_provider_symbol="EUR/USD",
            start="2026-01-01",
            end="2026-01-10",
            max_edge_gap_days=4,
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("timestamp", "2026-01-01T00:00:01Z", "exact UTC minute"),
        ("symbol", None, "null/blank symbol"),
        ("tf", "", "null/blank timeframe"),
        ("volume", "not-a-number", "invalid volume"),
    ],
)
def test_normalization_rejects_misaligned_or_corrupt_source_fields(
    field,
    bad_value,
    message,
):
    rows = _minute_rows("2026-01-01T00:00:00Z", 2)
    rows[0][field] = bad_value
    with pytest.raises(LSEIngestError, match=message):
        normalize_lse_candles(
            pd.DataFrame(rows),
            expected_provider_symbol="EUR/USD",
            start="2026-01-01T00:00:00Z",
            end="2026-01-01T00:02:00Z",
        )


def test_internal_history_gap_is_rejected_even_when_edges_are_covered():
    rows = _minute_rows("2026-01-01T00:00:00Z", 2)
    rows += _minute_rows("2026-01-09T23:58:00Z", 2)
    with pytest.raises(LSEIngestError, match="internal gap"):
        normalize_lse_candles(
            pd.DataFrame(rows),
            expected_provider_symbol="EUR/USD",
            start="2026-01-01",
            end="2026-01-10",
            max_edge_gap_days=4,
            max_internal_gap_days=4,
        )


def test_resample_drops_incomplete_source_buckets_and_reports_count():
    rows = _minute_rows("2026-01-01T00:00:00Z", 17)
    minute, _ = normalize_lse_candles(
        pd.DataFrame(rows),
        expected_provider_symbol="EUR/USD",
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T00:30:00Z",
    )
    bars = resample_lse_candles(
        minute,
        timeframe="15m",
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T00:30:00Z",
    )
    assert len(bars) == 1
    assert bars.attrs["bucket_quality"]["dropped_low_coverage"] == 1

    sparse_rows = [rows[0], rows[14]]
    sparse, _ = normalize_lse_candles(
        pd.DataFrame(sparse_rows),
        expected_provider_symbol="EUR/USD",
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T00:15:00Z",
    )
    with pytest.raises(LSEIngestError, match="min_bucket_coverage"):
        resample_lse_candles(
            sparse,
            timeframe="15m",
            start="2026-01-01T00:00:00Z",
            end="2026-01-01T00:15:00Z",
        )
