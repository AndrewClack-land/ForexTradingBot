from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

import monitoring.orca_eodhd_ingest as ingest
from monitoring.orca_eodhd_ingest import (
    ORCA_PAPER_UNIVERSE,
    EodhdClient,
    OrcaEodhdConfig,
    OrcaEodhdIngestError,
    build_aligned_eodhd_panel,
    fetch_orca_universe,
    publish_panel_atomically,
    refresh_eodhd_panel,
)
from monitoring.orca_monitor import load_price_panel


UTC = timezone.utc
START = date(2024, 1, 1)


def _row(ticker_index: int, offset: int) -> dict[str, object]:
    return {
        "date": (START + timedelta(days=offset)).isoformat(),
        "close": 9000.0 + offset,
        "adjusted_close": 100.0 + ticker_index + offset / 100.0,
    }


def _responses(
    rows: int = 380,
    *,
    omissions: dict[str, set[int]] | None = None,
) -> dict[str, list[dict[str, object]]]:
    missing = omissions or {}
    return {
        ticker: [
            _row(ticker_index, offset)
            for offset in range(rows)
            if offset not in missing.get(ticker, set())
        ]
        for ticker_index, ticker in enumerate(ORCA_PAPER_UNIVERSE)
    }


def _as_of(rows: int) -> datetime:
    latest_exchange_date = START + timedelta(days=rows - 1)
    return datetime.combine(
        latest_exchange_date + timedelta(days=1),
        time(hour=12),
        tzinfo=UTC,
    )


class _Response:
    status = 200

    def __init__(self, payload: str) -> None:
        self._payload = payload.encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def read(self) -> bytes:
        return self._payload


def test_official_request_contract_and_token_is_not_exposed() -> None:
    requests = []

    def opener(request: object, *, timeout: float) -> _Response:
        requests.append((request, timeout))
        return _Response('[{"date":"2026-08-19","adjusted_close":100.5}]')

    token = "secret + token"
    client = EodhdClient(
        token,
        timeout_seconds=7,
        from_date=date(2015, 1, 1),
        to_date=date(2026, 8, 19),
        opener=opener,
    )
    rows = client.fetch_daily("SPY")

    assert rows[0]["adjusted_close"] == 100.5
    request, timeout = requests[0]
    parsed = urlparse(request.full_url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "eodhd.com"
    assert parsed.path == "/api/eod/SPY.US"
    assert parse_qs(parsed.query) == {
        "api_token": [token],
        "fmt": ["json"],
        "period": ["d"],
        "order": ["a"],
        "from": ["2015-01-01"],
        "to": ["2026-08-19"],
    }
    assert timeout == 7

    def failing_opener(request: object, *, timeout: float) -> object:
        del timeout
        raise OSError(request.full_url)

    failing = EodhdClient(token, opener=failing_opener)
    with pytest.raises(OrcaEodhdIngestError) as caught:
        failing.fetch_daily("SPY")
    assert token not in str(caught.value)
    assert "secret+%2B+token" not in str(caught.value)


def test_explicit_history_range_is_validated() -> None:
    with pytest.raises(OrcaEodhdIngestError, match="to_date cannot precede"):
        EodhdClient(
            "token",
            from_date=date(2026, 1, 2),
            to_date=date(2026, 1, 1),
        )

    config = OrcaEodhdConfig(
        output_path=Path("prices.parquet"),
        history_start_date="2015-01-01",  # type: ignore[arg-type]
    )
    assert config.history_start_date == date(2015, 1, 1)


def test_fetch_requires_every_exact_paper_symbol_in_order() -> None:
    class Client:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def fetch_daily(self, ticker: str) -> list[dict[str, object]]:
            self.seen.append(ticker)
            return [_row(0, 0)]

    client = Client()
    result = fetch_orca_universe(client)  # type: ignore[arg-type]
    assert tuple(result) == ORCA_PAPER_UNIVERSE
    assert tuple(client.seen) == ORCA_PAPER_UNIVERSE
    assert len(result) == 24


def test_panel_uses_adjusted_close_and_conservative_availability() -> None:
    responses = _responses(380)
    responses["SPY"].append(
        {
            "date": (START + timedelta(days=380)).isoformat(),
            "close": 1.0,
            "adjusted_close": 999.0,
        }
    )
    panel = build_aligned_eodhd_panel(
        responses,
        as_of_utc=_as_of(380),
    )

    assert tuple(panel.columns) == ORCA_PAPER_UNIVERSE
    assert len(panel) == 380
    assert panel.index[0] == pd.Timestamp("2024-01-02T12:00:00Z")
    assert panel.index[-1] == pd.Timestamp(_as_of(380))
    assert panel.iloc[0]["SPY"] == pytest.approx(100.0)
    assert panel.iloc[-1]["SPY"] != 999.0
    assert panel.index.tz is not None


def test_ffill_is_limited_to_five_then_remaining_gap_is_dropped() -> None:
    responses = _responses(20, omissions={"UUP": set(range(10, 16))})
    panel = build_aligned_eodhd_panel(
        responses,
        as_of_utc=_as_of(20),
        minimum_aligned_rows=1,
        ffill_limit=5,
    )

    missing_sixth = pd.Timestamp(
        datetime.combine(
            START + timedelta(days=15 + 1),
            time(hour=12),
            tzinfo=UTC,
        )
    )
    assert missing_sixth not in panel.index
    assert len(panel) == 19
    assert panel.loc[panel.index[14], "UUP"] == panel.loc[panel.index[9], "UUP"]


def test_missing_symbol_and_short_alignment_fail_closed() -> None:
    missing = _responses(380)
    missing.pop("UUP")
    with pytest.raises(OrcaEodhdIngestError, match="universe mismatch"):
        build_aligned_eodhd_panel(missing, as_of_utc=_as_of(380))

    with pytest.raises(OrcaEodhdIngestError, match="at least 372"):
        build_aligned_eodhd_panel(_responses(371), as_of_utc=_as_of(371))


def test_duplicate_or_invalid_adjusted_close_fails_closed() -> None:
    duplicated = _responses(380)
    duplicated["SPY"].append(dict(duplicated["SPY"][0]))
    with pytest.raises(OrcaEodhdIngestError, match="duplicate date"):
        build_aligned_eodhd_panel(duplicated, as_of_utc=_as_of(380))

    invalid = _responses(380)
    invalid["GLD"][10]["adjusted_close"] = 0.0
    with pytest.raises(OrcaEodhdIngestError, match="finite and positive"):
        build_aligned_eodhd_panel(invalid, as_of_utc=_as_of(380))


def test_atomic_csv_is_accepted_by_monitor_loader(tmp_path: Path) -> None:
    panel = build_aligned_eodhd_panel(
        _responses(380),
        as_of_utc=_as_of(380),
    )
    target = tmp_path / "prices.csv"
    config = OrcaEodhdConfig(output_path=target)
    publish_panel_atomically(panel, config)

    loaded = load_price_panel(target)
    assert tuple(loaded.columns) == ORCA_PAPER_UNIVERSE
    assert loaded.index.equals(panel.index)
    assert not list(tmp_path.glob(".prices.*.csv"))


def test_failed_candidate_validation_preserves_last_good(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "prices.csv"
    target.write_text("last-good\n", encoding="utf-8")
    panel = build_aligned_eodhd_panel(
        _responses(380),
        as_of_utc=_as_of(380),
    )

    def reject(path: Path) -> pd.DataFrame:
        del path
        raise ValueError("injected rejection")

    monkeypatch.setattr(ingest, "load_price_panel", reject)
    with pytest.raises(OrcaEodhdIngestError, match="monitor rejected"):
        publish_panel_atomically(panel, OrcaEodhdConfig(output_path=target))
    assert target.read_text(encoding="utf-8") == "last-good\n"
    assert not list(tmp_path.glob(".prices.*.csv"))


def test_refresh_publishes_only_after_all_24_fetches(tmp_path: Path) -> None:
    responses = _responses(380)

    class Client:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def fetch_daily(self, ticker: str) -> list[dict[str, object]]:
            self.calls.append(ticker)
            return responses[ticker]

    client = Client()
    result = refresh_eodhd_panel(
        api_token="not-used-by-injected-client",
        config=OrcaEodhdConfig(output_path=tmp_path / "prices.csv"),
        as_of_utc=_as_of(380),
        client=client,  # type: ignore[arg-type]
    )
    assert result["status"] == "PUBLISHED"
    assert result["asset_count"] == 24
    assert tuple(client.calls) == ORCA_PAPER_UNIVERSE


def test_cli_redacts_raw_and_urlencoded_token(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    token = "top secret+value"
    monkeypatch.setenv("EODHD_API_TOKEN", token)

    def fail(**kwargs: object) -> dict[str, object]:
        del kwargs
        raise RuntimeError(f"raw={token}; encoded=top+secret%2Bvalue")

    monkeypatch.setattr(ingest, "refresh_eodhd_panel", fail)
    assert ingest.main(["--output", "prices.csv"]) == 1
    captured = capsys.readouterr()
    assert token not in captured.err
    assert "top+secret%2Bvalue" not in captured.err
    assert "[REDACTED]" in captured.err


def test_systemd_artifacts_enforce_native_pipeline_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    service = (root / "deploy" / "orca-eodhd.service").read_text(encoding="utf-8")
    timer = (root / "deploy" / "orca-eodhd.timer").read_text(encoding="utf-8")
    environment = (root / "deploy" / "orca-eodhd.env.example").read_text(
        encoding="utf-8"
    )

    assert (
        "ExecStart=/opt/forexbot-orca/venv/bin/python -m monitoring.orca_eodhd_ingest"
        in service
    )
    assert "WorkingDirectory=/home/ubuntu/forexbot" in service
    assert "EnvironmentFile=/etc/forexbot-orca/orca-eodhd.env" in service
    assert "ReadWritePaths=/srv/forexbot-backtest/orca" in service
    assert "ProtectHome=read-only" in service
    assert "OnSuccess=orca-prediction.service" in service
    assert "00:15:00 UTC" in timer
    assert "12:15:00 UTC" in timer
    assert "Persistent=true" in timer
    assert "EODHD_API_TOKEN=" in environment
    assert (
        "ORCA_EODHD_OUTPUT_PATH=/srv/forexbot-backtest/orca/prices.parquet"
    ) in environment
    assert "ORCA_EODHD_START_DATE=2015-01-01" in environment
    assert "ORCA_EODHD_MIN_ALIGNED_ROWS=372" in environment
    assert "ORCA_EODHD_FFILL_LIMIT=5" in environment
    assert "ORCA_EODHD_AVAILABILITY_HOUR_UTC=12" in environment
