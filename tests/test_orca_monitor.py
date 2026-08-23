from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import pytest

from core.orca_spectral import (
    ORCA_FX_GOLD_4_PROFILE,
    OrcaSpectralConfig,
    OrcaValidationError,
)
from monitoring.orca_monitor import (
    ORCA_MONITOR_SCHEMA,
    ORCA_PREDICTION_SCHEMA,
    AtomicSnapshotStore,
    JsonPredictionProvider,
    OrcaMonitorConfig,
    OrcaMonitorEngine,
    OrcaRequestHandler,
    build_monitor_snapshot,
    classify_regime,
    compute_prediction_hash,
    compute_snapshot_hash,
    load_price_panel,
    _parser,
    make_bound_handler,
    materialize_current_snapshot,
    resolve_universe_arguments,
)


AS_OF = datetime(2026, 8, 17, 22, 0, tzinfo=timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _prices(rows: int = 90, assets: int = 6) -> pd.DataFrame:
    index = pd.date_range(
        AS_OF - timedelta(days=rows),
        periods=rows,
        freq="1D",
        tz="UTC",
    )
    time = np.arange(rows, dtype=float)
    values = {}
    for column in range(assets):
        common = 0.0012 * time
        cycle = 0.02 * np.sin(time / (4.0 + column))
        offset = 0.003 * column * np.cos(time / 7.0)
        values[f"A{column + 1}"] = 100.0 * np.exp(common + cycle + offset)
    return pd.DataFrame(values, index=index)


def _spectral_config() -> OrcaSpectralConfig:
    return OrcaSpectralConfig(
        rolling_windows=(20, 30),
        ewm_halflife=10,
        ewm_min_periods=20,
    )


def _monitor_config() -> OrcaMonitorConfig:
    return OrcaMonitorConfig(
        refresh_seconds=10,
        stale_after_seconds=200_000,
        expected_assets=6,
        minimum_assets=5,
    )


def _snapshot_without_prediction() -> dict[str, object]:
    return build_monitor_snapshot(
        _prices(),
        generated_at_utc=AS_OF,
        spectral_config=_spectral_config(),
        monitor_config=_monitor_config(),
    )


def _prediction(snapshot: dict[str, object], **updates: object) -> dict[str, object]:
    market_as_of = datetime.fromisoformat(
        str(snapshot["as_of_market_utc"]).replace("Z", "+00:00")
    )
    payload: dict[str, object] = {
        "schema_version": ORCA_PREDICTION_SCHEMA,
        "as_of_market_utc": snapshot["as_of_market_utc"],
        "known_at_utc": _iso(AS_OF - timedelta(hours=1)),
        "trained_through_utc": _iso(market_as_of - timedelta(days=1)),
        "feature_registry_hash": snapshot["feature_registry_hash"],
        "feature_values_hash": snapshot["feature_values_hash"],
        "model_id": "model-1",
        "model_artifact_hash": "b" * 64,
        "status": "OOS_VALIDATED",
        "p_rally": 0.31,
        "p_crash": 0.09,
        "rally_rank": 0.81,
        "crash_rank": 0.20,
        "calibration_status": "UNCALIBRATED",
        "validation": {"bcd_auc": 0.73},
        "executing": False,
    }
    payload.update(updates)
    payload["prediction_hash"] = compute_prediction_hash(payload)
    return payload


def test_regime_uses_only_explicit_paper_zones() -> None:
    assert classify_regime(rally_rank=0.80, crash_rank=0.20) == {
        "regime": "RALLY",
        "risk_mode": "RISK_ON",
        "suggested_equity_exposure": 1.5,
        "exposure_status": "PAPER_EXPLICIT_ZONE",
    }
    assert (
        classify_regime(rally_rank=0.30, crash_rank=0.70)["suggested_equity_exposure"]
        == 0.0
    )
    intermediate = classify_regime(rally_rank=0.50, crash_rank=0.45)
    assert intermediate["regime"] == "CAUTION"
    assert intermediate["suggested_equity_exposure"] is None
    assert intermediate["exposure_status"] == "INTERMEDIATE_MAP_NOT_PUBLISHED"


def test_snapshot_matches_spectral_dataclasses_and_market_freshness() -> None:
    base = _snapshot_without_prediction()
    snapshot = build_monitor_snapshot(
        _prices(),
        generated_at_utc=AS_OF,
        spectral_config=_spectral_config(),
        monitor_config=_monitor_config(),
        prediction=_prediction(base),
    )

    assert snapshot["schema_version"] == ORCA_MONITOR_SCHEMA
    assert snapshot["executing"] is False
    assert snapshot["regime"] == "RALLY"
    assert snapshot["asset_count"] == 6
    assert snapshot["as_of_market_utc"] == "2026-08-16T22:00:00Z"
    assert snapshot["age_seconds"] == pytest.approx(86_400)
    assert snapshot["freshness"] == "FRESH"
    assert snapshot["snapshot_hash"] == compute_snapshot_hash(snapshot)
    assert len(snapshot["networks"]) == 3
    assert snapshot["model"]["validation"]["bcd_auc"] == pytest.approx(0.73)

    for network in snapshot["networks"]:
        assert set(network["node_coordinates_3d"]) == set(snapshot["symbols"])
        assert set(network["signed_edges"]) == {"0.3", "0.5", "0.7"}
        assert set(network["graph_metrics"]) == {"0.3", "0.5", "0.7"}
        assert len(network["eigenvectors"]) == 6
        assert "lambda1_lambda2" in network
        assert "marchenko_pastur_q" in network
        assert "spectral_gap_ratio" not in network
        assert "coordinates" not in network
        assert "edges" not in network


def test_prediction_provider_enforces_hash_exact_features_and_pit(tmp_path) -> None:
    base = _snapshot_without_prediction()
    path = tmp_path / "prediction.json"
    payload = _prediction(base)
    path.write_text(json.dumps(payload), encoding="utf-8")
    provider = JsonPredictionProvider(path)
    arguments = {
        "as_of_market_utc": datetime.fromisoformat(
            str(base["as_of_market_utc"]).replace("Z", "+00:00")
        ),
        "known_at_utc": AS_OF,
        "feature_registry_hash": str(base["feature_registry_hash"]),
        "feature_values_hash": str(base["feature_values_hash"]),
    }
    result = provider.read_prediction(**arguments)
    assert result is not None
    assert result["p_rally"] == pytest.approx(0.31)

    tampered = dict(payload, p_rally=0.99)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        provider.read_prediction(**arguments)

    future = _prediction(base, known_at_utc=_iso(AS_OF + timedelta(seconds=1)))
    path.write_text(json.dumps(future), encoding="utf-8")
    with pytest.raises(ValueError, match="not known"):
        provider.read_prediction(**arguments)

    stale = _prediction(
        base,
        as_of_market_utc=_iso(AS_OF - timedelta(days=2)),
    )
    path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly match"):
        provider.read_prediction(**arguments)

    wrong_values = _prediction(base, feature_values_hash="c" * 64)
    path.write_text(json.dumps(wrong_values), encoding="utf-8")
    with pytest.raises(ValueError, match="feature values"):
        provider.read_prediction(**arguments)

    missing_status = dict(payload)
    missing_status.pop("status")
    missing_status["prediction_hash"] = compute_prediction_hash(missing_status)
    path.write_text(json.dumps(missing_status), encoding="utf-8")
    with pytest.raises(ValueError, match="status"):
        provider.read_prediction(**arguments)


def test_atomic_store_failed_replace_preserves_last_good(tmp_path, monkeypatch) -> None:
    store = AtomicSnapshotStore(tmp_path / "latest.json")
    first = _snapshot_without_prediction()
    store.publish(first)
    assert store.read() == first

    second = dict(first, freshness="STALE")
    second["snapshot_hash"] = compute_snapshot_hash(second)

    def fail_replace(source: object, target: object) -> None:
        del source, target
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        store.publish(second)
    assert store.read() == first
    assert not list(tmp_path.glob("*.tmp"))


def test_refresh_failure_preserves_last_good_and_sets_health_error(tmp_path) -> None:
    prices_path = tmp_path / "prices.csv"
    _prices().rename_axis("timestamp").to_csv(prices_path)
    store = AtomicSnapshotStore(tmp_path / "latest.json")
    engine = OrcaMonitorEngine(
        prices_path=prices_path,
        snapshot_store=store,
        spectral_config=_spectral_config(),
        monitor_config=_monitor_config(),
        clock=lambda: AS_OF,
    )
    first = engine.refresh()
    assert store.read() == first
    successful_refresh = engine.last_refresh_utc

    prices_path.write_text("timestamp,A1\ninvalid,not-a-number\n", encoding="utf-8")
    with pytest.raises(Exception):
        engine.refresh()
    assert store.read() == first
    assert engine.last_refresh_utc == successful_refresh
    assert engine.last_error is not None


def test_bound_handler_recomputes_age_and_health_fails_closed(tmp_path) -> None:
    dashboard = tmp_path / "dashboard"
    dashboard.mkdir()
    (dashboard / "orca_dashboard.html").write_text("dashboard", encoding="ascii")
    store = AtomicSnapshotStore(tmp_path / "latest.json")
    stored = _snapshot_without_prediction()
    store.publish(stored)
    clock = [AS_OF]
    engine = OrcaMonitorEngine(
        prices_path=tmp_path / "unused.csv",
        snapshot_store=store,
        spectral_config=_spectral_config(),
        monitor_config=_monitor_config(),
        clock=lambda: clock[0],
    )
    handler = make_bound_handler(engine=engine, dashboard_directory=dashboard)
    assert isinstance(handler, type)
    assert issubclass(handler, OrcaRequestHandler)
    assert handler.__name__ == "BoundHandler"

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    try:
        connection.request("GET", "/api/snapshot")
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        assert response.status == 200
        assert payload["schema_version"] == ORCA_MONITOR_SCHEMA
        assert payload["age_seconds"] == pytest.approx(86_400)
        assert payload["served_at_utc"] == _iso(AS_OF)
        assert payload["snapshot_hash"] == compute_snapshot_hash(payload)
        assert response.getheader("X-Content-Type-Options") == "nosniff"
        assert response.getheader("Content-Security-Policy")

        connection.request("GET", "/healthz")
        response = connection.getresponse()
        health = json.loads(response.read().decode("utf-8"))
        assert response.status == 503
        assert health["status"] == "STARTING"
        assert health["reason"] == "NO_SUCCESSFUL_REFRESH"

        engine.last_refresh_utc = AS_OF
        connection.request("GET", "/healthz")
        response = connection.getresponse()
        health = json.loads(response.read().decode("utf-8"))
        assert response.status == 200
        assert health["status"] == "OK"

        clock[0] = AS_OF + timedelta(days=2)
        connection.request("GET", "/api/snapshot")
        response = connection.getresponse()
        stale = json.loads(response.read().decode("utf-8"))
        assert response.status == 200
        assert stale["age_seconds"] == pytest.approx(3 * 86_400)
        assert stale["freshness"] == "STALE"
        assert stale["regime"] == "UNAVAILABLE"
        assert stale["suggested_equity_exposure"] is None
        assert stale["exposure_status"] == "STALE_MARKET_DATA"
        assert stale["snapshot_hash"] == compute_snapshot_hash(stale)

        connection.request("GET", "/healthz")
        response = connection.getresponse()
        health = json.loads(response.read().decode("utf-8"))
        assert response.status == 503
        assert health["reason"] == "STALE"
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
    assert not worker.is_alive()


def test_snapshot_store_requires_schema_nonexecuting_and_hash(tmp_path) -> None:
    store = AtomicSnapshotStore(tmp_path / "latest.json")
    valid = _snapshot_without_prediction()

    missing_hash = dict(valid)
    missing_hash.pop("snapshot_hash")
    with pytest.raises(ValueError, match="SHA-256"):
        store.publish(missing_hash)

    wrong_schema = dict(valid, schema_version="orca-monitor-snapshot-v0")
    wrong_schema["snapshot_hash"] = compute_snapshot_hash(wrong_schema)
    with pytest.raises(ValueError, match="schema_version"):
        store.publish(wrong_schema)

    executing = dict(valid, executing=True)
    executing["snapshot_hash"] = compute_snapshot_hash(executing)
    with pytest.raises(ValueError, match="executing=false"):
        store.publish(executing)

    store.publish(valid)
    tampered = dict(valid, freshness="STALE")
    store.path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        store.read()

    executing["snapshot_hash"] = compute_snapshot_hash(executing)
    store.path.write_text(json.dumps(executing), encoding="utf-8")
    with pytest.raises(ValueError, match="executing=false"):
        store.read()


@pytest.mark.parametrize(
    ("timestamp", "message"),
    [
        ("2026-08-01", "date-only"),
        ("2026-08-01T22:00:00", "timezone"),
        ("2026-08-01T00:00:00Z", "implicit-midnight"),
    ],
)
def test_price_loader_rejects_noncausal_d1_labels(
    tmp_path,
    timestamp: str,
    message: str,
) -> None:
    path = tmp_path / "prices.csv"
    path.write_text(
        f"timestamp,A1,A2\n{timestamp},100,101\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        load_price_panel(path)


def test_price_loader_accepts_explicit_nonmidnight_availability(tmp_path) -> None:
    path = tmp_path / "prices.csv"
    path.write_text(
        "timestamp,A1,A2\n"
        "2026-08-01T22:00:00Z,100,101\n"
        "2026-08-02T22:00:00+00:00,101,102\n",
        encoding="utf-8",
    )
    panel = load_price_panel(path)
    assert str(panel.index.tz) == "UTC"
    assert list(panel.index.hour) == [22, 22]


def test_regime_requires_fresh_full_universe_and_allowed_model_status() -> None:
    base = _snapshot_without_prediction()
    partial = build_monitor_snapshot(
        _prices(),
        generated_at_utc=AS_OF,
        spectral_config=_spectral_config(),
        monitor_config=_monitor_config(),
        prediction=_prediction(base, status="OOS_PARTIAL"),
    )
    assert partial["freshness"] == "FRESH"
    assert partial["regime"] == "UNAVAILABLE"
    assert partial["exposure_status"] == "MODEL_STATUS_NOT_ALLOWED"

    degraded_base = build_monitor_snapshot(
        _prices(assets=5),
        generated_at_utc=AS_OF,
        spectral_config=_spectral_config(),
        monitor_config=_monitor_config(),
    )
    degraded = build_monitor_snapshot(
        _prices(assets=5),
        generated_at_utc=AS_OF,
        spectral_config=_spectral_config(),
        monitor_config=_monitor_config(),
        prediction=_prediction(degraded_base),
    )
    assert degraded["freshness"] == "DEGRADED_UNIVERSE"
    assert degraded["full_universe"] is False
    assert degraded["regime"] == "UNAVAILABLE"
    assert degraded["exposure_status"] == "FULL_UNIVERSE_REQUIRED"

    overfull_base = build_monitor_snapshot(
        _prices(assets=7),
        generated_at_utc=AS_OF,
        spectral_config=_spectral_config(),
        monitor_config=_monitor_config(),
    )
    overfull = build_monitor_snapshot(
        _prices(assets=7),
        generated_at_utc=AS_OF,
        spectral_config=_spectral_config(),
        monitor_config=_monitor_config(),
        prediction=_prediction(overfull_base),
    )
    assert overfull["freshness"] == "DEGRADED_UNIVERSE"
    assert overfull["regime_eligible"] is False

    stale_config = OrcaMonitorConfig(
        refresh_seconds=10,
        stale_after_seconds=60,
        expected_assets=6,
        minimum_assets=5,
    )
    stale_base = build_monitor_snapshot(
        _prices(),
        generated_at_utc=AS_OF,
        spectral_config=_spectral_config(),
        monitor_config=stale_config,
    )
    stale = build_monitor_snapshot(
        _prices(),
        generated_at_utc=AS_OF,
        spectral_config=_spectral_config(),
        monitor_config=stale_config,
        prediction=_prediction(stale_base),
    )
    assert stale["freshness"] == "STALE"
    assert stale["regime"] == "UNAVAILABLE"
    assert stale["exposure_status"] == "STALE_MARKET_DATA"


def test_monitor_imports_no_execution_modules_and_dashboard_is_ascii_safe() -> None:
    repository = Path(__file__).resolve().parents[1]
    monitor_path = repository / "monitoring" / "orca_monitor.py"
    source = monitor_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint(
        {"MetaTrader5", "main", "config", "executor", "executors"}
    )

    dashboard_source = (
        (repository / "monitoring" / "orca_dashboard.html").read_bytes().decode("ascii")
    )
    assert "innerHTML" not in dashboard_source
    assert "node_coordinates_3d" in dashboard_source
    assert "signed_edges" in dashboard_source
    assert "bcd_auc" in dashboard_source
    assert "network.coordinates" not in dashboard_source
    assert "network.edges" not in dashboard_source
    assert "Loading..." in dashboard_source
    assert "lambda ${index + 1}" in dashboard_source


def test_expected_symbols_pins_the_exact_ordered_universe() -> None:
    prices = _prices()
    pinned = OrcaMonitorConfig(
        refresh_seconds=10,
        stale_after_seconds=200_000,
        expected_assets=6,
        minimum_assets=5,
        expected_symbols=tuple(prices.columns),
    )

    snapshot = build_monitor_snapshot(
        prices,
        generated_at_utc=AS_OF,
        spectral_config=_spectral_config(),
        monitor_config=pinned,
    )
    assert snapshot["symbols"] == list(prices.columns)
    assert snapshot["full_universe"] is True

    # A panel with the right asset count but a different universe is rejected
    # instead of being served as a full universe.
    renamed = prices.rename(columns={prices.columns[0]: "SWAPPED"})
    with pytest.raises(ValueError, match="universe/order differs"):
        build_monitor_snapshot(
            renamed,
            generated_at_utc=AS_OF,
            spectral_config=_spectral_config(),
            monitor_config=pinned,
        )

    reordered = prices.loc[:, list(reversed(prices.columns))]
    with pytest.raises(ValueError, match="universe/order differs"):
        build_monitor_snapshot(
            reordered,
            generated_at_utc=AS_OF,
            spectral_config=_spectral_config(),
            monitor_config=pinned,
        )


def test_expected_symbols_are_revalidated_when_a_snapshot_is_served() -> None:
    prices = _prices()
    pinned = OrcaMonitorConfig(
        refresh_seconds=10,
        stale_after_seconds=200_000,
        expected_assets=6,
        minimum_assets=5,
        expected_symbols=tuple(prices.columns),
    )
    snapshot = build_monitor_snapshot(
        prices,
        generated_at_utc=AS_OF,
        spectral_config=_spectral_config(),
        monitor_config=pinned,
    )

    served = materialize_current_snapshot(
        snapshot,
        current_time_utc=AS_OF,
        monitor_config=pinned,
    )
    assert served["freshness"] == "FRESH"

    stale_universe = dict(snapshot)
    stale_universe["symbols"] = list(reversed(snapshot["symbols"]))
    with pytest.raises(ValueError, match="universe/order differs"):
        materialize_current_snapshot(
            stale_universe,
            current_time_utc=AS_OF,
            monitor_config=pinned,
        )


def test_expected_symbols_must_agree_with_expected_assets() -> None:
    with pytest.raises(ValueError, match="must match expected_assets"):
        OrcaMonitorConfig(expected_assets=6, expected_symbols=("A", "B"))
    with pytest.raises(ValueError, match="must be unique"):
        OrcaMonitorConfig(
            expected_assets=2,
            minimum_assets=2,
            expected_symbols=("A", "A"),
        )
    with pytest.raises(ValueError, match="cannot contain blank"):
        OrcaMonitorConfig(
            expected_assets=2,
            minimum_assets=2,
            expected_symbols=("A", " "),
        )


def test_blank_expected_symbols_preserve_the_legacy_count_only_contract() -> None:
    prices = _prices()
    snapshot = build_monitor_snapshot(
        prices,
        generated_at_utc=AS_OF,
        spectral_config=_spectral_config(),
        monitor_config=_monitor_config(),
    )
    assert snapshot["full_universe"] is True
    renamed = prices.rename(columns={prices.columns[0]: "SWAPPED"})
    assert build_monitor_snapshot(
        renamed,
        generated_at_utc=AS_OF,
        spectral_config=_spectral_config(),
        monitor_config=_monitor_config(),
    )["asset_count"] == 6


def test_universe_profile_resolves_the_paired_spectral_contract() -> None:
    config, symbols, assets, minimum = resolve_universe_arguments(
        universe_profile="orca-fx-gold-4-v1",
        expected_symbols="",
        expected_assets=None,
    )
    assert symbols == ORCA_FX_GOLD_4_PROFILE.symbols
    assert assets == 4
    assert minimum == 4
    # The four-asset profile cannot silently inherit the 24-asset AR5 contract.
    assert config is not None
    assert config.absorption_ranks == (1, 2, 3)
    assert config.universe_profile is ORCA_FX_GOLD_4_PROFILE


def test_universe_profile_refuses_conflicting_explicit_arguments() -> None:
    with pytest.raises(ValueError, match="conflicts with profile"):
        resolve_universe_arguments(
            universe_profile="orca-fx-gold-4-v1",
            expected_symbols="EURUSD GBPUSD USDCAD XAUUSD",
            expected_assets=None,
        )
    with pytest.raises(ValueError, match="conflicts with profile"):
        resolve_universe_arguments(
            universe_profile="orca-fx-gold-4-v1",
            expected_symbols="",
            expected_assets=24,
        )
    with pytest.raises(OrcaValidationError, match="unknown ORCA universe profile"):
        resolve_universe_arguments(
            universe_profile="orca-fx-gold-5-v1",
            expected_symbols="",
            expected_assets=None,
        )


def test_explicit_symbols_without_a_profile_must_match_the_declared_count() -> None:
    config, symbols, assets, minimum = resolve_universe_arguments(
        universe_profile="",
        expected_symbols="EURUSD, GBPUSD, USDCAD, GOLD",
        expected_assets=4,
    )
    assert config is None
    assert symbols == ("EURUSD", "GBPUSD", "USDCAD", "GOLD")
    assert (assets, minimum) == (4, 4)

    with pytest.raises(ValueError, match="must match --expected-assets"):
        resolve_universe_arguments(
            universe_profile="",
            expected_symbols="EURUSD GBPUSD",
            expected_assets=4,
        )


def test_a_leftover_expected_assets_env_conflicts_with_a_profile(monkeypatch) -> None:
    monkeypatch.setenv("ORCA_EXPECTED_ASSETS", "24")
    monkeypatch.setenv("ORCA_UNIVERSE_PROFILE", "orca-fx-gold-4-v1")
    monkeypatch.setenv("ORCA_PRICES_PATH", "unused.parquet")

    args = _parser().parse_args([])
    assert args.expected_assets == 24
    with pytest.raises(ValueError, match="conflicts with profile"):
        resolve_universe_arguments(
            universe_profile=args.universe_profile,
            expected_symbols=args.expected_symbols,
            expected_assets=args.expected_assets,
        )

    # Removing the stale variable is what actually enables the profile.
    monkeypatch.delenv("ORCA_EXPECTED_ASSETS")
    args = _parser().parse_args([])
    assert args.expected_assets is None
    _, symbols, assets, _ = resolve_universe_arguments(
        universe_profile=args.universe_profile,
        expected_symbols=args.expected_symbols,
        expected_assets=args.expected_assets,
    )
    assert (symbols, assets) == (ORCA_FX_GOLD_4_PROFILE.symbols, 4)
