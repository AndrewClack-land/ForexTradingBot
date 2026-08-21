from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os

import numpy as np
import pandas as pd
import pytest

from backtest.orca_feature_frame import (
    DynamicsConfig,
    ORCA_FEATURE_FRAME_SCHEMA,
    ORCA_PIPELINE_SCHEMA,
    OrcaFeatureFrame,
    build_orca_feature_frame,
)
from backtest.orca_random_forest import (
    ORCA_FEATURE_REGISTRY_SCHEMA,
    ORCA_RANDOM_FOREST_SCHEMA,
    compute_artifact_hash,
    predict_orca_frozen,
)
from core.orca_spectral import OrcaSpectralConfig
from monitoring.orca_monitor import (
    JsonPredictionProvider,
    OrcaMonitorConfig,
    build_monitor_snapshot,
    load_price_panel,
)
from monitoring.orca_prediction import (
    AtomicPredictionStore,
    OrcaPredictionConfig,
    OrcaPredictionEngine,
    OrcaPredictionError,
    causal_empirical_rank,
    compute_feature_values_hash,
)


KNOWN_AT = datetime(2026, 8, 20, 23, 0, tzinfo=timezone.utc)


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _prices(rows: int = 80, assets: int = 6) -> pd.DataFrame:
    index = pd.date_range(
        KNOWN_AT - timedelta(days=rows),
        periods=rows,
        freq="1D",
        tz="UTC",
    )
    time = np.arange(rows, dtype=float)
    values: dict[str, np.ndarray] = {}
    for number in range(assets):
        trend = 0.001 * time
        common = 0.018 * np.sin(time / 4.0)
        specific = 0.004 * np.cos(time / (2.0 + number) + number)
        values[f"A{number + 1}"] = (100.0 + number) * np.exp(
            trend + (1.0 - 0.06 * number) * common + specific
        )
    return pd.DataFrame(values, index=index)


def _spectral_config() -> OrcaSpectralConfig:
    return OrcaSpectralConfig(
        rolling_windows=(6, 10),
        ewm_halflife=4.0,
        ewm_min_periods=6,
        absorption_ranks=(1, 3, 5),
        graph_thresholds=(0.3, 0.5, 0.7),
    )


def _feature_registry(names: list[str]) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": ORCA_FEATURE_REGISTRY_SCHEMA,
        "names": names,
        "count": len(names),
        "order_is_contractual": True,
    }
    return {**body, "hash": _canonical_hash(body)}


def _frame_registry(names: list[str]) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": ORCA_FEATURE_FRAME_SCHEMA,
        "feature_names": names,
        "feature_count": len(names),
    }
    return {**body, "hash": _canonical_hash(body)}


def _tree(*, positive_left: float, positive_right: float) -> dict[str, object]:
    return {
        "children_left": [1, -1, -1],
        "children_right": [2, -1, -1],
        "feature": [0, -2, -2],
        "threshold": [0.0, -2.0, -2.0],
        "value": [
            [5.0, 5.0],
            [10.0 * (1.0 - positive_left), 10.0 * positive_left],
            [10.0 * (1.0 - positive_right), 10.0 * positive_right],
        ],
    }


def _artifact(
    names: list[str],
    *,
    market_as_of: pd.Timestamp,
) -> dict[str, object]:
    before = [market_as_of - pd.Timedelta(days=6 - number) for number in range(4)]
    oos = []
    rally_probabilities = (0.1, 0.4, 0.8, 0.9)
    crash_probabilities = (0.9, 0.6, 0.2, 0.1)
    for number, timestamp in enumerate(before):
        oos.append(
            {
                "position": number,
                "timestamp": timestamp.isoformat(),
                "label_known_at": (timestamp + pd.Timedelta(days=1)).isoformat(),
                "rally_label": number % 2,
                "crash_label": (number + 1) % 2,
                "rally_probability": rally_probabilities[number],
                "crash_probability": crash_probabilities[number],
            }
        )
    # Equal-as-of data is deliberately present in the immutable research
    # artifact.  It must not enter either the current rank or validation.
    oos.append(
        {
            "position": 4,
            "timestamp": market_as_of.isoformat(),
            "label_known_at": (market_as_of + pd.Timedelta(days=1)).isoformat(),
            "rally_label": 1,
            "crash_label": 1,
            "rally_probability": 1.0,
            "crash_probability": 1.0,
        }
    )
    feature_count = len(names)
    zeroes = [0.0] * feature_count
    ones = [1.0] * feature_count
    artifact: dict[str, object] = {
        "schema": ORCA_RANDOM_FOREST_SCHEMA,
        "calibration": {
            "status": "uncalibrated",
            "method": None,
            "claim": "test fixture",
        },
        "feature_registry": _feature_registry(names),
        "folds": [
            {
                "index": 0,
                "lineage": {
                    "test_start": 0,
                    "test_end": 5,
                    "train_last_label_known_at": (
                        market_as_of - pd.Timedelta(days=10)
                    ).isoformat(),
                },
                "preprocessor": {"center": zeroes, "scale": ones},
                "rally_forest": {
                    "classes": [0, 1],
                    "trees": [_tree(positive_left=0.1, positive_right=0.9)],
                },
                "crash_forest": {
                    "classes": [0, 1],
                    "trees": [_tree(positive_left=0.8, positive_right=0.2)],
                },
                "oos": oos,
            }
        ],
    }
    artifact["artifact_hash"] = compute_artifact_hash(artifact)
    return artifact


def _fake_builder_registry() -> dict[str, object]:
    return _frame_registry(["signal"])


def _fake_builder(
    prices: pd.DataFrame,
    *,
    benchmark_symbol: str,
    as_of_utc: datetime,
    spectral_config: OrcaSpectralConfig,
) -> OrcaFeatureFrame:
    del spectral_config
    assert benchmark_symbol in prices.columns
    values = np.linspace(-1.0, 1.0, len(prices))
    features = pd.DataFrame({"signal": values}, index=prices.index)
    return OrcaFeatureFrame(
        features=features,
        registry=_fake_builder_registry(),
        benchmark_symbol=benchmark_symbol,
        as_of_utc=as_of_utc,
    )


def _write_prices(prices: pd.DataFrame, path) -> None:
    output = prices.copy()
    output.insert(0, "timestamp", output.index)
    output.to_csv(path, index=False)


def _write_pipeline(path, artifact, registry) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": ORCA_PIPELINE_SCHEMA,
                "status": "research_only",
                "auto_promote": False,
                "feature_registry": registry,
                "model_artifact": artifact,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def _engine(tmp_path) -> tuple[OrcaPredictionEngine, dict[str, object], pd.DataFrame]:
    prices = _prices()
    prices_path = tmp_path / "prices.csv"
    artifact_path = tmp_path / "promoted.json"
    output_path = tmp_path / "prediction.json"
    _write_prices(prices, prices_path)
    artifact = _artifact(["signal"], market_as_of=prices.index[-1])
    _write_pipeline(artifact_path, artifact, _fake_builder_registry())
    config = OrcaPredictionConfig(
        promoted_artifact_hash=str(artifact["artifact_hash"]),
        promoted_feature_registry_hash=str(_fake_builder_registry()["hash"]),
        stale_after_seconds=200_000,
        model_id="fixture-model",
    )
    engine = OrcaPredictionEngine(
        prices_path=prices_path,
        artifact_path=artifact_path,
        output_path=output_path,
        benchmark_symbol="A1",
        config=config,
        spectral_config=_spectral_config(),
        feature_builder=_fake_builder,
        now=lambda: KNOWN_AT,
    )
    return engine, artifact, prices


def test_causal_rank_is_tie_aware_and_excludes_equal_or_future_timestamps() -> None:
    cutoff = datetime(2026, 1, 10, tzinfo=timezone.utc)
    observations = [
        (cutoff - timedelta(days=2), 0.2),
        (cutoff - timedelta(days=1), 0.5),
        (cutoff, 0.99),
        (cutoff + timedelta(days=1), 0.0),
    ]
    rank, count = causal_empirical_rank(
        0.5,
        observations,
        as_of_market_utc=cutoff,
    )
    assert count == 2
    assert rank == pytest.approx(0.75)


def test_publisher_matches_provider_and_frozen_inference(tmp_path) -> None:
    engine, artifact, prices = _engine(tmp_path)
    payload = engine.refresh()

    expected = predict_orca_frozen(
        artifact,
        _fake_builder(
            prices,
            benchmark_symbol="A1",
            as_of_utc=KNOWN_AT,
            spectral_config=_spectral_config(),
        ).features.iloc[[-1]],
        fold_index=0,
    )
    assert payload["p_rally"] == pytest.approx(expected.iloc[0]["rally_probability"])
    assert payload["p_crash"] == pytest.approx(expected.iloc[0]["crash_probability"])
    assert payload["calibration_status"] == "UNCALIBRATED"
    assert payload["executing"] is False
    assert payload["rank_oos_counts"] == {"rally": 4, "crash": 4}
    assert payload["validation"]["observation_count"] == 4
    assert payload["validation"]["status"] == "DEFINED"

    base = build_monitor_snapshot(
        load_price_panel(engine.prices_path),
        generated_at_utc=KNOWN_AT,
        spectral_config=_spectral_config(),
        monitor_config=OrcaMonitorConfig(
            refresh_seconds=60,
            stale_after_seconds=200_000,
            expected_assets=6,
            minimum_assets=5,
        ),
    )
    provider = JsonPredictionProvider(tmp_path / "prediction.json")
    read = provider.read_prediction(
        as_of_market_utc=pd.Timestamp(base["as_of_market_utc"]).to_pydatetime(),
        known_at_utc=KNOWN_AT,
        feature_registry_hash=str(base["feature_registry_hash"]),
        feature_values_hash=str(base["feature_values_hash"]),
    )
    assert read is not None
    assert read["prediction_hash"] == payload["prediction_hash"]
    assert (
        compute_feature_values_hash(payload["feature_values"])
        == payload["feature_values_hash"]
    )


def test_pipeline_registry_artifact_pin_and_tamper_fail_closed(tmp_path) -> None:
    engine, artifact, _ = _engine(tmp_path)
    first = engine.refresh()
    last_good = engine.output.path.read_bytes()

    wrong_registry = OrcaPredictionConfig(
        promoted_artifact_hash=str(artifact["artifact_hash"]),
        promoted_feature_registry_hash="a" * 64,
        stale_after_seconds=200_000,
    )
    registry_engine = OrcaPredictionEngine(
        prices_path=engine.prices_path,
        artifact_path=engine.artifact_path,
        output_path=engine.output.path,
        benchmark_symbol="A1",
        config=wrong_registry,
        spectral_config=_spectral_config(),
        feature_builder=_fake_builder,
        now=lambda: KNOWN_AT,
    )
    with pytest.raises(OrcaPredictionError, match="registry"):
        registry_engine.refresh()
    assert engine.output.path.read_bytes() == last_good

    tampered = copy.deepcopy(artifact)
    tampered["folds"][0]["rally_forest"]["trees"][0]["threshold"][0] = 99.0
    _write_pipeline(engine.artifact_path, tampered, _fake_builder_registry())
    with pytest.raises(OrcaPredictionError, match="artifact hash"):
        engine.refresh()
    assert engine.output.path.read_bytes() == last_good
    assert first["prediction_hash"] == json.loads(last_good)["prediction_hash"]


def test_atomic_replace_failure_preserves_last_good(tmp_path, monkeypatch) -> None:
    engine, _, _ = _engine(tmp_path)
    first = engine.refresh()
    store = AtomicPredictionStore(engine.output.path)
    changed = dict(first, status="OOS_PARTIAL")
    from monitoring.orca_monitor import compute_prediction_hash

    changed["prediction_hash"] = compute_prediction_hash(changed)
    original = engine.output.path.read_bytes()

    def fail_replace(source: object, target: object) -> None:
        del source, target
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        store.publish(changed)
    assert engine.output.path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_stale_and_future_model_are_rejected_before_publication(tmp_path) -> None:
    engine, artifact, prices = _engine(tmp_path)
    stale_engine = OrcaPredictionEngine(
        prices_path=engine.prices_path,
        artifact_path=engine.artifact_path,
        output_path=engine.output.path,
        benchmark_symbol="A1",
        config=OrcaPredictionConfig(
            promoted_artifact_hash=str(artifact["artifact_hash"]),
            promoted_feature_registry_hash=str(_fake_builder_registry()["hash"]),
            stale_after_seconds=1.0,
        ),
        spectral_config=_spectral_config(),
        feature_builder=_fake_builder,
        now=lambda: KNOWN_AT,
    )
    with pytest.raises(OrcaPredictionError, match="stale"):
        stale_engine.refresh()
    assert not engine.output.path.exists()

    future_artifact = copy.deepcopy(artifact)
    future_artifact["folds"][0]["lineage"]["train_last_label_known_at"] = (
        prices.index[-1] + pd.Timedelta(days=1)
    ).isoformat()
    future_artifact["artifact_hash"] = compute_artifact_hash(future_artifact)
    _write_pipeline(
        engine.artifact_path,
        future_artifact,
        _fake_builder_registry(),
    )
    future_engine = OrcaPredictionEngine(
        prices_path=engine.prices_path,
        artifact_path=engine.artifact_path,
        output_path=engine.output.path,
        benchmark_symbol="A1",
        config=OrcaPredictionConfig(
            promoted_artifact_hash=str(future_artifact["artifact_hash"]),
            promoted_feature_registry_hash=str(_fake_builder_registry()["hash"]),
            stale_after_seconds=200_000,
        ),
        spectral_config=_spectral_config(),
        feature_builder=_fake_builder,
        now=lambda: KNOWN_AT,
    )
    with pytest.raises(OrcaPredictionError, match="trained beyond"):
        future_engine.refresh()


def test_real_feature_builder_registry_and_monitor_hashes_are_compatible(
    tmp_path,
) -> None:
    prices = _prices(90)
    spectral = _spectral_config()

    def real_builder(
        source: pd.DataFrame,
        *,
        benchmark_symbol: str,
        as_of_utc: datetime,
        spectral_config: OrcaSpectralConfig,
    ) -> OrcaFeatureFrame:
        return build_orca_feature_frame(
            source,
            benchmark_symbol=benchmark_symbol,
            as_of_utc=as_of_utc,
            spectral_config=spectral_config,
            dynamics_config=DynamicsConfig(
                horizons=(2, 4),
                percentile_window=10,
            ),
        )

    frame = real_builder(
        prices,
        benchmark_symbol="A1",
        as_of_utc=KNOWN_AT,
        spectral_config=spectral,
    )
    assert not np.isinf(frame.features.iloc[-1].to_numpy(dtype=float)).any()
    artifact = _artifact(list(frame.features.columns), market_as_of=prices.index[-1])
    prices_path = tmp_path / "prices.csv"
    artifact_path = tmp_path / "promoted.json"
    output_path = tmp_path / "prediction.json"
    _write_prices(prices, prices_path)
    _write_pipeline(artifact_path, artifact, frame.registry)
    engine = OrcaPredictionEngine(
        prices_path=prices_path,
        artifact_path=artifact_path,
        output_path=output_path,
        benchmark_symbol="A1",
        config=OrcaPredictionConfig(
            promoted_artifact_hash=str(artifact["artifact_hash"]),
            promoted_feature_registry_hash=str(frame.registry["hash"]),
            stale_after_seconds=200_000,
        ),
        spectral_config=spectral,
        feature_builder=real_builder,
        now=lambda: KNOWN_AT,
    )
    payload = engine.refresh()
    snapshot = build_monitor_snapshot(
        load_price_panel(prices_path),
        generated_at_utc=KNOWN_AT + timedelta(seconds=1),
        spectral_config=spectral,
        monitor_config=OrcaMonitorConfig(
            refresh_seconds=60,
            stale_after_seconds=200_000,
            expected_assets=6,
            minimum_assets=5,
        ),
        prediction=payload,
    )
    assert snapshot["model"]["status"] == "OOS_VALIDATED"
    assert snapshot["model"]["model_artifact_hash"] == artifact["artifact_hash"]
