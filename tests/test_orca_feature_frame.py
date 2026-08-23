from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from backtest.orca_feature_frame import (
    DynamicsConfig,
    OrcaFeatureFrame,
    OrcaFeatureFrameError,
    TraditionalFeatureConfig,
    build_and_fit_orca_pipeline,
    build_orca_feature_frame,
    validate_feature_registry,
)
from backtest.orca_random_forest import (
    FoldWindow,
    RandomForestConfig,
    TargetConfig,
)
from core.orca_spectral import (
    OrcaSpectralConfig,
    build_orca_spectral_snapshots,
    spectral_feature_registry_hash,
    spectral_feature_row,
)


def _spectral_config() -> OrcaSpectralConfig:
    return OrcaSpectralConfig(
        rolling_windows=(6, 10),
        ewm_halflife=4.0,
        ewm_min_periods=6,
        absorption_ranks=(1, 3, 5),
        graph_thresholds=(0.3, 0.5, 0.7),
    )


def _prices(rows: int = 100) -> pd.DataFrame:
    index = pd.date_range(
        "2023-01-01 22:00:00",
        periods=rows,
        freq="D",
        tz="UTC",
    )
    base = np.resize(
        np.asarray(
            [
                0.002,
                0.025,
                0.022,
                -0.082,
                0.018,
                0.035,
                -0.074,
                0.028,
                0.016,
                -0.012,
                0.006,
                0.001,
            ],
            dtype=float,
        ),
        rows - 1,
    )
    time = np.arange(rows - 1, dtype=float)
    columns: dict[str, np.ndarray] = {}
    for number, symbol in enumerate(("SPY", "QQQ", "TLT", "GLD", "UUP", "HYG")):
        sign = -0.55 if symbol in {"TLT", "UUP"} else 1.0 - 0.06 * number
        idiosyncratic = 0.0035 * np.sin(time / (2.0 + number) + number)
        returns = np.clip(sign * base + idiosyncratic, -0.2, 0.2)
        columns[symbol] = np.concatenate(
            ([100.0 + number], (100.0 + number) * np.cumprod(1.0 + returns))
        )
    return pd.DataFrame(columns, index=index)


def _as_of(prices: pd.DataFrame):
    return prices.index[-1].to_pydatetime() + timedelta(minutes=1)


def _small_forest() -> RandomForestConfig:
    return RandomForestConfig(
        n_estimators=7,
        max_depth=3,
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=11,
        n_jobs=1,
    )


def test_closed_d1_as_of_contract_fails_closed():
    prices = _prices(40)
    with pytest.raises(OrcaFeatureFrameError, match="after as_of"):
        build_orca_feature_frame(
            prices,
            benchmark_symbol="SPY",
            as_of_utc=prices.index[-1].to_pydatetime() - timedelta(seconds=1),
            spectral_config=_spectral_config(),
        )
    with pytest.raises(OrcaFeatureFrameError, match="timezone-aware"):
        build_orca_feature_frame(
            prices,
            benchmark_symbol="SPY",
            as_of_utc=pd.Timestamp("2026-01-01").to_pydatetime(),
            spectral_config=_spectral_config(),
        )


def test_static_spectral_rows_match_core_contract_at_each_available_date():
    prices = _prices(55)
    config = _spectral_config()
    frame = build_orca_feature_frame(
        prices,
        benchmark_symbol="SPY",
        as_of_utc=_as_of(prices),
        spectral_config=config,
    )
    for position in (15, 30, 54):
        partial = prices.iloc[: position + 1]
        bundle = build_orca_spectral_snapshots(
            partial,
            as_of_utc=partial.index[-1].to_pydatetime(),
            config=config,
        )
        expected = spectral_feature_row(bundle)
        for name, value in expected.items():
            assert frame.features[name].iloc[position] == pytest.approx(value)
    assert frame.registry["core_spectral_feature_registry_hash"] == (
        spectral_feature_registry_hash(bundle)
    )


def test_registry_is_ordered_deterministic_and_does_not_claim_exact_206():
    prices = _prices(70)
    left = build_orca_feature_frame(
        prices,
        benchmark_symbol="SPY",
        as_of_utc=_as_of(prices),
        spectral_config=_spectral_config(),
    )
    right = build_orca_feature_frame(
        prices.copy(),
        benchmark_symbol="SPY",
        as_of_utc=_as_of(prices),
        spectral_config=_spectral_config(),
    )
    assert left.registry == right.registry
    assert left.registry["feature_names"] == list(left.features.columns)
    assert left.registry["feature_count"] == left.features.shape[1]
    assert left.registry["ordered_universe"] == list(prices.columns)
    assert len(left.registry["hash"]) == 64
    assert left.registry["paper_exact_206_replication"] is False
    validate_feature_registry(left)

    reordered = OrcaFeatureFrame(
        features=left.features.iloc[:, ::-1],
        registry=left.registry,
        benchmark_symbol=left.benchmark_symbol,
        as_of_utc=left.as_of_utc,
    )
    with pytest.raises(OrcaFeatureFrameError, match="order differs"):
        validate_feature_registry(reordered)

    reordered_prices = prices.loc[:, list(reversed(prices.columns))]
    reordered_universe = build_orca_feature_frame(
        reordered_prices,
        benchmark_symbol="SPY",
        as_of_utc=_as_of(reordered_prices),
        spectral_config=_spectral_config(),
    )
    assert reordered_universe.registry["ordered_universe"] == list(
        reversed(prices.columns)
    )
    assert reordered_universe.registry["hash"] != left.registry["hash"]


def test_traditional_contract_has_expected_causal_values():
    prices = _prices(90)
    frame = build_orca_feature_frame(
        prices,
        benchmark_symbol="SPY",
        as_of_utc=_as_of(prices),
        spectral_config=_spectral_config(),
    )
    features = frame.features
    close = prices["SPY"]
    returns = prices.pct_change(fill_method=None)
    assert features["traditional_v1__return_001d"].iloc[-1] == pytest.approx(
        close.pct_change(fill_method=None).iloc[-1]
    )
    assert features["traditional_v1__return_005d"].iloc[-1] == pytest.approx(
        close.iloc[-1] / close.iloc[-6] - 1.0
    )
    assert features["traditional_v1__price_to_sma_050d"].iloc[-1] == (
        pytest.approx(close.iloc[-1] / close.iloc[-50:].mean())
    )
    assert features["traditional_v1__cross_asset_dispersion_001d"].iloc[-1] == (
        pytest.approx(returns.iloc[-1].std(ddof=0))
    )
    traditional_names = frame.registry["traditional_feature_names"]
    assert "traditional_v1__realized_vol_020d" in traditional_names
    assert "traditional_v1__downside_vol_020d" in traditional_names
    assert "traditional_v1__drawdown_060d" in traditional_names
    assert "traditional_v1__rsi_014d" in traditional_names
    assert "traditional_v1__skew_020d" in traditional_names
    assert "traditional_v1__kurtosis_060d" in traditional_names
    assert "traditional_v1__vol_of_vol_005_020d" in traditional_names


def test_spectral_dynamics_match_manual_diff_roc_z_and_acceleration():
    prices = _prices(100)
    dynamics = DynamicsConfig(horizons=(5, 10, 20), percentile_window=30)
    frame = build_orca_feature_frame(
        prices,
        benchmark_symbol="SPY",
        as_of_utc=_as_of(prices),
        spectral_config=_spectral_config(),
        dynamics_config=dynamics,
    )
    base_name = "rolling_10d__ar_1"
    base = frame.features[base_name]
    position = len(base) - 1
    assert frame.features[f"{base_name}__dyn_diff_005d"].iloc[position] == (
        pytest.approx(base.iloc[position] - base.iloc[position - 5])
    )
    assert frame.features[f"{base_name}__dyn_roc_005d"].iloc[position] == (
        pytest.approx(base.iloc[position] / base.iloc[position - 5] - 1.0)
    )
    expected_acceleration = (
        base.iloc[position] - 2.0 * base.iloc[position - 5] + base.iloc[position - 10]
    )
    assert frame.features[f"{base_name}__dyn_acceleration_005d"].iloc[
        position
    ] == pytest.approx(expected_acceleration)
    window = base.iloc[position - 9 : position + 1]
    expected_z = (base.iloc[position] - window.mean()) / window.std(ddof=0)
    assert frame.features[f"{base_name}__dyn_z_005d"].iloc[position] == (
        pytest.approx(expected_z)
    )
    percentile = frame.features[f"{base_name}__dyn_percentile_030d"].iloc[position]
    assert 0.0 < percentile <= 1.0


def test_default_dynamics_contract_uses_trailing_252_percentile():
    assert DynamicsConfig().to_mapping()["percentile_window"] == 252
    prices = _prices(70)
    frame = build_orca_feature_frame(
        prices,
        benchmark_symbol="SPY",
        as_of_utc=_as_of(prices),
        spectral_config=_spectral_config(),
    )
    assert any(
        name.endswith("__dyn_percentile_252d")
        for name in frame.registry["dynamic_feature_names"]
    )


def test_warmup_rows_may_be_nan_but_frame_never_contains_infinity():
    prices = _prices(70)
    frame = build_orca_feature_frame(
        prices,
        benchmark_symbol="SPY",
        as_of_utc=_as_of(prices),
        spectral_config=_spectral_config(),
    )
    static_names = frame.registry["static_spectral_feature_names"]
    assert frame.features.loc[:, static_names].iloc[:10].isna().any().any()
    assert frame.features.loc[:, static_names].iloc[-1].notna().all()
    assert not np.isinf(frame.features.to_numpy(dtype=float)).any()


def test_future_price_mutation_cannot_change_past_feature_rows():
    prices = _prices(105)
    original = build_orca_feature_frame(
        prices,
        benchmark_symbol="SPY",
        as_of_utc=_as_of(prices),
        spectral_config=_spectral_config(),
        dynamics_config=DynamicsConfig(percentile_window=30),
    )
    cutoff = 72
    mutated = prices.copy()
    multipliers = np.asarray([1.4, 0.8, 1.2, 0.7, 1.1, 0.9])
    mutated.iloc[cutoff + 1 :] *= multipliers
    rebuilt = build_orca_feature_frame(
        mutated,
        benchmark_symbol="SPY",
        as_of_utc=_as_of(mutated),
        spectral_config=_spectral_config(),
        dynamics_config=DynamicsConfig(percentile_window=30),
    )
    pdt.assert_frame_equal(
        original.features.iloc[: cutoff + 1],
        rebuilt.features.iloc[: cutoff + 1],
        check_exact=True,
    )


def test_research_pipeline_calls_existing_rf_without_auto_promotion():
    prices = _prices(90)
    payload = build_and_fit_orca_pipeline(
        prices,
        benchmark_symbol="SPY",
        as_of_utc=_as_of(prices),
        spectral_config=_spectral_config(),
        dynamics_config=DynamicsConfig(percentile_window=30),
        fold_window=FoldWindow(
            train_size=35,
            gap_size=3,
            test_size=12,
            folds=2,
        ),
        target_config=TargetConfig(horizon=3),
        forest_config=_small_forest(),
    )
    assert payload["status"] == "research_only"
    assert payload["auto_promote"] is False
    assert payload["model_artifact"]["calibration"]["status"] == "uncalibrated"
    assert (
        payload["feature_registry"]["feature_names"]
        == (payload["model_artifact"]["feature_registry"]["names"])
    )


def test_config_registry_changes_when_traditional_contract_changes():
    prices = _prices(70)
    baseline = build_orca_feature_frame(
        prices,
        benchmark_symbol="SPY",
        as_of_utc=_as_of(prices),
        spectral_config=_spectral_config(),
    )
    changed = build_orca_feature_frame(
        prices,
        benchmark_symbol="SPY",
        as_of_utc=_as_of(prices),
        spectral_config=_spectral_config(),
        traditional_config=TraditionalFeatureConfig(sma_windows=(5, 10, 20, 50)),
    )
    assert baseline.registry["hash"] != changed.registry["hash"]
    assert list(baseline.features.columns) != list(changed.features.columns)
