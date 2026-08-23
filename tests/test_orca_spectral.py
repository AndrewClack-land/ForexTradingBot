from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pandas as pd
import pytest

from core.orca_spectral import (
    BCD_AUC_SCHEMA,
    ORCA_FX_GOLD_4_PROFILE,
    ORCA_FX_GOLD_4_SPECTRAL_CONFIG,
    ORCA_SPECTRAL_SCHEMA,
    OrcaInsufficientDataError,
    OrcaSpectralConfig,
    OrcaValidationError,
    balanced_crisis_detection_auc,
    build_correlation_snapshots,
    build_orca_spectral_snapshots,
    resolve_universe_profile,
    simple_returns,
    spectral_config_for_profile,
    spectral_feature_registry_hash,
    spectral_feature_row,
    tie_aware_binary_auc,
    validate_d1_price_panel,
)


UTC = timezone.utc


def _config(**overrides: object) -> OrcaSpectralConfig:
    values: dict[str, object] = {
        "rolling_windows": (4, 6),
        "ewm_halflife": 2.0,
        "ewm_min_periods": 4,
        "absorption_ranks": (1, 3, 5),
        "graph_thresholds": (0.3, 0.5, 0.7),
    }
    values.update(overrides)
    return OrcaSpectralConfig(**values)


def _price_panel(
    *,
    symbols: tuple[str, ...] = ("A", "B", "C", "D", "E"),
) -> pd.DataFrame:
    common = np.array(
        [0.010, -0.020, 0.030, -0.010, 0.025, -0.015, 0.020, -0.005],
        dtype=float,
    )
    columns: list[np.ndarray] = []
    for index, _symbol in enumerate(symbols):
        sign = 1.0 if index < 3 else -1.0
        columns.append(sign * common)
    returns = np.column_stack(columns)
    values = np.vstack(
        [
            np.full(len(symbols), 100.0),
            100.0 * np.cumprod(1.0 + returns, axis=0),
        ]
    )
    dates = pd.date_range(
        "2026-01-01 22:00:00",
        periods=len(values),
        freq="D",
        tz="UTC",
    )
    return pd.DataFrame(values, index=dates, columns=symbols)


def _as_of(panel: pd.DataFrame) -> datetime:
    return panel.index[-1].to_pydatetime() + timedelta(minutes=1)


def test_config_is_versioned_validated_and_hash_stable() -> None:
    left = _config()
    right = _config()

    assert left.schema_version == ORCA_SPECTRAL_SCHEMA
    assert left.config_id == right.config_id
    assert len(left.config_id) == 64
    assert left.config_id != _config(ewm_halflife=3.0).config_id

    with pytest.raises(OrcaValidationError, match="schema_version"):
        _config(schema_version="orca-spectral-v0")
    with pytest.raises(OrcaValidationError, match="strictly increasing"):
        _config(rolling_windows=(6, 4))
    with pytest.raises(OrcaValidationError, match="strictly inside"):
        _config(graph_thresholds=(0.3, 1.0))


def test_validate_d1_panel_returns_utc_copy_and_enforces_as_of() -> None:
    panel = _price_panel()
    validated = validate_d1_price_panel(panel, as_of_utc=_as_of(panel))

    assert validated is not panel
    assert str(validated.index.tz) == "UTC"
    np.testing.assert_allclose(validated, panel)

    with pytest.raises(OrcaValidationError, match="after as_of"):
        validate_d1_price_panel(
            panel,
            as_of_utc=panel.index[-1].to_pydatetime() - timedelta(seconds=1),
        )
    with pytest.raises(OrcaValidationError, match="timezone-aware"):
        validate_d1_price_panel(
            panel,
            as_of_utc=datetime(2026, 1, 20),
        )


@pytest.mark.parametrize("defect", ["naive", "duplicate_day", "nan", "zero"])
def test_validate_d1_panel_fails_closed_on_bad_alignment(defect: str) -> None:
    panel = _price_panel()
    if defect == "naive":
        panel.index = panel.index.tz_localize(None)
    elif defect == "duplicate_day":
        index = list(panel.index)
        index[-1] = index[-2] + timedelta(hours=1)
        panel.index = pd.DatetimeIndex(index)
    elif defect == "nan":
        panel.iloc[2, 1] = np.nan
    else:
        panel.iloc[2, 1] = 0.0

    with pytest.raises(OrcaValidationError):
        validate_d1_price_panel(
            panel,
            as_of_utc=datetime(2026, 2, 1, tzinfo=UTC),
        )


def test_simple_returns_never_fills_and_recovers_price_changes() -> None:
    panel = _price_panel()
    returns = simple_returns(panel)

    assert len(returns) == len(panel) - 1
    assert returns.iloc[0, 0] == pytest.approx(0.010)
    assert returns.iloc[0, 3] == pytest.approx(-0.010)

    broken = panel.copy()
    broken.iloc[3, 0] = np.nan
    with pytest.raises(OrcaValidationError, match="finite"):
        simple_returns(broken)


def test_correlation_estimators_use_configured_windows_and_ewm_ess() -> None:
    panel = _price_panel()
    config = _config()
    snapshots = build_correlation_snapshots(
        panel,
        as_of_utc=_as_of(panel),
        config=config,
    )

    assert [item.estimator for item in snapshots] == [
        "rolling_4d",
        "rolling_6d",
        "ewm_hl2d",
    ]
    assert [item.observation_count for item in snapshots] == [4, 6, 8]
    assert snapshots[0].effective_sample_length == 4.0
    assert snapshots[1].effective_sample_length == 6.0

    age = np.arange(7, -1, -1, dtype=float)
    weights = np.exp2(-age / 2.0)
    expected_ess = weights.sum() ** 2 / np.square(weights).sum()
    assert snapshots[2].effective_sample_length == pytest.approx(expected_ess)

    for item in snapshots:
        np.testing.assert_allclose(item.correlation, item.correlation.T)
        np.testing.assert_allclose(np.diag(item.correlation), 1.0)
        assert not item.correlation.flags.writeable


def test_correlation_fails_closed_for_too_few_rows_or_constant_series() -> None:
    panel = _price_panel()
    with pytest.raises(OrcaInsufficientDataError, match="rolling_20d"):
        build_correlation_snapshots(
            panel,
            as_of_utc=_as_of(panel),
            config=_config(rolling_windows=(4, 20)),
        )

    constant = panel.copy()
    constant["E"] = 100.0
    with pytest.raises(OrcaValidationError, match="constant return"):
        build_correlation_snapshots(
            constant,
            as_of_utc=_as_of(constant),
            config=_config(),
        )


def test_rank_one_market_has_expected_spectrum_graph_and_signed_edges() -> None:
    panel = _price_panel()
    bundle = build_orca_spectral_snapshots(
        panel,
        as_of_utc=_as_of(panel),
        config=_config(),
    )
    snapshot = bundle.snapshot("rolling_6d")

    np.testing.assert_allclose(snapshot.eigenvalues, [5.0, 0.0, 0.0, 0.0, 0.0])
    assert snapshot.absorption_ratios[1] == pytest.approx(1.0)
    assert snapshot.absorption_ratios[3] == pytest.approx(1.0)
    assert snapshot.absorption_ratios[5] == pytest.approx(1.0)
    assert snapshot.eigenvalue_entropy == pytest.approx(0.0)
    assert snapshot.effective_rank == pytest.approx(1.0)
    assert snapshot.lambda1_lambda2 is None
    assert snapshot.condition_number is None
    assert snapshot.dominant_eigenvector_hhi == pytest.approx(1.0 / 5.0)
    assert snapshot.dominant_eigenvector_participation == pytest.approx(5.0)

    graph = snapshot.graph_metrics[0.7]
    assert graph.edge_count == 10
    assert graph.edge_density == pytest.approx(1.0)
    assert graph.mean_degree == pytest.approx(4.0)
    assert graph.degree_std == pytest.approx(0.0)
    assert graph.max_degree == 4
    assert graph.isolated_nodes == 0
    assert graph.degree_centralization == pytest.approx(0.0)
    assert graph.global_clustering == pytest.approx(1.0)

    edge_lookup = {
        (edge.source, edge.target): edge for edge in snapshot.signed_edges[0.7]
    }
    assert edge_lookup[("A", "B")].sign == 1
    assert edge_lookup[("A", "D")].sign == -1
    assert edge_lookup[("A", "D")].absolute_correlation == pytest.approx(1.0)


def test_marchenko_pastur_uses_effective_sample_length() -> None:
    panel = _price_panel()
    bundle = build_orca_spectral_snapshots(
        panel,
        as_of_utc=_as_of(panel),
        config=_config(),
    )
    rolling = bundle.snapshot("rolling_6d")

    expected_q = 5.0 / 6.0
    assert rolling.marchenko_pastur_q == pytest.approx(expected_q)
    assert rolling.marchenko_pastur_lower == pytest.approx(
        (1.0 - math.sqrt(expected_q)) ** 2
    )
    assert rolling.marchenko_pastur_upper == pytest.approx(
        (1.0 + math.sqrt(expected_q)) ** 2
    )
    assert rolling.marchenko_pastur_outlier_count == 5
    assert rolling.marchenko_pastur_upper_outlier_count == 1
    assert rolling.marchenko_pastur_lambda1_excess == pytest.approx(
        5.0 - rolling.marchenko_pastur_upper
    )

    ewm = bundle.snapshot("ewm_hl2d")
    assert ewm.marchenko_pastur_q == pytest.approx(5.0 / ewm.effective_sample_length)


def test_ar5_fails_closed_for_four_asset_universe() -> None:
    panel = _price_panel(symbols=("A", "B", "C", "D"))
    with pytest.raises(OrcaInsufficientDataError, match="AR5"):
        build_orca_spectral_snapshots(
            panel,
            as_of_utc=_as_of(panel),
            config=_config(),
        )


def test_three_dimensional_coordinates_are_deterministic_and_non_dominant() -> None:
    panel = _price_panel()
    first = build_orca_spectral_snapshots(
        panel,
        as_of_utc=_as_of(panel),
        config=_config(),
    ).snapshot("rolling_6d")
    second = build_orca_spectral_snapshots(
        panel,
        as_of_utc=_as_of(panel),
        config=_config(),
    ).snapshot("rolling_6d")

    assert dict(first.node_coordinates_3d) == dict(second.node_coordinates_3d)
    assert tuple(first.node_coordinates_3d) == ("A", "B", "C", "D", "E")
    # A rank-one market has no non-dominant eigenvalue mass.
    assert all(
        coordinates == pytest.approx((0.0, 0.0, 0.0))
        for coordinates in first.node_coordinates_3d.values()
    )


def test_bundle_arrays_are_read_only() -> None:
    panel = _price_panel()
    snapshot = build_orca_spectral_snapshots(
        panel,
        as_of_utc=_as_of(panel),
        config=_config(),
    ).snapshots[0]

    with pytest.raises(ValueError):
        snapshot.eigenvalues[0] = 0.0
    with pytest.raises(ValueError):
        snapshot.eigenvectors[0, 0] = 0.0
    with pytest.raises(TypeError):
        snapshot.absorption_ratios[1] = 0.0  # type: ignore[index]


def test_flat_feature_row_has_stable_registry_and_only_finite_values() -> None:
    panel = _price_panel()
    first_bundle = build_orca_spectral_snapshots(
        panel,
        as_of_utc=_as_of(panel),
        config=_config(),
    )
    changed = panel.copy()
    changed.iloc[-1, 0] *= 1.001
    second_bundle = build_orca_spectral_snapshots(
        changed,
        as_of_utc=_as_of(changed),
        config=_config(),
    )

    row = spectral_feature_row(first_bundle)
    assert row
    assert all(math.isfinite(value) for value in row.values())
    assert list(row) == list(spectral_feature_row(first_bundle))
    assert "rolling_6d__ar_5" in row
    assert row["rolling_6d__lambda1_lambda2_missing"] == 1.0
    assert row["rolling_6d__condition_number_missing"] == 1.0
    assert spectral_feature_registry_hash(first_bundle) == (
        spectral_feature_registry_hash(second_bundle)
    )


def test_tie_aware_auc_matches_mann_whitney_definition() -> None:
    assert tie_aware_binary_auc([0, 1], [0.0, 1.0]) == pytest.approx(1.0)
    assert tie_aware_binary_auc([0, 1], [1.0, 0.0]) == pytest.approx(0.0)
    assert tie_aware_binary_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert tie_aware_binary_auc(
        [0, 1, 0, 1],
        [0.1, 0.4, 0.4, 0.8],
    ) == pytest.approx(0.875)


def test_bcd_auc_is_exact_geometric_mean_of_two_auc_values() -> None:
    result = balanced_crisis_detection_auc(
        [0, 1],
        [0.0, 1.0],
        [0, 1],
        [0.5, 0.5],
    )

    assert result.schema_version == BCD_AUC_SCHEMA
    assert result.status == "DEFINED"
    assert result.rally_auc == pytest.approx(1.0)
    assert result.crash_auc == pytest.approx(0.5)
    assert result.bcd_auc == pytest.approx(math.sqrt(0.5))
    assert result.observation_count == 2
    assert result.reason is None


def test_bcd_auc_is_undefined_when_either_task_has_one_class() -> None:
    result = balanced_crisis_detection_auc(
        [0, 1, 0],
        [0.1, 0.9, 0.2],
        [0, 0, 0],
        [0.1, 0.2, 0.3],
    )

    assert result.status == "UNDEFINED"
    assert result.bcd_auc is None
    assert result.rally_auc == pytest.approx(1.0)
    assert result.crash_auc is None
    assert result.reason == "missing_class:crash"
    assert result.crash_positive_count == 0
    assert result.crash_negative_count == 3


@pytest.mark.parametrize(
    ("labels", "scores", "match"),
    [
        ([0, 2], [0.1, 0.2], "only 0 and 1"),
        ([0, 1], [0.1, np.nan], "finite"),
        ([0, 1], [0.1], "equal non-zero length"),
    ],
)
def test_auc_rejects_invalid_inputs(
    labels: list[float],
    scores: list[float],
    match: str,
) -> None:
    with pytest.raises(OrcaValidationError, match=match):
        tie_aware_binary_auc(labels, scores)


def test_bcd_auc_requires_one_date_aligned_oos_population() -> None:
    with pytest.raises(OrcaValidationError, match="date-aligned"):
        balanced_crisis_detection_auc(
            [0, 1],
            [0.1, 0.9],
            [0, 1, 0],
            [0.1, 0.9, 0.2],
        )


def test_universe_profile_registry_resolves_and_fails_closed() -> None:
    profile = resolve_universe_profile("orca-fx-gold-4-v1")
    assert profile is ORCA_FX_GOLD_4_PROFILE
    assert profile.symbols == ("EURUSD", "GBPUSD", "USDCAD", "GOLD")
    assert profile.expected_assets == 4
    assert profile.minimum_assets == 4

    config = spectral_config_for_profile("orca-fx-gold-4-v1")
    assert config is ORCA_FX_GOLD_4_SPECTRAL_CONFIG
    # AR5 is undefined for four assets, so the profile carries its own ranks.
    assert config.absorption_ranks == (1, 2, 3)

    for unknown in ("", "   ", "orca-fx-gold-4", "ORCA-FX-GOLD-4-V1"):
        with pytest.raises(OrcaValidationError, match="unknown ORCA universe profile"):
            resolve_universe_profile(unknown)


def test_a_profile_bound_config_pins_the_exact_ordered_universe() -> None:
    panel = _price_panel(symbols=ORCA_FX_GOLD_4_PROFILE.symbols)
    config = _config(
        absorption_ranks=ORCA_FX_GOLD_4_PROFILE.absorption_ranks,
        universe_profile=ORCA_FX_GOLD_4_PROFILE,
    )

    bundle = build_orca_spectral_snapshots(panel, as_of_utc=_as_of(panel), config=config)
    assert bundle.symbols == ORCA_FX_GOLD_4_PROFILE.symbols
    assert bundle.profile_id == "orca-fx-gold-4-v1"
    assert bundle.profile_contract_hash == ORCA_FX_GOLD_4_PROFILE.contract_hash

    renamed = panel.rename(columns={"USDCAD": "USDCHF"})
    with pytest.raises(OrcaValidationError, match="requires exact ordered"):
        build_orca_spectral_snapshots(
            renamed,
            as_of_utc=_as_of(renamed),
            config=config,
        )


def test_a_profile_bound_registry_hash_covers_the_universe_not_only_names() -> None:
    panel = _price_panel(symbols=ORCA_FX_GOLD_4_PROFILE.symbols)
    as_of = _as_of(panel)

    unpinned = build_orca_spectral_snapshots(
        panel,
        as_of_utc=as_of,
        config=_config(absorption_ranks=(1, 2, 3)),
    )
    pinned = build_orca_spectral_snapshots(
        panel,
        as_of_utc=as_of,
        config=_config(
            absorption_ranks=(1, 2, 3),
            universe_profile=ORCA_FX_GOLD_4_PROFILE,
        ),
    )

    assert spectral_feature_row(unpinned) == spectral_feature_row(pinned)
    # Identical feature values, but only the pinned bundle commits to which
    # four assets produced them, so the two registries must not collide.
    assert spectral_feature_registry_hash(unpinned) != spectral_feature_registry_hash(
        pinned
    )
    assert unpinned.profile_id is None
    assert unpinned.profile_contract_hash is None
