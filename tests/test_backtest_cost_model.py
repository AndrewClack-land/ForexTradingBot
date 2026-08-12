from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backtest.cost_model import (
    CostModelError,
    CostProfile,
    apply_costs_to_setups,
    estimate_cost_r,
    rollover_units,
)


def _profile(
    *,
    created_at_utc: str = "2026-08-11T00:00:00Z",
    measured_through_utc: str | None = None,
) -> CostProfile:
    payload = {
        "schema": "fx-cost-profile-v1",
        "profile_id": "fxpro-forward-2026w32",
        "account_currency": "USD",
        "measured_from": "forward FxPro ticks plus broker contract terms",
        "created_at_utc": created_at_utc,
        "rollover_timezone": "UTC",
        "rollover_hour": 0,
        "triple_swap_weekday": 2,
        "symbols": {
            "EURUSD": {
                "base_currency": "EUR",
                "quote_currency": "USD",
                "contract_size": 100000,
                "spread_price": 0.00010,
                "slippage_price_per_side": 0.00002,
                "commission_round_turn_per_lot": 7.0,
                "swap_long_per_lot_rollover": -2.0,
                "swap_short_per_lot_rollover": 1.0,
            },
            "USDCAD": {
                "base_currency": "USD",
                "quote_currency": "CAD",
                "contract_size": 100000,
                "spread_price": 0.00012,
                "slippage_price_per_side": 0.00002,
                "commission_round_turn_per_lot": 7.0,
                "swap_long_per_lot_rollover": -2.0,
                "swap_short_per_lot_rollover": -1.0,
            },
        },
    }
    if measured_through_utc is not None:
        payload["measured_through_utc"] = measured_through_utc
    return CostProfile.from_mapping(payload)


def test_friction_cost_is_expressed_in_planned_risk_units():
    estimate = estimate_cost_r(
        _profile(),
        symbol="EURUSD",
        side="LONG",
        entry_price=1.10,
        stop_price=1.09,
    )
    # spread .00010 + two-sided slippage .00004 + $7/lot=.00007
    assert estimate.total_cost_r == pytest.approx(0.021)
    assert estimate.swap_r == 0.0


def test_base_account_pair_uses_dynamic_quote_conversion():
    estimate = estimate_cost_r(
        _profile(),
        symbol="USDCAD",
        side="LONG",
        entry_price=1.35,
        stop_price=1.34,
    )
    # $7 / (100k CAD * 1/1.35 CAD->USD) = 0.0000945 CAD.
    assert estimate.commission_r == pytest.approx(0.00945)


def test_rollover_counts_weekdays_and_triple_wednesday():
    units = rollover_units(
        datetime(2026, 8, 11, 23, tzinfo=timezone.utc),
        datetime(2026, 8, 13, 1, tzinfo=timezone.utc),
        timezone_name="UTC",
        rollover_hour=0,
        triple_swap_weekday=2,
    )
    # Wednesday 00:00 counts 3 and Thursday 00:00 counts 1.
    assert units == 4


def test_realized_rows_preserve_gross_and_replace_metric_net(tmp_path):
    profile = _profile()
    rows = apply_costs_to_setups(
        [{
            "setup_id": "s1",
            "symbol": "EURUSD",
            "side": "LONG",
            "entry": 1.10,
            "stop": 1.09,
            "entry_time": "2026-08-11T10:00:00Z",
            "exit_time": "2026-08-11T12:00:00Z",
            "net_r": 0.5,
        }],
        profile,
    )
    assert rows[0]["gross_net_r"] == 0.5
    assert rows[0]["transaction_cost_r"] == pytest.approx(0.021)
    assert rows[0]["net_after_cost_r"] == pytest.approx(0.479)
    assert rows[0]["net_r"] == rows[0]["net_after_cost_r"]


def test_missing_conversion_or_provenance_fails_closed():
    payload = {
        "schema": "fx-cost-profile-v1",
        "profile_id": "bad",
        "account_currency": "USD",
        "measured_from": "",
        "created_at_utc": "2026-08-11T00:00:00Z",
        "rollover_timezone": "UTC",
        "symbols": {},
    }
    with pytest.raises(CostModelError):
        CostProfile.from_mapping(payload)


def test_audit_payload_round_trips_with_same_source_hash():
    profile = _profile()
    restored = CostProfile.from_mapping(profile.to_dict())

    assert restored.profile_sha256 == profile.profile_sha256
    assert restored.profile_id == profile.profile_id
    assert restored.measured_through_utc is None
    assert "measured_through_utc" not in restored.to_dict()


def test_structured_measurement_timestamp_round_trips_and_is_audited():
    profile = _profile(
        created_at_utc="2025-12-31T22:00:00Z",
        measured_through_utc="2025-12-31T21:00:00Z",
    )

    audit = profile.point_in_time_audit([
        "2026-01-01T00:00:00Z",
        "2026-06-01T00:00:00Z",
    ])
    restored = CostProfile.from_mapping(profile.to_dict())

    assert audit["status"] == "CAUSAL_FOR_ALL_FOLDS"
    assert audit["causal_for_all_fold_test_starts"] is True
    assert audit["issues"] == []
    assert audit["fold_test_start_count"] == 2
    assert restored.measured_through_utc == (
        profile.measured_through_utc
    )
    assert restored.profile_sha256 == profile.profile_sha256
