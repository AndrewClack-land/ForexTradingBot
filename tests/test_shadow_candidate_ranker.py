from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backtest.cost_model import CostProfile
from core.shadow_candidate_ranker import (
    PortfolioCorrelationProfile,
    build_shadow_rankings,
    correlation_penalty_r,
)


def _cost_profile() -> CostProfile:
    return CostProfile.from_mapping({
        "schema": "fx-cost-profile-v1",
        "profile_id": "costs",
        "account_currency": "USD",
        "measured_from": "test fixture",
        "created_at_utc": "2026-08-01T00:00:00Z",
        "rollover_timezone": "UTC",
        "rollover_hour": 0,
        "triple_swap_weekday": 2,
        "symbols": {
            "EURUSD": {
                "base_currency": "EUR",
                "quote_currency": "USD",
                "contract_size": 100000,
                "spread_price": 0.0001,
                "slippage_price_per_side": 0.00002,
                "commission_round_turn_per_lot": 7.0,
                "swap_long_per_lot_rollover": -2.0,
                "swap_short_per_lot_rollover": 1.0,
            },
        },
    })


def _correlation_profile(
    *,
    trained_end="2026-08-01T00:00:00Z",
) -> PortfolioCorrelationProfile:
    return PortfolioCorrelationProfile.from_mapping({
        "schema": "portfolio-correlation-profile-v1",
        "profile_id": "corr",
        "trained_start_utc": "2025-08-01T00:00:00Z",
        "trained_end_utc": trained_end,
        "return_timeframe": "1H",
        "method": "pearson_shrunk",
        "shrinkage": 0.2,
        "penalty_scale_r": 0.2,
        "max_penalty_r": 0.3,
        "correlations": {
            "EURUSD": {"EURUSD": 1.0, "GBPUSD": 0.8},
            "GBPUSD": {"EURUSD": 0.8, "GBPUSD": 1.0},
        },
    })


class _Scorer:
    model_id = "shadow-model"

    def score(self, signal, *, symbol):
        assert symbol == "EURUSD"
        kind = signal["trigger_kind"]
        if kind == "unseen":
            return {
                "shadow_score": 9.0,
                "shadow_trigger_in_vocab": False,
            }
        return {
            "shadow_score": {
                "long_trigger": 0.50,
                "short_trigger": 0.45,
            }[kind],
            "shadow_trigger_in_vocab": True,
        }


def _candidate(side, kind):
    return {
        "signal": "ENTER",
        "side": side,
        "trigger_kind": kind,
        "trigger_reason": kind,
        "entry_price": 1.10,
        "entry_min": 1.10,
        "entry_max": 1.10,
        "stop_price": 1.09 if side == "LONG" else 1.11,
        "tp_prices": [1.12, 1.13, 1.14],
    }


def test_directional_correlation_penalty_rewards_actual_hedge():
    profile = _correlation_profile()
    decision = datetime(2026, 8, 2, tzinfo=timezone.utc)

    long_penalty = correlation_penalty_r(
        profile,
        symbol="EURUSD",
        side="LONG",
        active_exposures={"GBPUSD": "LONG"},
        decision_time_utc=decision,
    )
    short_penalty = correlation_penalty_r(
        profile,
        symbol="EURUSD",
        side="SHORT",
        active_exposures={"GBPUSD": "LONG"},
        decision_time_utc=decision,
    )

    assert long_penalty == pytest.approx(0.16)
    assert short_penalty == 0.0


def test_net_and_correlation_rank_can_reverse_gross_choice():
    decision = datetime(2026, 8, 2, 12, tzinfo=timezone.utc)
    model_id, rows = build_shadow_rankings(
        symbol="EURUSD",
        candidates=[
            _candidate("LONG", "long_trigger"),
            _candidate("SHORT", "short_trigger"),
            _candidate("LONG", "unseen"),
        ],
        observed_at_utc=decision,
        decision_bar_close=decision,
        scorer=_Scorer(),
        cost_profile=_cost_profile(),
        correlation_profile=_correlation_profile(),
        active_exposures={"GBPUSD": "LONG"},
    )

    assert len(model_id) == 24
    selected = [row for row in rows if row["selected"]]
    assert len(selected) == 1
    assert selected[0]["expected_gross_r"] == pytest.approx(0.45)
    assert selected[0]["correlation_penalty_r"] == 0.0
    unseen = next(
        row
        for row in rows
        if row["ranking_status"] == "UNMODELED_TRIGGER"
    )
    assert unseen["rank_position"] is None
    assert unseen["selected"] is False


def test_future_dated_correlation_profile_is_not_causal():
    with pytest.raises(ValueError, match="future-dated"):
        correlation_penalty_r(
            _correlation_profile(
                trained_end="2026-08-03T00:00:00Z",
            ),
            symbol="EURUSD",
            side="LONG",
            active_exposures={"GBPUSD": "LONG"},
            decision_time_utc=datetime(
                2026,
                8,
                2,
                tzinfo=timezone.utc,
            ),
        )
