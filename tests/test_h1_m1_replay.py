from __future__ import annotations

import copy

import pandas as pd
import pytest

from backtest.cost_model import CostProfile
from backtest.h1_m1_replay import (
    H1M1FoldWindow,
    H1M1ReplayConfig,
    run_h1_m1_paired_replay,
)
from backtest.h1_m1_trigger import H1Context


def _frame(rows):
    return pd.DataFrame(
        [
            {
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
            }
            for row in rows
        ],
        index=pd.DatetimeIndex(
            [row[0] for row in rows],
            name="timestamp",
        ),
    )


def _context(
    *,
    trigger_kind="rejection_block_1h",
    expiry="2026-01-01T00:05:00Z",
):
    return H1Context.from_signal(
        symbol="EURUSD",
        signal={
            "signal": "ENTER",
            "side": "LONG",
            "trigger_kind": trigger_kind,
            "trigger_event_id": "parent-event-1",
            "arbitration_group_id": "EURUSD@2026-01-01T00:01:00Z",
            "entry_price": 1.0995,
            "entry_min": 1.0990,
            "entry_max": 1.1000,
            "stop_price": 1.0950,
            "tp_prices": [1.1050, 1.1100, 1.1150],
        },
        decision_time="2026-01-01T00:01:00Z",
        expires_at=expiry,
    )


def _completed_reclaim_frame():
    return _frame([
        ("2026-01-01T00:00:00Z", 1.1010, 1.1020, 1.1005, 1.1010),
        ("2026-01-01T00:01:00Z", 1.0998, 1.1020, 1.0995, 1.1015),
        ("2026-01-01T00:02:00Z", 1.0998, 1.1004, 1.0996, 1.1001),
        ("2026-01-01T00:03:00Z", 1.1005, 1.1060, 1.1001, 1.1055),
        ("2026-01-01T00:04:00Z", 1.1055, 1.1160, 1.1050, 1.1155),
        ("2026-01-01T00:05:00Z", 1.1155, 1.1160, 1.1150, 1.1155),
        ("2026-01-01T00:06:00Z", 1.1155, 1.1160, 1.1150, 1.1155),
    ])


def _config():
    return H1M1ReplayConfig(outcome_horizon=pd.Timedelta(minutes=5))


def _cost_profile():
    return CostProfile.from_mapping({
        "schema": "fx-cost-profile-v1",
        "profile_id": "paired-test-costs",
        "account_currency": "USD",
        "measured_from": "fixed test scenario",
        "created_at_utc": "2025-12-31T00:00:00Z",
        "measured_through_utc": "2025-12-30T00:00:00Z",
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
        },
    })


def test_paired_arm_uses_strict_post_trigger_open_and_audits_parent():
    result = run_h1_m1_paired_replay(
        contexts=[_context()],
        m1_by_symbol={"EURUSD": _completed_reclaim_frame()},
        config=_config(),
    )

    row = result.rows[0]
    assert row["baseline_entry_time"] == "2026-01-01T00:02:00+00:00"
    assert row["m1_reclaim_trigger_time"] == (
        "2026-01-01T00:02:00+00:00"
    )
    assert row["m1_reclaim_entry_time"] == (
        "2026-01-01T00:03:00+00:00"
    )
    assert pd.Timestamp(row["m1_reclaim_entry_time"]) > pd.Timestamp(
        row["m1_reclaim_trigger_time"]
    )
    assert row["m1_reclaim_entry_rule"] == (
        "first_m1_open_strictly_after_trigger_time"
    )
    assert row["parent_arbitration_group_id"].startswith("EURUSD@")
    assert row["parent_trigger_event_id"] == "parent-event-1"
    assert row["context_expires_at"] == "2026-01-01T00:05:00+00:00"
    assert row["pair_status"] == "BOTH_ENTERED"
    assert row["policy_complete"] is True
    assert row["common_fill"] is True
    assert result.summary["complete_pairs"] == 1
    assert result.summary["common_fill_pairs"] == 1


def test_intervening_bar_invalidates_before_strict_m1_entry():
    frame = _completed_reclaim_frame()
    frame.loc[pd.Timestamp("2026-01-01T00:02:00Z"), "low"] = 1.0940

    result = run_h1_m1_paired_replay(
        contexts=[_context()],
        m1_by_symbol={"EURUSD": frame},
        config=_config(),
    )

    row = result.rows[0]
    assert row["baseline_label_status"] == "CLOSED"
    assert row["baseline_policy_r"] == pytest.approx(-1.0)
    assert row["m1_reclaim_trigger_status"] == "INVALIDATED_STOP"
    assert row["m1_reclaim_disposition"] == "NO_FILL"
    assert row["m1_reclaim_invalidated_at"] == (
        "2026-01-01T00:03:00+00:00"
    )
    assert row["m1_reclaim_policy_r"] == 0.0
    assert row["pair_status"] == "BASELINE_ONLY"
    assert row["policy_delta_r"] == pytest.approx(1.0)


def test_bars_at_or_after_common_cutoff_cannot_change_result():
    frame = _completed_reclaim_frame()
    mutated = frame.copy()
    mutated.loc[
        mutated.index >= pd.Timestamp("2026-01-01T00:06:00Z"),
        ["open", "high", "low", "close"],
    ] = [1.2000, 1.2100, 1.1900, 1.2050]
    mutated.loc[pd.Timestamp("2026-01-01T00:07:00Z")] = {
        "open": 1.3000,
        "high": 1.3100,
        "low": 1.2900,
        "close": 1.3050,
    }

    original = run_h1_m1_paired_replay(
        contexts=[_context()],
        m1_by_symbol={"EURUSD": frame},
        config=_config(),
    )
    changed = run_h1_m1_paired_replay(
        contexts=[_context()],
        m1_by_symbol={"EURUSD": mutated},
        config=_config(),
    )

    assert changed.as_dict() == original.as_dict()


def test_invalid_or_duplicate_rows_after_cutoff_cannot_change_result():
    frame = _completed_reclaim_frame()
    future = _frame([
        ("2026-01-01T00:06:00Z", 1.0, 0.5, 2.0, 1.0),
        ("2026-01-01T00:07:00Z", 1.0, 0.5, 2.0, 1.0),
        ("2026-01-01T00:07:00Z", 1.0, 0.4, 3.0, 1.0),
    ])
    mutated = pd.concat([
        frame.loc[frame.index < pd.Timestamp("2026-01-01T00:06:00Z")],
        future,
    ])

    original = run_h1_m1_paired_replay(
        contexts=[_context()],
        m1_by_symbol={"EURUSD": frame},
        config=_config(),
    )
    changed = run_h1_m1_paired_replay(
        contexts=[_context()],
        m1_by_symbol={"EURUSD": mutated},
        config=_config(),
    )

    assert changed.as_dict() == original.as_dict()


def test_truncated_entry_window_is_censored_not_zero_r():
    frame = _frame([
        ("2026-01-01T00:00:00Z", 1.1010, 1.1020, 1.1005, 1.1010),
        ("2026-01-01T00:01:00Z", 1.1010, 1.1020, 1.1005, 1.1010),
        ("2026-01-01T00:02:00Z", 1.1010, 1.1020, 1.1005, 1.1010),
    ])

    result = run_h1_m1_paired_replay(
        contexts=[
            _context(expiry="2026-01-01T00:10:00Z"),
        ],
        m1_by_symbol={"EURUSD": frame},
        config=H1M1ReplayConfig(
            outcome_horizon=pd.Timedelta(minutes=20)
        ),
    )

    row = result.rows[0]
    assert row["baseline_disposition"] == "CENSORED_ENTRY_WINDOW"
    assert row["m1_reclaim_disposition"] == "CENSORED_ENTRY_WINDOW"
    assert row["policy_complete"] is False
    assert row["policy_delta_r"] is None
    assert result.summary["complete_pairs"] == 0
    assert result.summary["excluded_pairs"] == 1


def test_parent_population_rejects_existing_m15_pivot_reclaim():
    with pytest.raises(ValueError, match="only accepts"):
        run_h1_m1_paired_replay(
            contexts=[_context(trigger_kind="h1_pivot_reclaim_15m")],
            m1_by_symbol={"EURUSD": _completed_reclaim_frame()},
            config=_config(),
        )


def test_duplicate_context_ids_are_rejected_before_replay():
    context = _context()
    with pytest.raises(ValueError, match="context_id values must be unique"):
        run_h1_m1_paired_replay(
            contexts=[context, copy.copy(context)],
            m1_by_symbol={"EURUSD": _completed_reclaim_frame()},
            config=_config(),
        )


def test_missing_m1_inside_entry_window_is_censored_not_no_fill():
    frame = _frame([
        ("2026-01-01T00:00:00Z", 1.1010, 1.1020, 1.1005, 1.1010),
        ("2026-01-01T00:01:00Z", 1.1010, 1.1020, 1.1005, 1.1010),
        # 00:02 is missing.
        ("2026-01-01T00:03:00Z", 1.1010, 1.1020, 1.1005, 1.1010),
        ("2026-01-01T00:04:00Z", 1.1010, 1.1020, 1.1005, 1.1010),
        ("2026-01-01T00:05:00Z", 1.1010, 1.1020, 1.1005, 1.1010),
    ])

    result = run_h1_m1_paired_replay(
        contexts=[_context()],
        m1_by_symbol={"EURUSD": frame},
        config=_config(),
    )

    row = result.rows[0]
    assert row["baseline_disposition"] == "CENSORED_ENTRY_WINDOW"
    assert row["m1_reclaim_trigger_status"] == "DATA_GAP"
    assert row["policy_complete"] is False
    assert result.summary["complete_pairs"] == 0


def test_missing_m1_trigger_bar_censors_only_m1_arm():
    frame = _frame([
        ("2026-01-01T00:00:00Z", 1.1010, 1.1020, 1.1005, 1.1010),
        # Decision-time 00:01 trigger candle is missing, while every
        # executable baseline open remains available.
        ("2026-01-01T00:02:00Z", 1.0995, 1.1000, 1.0990, 1.0995),
        ("2026-01-01T00:03:00Z", 1.0995, 1.1000, 1.0990, 1.0995),
        ("2026-01-01T00:04:00Z", 1.0995, 1.1000, 1.0990, 1.0995),
        ("2026-01-01T00:05:00Z", 1.0995, 1.1000, 1.0990, 1.0995),
        ("2026-01-01T00:06:00Z", 1.0995, 1.1000, 1.0990, 1.0995),
    ])

    result = run_h1_m1_paired_replay(
        contexts=[_context()],
        m1_by_symbol={"EURUSD": frame},
        config=_config(),
    )

    row = result.rows[0]
    assert row["baseline_disposition"] == "FILLED"
    assert row["baseline_label_status"] == "FORCED_TIME"
    assert row["m1_reclaim_disposition"] == "CENSORED_ENTRY_WINDOW"
    assert row["m1_reclaim_trigger_status"] == "DATA_GAP"
    assert row["policy_complete"] is False


def test_missing_m1_inside_used_outcome_path_is_censored():
    frame = _frame([
        ("2026-01-01T00:00:00Z", 1.1010, 1.1020, 1.1005, 1.1010),
        ("2026-01-01T00:01:00Z", 1.1010, 1.1020, 1.1005, 1.1010),
        ("2026-01-01T00:02:00Z", 1.0995, 1.1000, 1.0990, 1.0995),
        # 00:03 is missing before the later stop observation.
        ("2026-01-01T00:04:00Z", 1.0960, 1.0970, 1.0940, 1.0950),
        ("2026-01-01T00:05:00Z", 1.0950, 1.0960, 1.0940, 1.0950),
        ("2026-01-01T00:06:00Z", 1.0950, 1.0960, 1.0940, 1.0950),
    ])

    result = run_h1_m1_paired_replay(
        contexts=[_context(expiry="2026-01-01T00:03:00Z")],
        m1_by_symbol={"EURUSD": frame},
        config=_config(),
    )

    row = result.rows[0]
    assert row["baseline_disposition"] == "FILLED"
    assert row["baseline_label_status"] == "CENSORED"
    assert "missing M1 bar" in row["baseline_label_reason"]
    assert row["policy_complete"] is False


def test_explicit_fold_clips_entry_and_outcome_without_next_fold_data():
    context = _context(expiry="2026-01-01T00:10:00Z")
    frame = _frame([
        ("2026-01-01T00:00:00Z", 1.1010, 1.1020, 1.1005, 1.1010),
        ("2026-01-01T00:01:00Z", 1.1010, 1.1020, 1.1005, 1.1010),
        ("2026-01-01T00:02:00Z", 1.0995, 1.1000, 1.0990, 1.0995),
        ("2026-01-01T00:03:00Z", 1.0995, 1.1000, 1.0990, 1.0995),
        ("2026-01-01T00:04:00Z", 1.0995, 1.1000, 1.0990, 1.0995),
        ("2026-01-01T00:05:00Z", 1.2000, 1.2100, 1.1900, 1.2050),
    ])
    fold = H1M1FoldWindow(
        fold_index=3,
        test_start=pd.Timestamp("2026-01-01T00:00:00Z"),
        test_end=pd.Timestamp("2026-01-01T00:05:00Z"),
    )

    result = run_h1_m1_paired_replay(
        contexts=[context],
        m1_by_symbol={"EURUSD": frame},
        config=H1M1ReplayConfig(outcome_horizon="20min"),
        fold_by_context={context.context_id: fold},
    )

    row = result.rows[0]
    assert row["fold_index"] == 3
    assert row["fold_clipped"] is True
    assert row["outcome_cutoff"] == "2026-01-01T00:05:00+00:00"
    assert result.summary["oos_boundary_safe"] is True


def test_cost_profile_changes_policy_basis_and_reports_entry_metrics():
    context = _context()
    fold = H1M1FoldWindow(
        fold_index=0,
        test_start=pd.Timestamp("2026-01-01T00:00:00Z"),
        test_end=pd.Timestamp("2026-01-02T00:00:00Z"),
    )
    result = run_h1_m1_paired_replay(
        contexts=[context],
        m1_by_symbol={"EURUSD": _completed_reclaim_frame()},
        config=_config(),
        cost_profile=_cost_profile(),
        fold_by_context={context.context_id: fold},
    )

    row = result.rows[0]
    assert row["baseline_transaction_cost_r"] > 0.0
    assert row["m1_reclaim_transaction_cost_r"] > 0.0
    assert row["baseline_policy_r"] == pytest.approx(
        row["baseline_gross_r"] - row["baseline_transaction_cost_r"]
    )
    assert result.summary["metric_basis"] == (
        "net_after_transaction_costs"
    )
    assert result.summary[
        "cost_profile_created_before_all_decisions"
    ] is True
    assert result.summary["cost_evidence_scope"] == (
        "POINT_IN_TIME_PROFILE_POST_HOC_OUTCOME_COSTS"
    )
    assert result.summary["baseline_entry_metrics"]["setups_closed"] == 1
    assert result.summary["baseline_gross_entry_metrics"]["net_r"] > (
        result.summary["baseline_entry_metrics"]["net_r"]
    )
