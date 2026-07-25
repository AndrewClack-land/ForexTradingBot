from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

import backtest.counterfactual as counterfactual_module
from backtest.counterfactual import (
    TRIGGER_MANIFEST,
    run_counterfactual_backtest,
)
from backtest.data import HistoricalDataset
from backtest.strategy_runner import NarrativeBacktestConfig
from core.narrative_scoring import build_factor_vector
from core.strategy_narrative import CandidateEntry


def _frame(index, rows):
    return pd.DataFrame(
        rows,
        columns=["open", "high", "low", "close"],
        index=pd.DatetimeIndex(index, name="timestamp"),
    )


def _dataset(tmp_path):
    m15_index = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=13,
        freq="15min",
    )
    m1_index = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=211,
        freq="1min",
    )
    frames = {
        ("EURUSD", "1d"): _frame(
            ["2025-12-31T00:00:00Z"],
            [(100.0, 101.0, 99.0, 100.0)],
        ),
        ("EURUSD", "4h"): _frame(
            ["2025-12-31T20:00:00Z"],
            [(100.0, 101.0, 99.0, 100.0)],
        ),
        ("EURUSD", "1h"): _frame(
            ["2025-12-31T23:00:00Z"],
            [(100.0, 101.0, 99.0, 100.0)],
        ),
        ("EURUSD", "15m"): _frame(
            m15_index,
            [(100.0, 101.0, 99.0, 100.0)] * len(m15_index),
        ),
        ("EURUSD", "1m"): _frame(
            m1_index,
            [(100.0, 100.0, 100.0, 100.0)] * len(m1_index),
        ),
    }
    return HistoricalDataset(
        tmp_path,
        frames,
        source_files=[],
        raw_counts={key: len(frame) for key, frame in frames.items()},
    )


class _AllSideAllTriggerStrategy:
    calls = Counter()

    def __init__(self):
        self._last_htf_context = None
        self._last_factor_vector = None
        self.risk_per_trade = 0.01

    def calc_narrative(self, *_):
        self._last_factor_vector = build_factor_vector(
            {
                "h1_premium_discount": {
                    "present": True,
                    "side": "LONG",
                },
                "false_breakout_4h": {
                    "present": True,
                    "side": "SHORT",
                },
                "true_breakout_15m": {
                    "present": True,
                    "side": "LONG",
                },
                "order_block_1h": {
                    "present": True,
                    "side": "LONG",
                },
                "rejection_block_1h": {
                    "present": True,
                    "side": "SHORT",
                },
            },
            base_margin=2,
        )
        # The counterfactual path must ignore this returned production bias.
        return "NEUTRAL", "frozen test context"

    def calc_fvg_regime_1h(self, *_):
        return "NEUTRAL", "FVG neutral"

    @staticmethod
    def _entry(side, kind, offset):
        center = 100.0 + offset
        return CandidateEntry(
            side=side,
            entry_price=center,
            entry_min=center - 0.04,
            entry_max=center + 0.04,
            tf="15M",
            reason=f"{kind} {side}",
            trigger_kind=kind,
            lock_entry_range=True,
        )

    def trigger_15m_rejection_block(self, _, side):
        type(self).calls[("rejection_block_15m", side)] += 1
        return self._entry(side, "rejection_block_15m", 0.00)

    def trigger_15m_absorption(self, _, side, event, *, symbol):
        assert event is None
        assert symbol == "EURUSD"
        type(self).calls[("absorption_15m", side)] += 1
        return self._entry(side, "absorption_15m", 0.10)

    def trigger_h1_pivot_reclaim_on_15m(self, _h1, _m15, side):
        type(self).calls[("h1_pivot_reclaim_15m", side)] += 1
        return self._entry(side, "h1_pivot_reclaim_15m", 0.00)

    def trigger_orderblock_touch(self, _h1, _ctx, side):
        type(self).calls[("order_block_1h", side)] += 1
        return self._entry(side, "order_block_1h", 0.00)

    def trigger_15m_turtle_soup(self, *_args, **_kwargs):
        raise AssertionError("retired Turtle Soup must never be called")

    def calc_stop_and_tps(
        self,
        entry,
        side,
        *_args,
        **_kwargs,
    ):
        if side == "LONG":
            return entry - 10.0, [entry + 10.0, entry + 20.0, entry + 30.0]
        return entry + 10.0, [entry - 10.0, entry - 20.0, entry - 30.0]


def _config(
    *,
    rejection_block_entry_enabled=True,
    orderblock_entry_enabled=True,
):
    return NarrativeBacktestConfig.build(
        symbols=["EURUSD"],
        start="2026-01-01T00:15:00Z",
        end="2026-01-01T03:15:00Z",
        initial_capital=10_000,
        risk_fraction=0.01,
        history_limit=1,
        min_context_bars=1,
        min_daily_bars=1,
        entry_ttl="15min",
        max_holding="15min",
        intrabar_policy="stop-first",
        profile="signal-quality",
        train="60min",
        test="30min",
        step="30min",
        rejection_block_entry_enabled=rejection_block_entry_enabled,
        orderblock_entry_enabled=orderblock_entry_enabled,
    )


def test_universe_regenerates_both_sides_and_every_trigger_inside_train():
    _AllSideAllTriggerStrategy.calls.clear()
    result = run_counterfactual_backtest(
        _dataset("counterfactual-memory"),
        _config(),
        strategy_factory=_AllSideAllTriggerStrategy,
        min_train_opportunities=1,
    )

    assert result.summary["turtle_soup_called"] is False
    assert tuple(result.summary["trigger_manifest"]) == TRIGGER_MANIFEST
    assert result.summary["decision_events"] > 0
    assert {
        row["production_bias_ignored_for_generation"]
        for row in result.decision_events
    } == {"NEUTRAL"}
    assert {row["side"] for row in result.opportunities} == {
        "LONG",
        "SHORT",
    }
    assert {row["trigger_kind"] for row in result.opportunities} == set(
        TRIGGER_MANIFEST
    )
    by_decision_side = Counter(
        (row["decision_event_id"], row["side"])
        for row in result.opportunities
    )
    assert set(by_decision_side.values()) == {len(TRIGGER_MANIFEST)}
    for trigger in TRIGGER_MANIFEST:
        assert _AllSideAllTriggerStrategy.calls[(trigger, "LONG")] > 0
        assert _AllSideAllTriggerStrategy.calls[(trigger, "SHORT")] > 0


def test_train_only_weights_replay_oos_without_threshold_and_keep_one_pct_risk(
    tmp_path,
):
    result = run_counterfactual_backtest(
        _dataset(tmp_path / "data"),
        _config(),
        strategy_factory=_AllSideAllTriggerStrategy,
        min_train_opportunities=1,
    )

    assert result.summary["optimizer"]["models_fit"] > 0
    assert result.summary["optimizer"]["no_hard_threshold"] is True
    assert result.selections
    assert all(
        row["hard_score_threshold_applied"] is False
        for row in result.selections
    )
    assert all(
        setup["risk_amount"] == pytest.approx(100.0)
        for setup in result.setups
    )
    assert result.summary["risk"]["risk_fraction"] == pytest.approx(0.01)
    assert result.summary["risk"]["changed_by_optimizer"] is False
    assert {
        row["label_status"] for row in result.labels
    } >= {"FILLED", "NO_FILL"}

    report = result.write(tmp_path / "report")
    assert (report / "frozen_weight_models.json").is_file()
    assert (report / "decision_events.csv").is_file()
    assert (report / "technical_opportunities.csv").is_file()
    assert (report / "opportunity_labels.csv").is_file()
    assert (report / "manifest.json").is_file()


def test_not_fit_fold_is_audited_but_never_trades_reference_weights(
    tmp_path,
):
    result = run_counterfactual_backtest(
        _dataset(tmp_path / "data"),
        _config(),
        strategy_factory=_AllSideAllTriggerStrategy,
        min_train_opportunities=10_000,
    )

    assert not result.setups
    assert not result.executions
    assert not result.selections or {
        row["selection_status"] for row in result.selections
    } == {"SKIPPED_NOT_FIT"}
    assert result.summary["oos_selected_candidates"] == 0
    assert result.summary["oos_decisions_skipped_not_fit"] > 0

    report = result.write(tmp_path / "not-fit-report")
    for filename in (
        "executions.csv",
        "setups.csv",
        "legs.csv",
    ):
        assert (report / filename).read_text(
            encoding="utf-8"
        ).strip()


def test_disabled_trigger_is_still_labelled_but_blocked_only_in_replay():
    result = run_counterfactual_backtest(
        _dataset("counterfactual-disabled-trigger"),
        _config(rejection_block_entry_enabled=False),
        strategy_factory=_AllSideAllTriggerStrategy,
        min_train_opportunities=1,
    )

    rb_opportunities = [
        row
        for row in result.opportunities
        if row["trigger_kind"] == "rejection_block_15m"
    ]
    assert rb_opportunities
    assert {
        row["gate"] for row in rb_opportunities
    } == {"BLOCK_TRIGGER_DISABLED"}
    rb_labels = [
        row
        for row in result.labels
        if row["trigger_kind"] == "rejection_block_15m"
    ]
    assert rb_labels
    assert all(
        row["label_status"] != "INVALID"
        for row in rb_labels
    )
    assert all(
        row["trigger_kind"] != "rejection_block_15m"
        for row in result.selections
        if row["selection_status"] == "RANKED_FOR_REPLAY"
    )


def test_orderblock_detector_is_forced_on_for_training_then_replay_blocked():
    class _FlagAwareOrderBlockStrategy(_AllSideAllTriggerStrategy):
        orderblock_entry_enabled = False

        def trigger_orderblock_touch(self, h1, context, side):
            if not self.orderblock_entry_enabled:
                return None
            return super().trigger_orderblock_touch(h1, context, side)

    result = run_counterfactual_backtest(
        _dataset("counterfactual-disabled-orderblock"),
        _config(orderblock_entry_enabled=False),
        strategy_factory=_FlagAwareOrderBlockStrategy,
        min_train_opportunities=1,
    )

    order_blocks = [
        row
        for row in result.opportunities
        if row["trigger_kind"] == "order_block_1h"
    ]
    assert order_blocks
    assert {
        row["gate"] for row in order_blocks
    } == {"BLOCK_TRIGGER_DISABLED"}
    assert all(
        row["label_status"] != "INVALID"
        for row in result.labels
        if row["trigger_kind"] == "order_block_1h"
    )


def test_summary_metrics_sort_cross_market_exits_chronologically(
    monkeypatch,
):
    setups = [
        {
            "policy": "stop-first",
            "symbol": "GBPUSD",
            "decision_time": "2026-01-01T02:30:00+00:00",
            "entry_time": "2026-01-01T02:45:00+00:00",
            "exit_time": "2026-01-01T03:00:00+00:00",
            "status": "CLOSED",
            "net_r": -1.0,
        },
        {
            "policy": "stop-first",
            "symbol": "EURUSD",
            "decision_time": "2026-01-01T00:30:00+00:00",
            "entry_time": "2026-01-01T00:45:00+00:00",
            "exit_time": "2026-01-01T01:00:00+00:00",
            "status": "CLOSED",
            "net_r": -1.0,
        },
        {
            "policy": "stop-first",
            "symbol": "EURUSD",
            "decision_time": "2026-01-01T01:30:00+00:00",
            "entry_time": "2026-01-01T01:45:00+00:00",
            "exit_time": "2026-01-01T02:00:00+00:00",
            "status": "CLOSED",
            "net_r": 2.0,
        },
    ]

    def fake_replay(**_kwargs):
        return setups, [], [], []

    monkeypatch.setattr(
        counterfactual_module,
        "_run_selected_replay",
        fake_replay,
    )
    result = run_counterfactual_backtest(
        _dataset("counterfactual-metric-order"),
        _config(),
        strategy_factory=_AllSideAllTriggerStrategy,
        min_train_opportunities=1,
    )

    metrics = result.summary["oos_primary_metrics"]
    assert metrics["max_drawdown_r"] == pytest.approx(1.0)
    assert metrics["longest_loss_streak"] == 1


def test_ranked_replay_falls_through_no_fill_then_stops_after_first_fill():
    class _FallbackStrategy(_AllSideAllTriggerStrategy):
        def trigger_15m_rejection_block(self, _, side):
            return self._entry(side, "rejection_block_15m", 0.10)

    result = run_counterfactual_backtest(
        _dataset("counterfactual-ranked-fallback"),
        _config(),
        strategy_factory=_FallbackStrategy,
        min_train_opportunities=1,
    )

    by_decision = {}
    for row in result.executions:
        by_decision.setdefault(row["decision_time"], []).append(row)
    assert any(
        "EXPIRED_ENTRY_RANGE" in {
            row["disposition"] for row in rows
        }
        and "FILLED" in {row["disposition"] for row in rows}
        and "BLOCK_ALTERNATIVE" in {
            row["disposition"] for row in rows
        }
        for rows in by_decision.values()
    )
    assert all(
        sum(row["disposition"] == "FILLED" for row in rows) <= 1
        for rows in by_decision.values()
    )
    assert not {
        setup.get("forced_exit_reason")
        for setup in result.setups
    } & {"FOLD_END"}
