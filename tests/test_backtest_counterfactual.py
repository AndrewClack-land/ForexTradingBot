from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pandas as pd
import pytest

import backtest.counterfactual as counterfactual_module
from backtest.counterfactual import (
    TRIGGER_MANIFEST,
    run_counterfactual_backtest,
)
from backtest.data import HistoricalDataset
from backtest.strategy_runner import (
    NarrativeBacktestConfig,
    StrategyBacktestError,
)
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

    def trigger_15m_cluster_rejection(self, _, side, event, *, symbol):
        assert event is None
        assert symbol == "EURUSD"
        type(self).calls[("fxpro_cluster_rejection_15m", side)] += 1
        return self._entry(side, "fxpro_cluster_rejection_15m", 0.05)

    def trigger_15m_quote_pressure_rejection(self, _, side, event, *, symbol):
        assert event is None
        assert symbol == "EURUSD"
        type(self).calls[("fxpro_quote_pressure_rejection_15m", side)] += 1
        return self._entry(side, "fxpro_quote_pressure_rejection_15m", 0.10)

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
    entry_ttl="15min",
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
        entry_ttl=entry_ttl,
        max_holding="15min",
        intrabar_policy="stop-first",
        profile="signal-quality",
        train="60min",
        test="30min",
        step="30min",
        rejection_block_entry_enabled=rejection_block_entry_enabled,
        orderblock_entry_enabled=orderblock_entry_enabled,
    )


def test_optimize_v2_rejects_cross_decision_pending_overlap():
    with pytest.raises(
        StrategyBacktestError,
        match=r"entry_ttl <= 15 minutes",
    ):
        run_counterfactual_backtest(
            _dataset("counterfactual-overlap"),
            _config(entry_ttl="30min"),
            strategy_factory=_AllSideAllTriggerStrategy,
            min_train_opportunities=1,
        )


def test_private_replay_rejects_overlapping_pending_decisions():
    decision_time = pd.Timestamp("2026-01-01T00:15:00Z")
    signal = {
        "side": "LONG",
        "entry_price": 100.0,
        "entry_min": 99.9,
        "entry_max": 100.1,
        "stop_price": 99.0,
        "tp_prices": [101.0, 102.0, 103.0],
        "trigger_kind": "h1_pivot_reclaim_15m",
        "optimizer_rank": 1,
    }
    selected = [
        counterfactual_module._Candidate(
            candidate_id="overlap-a",
            fold_index=0,
            symbol="EURUSD",
            decision_time=decision_time,
            gate="ENTER",
            gate_reason="eligible",
            signal=signal,
        ),
        counterfactual_module._Candidate(
            candidate_id="overlap-b",
            fold_index=0,
            symbol="EURUSD",
            decision_time=decision_time + pd.Timedelta(minutes=5),
            gate="ENTER",
            gate_reason="eligible",
            signal=signal,
        ),
    ]
    period = counterfactual_module._Period(
        fold_index=0,
        train_start=None,
        train_end=None,
        test_start=decision_time,
        test_end=decision_time + pd.Timedelta(minutes=30),
    )

    with pytest.raises(
        StrategyBacktestError,
        match=r"overlapping pending entry windows",
    ):
        counterfactual_module._run_selected_replay(
            selected=selected,
            prepared_by_symbol={"EURUSD": {}},
            periods=[period],
            config=_config(),
            progress=None,
        )


def test_adjacent_decisions_at_ttl_boundary_do_not_overlap():
    decision_time = pd.Timestamp("2026-01-01T00:15:00Z")
    signal = {
        "side": "LONG",
        "entry_price": 100.0,
        "entry_min": 99.9,
        "entry_max": 100.1,
        "stop_price": 99.0,
        "tp_prices": [101.0, 102.0, 103.0],
        "trigger_kind": "h1_pivot_reclaim_15m",
        "optimizer_rank": 1,
    }
    selected = [
        counterfactual_module._Candidate(
            candidate_id=f"boundary-{index}",
            fold_index=0,
            symbol="EURUSD",
            decision_time=decision_time + pd.Timedelta(minutes=15 * index),
            gate="ENTER",
            gate_reason="eligible",
            signal=signal,
        )
        for index in range(2)
    ]
    m1_index = pd.date_range(
        "2026-01-01T00:16:00Z",
        periods=30,
        freq="1min",
    )
    m1 = _frame(
        m1_index,
        [(101.5, 101.5, 101.5, 101.5)] * len(m1_index),
    )

    ordered = counterfactual_module._causal_ranked_replay_order(
        candidates=selected,
        prepared={"1m": SimpleNamespace(frame=m1)},
        config=_config(),
    )

    assert [row.candidate_id for row in ordered] == [
        "boundary-0",
        "boundary-1",
    ]


def test_entry_ttl_deadline_is_exclusive():
    decision_time = pd.Timestamp("2026-01-01T00:15:00Z")
    index = pd.DatetimeIndex(
        [
            "2026-01-01T00:16:00Z",
            "2026-01-01T00:29:00Z",
            "2026-01-01T00:30:00Z",
        ]
    )
    signal = {
        "side": "LONG",
        "entry_price": 100.0,
        "entry_min": 100.0,
        "entry_max": 100.0,
        "stop_price": 99.0,
        "tp_prices": [101.0, 102.0, 103.0],
    }
    deadline_only = _frame(
        index,
        [
            (101.5, 101.5, 101.5, 101.5),
            (101.5, 101.5, 101.5, 101.5),
            (100.0, 100.0, 100.0, 100.0),
        ],
    )
    before_deadline = deadline_only.copy()
    before_deadline.loc[index[1], ["open", "high", "low", "close"]] = 100.0

    _, deadline_fill, _, _ = counterfactual_module._find_fill(
        m1=deadline_only,
        decision_time=decision_time,
        period_end=pd.Timestamp("2026-01-01T01:00:00Z"),
        signal=signal,
        config=_config(),
    )
    _, causal_fill, _, _ = counterfactual_module._find_fill(
        m1=before_deadline,
        decision_time=decision_time,
        period_end=pd.Timestamp("2026-01-01T01:00:00Z"),
        signal=signal,
        config=_config(),
    )

    assert deadline_fill is None
    assert causal_fill == pd.Timestamp("2026-01-01T00:29:00Z")


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
    assert (report / "trigger_attribution.csv").is_file()
    assert (report / "manifest.json").is_file()
    trigger_rows = list(result.trigger_attribution)
    assert trigger_rows
    assert {
        row["trigger_kind"] for row in trigger_rows
        if row["dimension"] == "trigger"
    } == set(TRIGGER_MANIFEST)


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


def test_ranked_replay_never_fills_retroactively_after_a_full_ttl_no_fill():
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
        {"FILLED", "BLOCK_ALTERNATIVE"} <= {row["disposition"] for row in rows}
        for rows in by_decision.values()
    )
    assert not any(
        {"EXPIRED_ENTRY_RANGE", "FILLED"}
        <= {row["disposition"] for row in rows}
        for rows in by_decision.values()
    )
    assert all(
        sum(row["disposition"] == "FILLED" for row in rows) <= 1
        for rows in by_decision.values()
    )
    assert not {setup.get("forced_exit_reason") for setup in result.setups} & {
        "FOLD_END"
    }


def test_ranked_replay_uses_earliest_fill_then_rank_for_timestamp_tie():
    decision_time = pd.Timestamp("2026-01-01T00:15:00Z")
    index = pd.date_range(
        "2026-01-01T00:16:00Z",
        periods=31,
        freq="1min",
    )
    opens = [100.1, 100.0, *([100.2] * (len(index) - 2))]
    m1 = _frame(
        index,
        [(price, price, price, price) for price in opens],
    )

    def candidate(
        candidate_id,
        *,
        rank,
        entry,
        trigger_kind,
    ):
        return counterfactual_module._Candidate(
            candidate_id=candidate_id,
            fold_index=0,
            symbol="EURUSD",
            decision_time=decision_time,
            gate="ENTER",
            gate_reason="eligible",
            signal={
                "side": "LONG",
                "entry_price": entry,
                "entry_min": entry,
                "entry_max": entry,
                "stop_price": entry - 1.0,
                "tp_prices": [entry + 1.0, entry + 2.0, entry + 3.0],
                "trigger_kind": trigger_kind,
                "trigger_reason": trigger_kind,
                "optimizer_rank": rank,
            },
        )

    selected = [
        candidate(
            "rank-1-no-fill",
            rank=1,
            entry=101.0,
            trigger_kind="rejection_block_15m",
        ),
        candidate(
            "rank-2-later-fill",
            rank=2,
            entry=100.0,
            trigger_kind="fxpro_quote_pressure_rejection_15m",
        ),
        candidate(
            "a-rank-4-earliest-fill",
            rank=4,
            entry=100.1,
            trigger_kind="order_block_1h",
        ),
        candidate(
            "z-rank-3-earliest-fill",
            rank=3,
            entry=100.1,
            trigger_kind="h1_pivot_reclaim_15m",
        ),
    ]
    period = counterfactual_module._Period(
        fold_index=0,
        train_start=None,
        train_end=None,
        test_start=decision_time,
        test_end=pd.Timestamp("2026-01-01T00:45:00Z"),
    )

    setups, _, executions, _ = counterfactual_module._run_selected_replay(
        selected=selected,
        prepared_by_symbol={"EURUSD": {"1m": SimpleNamespace(frame=m1)}},
        periods=[period],
        config=_config(),
        progress=None,
    )

    filled = [row for row in executions if row["disposition"] == "FILLED"]
    assert [row["candidate_id"] for row in filled] == [
        "z-rank-3-earliest-fill"
    ]
    assert filled[0]["fill_time"] == "2026-01-01T00:16:00+00:00"
    assert setups[0]["candidate_id"] == "z-rank-3-earliest-fill"
    assert {
        row["candidate_id"]
        for row in executions
        if row["disposition"] == "BLOCK_ALTERNATIVE"
    } == {
        "rank-1-no-fill",
        "rank-2-later-fill",
        "a-rank-4-earliest-fill",
    }
    assert "EXPIRED_ENTRY_RANGE" not in {
        row["disposition"] for row in executions
    }
