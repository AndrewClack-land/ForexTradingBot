from __future__ import annotations

from copy import copy
from types import SimpleNamespace

import pandas as pd
import pytest

import core.strategy_narrative as strategy_module
from backtest.counterfactual import _materialize_entry
from core.htf_context import PineRejectionBlockTracker
from core.strategy_narrative import CandidateEntry, NarrativeStrategy


def _frame(periods: int = 24) -> pd.DataFrame:
    index = pd.date_range(
        "2026-01-01",
        periods=periods,
        freq="1h",
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "open": [1.10] * periods,
            "high": [1.12] * periods,
            "low": [1.08] * periods,
            "close": [1.11] * periods,
            "volume": [1.0] * periods,
        },
        index=index,
    )


def _block(
    *,
    side: str,
    created_idx: int,
    available_idx: int,
    zone_low: float,
    zone_high: float,
    retested: bool = False,
    broken: bool = False,
    valid: bool = True,
    detector_version: str = "pine-v1-causal",
    pivot_time=None,
    available_time=None,
    baseline_unverified: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        side=side,
        created_idx=created_idx,
        available_idx=available_idx,
        zone_low=zone_low,
        zone_high=zone_high,
        retested=retested,
        broken=broken,
        valid=valid,
        detector_version=detector_version,
        pivot_time=pivot_time,
        available_time=available_time,
        wick_ratio=2.5,
        atr_at_pivot=0.01,
        baseline_unverified=baseline_unverified,
    )


@pytest.mark.parametrize(
    ("symbol", "min_tick"),
    [("EURUSD", 0.00001), ("GOLD", 0.01)],
)
def test_live_htf_context_uses_pine_contract_and_symbol_tick(
    symbol,
    min_tick,
):
    strategy = NarrativeStrategy()
    frame = _frame()

    ctx = strategy._build_htf_context(frame, frame, frame, symbol)

    assert ctx is not None
    assert ctx.rb_detector_version == PineRejectionBlockTracker.VERSION
    assert isinstance(ctx.rb_tracker, PineRejectionBlockTracker)
    assert ctx.min_tick == pytest.approx(min_tick)


@pytest.mark.parametrize(
    ("side", "zone_low", "zone_high", "entry", "stop"),
    [
        ("LONG", 1.0800, 1.1000, 1.1000, 1.0800),
        ("SHORT", 1.1000, 1.1200, 1.1000, 1.1200),
    ],
)
def test_h1_rb_builds_exact_proximal_limit_and_distal_stop(
    side,
    zone_low,
    zone_high,
    entry,
    stop,
):
    frame = _frame()
    rb = _block(
        side=side,
        created_idx=16,
        available_idx=19,
        zone_low=zone_low,
        zone_high=zone_high,
    )
    ctx = SimpleNamespace(rejection_blocks=[rb])
    strategy = NarrativeStrategy()

    first = strategy.trigger_h1_rejection_block(
        frame,
        side,
        ctx=ctx,
        symbol="EURUSD",
    )
    repeated = strategy.trigger_h1_rejection_block(
        frame,
        side,
        ctx=ctx,
        symbol="EURUSD",
    )

    assert first is not None
    assert first.entry_price == pytest.approx(entry)
    assert first.entry_min == pytest.approx(entry)
    assert first.entry_max == pytest.approx(entry)
    assert first.zone_low == pytest.approx(zone_low)
    assert first.zone_high == pytest.approx(zone_high)
    assert first.stop_override == pytest.approx(stop)
    assert first.lock_entry_range is True
    assert first.lock_stop_override is True
    assert first.entry_order_type == "LIMIT_RETEST"
    assert first.entry_ttl_min is None
    assert first.trigger_kind == "rejection_block_1h"
    assert first.trigger_event_id == repeated.trigger_event_id
    assert first.trigger_event_id == (
        f"rb:h1:touch:v1:EURUSD:{side}:"
        f"{pd.Timestamp(frame.index[16]).isoformat()}"
    )
    assert first.trigger_meta["schema"] == "rb-h1-exact-retest/v1"
    assert first.trigger_meta["available_index"] == 19
    assert first.trigger_meta["execution_ttl_source"] == "policy"


def test_h1_rb_ignores_retested_broken_invalid_and_legacy_blocks():
    frame = _frame()
    blocks = [
        _block(
            side="LONG",
            created_idx=18,
            available_idx=21,
            zone_low=1.01,
            zone_high=1.02,
            retested=True,
        ),
        _block(
            side="LONG",
            created_idx=17,
            available_idx=20,
            zone_low=1.02,
            zone_high=1.03,
            broken=True,
        ),
        _block(
            side="LONG",
            created_idx=16,
            available_idx=19,
            zone_low=1.03,
            zone_high=1.04,
            valid=False,
        ),
        _block(
            side="LONG",
            created_idx=15,
            available_idx=18,
            zone_low=1.04,
            zone_high=1.05,
            detector_version="legacy",
        ),
        _block(
            side="LONG",
            created_idx=14,
            available_idx=17,
            zone_low=1.05,
            zone_high=1.06,
            baseline_unverified=True,
            valid=False,
            retested=True,
        ),
    ]

    entry = NarrativeStrategy().trigger_h1_rejection_block(
        frame,
        "LONG",
        ctx=SimpleNamespace(rejection_blocks=blocks),
    )

    assert entry is None


def test_old_persisted_block_triggers_without_current_window_indices():
    frame = _frame()
    pivot_time = "2025-12-20T03:00:00+00:00"
    available_time = "2025-12-20T07:00:00+00:00"
    rb = _block(
        side="LONG",
        created_idx=999,
        available_idx=1002,
        zone_low=1.08,
        zone_high=1.10,
        pivot_time=pivot_time,
        available_time=available_time,
    )
    ctx = SimpleNamespace(
        rejection_blocks=[rb],
        rb_tracker=SimpleNamespace(symbol="GBPUSD"),
    )

    entry = NarrativeStrategy().trigger_h1_rejection_block(
        frame,
        "LONG",
        ctx=ctx,
    )

    assert entry is not None
    assert entry.trigger_event_id == (
        "rb:h1:touch:v1:GBPUSD:LONG:"
        f"{pivot_time}"
    )
    assert entry.trigger_meta["pivot_index"] == 999
    assert entry.trigger_meta["available_index"] == 1002
    assert entry.trigger_meta["available_time"] == available_time


def test_symbol_is_part_of_durable_rb_event_identity():
    frame = _frame()
    rb = _block(
        side="LONG",
        created_idx=16,
        available_idx=19,
        zone_low=1.08,
        zone_high=1.10,
    )
    ctx = SimpleNamespace(rejection_blocks=[rb])
    strategy = NarrativeStrategy()

    eur = strategy.trigger_h1_rejection_block(
        frame,
        "LONG",
        ctx=ctx,
        symbol="EURUSD",
    )
    gbp = strategy.trigger_h1_rejection_block(
        frame,
        "LONG",
        ctx=ctx,
        symbol="GBPUSD",
    )

    assert eur is not None and gbp is not None
    assert eur.trigger_event_id != gbp.trigger_event_id
    assert ":EURUSD:LONG:" in eur.trigger_event_id
    assert ":GBPUSD:LONG:" in gbp.trigger_event_id


def test_shadow_shallow_copy_detaches_mutable_rb_ledgers():
    strategy = NarrativeStrategy()
    tracker = strategy._pine_rb_tracker("EURUSD")
    tracker.build(_frame(8))
    snapshot = copy(strategy)

    assert snapshot is not strategy
    assert snapshot._pine_rb_trackers["EURUSD"] is not tracker
    assert (
        snapshot.export_rejection_block_h1_state()
        == strategy.export_rejection_block_h1_state()
    )
    snapshot._pine_rb_trackers["EURUSD"]._dirty = True
    tracker.mark_clean()
    assert snapshot.rejection_block_h1_state_dirty is True
    assert strategy.rejection_block_h1_state_dirty is False


def test_h1_rb_ctx_none_uses_pine_tracker_fallback(monkeypatch):
    frame = _frame()
    rb = _block(
        side="LONG",
        created_idx=16,
        available_idx=19,
        zone_low=1.08,
        zone_high=1.10,
    )
    calls = []

    class _Tracker:
        VERSION = "pine-v1-causal"

        def build(self, observed):
            calls.append(observed)
            return [rb]

    monkeypatch.setattr(
        strategy_module,
        "PineRejectionBlockTracker",
        _Tracker,
    )

    entry = NarrativeStrategy().trigger_h1_rejection_block(frame, "LONG")

    assert entry is not None
    assert calls == [frame]


def test_exact_structural_stop_bypasses_atr_clamp_and_validates_side():
    strategy = NarrativeStrategy()
    strategy._atr = lambda *_args, **_kwargs: 0.01
    frame = _frame()

    exact_stop, _ = strategy.calc_stop_and_tps(
        1.10,
        "LONG",
        frame,
        frame,
        custom_stop=1.099,
        lock_custom_stop=True,
    )
    bounded_stop, _ = strategy.calc_stop_and_tps(
        1.10,
        "LONG",
        frame,
        frame,
        custom_stop=1.099,
    )

    assert exact_stop == pytest.approx(1.099)
    assert bounded_stop == pytest.approx(1.0925)
    with pytest.raises(ValueError, match="below entry"):
        strategy.calc_stop_and_tps(
            1.10,
            "LONG",
            frame,
            frame,
            custom_stop=1.101,
            lock_custom_stop=True,
        )


def test_payload_audits_exact_stop_order_type_and_policy_ttl():
    strategy = NarrativeStrategy()
    strategy._atr = lambda *_args, **_kwargs: 0.01
    frame = _frame()
    entry = CandidateEntry(
        side="LONG",
        entry_price=1.10,
        entry_min=1.10,
        entry_max=1.10,
        tf="1H",
        reason="RB exact retest",
        zone_low=1.099,
        zone_high=1.10,
        stop_override=1.099,
        lock_entry_range=True,
        trigger_kind="rejection_block_1h",
        lock_stop_override=True,
        entry_order_type="LIMIT_RETEST",
        entry_ttl_min=None,
    )

    payload = strategy._build_signal_payload(
        entry=entry,
        side_bias="LONG",
        narrative_text="test",
        factor_vector={},
        fvg_side="NEUTRAL",
        fvg_text="none",
        df_15M=frame,
        df_1H=frame,
        df_4H=frame,
        symbol="EURUSD",
    )

    assert payload["entry_min"] == pytest.approx(1.10)
    assert payload["entry_max"] == pytest.approx(1.10)
    assert payload["stop_price"] == pytest.approx(1.099)
    assert payload["structural_stop_price"] == pytest.approx(1.099)
    assert payload["stop_mode"] == "structural_exact"
    assert payload["stop_atr_ratio"] == pytest.approx(0.1)
    assert payload["stop_atr_h1"] == pytest.approx(0.1)
    assert payload["lock_stop_override"] is True
    assert payload["entry_order_type"] == "LIMIT_RETEST"
    assert payload["entry_ttl_min"] is None


def test_counterfactual_materialization_preserves_exact_stop_contract():
    strategy = NarrativeStrategy()
    strategy.single_tp_mode = False
    strategy.tp_rr_levels = [1.0, 2.0, 3.0]
    strategy._atr = lambda *_args, **_kwargs: 0.01
    frame = _frame()
    entry = CandidateEntry(
        side="SHORT",
        entry_price=1.10,
        entry_min=1.10,
        entry_max=1.10,
        tf="1H",
        reason="RB exact retest",
        zone_low=1.10,
        zone_high=1.101,
        stop_override=1.101,
        lock_entry_range=True,
        trigger_kind="rejection_block_1h",
        lock_stop_override=True,
        entry_order_type="LIMIT_RETEST",
        entry_ttl_min=None,
    )

    payload = _materialize_entry(
        strategy=strategy,
        entry=entry,
        trigger_kind="rejection_block_1h",
        data={"4H": frame, "1H": frame, "15M": frame},
        symbol="GBPUSD",
        narrative="test",
        factor_vector={},
        fvg_side="NEUTRAL",
        fvg_text="none",
    )

    assert payload["entry_price"] == pytest.approx(1.10)
    assert payload["stop_price"] == pytest.approx(1.101)
    assert payload["stop_mode"] == "structural_exact"
    assert payload["stop_atr_ratio"] == pytest.approx(0.1)
    assert payload["stop_atr_h1"] == pytest.approx(0.1)
    assert payload["entry_order_type"] == "LIMIT_RETEST"
