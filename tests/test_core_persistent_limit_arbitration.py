"""Public-path contracts for market versus persistent-LIMIT arbitration."""

from __future__ import annotations

import sys
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pandas as pd
import pytest

import main
from core.persistent_limit import (
    PendingLimitLeg,
    PendingLimitPlan,
    PendingLimitState,
)
from core.rb_consumed_events import RBConsumedEventStore
from core.shadow_tick_watcher import QuoteTick
from core.strategy_narrative import ActiveTrade


SYMBOL = "EURUSD"
PLANNED_ENTRY = 1.1000
ENTRY_MIN = 1.0998
ENTRY_MAX = 1.1002


class _Profiler:
    @staticmethod
    def section(_name):
        return nullcontext()

    @staticmethod
    def dump(*, prefix):
        assert prefix == "[Profiler:core]"


class _Executor:
    def __init__(self):
        self.settings = SimpleNamespace(magic=4242)
        self.prepared: list[dict] = []
        self.placements: list[dict] = []

    @staticmethod
    def connection_alive() -> bool:
        return True

    @staticmethod
    def get_current_price(_symbol, _side):
        return PLANNED_ENTRY

    def prepare_limit_entry(
        self,
        symbol,
        *,
        side,
        limit_price,
        stop_price,
    ) -> dict:
        self.prepared.append(
            {
                "symbol": symbol,
                "side": side,
                "limit_price": limit_price,
                "stop_price": stop_price,
            }
        )
        return {
            "volume": 0.1,
            "volume_step": 0.01,
            "volume_min": 0.01,
            "risk_budget_amount": 100.0,
        }

    @staticmethod
    def account_is_hedging() -> bool:
        return True

    @staticmethod
    def pending_limit_capabilities(_symbol) -> dict:
        return {
            "limit_allowed": True,
            "specified_expiration": True,
            "account_hedging": True,
            "ready": True,
        }

    def place_limit_leg(self, symbol, **request) -> dict:
        self.placements.append({"symbol": symbol, **request})
        return {
            "ticket": 7000 + len(self.placements),
            "deal": 0,
            "risk_amount": 10.0,
        }


def _signal(side: str) -> dict:
    side = side.upper()
    return {
        "signal": "ENTER",
        "side": side,
        "entry_price": PLANNED_ENTRY,
        "entry_min": ENTRY_MIN,
        "entry_max": ENTRY_MAX,
        "stop_price": 1.0950 if side == "LONG" else 1.1050,
        "tp_price": 1.1150 if side == "LONG" else 1.0850,
        "tp_prices": [1.1150 if side == "LONG" else 1.0850],
        "tf": "15M",
        "narrative": "market/LIMIT arbitration fixture",
        "trigger_kind": "pivot_reclaim_h1",
        "trigger_event_id": f"arbitration-{side.lower()}",
    }


def _exact_rb_signal(side: str) -> dict:
    signal = _signal(side)
    side_key = side.upper()
    signal.update(
        {
            "entry_min": PLANNED_ENTRY,
            "entry_max": PLANNED_ENTRY,
            "entry_order_type": "LIMIT_RETEST",
            "lock_entry_range": True,
            "lock_stop_override": True,
            "tf": "1H",
            "trigger_kind": "rejection_block_1h",
            "trigger_event_id": (
                "rb:h1:touch:v1:EURUSD:"
                f"{side_key.lower()}:2026-08-24T08:00:00+00:00"
            ),
        }
    )
    return signal


def _active_trade(side: str) -> ActiveTrade:
    signal = _signal(side)
    return ActiveTrade(
        side=side,
        entry=PLANNED_ENTRY,
        stop=signal["stop_price"],
        tp_prices=signal["tp_prices"],
        tf="15M",
        narrative="market/LIMIT arbitration fixture",
        symbol=SYMBOL,
    )


def _core(
    monkeypatch,
    *,
    side: str,
    quote: QuoteTick | None,
    pending_plan: PendingLimitPlan | None = None,
    rb_state_path=None,
):
    monkeypatch.setattr(main, "PERSISTENT_LIMIT_ENABLED", True)
    monkeypatch.setattr(
        main,
        "PERSISTENT_LIMIT_SYMBOLS",
        frozenset({SYMBOL}),
    )
    monkeypatch.setattr(main, "PARTIAL_TP_MODE", "monitor")
    monkeypatch.setattr(main, "save_active_trades", lambda *_args: None)
    mt5 = sys.modules["MetaTrader5"]
    monkeypatch.setattr(mt5, "positions_get", lambda *, symbol: ())

    executor = _Executor()
    market_calls: list[dict] = []
    core = main.Core.__new__(main.Core)
    core.mt5_executor = executor
    core.pending_limits = (
        {SYMBOL: pending_plan} if pending_plan is not None else {}
    )
    core._pending_state_safe = True
    core._rb_consumed_events = RBConsumedEventStore()
    core._rb_consumed_events_safe = True
    core._rb_consumed_events_lock = threading.RLock()
    if rb_state_path is not None:
        monkeypatch.setattr(
            main,
            "RB_H1_CONSUMED_EVENTS_PATH",
            rb_state_path,
        )
    core._pending_entry_events = {}
    core._pending_reconcile_error_ts = 0.0
    core._management_lock = threading.RLock()
    core.active_trades = {}
    core._entries_today = {}
    core._trigger_signatures = {}
    core._entry_cooldowns = {}
    core._last_reconnect_ts = 0.0
    core._executor_retry_ts = 0.0
    core._entry_cooldown_sec = 300.0
    core._stale_cooldown_sec = 60.0
    core.shadow_tick_watcher = None
    core.universe = {SYMBOL: SYMBOL}
    core.global_context = {
        "session": "ALL",
        "session_allowed": True,
    }
    core.TIME_BUDGET_SEC = 60.0
    core.profiler = _Profiler()
    core.log_tick = False
    core.strategy = SimpleNamespace(
        generate_signal=lambda _data, *, symbol: dict(_signal(side))
    )
    core.ai = SimpleNamespace(
        on_signal=lambda _symbol, signal, _data, _active: dict(signal)
    )
    core._get_symbols = lambda: [SYMBOL]
    core._update_global_context = lambda: None
    # A reconciliation failure/deferral must not authorize a fresh order.  The
    # pending-plan test deliberately keeps the durable book unchanged here.
    core._manage_pending_limits = lambda: None
    core._record_shadow_scan_raw = lambda *_args, **_kwargs: None
    frame = pd.DataFrame({"close": [PLANNED_ENTRY]})
    core._build_tf_data = lambda _symbol: {"1H": frame}
    core._strategy_view = lambda data, *, symbol: data
    core._apply_global_filters = lambda _symbol, signal: signal
    core._apply_session_filter = lambda _symbol, signal: signal
    core._apply_vol_regime_filter = (
        lambda _symbol, signal, _data: signal
    )
    core._attach_shadow_score = lambda _symbol, signal: signal
    core._capture_quote = lambda _symbol: quote
    core._save_pending_limit_state = lambda: None
    core._log_signal = lambda *_args: None

    def execute_market(symbol: str, signal: dict) -> ActiveTrade:
        market_calls.append({"symbol": symbol, "signal": dict(signal)})
        return _active_trade(side)

    core._execute_entry_signal = execute_market
    return core, executor, market_calls


@pytest.mark.parametrize(
    ("side", "quote"),
    [
        ("LONG", QuoteTick(1_000, bid=1.1002, ask=1.1003)),
        ("SHORT", QuoteTick(1_000, bid=1.0997, ask=1.0998)),
    ],
)
def test_approach_side_arms_native_limit_at_exact_planned_entry(
    monkeypatch,
    side,
    quote,
):
    core, executor, market_calls = _core(
        monkeypatch,
        side=side,
        quote=quote,
    )

    result = core.get_signals()[SYMBOL]

    assert result["signal"] == "PENDING_LIMIT"
    assert result["pending_limit"]["limit_price"] == pytest.approx(
        PLANNED_ENTRY
    )
    assert core.pending_limits[SYMBOL].limit_price == pytest.approx(
        PLANNED_ENTRY
    )
    assert executor.prepared[0]["limit_price"] == pytest.approx(
        PLANNED_ENTRY
    )
    assert executor.placements[0]["limit_price"] == pytest.approx(
        PLANNED_ENTRY
    )
    assert market_calls == []


def test_signal_specific_ttl_reaches_native_pending_plan(monkeypatch):
    core, executor, market_calls = _core(
        monkeypatch,
        side="LONG",
        quote=QuoteTick(1_500, bid=1.1002, ask=1.1003),
    )
    signal = _signal("LONG")
    signal["entry_ttl_min"] = 30
    core.strategy = SimpleNamespace(
        generate_signal=lambda _data, *, symbol: dict(signal)
    )

    result = core.get_signals()[SYMBOL]
    plan = core.pending_limits[SYMBOL]

    assert result["signal"] == "PENDING_LIMIT"
    assert plan.expires_at - plan.created_at == pytest.approx(
        30 * 60,
        abs=1.0,
    )
    assert executor.placements[0]["expires_at"] == pytest.approx(
        plan.expires_at
    )
    assert market_calls == []


def test_exact_limit_retest_never_falls_through_when_symbol_not_routed(
    monkeypatch,
):
    core, executor, market_calls = _core(
        monkeypatch,
        side="LONG",
        quote=QuoteTick(1_600, bid=1.0999, ask=1.1000),
    )
    signal = _signal("LONG")
    signal["entry_order_type"] = "LIMIT_RETEST"
    core.strategy = SimpleNamespace(
        generate_signal=lambda _data, *, symbol: dict(signal)
    )
    monkeypatch.setattr(main, "PERSISTENT_LIMIT_SYMBOLS", frozenset())

    result = core.get_signals()[SYMBOL]

    assert result["signal"] == "WAIT_LIMIT_UNSUPPORTED"
    assert "cannot fall through to a market entry" in result["info"]
    assert executor.placements == []
    assert market_calls == []


@pytest.mark.parametrize(
    ("side", "quote"),
    [
        ("LONG", QuoteTick(2_000, bid=1.0999, ask=1.1000)),
        ("SHORT", QuoteTick(2_000, bid=1.1000, ask=1.1001)),
    ],
)
def test_price_inside_two_sided_zone_keeps_existing_market_entry(
    monkeypatch,
    side,
    quote,
):
    core, executor, market_calls = _core(
        monkeypatch,
        side=side,
        quote=quote,
    )

    result = core.get_signals()[SYMBOL]

    assert result["signal"] == "ENTER"
    assert len(market_calls) == 1
    assert executor.placements == []
    assert core.pending_limits == {}


@pytest.mark.parametrize(
    ("side", "quote"),
    [
        ("LONG", QuoteTick(3_000, bid=1.0996, ask=1.0997)),
        ("SHORT", QuoteTick(3_000, bid=1.1003, ask=1.1004)),
    ],
)
def test_price_already_beyond_zone_rejects_gap_market_fill(
    monkeypatch,
    side,
    quote,
):
    core, executor, market_calls = _core(
        monkeypatch,
        side=side,
        quote=quote,
    )

    result = core.get_signals()[SYMBOL]

    assert result["signal"] == "WAIT_LIMIT_INVALIDATED"
    assert "gap fill rejected" in result["info"]
    assert market_calls == []
    assert executor.placements == []
    assert core.pending_limits == {}


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_missing_post_gate_causal_quote_fails_closed(
    monkeypatch,
    side,
):
    core, executor, market_calls = _core(
        monkeypatch,
        side=side,
        quote=None,
    )

    result = core.get_signals()[SYMBOL]

    assert result["signal"] == "WAIT_LIMIT_QUOTE"
    assert "fail-closed" in result["info"]
    assert market_calls == []
    assert executor.placements == []
    assert core.pending_limits == {}


@pytest.mark.parametrize(
    ("side", "quote"),
    [
        ("LONG", QuoteTick(5_000, bid=1.1002, ask=1.1003)),
        ("SHORT", QuoteTick(5_000, bid=1.0997, ask=1.0998)),
    ],
)
def test_exact_rb_approach_consumes_before_native_limit_placement(
    monkeypatch,
    tmp_path,
    side,
    quote,
):
    state_path = tmp_path / "rb_consumed.json"
    core, executor, market_calls = _core(
        monkeypatch,
        side=side,
        quote=quote,
        rb_state_path=state_path,
    )
    signal = _exact_rb_signal(side)
    core.strategy = SimpleNamespace(
        generate_signal=lambda _data, *, symbol: dict(signal)
    )
    original_place = executor.place_limit_leg

    def assert_reserved_before_place(symbol, **request):
        assert state_path.exists()
        assert core._rb_consumed_events.is_consumed(
            signal["trigger_event_id"]
        )
        return original_place(symbol, **request)

    executor.place_limit_leg = assert_reserved_before_place

    result = core.get_signals()[SYMBOL]

    assert result["signal"] == "PENDING_LIMIT"
    assert len(executor.placements) == 1
    assert market_calls == []
    assert core._rb_consumed_events.is_consumed(
        signal["trigger_event_id"]
    )


@pytest.mark.parametrize(
    ("side", "quote"),
    [
        ("LONG", QuoteTick(6_000, bid=1.0999, ask=1.1000)),
        ("SHORT", QuoteTick(6_000, bid=1.1000, ask=1.1001)),
    ],
)
def test_exact_rb_first_touch_never_becomes_market_order(
    monkeypatch,
    tmp_path,
    side,
    quote,
):
    core, executor, market_calls = _core(
        monkeypatch,
        side=side,
        quote=quote,
        rb_state_path=tmp_path / "rb_consumed.json",
    )
    signal = _exact_rb_signal(side)
    core.strategy = SimpleNamespace(
        generate_signal=lambda _data, *, symbol: dict(signal)
    )

    first = core.get_signals()[SYMBOL]
    repeated = core.get_signals()[SYMBOL]

    assert first["signal"] == "WAIT_LIMIT_FIRST_TOUCH"
    assert repeated["signal"] == "SKIP_CONSUMED_RB_EVENT"
    assert core._rb_consumed_events.is_consumed(
        signal["trigger_event_id"]
    )
    assert executor.placements == []
    assert market_calls == []
    assert core.pending_limits == {}


@pytest.mark.parametrize(
    ("side", "quote"),
    [
        ("LONG", QuoteTick(7_000, bid=1.0997, ask=1.0998)),
        ("SHORT", QuoteTick(7_000, bid=1.1002, ask=1.1003)),
    ],
)
def test_exact_rb_crossed_before_arm_is_consumed_without_market(
    monkeypatch,
    tmp_path,
    side,
    quote,
):
    core, executor, market_calls = _core(
        monkeypatch,
        side=side,
        quote=quote,
        rb_state_path=tmp_path / "rb_consumed.json",
    )
    signal = _exact_rb_signal(side)
    core.strategy = SimpleNamespace(
        generate_signal=lambda _data, *, symbol: dict(signal)
    )

    result = core.get_signals()[SYMBOL]

    assert result["signal"] == "WAIT_LIMIT_INVALIDATED"
    assert core._rb_consumed_events.is_consumed(
        signal["trigger_event_id"]
    )
    assert executor.placements == []
    assert market_calls == []
    assert core.pending_limits == {}


def test_exact_rb_ledger_write_failure_sends_no_broker_order(
    monkeypatch,
    tmp_path,
):
    core, executor, market_calls = _core(
        monkeypatch,
        side="LONG",
        quote=QuoteTick(8_000, bid=1.1002, ask=1.1003),
        rb_state_path=tmp_path / "rb_consumed.json",
    )
    signal = _exact_rb_signal("LONG")
    core.strategy = SimpleNamespace(
        generate_signal=lambda _data, *, symbol: dict(signal)
    )

    def fail_save(_store, _path):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(RBConsumedEventStore, "save", fail_save)

    result = core.get_signals()[SYMBOL]

    assert result["signal"] == "WAIT_RB_EVENT_STATE"
    assert core._rb_consumed_events_safe is False
    assert executor.placements == []
    assert market_calls == []
    assert core.pending_limits == {}


def _durable_plan(*, filled_unacked: bool) -> PendingLimitPlan:
    now = time.time()
    signal = _signal("LONG")
    plan = PendingLimitPlan.create(
        symbol=SYMBOL,
        signal=signal,
        trigger_signature="LONG|pivot_reclaim_h1|existing",
        expires_at=now + 900.0,
        magic=4242,
        legs=(
            PendingLimitLeg(
                index=1,
                target_price=signal["tp_price"],
                requested_volume=0.1,
            ),
        ),
        now=now,
        plan_id=(
            "existing-filled-unacked-plan"
            if filled_unacked
            else "existing-working-plan"
        ),
    ).attach_order(1, 101, now=now + 0.1)
    if not filled_unacked:
        assert plan.state is PendingLimitState.PLACED
        return plan
    filled = plan.record_fill(
        1,
        0.1,
        PLANNED_ENTRY,
        deal_ticket=501,
        position_id=9001,
        now=now + 0.2,
    )
    assert filled.state is PendingLimitState.FILLED
    assert filled.entry_announcement_pending
    return filled


@pytest.mark.parametrize("filled_unacked", [False, True])
def test_any_durable_plan_blocks_a_second_fresh_entry(
    monkeypatch,
    filled_unacked,
):
    plan = _durable_plan(filled_unacked=filled_unacked)
    core, executor, market_calls = _core(
        monkeypatch,
        side="LONG",
        quote=QuoteTick(4_000, bid=1.0999, ask=1.1000),
        pending_plan=plan,
    )

    result = core.get_signals()[SYMBOL]

    assert market_calls == [], (
        "a durable pending-plan identity must reserve the symbol until Core "
        "reconciliation explicitly retires it"
    )
    assert executor.placements == []
    assert result["signal"] != "ENTER"
    assert core.pending_limits[SYMBOL].plan_id == plan.plan_id
