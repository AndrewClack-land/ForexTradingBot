"""Adversarial Core tests for emergency conflict-position auto-close."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

import main
from core.persistent_limit import (
    PendingLimitLeg,
    PendingLimitPlan,
    PendingLimitState,
)


SYMBOL = "EURUSD"
PLAN_POSITION_ID = 9001
ACTIVE_POSITION_ID = 8000
MAGIC = 4242


def _conflict_plan() -> PendingLimitPlan:
    now = time.time()
    signal = {
        "signal": "ENTER",
        "side": "LONG",
        "entry_price": 1.1000,
        "entry_min": 1.0998,
        "entry_max": 1.1002,
        "stop_price": 1.0950,
        "tp_price": 1.1150,
        "tp_prices": [1.1150],
        "tf": "15M",
        "idea_id": "new-conflicting-limit-idea",
    }
    plan = PendingLimitPlan.create(
        symbol=SYMBOL,
        signal=signal,
        trigger_signature="LONG|pivot_reclaim_h1|conflict-close",
        expires_at=now + 900.0,
        magic=MAGIC,
        legs=(PendingLimitLeg(1, 1.1150, 0.1),),
        now=now,
        plan_id="conflict-autoclose-plan-00000001",
    ).attach_order(1, 101, now=now + 0.1)
    plan = plan.record_fill(
        1,
        0.1,
        1.0999,
        deal_ticket=501,
        position_id=PLAN_POSITION_ID,
        now=now + 0.2,
    )
    assert plan.state is PendingLimitState.FILLED
    return main.Core._mark_pending_conflict(plan)


def _plan_position(plan: PendingLimitPlan) -> dict:
    return {
        "symbol": plan.symbol,
        "ticket": PLAN_POSITION_ID,
        "comment": plan.broker_comment_for_leg(1),
        "magic": MAGIC,
        "volume": 0.1,
    }


def _active_position() -> dict:
    return {
        "symbol": SYMBOL,
        "ticket": ACTIVE_POSITION_ID,
        "comment": "EXISTING:active-idea",
        "magic": MAGIC,
        "volume": 0.2,
    }


class _AutoCloseExecutor:
    def __init__(
        self,
        plan: PendingLimitPlan,
        *,
        positions: list[dict] | None = None,
        capabilities: dict | None = None,
        close_outcomes: list[object] | None = None,
        executor_magic: int = MAGIC,
    ) -> None:
        self.plan = plan
        self.settings = SimpleNamespace(magic=executor_magic)
        self.positions = [
            dict(row)
            for row in (
                positions
                if positions is not None
                else [_active_position(), _plan_position(plan)]
            )
        ]
        self.capabilities = dict(
            capabilities
            if capabilities is not None
            else {"ready": True, "account_hedging": True}
        )
        self.close_outcomes = list(close_outcomes or [True])
        self.close_calls: list[dict] = []
        self.capability_calls: list[str] = []
        self.cancel_calls: list[int] = []

    @staticmethod
    def list_pending_orders() -> list[dict]:
        return []

    def list_positions(self) -> list[dict]:
        return [dict(row) for row in self.positions]

    @staticmethod
    def get_pending_order_fills(
        _tickets,
        *,
        comments,
        lookback_days,
    ) -> dict:
        assert comments
        assert lookback_days >= 1
        return {}

    @staticmethod
    def get_pending_order_histories(
        comments,
        *,
        lookback_days,
    ) -> dict:
        assert comments
        assert lookback_days >= 1
        return {}

    def pending_limit_capabilities(self, symbol: str) -> dict:
        self.capability_calls.append(symbol)
        return dict(self.capabilities)

    def close_trade(
        self,
        symbol,
        *,
        position_id,
        volume,
        expected_comment,
    ) -> bool:
        self.close_calls.append(
            {
                "symbol": symbol,
                "position_id": int(position_id),
                "volume": volume,
                "expected_comment": expected_comment,
            }
        )
        outcome = self.close_outcomes.pop(0) if self.close_outcomes else True
        if isinstance(outcome, BaseException):
            raise outcome
        return bool(outcome)

    def cancel_pending_order(self, ticket, *, expected_comment_prefix):
        self.cancel_calls.append(int(ticket))
        raise AssertionError(
            "a fully filled leg has no pending remainder to cancel: "
            f"{expected_comment_prefix}"
        )


def _core(plan: PendingLimitPlan, executor: _AutoCloseExecutor):
    core = main.Core.__new__(main.Core)
    core.mt5_executor = executor
    core.pending_limits = {plan.symbol: plan}
    core._pending_state_safe = True
    core._pending_reconcile_error_ts = 0.0
    core._pending_entry_events = {}
    core._management_lock = threading.RLock()
    existing_idea = SimpleNamespace(
        idea_id="existing-active-idea",
        side="SHORT",
        symbol=plan.symbol,
    )
    core.active_trades = {plan.symbol: existing_idea}
    core.universe = {plan.symbol: plan.symbol}
    core.shadow_tick_watcher = None
    core._capture_quote = lambda _symbol: None
    core._materialize_pending_trade = lambda *_args, **_kwargs: pytest.fail(
        "a conflict plan must never merge into the existing active idea"
    )
    snapshots: list[dict] = []

    def persist_snapshot() -> None:
        snapshots.append(
            {
                symbol: pending.to_dict()
                for symbol, pending in core.pending_limits.items()
            }
        )

    core._save_pending_limit_state = persist_snapshot
    return core, existing_idea, snapshots


def test_autoclose_flag_off_never_probes_or_closes(monkeypatch):
    monkeypatch.setattr(
        main,
        "PERSISTENT_LIMIT_CONFLICT_AUTOCLOSE_ENABLED",
        False,
    )
    plan = _conflict_plan()
    executor = _AutoCloseExecutor(plan)
    core, existing_idea, _snapshots = _core(plan, executor)

    core._manage_pending_limits()

    assert executor.capability_calls == []
    assert executor.close_calls == []
    assert core.pending_limits[SYMBOL].reason == main._PENDING_CONFLICT_REASON
    assert core.active_trades[SYMBOL] is existing_idea


def test_exact_recorded_identity_closes_only_plan_position(monkeypatch):
    monkeypatch.setattr(
        main,
        "PERSISTENT_LIMIT_CONFLICT_AUTOCLOSE_ENABLED",
        True,
    )
    plan = _conflict_plan()
    executor = _AutoCloseExecutor(plan, close_outcomes=[True])
    core, existing_idea, snapshots = _core(plan, executor)

    core._manage_pending_limits()

    assert executor.capability_calls == [SYMBOL]
    assert executor.close_calls == [
        {
            "symbol": SYMBOL,
            "position_id": PLAN_POSITION_ID,
            "volume": None,
            "expected_comment": plan.broker_comment_for_leg(1),
        }
    ]
    assert all(
        call["position_id"] != ACTIVE_POSITION_ID
        for call in executor.close_calls
    )
    assert core.active_trades[SYMBOL] is existing_idea
    assert core.pending_limits[SYMBOL].reason == (
        main._PENDING_CONFLICT_CLOSE_REQUESTED_REASON
    )
    assert snapshots[-1][SYMBOL]["reason"] == (
        main._PENDING_CONFLICT_CLOSE_REQUESTED_REASON
    )


def _wrong_identity_case(
    plan: PendingLimitPlan,
    case: str,
) -> tuple[list[dict], int]:
    candidate = _plan_position(plan)
    executor_magic = MAGIC
    if case == "recorded_id":
        candidate["ticket"] = PLAN_POSITION_ID + 99
    elif case == "comment":
        candidate["comment"] = "FBPL:not-this-plan:T1"
    elif case == "symbol":
        candidate["symbol"] = "GBPUSD"
    elif case == "position_magic":
        candidate["magic"] = MAGIC + 1
    elif case == "executor_magic":
        executor_magic = MAGIC + 1
    else:  # pragma: no cover - test parameter contract
        raise AssertionError(case)
    return [_active_position(), candidate], executor_magic


@pytest.mark.parametrize(
    "case",
    [
        "recorded_id",
        "comment",
        "symbol",
        "position_magic",
        "executor_magic",
    ],
)
def test_any_wrong_identity_never_calls_close(monkeypatch, case):
    monkeypatch.setattr(
        main,
        "PERSISTENT_LIMIT_CONFLICT_AUTOCLOSE_ENABLED",
        True,
    )
    plan = _conflict_plan()
    positions, executor_magic = _wrong_identity_case(plan, case)
    executor = _AutoCloseExecutor(
        plan,
        positions=positions,
        executor_magic=executor_magic,
    )
    core, existing_idea, _snapshots = _core(plan, executor)

    core._manage_pending_limits()

    assert executor.close_calls == []
    assert SYMBOL in core.pending_limits
    assert core._pending_state_safe is False
    assert core.active_trades[SYMBOL] is existing_idea


@pytest.mark.parametrize(
    "first_outcome",
    [RuntimeError("broker close transport failure"), False],
    ids=["exception", "false"],
)
def test_failed_close_retries_on_next_manager_tick_without_30s_delay(
    monkeypatch,
    first_outcome,
):
    monkeypatch.setattr(
        main,
        "PERSISTENT_LIMIT_CONFLICT_AUTOCLOSE_ENABLED",
        True,
    )
    plan = _conflict_plan()
    executor = _AutoCloseExecutor(
        plan,
        close_outcomes=[first_outcome, True],
    )
    core, existing_idea, _snapshots = _core(plan, executor)

    core._manage_pending_limits()
    assert len(executor.close_calls) == 1
    assert core.pending_limits[SYMBOL].reason == (
        main._PENDING_CONFLICT_CLOSE_FAILED_REASON
    )

    # No clock advance: failed attempts are retryable on the very next fast
    # manager tick.  The 30-second publication grace applies only to success.
    core._manage_pending_limits()

    assert len(executor.close_calls) == 2
    assert core.pending_limits[SYMBOL].reason == (
        main._PENDING_CONFLICT_CLOSE_REQUESTED_REASON
    )
    assert core.active_trades[SYMBOL] is existing_idea


def test_success_waits_for_absence_then_persists_manual_recovery_state(
    monkeypatch,
):
    monkeypatch.setattr(
        main,
        "PERSISTENT_LIMIT_CONFLICT_AUTOCLOSE_ENABLED",
        True,
    )
    plan = _conflict_plan()
    executor = _AutoCloseExecutor(plan, close_outcomes=[True, True])
    core, existing_idea, snapshots = _core(plan, executor)

    core._manage_pending_limits()
    assert len(executor.close_calls) == 1
    assert SYMBOL in core.pending_limits
    assert core.pending_limits[SYMBOL].reason == (
        main._PENDING_CONFLICT_CLOSE_REQUESTED_REASON
    )

    # Broker still publishes the just-closed position. Success must not be
    # retried or interpreted as confirmed absence during the grace interval.
    core._manage_pending_limits()
    assert len(executor.close_calls) == 1
    assert core.pending_limits[SYMBOL].reason == (
        main._PENDING_CONFLICT_CLOSE_REQUESTED_REASON
    )

    executor.positions = [_active_position()]
    core._manage_pending_limits()

    resolved = core.pending_limits[SYMBOL]
    assert resolved.reason == main._PENDING_CONFLICT_CLOSE_RESOLVED_REASON
    assert "manual recovery required" in resolved.reason
    assert snapshots[-1][SYMBOL]["reason"] == resolved.reason
    assert core.active_trades[SYMBOL] is existing_idea

    core._manage_pending_limits()
    assert SYMBOL in core.pending_limits
    assert executor.close_calls[0]["position_id"] == PLAN_POSITION_ID
    assert len(executor.close_calls) == 1
    assert core.active_trades[SYMBOL] is existing_idea


def test_capability_not_ready_never_calls_close(monkeypatch):
    monkeypatch.setattr(
        main,
        "PERSISTENT_LIMIT_CONFLICT_AUTOCLOSE_ENABLED",
        True,
    )
    plan = _conflict_plan()
    executor = _AutoCloseExecutor(
        plan,
        capabilities={"ready": False, "account_hedging": True},
    )
    core, existing_idea, _snapshots = _core(plan, executor)

    core._manage_pending_limits()

    assert executor.capability_calls == [SYMBOL]
    assert executor.close_calls == []
    assert SYMBOL in core.pending_limits
    assert core._pending_state_safe is False
    assert core.active_trades[SYMBOL] is existing_idea
