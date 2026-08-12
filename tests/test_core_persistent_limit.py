"""Focused Core integration contract for persistent exact-retest LIMITs.

The tests instantiate ``Core`` without running its constructor and use broker
fakes only.  They pin the unsafe boundaries between durable intent, MT5
ownership reconciliation, and promotion into ``ActiveTrade``.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import main
from core.persistent_limit import (
    BROKER_COMMENT_PREFIX,
    PendingLimitLeg,
    PendingLimitPlan,
    PendingLimitState,
)
from executors.mt5_executor import PendingOrderRejected


NOW = 2_000_000_000.0
PLAN_ID = "core-limit-plan-00000000000001"


def _signal() -> dict:
    return {
        "signal": "ENTER",
        "side": "LONG",
        "entry_price": 1.1000,
        "entry_min": 1.0998,
        "entry_max": 1.1002,
        "stop_price": 1.0950,
        "tp_price": 1.1150,
        "tp_prices": [1.1050, 1.1100, 1.1150],
        "tf": "15M",
        "narrative": "focused integration fixture",
        "idea_id": "idea-core-limit-0001",
    }


def _plan(*, attach_tickets: bool) -> PendingLimitPlan:
    signal = _signal()
    plan = PendingLimitPlan.create(
        symbol="EURUSD",
        signal=signal,
        trigger_signature="LONG|pivot|EURUSD|closed-M15",
        expires_at=NOW + 3600.0,
        magic=4242,
        legs=tuple(
            PendingLimitLeg(
                index=index,
                target_price=target,
                requested_volume=0.1,
            )
            for index, target in enumerate(signal["tp_prices"], start=1)
        ),
        now=NOW,
        plan_id=PLAN_ID,
    )
    if attach_tickets:
        for offset, (index, ticket) in enumerate(
            ((1, 101), (2, 102), (3, 103)),
            start=1,
        ):
            plan = plan.attach_order(index, ticket, now=NOW + offset)
    return plan


def _core(executor, *, plans=None):
    core = main.Core.__new__(main.Core)
    core.universe = {"EURUSD": "EURUSD"}
    core.mt5_executor = executor
    core.pending_limits = dict(plans or {})
    core._pending_state_safe = True
    core._pending_entry_events = {}
    core._pending_reconcile_error_ts = 0.0
    core._management_lock = threading.RLock()
    core.active_trades = {}
    core._entries_today = {}
    core._trigger_signatures = {}
    core._entry_cooldowns = {}
    core.shadow_tick_watcher = None
    core._pending_limit_capability_cache = {}
    core._roll_daily_counters = lambda: None
    core._capture_quote = lambda _symbol: None
    return core


class PlacementExecutor:
    def __init__(self):
        self.settings = SimpleNamespace(magic=4242)
        self.core = None
        self.snapshots = None
        self.calls = []

    def prepare_limit_entry(self, symbol, *, side, limit_price, stop_price):
        assert (symbol, side) == ("EURUSD", "LONG")
        assert limit_price == pytest.approx(1.1000)
        assert stop_price == pytest.approx(1.0950)
        return {
            "volume": 0.30,
            "volume_step": 0.01,
            "volume_min": 0.01,
            "risk_budget_amount": 100.0,
        }

    @staticmethod
    def account_is_hedging():
        return True

    def place_limit_leg(self, symbol, **request):
        # Every unsafe broker call must have an exactly matching durable
        # snapshot already on disk, including tickets returned by earlier legs.
        current = self.core.pending_limits[symbol]
        assert self.snapshots
        assert self.snapshots[-1][symbol] == current.to_dict()
        self.calls.append((symbol, dict(request)))
        index = len(self.calls)
        return {
            "ticket": 7000 + index,
            "deal": 0,
            "risk_amount": 30.0,
        }


class DefinitiveRejectExecutor(PlacementExecutor):
    def place_limit_leg(self, symbol, **request):
        current = self.core.pending_limits[symbol]
        assert self.snapshots[-1][symbol] == current.to_dict()
        self.calls.append((symbol, dict(request)))
        raise PendingOrderRejected(
            "MT5 pending order rejected "
            "(retcode=10022, comment='Invalid expiration')",
            retcode=10022,
            broker_comment="Invalid expiration",
            submitted=True,
        )


class ReconcileExecutor:
    def __init__(
        self,
        *,
        live_orders,
        fills=None,
        positions=None,
        close_results=None,
        capabilities=None,
    ):
        self.settings = SimpleNamespace(magic=4242)
        self.live_orders = [dict(row) for row in live_orders]
        self.fills = dict(fills or {})
        self.positions = [dict(row) for row in (positions or [])]
        self.cancelled = []
        self.fill_queries = []
        self.placement_calls = 0
        self.close_results = list(close_results or [])
        self.close_calls = []
        self.capabilities = dict(
            capabilities
            or {
                "limit_allowed": True,
                "specified_expiration": True,
                "account_hedging": True,
                "ready": True,
            }
        )

    def list_pending_orders(self):
        return [dict(row) for row in self.live_orders]

    def list_positions(self):
        return [dict(row) for row in self.positions]

    def get_pending_order_fills(self, tickets, *, comments, lookback_days):
        self.fill_queries.append(
            (tuple(tickets), tuple(comments), int(lookback_days))
        )
        return self.fills

    def cancel_pending_order(self, ticket, *, expected_comment_prefix):
        assert expected_comment_prefix.startswith(BROKER_COMMENT_PREFIX)
        self.cancelled.append(int(ticket))
        self.live_orders = [
            row
            for row in self.live_orders
            if int(row.get("ticket") or 0) != int(ticket)
        ]
        return True

    def pending_limit_capabilities(self, _symbol):
        return dict(self.capabilities)

    def close_trade(
        self,
        symbol,
        *,
        position_id,
        volume,
        expected_comment=None,
    ):
        self.close_calls.append(
            (
                str(symbol),
                int(position_id),
                volume,
                str(expected_comment or ""),
            )
        )
        if not self.close_results:
            raise AssertionError("unexpected emergency close")
        result = self.close_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return bool(result)

    @staticmethod
    def get_pending_order_history(_ticket):
        return {"terminal": True, "fill_expected": False}

    def place_limit_leg(self, *_args, **_kwargs):
        self.placement_calls += 1
        raise AssertionError("reconciliation must never place a duplicate")


def _conflict_runtime(
    plan,
    *,
    live_orders,
    fills,
    positions,
    close_results,
    capabilities=None,
):
    executor = ReconcileExecutor(
        live_orders=live_orders,
        fills=fills,
        positions=positions,
        close_results=close_results,
        capabilities=capabilities,
    )
    core = _core(executor, plans={plan.symbol: plan})
    existing_trade = SimpleNamespace(idea_id="different-live-idea")
    core.active_trades[plan.symbol] = existing_trade
    snapshots = []
    core._save_pending_limit_state = lambda: snapshots.append(
        {
            symbol: pending.to_dict()
            for symbol, pending in core.pending_limits.items()
        }
    )
    core._materialize_pending_trade = (
        lambda *_args, **_kwargs: pytest.fail(
            "a conflicting fill must never be materialized"
        )
    )
    return core, executor, existing_trade, snapshots


def _opening_fill(order_ticket, position_id, comment, *, leg=1):
    return {
        "order_ticket": order_ticket,
        "deals": [
            {
                "deal_ticket": 500 + leg,
                "position_id": position_id,
                "volume": 0.1,
                "price": 1.0999,
                "time_msc": int((NOW + 10.0 + leg) * 1000),
                "comment": comment,
            }
        ],
    }


def test_arm_persists_exact_intent_before_every_broker_placement(monkeypatch):
    executor = PlacementExecutor()
    core = _core(executor)
    snapshots = []
    core._save_pending_limit_state = lambda: snapshots.append(
        {
            symbol: plan.to_dict()
            for symbol, plan in core.pending_limits.items()
        }
    )
    executor.core = core
    executor.snapshots = snapshots
    monkeypatch.setattr(main, "PARTIAL_TP_MODE", "split")

    observed_at = datetime.fromtimestamp(NOW, tz=timezone.utc)
    plan = core._arm_pending_limit(
        "EURUSD",
        _signal(),
        trigger_signature="LONG|pivot|EURUSD|closed-M15",
        expires_at=NOW + 900.0,
        shadow_plan_id=PLAN_ID,
        observed_at=observed_at,
    )

    assert len(executor.calls) == 3
    assert snapshots[0]["EURUSD"]["state"] == "placing"
    assert all(
        leg["broker_order_ticket"] is None
        for leg in snapshots[0]["EURUSD"]["legs"]
    )
    assert plan.state is PendingLimitState.PLACED
    assert plan.limit_price == pytest.approx(1.1000)
    assert plan.expires_at == NOW + 900.0
    assert plan.working_order_tickets == (7001, 7002, 7003)
    assert all(
        request["limit_price"] == pytest.approx(plan.limit_price)
        and request["expires_at"] == plan.expires_at
        and request["comment"]
        == plan.broker_comment_for_leg(index)
        for index, (_symbol, request) in enumerate(executor.calls, start=1)
    )
    assert core._trigger_signatures["EURUSD"] == {
        "LONG|pivot|EURUSD|closed-M15"
    }


def test_definitive_rejection_is_failed_and_retired_without_ttl_wait(
    monkeypatch,
):
    executor = DefinitiveRejectExecutor()
    core = _core(executor)
    snapshots = []
    core._save_pending_limit_state = lambda: snapshots.append(
        {
            symbol: plan.to_dict()
            for symbol, plan in core.pending_limits.items()
        }
    )
    executor.core = core
    executor.snapshots = snapshots
    monkeypatch.setattr(main, "PARTIAL_TP_MODE", "split")
    monkeypatch.setattr(main.time, "time", lambda: NOW + 1.0)

    with pytest.raises(PendingOrderRejected, match="retcode=10022"):
        core._arm_pending_limit(
            "EURUSD",
            _signal(),
            trigger_signature="LONG|pivot|EURUSD|closed-M15",
            expires_at=NOW + 900.0,
            shadow_plan_id=PLAN_ID,
            observed_at=datetime.fromtimestamp(NOW, tz=timezone.utc),
        )

    failed = core.pending_limits["EURUSD"]
    assert failed.state is PendingLimitState.FAILED
    assert failed.reason.startswith("definitive pending rejection: ")
    assert failed.legs[0].submission_attempted
    assert failed.working_order_tickets == ()

    core.mt5_executor = ReconcileExecutor(
        live_orders=[],
        fills={},
        positions=[],
    )
    core._manage_pending_limits()
    assert core.pending_limits == {}


def test_partial_fill_cancels_siblings_then_materializes_active_trade(monkeypatch):
    plan = _plan(attach_tickets=True)
    comments = {
        index: plan.broker_comment_for_leg(index) for index in (1, 2, 3)
    }
    executor = ReconcileExecutor(
        live_orders=[
            {"ticket": 102, "comment": comments[2]},
            {"ticket": 103, "comment": comments[3]},
        ],
        fills={
            101: {
                "order_ticket": 101,
                "deals": [
                    {
                        "deal_ticket": 501,
                        "position_id": 9001,
                        "volume": 0.1,
                        "price": 1.0999,
                        "comment": comments[1],
                    }
                ],
            }
        },
    )
    core = _core(executor, plans={"EURUSD": plan})
    timeline = []
    core._save_pending_limit_state = lambda: timeline.append(
        ("pending", tuple(sorted(core.pending_limits)))
    )
    monkeypatch.setattr(
        main,
        "save_active_trades",
        lambda trades, _path: timeline.append(
            ("active", tuple(sorted(trades)))
        ),
    )

    core._manage_pending_limits()

    assert executor.cancelled == [102, 103]
    assert core.pending_limits["EURUSD"].state is PendingLimitState.FILLED
    assert core.pending_limits["EURUSD"].entry_announcement_pending
    trade = core.active_trades["EURUSD"]
    assert trade.side == "LONG"
    assert trade.entry == pytest.approx(1.0999)
    assert trade.volume == pytest.approx(0.1)
    assert trade.split_position_ids == [9001]
    assert trade.split_legs[9001]["tp_index"] == 1
    event = core._pending_entry_events["EURUSD"]
    assert event["signal"] == "ENTER"
    assert event["execution"]["mode"] == "pending_limit_split"
    active_index = next(i for i, row in enumerate(timeline) if row[0] == "active")
    assert core.ack_pending_limit_entry(plan.plan_id)
    core._manage_pending_limits()
    retired_index = next(
        i
        for i, row in enumerate(timeline)
        if row == ("pending", ())
    )
    assert active_index < retired_index


def test_conflicting_fill_never_merges_and_stays_durable_after_exposure_absent(
    monkeypatch,
):
    monkeypatch.setattr(
        main,
        "PERSISTENT_LIMIT_CONFLICT_AUTOCLOSE_ENABLED",
        False,
    )
    clock = [NOW + 100.0]
    monkeypatch.setattr(main.time, "time", lambda: clock[0])
    plan = _plan(attach_tickets=True)
    comments = {
        index: plan.broker_comment_for_leg(index) for index in (1, 2, 3)
    }
    executor = ReconcileExecutor(
        live_orders=[
            {"ticket": 102, "comment": comments[2]},
            {"ticket": 103, "comment": comments[3]},
        ],
        fills={
            101: {
                "order_ticket": 101,
                "deals": [
                    {
                        "deal_ticket": 501,
                        "position_id": 9001,
                        "volume": 0.1,
                        "price": 1.0999,
                        "time_msc": int((NOW + 10.0) * 1000),
                        "comment": comments[1],
                    }
                ],
            }
        },
        positions=[
            {
                "symbol": "EURUSD",
                "ticket": 8000,
                "comment": "EXISTING:idea",
            },
            {
                "symbol": "EURUSD",
                "ticket": 9001,
                "comment": comments[1],
            },
        ],
    )
    core = _core(executor, plans={"EURUSD": plan})
    existing_trade = SimpleNamespace(idea_id="different-live-idea")
    core.active_trades["EURUSD"] = existing_trade
    snapshots = []
    core._save_pending_limit_state = lambda: snapshots.append(
        {
            symbol: pending.to_dict()
            for symbol, pending in core.pending_limits.items()
        }
    )
    core._materialize_pending_trade = lambda *_args, **_kwargs: pytest.fail(
        "a conflicting fill must never be materialized"
    )

    core._manage_pending_limits()

    assert executor.cancelled == [102, 103]
    assert core.active_trades["EURUSD"] is existing_trade
    assert core._pending_entry_events == {}
    assert "EURUSD" in core.pending_limits
    conflict = core.pending_limits["EURUSD"]
    assert conflict.state is PendingLimitState.CANCELLING
    assert conflict.reason == main._PENDING_CONFLICT_REASON
    assert core._pending_state_safe is True
    assert any(
        row["EURUSD"]["reason"] == main._PENDING_CONFLICT_REASON
        for row in snapshots
        if "EURUSD" in row
    )

    # Even if the old in-memory idea disappears, the durable conflict marker
    # prevents a historical fill from being promoted on a later pass.
    core.active_trades.pop("EURUSD")
    executor.positions = [
        {
            "symbol": "EURUSD",
            "ticket": 9001,
            "comment": comments[1],
        }
    ]
    clock[0] = NOW + 140.0
    core._manage_pending_limits()
    assert "EURUSD" in core.pending_limits
    assert core.active_trades == {}
    assert core._pending_entry_events == {}

    # Absence is not enough to prove that the conflicting position was closed
    # safely.  Preserve the durable marker for explicit/manual recovery even
    # after the bounded publication grace window has elapsed.
    executor.positions = []
    clock[0] = NOW + 180.0
    core._manage_pending_limits()
    assert "EURUSD" in core.pending_limits
    assert (
        core.pending_limits["EURUSD"].reason
        == main._PENDING_CONFLICT_REASON
    )
    assert core.active_trades == {}
    assert core._pending_entry_events == {}

    # The no-close fallback is intentionally indefinite, rather than a second
    # timer that could silently retire the only durable conflict evidence.
    clock[0] = NOW + 10_000.0
    core._manage_pending_limits()
    assert "EURUSD" in core.pending_limits
    assert (
        core.pending_limits["EURUSD"].reason
        == main._PENDING_CONFLICT_REASON
    )
    assert core.active_trades == {}
    assert core._pending_entry_events == {}


def test_emergency_autoclose_closes_only_exact_owned_position_and_retains_audit(
    monkeypatch,
):
    clock = [NOW + 100.0]
    monkeypatch.setattr(main.time, "time", lambda: clock[0])
    monkeypatch.setattr(
        main,
        "PERSISTENT_LIMIT_CONFLICT_AUTOCLOSE_ENABLED",
        True,
    )
    plan = _plan(attach_tickets=True)
    comments = {
        index: plan.broker_comment_for_leg(index)
        for index in (1, 2, 3)
    }
    positions = [
        {
            "symbol": "EURUSD",
            "ticket": 8000,
            "comment": "EXISTING:idea",
        },
        {
            "symbol": "EURUSD",
            "ticket": 9001,
            "comment": comments[1],
        },
    ]
    core, executor, existing_trade, snapshots = _conflict_runtime(
        plan,
        live_orders=[
            {"ticket": 102, "comment": comments[2]},
            {"ticket": 103, "comment": comments[3]},
        ],
        fills={
            101: _opening_fill(101, 9001, comments[1]),
        },
        positions=positions,
        close_results=[True],
    )

    core._manage_pending_limits()

    assert executor.cancelled == [102, 103]
    assert executor.close_calls == [
        ("EURUSD", 9001, None, comments[1])
    ]
    assert core.active_trades["EURUSD"] is existing_trade
    assert core._pending_entry_events == {}
    assert core.pending_limits["EURUSD"].reason == (
        main._PENDING_CONFLICT_CLOSE_REQUESTED_REASON
    )
    assert any(
        row["EURUSD"]["reason"]
        == main._PENDING_CONFLICT_CLOSE_REQUESTED_REASON
        for row in snapshots
        if "EURUSD" in row
    )

    # A successful request is not proof of absence. A still-visible position
    # is retained and not immediately closed a second time.
    clock[0] = NOW + 101.0
    core._manage_pending_limits()
    assert len(executor.close_calls) == 1
    assert "EURUSD" in core.pending_limits

    # A later healthy broker snapshot proves exposure absence, but the plan is
    # still a durable same-symbol recovery/audit marker and cannot auto-resume.
    executor.positions = [positions[0]]
    clock[0] = NOW + 102.0
    core._manage_pending_limits()
    assert "EURUSD" in core.pending_limits
    assert core.pending_limits["EURUSD"].reason == (
        main._PENDING_CONFLICT_CLOSE_RESOLVED_REASON
    )
    assert core.active_trades["EURUSD"] is existing_trade
    assert core._pending_entry_events == {}

    clock[0] = NOW + 10_000.0
    core._manage_pending_limits()
    assert "EURUSD" in core.pending_limits
    assert core.pending_limits["EURUSD"].reason == (
        main._PENDING_CONFLICT_CLOSE_RESOLVED_REASON
    )
    assert len(executor.close_calls) == 1


@pytest.mark.parametrize(
    "identity_case",
    ["wrong_comment", "wrong_id", "wrong_symbol"],
)
def test_emergency_autoclose_never_closes_ambiguous_identity(
    monkeypatch,
    identity_case,
):
    monkeypatch.setattr(main.time, "time", lambda: NOW + 100.0)
    monkeypatch.setattr(
        main,
        "PERSISTENT_LIMIT_CONFLICT_AUTOCLOSE_ENABLED",
        True,
    )
    plan = _plan(attach_tickets=True)
    comments = {
        index: plan.broker_comment_for_leg(index)
        for index in (1, 2, 3)
    }
    adversarial = {
        "symbol": "EURUSD",
        "ticket": 9001,
        "comment": comments[1],
    }
    if identity_case == "wrong_comment":
        adversarial["comment"] = "EXISTING:not-the-plan"
    elif identity_case == "wrong_id":
        adversarial["ticket"] = 9999
    else:
        adversarial["symbol"] = "GBPUSD"
    core, executor, existing_trade, _snapshots = _conflict_runtime(
        plan,
        live_orders=[
            {"ticket": 102, "comment": comments[2]},
            {"ticket": 103, "comment": comments[3]},
        ],
        fills={
            101: _opening_fill(101, 9001, comments[1]),
        },
        positions=[
            {
                "symbol": "EURUSD",
                "ticket": 8000,
                "comment": "EXISTING:idea",
            },
            adversarial,
        ],
        close_results=[True],
    )

    core._manage_pending_limits()

    assert executor.cancelled == [102, 103]
    assert executor.close_calls == []
    assert "EURUSD" in core.pending_limits
    assert core.active_trades["EURUSD"] is existing_trade
    assert core._pending_entry_events == {}
    assert core._pending_state_safe is False


def test_emergency_autoclose_mixed_failure_retries_next_management_tick(
    monkeypatch,
):
    clock = [NOW + 100.0]
    monkeypatch.setattr(main.time, "time", lambda: clock[0])
    monkeypatch.setattr(
        main,
        "PERSISTENT_LIMIT_CONFLICT_AUTOCLOSE_ENABLED",
        True,
    )
    plan = _plan(attach_tickets=True)
    comments = {
        index: plan.broker_comment_for_leg(index)
        for index in (1, 2, 3)
    }
    existing_row = {
        "symbol": "EURUSD",
        "ticket": 8000,
        "comment": "EXISTING:idea",
    }
    position_1 = {
        "symbol": "EURUSD",
        "ticket": 9001,
        "comment": comments[1],
    }
    position_2 = {
        "symbol": "EURUSD",
        "ticket": 9002,
        "comment": comments[2],
    }
    core, executor, existing_trade, _snapshots = _conflict_runtime(
        plan,
        live_orders=[{"ticket": 103, "comment": comments[3]}],
        fills={
            101: _opening_fill(101, 9001, comments[1], leg=1),
            102: _opening_fill(102, 9002, comments[2], leg=2),
        },
        positions=[existing_row, position_1, position_2],
        close_results=[True, False, True],
    )

    core._manage_pending_limits()

    assert executor.cancelled == [103]
    assert executor.close_calls == [
        ("EURUSD", 9001, None, comments[1]),
        ("EURUSD", 9002, None, comments[2]),
    ]
    assert core.pending_limits["EURUSD"].reason == (
        main._PENDING_CONFLICT_CLOSE_FAILED_REASON
    )
    assert core.active_trades["EURUSD"] is existing_trade

    # The first exact close is now absent. FAILED retries the remaining exact
    # position on the very next management tick; no 30-second grace applies.
    executor.positions = [existing_row, position_2]
    clock[0] = NOW + 101.0
    core._manage_pending_limits()
    assert executor.close_calls[-1] == (
        "EURUSD",
        9002,
        None,
        comments[2],
    )
    assert len(executor.close_calls) == 3
    assert core.pending_limits["EURUSD"].reason == (
        main._PENDING_CONFLICT_CLOSE_REQUESTED_REASON
    )
    assert "EURUSD" in core.pending_limits

    executor.positions = [existing_row]
    clock[0] = NOW + 102.0
    core._manage_pending_limits()
    assert core.pending_limits["EURUSD"].reason == (
        main._PENDING_CONFLICT_CLOSE_RESOLVED_REASON
    )
    assert core.active_trades["EURUSD"] is existing_trade
    assert core._pending_entry_events == {}


def test_emergency_autoclose_requires_hedging_capability(monkeypatch):
    monkeypatch.setattr(main.time, "time", lambda: NOW + 100.0)
    monkeypatch.setattr(
        main,
        "PERSISTENT_LIMIT_CONFLICT_AUTOCLOSE_ENABLED",
        True,
    )
    plan = _plan(attach_tickets=True)
    comments = {
        index: plan.broker_comment_for_leg(index)
        for index in (1, 2, 3)
    }
    core, executor, existing_trade, _snapshots = _conflict_runtime(
        plan,
        live_orders=[
            {"ticket": 102, "comment": comments[2]},
            {"ticket": 103, "comment": comments[3]},
        ],
        fills={
            101: _opening_fill(101, 9001, comments[1]),
        },
        positions=[
            {
                "symbol": "EURUSD",
                "ticket": 8000,
                "comment": "EXISTING:idea",
            },
            {
                "symbol": "EURUSD",
                "ticket": 9001,
                "comment": comments[1],
            },
        ],
        close_results=[True],
        capabilities={
            "limit_allowed": True,
            "specified_expiration": True,
            "account_hedging": False,
            "ready": False,
        },
    )

    core._manage_pending_limits()

    assert executor.cancelled == [102, 103]
    assert executor.close_calls == []
    assert "EURUSD" in core.pending_limits
    assert core.active_trades["EURUSD"] is existing_trade
    assert core._pending_state_safe is False


def test_emergency_close_classifier_refuses_missing_recorded_position_id():
    plan = _plan(attach_tickets=True).record_fill(
        1,
        0.1,
        1.0999,
        deal_ticket=501,
        position_id=None,
        now=NOW + 10.0,
    )
    targets, errors = main.Core._pending_plan_close_targets(
        plan,
        [
            {
                "symbol": "EURUSD",
                "ticket": 9001,
                "comment": plan.broker_comment_for_leg(1),
            }
        ],
        executor_magic=4242,
    )

    assert targets == ()
    assert errors
    assert any("recorded" in error for error in errors)


def test_pending_conflict_ownership_classifier_is_exact_and_conservative():
    plan = _plan(attach_tickets=True).record_fill(
        1,
        0.1,
        1.0999,
        deal_ticket=501,
        position_id=9001,
        now=NOW + 10.0,
    )
    comment_1 = plan.broker_comment_for_leg(1)
    comment_2 = plan.broker_comment_for_leg(2)

    owned, errors = main.Core._pending_plan_owned_position_ids(
        plan,
        [
            {
                "symbol": "EURUSD",
                "ticket": 8000,
                "comment": "EXISTING:idea",
            },
            {
                "symbol": "EURUSD",
                "ticket": 9001,
                "comment": comment_1,
            },
            {
                "symbol": "EURUSD",
                "ticket": 9002,
                "comment": comment_2,
            },
        ],
    )
    assert owned == (9001, 9002)
    assert errors == ()

    owned, errors = main.Core._pending_plan_owned_position_ids(
        plan,
        [
            {
                "symbol": "EURUSD",
                "ticket": 9001,
                "comment": "EXISTING:idea",
            },
            {
                "symbol": "EURUSD",
                "ticket": 9999,
                "comment": comment_1,
            },
        ],
    )
    assert owned == ()
    assert errors
    assert any("comment" in error for error in errors)
    assert any("unexpected" in error for error in errors)


def test_restart_recovers_ticket_then_cancels_incomplete_basket_without_duplicate():
    plan = _plan(attach_tickets=False)
    comment = plan.broker_comment_for_leg(1)
    executor = ReconcileExecutor(
        live_orders=[{"ticket": 777, "comment": comment}],
    )
    core = _core(executor, plans={"EURUSD": plan})
    snapshots = []
    core._save_pending_limit_state = lambda: snapshots.append(
        {
            symbol: pending.to_dict()
            for symbol, pending in core.pending_limits.items()
        }
    )
    core._pending_cancel_reason = lambda _plan, *, now, quote: None

    core._manage_pending_limits(startup=True)
    assert executor.cancelled == [777]
    assert "EURUSD" not in core.pending_limits

    core._manage_pending_limits(startup=True)

    assert executor.placement_calls == 0


def test_corrupt_state_flag_blocks_new_entry_before_other_guards_run():
    core = main.Core.__new__(main.Core)
    core._pending_state_safe = False
    sig = {"signal": "ENTER"}

    blocked = core._apply_entry_guards(
        "EURUSD",
        sig,
        side="LONG",
        trig_sig="trigger",
    )

    assert blocked is True
    assert sig["signal"] == "WAIT_PENDING_STATE"
    assert "fail-closed" in sig["info"]


def test_orphan_broker_identity_marks_state_ambiguous_and_blocks_entry():
    executor = ReconcileExecutor(
        live_orders=[
            {
                "ticket": 999,
                "comment": f"{BROKER_COMMENT_PREFIX}orphan:T1",
            }
        ]
    )
    core = _core(executor)
    core._save_pending_limit_state = lambda: None

    core._manage_pending_limits(startup=True)

    assert core._pending_state_safe is False
    sig = {"signal": "ENTER"}
    assert core._apply_entry_guards(
        "EURUSD",
        sig,
        side="LONG",
        trig_sig="new-trigger",
    )
    assert sig["signal"] == "WAIT_PENDING_STATE"
