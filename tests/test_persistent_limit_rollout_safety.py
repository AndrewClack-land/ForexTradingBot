"""Independent safety proof for the two-stage persistent-LIMIT rollout."""

from __future__ import annotations

import ast
import inspect
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import main
from core.persistent_limit import (
    BROKER_COMMENT_PREFIX,
    PendingLimitLeg,
    PendingLimitPlan,
    PendingLimitState,
)
from core.shadow_tick_watcher import MT5ReadOnlyFacade


ROOT = Path(__file__).resolve().parents[1]


def _config_getenv_default(name: str) -> str:
    tree = ast.parse((ROOT / "config.py").read_text(encoding="utf-8"))
    matches: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        function = node.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == "getenv"
            and isinstance(function.value, ast.Name)
            and function.value.id == "os"
        ):
            continue
        key, default = node.args[:2]
        if (
            isinstance(key, ast.Constant)
            and key.value == name
            and isinstance(default, ast.Constant)
            and isinstance(default.value, str)
        ):
            matches.append(default.value)
    assert len(matches) == 1, f"expected one os.getenv default for {name}"
    return matches[0]


def _example_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in (ROOT / ".env.example").read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def test_both_rollout_features_default_off_in_code_and_template():
    example = _example_env()
    for name in (
        "PERSISTENT_LIMIT_ENABLED",
        "SHADOW_TICK_WATCHER_ENABLED",
    ):
        assert _config_getenv_default(name) == "0"
        assert example[name] == "0"


@pytest.mark.parametrize(
    "surface",
    ["config fallback", ".env.example"],
)
def test_persistent_limit_canary_defaults_to_eurusd_only(surface):
    if surface == "config fallback":
        symbols = _config_getenv_default("PERSISTENT_LIMIT_SYMBOLS")
    else:
        symbols = _example_env()["PERSISTENT_LIMIT_SYMBOLS"]

    normalized = {
        symbol.strip().upper()
        for symbol in symbols.split(",")
        if symbol.strip()
    }
    assert normalized == {"EURUSD"}, (
        f"{surface} must not widen the first broker canary beyond EURUSD"
    )


def test_startup_passes_only_a_structurally_read_only_facade_to_watcher():
    order_calls: list[tuple] = []

    class DangerousModule:
        COPY_TICKS_INFO = 1

        @staticmethod
        def symbol_info_tick(symbol):
            return SimpleNamespace(
                symbol=symbol,
                time_msc=1_000,
                bid=1.1,
                ask=1.1001,
                flags=0,
            )

        @staticmethod
        def copy_ticks_range(symbol, start, end, flags):
            del symbol, start, end, flags
            return []

        @staticmethod
        def order_send(*args, **kwargs):
            order_calls.append((args, kwargs))
            raise AssertionError("watcher reached an order capability")

    facade = MT5ReadOnlyFacade.from_module(DangerousModule)

    assert set(MT5ReadOnlyFacade.__slots__) == {
        "_symbol_info_tick",
        "_copy_ticks_range",
        "copy_ticks_info",
    }
    assert not hasattr(facade, "__dict__")
    for forbidden in (
        "order_send",
        "order_check",
        "orders_get",
        "positions_get",
        "deal_send",
        "trade",
    ):
        assert not hasattr(facade, forbidden)
    assert facade.symbol_info_tick("EURUSD").bid == pytest.approx(1.1)
    assert facade.copy_ticks_range(
        "EURUSD",
        SimpleNamespace(),
        SimpleNamespace(),
    ) == []
    assert order_calls == []

    startup_source = inspect.getsource(main.Core.__init__)
    assert "mt5=MT5ReadOnlyFacade.from_module(_mt5)" in startup_source
    assert "mt5=_mt5" not in startup_source


def _working_plan() -> PendingLimitPlan:
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
        "idea_id": "rollback-canary-idea",
    }
    return PendingLimitPlan.create(
        symbol="EURUSD",
        signal=signal,
        trigger_signature="LONG|pivot_reclaim_h1|rollback",
        expires_at=now + 900.0,
        magic=4242,
        legs=(PendingLimitLeg(1, 1.1150, 0.1),),
        now=now,
        plan_id="rollback-canary-plan-00000001",
    ).attach_order(1, 101, now=now + 0.1)


class _RollbackExecutor:
    def __init__(self, plan: PendingLimitPlan):
        self.plan = plan
        self.core = None
        self.snapshots: list[dict] = []
        self.cancelled: list[tuple[int, str]] = []
        self.placement_calls = 0
        self.live_orders = [
            {
                "ticket": 101,
                "symbol": plan.symbol,
                "comment": plan.broker_comment_for_leg(1),
            },
            {
                "ticket": 909,
                "symbol": plan.symbol,
                "comment": "MANUAL:unrelated",
            },
        ]

    def list_pending_orders(self):
        return [dict(row) for row in self.live_orders]

    @staticmethod
    def list_positions():
        return []

    @staticmethod
    def get_pending_order_fills(
        _tickets,
        *,
        comments,
        lookback_days,
    ):
        assert comments
        assert lookback_days >= 1
        return {}

    def get_pending_order_histories(
        self,
        comments,
        *,
        lookback_days,
    ):
        assert lookback_days >= 1
        comment = self.plan.broker_comment_for_leg(1)
        assert comment in comments
        return {
            comment: [
                {
                    "ticket": 101,
                    "terminal": True,
                    "fill_expected": False,
                    "had_execution": False,
                    "time_done_msc": 0,
                }
            ]
        }

    @staticmethod
    def get_pending_order_history(ticket):
        assert ticket == 101
        return {
            "ticket": 101,
            "terminal": True,
            "fill_expected": False,
            "had_execution": False,
            "time_done_msc": 0,
        }

    def cancel_pending_order(self, ticket, *, expected_comment_prefix):
        assert self.core is not None
        assert ticket == 101
        assert expected_comment_prefix == self.plan.broker_comment
        assert self.snapshots
        persisted = self.snapshots[-1][self.plan.symbol]
        assert persisted["state"] == PendingLimitState.CANCELLING.value
        assert "feature disabled" in persisted["reason"]
        self.cancelled.append((int(ticket), expected_comment_prefix))
        self.live_orders = [
            row
            for row in self.live_orders
            if int(row["ticket"]) != int(ticket)
        ]
        return True

    def place_limit_leg(self, *_args, **_kwargs):
        self.placement_calls += 1
        raise AssertionError("rollback reconciliation must never place")


def test_disabling_flag_durably_cancels_only_known_pending_order(
    monkeypatch,
):
    monkeypatch.setattr(main, "PERSISTENT_LIMIT_ENABLED", False)
    monkeypatch.setattr(
        main,
        "PERSISTENT_LIMIT_SYMBOLS",
        frozenset({"EURUSD"}),
    )
    plan = _working_plan()
    executor = _RollbackExecutor(plan)
    core = main.Core.__new__(main.Core)
    core.mt5_executor = executor
    core.pending_limits = {plan.symbol: plan}
    core.active_trades = {}
    core._pending_state_safe = True
    core._pending_reconcile_error_ts = 0.0
    core._pending_entry_events = {}
    core._management_lock = threading.RLock()
    core.shadow_tick_watcher = None
    core._capture_quote = lambda _symbol: None

    def persist_snapshot():
        executor.snapshots.append(
            {
                symbol: pending.to_dict()
                for symbol, pending in core.pending_limits.items()
            }
        )

    core._save_pending_limit_state = persist_snapshot
    executor.core = core

    core._manage_pending_limits(startup=True)

    assert executor.cancelled == [(101, plan.broker_comment)]
    assert executor.placement_calls == 0
    assert executor.live_orders == [
        {
            "ticket": 909,
            "symbol": plan.symbol,
            "comment": "MANUAL:unrelated",
        }
    ]
    assert plan.symbol not in core.pending_limits
    assert executor.snapshots[-1] == {}
    assert plan.broker_comment.startswith(BROKER_COMMENT_PREFIX)
