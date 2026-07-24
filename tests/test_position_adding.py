from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

import main
from core.position_adding import PyramidManager, PyramidSettings
from core.state_store import load_active_trades, save_active_trades
from core.strategy_narrative import ActiveTrade

CAPITAL = 10_000.0
RISK_PER_LOT = 1_000.0  # 0.1 lot == 1% of CAPITAL


class FakeRiskExecutor:
    """Money model: risk scales with volume, and is zero at break-even."""

    def __init__(self, *, risk_fraction=0.01, capital=CAPITAL):
        self.risk_fraction = risk_fraction
        self.capital = capital
        self.move_calls = []
        self.move_ok = True

    def risk_capital_base(self):
        return self.capital

    def planned_entry_risk_fraction(self):
        return self.risk_fraction

    def open_risk_amount(self, symbol, *, side, entry_price, stop_price, volume):
        if volume <= 0:
            return 0.0
        if side == "LONG" and stop_price >= entry_price:
            return 0.0
        if side == "SHORT" and stop_price <= entry_price:
            return 0.0
        return RISK_PER_LOT * volume

    # break-even plumbing used by Core._move_entry_to_breakeven
    def move_stop_all(self, symbol, *, position_ids, new_stop):
        self.move_calls.append((symbol, list(position_ids), new_stop))
        return len(position_ids) if self.move_ok else 0

    def connection_alive(self):
        return True

    def get_open_position_ids(self, symbol):
        return set()

    def get_position_close_info(self, ticket):
        return None


def _entry(index: int, *, entry: float, stop: float, tickets, volume=0.1) -> ActiveTrade:
    trade = ActiveTrade(
        side="LONG",
        entry=entry,
        stop=stop,
        tp_prices=[entry + 0.0100, entry + 0.0200, entry + 0.0300],
        tf="15M",
        narrative="idea",
        symbol="EURUSD",
    )
    trade.volume = volume
    trade.volume_remaining = volume
    trade.entry_index = index
    trade.idea_id = "idea-1"
    trade.split_position_ids = list(tickets)
    trade.split_legs = {
        ticket: {
            "tp_index": n,
            "tp": trade.tp_prices[n - 1],
            "volume": volume / len(tickets),
            "status": "open",
        }
        for n, ticket in enumerate(tickets, start=1)
    }
    return trade


def _idea(*entries: ActiveTrade) -> ActiveTrade:
    primary, addons = entries[0], list(entries[1:])
    primary.addons = addons
    primary.idea_trigger_signatures = [f"LONG|trig{e.entry_index}|z1" for e in entries]
    return primary


def _manager(**overrides) -> PyramidManager:
    settings = dict(
        enabled=True,
        max_entries=3,
        max_idea_risk_pct=0.02,
        min_progress_r=0.5,
        max_progress_pct=0.5,
    )
    settings.update(overrides)
    return PyramidManager(PyramidSettings(**settings))


def _sig(**overrides):
    entry = overrides.get("entry_price", 1.0060)
    sig = {
        "signal": "ENTER",
        "side": "LONG",
        "entry_price": entry,
        "stop_price": entry - 0.0050,
        "tp_prices": [entry + 0.0100, entry + 0.0200, entry + 0.0300],
    }
    sig.update(overrides)
    return sig


# ---------------- decision gates ----------------


def test_disabled_manager_never_adds():
    idea = _idea(_entry(1, entry=1.0000, stop=0.9950, tickets=[101]))
    decision = _manager(enabled=False).evaluate(
        idea, _sig(), executor=FakeRiskExecutor(), last_price=1.0060,
        trigger_signature="LONG|new|z2",
    )
    assert decision.allowed is False


def test_opposite_side_signal_is_not_an_addon():
    idea = _idea(_entry(1, entry=1.0000, stop=0.9950, tickets=[101]))
    decision = _manager().evaluate(
        idea, _sig(side="SHORT"), executor=FakeRiskExecutor(), last_price=1.0060,
        trigger_signature="SHORT|new|z2",
    )
    assert decision.allowed is False
    assert "направлени" in decision.reason


def test_entry_limit_stops_the_pyramid():
    idea = _idea(
        _entry(1, entry=1.0000, stop=0.9950, tickets=[101]),
        _entry(2, entry=1.0030, stop=0.9980, tickets=[102]),
        _entry(3, entry=1.0060, stop=1.0010, tickets=[103]),
    )
    decision = _manager().evaluate(
        idea, _sig(), executor=FakeRiskExecutor(), last_price=1.0100,
        trigger_signature="LONG|new|z4",
    )
    assert decision.allowed is False
    assert "лимит 3 входов" in decision.reason


def test_repeated_trigger_is_not_a_new_confirmation():
    idea = _idea(_entry(1, entry=1.0000, stop=0.9950, tickets=[101]))
    decision = _manager().evaluate(
        idea, _sig(), executor=FakeRiskExecutor(), last_price=1.0060,
        trigger_signature="LONG|trig1|z1",
    )
    assert decision.allowed is False
    assert "повторяет" in decision.reason


def test_addon_requires_the_idea_to_be_in_profit():
    idea = _idea(_entry(1, entry=1.0000, stop=0.9950, tickets=[101]))
    decision = _manager().evaluate(
        idea, _sig(), executor=FakeRiskExecutor(), last_price=1.0010,  # 0.2R
        trigger_signature="LONG|new|z2",
    )
    assert decision.allowed is False
    assert "не подтверждена" in decision.reason


def test_kolachi_rule_blocks_adding_past_half_way_to_target():
    # Entry 1.0000, final TP 1.0300 → 1.0200 is 67% of the way.
    idea = _idea(_entry(1, entry=1.0000, stop=0.9950, tickets=[101]))
    decision = _manager().evaluate(
        idea, _sig(), executor=FakeRiskExecutor(), last_price=1.0200,
        trigger_signature="LONG|new|z2",
    )
    assert decision.allowed is False
    assert "пути до финального TP" in decision.reason


# ---------------- risk invariant ----------------


def test_second_entry_fits_without_break_even():
    idea = _idea(_entry(1, entry=1.0000, stop=0.9950, tickets=[101]))
    decision = _manager().evaluate(
        idea, _sig(), executor=FakeRiskExecutor(), last_price=1.0060,
        trigger_signature="LONG|new|z2",
    )
    assert decision.allowed is True
    assert decision.entry_index == 2
    assert decision.entries_to_breakeven == []
    assert decision.idea_risk_pct == 0.01
    assert decision.projected_risk_pct == 0.02


def test_third_entry_moves_only_the_oldest_entry_to_break_even():
    first = _entry(1, entry=1.0000, stop=0.9950, tickets=[101])
    second = _entry(2, entry=1.0030, stop=0.9980, tickets=[102])
    idea = _idea(first, second)
    decision = _manager().evaluate(
        idea, _sig(), executor=FakeRiskExecutor(), last_price=1.0060,
        trigger_signature="LONG|new|z3",
    )
    assert decision.allowed is True
    assert decision.entry_index == 3
    # Entry 2 keeps its stop — only entry 1 frees the room the third entry needs.
    assert decision.entries_to_breakeven == [first]
    assert decision.idea_risk_pct == 0.02
    assert decision.projected_risk_pct == 0.02


def test_failed_idea_costs_the_cap_not_the_sum_of_entries():
    """Scenario 2: entry 1 at BE, entries 2 and 3 still at their stops."""
    first = _entry(1, entry=1.0000, stop=1.0000, tickets=[101])  # moved to BE
    first.moved_to_be = True
    idea = _idea(
        first,
        _entry(2, entry=1.0030, stop=0.9980, tickets=[102]),
        _entry(3, entry=1.0060, stop=1.0010, tickets=[103]),
    )
    manager = _manager()
    assert manager.idea_risk_pct(FakeRiskExecutor(), idea) == 0.02


def test_addon_blocked_when_break_even_cannot_free_enough_room():
    # Cap 1% with 1% already at risk: the newest entry may not be moved to BE,
    # so there is no room left for another entry.
    idea = _idea(_entry(1, entry=1.0000, stop=0.9950, tickets=[101]))
    decision = _manager(max_idea_risk_pct=0.01).evaluate(
        idea, _sig(), executor=FakeRiskExecutor(), last_price=1.0060,
        trigger_signature="LONG|new|z2",
    )
    assert decision.allowed is False
    assert "не помещается в лимит" in decision.reason


def test_partially_closed_entry_releases_risk():
    entry = _entry(1, entry=1.0000, stop=0.9950, tickets=[101, 102], volume=0.1)
    entry.volume_remaining = 0.05
    assert _manager().idea_risk_pct(FakeRiskExecutor(), _idea(entry)) == 0.005


# ---------------- persistence ----------------


def test_idea_survives_a_restart_with_each_entry_intact(tmp_path: Path):
    idea = _idea(
        _entry(1, entry=1.0000, stop=1.0000, tickets=[101]),
        _entry(2, entry=1.0030, stop=0.9980, tickets=[102, 103]),
    )
    idea.moved_to_be = True
    idea.be_for_idea_risk = True
    path = tmp_path / "active_trades.json"

    save_active_trades({"EURUSD": idea}, path)
    restored = load_active_trades(path)["EURUSD"]

    assert restored.idea_id == "idea-1"
    assert restored.entry_index == 1
    assert restored.moved_to_be is True
    assert restored.be_for_idea_risk is True
    assert restored.idea_trigger_signatures == idea.idea_trigger_signatures
    assert len(restored.addons) == 1
    addon = restored.addons[0]
    assert addon.entry_index == 2
    assert addon.entry == 1.0030
    assert addon.split_position_ids == [102, 103]
    assert set(addon.split_legs) == {102, 103}
    assert len(restored.entries()) == 2


def test_legacy_state_without_addons_still_loads(tmp_path: Path):
    path = tmp_path / "active_trades.json"
    path.write_text(
        json.dumps({
            "EURUSD": {
                "side": "LONG", "entry": 1.0, "stop": 0.99, "tp_prices": [1.1],
                "tf": "15M", "narrative": "legacy", "symbol": "EURUSD",
                "split_position_ids": [777],
            }
        }),
        encoding="utf-8",
    )
    restored = load_active_trades(path)["EURUSD"]
    assert restored.entry_index == 1
    assert restored.addons == []
    assert restored.entries() == [restored]


# ---------------- Core integration ----------------


def _core(executor, trade):
    core = main.Core.__new__(main.Core)
    core.mt5_executor = executor
    core.active_trades = {trade.symbol: trade}
    core._broker_missing_counts = {}
    core._broker_missing_confirm = 2
    core._management_lock = threading.RLock()
    core._entry_cooldowns = {}
    core._post_sl_cooldown_sec = 3600.0
    return core


def test_hydration_keeps_each_entry_on_its_own_tickets():
    first = _entry(1, entry=1.0000, stop=0.9950, tickets=[101])
    second = _entry(2, entry=1.0030, stop=0.9980, tickets=[102])
    idea = _idea(first, second)
    positions = [
        {"ticket": 102, "volume": 0.2, "tp": second.tp_prices[0], "comment": "TP 1"},
        {"ticket": 101, "volume": 0.3, "tp": first.tp_prices[0], "comment": "TP 1"},
        {"ticket": 999, "volume": 0.1, "tp": first.tp_prices[1], "comment": "TP 2"},
    ]

    buckets = main.Core._partition_positions_by_entry(idea, positions)

    assert [entry.entry_index for entry, _ in buckets] == [1, 2]
    # Unknown ticket 999 falls back to the primary entry.
    assert [p["ticket"] for p in buckets[0][1]] == [101, 999]
    assert [p["ticket"] for p in buckets[1][1]] == [102]


def test_idea_is_final_only_when_every_entry_is_closed(monkeypatch):
    first = _entry(1, entry=1.0000, stop=0.9950, tickets=[101])
    second = _entry(2, entry=1.0030, stop=0.9980, tickets=[102])
    idea = _idea(first, second)

    class Executor(FakeRiskExecutor):
        open_ids = {102}
        closes = {101: {"reason": "SL", "reason_code": 4, "deal_ticket": 1,
                        "price": 0.9950, "volume": 0.1, "profit": -10.0,
                        "commission": 0.0, "swap": 0.0, "time": 1}}

        def get_open_position_ids(self, symbol):
            return set(self.open_ids)

        def get_position_close_info(self, ticket):
            return self.closes.get(ticket)

    executor = Executor()
    core = _core(executor, idea)

    for _ in range(3):
        result = core._poll_idea_lifecycle("EURUSD", idea)
    # Entry 1 is gone, entry 2 is still open → the idea is not finished.
    assert result["final"] is False
    assert first.split_legs[101]["status"] == "closed"
    assert second.split_legs[102]["status"] == "open"

    executor.open_ids = set()
    executor.closes[102] = dict(executor.closes[101], deal_ticket=2)
    for _ in range(3):
        result = core._poll_idea_lifecycle("EURUSD", idea)
    assert result["final"] is True


def _addon_core(executor, idea, monkeypatch, *, addon_entry=1.0060):
    core = _core(executor, idea)
    core.pyramid = _manager()
    core._entries_today = {}
    core._trigger_signatures = {}
    core._stale_cooldown_sec = 60.0
    core._entry_cooldown_sec = 300.0
    core.strategy = type("S", (), {
        "generate_signal": staticmethod(
            lambda data, symbol="": _sig(
                entry_price=addon_entry,
                stop_price=addon_entry - 0.0050,
                trigger_reason="turtle_soup",
            )
        )
    })()
    core.ai = type("A", (), {
        "on_signal": staticmethod(lambda symbol, sig, data, trades: sig)
    })()
    monkeypatch.setattr(main, "POSITION_ADDING_ENABLED", True)
    monkeypatch.setattr(main, "IDEA_MAX_ENTRIES", 3)
    monkeypatch.setattr(main.Core, "_apply_global_filters", lambda self, s, sig: sig)
    monkeypatch.setattr(main.Core, "_apply_session_filter", lambda self, s, sig: sig)
    monkeypatch.setattr(
        main.Core, "_apply_vol_regime_filter", lambda self, s, sig, data: sig
    )
    monkeypatch.setattr(
        main.Core, "_apply_entry_guards",
        lambda self, symbol, sig, side, trig_sig: False,
    )
    return core


def _executed_addon(index: int, entry: float) -> ActiveTrade:
    return _entry(index, entry=entry, stop=entry - 0.0050, tickets=[100 + index])


def test_third_entry_moves_first_to_break_even_before_opening(monkeypatch):
    first = _entry(1, entry=1.0000, stop=0.9950, tickets=[101])
    second = _entry(2, entry=1.0030, stop=0.9980, tickets=[102])
    idea = _idea(first, second)
    executor = FakeRiskExecutor()
    core = _addon_core(executor, idea, monkeypatch)
    calls = []
    monkeypatch.setattr(
        main.Core, "_execute_entry_signal",
        lambda self, symbol, sig: calls.append(("execute", executor.move_calls[:]))
        or _executed_addon(3, 1.0060),
    )

    result = core._try_position_add("EURUSD", idea, {}, 1.0060)

    assert result is not None
    assert result["entry_index"] == 3
    # The break-even that frees the risk budget happens BEFORE the new order.
    assert calls[0][1] == [("EURUSD", [101], 1.0000)]
    assert first.moved_to_be is True and first.be_for_idea_risk is True
    assert second.moved_to_be is False
    assert [e.entry_index for e in idea.entries()] == [1, 2, 3]
    assert core._entries_today["EURUSD"] == 1
    assert len(idea.idea_trigger_signatures) == 3


def test_addon_is_cancelled_when_break_even_fails(monkeypatch):
    first = _entry(1, entry=1.0000, stop=0.9950, tickets=[101])
    second = _entry(2, entry=1.0030, stop=0.9980, tickets=[102])
    idea = _idea(first, second)
    executor = FakeRiskExecutor()
    executor.move_ok = False
    core = _addon_core(executor, idea, monkeypatch)
    monkeypatch.setattr(
        main.Core, "_execute_entry_signal",
        lambda self, symbol, sig: pytest.fail("must not open while risk exceeds the cap"),
    )

    assert core._try_position_add("EURUSD", idea, {}, 1.0060) is None
    assert idea.addons == [second]
    assert first.moved_to_be is False


def test_single_tp_signal_is_not_added_to_a_split_idea(monkeypatch):
    """A monitor-mode entry inside a split idea would never reach EXIT_BROKER."""
    idea = _idea(_entry(1, entry=1.0000, stop=0.9950, tickets=[101]))
    core = _addon_core(FakeRiskExecutor(), idea, monkeypatch)
    core.strategy = type("S", (), {
        "generate_signal": staticmethod(
            lambda data, symbol="": _sig(tp_prices=[1.0160])
        )
    })()
    monkeypatch.setattr(
        main.Core, "_execute_entry_signal",
        lambda self, symbol, sig: pytest.fail("single-TP add-on must be refused"),
    )

    assert core._try_position_add("EURUSD", idea, {}, 1.0060) is None


def test_no_addon_while_the_idea_is_not_confirmed(monkeypatch):
    idea = _idea(_entry(1, entry=1.0000, stop=0.9950, tickets=[101]))
    core = _addon_core(FakeRiskExecutor(), idea, monkeypatch)
    monkeypatch.setattr(
        main.Core, "_execute_entry_signal",
        lambda self, symbol, sig: pytest.fail("idea is only 0.2R in profit"),
    )

    assert core._try_position_add("EURUSD", idea, {}, 1.0010) is None
    assert idea.addons == []


def test_break_even_uses_each_entry_own_fill_price():
    first = _entry(1, entry=1.0000, stop=0.9950, tickets=[101])
    second = _entry(2, entry=1.0030, stop=0.9980, tickets=[102])
    first.tp_hit = 1
    second.tp_hit = 1
    idea = _idea(first, second)
    executor = FakeRiskExecutor()
    core = _core(executor, idea)

    events = core._move_idea_entries_to_breakeven("EURUSD", idea)

    assert executor.move_calls == [
        ("EURUSD", [101], 1.0000),
        ("EURUSD", [102], 1.0030),
    ]
    assert [e["price"] for e in events] == [1.0000, 1.0030]
    assert events[1]["entry_index"] == 2
    assert first.stop == 1.0000
    assert second.stop == 1.0030
