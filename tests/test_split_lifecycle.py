from __future__ import annotations

import threading

import main
from core.strategy_narrative import ActiveTrade


class FakeExecutor:
    def __init__(self, *, open_ids, close_info):
        self.open_ids = set(open_ids)
        self.close_info = close_info
        self.move_calls = []

    def get_open_position_ids(self, symbol):
        return set(self.open_ids)

    def get_position_close_info(self, ticket):
        value = self.close_info.get(ticket)
        if isinstance(value, list):
            return value.pop(0) if value else None
        return value

    def get_position_close_reason(self, ticket):
        value = self.close_info.get(ticket)
        if isinstance(value, dict) and value.get("reason") in {"TP", "SL"}:
            return value["reason"]
        return None

    def get_current_price(self, symbol, side):
        # Deliberately below TP1 for a LONG: BE must be driven by deal history,
        # not by the price at this later management tick.
        return 1.0500

    def move_stop_all(self, symbol, *, position_ids, new_stop):
        self.move_calls.append((symbol, list(position_ids), new_stop))
        return len(position_ids)

    def connection_alive(self):
        return True


def _trade(*tickets: int) -> ActiveTrade:
    trade = ActiveTrade(
        side="LONG",
        entry=1.0000,
        stop=0.9500,
        tp_prices=[1.1000, 1.2000],
        tf="15m",
        narrative="test",
        symbol="EURUSD",
    )
    trade.volume = 0.2
    trade.volume_remaining = 0.2
    trade.split_position_ids = list(tickets)
    trade.split_legs = {
        ticket: {
            "tp_index": index,
            "tp": trade.tp_prices[index - 1],
            "volume": 0.1,
            "status": "open",
        }
        for index, ticket in enumerate(tickets, start=1)
    }
    return trade


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


def _tp_close(price=1.1000):
    return {
        "reason": "TP",
        "reason_code": 5,
        "deal_ticket": 9001,
        "price": price,
        "volume": 0.1,
        "profit": 10.0,
        "commission": -0.5,
        "swap": 0.0,
        "time": 123456,
    }


def test_mapping_preserves_exact_tp_index_and_event_is_idempotent():
    trade = _trade(101, 102)
    executor = FakeExecutor(open_ids={102}, close_info={101: _tp_close()})
    core = _core(executor, trade)

    first = core._poll_split_lifecycle("EURUSD", trade, last_price=1.0500)

    assert first["events"] == [
        {
            "type": "TP",
            "tp_index": 1,
            "tp_price": 1.1,
            "hit_price": 1.1,
            "source": "broker_deal",
            "position_id": 101,
            "close_deal_ticket": 9001,
        }
    ]
    assert trade.tp_hit == 1
    assert trade.split_position_ids == [102]
    assert trade.split_legs[101]["status"] == "closed"

    second = core._poll_split_lifecycle("EURUSD", trade, last_price=1.0600)
    assert second["events"] == []
    assert trade.tp_hit == 1


def test_hydration_corrects_legacy_index_from_broker_comment():
    trade = _trade(151)
    trade.tp_prices = [1.1, 1.2, 1.3]
    trade.split_legs[151] = {
        "tp_index": 1,
        "tp": 1.1,
        "volume": 0.1,
        "status": "open",
        "legacy_inferred": True,
    }
    core = main.Core.__new__(main.Core)

    changed = core._ensure_split_leg_mapping(
        trade,
        positions=[
            {
                "ticket": 151,
                "tp": 1.3,
                "volume": 0.1,
                "comment": "setup TP3",
            }
        ],
    )

    assert changed is True
    assert trade.split_legs[151]["tp_index"] == 3
    assert trade.split_legs[151]["tp"] == 1.3
    assert "legacy_inferred" not in trade.split_legs[151]


def test_delayed_deal_history_retries_then_moves_be_after_retrace(monkeypatch):
    trade = _trade(201, 202)
    executor = FakeExecutor(
        open_ids={202},
        close_info={201: [None, _tp_close()]},
    )
    core = _core(executor, trade)
    monkeypatch.setattr(main, "MOVE_BE_AFTER_TP1", True)
    monkeypatch.setattr(main, "save_active_trades", lambda *args, **kwargs: None)

    assert core.manage_active_trades() == {}
    assert trade.split_legs[201]["status"] == "pending_history"
    assert trade.moved_to_be is False

    signals = core.manage_active_trades()

    assert [event["type"] for event in signals["EURUSD"]["events"]] == ["TP", "BE"]
    assert executor.move_calls == [("EURUSD", [202], 1.0)]
    assert trade.moved_to_be is True
    assert trade.stop == 1.0

    # Closed state suppresses duplicate TP/BE messages on later fast ticks.
    assert core.manage_active_trades() == {}


def test_final_broker_exit_is_emitted_once(monkeypatch):
    trade = _trade(301)
    executor = FakeExecutor(open_ids=set(), close_info={301: _tp_close()})
    core = _core(executor, trade)
    registered = []
    core._register_broker_close = lambda symbol, tr, sig: registered.append(symbol)
    core._log_signal = lambda *args, **kwargs: None
    monkeypatch.setattr(main, "save_active_trades", lambda *args, **kwargs: None)

    first = core.manage_active_trades()
    assert first["EURUSD"]["signal"] == "HOLD"
    assert [event["type"] for event in first["EURUSD"]["events"]] == ["TP"]

    second = core.manage_active_trades()
    assert second["EURUSD"]["signal"] == "EXIT_BROKER"
    assert registered == ["EURUSD"]
    assert "EURUSD" not in core.active_trades

    assert core.manage_active_trades() == {}
    assert registered == ["EURUSD"]
