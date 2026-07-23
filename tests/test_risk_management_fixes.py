from __future__ import annotations

import threading

import main
from core.strategy_narrative import ActiveTrade


def _trade(symbol="EURUSD"):
    trade = ActiveTrade(
        side="LONG",
        entry=1.0,
        stop=0.95,
        tp_prices=[1.05, 1.10, 1.15],
        tf="15M",
        narrative="test",
        symbol=symbol,
    )
    trade.volume = 0.3
    trade.volume_remaining = 0.3
    return trade


def test_small_volume_allocation_prioritizes_nearest_targets():
    assert main._compute_tp_volumes(0.01, 3, step=0.01) == [0.01, 0.0, 0.0]
    assert main._compute_tp_volumes(0.02, 3, step=0.01) == [0.01, 0.01, 0.0]
    assert main._compute_tp_volumes(0.03, 3, step=0.01) == [0.01, 0.01, 0.01]

    allocated = main._compute_tp_volumes(0.31, 3, step=0.01)
    assert sum(allocated) == 0.31
    assert allocated[0] >= allocated[1] >= allocated[2]


class CloseExecutor:
    def __init__(self, close_result=True):
        self.close_result = close_result
        self.calls = []

    def get_current_price(self, symbol, side):
        return 1.02

    def close_trade(self, symbol, *, position_id, volume):
        self.calls.append((symbol, position_id, volume))
        return self.close_result


def _management_core(trade, executor):
    core = main.Core.__new__(main.Core)
    core.mt5_executor = executor
    core.active_trades = {trade.symbol: trade}
    core._management_lock = threading.RLock()
    core._broker_missing_counts = {}
    core._broker_missing_confirm = 2
    core._log_signal = lambda *args, **kwargs: None
    return core


def test_fast_management_executes_scheduled_flat_without_candles(monkeypatch):
    trade = _trade()
    trade.mt5_position_id = 77
    executor = CloseExecutor(close_result=True)
    core = _management_core(trade, executor)
    monkeypatch.setattr(core, "_is_friday_weekend_close", lambda: False)
    monkeypatch.setattr(core, "_is_daily_flat_close", lambda: True)
    monkeypatch.setattr(main, "save_active_trades", lambda *args, **kwargs: None)

    result = core.manage_active_trades()

    assert result["EURUSD"]["signal"] == "EXIT_TIME"
    assert "EURUSD" not in core.active_trades
    assert executor.calls == [("EURUSD", 77, 0.3)]


def test_unconfirmed_scheduled_close_keeps_trade_for_retry(monkeypatch):
    trade = _trade()
    trade.mt5_position_id = 77
    executor = CloseExecutor(close_result=False)
    core = _management_core(trade, executor)
    monkeypatch.setattr(core, "_is_friday_weekend_close", lambda: False)
    monkeypatch.setattr(core, "_is_daily_flat_close", lambda: True)
    monkeypatch.setattr(main, "save_active_trades", lambda *args, **kwargs: None)

    result = core.manage_active_trades()

    assert result["EURUSD"]["signal"] == "HOLD"
    assert "EURUSD" in core.active_trades


def test_scheduled_single_close_attaches_broker_net(monkeypatch):
    trade = _trade()
    trade.mt5_position_id = 77

    class MetricsExecutor(CloseExecutor):
        def get_position_close_info(self, position_id):
            assert position_id == 77
            return {
                "price": 1.02,
                "volume": 0.3,
                "profit": 25.0,
                "commission": -2.0,
                "swap": 0.0,
                "fee": -0.5,
                "net": 22.5,
            }

    executor = MetricsExecutor(close_result=True)
    core = _management_core(trade, executor)
    monkeypatch.setattr(core, "_is_friday_weekend_close", lambda: False)
    monkeypatch.setattr(core, "_is_daily_flat_close", lambda: True)
    monkeypatch.setattr(main, "save_active_trades", lambda *args, **kwargs: None)

    result = core.manage_active_trades()["EURUSD"]

    assert result["signal"] == "EXIT_TIME"
    assert result["pnl_complete"] is True
    assert result["realized_net"] == 22.5
    assert result["outcome"] == "TP"


def test_mixed_tp_and_be_stop_uses_realized_net_outcome():
    trade = _trade()
    trade.tp_hit = 1
    trade.split_legs = {
        1: {"status": "closed", "close_reason": "TP", "close_profit": 12.0,
            "close_commission": -0.5, "close_swap": 0.0, "close_volume": 0.15,
            "close_price": 1.05},
        2: {"status": "closed", "close_reason": "SL", "close_profit": 0.0,
            "close_commission": -0.3, "close_swap": 0.0, "close_volume": 0.09,
            "close_price": 1.0},
        3: {"status": "closed", "close_reason": "SL", "close_profit": 0.0,
            "close_commission": -0.2, "close_swap": 0.0, "close_volume": 0.06,
            "close_price": 1.0},
    }
    updates = []
    core = main.Core.__new__(main.Core)
    core.mt5_executor = type(
        "Executor", (), {"get_position_close_reason": lambda self, pid: "SL"}
    )()
    core.ai_store = type(
        "Store", (), {"update_on_close": lambda self, *args, **kwargs: updates.append((args, kwargs))}
    )()
    core._entry_cooldowns = {}
    core._post_sl_cooldown_sec = 3600.0
    manage = {}

    core._register_broker_close("EURUSD", trade, manage)

    assert manage["outcome"] == "TP"
    assert manage["realized_net"] == 11.0
    assert manage["pnl_complete"] is True
    assert "EURUSD" not in core._entry_cooldowns
    assert updates and updates[0][0][1] == "TP"


def test_non_split_broker_close_never_reports_false_zero_net():
    trade = _trade()
    trade.mt5_position_id = 77
    updates = []
    core = main.Core.__new__(main.Core)
    core.mt5_executor = type(
        "Executor", (), {"get_position_close_reason": lambda self, pid: "TP"}
    )()
    core.ai_store = type(
        "Store", (), {"update_on_close": lambda self, *args, **kwargs: updates.append((args, kwargs))}
    )()
    core._entry_cooldowns = {}
    core._post_sl_cooldown_sec = 3600.0
    manage = {}

    core._register_broker_close("EURUSD", trade, manage)

    assert manage["outcome"] == "TP"
    assert manage["pnl_complete"] is False
    assert "realized_net" not in manage
    assert updates and updates[0][0][1] == "TP"
