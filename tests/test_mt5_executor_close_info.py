from __future__ import annotations

from types import SimpleNamespace

from executors import mt5_executor


def test_close_info_uses_latest_exit_deal_and_keeps_non_tp_reason(monkeypatch):
    entry = SimpleNamespace(
        entry=0, reason=3, ticket=1, time=10, time_msc=10_000,
        price=1.0, volume=0.1, profit=0.0, commission=0.0, swap=0.0,
    )
    older_exit = SimpleNamespace(
        entry=1, reason=5, ticket=2, time=20, time_msc=20_000,
        price=1.1, volume=0.05, profit=5.0, commission=-0.1, swap=0.0,
    )
    latest_exit = SimpleNamespace(
        entry=1, reason=0, ticket=3, time=30, time_msc=30_000,
        price=1.09, volume=0.05, profit=4.0, commission=-0.1, swap=0.0,
    )
    monkeypatch.setattr(mt5_executor.mt5, "DEAL_ENTRY_OUT", 1, raising=False)
    monkeypatch.setattr(mt5_executor.mt5, "DEAL_REASON_TP", 5, raising=False)
    monkeypatch.setattr(mt5_executor.mt5, "DEAL_REASON_CLIENT", 0, raising=False)
    monkeypatch.setattr(
        mt5_executor.mt5,
        "history_deals_get",
        lambda **kwargs: (entry, older_exit, latest_exit),
    )
    executor = mt5_executor.MT5Executor.__new__(mt5_executor.MT5Executor)

    info = executor.get_position_close_info(123)

    assert info["deal_ticket"] == 3
    assert info["reason"] == "MANUAL"
    assert executor.get_position_close_reason(123) is None


def test_close_trade_uses_strict_ticket_lookup(monkeypatch):
    executor = mt5_executor.MT5Executor.__new__(mt5_executor.MT5Executor)
    executor.logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
    calls = []
    executor._find_position = lambda symbol, ticket, strict=False: calls.append(
        (symbol, ticket, strict)
    )

    assert executor.close_trade("EURUSD", position_id=77, volume=None) is False
    assert calls == [("EURUSD", 77, True)]
