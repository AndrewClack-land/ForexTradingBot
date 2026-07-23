from __future__ import annotations

from types import SimpleNamespace

import pytest

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
    assert info["volume"] == 0.1
    assert info["profit"] == 9.0
    assert info["commission"] == -0.2
    assert info["fee"] == 0.0
    assert info["net"] == 8.8
    assert executor.get_position_close_reason(123) is None


def test_close_info_includes_entry_commission_and_all_broker_fees(monkeypatch):
    entry = SimpleNamespace(
        entry=0, reason=3, ticket=1, time=10, time_msc=10_000,
        price=1.0, volume=0.1, profit=0.0, commission=-0.30, swap=0.0, fee=-0.05,
    )
    exit_deal = SimpleNamespace(
        entry=1, reason=5, ticket=2, time=20, time_msc=20_000,
        price=1.1, volume=0.1, profit=5.0, commission=-0.10, swap=0.0, fee=-0.02,
    )
    monkeypatch.setattr(mt5_executor.mt5, "DEAL_ENTRY_OUT", 1, raising=False)
    monkeypatch.setattr(mt5_executor.mt5, "DEAL_REASON_TP", 5, raising=False)
    monkeypatch.setattr(
        mt5_executor.mt5,
        "history_deals_get",
        lambda **kwargs: (entry, exit_deal),
    )
    executor = mt5_executor.MT5Executor.__new__(mt5_executor.MT5Executor)

    info = executor.get_position_close_info(123)

    assert info["profit"] == 5.0
    assert info["commission"] == pytest.approx(-0.4)
    assert info["fee"] == pytest.approx(-0.07)
    assert info["net"] == pytest.approx(4.53)


def test_close_trade_uses_strict_ticket_lookup(monkeypatch):
    executor = mt5_executor.MT5Executor.__new__(mt5_executor.MT5Executor)
    executor.logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
    calls = []
    executor._find_position = lambda symbol, ticket, strict=False: calls.append(
        (symbol, ticket, strict)
    )

    assert executor.close_trade("EURUSD", position_id=77, volume=None) is False
    assert calls == [("EURUSD", 77, True)]


def test_move_stop_defers_when_exact_be_is_inside_broker_gap(monkeypatch):
    executor = mt5_executor.MT5Executor.__new__(mt5_executor.MT5Executor)
    executor.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    executor._find_position = lambda *args, **kwargs: SimpleNamespace(
        ticket=77,
        type=mt5_executor.mt5.POSITION_TYPE_BUY,
        sl=0.9950,
        tp=1.0200,
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=1.0001, ask=1.0003),
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "symbol_info",
        lambda symbol: SimpleNamespace(
            point=0.0001,
            tick_size=0.0001,
            trade_stops_level=0,
            spread=2,
            trade_freeze_level=0,
        ),
    )
    monkeypatch.setattr(
        mt5_executor,
        "_send_request",
        lambda request: (_ for _ in ()).throw(AssertionError("must not modify a clamped BE")),
    )

    assert executor.move_stop("EURUSD", position_id=77, new_stop=1.0000) is False


def test_move_stop_treats_already_better_stop_as_idempotent_success(monkeypatch):
    executor = mt5_executor.MT5Executor.__new__(mt5_executor.MT5Executor)
    executor.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    executor._find_position = lambda *args, **kwargs: SimpleNamespace(
        ticket=77,
        type=mt5_executor.mt5.POSITION_TYPE_BUY,
        sl=1.0010,
        tp=1.0200,
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=1.0100, ask=1.0102),
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "symbol_info",
        lambda symbol: SimpleNamespace(
            point=0.0001,
            tick_size=0.0001,
            trade_stops_level=0,
            spread=2,
            trade_freeze_level=0,
        ),
    )
    monkeypatch.setattr(
        mt5_executor,
        "_send_request",
        lambda request: (_ for _ in ()).throw(AssertionError("already protected")),
    )

    assert executor.move_stop("EURUSD", position_id=77, new_stop=1.0000) is True
