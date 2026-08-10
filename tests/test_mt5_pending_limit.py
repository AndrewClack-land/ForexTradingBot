from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from executors import mt5_executor


def _executor():
    executor = mt5_executor.MT5Executor.__new__(mt5_executor.MT5Executor)
    executor.settings = SimpleNamespace(magic=4242, slippage=12)
    executor.logger = SimpleNamespace(
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
    )
    return executor


def _install_constants(monkeypatch):
    constants = {
        "SYMBOL_ORDER_LIMIT": 2,
        "SYMBOL_EXPIRATION_SPECIFIED": 4,
        "ORDER_TYPE_BUY_LIMIT": 2,
        "ORDER_TYPE_SELL_LIMIT": 3,
        "ORDER_TYPE_BUY_STOP": 4,
        "ORDER_TYPE_SELL_STOP": 5,
        "ORDER_TYPE_BUY_STOP_LIMIT": 6,
        "ORDER_TYPE_SELL_STOP_LIMIT": 7,
        "ORDER_FILLING_RETURN": 2,
        "ORDER_TIME_SPECIFIED": 2,
        "TRADE_ACTION_PENDING": 5,
        "TRADE_ACTION_REMOVE": 8,
        "TRADE_RETCODE_PLACED": 10008,
        "TRADE_RETCODE_DONE": 10009,
        "TRADE_RETCODE_DONE_PARTIAL": 10010,
        "TRADE_RETCODE_ORDER_CHANGED": 10023,
        "DEAL_ENTRY_IN": 0,
    }
    for name, value in constants.items():
        monkeypatch.setattr(mt5_executor.mt5, name, value, raising=False)


def _info(**overrides):
    payload = {
        "order_mode": 2,
        "expiration_mode": 4,
        "point": 0.0001,
        "trade_tick_size": 0.0001,
        "digits": 4,
        "trade_stops_level": 1,
        "trade_freeze_level": 0,
        "spread": 1,
        "volume_min": 0.01,
        "volume_step": 0.01,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_prepare_limit_entry_requires_server_side_expiry(monkeypatch):
    _install_constants(monkeypatch)
    executor = _executor()
    executor._get_symbol_info = lambda symbol: _info(expiration_mode=1)

    with pytest.raises(RuntimeError, match="specified expiration"):
        executor.prepare_limit_entry(
            "EURUSD",
            side="LONG",
            limit_price=1.1000,
            stop_price=1.0950,
        )


def test_prepare_limit_entry_sizes_at_exact_limit(monkeypatch):
    _install_constants(monkeypatch)
    executor = _executor()
    info = _info()
    executor._get_symbol_info = lambda symbol: info
    captured = {}

    def size(symbol, side, entry, stop):
        captured.update(
            symbol=symbol, side=side, entry=entry, stop=stop
        )
        limit = SimpleNamespace(
            budget_amount=100.0,
            capital_base=10_000.0,
            fraction=0.01,
            account_currency="USD",
        )
        return SimpleNamespace(
            volume=0.30,
            risk_amount=99.0,
            risk_per_lot=330.0,
            limit=limit,
        )

    executor._size_volume_for_risk = size
    result = executor.prepare_limit_entry(
        "EURUSD",
        side="LONG",
        limit_price=1.1000,
        stop_price=1.0950,
    )

    assert captured == {
        "symbol": "EURUSD",
        "side": "LONG",
        "entry": 1.1000,
        "stop": 1.0950,
    }
    assert result["volume"] == 0.30
    assert result["risk_budget_amount"] == 100.0


def test_place_limit_leg_uses_specified_expiry_and_return(monkeypatch):
    _install_constants(monkeypatch)
    executor = _executor()
    info = _info()
    executor._get_symbol_info = lambda symbol: info
    executor._risk_per_lot = (
        lambda symbol, side, entry, stop, *, info: 300.0
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=1.1048, ask=1.1050),
    )
    sent = {}

    def fake_send(request):
        sent.update(request)
        return SimpleNamespace(
            retcode=10008,
            order=7001,
            deal=0,
            volume=0.10,
            price=1.1000,
        )

    monkeypatch.setattr(mt5_executor, "_send_request", fake_send)
    expiry = datetime.now(timezone.utc).timestamp() + 900
    result = executor.place_limit_leg(
        "EURUSD",
        side="LONG",
        volume=0.10,
        limit_price=1.1000,
        stop_price=1.0950,
        tp_price=1.1100,
        expires_at=expiry,
        comment="FBPL:abc TP1",
        risk_budget_amount=100.0,
    )

    assert sent["action"] == 5
    assert sent["type"] == 2
    assert sent["type_time"] == 2
    assert sent["type_filling"] == 2
    assert sent["expiration"] == int(expiry)
    assert result["ticket"] == 7001


def test_place_limit_leg_rejects_non_retest_side_of_market(monkeypatch):
    _install_constants(monkeypatch)
    executor = _executor()
    executor._get_symbol_info = lambda symbol: _info()
    monkeypatch.setattr(
        mt5_executor.mt5,
        "symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=1.0998, ask=1.1000),
    )

    with pytest.raises(RuntimeError, match="not below ASK"):
        executor.place_limit_leg(
            "EURUSD",
            side="LONG",
            volume=0.10,
            limit_price=1.1000,
            stop_price=1.0950,
            tp_price=1.1100,
            expires_at=datetime.now(timezone.utc).timestamp() + 900,
            comment="FBPL:abc TP1",
            risk_budget_amount=100.0,
        )


def test_cancel_pending_order_is_owned_and_idempotent(monkeypatch):
    _install_constants(monkeypatch)
    executor = _executor()
    order = SimpleNamespace(
        ticket=77, magic=4242, comment="FBPL:abc:T1"
    )
    rows = {77: [order], 78: []}
    monkeypatch.setattr(
        mt5_executor.mt5,
        "orders_get",
        lambda *, ticket: rows[ticket],
    )
    sent = []
    monkeypatch.setattr(
        mt5_executor,
        "_send_request",
        lambda request: (
            sent.append(request)
            or SimpleNamespace(retcode=10009)
        ),
    )

    assert executor.cancel_pending_order(
        77, expected_comment_prefix="FBPL:abc"
    ) is True
    assert executor.cancel_pending_order(
        78, expected_comment_prefix="FBPL:abc"
    ) is False
    assert sent == [{"action": 8, "order": 77}]


def test_get_pending_order_fill_aggregates_opening_deals(monkeypatch):
    _install_constants(monkeypatch)
    executor = _executor()
    deals = [
        SimpleNamespace(
            ticket=1,
            order=77,
            magic=4242,
            entry=0,
            volume=0.04,
            price=1.1000,
            position_id=901,
            time_msc=1000,
        ),
        SimpleNamespace(
            ticket=2,
            order=77,
            magic=4242,
            entry=0,
            volume=0.06,
            price=1.1002,
            position_id=901,
            time_msc=1001,
        ),
        SimpleNamespace(
            ticket=3,
            order=77,
            magic=4242,
            entry=1,
            volume=0.10,
            price=1.1100,
            position_id=901,
            time_msc=2000,
        ),
    ]
    monkeypatch.setattr(
        mt5_executor.mt5,
        "history_deals_get",
        lambda start, end: deals,
    )

    fill = executor.get_pending_order_fill(77)

    assert fill is not None
    assert fill["volume"] == pytest.approx(0.10)
    assert fill["price"] == pytest.approx(1.10012)
    assert fill["position_ids"] == [901]
    assert fill["deal_tickets"] == [1, 2]


def test_none_live_order_book_is_not_treated_as_empty(monkeypatch):
    _install_constants(monkeypatch)
    executor = _executor()
    monkeypatch.setattr(
        mt5_executor.mt5, "orders_get", lambda **kwargs: None
    )
    monkeypatch.setattr(
        mt5_executor.mt5, "last_error", lambda: (-1, "disconnected")
    )

    with pytest.raises(RuntimeError, match="orders_get failed"):
        executor.list_pending_orders()
    with pytest.raises(RuntimeError, match=r"orders_get\(77\) failed"):
        executor.cancel_pending_order(
            77, expected_comment_prefix="FBPL:abc"
        )


def test_cancel_requires_magic_and_plan_comment(monkeypatch):
    _install_constants(monkeypatch)
    executor = _executor()
    order = SimpleNamespace(
        ticket=77, magic=4242, comment="FBPL:different:T1"
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "orders_get",
        lambda *, ticket: [order],
    )

    with pytest.raises(RuntimeError, match="does not match"):
        executor.cancel_pending_order(
            77, expected_comment_prefix="FBPL:expected"
        )
