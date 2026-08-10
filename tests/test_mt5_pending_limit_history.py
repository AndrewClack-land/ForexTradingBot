from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from executors import mt5_executor


def _executor():
    executor = mt5_executor.MT5Executor.__new__(mt5_executor.MT5Executor)
    executor.settings = SimpleNamespace(magic=4242, slippage=12)
    return executor


def _install_constants(monkeypatch):
    constants = {
        "SYMBOL_ORDER_LIMIT": 2,
        "SYMBOL_EXPIRATION_SPECIFIED": 4,
        "ORDER_TYPE_BUY_LIMIT": 2,
        "ORDER_TYPE_SELL_LIMIT": 3,
        "ORDER_STATE_CANCELED": 2,
        "ORDER_STATE_PARTIAL": 3,
        "ORDER_STATE_FILLED": 4,
        "ORDER_STATE_REJECTED": 5,
        "ORDER_STATE_EXPIRED": 6,
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


def test_batch_history_filters_exact_comment_and_magic_and_reports_execution(
    monkeypatch,
):
    _install_constants(monkeypatch)
    executor = _executor()
    comments = ["FBPL:plan:T1", "FBPL:plan:T2", "FBPL:plan:T3"]
    rows = [
        SimpleNamespace(
            ticket=11,
            magic=4242,
            comment=comments[0],
            state=2,
            type=2,
            volume_initial=0.30,
            volume_current=0.20,
            price_open=1.1000,
            time_done_msc=2000,
        ),
        SimpleNamespace(
            ticket=12,
            magic=4242,
            comment=comments[1],
            state=2,
            type=2,
            volume_initial=0.30,
            volume_current=0.30,
            price_open=1.1000,
            time_done_msc=2001,
        ),
        # FILLED/PARTIAL state is affirmative evidence even if a historical
        # adapter exposes no useful initial-current delta.
        SimpleNamespace(
            ticket=13,
            magic=4242,
            comment=comments[2],
            state=4,
            type=2,
            volume_initial=0.30,
            volume_current=0.30,
            price_open=1.1000,
            time_done_msc=2002,
        ),
        SimpleNamespace(
            ticket=21,
            magic=9999,
            comment=comments[0],
            state=4,
            type=2,
            volume_initial=0.30,
            volume_current=0.0,
            price_open=1.1000,
            time_done_msc=2003,
        ),
        SimpleNamespace(
            ticket=22,
            magic=4242,
            comment=f"{comments[0]}-suffix",
            state=4,
            type=2,
            volume_initial=0.30,
            volume_current=0.0,
            price_open=1.1000,
            time_done_msc=2004,
        ),
        SimpleNamespace(
            ticket=0,
            magic=4242,
            comment=comments[0],
            state=4,
            type=2,
            volume_initial=0.30,
            volume_current=0.0,
            price_open=1.1000,
            time_done_msc=2005,
        ),
    ]
    monkeypatch.setattr(
        mt5_executor.mt5,
        "history_orders_get",
        lambda start, end: rows,
    )

    result = executor.get_pending_order_histories(comments)

    assert set(result) == set(comments)
    assert [row["ticket"] for values in result.values() for row in values] == [
        11,
        12,
        13,
    ]
    assert result[comments[0]][0]["executed_volume"] == pytest.approx(0.10)
    assert result[comments[0]][0]["had_execution"] is True
    assert result[comments[1]][0]["executed_volume"] == 0.0
    assert result[comments[1]][0]["had_execution"] is False
    assert result[comments[2]][0]["executed_volume"] == 0.0
    assert result[comments[2]][0]["had_execution"] is True


def test_history_query_none_fails_closed_for_batch_and_ticket(monkeypatch):
    _install_constants(monkeypatch)
    executor = _executor()
    monkeypatch.setattr(
        mt5_executor.mt5,
        "history_orders_get",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "last_error",
        lambda: (-1, "disconnected"),
    )

    with pytest.raises(RuntimeError, match="history_orders_get range failed"):
        executor.get_pending_order_histories(["FBPL:plan:T1"])
    with pytest.raises(RuntimeError, match=r"history_orders_get[(]77[)] failed"):
        executor.get_pending_order_history(77)


def test_prepare_limit_entry_rejects_unknown_order_mode(monkeypatch):
    _install_constants(monkeypatch)
    executor = _executor()
    executor._get_symbol_info = lambda symbol: _info(order_mode=0)
    executor._size_volume_for_risk = lambda *args, **kwargs: pytest.fail(
        "sizing must not run when LIMIT support is unknown"
    )

    with pytest.raises(RuntimeError, match="LIMIT orders are not allowed"):
        executor.prepare_limit_entry(
            "EURUSD",
            side="LONG",
            limit_price=1.1000,
            stop_price=1.0950,
        )


def test_place_limit_leg_rejects_unknown_order_mode_before_send(monkeypatch):
    _install_constants(monkeypatch)
    executor = _executor()
    executor._get_symbol_info = lambda symbol: _info(order_mode=0)
    monkeypatch.setattr(
        mt5_executor.mt5,
        "symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=1.1048, ask=1.1050),
    )
    monkeypatch.setattr(
        mt5_executor,
        "_send_request",
        lambda request: pytest.fail("order_send must not run"),
    )

    with pytest.raises(RuntimeError, match="LIMIT orders are not allowed"):
        executor.place_limit_leg(
            "EURUSD",
            side="LONG",
            volume=0.10,
            limit_price=1.1000,
            stop_price=1.0950,
            tp_price=1.1100,
            expires_at=datetime.now(timezone.utc).timestamp() + 900,
            comment="FBPL:plan:T1",
            risk_budget_amount=100.0,
        )


@pytest.mark.parametrize(
    "risk_budget",
    [None, 0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_place_limit_leg_rejects_invalid_required_risk_budget(
    monkeypatch,
    risk_budget,
):
    _install_constants(monkeypatch)
    executor = _executor()
    executor._get_symbol_info = lambda symbol: _info()
    executor._risk_per_lot = (
        lambda symbol, side, entry, stop, *, info: 300.0
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=1.1048, ask=1.1050),
    )
    monkeypatch.setattr(
        mt5_executor,
        "_send_request",
        lambda request: pytest.fail("order_send must not run"),
    )

    with pytest.raises(mt5_executor.RiskLimitError, match="risk budget"):
        executor.place_limit_leg(
            "EURUSD",
            side="LONG",
            volume=0.10,
            limit_price=1.1000,
            stop_price=1.0950,
            tp_price=1.1100,
            expires_at=datetime.now(timezone.utc).timestamp() + 900,
            comment="FBPL:plan:T1",
            risk_budget_amount=risk_budget,
        )


def test_place_limit_leg_enforces_cumulative_risk_budget(monkeypatch):
    _install_constants(monkeypatch)
    executor = _executor()
    executor._get_symbol_info = lambda symbol: _info()
    executor._risk_per_lot = (
        lambda symbol, side, entry, stop, *, info: 300.0
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=1.1048, ask=1.1050),
    )
    monkeypatch.setattr(
        mt5_executor,
        "_send_request",
        lambda request: pytest.fail("order_send must not run"),
    )

    with pytest.raises(mt5_executor.RiskLimitError, match="cumulative risk"):
        executor.place_limit_leg(
            "EURUSD",
            side="LONG",
            volume=0.10,
            limit_price=1.1000,
            stop_price=1.0950,
            tp_price=1.1100,
            expires_at=datetime.now(timezone.utc).timestamp() + 900,
            comment="FBPL:plan:T1",
            risk_budget_amount=100.0,
            risk_used_amount=80.0,
        )
