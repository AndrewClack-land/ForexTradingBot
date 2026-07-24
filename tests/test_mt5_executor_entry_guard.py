from __future__ import annotations

from types import SimpleNamespace

import pytest

from executors import mt5_executor


def _executor():
    executor = mt5_executor.MT5Executor.__new__(mt5_executor.MT5Executor)
    executor.logger = SimpleNamespace(
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
    )
    return executor


def _market(monkeypatch, *, bid: float, ask: float, point: float = 0.0001):
    monkeypatch.setattr(
        mt5_executor.mt5,
        "symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=bid, ask=ask),
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "symbol_info",
        lambda symbol: SimpleNamespace(
            point=point,
            tick_size=point,
            trade_stops_level=0,
            trade_freeze_level=0,
            spread=max(1, round((ask - bid) / point)),
        ),
    )


@pytest.mark.parametrize(
    ("side", "bid", "ask", "entry_min", "entry_max"),
    [
        ("LONG", 1.1010, 1.1012, 1.0990, 1.1000),
        ("SHORT", 1.0988, 1.0990, 1.1000, 1.1010),
    ],
)
def test_send_order_rejects_price_outside_entry_zone(
    monkeypatch, side, bid, ask, entry_min, entry_max
):
    _market(monkeypatch, bid=bid, ask=ask)
    executor = _executor()

    with pytest.raises(RuntimeError, match="outside entry range"):
        executor._send_order(
            "EURUSD",
            side,
            0.1,
            1.1000,
            1.0950 if side == "LONG" else 1.1050,
            1.1100 if side == "LONG" else 1.0900,
            None,
            entry_min=entry_min,
            entry_max=entry_max,
        )


@pytest.mark.parametrize(
    ("side", "stop", "target"),
    [
        ("LONG", 1.0950, 1.1001),
        ("SHORT", 1.1050, 1.1001),
    ],
)
def test_send_order_rejects_target_already_reached(monkeypatch, side, stop, target):
    _market(monkeypatch, bid=1.1000, ask=1.1002)
    executor = _executor()

    with pytest.raises(RuntimeError, match="target .* already reached"):
        executor._send_order(
            "EURUSD",
            side,
            0.1,
            1.1000,
            stop,
            target,
            None,
        )


def test_execute_entry_forwards_entry_zone(monkeypatch):
    _market(monkeypatch, bid=1.1000, ask=1.1002)
    executor = _executor()
    executor._calc_volume = lambda symbol, entry, stop, *, side: 0.1
    captured = {}

    def fake_send_order(*args, **kwargs):
        captured.update(kwargs)
        return {
            "price": 1.1002,
            "deal": 44,
            "ticket": 33,
            "volume": 0.1,
            "stop_price": 1.0950,
            "risk_amount": 10.0,
            "risk_budget_amount": 10.0,
            "risk_capital_base": 1000.0,
            "risk_pct": 0.01,
        }

    executor._send_order = fake_send_order
    executor._find_position_id_from_deal = lambda deal: 55

    result = executor.execute_entry(
        "EURUSD",
        side="LONG",
        entry_price=1.1000,
        stop_price=1.0950,
        tp_price=1.1100,
        entry_min=1.0995,
        entry_max=1.1005,
    )

    assert captured["entry_min"] == 1.0995
    assert captured["entry_max"] == 1.1005
    assert result["position_id"] == 55


def test_split_subminimum_volumes_are_merged_into_tp1(monkeypatch):
    executor = _executor()
    executor.settings = SimpleNamespace(magic=123)
    risk_limit = mt5_executor._RiskLimit(
        capital_base=1000.0,
        budget_amount=10.0,
        fraction=0.01,
        margin_free=1000.0,
        account_currency="USD",
    )
    executor._risk_limit = lambda: risk_limit
    executor._size_volume_for_risk = lambda *args, **kwargs: mt5_executor._RiskSizing(
        volume=0.1,
        risk_amount=10.0,
        risk_per_lot=100.0,
        limit=risk_limit,
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "symbol_info",
        lambda symbol: SimpleNamespace(volume_min=0.1, volume_step=0.01),
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=0.99, ask=1.0),
    )
    sent = []

    def fake_send(symbol, side, volume, entry, stop, tp, comment, **kwargs):
        sent.append((volume, tp))
        return {
            "ticket": 10,
            "deal": 20,
            "price": entry,
            "volume": volume,
            "stop_price": stop,
            "risk_amount": volume * 100.0,
            "risk_budget_amount": risk_limit.budget_amount,
            "risk_capital_base": risk_limit.capital_base,
            "risk_pct": risk_limit.fraction,
        }

    executor._send_order = fake_send
    executor._find_position_id_from_deal = lambda deal: 30

    legs = executor.execute_split_entry(
        "EURUSD",
        side="LONG",
        entry_price=1.0,
        stop_price=0.9,
        tp_prices=[1.1, 1.2, 1.3],
        volumes_per_tp=[0.05, 0.03, 0.02],
    )

    assert sent == [(0.1, 1.1)]
    assert legs[0]["tp_index"] == 1
