from __future__ import annotations

from types import SimpleNamespace

import pytest

from executors import mt5_executor


def _executor():
    executor = mt5_executor.MT5Executor.__new__(mt5_executor.MT5Executor)
    executor.logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
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
    executor._calc_volume = lambda symbol, entry, stop: 0.1
    captured = {}

    def fake_send_order(*args, **kwargs):
        captured.update(kwargs)
        return {"price": 1.1002, "deal": 44, "ticket": 33}

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
