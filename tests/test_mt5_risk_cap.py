from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from executors import mt5_executor


def _executor(
    *,
    risk_pct: float = 0.01,
    commission_per_lot: float = 0.0,
    max_volume: float = 100.0,
    initial_capital: float = 10_000.0,
    risk_state_path=None,
    login: int = 123456,
    slippage: int = 0,
):
    executor = mt5_executor.MT5Executor.__new__(mt5_executor.MT5Executor)
    executor.settings = SimpleNamespace(
        login=login,
        risk_pct=risk_pct,
        initial_capital=initial_capital,
        risk_state_path=(
            str(risk_state_path) if risk_state_path is not None else None
        ),
        commission_per_lot=commission_per_lot,
        max_volume=max_volume,
        magic=123456,
        slippage=slippage,
    )
    executor.logger = SimpleNamespace(
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
    )
    executor._fill_mode_cache = {}
    executor._initial_capital = None
    return executor


def _account(
    monkeypatch,
    *,
    balance: float = 10_000.0,
    equity: float = 10_000.0,
    margin_free: float = 10_000.0,
    login: int = 123456,
):
    monkeypatch.setattr(
        mt5_executor.mt5,
        "account_info",
        lambda: SimpleNamespace(
            login=login,
            balance=balance,
            equity=equity,
            margin_free=margin_free,
            currency="USD",
        ),
    )
    monkeypatch.setattr(mt5_executor.mt5, "last_error", lambda: (0, "ok"))


def _symbol_info(
    *,
    point: float = 1.0,
    volume_min: float = 0.001,
    volume_max: float = 100.0,
    volume_step: float = 0.001,
    stops_level: int = 0,
    spread: int = 1,
):
    return SimpleNamespace(
        visible=True,
        point=point,
        trade_tick_size=point,
        tick_size=point,
        volume_min=volume_min,
        volume_max=volume_max,
        volume_step=volume_step,
        trade_stops_level=stops_level,
        trade_freeze_level=0,
        spread=spread,
        fillings=0,
    )


def _sizing_market(
    monkeypatch,
    *,
    info=None,
    loss_per_lot: float = 100.0,
):
    info = info or _symbol_info()
    monkeypatch.setattr(mt5_executor.mt5, "symbol_info", lambda symbol: info)
    monkeypatch.setattr(mt5_executor.mt5, "symbol_select", lambda symbol, selected: True)
    monkeypatch.setattr(
        mt5_executor.mt5,
        "order_calc_profit",
        lambda order_type, symbol, volume, entry, stop: -loss_per_lot * volume,
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "order_calc_margin",
        lambda order_type, symbol, volume, entry: None,
    )
    return info


def test_risk_limit_hard_caps_configured_two_percent_at_one_percent(monkeypatch):
    executor = _executor(risk_pct=0.02)
    _account(monkeypatch, balance=10_000.0, equity=12_000.0)

    limit = executor._risk_limit()

    assert limit.fraction == pytest.approx(0.01)
    assert limit.capital_base == pytest.approx(10_000.0)
    assert limit.budget_amount == pytest.approx(100.0)


def test_risk_limit_preserves_lower_half_percent(monkeypatch):
    executor = _executor(risk_pct=0.005)
    _account(monkeypatch, balance=10_000.0, equity=10_000.0)

    limit = executor._risk_limit()

    assert limit.fraction == pytest.approx(0.005)
    assert limit.budget_amount == pytest.approx(50.0)


@pytest.mark.parametrize(
    ("balance", "equity"),
    [
        (20_000.0, 25_000.0),
        (4_000.0, 3_000.0),
    ],
)
def test_risk_limit_stays_on_fixed_initial_capital(monkeypatch, balance, equity):
    executor = _executor(initial_capital=7_500.0)
    _account(monkeypatch, balance=balance, equity=equity)

    limit = executor._risk_limit()

    assert limit.capital_base == pytest.approx(7_500.0)
    assert limit.budget_amount == pytest.approx(75.0)


def test_auto_captured_initial_capital_reloads_for_same_account_after_restart(
    monkeypatch, tmp_path
):
    state_path = tmp_path / "risk_capital.json"
    first = _executor(
        initial_capital=0.0,
        risk_state_path=state_path,
        login=778899,
    )
    _account(
        monkeypatch,
        balance=6_000.0,
        equity=5_500.0,
        login=778899,
    )

    first_limit = first._risk_limit()
    persisted = json.loads(state_path.read_text(encoding="utf-8"))

    assert first_limit.capital_base == pytest.approx(6_000.0)
    assert first_limit.budget_amount == pytest.approx(60.0)
    assert persisted["account_login"] == 778899
    assert persisted["initial_capital"] == pytest.approx(6_000.0)

    restarted = _executor(
        initial_capital=0.0,
        risk_state_path=state_path,
        login=778899,
    )
    _account(
        monkeypatch,
        balance=15_000.0,
        equity=14_000.0,
        login=778899,
    )

    restarted_limit = restarted._risk_limit()

    assert restarted_limit.capital_base == pytest.approx(6_000.0)
    assert restarted_limit.budget_amount == pytest.approx(60.0)


def test_small_account_has_no_one_dollar_floor(monkeypatch):
    executor = _executor(initial_capital=50.0)
    _account(monkeypatch, balance=50.0, equity=50.0)
    _sizing_market(monkeypatch, loss_per_lot=100.0)

    sizing = executor._size_volume_for_risk("TEST", "LONG", 100.0, 90.0)

    assert sizing.limit.budget_amount == pytest.approx(0.50)
    assert sizing.volume == pytest.approx(0.005)
    assert sizing.risk_amount == pytest.approx(0.50)


@pytest.mark.parametrize(
    ("side", "stop", "expected_target"),
    [
        ("LONG", 90.0, 90.5),
        ("SHORT", 110.0, 109.5),
    ],
)
def test_minimum_lot_returns_risk_compatible_entry_nearer_to_stop(
    monkeypatch, side, stop, expected_target
):
    executor = _executor(initial_capital=50.0)
    _account(monkeypatch)
    info = _symbol_info(
        point=0.1,
        volume_min=0.01,
        volume_step=0.01,
        spread=1,
    )
    monkeypatch.setattr(mt5_executor.mt5, "symbol_info", lambda symbol: info)
    monkeypatch.setattr(mt5_executor.mt5, "symbol_select", lambda symbol, selected: True)

    def directional_profit(order_type, symbol, volume, entry, exit_price):
        if order_type == mt5_executor.mt5.ORDER_TYPE_BUY:
            return (exit_price - entry) * 100.0 * volume
        return (entry - exit_price) * 100.0 * volume

    monkeypatch.setattr(mt5_executor.mt5, "order_calc_profit", directional_profit)
    monkeypatch.setattr(
        mt5_executor.mt5,
        "order_calc_margin",
        lambda order_type, symbol, volume, entry: None,
    )

    with pytest.raises(mt5_executor.RiskCapacityError) as raised:
        executor._size_volume_for_risk("TEST", side, 100.0, stop)

    error = raised.value
    assert error.target_entry == pytest.approx(expected_target)
    assert abs(error.target_entry - stop) < abs(100.0 - stop)
    assert error.required_capital == pytest.approx(1_000.0)
    assert error.minimum_volume == pytest.approx(0.01)
    assert error.to_payload() == pytest.approx(
        {
            "target_entry": expected_target,
            "required_initial_capital": 1_000.0,
            "broker_min_volume": 0.01,
        }
    )


def test_arbitrary_volume_step_is_always_floored_not_rounded_up(monkeypatch):
    executor = _executor(initial_capital=195.0)
    _account(monkeypatch, balance=195.0, equity=195.0)
    _sizing_market(
        monkeypatch,
        info=_symbol_info(volume_min=0.001, volume_step=0.001),
        loss_per_lot=100.0,
    )

    sizing = executor._size_volume_for_risk("TEST", "LONG", 100.0, 90.0)

    assert sizing.limit.budget_amount == pytest.approx(1.95)
    assert sizing.volume == pytest.approx(0.019)
    assert sizing.risk_amount == pytest.approx(1.90)
    assert sizing.risk_amount <= sizing.limit.budget_amount


@pytest.mark.parametrize(
    ("side", "stop", "expected_order_type"),
    [
        ("LONG", 90.0, mt5_executor.mt5.ORDER_TYPE_BUY),
        ("SHORT", 110.0, mt5_executor.mt5.ORDER_TYPE_SELL),
    ],
)
def test_risk_per_lot_uses_directional_order_calc_profit(
    monkeypatch, side, stop, expected_order_type
):
    executor = _executor()
    info = _symbol_info()
    monkeypatch.setattr(mt5_executor.mt5, "symbol_info", lambda symbol: info)
    calls = []

    def fake_profit(order_type, symbol, volume, entry, adverse_stop):
        calls.append((order_type, symbol, volume, entry, adverse_stop))
        if order_type == mt5_executor.mt5.ORDER_TYPE_BUY:
            return (adverse_stop - entry) * 25.0 * volume
        return (entry - adverse_stop) * 25.0 * volume

    monkeypatch.setattr(mt5_executor.mt5, "order_calc_profit", fake_profit)

    risk_per_lot = executor._risk_per_lot("TEST", side, 100.0, stop)

    assert risk_per_lot == pytest.approx(250.0)
    assert calls == [(expected_order_type, "TEST", 1.0, 100.0, stop)]


def test_commission_is_included_in_risk_per_lot(monkeypatch):
    executor = _executor(commission_per_lot=7.0)
    _sizing_market(monkeypatch, loss_per_lot=100.0)

    assert executor._risk_per_lot("TEST", "LONG", 100.0, 90.0) == pytest.approx(
        107.0
    )


def test_entry_deviation_is_reserved_in_addition_to_stop_distance(monkeypatch):
    executor = _executor(slippage=2)
    info = _symbol_info(point=0.1)
    monkeypatch.setattr(mt5_executor.mt5, "symbol_info", lambda symbol: info)

    def directional_profit(order_type, symbol, volume, entry, exit_price):
        return (exit_price - entry) * 100.0 * volume

    monkeypatch.setattr(mt5_executor.mt5, "order_calc_profit", directional_profit)

    risk_per_lot = executor._risk_per_lot("TEST", "LONG", 100.0, 99.0)

    assert risk_per_lot == pytest.approx(120.0)


def test_send_order_resizes_against_final_broker_clamped_stop(monkeypatch):
    executor = _executor()
    _account(monkeypatch)
    info = _symbol_info(
        point=1.0,
        volume_min=0.1,
        volume_step=0.1,
        stops_level=20,
        spread=1,
    )
    monkeypatch.setattr(mt5_executor.mt5, "symbol_info", lambda symbol: info)
    monkeypatch.setattr(
        mt5_executor.mt5,
        "symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=99.0, ask=100.0),
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "order_calc_profit",
        lambda order_type, symbol, volume, entry, stop: (stop - entry)
        * 10.0
        * volume,
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "order_calc_margin",
        lambda order_type, symbol, volume, entry: None,
    )
    executor._resolve_fill_modes = lambda symbol: [7]
    sent = []

    def fake_send(request):
        sent.append(dict(request))
        return SimpleNamespace(
            retcode=mt5_executor.mt5.TRADE_RETCODE_DONE,
            order=101,
            price=100.0,
            deal=202,
        )

    monkeypatch.setattr(mt5_executor, "_send_request", fake_send)

    result = executor._send_order(
        "TEST",
        "LONG",
        1.0,
        100.0,
        95.0,
        150.0,
        None,
    )

    assert sent[0]["sl"] == pytest.approx(80.0)
    assert sent[0]["volume"] == pytest.approx(0.5)
    assert result["stop_price"] == pytest.approx(80.0)
    assert result["volume"] == pytest.approx(0.5)
    assert result["risk_amount"] == pytest.approx(100.0)
    assert result["risk_amount"] <= result["risk_budget_amount"]


def test_send_order_tracks_actual_partial_fill_volume_and_risk(monkeypatch):
    executor = _executor()
    _account(monkeypatch)
    info = _symbol_info(
        point=1.0,
        volume_min=0.1,
        volume_step=0.1,
        spread=1,
    )
    monkeypatch.setattr(mt5_executor.mt5, "symbol_info", lambda symbol: info)
    monkeypatch.setattr(
        mt5_executor.mt5,
        "symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=99.0, ask=100.0),
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "order_calc_profit",
        lambda order_type, symbol, volume, entry, stop: (stop - entry)
        * 10.0
        * volume,
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "order_calc_margin",
        lambda order_type, symbol, volume, entry: None,
    )
    executor._resolve_fill_modes = lambda symbol: [7]
    monkeypatch.setattr(
        mt5_executor,
        "_send_request",
        lambda request: SimpleNamespace(
            retcode=getattr(mt5_executor.mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010),
            order=101,
            price=100.0,
            deal=202,
            volume=0.4,
        ),
    )

    result = executor._send_order(
        "TEST",
        "LONG",
        1.0,
        100.0,
        90.0,
        150.0,
        None,
    )

    assert result["partial_fill"] is True
    assert result["volume"] == pytest.approx(0.4)
    assert result["risk_amount"] == pytest.approx(40.0)
    assert result["risk_amount"] <= result["risk_budget_amount"]


def test_split_entry_caps_aggregate_stop_risk_to_one_budget(monkeypatch):
    executor = _executor()
    _account(monkeypatch)
    info = _symbol_info(
        point=1.0,
        volume_min=0.1,
        volume_step=0.1,
        stops_level=0,
        spread=1,
    )
    monkeypatch.setattr(mt5_executor.mt5, "symbol_info", lambda symbol: info)
    monkeypatch.setattr(
        mt5_executor.mt5,
        "symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=99.0, ask=100.0),
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "order_calc_profit",
        lambda order_type, symbol, volume, entry, stop: (stop - entry)
        * 10.0
        * volume,
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "order_calc_margin",
        lambda order_type, symbol, volume, entry: None,
    )
    executor._resolve_fill_modes = lambda symbol: [7]
    next_ticket = iter(range(1, 10))

    def fake_send(request):
        ticket = next(next_ticket)
        return SimpleNamespace(
            retcode=mt5_executor.mt5.TRADE_RETCODE_DONE,
            order=ticket,
            price=100.0,
            deal=ticket + 100,
        )

    monkeypatch.setattr(mt5_executor, "_send_request", fake_send)
    executor._find_position_id_from_deal = lambda deal: deal + 1000

    legs = executor.execute_split_entry(
        "TEST",
        side="LONG",
        entry_price=100.0,
        stop_price=90.0,
        tp_prices=[110.0, 120.0, 130.0],
        volumes_per_tp=[1.0, 1.0, 1.0],
    )

    total_volume = sum(leg["volume"] for leg in legs)
    total_risk = sum(leg["risk_amount"] for leg in legs)
    budget = legs[0]["risk_budget_amount"]

    assert [leg["volume"] for leg in legs] == pytest.approx([0.4, 0.3, 0.3])
    assert total_volume == pytest.approx(1.0)
    assert total_risk == pytest.approx(100.0)
    assert total_risk <= budget
