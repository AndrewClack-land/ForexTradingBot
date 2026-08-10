from __future__ import annotations

from types import SimpleNamespace

import pytest

from executors import mt5_executor


def _executor():
    executor = mt5_executor.MT5Executor.__new__(mt5_executor.MT5Executor)
    executor.settings = SimpleNamespace(magic=4242, slippage=12)
    executor.logger = SimpleNamespace(
        warning=lambda *args, **kwargs: None,
    )
    executor._fill_mode_cache = {}
    return executor


def _position(**overrides):
    payload = {
        "ticket": 77,
        "symbol": "EURUSD",
        "magic": 4242,
        "comment": "FBPL:abc123:T1",
        "volume": 0.10,
        "type": 0,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _install_close_constants(monkeypatch):
    constants = {
        "POSITION_TYPE_BUY": 0,
        "ORDER_TYPE_BUY": 0,
        "ORDER_TYPE_SELL": 1,
        "TRADE_ACTION_DEAL": 1,
        "TRADE_RETCODE_DONE": 10009,
        "TRADE_RETCODE_INVALID_FILLING": 10030,
    }
    for name, value in constants.items():
        monkeypatch.setattr(mt5_executor.mt5, name, value, raising=False)


def test_close_trade_missing_exact_ticket_never_falls_back_to_same_symbol(
    monkeypatch,
):
    _install_close_constants(monkeypatch)
    executor = _executor()
    calls = []

    def positions_get(**kwargs):
        calls.append(kwargs)
        if "ticket" in kwargs:
            return ()
        return (_position(ticket=88, comment="existing-idea"),)

    monkeypatch.setattr(mt5_executor.mt5, "positions_get", positions_get)
    monkeypatch.setattr(
        mt5_executor.mt5,
        "symbol_info_tick",
        lambda symbol: pytest.fail("quote lookup must not run"),
    )
    monkeypatch.setattr(
        mt5_executor,
        "_send_request",
        lambda request: pytest.fail("order_send must not run"),
    )

    assert (
        executor.close_trade(
            "EURUSD",
            position_id=77,
            volume=None,
            expected_comment="FBPL:abc123:T1",
        )
        is False
    )
    assert calls == [{"ticket": 77}]


def test_close_trade_position_query_none_fails_closed(monkeypatch):
    _install_close_constants(monkeypatch)
    executor = _executor()
    monkeypatch.setattr(
        mt5_executor.mt5,
        "positions_get",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "last_error",
        lambda: (-1, "disconnected"),
    )
    monkeypatch.setattr(
        mt5_executor,
        "_send_request",
        lambda request: pytest.fail("order_send must not run"),
    )

    with pytest.raises(RuntimeError, match="positions_get"):
        executor.close_trade(
            "EURUSD",
            position_id=77,
            volume=None,
            expected_comment="FBPL:abc123:T1",
        )


def test_close_trade_legacy_symbol_lookup_never_returns_unowned_position(
    monkeypatch,
):
    _install_close_constants(monkeypatch)
    executor = _executor()
    monkeypatch.setattr(
        mt5_executor.mt5,
        "positions_get",
        lambda **kwargs: (_position(ticket=88, magic=9999),),
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "symbol_info_tick",
        lambda symbol: pytest.fail("quote lookup must not run"),
    )
    monkeypatch.setattr(
        mt5_executor,
        "_send_request",
        lambda request: pytest.fail("order_send must not run"),
    )

    assert executor.close_trade(
        "EURUSD",
        position_id=None,
        volume=None,
    ) is False


@pytest.mark.parametrize(
    ("position", "message"),
    [
        (_position(ticket=78), "requested ticket"),
        (_position(symbol="GBPUSD"), "does not match"),
        (_position(magic=9999), "not owned"),
    ],
)
def test_close_trade_rejects_inexact_ticket_symbol_or_magic(
    monkeypatch,
    position,
    message,
):
    _install_close_constants(monkeypatch)
    executor = _executor()
    monkeypatch.setattr(
        mt5_executor.mt5,
        "positions_get",
        lambda **kwargs: (position,),
    )
    monkeypatch.setattr(
        mt5_executor,
        "_send_request",
        lambda request: pytest.fail("order_send must not run"),
    )

    with pytest.raises(RuntimeError, match=message):
        executor.close_trade(
            "EURUSD",
            position_id=77,
            volume=None,
            expected_comment="FBPL:abc123:T1",
        )


def test_close_trade_requires_exact_expected_comment(monkeypatch):
    _install_close_constants(monkeypatch)
    executor = _executor()
    monkeypatch.setattr(
        mt5_executor.mt5,
        "positions_get",
        lambda **kwargs: (_position(comment="FBPL:abc123:T2"),),
    )
    monkeypatch.setattr(
        mt5_executor,
        "_send_request",
        lambda request: pytest.fail("order_send must not run"),
    )

    with pytest.raises(RuntimeError, match="does not exactly match"):
        executor.close_trade(
            "EURUSD",
            position_id=77,
            volume=None,
            expected_comment="FBPL:abc123:T1",
        )


@pytest.mark.parametrize("position_id", [0, -1, True])
def test_close_trade_rejects_non_positive_explicit_position_id(
    monkeypatch,
    position_id,
):
    _install_close_constants(monkeypatch)
    executor = _executor()
    monkeypatch.setattr(
        mt5_executor.mt5,
        "positions_get",
        lambda **kwargs: pytest.fail("position lookup must not run"),
    )

    with pytest.raises(RuntimeError, match="positive integer"):
        executor.close_trade(
            "EURUSD",
            position_id=position_id,
            volume=None,
            expected_comment="FBPL:abc123:T1",
        )


def test_close_trade_sends_only_after_fresh_exact_ownership_match(monkeypatch):
    _install_close_constants(monkeypatch)
    executor = _executor()
    position = _position()
    monkeypatch.setattr(
        mt5_executor.mt5,
        "positions_get",
        lambda **kwargs: (position,),
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=1.1000, ask=1.1002),
    )
    executor._resolve_fill_modes = lambda symbol: [7]
    sent = []
    monkeypatch.setattr(
        mt5_executor,
        "_send_request",
        lambda request: (
            sent.append(dict(request))
            or SimpleNamespace(retcode=10009)
        ),
    )

    assert executor.close_trade(
        "eurusd",
        position_id=77,
        volume=None,
        expected_comment="FBPL:abc123:T1",
    )
    assert sent == [
        {
            "action": 1,
            "symbol": "EURUSD",
            "position": 77,
            "volume": 0.10,
            "type": 1,
            "price": 1.1000,
            "deviation": 12,
            "magic": 4242,
            "comment": "Close by bot",
            "type_filling": 7,
        }
    ]


def test_pending_limit_capabilities_is_strictly_read_only(monkeypatch):
    _install_close_constants(monkeypatch)
    monkeypatch.setattr(
        mt5_executor.mt5,
        "SYMBOL_ORDER_LIMIT",
        2,
        raising=False,
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "SYMBOL_EXPIRATION_SPECIFIED",
        4,
        raising=False,
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "ACCOUNT_MARGIN_MODE_RETAIL_HEDGING",
        2,
        raising=False,
    )
    executor = _executor()
    monkeypatch.setattr(
        mt5_executor.mt5,
        "symbol_info",
        lambda symbol: SimpleNamespace(order_mode=2, expiration_mode=4),
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "account_info",
        lambda: SimpleNamespace(margin_mode=2),
    )
    monkeypatch.setattr(
        mt5_executor.mt5,
        "symbol_select",
        lambda *args, **kwargs: pytest.fail("symbol_select must not run"),
    )
    monkeypatch.setattr(
        mt5_executor,
        "_send_request",
        lambda request: pytest.fail("order_send must not run"),
    )
    executor._risk_limit = lambda: pytest.fail("risk calculation must not run")

    assert executor.pending_limit_capabilities("EURUSD") == {
        "limit_allowed": True,
        "specified_expiration": True,
        "account_hedging": True,
        "ready": True,
    }


def test_pending_limit_capabilities_fails_closed_when_data_missing(
    monkeypatch,
):
    _install_close_constants(monkeypatch)
    executor = _executor()
    monkeypatch.setattr(
        mt5_executor.mt5,
        "symbol_info",
        lambda symbol: None,
    )
    monkeypatch.setattr(mt5_executor.mt5, "account_info", lambda: None)

    assert executor.pending_limit_capabilities("EURUSD") == {
        "limit_allowed": False,
        "specified_expiration": False,
        "account_hedging": False,
        "ready": False,
    }
