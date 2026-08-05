"""The invalidated-setup check must run before sizing, not after it.

Sizing uses the live tick as the entry while the stop stays at its planned
level.  When price has already crossed the stop, the sizing invariant "stop is
on the correct side of entry" fails first and reports a generic risk error, so
the accurate diagnosis inside ``_send_order`` is never reached.  The live
journal for 2026-07-13..08-05 shows 25 such rejections logged as
``LONG SL must be below entry`` / ``SHORT SL must be above entry`` — a message
that reads like a malformed signal rather than an invalidated setup.
"""

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


def _market(monkeypatch, *, bid, ask, point=0.0001):
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
            volume_min=0.01,
            volume_step=0.01,
        ),
    )


@pytest.mark.parametrize(
    ("side", "bid", "ask", "stop"),
    [
        # LONG: ask has fallen to/through the planned stop.
        ("LONG", 1.0948, 1.0950, 1.0950),
        ("LONG", 1.0940, 1.0942, 1.0950),
        # SHORT: bid has risen to/through the planned stop.
        ("SHORT", 1.1050, 1.1052, 1.1050),
        ("SHORT", 1.1060, 1.1062, 1.1050),
    ],
)
def test_execute_entry_reports_an_invalidated_setup_not_a_risk_error(
    monkeypatch, side, bid, ask, stop
):
    _market(monkeypatch, bid=bid, ask=ask)
    executor = _executor()

    def _fail(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("sizing ran before the invalidated-setup check")

    monkeypatch.setattr(executor, "_calc_volume", _fail)

    with pytest.raises(RuntimeError, match="setup invalidated"):
        executor.execute_entry(
            "EURUSD",
            side=side,
            entry_price=1.1000,
            stop_price=stop,
            tp_price=None,
        )


@pytest.mark.parametrize(
    ("side", "bid", "ask", "stop"),
    [
        ("LONG", 1.0948, 1.0950, 1.0950),
        ("SHORT", 1.1050, 1.1052, 1.1050),
    ],
)
def test_execute_split_entry_checks_before_the_preflight_sizing(
    monkeypatch, side, bid, ask, stop
):
    _market(monkeypatch, bid=bid, ask=ask)
    executor = _executor()

    def _fail(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("preflight sizing ran before the check")

    monkeypatch.setattr(executor, "_risk_limit", _fail)
    monkeypatch.setattr(executor, "_size_volume_for_risk", _fail)

    with pytest.raises(RuntimeError, match="setup invalidated"):
        executor.execute_split_entry(
            "EURUSD",
            side=side,
            entry_price=1.1000,
            stop_price=stop,
            tp_prices=[1.1050, 1.1100, 1.1150],
            volumes_per_tp=[0.05, 0.03, 0.02],
        )


@pytest.mark.parametrize(
    ("side", "price", "stop"),
    [
        ("LONG", 1.1000, 1.0950),  # healthy room below
        ("SHORT", 1.1000, 1.1050),  # healthy room above
    ],
)
def test_a_valid_setup_passes_the_check_untouched(side, price, stop):
    executor = _executor()

    executor._assert_setup_not_invalidated("EURUSD", side, price, stop)


def test_missing_or_non_finite_inputs_do_not_raise():
    """The check refuses setups; it never invents a rejection from bad input."""

    executor = _executor()

    executor._assert_setup_not_invalidated("EURUSD", "LONG", 1.1, None)
    executor._assert_setup_not_invalidated("EURUSD", "LONG", float("nan"), 1.09)
    executor._assert_setup_not_invalidated("EURUSD", "LONG", 1.1, float("inf"))
    executor._assert_setup_not_invalidated("EURUSD", "LONG", 1.1, "not-a-price")


def test_send_order_still_carries_its_own_check(monkeypatch):
    """Defence in depth: the deeper guard is not removed by the earlier one."""

    _market(monkeypatch, bid=1.0948, ask=1.0950)
    executor = _executor()

    with pytest.raises(RuntimeError, match="setup invalidated"):
        executor._send_order(
            "EURUSD",
            "LONG",
            0.1,
            1.1000,
            1.0950,
            None,
            None,
        )
