"""Realized slippage must be captured at the moment of the market send.

Cost stress puts the strategy's break-even near 0.6 pip per side, and that
threshold has stayed untestable because the request-time quote was never
persisted. These tests pin the arithmetic and its sign convention.
"""

from __future__ import annotations


def _slippage(fill: float, request: float, is_buy: bool) -> float:
    """Mirror of the executor's sign convention: positive == worse fill."""
    return fill - request if is_buy else request - fill


def test_positive_slippage_always_means_a_worse_fill():
    # BUY filled above the asked price is adverse.
    assert _slippage(1.10005, 1.10000, is_buy=True) > 0
    # SELL filled below the bid is adverse.
    assert _slippage(1.09995, 1.10000, is_buy=False) > 0


def test_price_improvement_is_negative_for_both_sides():
    assert _slippage(1.09995, 1.10000, is_buy=True) < 0
    assert _slippage(1.10005, 1.10000, is_buy=False) < 0


def test_exact_fill_is_zero_slippage():
    assert _slippage(1.10000, 1.10000, is_buy=True) == 0.0
    assert _slippage(1.10000, 1.10000, is_buy=False) == 0.0


def test_pip_conversion_matches_the_break_even_threshold():
    """0.6 pip on a 5-digit FX symbol is 0.00006 in price terms."""
    pip = 0.0001
    adverse = _slippage(1.10006, 1.10000, is_buy=True)
    assert round(adverse / pip, 6) == 0.6


def test_executor_exposes_the_telemetry_keys():
    """The keys the analysis depends on must not be renamed silently."""
    import inspect

    from executors import mt5_executor

    source = inspect.getsource(mt5_executor.MT5Executor._send_order)
    for key in (
        "request_price",
        "request_bid",
        "request_ask",
        "request_spread",
        "slippage_price",
        "slippage_pips",
    ):
        assert f'"{key}"' in source, f"{key} is no longer emitted"


def test_split_legs_carry_the_same_telemetry():
    """Split is the production path; measuring only monitor captures nothing."""
    import inspect

    from executors import mt5_executor

    source = inspect.getsource(mt5_executor.MT5Executor.execute_split_entry)
    for key in ("request_price", "request_spread", "slippage_pips"):
        assert f'"{key}"' in source, f"split legs no longer carry {key}"
