"""Entry spread gate: refuse a fill while the quoted spread is dislocated."""

from __future__ import annotations

import config


def test_defaults_sit_above_the_measured_in_session_p95():
    """Limits must pass normal trading and cut only genuine dislocations.

    Measured FxPro DOM in-session p95, in pips (2026-07-28 onward).
    """

    in_session_p95 = {"EURUSD": 0.34, "GBPUSD": 0.92, "USDCAD": 1.06}
    for symbol, p95 in in_session_p95.items():
        limit = config.MAX_ENTRY_SPREAD_PIPS[symbol]
        assert limit > p95, f"{symbol} limit would block normal spreads"
        # ...but still well under the off-session p99 blowouts (9-10 pips).
        assert limit < 3.0, f"{symbol} limit is too loose to catch a spike"


def test_gate_is_off_by_default():
    assert config.MAX_ENTRY_SPREAD_ENABLED is False


def test_env_override_parses_and_keeps_unlisted_defaults():
    limits = config._parse_spread_limits("EURUSD=0.9, GBPUSD=2.0")

    assert limits["EURUSD"] == 0.9
    assert limits["GBPUSD"] == 2.0
    # USDCAD was not overridden and must keep its measured default.
    assert limits["USDCAD"] == config._DEFAULT_MAX_ENTRY_SPREAD_PIPS["USDCAD"]


def test_malformed_entries_are_ignored_not_fatal():
    limits = config._parse_spread_limits("garbage,EURUSD=,GBPUSD=abc,=1.0")

    assert limits == config._DEFAULT_MAX_ENTRY_SPREAD_PIPS


def test_non_positive_limits_are_rejected():
    limits = config._parse_spread_limits("EURUSD=0,GBPUSD=-1")

    assert limits["EURUSD"] == config._DEFAULT_MAX_ENTRY_SPREAD_PIPS["EURUSD"]
    assert limits["GBPUSD"] == config._DEFAULT_MAX_ENTRY_SPREAD_PIPS["GBPUSD"]
