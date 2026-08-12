from __future__ import annotations

from types import SimpleNamespace

import main as main_module
from core.execution_quality import ExecutionQualityProfile


def _profile() -> ExecutionQualityProfile:
    return ExecutionQualityProfile.from_mapping({
        "schema": "execution-quality-profile-v1",
        "profile_id": "quality",
        "measured_from": "test",
        "created_at_utc": "2026-08-01T00:00:00Z",
        "rollover_timezone": "UTC",
        "rollover_hour": 22,
        "rollover_minute": 0,
        "rollover_before_minutes": 0,
        "rollover_after_minutes": 0,
        "news_filter_enabled": False,
        "news_before_minutes": 0,
        "news_after_minutes": 0,
        "news_min_impact": 3,
        "symbols": {
            "EURUSD": {
                "currencies": ["EUR", "USD"],
                "max_spread_price": 0.0002,
                "max_spread_r": 0.03,
            },
        },
    })


def _signal():
    return {
        "signal": "ENTER",
        "side": "LONG",
        "entry_price": 1.10,
        "stop_price": 1.09,
    }


def test_default_off_never_reads_broker_quote(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "EXECUTION_QUALITY_FILTER_ENABLED",
        False,
    )
    core = main_module.Core.__new__(main_module.Core)
    core.mt5_executor = SimpleNamespace(
        get_current_quote=lambda _symbol: (_ for _ in ()).throw(
            AssertionError("disabled filter read broker quote")
        )
    )

    signal = _signal()
    assert core._apply_execution_quality_filter(
        "EURUSD",
        signal,
    ) is signal


def test_enabled_gate_blocks_only_from_causal_quote(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "EXECUTION_QUALITY_FILTER_ENABLED",
        True,
    )
    core = main_module.Core.__new__(main_module.Core)
    core.execution_quality_profile = _profile()
    core.news_calendar = None
    core.universe = {"EURUSD": "EURUSD"}
    core.mt5_executor = SimpleNamespace(
        get_current_quote=lambda _symbol: {
            "bid": 1.1000,
            "ask": 1.1003,
            "time_msc": 123,
        }
    )

    result = core._apply_execution_quality_filter(
        "EURUSD",
        _signal(),
    )

    assert result["signal"] == "SKIP_SPREAD"
    assert result["execution_quality"]["quote_time_msc"] == 123


def test_enabled_gate_fails_closed_when_profile_missing(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "EXECUTION_QUALITY_FILTER_ENABLED",
        True,
    )
    core = main_module.Core.__new__(main_module.Core)
    core.execution_quality_profile = None
    core.mt5_executor = None

    result = core._apply_execution_quality_filter(
        "EURUSD",
        _signal(),
    )

    assert result["signal"] == "WAIT_EXECUTION_QUALITY_DATA"
