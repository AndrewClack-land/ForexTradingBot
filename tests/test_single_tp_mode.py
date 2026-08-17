"""Single-TP production contract: one aggregate 1:1.2 target per position.

The legacy 1R/2R/3R 50/30/20 split must stay in the code but inactive, the
dedicated admission rule must accept the 1.2 R/R, Telegram must render a
single TP line, and the sealed backtest contract must stay three-target.
"""

from __future__ import annotations

import types

import pandas as pd
import pytest

import config
import main
from bot.telegram_bot import TelegramBot
from core.m1.config import AIConfig, _default_min_rr
from core.position_adding import PyramidManager, PyramidSettings
from core.strategy_narrative import NarrativeStrategy


def _frames():
    index = pd.date_range("2026-08-03", periods=60, freq="1h", tz="UTC")
    base = 1.1000
    frame = pd.DataFrame(
        {
            "open": [base + 0.0002 * (i % 7) for i in range(60)],
            "high": [base + 0.0035 + 0.0002 * (i % 7) for i in range(60)],
            "low": [base - 0.0035 + 0.0002 * (i % 7) for i in range(60)],
            "close": [base + 0.0001 * (i % 9) for i in range(60)],
        },
        index=index,
    )
    return frame


def test_config_defaults_enable_single_tp_contract() -> None:
    assert config.SINGLE_TP_MODE_ENABLED is True
    assert config.SINGLE_TP_RR == pytest.approx(1.2)
    # Position-adding adaptation: the default progress floor scales to the
    # single 1.2R target instead of the legacy 0.5R.
    assert config.ADDON_MIN_PROGRESS_R == pytest.approx(0.3)


def test_strategy_produces_one_tp_at_contract_rr() -> None:
    strategy = NarrativeStrategy()
    assert strategy.single_tp_mode is True
    assert strategy.tp_rr_levels == [pytest.approx(1.2)]
    assert strategy.rr_min == pytest.approx(1.2)

    frame = _frames()
    entry = 1.1000
    stop, tps = strategy.calc_stop_and_tps(entry, "LONG", frame, frame)
    assert len(tps) == 1
    risk = abs(entry - stop)
    assert tps[0] == pytest.approx(entry + 1.2 * risk, rel=1e-9)

    stop_s, tps_s = strategy.calc_stop_and_tps(entry, "SHORT", frame, frame)
    assert len(tps_s) == 1
    assert tps_s[0] == pytest.approx(entry - 1.2 * abs(entry - stop_s), rel=1e-9)


def test_legacy_split_remains_available_but_inactive() -> None:
    strategy = NarrativeStrategy()
    strategy.single_tp_mode = False
    strategy.tp_rr_levels = [1.0, 2.0, 3.0]
    strategy.rr_min = 1.5

    frame = _frames()
    entry = 1.1000
    stop, tps = strategy.calc_stop_and_tps(entry, "LONG", frame, frame)
    risk = abs(entry - stop)
    assert len(tps) == 3
    assert tps == [
        pytest.approx(entry + rr * risk, rel=1e-9) for rr in (1.0, 2.0, 3.0)
    ]


def test_single_tp_volume_is_the_whole_position() -> None:
    assert main._compute_tp_volumes(0.50, 1) == [pytest.approx(0.50)]
    assert main._compute_tp_volumes(0.03, 1) == [pytest.approx(0.03)]
    # Legacy split still computes 50/30/20 when three TPs are requested.
    legacy = main._compute_tp_volumes(1.0, 3)
    assert legacy == [pytest.approx(0.5), pytest.approx(0.3), pytest.approx(0.2)]


def test_dedicated_admission_rule_accepts_contract_rr(monkeypatch) -> None:
    monkeypatch.delenv("AI_MIN_RR", raising=False)
    assert _default_min_rr() == pytest.approx(1.2)
    assert AIConfig().min_rr == pytest.approx(1.2)

    monkeypatch.setenv("AI_MIN_RR", "1.4")
    assert _default_min_rr() == pytest.approx(1.4)

    monkeypatch.delenv("AI_MIN_RR", raising=False)
    monkeypatch.setattr(config, "SINGLE_TP_MODE_ENABLED", False)
    assert _default_min_rr() == pytest.approx(1.3)


def test_telegram_renders_exactly_the_signal_tps() -> None:
    single = TelegramBot._format_signal(
        None,
        "EURUSD",
        {
            "side": "LONG",
            "entry_price": 1.1000,
            "stop_price": 1.0950,
            "tp_prices": [1.1060],
        },
    )
    assert "✅Тп: 1.10600" in single
    assert "Тп 2" not in single and "Тп 1:" not in single

    legacy = TelegramBot._format_signal(
        None,
        "EURUSD",
        {
            "side": "LONG",
            "entry_price": 1.1000,
            "stop_price": 1.0950,
            "tp_prices": [1.1050, 1.1100, 1.1150],
        },
    )
    assert "✅Тп 1: 1.10500" in legacy
    assert "✅Тп 3: 1.11500" in legacy


def test_position_adding_window_matches_single_tp_geometry() -> None:
    manager = PyramidManager(PyramidSettings(
        enabled=True,
        min_progress_r=0.3,
        max_progress_pct=0.5,
    ))
    entry = types.SimpleNamespace(
        entry=1.0000,
        stop=0.9900,
        tp_prices=[1.0120],  # single 1.2R target
    )
    trade = types.SimpleNamespace(side="LONG")

    too_early = manager._check_progress(trade, [entry], 1.0020)  # 0.2R
    assert too_early is not None and "не подтверждена" in too_early

    in_window = manager._check_progress(trade, [entry], 1.0040)  # 0.4R, 33%
    assert in_window is None

    too_late = manager._check_progress(trade, [entry], 1.0070)  # 0.7R, 58%
    assert too_late is not None and "финального TP" in too_late


def test_payload_reports_single_tp_rule(monkeypatch) -> None:
    strategy = NarrativeStrategy()
    captured = {}

    real = strategy.calc_stop_and_tps

    def spy(entry_price, side, df_1h, df_4h, custom_stop=None, symbol=""):
        stop, tps = real(
            entry_price, side, df_1h, df_4h,
            custom_stop=custom_stop, symbol=symbol,
        )
        captured["tps"] = tps
        return stop, tps

    monkeypatch.setattr(strategy, "calc_stop_and_tps", spy)
    frame = _frames()
    entry = types.SimpleNamespace(
        side="LONG",
        entry_price=1.1000,
        entry_min=1.0995,
        entry_max=1.1005,
        tf="15M",
        reason="test",
        trigger_kind="pivot_reclaim",
        trigger_event_id=None,
        trigger_meta=None,
        zone_low=None,
        zone_high=None,
        lock_entry_range=True,
        stop_override=None,
    )
    payload = strategy._build_signal_payload(
        entry=entry,
        side_bias="LONG",
        narrative_text="test narrative",
        factor_vector=None,
        fvg_side="NEUTRAL",
        fvg_text="",
        df_15M=frame,
        df_1H=frame,
        df_4H=frame,
        symbol="EURUSD",
    )
    assert payload["tp_prices"] == [
        pytest.approx(round(captured["tps"][0], 6))
    ]
    assert payload["weighted_rr_numeric"] == pytest.approx(1.2)
    assert payload["rr_numeric"] == pytest.approx(1.2)
    assert "single-TP" in payload["rr"]
