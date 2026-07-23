from __future__ import annotations

import math

import numpy as np
import pytest

import main
from core.vol_regime import (
    VolContext,
    build_vol_context,
    classify_regime,
    em_move,
    entry_gate,
    profile_for,
    realized_vol,
    GOLD_PROFILE,
    FX_PROFILE,
)


def _closes(n=120, sigma=0.01, start=100.0, seed=7):
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0, sigma, n)
    return list(start * np.exp(np.cumsum(log_ret)))


def _ctx(r_t=40.0, em_1d=10.0, spot=100.0):
    return VolContext(
        symbol="GOLD", r_t=r_t, regime=classify_regime(r_t),
        base_r_t=r_t, tape_stress=r_t,
        atm_iv=0.15, rv=0.13, vrp=0.02, spot=spot,
        em_1d=em_1d, em_5d=em_1d * math.sqrt(5), signals={},
    )


def _ohlc_15m(n=200, sigma=0.001, start=100.0, seed=3):
    import pandas as pd
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0, sigma, n)))
    open_ = np.roll(close, 1)
    open_[0] = start
    spread = np.abs(rng.normal(0, sigma, n)) * close
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close,
    })


# ── pure math ─────────────────────────────────────────────────────────────────

def test_realized_vol_matches_manual_calc():
    closes = _closes(60, sigma=0.01)
    rv = realized_vol(closes, window=21)
    log_ret = np.diff(np.log(closes))[-21:]
    expected = np.std(log_ret, ddof=1) * math.sqrt(252)
    assert rv == pytest.approx(expected)


def test_realized_vol_unusable_data():
    assert realized_vol([100.0] * 60) is None          # zero variance
    assert realized_vol([100.0, 101.0, 99.0]) is None  # too short


def test_em_move_formula():
    assert em_move(3000.0, 0.15, 1) == pytest.approx(3000.0 * 0.15 * math.sqrt(1 / 365))
    assert em_move(3000.0, 0.15, 5) == pytest.approx(3000.0 * 0.15 * math.sqrt(5 / 365))


def test_classify_regime_boundaries():
    assert classify_regime(0.0) == "CALM"
    assert classify_regime(30.0) == "NORMAL"
    assert classify_regime(50.0) == "ELEVATED"
    assert classify_regime(60.0) == "PANIC"
    assert classify_regime(100.0) == "PANIC"


def test_profiles():
    assert profile_for("GOLD") is GOLD_PROFILE
    assert profile_for("gold") is GOLD_PROFILE
    assert profile_for("EURUSD") is FX_PROFILE
    assert profile_for("UNKNOWN") is FX_PROFILE  # fallback


# ── context building ──────────────────────────────────────────────────────────

def test_build_vol_context_structure():
    ctx = build_vol_context("EURUSD", _closes(120, sigma=0.005, start=1.08), spot=1.0850)
    assert ctx is not None
    assert 0.0 <= ctx.r_t <= 100.0
    assert ctx.regime == classify_regime(ctx.r_t)
    assert ctx.spot == 1.0850
    assert ctx.atm_iv == pytest.approx(ctx.rv * FX_PROFILE.iv_mult + FX_PROFILE.iv_add)
    assert ctx.em_1d == pytest.approx(em_move(1.0850, ctx.atm_iv, 1))
    assert ctx.em_5d > ctx.em_1d
    assert len(ctx.signals) == 7


def test_build_vol_context_fails_open_on_short_data():
    assert build_vol_context("GOLD", [3000.0, 3010.0], spot=3005.0) is None


def test_score_increases_with_realized_vol():
    calm = build_vol_context("GOLD", _closes(120, sigma=0.004, start=3000))
    wild = build_vol_context("GOLD", _closes(120, sigma=0.05, start=3000))
    assert calm is not None and wild is not None
    assert wild.r_t > calm.r_t


def test_tape_metrics_and_overlay():
    from core.vol_regime import compute_tape_metrics, tape_stress_score

    df = _ohlc_15m(200)
    metrics = compute_tape_metrics(df, atm_iv=0.15)
    assert metrics is not None
    assert set(metrics) == {"vol_accel", "squeeze", "range_ratio", "momentum_1h_pct"}
    assert 0.0 <= tape_stress_score(metrics) <= 100.0

    assert compute_tape_metrics(df.head(50), atm_iv=0.15) is None  # too short
    assert compute_tape_metrics(None, atm_iv=0.15) is None


def test_context_blends_tape_stress():
    closes = _closes(120, sigma=0.01, start=100.0)
    base_only = build_vol_context("GOLD", closes)
    with_tape = build_vol_context("GOLD", closes, df_15m=_ohlc_15m(200))
    assert base_only is not None and with_tape is not None
    assert base_only.r_t == base_only.base_r_t
    assert with_tape.base_r_t == pytest.approx(base_only.base_r_t)
    expected = 0.62 * with_tape.base_r_t + 0.38 * with_tape.tape_stress
    assert with_tape.r_t == pytest.approx(expected)


def test_spot_falls_back_to_last_close():
    closes = _closes(120, sigma=0.01, start=1.30)
    ctx = build_vol_context("GBPUSD", closes, spot=None)
    assert ctx is not None
    assert ctx.spot == pytest.approx(closes[-1])


# ── entry gate ────────────────────────────────────────────────────────────────

def test_gate_blocks_panic_score():
    ok, reason = entry_gate(_ctx(r_t=80.0), 100.0, [101.0], max_r=75.0, em_tp_ratio=1.0)
    assert not ok
    assert "R(t)=80" in reason


def test_gate_blocks_tp1_beyond_expected_move():
    # em_1d=10, TP1 is 15 away → blocked at ratio 1.0
    ok, reason = entry_gate(_ctx(em_1d=10.0), 100.0, [115.0, 130.0], max_r=75.0, em_tp_ratio=1.0)
    assert not ok
    assert "TP1" in reason


def test_gate_uses_nearest_tp():
    # nearest TP (8 away) is inside EM even though the far ones are not
    ok, _ = entry_gate(_ctx(em_1d=10.0), 100.0, [130.0, 108.0, 120.0], max_r=75.0, em_tp_ratio=1.0)
    assert ok


def test_gate_passes_normal_setup_and_short_side():
    ok, _ = entry_gate(_ctx(r_t=40.0, em_1d=10.0), 100.0, [93.0], max_r=75.0, em_tp_ratio=1.0)
    assert ok


def test_gate_disabled_by_zero_thresholds():
    ok, _ = entry_gate(_ctx(r_t=99.0, em_1d=0.1), 100.0, [150.0], max_r=0.0, em_tp_ratio=0.0)
    assert ok


# ── main.py wiring ────────────────────────────────────────────────────────────

class _FakeCore:
    """Bare object driving Core._apply_vol_regime_filter without a real Core."""

    def __init__(self, ctx):
        self._ctx = ctx

    def _get_vol_context(self, symbol, data):
        return self._ctx


def _enter_sig(entry=100.0, tps=(101.0, 102.0, 103.0)):
    return {
        "signal": "ENTER",
        "side": "LONG",
        "entry_price": entry,
        "stop_price": entry - 1.0,
        "tp_prices": list(tps),
    }


def test_filter_blocks_panic(monkeypatch):
    monkeypatch.setattr(main, "VOL_REGIME_FILTER_ENABLED", True)
    monkeypatch.setattr(main, "VOL_REGIME_SYMBOLS", ["GOLD"])
    core = _FakeCore(_ctx(r_t=90.0, em_1d=10.0))
    out = main.Core._apply_vol_regime_filter(core, "GOLD", _enter_sig(), {})
    assert out["signal"] == "SKIP_VOL_REGIME"
    assert out["vol_regime"] == "PANIC"


def test_filter_blocks_far_tp1(monkeypatch):
    monkeypatch.setattr(main, "VOL_REGIME_FILTER_ENABLED", True)
    monkeypatch.setattr(main, "VOL_REGIME_SYMBOLS", ["GOLD"])
    monkeypatch.setattr(main, "EM_TP_MAX_RATIO", 1.0)
    core = _FakeCore(_ctx(r_t=40.0, em_1d=0.5))
    out = main.Core._apply_vol_regime_filter(core, "GOLD", _enter_sig(tps=(105.0,)), {})
    assert out["signal"] == "SKIP_EM_TP"


def test_filter_passes_and_annotates(monkeypatch):
    monkeypatch.setattr(main, "VOL_REGIME_FILTER_ENABLED", True)
    monkeypatch.setattr(main, "VOL_REGIME_SYMBOLS", ["GOLD"])
    core = _FakeCore(_ctx(r_t=42.0, em_1d=10.0))
    out = main.Core._apply_vol_regime_filter(core, "GOLD", _enter_sig(), {})
    assert out["signal"] == "ENTER"
    assert out["vol_R"] == 42.0
    assert out["vol_regime"] == "NORMAL"
    assert out["vol_em_1d"] == 10.0


def test_filter_skips_other_symbols_and_non_enter(monkeypatch):
    monkeypatch.setattr(main, "VOL_REGIME_FILTER_ENABLED", True)
    monkeypatch.setattr(main, "VOL_REGIME_SYMBOLS", ["GOLD"])
    core = _FakeCore(_ctx(r_t=99.0))
    sig = _enter_sig()
    assert main.Core._apply_vol_regime_filter(core, "EURUSD", sig, {})["signal"] == "ENTER"
    hold = {"signal": "HOLD"}
    assert main.Core._apply_vol_regime_filter(core, "GOLD", hold, {}) is hold


def test_filter_fails_open_without_context(monkeypatch):
    monkeypatch.setattr(main, "VOL_REGIME_FILTER_ENABLED", True)
    monkeypatch.setattr(main, "VOL_REGIME_SYMBOLS", ["GOLD"])
    core = _FakeCore(None)
    out = main.Core._apply_vol_regime_filter(core, "GOLD", _enter_sig(), {})
    assert out["signal"] == "ENTER"
    assert "vol_R" not in out
