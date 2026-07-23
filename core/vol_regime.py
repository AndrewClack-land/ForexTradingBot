"""
Vol-regime filter — port of the GOLD IV Surface 7-signal score
(E:\\Новый индикатор по GOLD → gold_iv_surface.py) adapted for the bot.

The original builds a synthetic (CME-style) option chain from realized vol
(ATM IV = RV * mult + add — there is no real options feed) and scores the
surface with 7 signals:

    R(t) = 100 * (1/7) * Σ s_i(t)      s_i ∈ [0, 1]

    s1 ATM IV level          s5 smile convexity (tail pricing)
    s2 put skew              s6 IV − RV spread (VRP)
    s3 call skew             s7 near-ATM surface distortion
    s4 term-structure inversion

Because the chain is synthetic, the base score is intentionally stable, so the
original project blends in a short-term "tape stress" overlay computed from
M15 candles (src/utils/vol_regime.py::sensitive_regime_score):

    final = 0.62 * base + 0.38 * tape_stress

Regimes: CALM < 30 ≤ NORMAL < 50 ≤ ELEVATED < 60 ≤ PANIC.
The PANIC threshold is lowered from the original 72: with the 0.62/0.38 blend
the score is mathematically capped at 0.62*base + 38, and a base above 55
needs gold RV > 45% — the original 72 gate could effectively never fire.

The bot uses R(t) as an entry brake (PANIC blocks new setups) and the
IV-implied Expected Move as a TP sanity check (TP1 further than the expected
1-day move is unlikely to be reached before the daily flat close).

Data source is the bot's own MT5 daily candles — no TradingView / yfinance.
Normalisation ranges are per-symbol: gold and FX majors live in different
vol universes (gold RV ~10-25%, EURUSD ~5-12%).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# Tenor grid of the synthetic chain (days) — as in the original
_MATURITIES = (7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365)
_N_STRIKES = 15
_RISK_FREE = 0.045

TRADING_DAYS = 252

# Regime thresholds (score 0-100) — see module docstring on the PANIC level
REGIME_NORMAL = 30.0
REGIME_ELEVATED = 50.0
REGIME_PANIC = 60.0

# Minimum M15 bars for the tape-stress overlay (24h of M15 candles)
_TAPE_MIN_BARS = 96


@dataclass(frozen=True)
class VolProfile:
    """Per-symbol calibration of the synthetic surface."""
    iv_mult: float   # ATM IV = RV * iv_mult + iv_add (typical VRP of the asset)
    iv_add: float
    iv_low: float    # s1 normalisation range for ATM IV level
    iv_high: float
    vrp_low: float   # s6 normalisation range for IV − RV
    vrp_high: float


# Gold keeps the original calibration; FX majors get ranges matching their
# much lower vol universe. Unknown symbols fall back to the FX profile.
GOLD_PROFILE = VolProfile(iv_mult=1.18, iv_add=0.02,
                          iv_low=0.08, iv_high=0.50,
                          vrp_low=-0.03, vrp_high=0.15)
FX_PROFILE = VolProfile(iv_mult=1.10, iv_add=0.005,
                        iv_low=0.04, iv_high=0.20,
                        vrp_low=-0.015, vrp_high=0.06)

_PROFILES: Dict[str, VolProfile] = {
    "GOLD": GOLD_PROFILE,
    "XAUUSD": GOLD_PROFILE,
    "EURUSD": FX_PROFILE,
    "GBPUSD": FX_PROFILE,
    "USDCAD": FX_PROFILE,
}


def profile_for(symbol: str) -> VolProfile:
    return _PROFILES.get(str(symbol).upper(), FX_PROFILE)


@dataclass
class VolContext:
    symbol: str
    r_t: float          # final score 0-100 (base blended with tape stress)
    regime: str         # CALM / NORMAL / ELEVATED / PANIC
    base_r_t: float     # 7-signal surface score before the overlay
    tape_stress: float  # M15 overlay component 0-100 (base_r_t when no M15 data)
    atm_iv: float       # annualised (synthetic)
    rv: float           # annualised realized vol (21D)
    vrp: float          # atm_iv - rv
    spot: float
    em_1d: float        # expected 1-day move, price units
    em_5d: float        # expected 5-day move, price units
    signals: Dict[str, Tuple[float, float]]  # name -> (value, score)


# ── math helpers ──────────────────────────────────────────────────────────────

def _norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def _norm(value: float, low: float, high: float) -> float:
    return float(np.clip((value - low) / (high - low + 1e-9), 0.0, 1.0))


def realized_vol(closes: Sequence[float], window: int = 21) -> Optional[float]:
    """Annualised close-to-close realized vol of the LAST `window` daily returns."""
    arr = np.asarray(closes, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < window + 1:
        return None
    log_ret = np.diff(np.log(arr))
    recent = log_ret[-window:]
    sd = float(np.std(recent, ddof=1))
    if not math.isfinite(sd) or sd <= 0:
        return None
    return sd * math.sqrt(TRADING_DAYS)


def em_move(spot: float, iv: float, days: float) -> float:
    """IV-implied expected move over `days`, in price units (1-sigma)."""
    return float(spot) * float(iv) * math.sqrt(days / 365.0)


def classify_regime(r_t: float) -> str:
    if r_t >= REGIME_PANIC:
        return "PANIC"
    if r_t >= REGIME_ELEVATED:
        return "ELEVATED"
    if r_t >= REGIME_NORMAL:
        return "NORMAL"
    return "CALM"


# ── synthetic chain + 7-signal score (numpy port, no scipy) ───────────────────

def _iv_model(m: np.ndarray, T: float, atm_iv: float,
              skew_slope: float = -0.15, smile_curv: float = 0.08,
              term_shape: float = 0.04) -> np.ndarray:
    m = np.clip(m, -3.5, 3.5)
    term_factor = 1.0 + term_shape / math.sqrt(T + 0.05)
    iv = atm_iv * term_factor * (1.0 + skew_slope * m + smile_curv * m ** 2)
    return np.clip(iv, 0.01, 2.50)


def compute_surface_score(spot: float, atm_iv: float, rv: float,
                          profile: VolProfile) -> Tuple[float, Dict[str, Tuple[float, float]]]:
    """R(t) 0-100 + per-signal (value, score). Deterministic given (spot, atm_iv, rv)."""
    S = float(spot)
    # Strike grid ±30% around spot. The original rounds strikes to $10 —
    # cosmetic for gold and wrong for FX (everything would round to 0), so
    # the grid is kept unrounded here.
    K = np.linspace(S * 0.72, S * 1.30, _N_STRIKES)

    atm_by_tenor: Dict[int, float] = {}
    per_tenor: Dict[int, Dict[str, np.ndarray]] = {}
    for T_days in _MATURITIES:
        T = T_days / 365.0
        F = S * math.exp(_RISK_FREE * T)
        m = np.log(K / F) / (atm_iv * math.sqrt(T) + 1e-9)
        iv = _iv_model(m, T, atm_iv)
        d1 = (np.log(S / K) + (_RISK_FREE + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T) + 1e-9)
        delta = np.asarray(_norm_cdf(d1), dtype=float)
        per_tenor[T_days] = {"K": K, "m": m, "iv": iv, "delta": delta}
        atm_by_tenor[T_days] = float(iv[np.argmin(np.abs(m))])

    signals: Dict[str, Tuple[float, float]] = {}

    # s1: ATM IV level
    signals["ATM_IV"] = (atm_iv, _norm(atm_iv, profile.iv_low, profile.iv_high))

    # s2/s3: 25Δ put/call skew on the 30-day tenor
    ref = per_tenor[30]
    ref_atm = atm_by_tenor[30]
    put_mask = ref["K"] < S
    call_mask = ref["K"] > S
    put_iv = (
        float(ref["iv"][put_mask][np.argmin(np.abs(ref["delta"][put_mask] - 0.75))])
        if put_mask.any() else ref_atm
    )
    call_iv = (
        float(ref["iv"][call_mask][np.argmin(np.abs(ref["delta"][call_mask] - 0.25))])
        if call_mask.any() else ref_atm
    )
    put_skew = put_iv - ref_atm
    call_skew = call_iv - ref_atm
    signals["Put_Skew"] = (put_skew, _norm(put_skew, 0.00, 0.12))
    signals["Call_Skew"] = (call_skew, _norm(call_skew, -0.02, 0.08))

    # s4: term-structure inversion (7D vs 180D)
    term_inv = atm_by_tenor[7] - atm_by_tenor[180]
    signals["Term_Structure"] = (term_inv, _norm(term_inv, -0.05, 0.20))

    # s5: smile convexity (25Δ fly vs ATM)
    curvature = (put_iv + call_iv) / 2.0 - ref_atm
    signals["Smile_Curvature"] = (curvature, _norm(curvature, 0.0, 0.10))

    # s6: VRP
    vrp = atm_iv - rv
    signals["VRP"] = (vrp, _norm(vrp, profile.vrp_low, profile.vrp_high))

    # s7: near-ATM (|moneyness| < 1) surface distortion
    near = np.concatenate([
        t["iv"][np.abs(t["m"]) < 1.0] for t in per_tenor.values()
    ])
    distortion = float(np.std(near) / (atm_iv + 1e-9)) if len(near) > 1 else 0.0
    signals["Surface_Distortion"] = (distortion, _norm(distortion, 0.0, 0.25))

    r_t = 100.0 * float(np.mean([s for _, s in signals.values()]))
    return r_t, signals


# ── M15 tape-stress overlay (port of sensitive_regime_score) ─────────────────

def compute_tape_metrics(df_15m, atm_iv: float) -> Optional[Dict[str, float]]:
    """Garman-Klass-based short-term stress features on M15 candles.

    Expects the bot's lowercase open/high/low/close columns. Returns None when
    fewer than _TAPE_MIN_BARS bars are available (overlay is skipped).
    """
    if df_15m is None or len(df_15m) < _TAPE_MIN_BARS:
        return None
    o = df_15m["open"].astype(float)
    h = df_15m["high"].astype(float)
    low = df_15m["low"].astype(float)
    c = df_15m["close"].astype(float)

    gk = (0.5 * np.log(h / low) ** 2
          - (2 * math.log(2) - 1) * np.log(c / o) ** 2).clip(lower=0)

    fast = float(gk.rolling(4).mean().iloc[-1])
    slow = float(gk.rolling(16).mean().iloc[-1])
    vol_accel = fast / (slow + 1e-12)

    bars_per_year = 96 * TRADING_DAYS
    rv_m15 = float(np.sqrt(gk.rolling(96).mean().iloc[-1] * bars_per_year))
    squeeze = rv_m15 / (float(atm_iv) + 1e-12)

    hl = (h - low)
    range_ratio = float(hl.iloc[-1] / (hl.rolling(20).mean().iloc[-1] + 1e-12))
    momentum_1h_pct = float(c.iloc[-1] / c.iloc[-5] - 1) * 100.0

    out = {
        "vol_accel": vol_accel,
        "squeeze": squeeze,
        "range_ratio": range_ratio,
        "momentum_1h_pct": momentum_1h_pct,
    }
    if not all(math.isfinite(v) for v in out.values()):
        return None
    return out


def tape_stress_score(metrics: Dict[str, float]) -> float:
    """0-100 stress of the current M15 tape (weights as in the original)."""
    m15_accel = _norm(metrics.get("vol_accel", 1.0), 0.75, 1.75)
    squeeze = _norm(metrics.get("squeeze", 0.80), 0.55, 1.25)
    range_ratio = _norm(metrics.get("range_ratio", 1.0), 0.60, 1.80)
    momentum = _norm(abs(metrics.get("momentum_1h_pct", 0.0)), 0.03, 0.30)
    return 100.0 * (
        0.35 * m15_accel + 0.25 * squeeze + 0.25 * range_ratio + 0.15 * momentum
    )


def build_vol_context(symbol: str, daily_closes: Sequence[float],
                      spot: Optional[float] = None,
                      df_15m=None) -> Optional[VolContext]:
    """Full context from the bot's daily (+optionally M15) candles. None when
    data is unusable — the caller must fail OPEN (a missing filter must not
    block trading)."""
    rv = realized_vol(daily_closes)
    if rv is None:
        return None
    if spot is None or not math.isfinite(float(spot)) or float(spot) <= 0:
        spot = float(np.asarray(daily_closes, dtype=float)[-1])
    spot = float(spot)

    profile = profile_for(symbol)
    atm_iv = rv * profile.iv_mult + profile.iv_add
    base_r_t, signals = compute_surface_score(spot, atm_iv, rv, profile)

    tape_metrics = compute_tape_metrics(df_15m, atm_iv)
    if tape_metrics is not None:
        tape = tape_stress_score(tape_metrics)
        r_t = float(np.clip(0.62 * base_r_t + 0.38 * tape, 0.0, 100.0))
    else:
        tape = base_r_t  # no M15 data — final score falls back to the base
        r_t = base_r_t

    return VolContext(
        symbol=str(symbol),
        r_t=r_t,
        regime=classify_regime(r_t),
        base_r_t=base_r_t,
        tape_stress=tape,
        atm_iv=atm_iv,
        rv=rv,
        vrp=atm_iv - rv,
        spot=spot,
        em_1d=em_move(spot, atm_iv, 1),
        em_5d=em_move(spot, atm_iv, 5),
        signals=signals,
    )


# ── entry gate (pure logic — unit-testable without the Core) ─────────────────

def entry_gate(
    ctx: VolContext,
    entry_price: Optional[float],
    tp_prices: Sequence[float],
    max_r: float,
    em_tp_ratio: float,
) -> Tuple[bool, str]:
    """(ok, reason). Blocks on PANIC score or TP1 beyond the expected 1-day move.

    em_tp_ratio <= 0 disables the EM check; max_r <= 0 disables the score check.
    """
    if max_r > 0 and ctx.r_t >= max_r:
        return False, (
            f"Режим волатильности {ctx.regime}: R(t)={ctx.r_t:.0f} >= {max_r:.0f}"
        )

    if em_tp_ratio > 0 and entry_price and tp_prices:
        entry = float(entry_price)
        tp1_dist = min(abs(float(tp) - entry) for tp in tp_prices)
        limit = em_tp_ratio * ctx.em_1d
        if ctx.em_1d > 0 and tp1_dist > limit:
            return False, (
                f"TP1 дальше ожидаемого хода: {tp1_dist:.5g} > "
                f"{em_tp_ratio:.2g}×EM1D ({ctx.em_1d:.5g})"
            )
    return True, "OK"
