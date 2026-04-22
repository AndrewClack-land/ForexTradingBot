from __future__ import annotations

from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd


def mark_pivots(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Pivot подтверждается только после наличия правой стороны (future bars),
    поэтому последние window баров обычно без pivot — это нормально.
    pivot:
      0 = none
      1 = pivot high
      2 = pivot low
    """
    df = df.copy()
    n = len(df)
    piv = np.zeros(n, dtype=int)

    if n < 2 * window + 1:
        df["pivot"] = piv
        return df

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)

    span = 2 * window + 1

    for i in range(span - 1, n):
        left = i - (span - 1)
        right = i
        center = i - window

        window_highs = highs[left:right + 1]
        window_lows = lows[left:right + 1]

        ch = highs[center]
        cl = lows[center]

        if ch >= window_highs.max():
            piv[center] = 1

        if cl <= window_lows.min() and piv[center] == 0:
            piv[center] = 2

    df["pivot"] = piv
    return df


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or df.empty or len(df) < period + 2:
        return 0.0
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c_prev = df["close"].astype(float).shift(1)
    tr = np.maximum(h - l, np.maximum((h - c_prev).abs(), (l - c_prev).abs()))
    v = tr.rolling(period).mean().iloc[-1]
    return float(v) if pd.notna(v) else 0.0


def _pick_h1_pivot_high(df_1h_piv: pd.DataFrame, price: float) -> Optional[Tuple[float, Any]]:
    piv = df_1h_piv[df_1h_piv["pivot"] == 1][["high"]].copy()
    if piv.empty:
        return None
    piv["high"] = piv["high"].astype(float)

    above = piv[piv["high"] > float(price)]
    if not above.empty:
        idx = above["high"].idxmin()  # ближайший сверху
        return float(above.loc[idx, "high"]), idx

    # fallback: последний pivot high (если нет выше цены)
    idx = piv.index[-1]
    return float(piv.iloc[-1]["high"]), idx


def _pick_h1_pivot_low(df_1h_piv: pd.DataFrame, price: float) -> Optional[Tuple[float, Any]]:
    piv = df_1h_piv[df_1h_piv["pivot"] == 2][["low"]].copy()
    if piv.empty:
        return None
    piv["low"] = piv["low"].astype(float)

    below = piv[piv["low"] < float(price)]
    if not below.empty:
        idx = below["low"].idxmax()  # ближайший снизу
        return float(below.loc[idx, "low"]), idx

    # fallback: последний pivot low (если нет ниже цены)
    idx = piv.index[-1]
    return float(piv.iloc[-1]["low"]), idx


def m15_pivot_trigger_h1(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    pivot_window_1h: int = 3,
    pivot_lookback_1h: int = 200,
    lookback_15m: int = 120,
    mode: str = "close_reclaim",          # "close_reclaim" | "two_close"
    wick_confirm: bool = True,
    min_wick_pct: float = 0.0,            # запасной фильтр
    min_wick_atr_k: float = 0.10,         # ОСНОВНОЙ: доля ATR(15m)
    atr_period_15m: int = 14,
) -> Optional[Dict[str, Any]]:
    """
    ✅ Пивоты строим ТОЛЬКО на 1H, вход ловим на 15M.

    threshold прокола = max(level * min_wick_pct/100, ATR(15m)*min_wick_atr_k)

    SHORT (от H1 pivot high):
      close_reclaim: last.high > lvl AND last.close < lvl
      two_close:     prev.close > lvl AND last.close < lvl

    LONG (от H1 pivot low):
      close_reclaim: last.low < lvl AND last.close > lvl
      two_close:     prev.close < lvl AND last.close > lvl
    """
    if df_15m is None or df_15m.empty or len(df_15m) < 20:
        return None
    if df_1h is None or df_1h.empty or len(df_1h) < (2 * pivot_window_1h + 5):
        return None

    need_cols_15 = {"high", "low", "close"}
    need_cols_1h = {"high", "low", "close"}
    if not need_cols_15.issubset(df_15m.columns) or not need_cols_1h.issubset(df_1h.columns):
        return None

    w15 = df_15m.tail(lookback_15m).copy()
    if len(w15) < 3:
        return None

    last = w15.iloc[-1]
    prev = w15.iloc[-2]
    price = float(last["close"])

    # ATR(15m) для универсального порога
    atr15 = _atr(w15, period=atr_period_15m)
    atr_thr = float(atr15) * float(min_wick_atr_k) if atr15 > 0 else 0.0

    def _threshold(level: float) -> float:
        pct_thr = abs(float(level)) * (float(min_wick_pct) / 100.0) if min_wick_pct > 0 else 0.0
        return max(pct_thr, atr_thr)

    def _overshoot_abs(up: bool, level: float) -> float:
        if up:
            return float(last["high"]) - float(level)
        return float(level) - float(last["low"])

    def _wick_ok(up: bool, level: float) -> bool:
        if not wick_confirm:
            return True
        level = float(level)
        if level <= 0:
            return False

        # факт прокола
        if up:
            if float(last["high"]) <= level:
                return False
        else:
            if float(last["low"]) >= level:
                return False

        # минимальная величина прокола
        thr = _threshold(level)
        return _overshoot_abs(up, level) >= thr

    # ===== 1H pivots =====
    df_1h_piv = mark_pivots(df_1h, window=int(pivot_window_1h))
    w1 = df_1h_piv.tail(int(pivot_lookback_1h)).copy()
    if "pivot" not in w1.columns:
        return None

    # ===== SHORT от H1 pivot high =====
    ph = _pick_h1_pivot_high(w1, price=price)
    if ph is not None:
        lvl_h, piv_ts = ph

        if mode == "two_close":
            cond = float(prev["close"]) > lvl_h and float(last["close"]) < lvl_h
        else:
            cond = float(last["high"]) > lvl_h and float(last["close"]) < lvl_h

        if cond and _wick_ok(True, lvl_h):
            return {
                "side": "SHORT",
                "kind": "TRAP_2C" if mode == "two_close" else "REJECT",
                "level": float(lvl_h),
                "pivot_kind": "H1_PIVOT_HIGH",
                "pivot_tf": "1H",
                "pivot_ts": piv_ts,
                "reclaim_close": float(last["close"]),
                "overshoot_abs": float(_overshoot_abs(True, lvl_h)),
                "atr_15m": float(atr15),
                "thr": float(_threshold(lvl_h)),
                "mode": mode,
            }

    # ===== LONG от H1 pivot low =====
    pl = _pick_h1_pivot_low(w1, price=price)
    if pl is not None:
        lvl_l, piv_ts = pl

        if mode == "two_close":
            cond = float(prev["close"]) < lvl_l and float(last["close"]) > lvl_l
        else:
            cond = float(last["low"]) < lvl_l and float(last["close"]) > lvl_l

        if cond and _wick_ok(False, lvl_l):
            return {
                "side": "LONG",
                "kind": "TRAP_2C" if mode == "two_close" else "REJECT",
                "level": float(lvl_l),
                "pivot_kind": "H1_PIVOT_LOW",
                "pivot_tf": "1H",
                "pivot_ts": piv_ts,
                "reclaim_close": float(last["close"]),
                "overshoot_abs": float(_overshoot_abs(False, lvl_l)),
                "atr_15m": float(atr15),
                "thr": float(_threshold(lvl_l)),
                "mode": mode,
            }

    return None
