# core/strategy_narrative.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Literal, Tuple, Any, List

import time
import numpy as np
import pandas as pd

from core.htf_context import HtfContext
from core.pivot_trigger import mark_pivots

try:
    import config as _cfg
except Exception:
    _cfg = None

ORDERBLOCK_ENTRY_ENABLED = bool(getattr(_cfg, "ORDERBLOCK_ENTRY_ENABLED", True))
ORDERBLOCK_TOUCH_ATR_K = float(getattr(_cfg, "ORDERBLOCK_TOUCH_ATR_K", 0.15))
ORDERBLOCK_TOUCH_MIN_ABS = float(getattr(_cfg, "ORDERBLOCK_TOUCH_MIN_ABS", 0.0005))
ORDERBLOCK_MAX_AGE_BARS = int(getattr(_cfg, "ORDERBLOCK_MAX_AGE_BARS", 240))
HTF_SCORE_MARGIN = int(getattr(_cfg, "HTF_SCORE_MARGIN", 1))

# M15 EMA trend filter
EMA_FAST = 72
EMA_SLOW = 89
MIN_BARS_FOR_EMA = 100

Side = Literal["LONG", "SHORT", "NEUTRAL"]


# ================== FRACTALS (INLINE) ==================
def detect_williams_fractals_3bar(df: pd.DataFrame) -> List[Dict[str, float]]:
    """
    Williams Fractals (ONLY 3-bar mode):

    high fractal:
        high[i-1] < high[i] and high[i+1] < high[i]

    low fractal:
        low[i-1] > low[i] and low[i+1] > low[i]

    Returns list:
      {"type": "high"|"low", "price": float, "index": int}
    """
    if df is None or df.empty:
        return []
    if not {"high", "low"}.issubset(df.columns):
        return []

    highs = df["high"].astype(float).to_list()
    lows = df["low"].astype(float).to_list()
    n = len(df)

    out: List[Dict[str, float]] = []
    for i in range(1, n - 1):
        if highs[i - 1] < highs[i] and highs[i + 1] < highs[i]:
            out.append({"type": "high", "price": float(highs[i]), "index": int(i)})
        if lows[i - 1] > lows[i] and lows[i + 1] > lows[i]:
            out.append({"type": "low", "price": float(lows[i]), "index": int(i)})

    return sorted(out, key=lambda x: int(x["index"]))


# ================== STRUCTURES ==================

@dataclass
class CandidateEntry:
    side: Side
    entry_price: float
    tf: str
    reason: str

    # entry range (for "send range" style)
    entry_min: Optional[float] = None
    entry_max: Optional[float] = None

    # setup zone (RB / TS / Pivot reclaim zone)
    zone_low: Optional[float] = None
    zone_high: Optional[float] = None

    # meta
    used_m1: bool = False
    stop_override: Optional[float] = None
    lock_entry_range: bool = False


@dataclass
class ActiveTrade:
    side: Side
    entry: float
    stop: float
    tp_prices: List[float]
    tf: str
    narrative: str
    symbol: str

    tp_hit: int = 0

    telegram_chat_id: Optional[int] = None
    telegram_message_id: Optional[int] = None
    ts_open: float = field(default_factory=lambda: time.time())
    last_price_ts: float = field(default_factory=lambda: time.time())

    volume: float = 0.0
    mt5_ticket: Optional[int] = None
    mt5_position_id: Optional[int] = None
    execution_comment: Optional[str] = None

    # Partial-close management: how much to close at each TP level.
    # Computed once after entry fill; last element always captures rounding remainder.
    volume_per_tp: List[float] = field(default_factory=list)
    # Tracks remaining open volume as partial closes execute.
    volume_remaining: float = 0.0

    # MK-style split entry: one sub-position per TP, each with its own broker TP.
    # When non-empty, MT5 closes each leg automatically — bot skips manual partial closes.
    split_position_ids: List[int] = field(default_factory=list)

    # SL moved to break-even (entry) after TP1 — done at most once per trade.
    moved_to_be: bool = False


# ================== STRATEGY ==================

class NarrativeStrategy:
    """
    Top-Down:
      1) Narrative / BIAS: 4H + 1H
      2) VC (optional): 1H (SNR/FVG)
      3) SETUP: 15M Rejection Block
      4) FALLBACK: 15M Turtle Soup
      5) FALLBACK2: H1 Pivots levels + 15M reclaim
      6) ENTRY DELIVERY: send ENTRY RANGE
         - всё локализуется вокруг последнего 15M close
         - если есть RB зона -> диапазон зажимается внутри неё
      7) Stop: 1H fractal (3-bar) + ATR buffer + candle floor
      8) TP: 4 take-profits by RR (TP2 = rr_min)
    """

    def __init__(self):
        self.risk_per_trade = 0.01
        self.rr_min = 1.5

        # 4 TPs
        self.tp_rr_levels = [1.0, float(self.rr_min), 2.0, 3.0]

        # Turtle Soup (15M)
        self.ts_lookback_bars_15m = 20
        self.ts_min_sweep_pct = 0.02
        self.ts_sweep_atr_k = 0.10

        # Rejection Block (15M)
        self.rb_pivot_lookback_left = 6
        self.rb_box_length = 18
        # HARD_RIGHT: confirm candle body must stay below/above pivot body (stricter than CLASSIC midpoint)
        self.rb_body_rule = "HARD_RIGHT"

        self.rb_use_wick_to_body_filter = False
        self.rb_wick_to_body_ratio = 3.0
        self.rb_min_wick_intrusion_pct = 5.0

        # Require bearish/bullish close on confirmation candle before entry
        self.rb_require_confirm_bearish_body = True
        self.rb_require_confirm_bullish_body = True

        # HTF context config
        self.htf_fractal_limit = 20
        self.htf_pivot_lookback = 2
        self.htf_dealing_window = 120  # ≈5 дней H1
        self.htf_dealing_rows = 10
        self.htf_dealing_pivot = 3
        self.htf_ob_lookback = 10
        self.htf_ob_show_last = 3
        self.htf_rb_box_length = 12
        self.htf_rb_use_wick_filter = False
        self.htf_rb_wick_ratio = 3.0
        self.htf_rb_intrusion_pct = 20.0
        self.htf_rb_body_rule = "HARD_RIGHT"

        # H1 pivots fallback
        self.use_h1_pivot_fallback = True
        self.h1_pivot_window = 3
        self.h1_pivot_lookback_h1 = 160
        self.h1_pivot_mode = "close_reclaim"  # close_reclaim | two_close
        self.h1_pivot_wick_atr_k_15m = 0.10

        # Entry range delivery
        self.entry_range_pad_atr_k = 0.10          # pad inside RB zone
        self.entry_range_fallback_atr_k = 0.15     # fallback range around price
        self.entry_range_m1_atr_k = 0.25           # NEW: tighter localization around 1M close

        # Stop logic — raised min_risk to avoid micro-stops on noise/spread
        self.stop_buffer_atr_k = 0.05
        self.min_risk_atr_k = 0.40
        self.max_risk_atr_k = 2.0

        self.use_prev_candle_stop_floor = True

        self.orderblock_entry_enabled = bool(ORDERBLOCK_ENTRY_ENABLED)
        self.orderblock_touch_atr_k = float(ORDERBLOCK_TOUCH_ATR_K or 0.15)
        self.orderblock_touch_min_abs = float(ORDERBLOCK_TOUCH_MIN_ABS or 0.0005)
        self.orderblock_max_age_bars = int(ORDERBLOCK_MAX_AGE_BARS or 240)
        self.htf_score_margin = max(1, int(HTF_SCORE_MARGIN or 1))

        self._last_htf_context: Optional[HtfContext] = None

    # ================== HELPERS ==================

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> float:
        if df is None or df.empty or len(df) < period + 2:
            return 0.0
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c_prev = df["close"].astype(float).shift(1)
        tr = np.maximum(h - l, np.maximum((h - c_prev).abs(), (l - c_prev).abs()))
        v = tr.rolling(period).mean().iloc[-1]
        return float(v) if pd.notna(v) else 0.0

    @staticmethod
    def _sma(series: pd.Series, n: int) -> float:
        v = series.tail(n).mean()
        return float(v) if pd.notna(v) else 0.0

    @staticmethod
    def _pct_of_price(value: float, price: float) -> float:
        if price <= 0:
            return 0.0
        return (value / price) * 100.0

    @staticmethod
    def _prev_candle(df: pd.DataFrame) -> Optional[pd.Series]:
        if df is None or df.empty:
            return None
        if len(df) >= 2:
            return df.iloc[-2]
        return df.iloc[-1]

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, x)))

    # ================== DR / FRACTALS (for Bias) ==================

    @staticmethod
    def detect_fractals_1_1(df: pd.DataFrame):
        highs = []
        lows = []
        for i in range(1, len(df) - 1):
            prev = df.iloc[i - 1]
            cur = df.iloc[i]
            nxt = df.iloc[i + 1]

            if float(cur["high"]) > float(prev["high"]) and float(cur["high"]) > float(nxt["high"]):
                highs.append({"index": i, "price": float(cur["high"])})
            if float(cur["low"]) < float(prev["low"]) and float(cur["low"]) < float(nxt["low"]):
                lows.append({"index": i, "price": float(cur["low"])})
        return highs, lows

    @staticmethod
    def get_dr_from_fractals(df: pd.DataFrame) -> Optional[Dict[str, float]]:
        highs, lows = NarrativeStrategy.detect_fractals_1_1(df)
        if not highs or not lows:
            return None

        last_close = float(df["close"].iloc[-1])
        highs_above = [h for h in highs if h["price"] > last_close]
        lows_below = [l for l in lows if l["price"] < last_close]
        if not highs_above or not lows_below:
            return None

        dr_high = min(highs_above, key=lambda x: x["price"])["price"]
        dr_low = max(lows_below, key=lambda x: x["price"])["price"]
        mid = (dr_high + dr_low) / 2.0
        return {"high": dr_high, "low": dr_low, "mid": mid}

    def _build_htf_context(self, df_D: pd.DataFrame, df_4H: pd.DataFrame, df_1H: pd.DataFrame) -> Optional[HtfContext]:
        if df_4H is None or df_1H is None or df_4H.empty or df_1H.empty:
            self._last_htf_context = None
            return None
        ctx = HtfContext(
            df_1h=df_1H,
            df_4h=df_4H,
            df_daily=df_D,
            fractal_limit=self.htf_fractal_limit,
            pivot_lookback=self.htf_pivot_lookback,
            dealing_range_window=self.htf_dealing_window,
            dealing_rows=self.htf_dealing_rows,
            dealing_pivot=self.htf_dealing_pivot,
            ob_lookback=self.htf_ob_lookback,
            ob_show_last=self.htf_ob_show_last,
            rb_box_length=self.htf_rb_box_length,
            rb_use_wick_filter=self.htf_rb_use_wick_filter,
            rb_wick_ratio=self.htf_rb_wick_ratio,
            rb_intrusion_pct=self.htf_rb_intrusion_pct,
            rb_body_rule=self.htf_rb_body_rule,
        )
        self._last_htf_context = ctx
        return ctx

    def calc_narrative(self, df_D: pd.DataFrame, df_4H: pd.DataFrame, df_1H: pd.DataFrame) -> Tuple[Side, str]:
        ctx = self._build_htf_context(df_D, df_4H, df_1H)
        if ctx is None:
            return self._legacy_calc_narrative(df_D, df_4H, df_1H)

        parts: List[str] = []
        score_long = 0
        score_short = 0

        # Daily premium/discount
        if ctx.daily_range is not None:
            dr = ctx.daily_range
            parts.append(
                f"DailyPD pos={dr.position} (PDH={dr.pdh:.5f}/PDL={dr.pdl:.5f})"
            )
            if dr.bias == "LONG":
                score_long += 2
            elif dr.bias == "SHORT":
                score_short += 2

        # 5-day dealing range rows
        if ctx.dealing_range is not None:
            drange = ctx.dealing_range
            current_row = drange.current_row
            if current_row is not None:
                parts.append(
                    f"DR row={current_row.idx} dom={current_row.dominant} prob={current_row.probability:.1f}% pos={drange.position}"
                )
                if current_row.dominant == "BULL" and current_row.probability >= 55:
                    score_long += 1
                elif current_row.dominant == "BEAR" and current_row.probability >= 55:
                    score_short += 1
            else:
                parts.append(f"DR pos={drange.position}")
                if drange.bias == "LONG":
                    score_long += 1
                elif drange.bias == "SHORT":
                    score_short += 1

        # Fractal power
        if ctx.fractals is not None:
            fp = ctx.fractals
            parts.append(
                f"Fractal power {fp.bullish_power:.1f}/{fp.bearish_power:.1f}% strength={fp.strength}"
            )
            if fp.dominant == "LONG":
                score_long += 1
                if fp.bullish_power >= 65:
                    score_long += 1
            elif fp.dominant == "SHORT":
                score_short += 1
                if fp.bearish_power >= 65:
                    score_short += 1

        zone_snippets: List[str] = []
        if ctx.order_blocks:
            ob = ctx.order_blocks[0]
            zone_snippets.append(
                f"OB {ob.side} @{ob.bottom:.5f}-{ob.top:.5f}{' breaker' if ob.breaker else ''}"
            )
            if ob.side == "LONG":
                score_long += 1
            elif ob.side == "SHORT":
                score_short += 1
        if ctx.rejection_blocks:
            rb = ctx.rejection_blocks[-1]
            state = "valid" if rb.valid and not rb.broken else "invalid"
            zone_snippets.append(
                f"RB {rb.side} @{min(rb.zone_low, rb.zone_high):.5f}-{max(rb.zone_low, rb.zone_high):.5f} {state}"
            )
            if rb.valid and not rb.broken:
                if rb.side == "LONG":
                    score_long += 1
                elif rb.side == "SHORT":
                    score_short += 1
        if zone_snippets:
            parts.append("; ".join(zone_snippets))

        if not parts:
            return self._legacy_calc_narrative(df_D, df_4H, df_1H)

        summary = " | ".join(parts)
        margin = getattr(self, "htf_score_margin", 1)
        if score_long >= score_short + margin:
            return "LONG", f"HTF Bias LONG (scores L/S={score_long}/{score_short}) | {summary}"
        if score_short >= score_long + margin:
            return "SHORT", f"HTF Bias SHORT (scores L/S={score_long}/{score_short}) | {summary}"

        legacy_side, legacy_text = self._legacy_calc_narrative(df_D, df_4H, df_1H)
        if legacy_side != "NEUTRAL":
            blended = f"HTF mixed (L/S={score_long}/{score_short}) | {summary}"
            return legacy_side, blended + " :: " + legacy_text
        return "NEUTRAL", f"HTF mixed (L/S={score_long}/{score_short}) | {summary}"

    def _legacy_calc_narrative(self, df_D: pd.DataFrame, df_4H: pd.DataFrame, df_1H: pd.DataFrame) -> Tuple[Side, str]:
        if df_4H is None or df_1H is None or df_4H.empty or df_1H.empty:
            return "NEUTRAL", "Нет данных для Narrative (4H/1H)"
        if len(df_4H) < 80 or len(df_1H) < 80:
            return "NEUTRAL", "Недостаточно данных для Narrative (4H/1H)"

        dr4 = self.get_dr_from_fractals(df_4H)
        if dr4 is None:
            return "NEUTRAL", "Нет валидного DR по 4H"

        price_4h = float(df_4H["close"].iloc[-1])
        pos_4h = "DISCOUNT" if price_4h < dr4["mid"] else "PREMIUM"

        sma4 = self._sma(df_4H["close"], 50)
        sma1 = self._sma(df_1H["close"], 30)

        p4 = float(df_4H["close"].iloc[-1])
        p1 = float(df_1H["close"].iloc[-1])

        trend4 = "LONG" if p4 > sma4 else "SHORT"
        trend1 = "LONG" if p1 > sma1 else "SHORT"

        score_long = 0
        score_short = 0

        score_long += 1 if pos_4h == "DISCOUNT" else 0
        score_short += 1 if pos_4h == "PREMIUM" else 0

        score_long += 1 if trend4 == "LONG" else 0
        score_short += 1 if trend4 == "SHORT" else 0

        score_long += 1 if trend1 == "LONG" else 0
        score_short += 1 if trend1 == "SHORT" else 0

        if score_long >= 2 and score_long > score_short:
            return "LONG", (
                f"Bias LONG (TopDown 4H→1H): 4H_DR={pos_4h}, 4H_trend={trend4}, 1H_trend={trend1} "
                f"(голоса LONG/SHORT = {score_long}/{score_short})"
            )
        if score_short >= 2 and score_short > score_long:
            return "SHORT", (
                f"Bias SHORT (TopDown 4H→1H): 4H_DR={pos_4h}, 4H_trend={trend4}, 1H_trend={trend1} "
                f"(голоса LONG/SHORT = {score_long}/{score_short})"
            )

        return "NEUTRAL", (
            f"Смешанный контекст (4H→1H): 4H_DR={pos_4h}, 4H_trend={trend4}, 1H_trend={trend1} "
            f"(голоса LONG/SHORT = {score_long}/{score_short})"
        )

    # ================== VC (optional) ==================

    @staticmethod
    def detect_snr_break(df: pd.DataFrame, lookback: int = 20) -> Optional[str]:
        if len(df) < lookback + 1:
            return None
        window = df.tail(lookback + 1)
        last = window.iloc[-1]
        prev = window.iloc[:-1]
        max_high = float(prev["high"].max())
        min_low = float(prev["low"].min())
        if float(last["close"]) > max_high:
            return "BULL"
        if float(last["close"]) < min_low:
            return "BEAR"
        return None

    @staticmethod
    def detect_fvg(df: pd.DataFrame, max_bars_back: int = 15) -> Optional[Dict[str, Any]]:
        if len(df) < 5:
            return None
        end = len(df)
        start = max(2, end - max_bars_back)
        for i in range(start, end):
            c0 = df.iloc[i - 2]
            c1 = df.iloc[i - 1]
            c2 = df.iloc[i]
            max_high = max(float(c0["high"]), float(c2["high"]))
            min_low = min(float(c0["low"]), float(c2["low"]))
            if float(c1["low"]) > max_high:
                return {"type": "BULL", "low": max_high, "high": float(c1["low"]), "index": i - 1}
            if float(c1["high"]) < min_low:
                return {"type": "BEAR", "high": min_low, "low": float(c1["high"]), "index": i - 1}
        return None

    def volume_confirmation(self, df_1H: pd.DataFrame) -> tuple[Optional[str], str]:
        snr = self.detect_snr_break(df_1H)
        fvg = self.detect_fvg(df_1H)
        if snr == "BULL":
            return "BULL", "SNR breakout вверх (1H)"
        if snr == "BEAR":
            return "BEAR", "SNR breakout вниз (1H)"
        if fvg is not None:
            return ("BULL", "Bullish FVG (1H)") if fvg["type"] == "BULL" else ("BEAR", "Bearish FVG (1H)")
        return None, "Нет явного VC (SNR/FVG) на 1H"

    # ================== 15M Rejection Block Trigger ==================

    @staticmethod
    def _body_top(open_: float, close_: float) -> float:
        return float(max(open_, close_))

    @staticmethod
    def _body_bottom(open_: float, close_: float) -> float:
        return float(min(open_, close_))

    @classmethod
    def _body_size(cls, open_: float, close_: float) -> float:
        return float(cls._body_top(open_, close_) - cls._body_bottom(open_, close_))

    @classmethod
    def _upper_wick_size(cls, high: float, open_: float, close_: float) -> float:
        return float(high - cls._body_top(open_, close_))

    @classmethod
    def _lower_wick_size(cls, low: float, open_: float, close_: float) -> float:
        return float(cls._body_bottom(open_, close_) - low)

    def trigger_15m_rejection_block(self, df_15m: pd.DataFrame, side: Side) -> Optional[CandidateEntry]:
        if df_15m is None or df_15m.empty:
            return None

        pivot_left = int(self.rb_pivot_lookback_left)
        if len(df_15m) < pivot_left + 3:
            return None

        i0 = len(df_15m) - 1
        i1 = i0 - 1
        i2 = i0 - 2

        c0 = df_15m.iloc[i0]
        c1 = df_15m.iloc[i1]
        c2 = df_15m.iloc[i2]

        o0, h0, l0, cl0 = float(c0["open"]), float(c0["high"]), float(c0["low"]), float(c0["close"])
        o1, h1, l1, cl1 = float(c1["open"]), float(c1["high"]), float(c1["low"]), float(c1["close"])
        o2, h2, l2, cl2 = float(c2["open"]), float(c2["high"]), float(c2["low"]), float(c2["close"])

        body_top_1 = self._body_top(o1, cl1)
        body_bot_1 = self._body_bottom(o1, cl1)

        body_top_0 = self._body_top(o0, cl0)
        body_top_2 = self._body_top(o2, cl2)
        body_bot_2 = self._body_bottom(o2, cl2)

        body_size_1 = self._body_size(o1, cl1)
        up_wick_1 = self._upper_wick_size(h1, o1, cl1)
        dn_wick_1 = self._lower_wick_size(l1, o1, cl1)

        up_wick_0 = self._upper_wick_size(h0, o0, cl0)
        dn_wick_0 = self._lower_wick_size(l0, o0, cl0)

        start = i1 - pivot_left
        if start < 0:
            return None
        hh = float(df_15m["high"].iloc[start:i1 + 1].max())
        ll = float(df_15m["low"].iloc[start:i1 + 1].min())

        isPivotHigh = (h1 > h0) and (h1 == hh)
        isPivotLow = (l1 < l0) and (l1 == ll)

        # Box-length freshness: pivot must be the dominant extreme over rb_box_length bars.
        # Prevents signals on stale pivots already tested within the broader window.
        box_len = int(self.rb_box_length)
        box_start = max(0, i1 - box_len + 1)
        box_max_high = float(df_15m["high"].iloc[box_start:i1 + 1].max())
        box_min_low  = float(df_15m["low"].iloc[box_start:i1 + 1].min())
        isPivotHigh = isPivotHigh and (h1 == box_max_high)
        isPivotLow  = isPivotLow  and (l1 == box_min_low)

        rule = str(self.rb_body_rule or "").upper()

        # ---- BEARISH RB ----
        middleTopWick = up_wick_1 > 0
        confirmTopWick = up_wick_0 > 0

        main_wick_bear = up_wick_1
        intrusion_amt = h0 - body_top_1
        intrusion_pct = (intrusion_amt / main_wick_bear) * 100.0 if main_wick_bear > 0 else 0.0
        wickIntrusion = (intrusion_pct >= float(self.rb_min_wick_intrusion_pct)) and (h0 < h1)

        midpoint_rule = body_top_0 <= (h1 + body_top_1) / 2.0
        hard_right = body_top_0 <= body_top_1
        hard_left = body_top_2 <= body_top_1

        if rule == "HARD_BOTH":
            bodyCondition = hard_left and hard_right
        elif rule == "HARD_RIGHT":
            bodyCondition = hard_right
        elif rule == "HARD_LEFT":
            bodyCondition = hard_left
        elif rule == "CLASSIC":
            bodyCondition = midpoint_rule
        else:
            bodyCondition = False

        wick_body_ok = True
        if bool(self.rb_use_wick_to_body_filter):
            wick_body_ok = up_wick_1 >= (body_size_1 * float(self.rb_wick_to_body_ratio))

        isBearishRB = (
            isPivotHigh
            and middleTopWick
            and confirmTopWick
            and wickIntrusion
            and bodyCondition
            and wick_body_ok
        )

        if isBearishRB and side == "SHORT":
            if self.rb_require_confirm_bearish_body and not (cl0 < o0):
                return None

            return CandidateEntry(
                side="SHORT",
                entry_price=float(cl0),
                tf="15M",
                reason=f"RejectionBlock 15M BEAR | intrusion={intrusion_pct:.1f}% (min={float(self.rb_min_wick_intrusion_pct):.1f}%) | rule={rule}",
                zone_low=float(body_top_1),
                zone_high=float(h1),
            )

        # ---- BULLISH RB ----
        middleBotWick = dn_wick_1 > 0
        confirmBotWick = dn_wick_0 > 0

        main_wick_bull = dn_wick_1
        intrusion_amt2 = body_bot_1 - l0
        intrusion_pct2 = (intrusion_amt2 / main_wick_bull) * 100.0 if main_wick_bull > 0 else 0.0
        wickIntrusion2 = (intrusion_pct2 >= float(self.rb_min_wick_intrusion_pct)) and (l0 > l1)

        midpoint_rule2 = self._body_bottom(o0, cl0) >= (l1 + body_bot_1) / 2.0
        hard_right2 = self._body_bottom(o0, cl0) >= body_bot_1
        hard_left2 = body_bot_2 >= body_bot_1

        if rule == "HARD_BOTH":
            bodyCondition2 = hard_left2 and hard_right2
        elif rule == "HARD_RIGHT":
            bodyCondition2 = hard_right2
        elif rule == "HARD_LEFT":
            bodyCondition2 = hard_left2
        elif rule == "CLASSIC":
            bodyCondition2 = midpoint_rule2
        else:
            bodyCondition2 = False

        wick_body_ok2 = True
        if bool(self.rb_use_wick_to_body_filter):
            wick_body_ok2 = dn_wick_1 >= (body_size_1 * float(self.rb_wick_to_body_ratio))

        isBullishRB = (
            isPivotLow
            and middleBotWick
            and confirmBotWick
            and wickIntrusion2
            and bodyCondition2
            and wick_body_ok2
        )

        if isBullishRB and side == "LONG":
            if self.rb_require_confirm_bullish_body and not (cl0 > o0):
                return None

            return CandidateEntry(
                side="LONG",
                entry_price=float(cl0),
                tf="15M",
                reason=f"RejectionBlock 15M BULL | intrusion={intrusion_pct2:.1f}% (min={float(self.rb_min_wick_intrusion_pct):.1f}%) | rule={rule}",
                zone_low=float(l1),
                zone_high=float(body_bot_1),
            )

        return None

    # ================== 15M Turtle Soup Trigger ==================

    @staticmethod
    def turtle_soup_15m_trigger(
        df_15m: pd.DataFrame,
        lookback: int = 20,
        min_sweep_pct: float = 0.03,
        mode: str = "close_reclaim",
    ) -> Optional[Dict[str, Any]]:
        if df_15m is None or df_15m.empty or len(df_15m) < lookback + 2:
            return None

        w = df_15m.tail(lookback + 1)
        prev_window = w.iloc[:-1]
        last = w.iloc[-1]
        prev = w.iloc[-2]

        level_high = float(prev_window["high"].max())
        level_low = float(prev_window["low"].min())

        def sweep_pct(up: bool, level: float) -> float:
            if level <= 0:
                return 0.0
            if up:
                return max(0.0, (float(last["high"]) - level) / level * 100.0)
            return max(0.0, (level - float(last["low"])) / level * 100.0)

        if mode == "two_close":
            cond_short = float(prev["close"]) > level_high and float(last["close"]) < level_high
        else:
            cond_short = float(last["high"]) > level_high and float(last["close"]) < level_high
        if cond_short and sweep_pct(True, level_high) >= float(min_sweep_pct):
            return {"side": "SHORT", "kind": "TURTLE_SOUP", "level": level_high, "sweep_pct": sweep_pct(True, level_high), "mode": mode}

        if mode == "two_close":
            cond_long = float(prev["close"]) < level_low and float(last["close"]) > level_low
        else:
            cond_long = float(last["low"]) < level_low and float(last["close"]) > level_low
        if cond_long and sweep_pct(False, level_low) >= float(min_sweep_pct):
            return {"side": "LONG", "kind": "TURTLE_SOUP", "level": level_low, "sweep_pct": sweep_pct(False, level_low), "mode": mode}

        return None

    def trigger_15m_turtle_soup(self, df_15M: pd.DataFrame, side: Side) -> Optional[CandidateEntry]:
        last_price = float(df_15M["close"].iloc[-1])
        atr15 = self._atr(df_15M, 14)
        atr15_pct = self._pct_of_price(atr15, last_price)
        min_sweep = max(float(self.ts_min_sweep_pct), float(self.ts_sweep_atr_k) * atr15_pct)

        trig = self.turtle_soup_15m_trigger(df_15M, lookback=self.ts_lookback_bars_15m, min_sweep_pct=min_sweep, mode="close_reclaim")
        if trig is not None and trig.get("side") == side:
            entry = float(df_15M["close"].iloc[-1])
            lvl = float(trig["level"])
            sp = float(trig.get("sweep_pct", 0.0))
            return CandidateEntry(side=side, entry_price=entry, tf="15M", reason=f"TurtleSoup 15M reclaim lvl={lvl:.5f} sweep={sp:.3f}% (min={min_sweep:.3f}%)")

        trig2 = self.turtle_soup_15m_trigger(df_15M, lookback=self.ts_lookback_bars_15m, min_sweep_pct=min_sweep, mode="two_close")
        if trig2 is not None and trig2.get("side") == side:
            entry = float(df_15M["close"].iloc[-1])
            lvl = float(trig2["level"])
            sp = float(trig2.get("sweep_pct", 0.0))
            return CandidateEntry(side=side, entry_price=entry, tf="15M", reason=f"TurtleSoup 15M TRAP_2C lvl={lvl:.5f} sweep={sp:.3f}% (min={min_sweep:.3f}%)")
        return None

    # ================== H1 PIVOTS + 15M reclaim trigger ==================

    def trigger_h1_pivot_reclaim_on_15m(self, df_1H: pd.DataFrame, df_15M: pd.DataFrame, side: Side) -> Optional[CandidateEntry]:
        if not self.use_h1_pivot_fallback:
            return None
        if df_1H is None or df_1H.empty or len(df_1H) < 50:
            return None
        if df_15M is None or df_15M.empty or len(df_15M) < 3:
            return None

        dfp = mark_pivots(df_1H, window=int(self.h1_pivot_window))
        w = dfp.tail(int(self.h1_pivot_lookback_h1)).copy()

        piv_highs = w[w["pivot"] == 1]
        piv_lows = w[w["pivot"] == 2]

        last15 = df_15M.iloc[-1]
        prev15 = df_15M.iloc[-2]

        atr15 = self._atr(df_15M, 14)
        thr = float(atr15) * float(self.h1_pivot_wick_atr_k_15m) if atr15 > 0 else 0.0

        mode = str(self.h1_pivot_mode or "close_reclaim").lower()

        if side == "SHORT" and not piv_highs.empty:
            lvl = float(piv_highs["high"].iloc[-1])
            if mode == "two_close":
                cond = float(prev15["close"]) > lvl and float(last15["close"]) < lvl
                overshoot = abs(float(prev15["close"]) - lvl)
            else:
                cond = float(last15["high"]) > lvl and float(last15["close"]) < lvl
                overshoot = float(last15["high"]) - lvl

            if cond and (overshoot >= thr):
                return CandidateEntry(side="SHORT", entry_price=float(last15["close"]), tf="15M",
                                      reason=f"H1 PivotHigh reclaim on 15M | lvl={lvl:.5f} overshoot={overshoot:.5f} thr={thr:.5f} mode={mode}")

        if side == "LONG" and not piv_lows.empty:
            lvl = float(piv_lows["low"].iloc[-1])
            if mode == "two_close":
                cond = float(prev15["close"]) < lvl and float(last15["close"]) > lvl
                overshoot = abs(lvl - float(prev15["close"]))
            else:
                cond = float(last15["low"]) < lvl and float(last15["close"]) > lvl
                overshoot = lvl - float(last15["low"])

            if cond and (overshoot >= thr):
                return CandidateEntry(side="LONG", entry_price=float(last15["close"]), tf="15M",
                                      reason=f"H1 PivotLow reclaim on 15M | lvl={lvl:.5f} overshoot={overshoot:.5f} thr={thr:.5f} mode={mode}")

        return None

    # ================== H1 ORDERBLOCK TOUCH ==================

    def trigger_orderblock_touch(self, df_1H: pd.DataFrame, ctx: Optional[HtfContext], side: Side) -> Optional[CandidateEntry]:
        if not self.orderblock_entry_enabled:
            return None
        if ctx is None or df_1H is None or df_1H.empty or side not in {"LONG", "SHORT"}:
            return None

        order_blocks = getattr(ctx, "order_blocks", []) or []
        relevant = [ob for ob in order_blocks if getattr(ob, "side", None) == side]
        if not relevant:
            return None

        last_price = float(df_1H["close"].iloc[-1])
        atr1 = max(self._atr(df_1H, 14), 1e-9)
        tolerance = max(float(self.orderblock_touch_atr_k) * atr1, float(self.orderblock_touch_min_abs))
        now_idx = len(df_1H) - 1

        selected = None
        best_dist = None

        for ob in relevant:
            if getattr(ob, "breaker", False):
                continue
            age = now_idx - int(getattr(ob, "created_idx", now_idx))
            if self.orderblock_max_age_bars and age > self.orderblock_max_age_bars:
                continue

            boundary = float(ob.top if side == "LONG" else ob.bottom)
            dist = abs(last_price - boundary)
            if dist > tolerance:
                continue

            zone_low = float(min(ob.bottom, ob.top))
            zone_high = float(max(ob.bottom, ob.top))
            if not (zone_low - tolerance <= last_price <= zone_high + tolerance):
                continue

            if best_dist is None or dist < best_dist:
                selected = ob
                best_dist = dist

        if selected is None:
            return None

        zone_low = float(min(selected.bottom, selected.top))
        zone_high = float(max(selected.bottom, selected.top))
        boundary = float(selected.top if side == "LONG" else selected.bottom)

        pad = min(tolerance, max(1e-9, zone_high - zone_low))
        if side == "LONG":
            entry_min = max(boundary - pad, zone_low)
            entry_max = boundary
            stop_override = zone_low
        else:
            entry_min = boundary
            entry_max = min(boundary + pad, zone_high)
            stop_override = zone_high

        entry = CandidateEntry(
            side=side,
            entry_price=float(boundary),
            tf="1H",
            reason=f"OrderBlock touch {side} (H1) zone={zone_low:.5f}-{zone_high:.5f}",
            zone_low=zone_low,
            zone_high=zone_high,
            entry_min=float(entry_min),
            entry_max=float(entry_max),
            stop_override=float(stop_override),
            lock_entry_range=True,
        )
        return entry

    # ================== ENTRY RANGE BUILD (M1 LOCALIZATION) ==================

    def _build_entry_range(
        self,
        entry: CandidateEntry,
        df_15M: pd.DataFrame,
        side: Side,
    ) -> CandidateEntry:
        """
        Локализация входа остаётся на 15M:
          - диапазон строится вокруг последнего 15M close
          - ATR-параметры берутся с 15M данных
          - RB-зона по-прежнему ограничивает диапазон
        """
        if df_15M is None or df_15M.empty:
            return entry

        last_close_15m = float(df_15M["close"].iloc[-1])
        atr15m = max(float(self._atr(df_15M, 14)), 1e-9)

        pad = float(self.entry_range_pad_atr_k) * atr15m
        fb15 = float(self.entry_range_fallback_atr_k) * atr15m

        anchor = float(last_close_15m)
        entry.used_m1 = False

        # If zone exists (RB) -> clamp inside zone
        if entry.zone_low is not None and entry.zone_high is not None:
            zlo = float(min(entry.zone_low, entry.zone_high))
            zhi = float(max(entry.zone_low, entry.zone_high))

            e_min = anchor - fb15
            e_max = anchor + fb15

            # clamp inside zone
            e_min = self._clamp(e_min, zlo, zhi)
            e_max = self._clamp(e_max, zlo, zhi)

            # apply pad inside zone (avoid inversion)
            pad2 = min(pad, max(0.0, (zhi - zlo) * 0.45))
            e_min = max(e_min, zlo + pad2)
            e_max = min(e_max, zhi - pad2)

            if e_max <= e_min:
                e_min, e_max = zlo, zhi

            entry.entry_min = float(e_min)
            entry.entry_max = float(e_max)
            return entry

        # No zone -> just range around anchor
        entry.entry_min = float(anchor - fb15)
        entry.entry_max = float(anchor + fb15)
        return entry

    # ================== STOP / TP ==================

    @staticmethod
    def _williams_fractal_stop_level(df_1H: pd.DataFrame, side: Side, ref_price: float) -> Optional[float]:
        fr = detect_williams_fractals_3bar(df_1H)
        if not fr:
            return None

        if side == "LONG":
            lows = [f for f in fr if f.get("type") == "low" and float(f["price"]) < float(ref_price)]
            if not lows:
                return None
            last_low = max(lows, key=lambda x: int(x["index"]))
            return float(last_low["price"])

        if side == "SHORT":
            highs = [f for f in fr if f.get("type") == "high" and float(f["price"]) > float(ref_price)]
            if not highs:
                return None
            last_high = max(highs, key=lambda x: int(x["index"]))
            return float(last_high["price"])

        return None

    def _candle_stop_floor(self, entry: float, side: Side, df_1H: pd.DataFrame, df_4H: pd.DataFrame, buf: float) -> Optional[float]:
        if not self.use_prev_candle_stop_floor:
            return None

        p1 = self._prev_candle(df_1H)
        p4 = self._prev_candle(df_4H)
        if p1 is None or p4 is None:
            return None

        prev_h1_high = float(p1["high"])
        prev_h1_low = float(p1["low"])
        prev_h4_high = float(p4["high"])
        prev_h4_low = float(p4["low"])

        if side == "SHORT":
            lvl = max(prev_h1_high, prev_h4_high)
            floor = lvl + buf
            floor = max(floor, entry + 1e-6)
            return float(floor)

        if side == "LONG":
            lvl = min(prev_h1_low, prev_h4_low)
            floor = lvl - buf
            floor = min(floor, entry - 1e-6)
            return float(floor)

        return None

    _TP1_PCT: Dict[str, float] = {
        "EURUSD": 0.0015,
        "GBPUSD": 0.0015,
        "USDCAD": 0.0015,
        "GOLD":   0.0045,
        "XAUUSD": 0.0045,
    }

    def calc_stop_and_tps(
        self,
        entry_price: float,
        side: Side,
        df_1H: pd.DataFrame,
        df_4H: pd.DataFrame,
        custom_stop: Optional[float] = None,
        symbol: str = "",
    ) -> tuple[float, List[float]]:
        atr1 = self._atr(df_1H, 14)
        atr1 = max(float(atr1), 1e-9)

        buf = float(self.stop_buffer_atr_k) * atr1
        min_risk = float(self.min_risk_atr_k) * atr1
        max_risk = float(self.max_risk_atr_k) * atr1

        if custom_stop is not None:
            stop = float(custom_stop)
        else:
            fract_level = self._williams_fractal_stop_level(df_1H, side, ref_price=float(entry_price))

            if fract_level is None:
                stop = (entry_price - atr1) if side == "LONG" else (entry_price + atr1)
            else:
                stop = (fract_level - buf) if side == "LONG" else (fract_level + buf)

        if side == "LONG":
            stop = min(stop, entry_price - 1e-6)
            stop = min(stop, entry_price - min_risk)
        else:
            stop = max(stop, entry_price + 1e-6)
            stop = max(stop, entry_price + min_risk)

        risk = abs(entry_price - stop)
        if risk > max_risk:
            stop = (entry_price - max_risk) if side == "LONG" else (entry_price + max_risk)
        elif risk < min_risk:
            stop = (entry_price - min_risk) if side == "LONG" else (entry_price + min_risk)

        if custom_stop is None:
            candle_floor = self._candle_stop_floor(entry_price, side, df_1H, df_4H, buf=buf)
            if candle_floor is not None:
                if side == "SHORT":
                    stop = max(stop, candle_floor)
                else:
                    stop = min(stop, candle_floor)

        if side == "LONG":
            stop = min(stop, entry_price - 1e-6)
        else:
            stop = max(stop, entry_price + 1e-6)

        risk = abs(entry_price - stop)

        base = [1.0, float(self.rr_min), 2.0, 3.0]
        rr_levels = sorted(set(float(x) for x in (self.tp_rr_levels + base)))
        rr_levels = rr_levels[:4]  # 4 levels

        tps: List[float] = []
        for rr in rr_levels:
            reward = risk * rr
            tp = (entry_price + reward) if side == "LONG" else (entry_price - reward)
            tps.append(float(tp))

        tps = sorted(tps) if side == "LONG" else sorted(tps, reverse=True)

        tp1_pct = self._TP1_PCT.get(symbol.upper())
        if tp1_pct is not None:
            tp1_val = (entry_price * (1 + tp1_pct)) if side == "LONG" else (entry_price * (1 - tp1_pct))
            tps[0] = float(tp1_val)
            tps = sorted(tps) if side == "LONG" else sorted(tps, reverse=True)

        return float(stop), tps

    # ================== M15 EMA TREND FILTER ==================

    @staticmethod
    def calc_m15_ema_trend(df_15m: pd.DataFrame) -> Tuple[str, str]:
        """
        Determines local trend direction on M15 using EMA72 and EMA89 as a dynamic
        support/resistance zone. Returns (m15_trend, description_text).

        Rules:
          bullish : price above EMA zone — close > EMA72 AND close > EMA89 AND EMA72 > EMA89
          bearish : price below EMA zone — close < EMA72 AND close < EMA89 AND EMA72 < EMA89
          neutral : price inside or between the EMAs
        """
        if df_15m is None or df_15m.empty:
            return (
                "neutral",
                "M15 trend: neutral. Insufficient M15 data for EMA calculation.",
            )

        if "close" not in df_15m.columns:
            return (
                "neutral",
                "M15 trend: neutral. Missing 'close' column in M15 data.",
            )

        if len(df_15m) < MIN_BARS_FOR_EMA:
            return (
                "neutral",
                f"M15 trend: neutral. Insufficient M15 data for EMA calculation "
                f"(have {len(df_15m)} bars, need at least {MIN_BARS_FOR_EMA}).",
            )

        close = df_15m["close"].astype(float)
        ema72 = close.ewm(span=EMA_FAST, adjust=False).mean()
        ema89 = close.ewm(span=EMA_SLOW, adjust=False).mean()

        last_close = float(close.iloc[-1])
        last_ema72 = float(ema72.iloc[-1])
        last_ema89 = float(ema89.iloc[-1])

        if not pd.notna(last_ema72) or not pd.notna(last_ema89):
            return (
                "neutral",
                "M15 trend: neutral. EMA values could not be calculated.",
            )

        ema_aligned_bull = last_ema72 > last_ema89
        ema_aligned_bear = last_ema72 < last_ema89

        # Price above both EMAs — bullish only when EMA cross is confirmed
        if last_close > last_ema72 and last_close > last_ema89:
            if ema_aligned_bull:
                return (
                    "bullish",
                    f"M15 trend: bullish. Price ({last_close:.5f}) above "
                    f"EMA72 ({last_ema72:.5f}) and EMA89 ({last_ema89:.5f}) — EMA72>EMA89 confirmed.",
                )
            return (
                "neutral",
                f"M15 trend: neutral (early bullish). Price ({last_close:.5f}) above "
                f"EMA72 ({last_ema72:.5f}) and EMA89 ({last_ema89:.5f}) — EMA72<EMA89 (transition).",
            )

        # Price below both EMAs — bearish only when EMA cross is confirmed
        if last_close < last_ema72 and last_close < last_ema89:
            if ema_aligned_bear:
                return (
                    "bearish",
                    f"M15 trend: bearish. Price ({last_close:.5f}) below "
                    f"EMA72 ({last_ema72:.5f}) and EMA89 ({last_ema89:.5f}) — EMA72<EMA89 confirmed.",
                )
            return (
                "neutral",
                f"M15 trend: neutral (early bearish). Price ({last_close:.5f}) below "
                f"EMA72 ({last_ema72:.5f}) and EMA89 ({last_ema89:.5f}) — EMA72>EMA89 (transition).",
            )

        # Price inside EMA zone → neutral
        return (
            "neutral",
            (
                f"M15 trend: neutral. Price ({last_close:.5f}) inside "
                f"EMA72/EMA89 zone ({last_ema72:.5f}/{last_ema89:.5f})."
            ),
        )

    # ================== MAIN SIGNAL ==================

    def generate_signal(self, data: Dict[str, pd.DataFrame], symbol: str = "") -> Dict[str, Any]:
        df_4H = data.get("4H")
        df_1H = data.get("1H")
        df_15M = data.get("15M")

        if any(d is None or d.empty for d in (df_4H, df_1H, df_15M)):
            return {"signal": "NO_DATA"}

        side_bias, narrative_text = self.calc_narrative(data.get("D"), df_4H, df_1H)
        ctx = self._last_htf_context
        if side_bias == "NEUTRAL":
            return {"signal": "NO_TREND", "narrative": narrative_text}

        m15_trend, m15_trend_text = self.calc_m15_ema_trend(df_15M)

        # Block entry when M15 EMA trend directly opposes HTF bias
        if m15_trend == "bearish" and side_bias == "LONG":
            return {"signal": "WAIT_M15_EMA", "narrative": narrative_text, "m15_trend": m15_trend_text, "m15_trend_raw": m15_trend}
        if m15_trend == "bullish" and side_bias == "SHORT":
            return {"signal": "WAIT_M15_EMA", "narrative": narrative_text, "m15_trend": m15_trend_text, "m15_trend_raw": m15_trend}

        vc_dir, vc_text = self.volume_confirmation(df_1H)
        if vc_dir == "BEAR" and side_bias == "LONG":
            return {"signal": "WAIT_PHASE", "narrative": narrative_text, "vc": vc_text + " | Конфликт с LONG bias"}
        if vc_dir == "BULL" and side_bias == "SHORT":
            return {"signal": "WAIT_PHASE", "narrative": narrative_text, "vc": vc_text + " | Конфликт с SHORT bias"}

        entry = self.trigger_15m_rejection_block(df_15M, side_bias)
        if entry is None:
            entry = self.trigger_15m_turtle_soup(df_15M, side_bias)
        if entry is None:
            entry = self.trigger_h1_pivot_reclaim_on_15m(df_1H, df_15M, side_bias)
        if entry is None:
            entry = self.trigger_orderblock_touch(df_1H, ctx, side_bias)

        if entry is None:
            return {"signal": "NO_TRIGGER", "narrative": narrative_text, "vc": vc_text}

        # build entry range (фиксированная 15M локализация)
        if not getattr(entry, "lock_entry_range", False):
            entry = self._build_entry_range(entry, df_15M, side_bias)

        # risk entry: LONG uses entry_max, SHORT uses entry_min
        if side_bias == "LONG":
            entry_for_risk = float(entry.entry_max if entry.entry_max is not None else entry.entry_price)
        else:
            entry_for_risk = float(entry.entry_min if entry.entry_min is not None else entry.entry_price)

        stop, tp_prices = self.calc_stop_and_tps(
            entry_for_risk,
            entry.side,
            df_1H,
            df_4H,
            custom_stop=getattr(entry, "stop_override", None),
            symbol=symbol,
        )
        rr_text = f"1:{float(self.rr_min):.2f}"

        payload: Dict[str, Any] = {
            "signal": "ENTER",
            "side": entry.side,

            "entry_min": round(float(entry.entry_min), 6) if entry.entry_min is not None else None,
            "entry_max": round(float(entry.entry_max), 6) if entry.entry_max is not None else None,

            # compat
            "entry_price": round(float(entry_for_risk), 6),
            "stop_price": round(float(stop), 6),

            # compat
            "tp_price": round(float(tp_prices[-1]), 6),
            "tp_prices": [round(float(x), 6) for x in tp_prices],  # 4 TPs

            "rr": rr_text,
            "rr_numeric": float(self.rr_min),

            "risk_percent": f"{self.risk_per_trade * 100:.2f}%",

            # IMPORTANT: show what timeframe we localized on
            "tf": "15M",
            "setup_tf": "15M",

            "narrative": narrative_text,
            "vc": vc_text,
            "trigger_reason": entry.reason,
            "m1_localized": False,
            "m15_trend": m15_trend_text,
            "m15_trend_raw": m15_trend,
        }

        if entry.zone_low is not None and entry.zone_high is not None:
            payload["zone_low"] = round(float(min(entry.zone_low, entry.zone_high)), 6)
            payload["zone_high"] = round(float(max(entry.zone_low, entry.zone_high)), 6)

        return payload
