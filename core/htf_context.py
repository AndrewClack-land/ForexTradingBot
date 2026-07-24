from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

Side = Literal["LONG", "SHORT", "NEUTRAL"]


@dataclass
class FractalLevel:
    kind: Literal["HIGH", "LOW"]
    price: float
    index: int
    broken: bool = False
    break_index: Optional[int] = None
    break_price: Optional[float] = None


@dataclass
class FractalPower:
    levels: List[FractalLevel] = field(default_factory=list)
    bullish_breaks: int = 0
    bearish_breaks: int = 0
    bullish_power: float = 50.0
    bearish_power: float = 50.0
    dominant: Side = "NEUTRAL"
    strength: str = "Neutral"

    @property
    def total_breaks(self) -> int:
        return int(self.bullish_breaks + self.bearish_breaks)

    def to_dict(self) -> dict:
        return {
            "bullish_breaks": int(self.bullish_breaks),
            "bearish_breaks": int(self.bearish_breaks),
            "bullish_power": float(self.bullish_power),
            "bearish_power": float(self.bearish_power),
            "dominant": self.dominant,
            "strength": self.strength,
        }


@dataclass
class HourlyRange:
    """Current price location inside the latest completed H1 candle range."""

    high: float
    low: float
    close: float
    position: Literal["PREMIUM", "DISCOUNT", "EQ"]

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0 if self.high > self.low else self.high

    @property
    def bias(self) -> Side:
        if self.position == "PREMIUM":
            return "SHORT"
        if self.position == "DISCOUNT":
            return "LONG"
        return "NEUTRAL"

    def to_dict(self) -> dict:
        return {
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "mid": float(self.mid),
            "position": self.position,
        }


@dataclass
class FractalBreakout:
    """Latest interaction with a confirmed Williams fractal level.

    FALSE_BREAK — wick pierced the level but the bar closed back inside → reversal vote.
    TRUE_BREAK  — the bar closed beyond the level → continuation vote.
    """
    kind: Literal["FALSE_BREAK", "TRUE_BREAK"]
    side: Side               # voting direction implied by the event
    level: float
    level_kind: Literal["HIGH", "LOW"]
    bar_index: int
    bars_ago: int
    timeframe: str

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "side": self.side,
            "level": float(self.level),
            "level_kind": self.level_kind,
            "bars_ago": int(self.bars_ago),
            "timeframe": self.timeframe,
        }


@dataclass
class OrderBlock:
    side: Side
    top: float
    bottom: float
    created_idx: int
    breaker: bool = False
    breaker_idx: Optional[int] = None
    breaker_price: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "type": "OB",
            "side": self.side,
            "top": self.top,
            "bottom": self.bottom,
            "breaker": self.breaker,
        }


@dataclass
class RejectionBlock:
    side: Side
    zone_high: float
    zone_low: float
    created_idx: int
    midline: float
    wick_ratio: float
    intrusion_pct: float
    broken: bool = False
    valid: bool = True

    def to_dict(self) -> dict:
        return {
            "type": "RB",
            "side": self.side,
            "zone_high": self.zone_high,
            "zone_low": self.zone_low,
            "valid": self.valid and not self.broken,
        }


class OrderBlockTracker:
    def __init__(self, *, swing_lookback: int = 10, show_last: int = 3, use_body: bool = False):
        self.swing_lookback = max(3, int(swing_lookback))
        self.show_last = max(1, int(show_last))
        self.use_body = bool(use_body)

    def build(self, df: pd.DataFrame) -> List[OrderBlock]:
        if df is None or df.empty or len(df) < self.swing_lookback + 5:
            return []
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        open_ = df["open"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        max_arr = np.maximum(close, open_) if self.use_body else high
        min_arr = np.minimum(close, open_) if self.use_body else low

        bullish: List[OrderBlock] = []
        bearish: List[OrderBlock] = []
        top_crossed = False
        bot_crossed = False
        n = len(df)

        for i in range(self.swing_lookback, n):
            if i - self.swing_lookback <= 0:
                continue
            window_high = high[i - self.swing_lookback : i + 1]
            window_low = low[i - self.swing_lookback : i + 1]
            if len(window_high) < self.swing_lookback + 1:
                continue

            # Detect bullish OB when price crosses swing high
            if not top_crossed and close[i] > np.max(window_high[:-1]):
                top_crossed = True
                maxima = max_arr[i - 1]
                minima = min_arr[i - 1]
                loc = i - 1
                for k in range(1, min(200, i)):
                    idx = i - k
                    if idx <= 0:
                        break
                    if min_arr[idx] <= minima:
                        minima = min_arr[idx]
                        maxima = max_arr[idx]
                        loc = idx
                bullish.insert(0, OrderBlock(side="LONG", top=float(maxima), bottom=float(minima), created_idx=loc))

            if top_crossed:
                active_bull = next((ob for ob in bullish if not ob.breaker), None)
                if active_bull is not None and close[i] < active_bull.bottom:
                    active_bull.breaker = True
                    active_bull.breaker_idx = i
                    active_bull.breaker_price = float(close[i])
                    top_crossed = False

            # A bullish OB broken down becomes a historical breaker zone. Once
            # price subsequently closes back above its top, the zone is spent.
            bullish = [
                ob for ob in bullish
                if not (ob.breaker and close[i] > ob.top)
            ]

            # Detect bearish OB when price crosses swing low
            if not bot_crossed and close[i] < np.min(window_low[:-1]):
                bot_crossed = True
                maxima = max_arr[i - 1]
                minima = min_arr[i - 1]
                loc = i - 1
                for k in range(1, min(200, i)):
                    idx = i - k
                    if idx <= 0:
                        break
                    if max_arr[idx] >= maxima:
                        maxima = max_arr[idx]
                        minima = min_arr[idx]
                        loc = idx
                bearish.insert(0, OrderBlock(side="SHORT", top=float(maxima), bottom=float(minima), created_idx=loc))

            if bot_crossed:
                active_bear = next((ob for ob in bearish if not ob.breaker), None)
                if active_bear is not None and close[i] > active_bear.top:
                    active_bear.breaker = True
                    active_bear.breaker_idx = i
                    active_bear.breaker_price = float(close[i])
                    bot_crossed = False

            bearish = [
                ob for ob in bearish
                if not (ob.breaker and close[i] < ob.bottom)
            ]

        # Do not concatenate LONG before SHORT: callers that inspect the first
        # zone would otherwise acquire a permanent bullish bias. Keep the most
        # recent zones from both sides in one chronological ordering.
        blocks = bullish[: self.show_last] + bearish[: self.show_last]
        return sorted(blocks, key=lambda ob: ob.created_idx, reverse=True)


class RejectionBlockTracker:
    def __init__(
        self,
        *,
        pivot_left: int = 1,
        box_length: int = 6,
        wick_to_body_ratio: float = 3.0,
        min_intrusion_pct: float = 25.0,
        use_wick_body_filter: bool = False,
        body_rule: str = "HARD_RIGHT",
    ) -> None:
        self.pivot_left = max(0, int(pivot_left))
        self.box_length = max(1, int(box_length))
        self.wick_to_body_ratio = float(max(0.1, wick_to_body_ratio))
        self.min_intrusion_pct = float(max(0.0, min_intrusion_pct))
        self.use_wick_body_filter = bool(use_wick_body_filter)
        self.body_rule = (body_rule or "HARD_RIGHT").upper()

    def build(self, df: pd.DataFrame) -> List[RejectionBlock]:
        if df is None or df.empty or len(df) < self.pivot_left + 3:
            return []
        open_ = df["open"].astype(float).to_numpy()
        high = df["high"].astype(float).to_numpy()
        low = df["low"].astype(float).to_numpy()
        close = df["close"].astype(float).to_numpy()
        n = len(df)
        rbs: List[RejectionBlock] = []

        for i in range(self.pivot_left + 2, n):
            i0 = i
            i1 = i - 1
            i2 = i - 2
            body_top = lambda o, c: max(o, c)
            body_bottom = lambda o, c: min(o, c)
            body_size = lambda o, c: abs(o - c)
            upper_wick = lambda h, o, c: h - body_top(o, c)
            lower_wick = lambda l, o, c: body_bottom(o, c) - l

            start = max(0, i1 - self.pivot_left)
            isPivotHigh = high[i1] > high[i0] and high[i1] >= np.max(high[start : i1 + 1])
            isPivotLow = low[i1] < low[i0] and low[i1] <= np.min(low[start : i1 + 1])

            if isPivotHigh:
                wick = upper_wick(high[i1], open_[i1], close[i1])
                if wick <= 0:
                    continue
                intrusion = high[i0] - body_top(open_[i1], close[i1])
                intrusion_pct = (intrusion / wick) * 100.0 if wick > 0 else 0.0
                if intrusion_pct < self.min_intrusion_pct or high[i0] >= high[i1]:
                    continue
                if self.use_wick_body_filter and wick < body_size(open_[i1], close[i1]) * self.wick_to_body_ratio:
                    continue
                if not self._body_rule_ok(body_top(open_[i0], close[i0]), body_top(open_[i1], close[i1]), body_top(open_[i2], close[i2])):
                    continue
                rbs.append(
                    RejectionBlock(
                        side="SHORT",
                        zone_high=float(high[i1]),
                        zone_low=float(body_top(open_[i1], close[i1])),
                        created_idx=i1,
                        midline=float((high[i1] + body_top(open_[i1], close[i1])) / 2.0),
                        wick_ratio=float(wick / body_size(open_[i1], close[i1]) if body_size(open_[i1], close[i1]) > 0 else 0),
                        intrusion_pct=float(intrusion_pct),
                    )
                )

            if isPivotLow:
                wick = lower_wick(low[i1], open_[i1], close[i1])
                if wick <= 0:
                    continue
                intrusion = body_bottom(open_[i1], close[i1]) - low[i0]
                intrusion_pct = (intrusion / wick) * 100.0 if wick > 0 else 0.0
                if intrusion_pct < self.min_intrusion_pct or low[i0] <= low[i1]:
                    continue
                if self.use_wick_body_filter and wick < body_size(open_[i1], close[i1]) * self.wick_to_body_ratio:
                    continue
                if not self._body_rule_ok(body_bottom(open_[i0], close[i0]), body_bottom(open_[i1], close[i1]), body_bottom(open_[i2], close[i2]), bullish=True):
                    continue
                rbs.append(
                    RejectionBlock(
                        side="LONG",
                        zone_high=float(body_bottom(open_[i1], close[i1])),
                        zone_low=float(low[i1]),
                        created_idx=i1,
                        midline=float((body_bottom(open_[i1], close[i1]) + low[i1]) / 2.0),
                        wick_ratio=float(wick / body_size(open_[i1], close[i1]) if body_size(open_[i1], close[i1]) > 0 else 0),
                        intrusion_pct=float(intrusion_pct),
                    )
                )

        # expire / mark broken
        last_idx = len(df) - 1
        for rb in rbs:
            if last_idx - rb.created_idx > self.box_length:
                rb.valid = False
            subsequent_closes = close[rb.created_idx + 1 :]
            if (
                rb.side == "LONG"
                and len(subsequent_closes)
                and bool(np.any(subsequent_closes < rb.zone_low))
            ):
                rb.broken = True
            if (
                rb.side == "SHORT"
                and len(subsequent_closes)
                and bool(np.any(subsequent_closes > rb.zone_high))
            ):
                rb.broken = True
        return rbs[-4:]

    def _body_rule_ok(self, body_current: float, body_rb: float, body_prev: float, bullish: bool = False) -> bool:
        rule = self.body_rule
        if rule == "HARD_BOTH":
            return (body_prev <= body_rb if bullish else body_prev >= body_rb) and (body_current <= body_rb if bullish else body_current >= body_rb)
        if rule == "HARD_LEFT":
            return body_prev <= body_rb if bullish else body_prev >= body_rb
        if rule == "HARD_RIGHT":
            return body_current <= body_rb if bullish else body_current >= body_rb
        if rule == "CLASSIC":
            mid = (body_rb + body_prev) / 2.0
            return body_current >= mid if bullish else body_current <= mid
        return True


class HtfContext:
    def __init__(
        self,
        df_1h: Optional[pd.DataFrame],
        df_4h: Optional[pd.DataFrame],
        df_15m: Optional[pd.DataFrame],
        *,
        fractal_limit: int = 20,
        pivot_lookback: int = 2,
        ob_lookback: int = 10,
        ob_show_last: int = 3,
        rb_box_length: int = 6,
        rb_use_wick_filter: bool = False,
        rb_wick_ratio: float = 3.0,
        rb_intrusion_pct: float = 25.0,
        rb_body_rule: str = "HARD_RIGHT",
    ) -> None:
        self.df_1h = df_1h
        self.df_4h = df_4h
        self.df_15m = df_15m

        self.fractal_limit = max(5, int(fractal_limit))
        self.pivot_lookback = max(1, int(pivot_lookback))

        self.ob_tracker = OrderBlockTracker(swing_lookback=ob_lookback, show_last=ob_show_last)
        self.rb_tracker = RejectionBlockTracker(
            pivot_left=1,
            box_length=rb_box_length,
            use_wick_body_filter=rb_use_wick_filter,
            wick_to_body_ratio=rb_wick_ratio,
            min_intrusion_pct=rb_intrusion_pct,
            body_rule=rb_body_rule,
        )

        self.fractals: Optional[FractalPower] = None
        self.hourly_range: Optional[HourlyRange] = None
        self.order_blocks: List[OrderBlock] = []
        self.rejection_blocks: List[RejectionBlock] = []
        self.false_breakout_4h: Optional[FractalBreakout] = None
        self.true_breakout_15m: Optional[FractalBreakout] = None

        if self.df_1h is not None and not self.df_1h.empty:
            self.fractals = self._calc_fractals()
            self.hourly_range = self._calc_hourly_range()
            self.order_blocks = self.ob_tracker.build(self.df_1h)
            self.rejection_blocks = self.rb_tracker.build(self.df_1h)
        if self.df_4h is not None and not self.df_4h.empty:
            latest_4h_breakout = self._calc_latest_fractal_breakout(
                self.df_4h,
                timeframe="4H",
            )
            if (
                latest_4h_breakout is not None
                and latest_4h_breakout.kind == "FALSE_BREAK"
            ):
                self.false_breakout_4h = latest_4h_breakout
        if self.df_15m is not None and not self.df_15m.empty:
            latest_15m_breakout = self._calc_latest_fractal_breakout(
                self.df_15m,
                timeframe="15M",
            )
            if (
                latest_15m_breakout is not None
                and latest_15m_breakout.kind == "TRUE_BREAK"
            ):
                self.true_breakout_15m = latest_15m_breakout

    # ---------------------- FRACTALS ----------------------

    def _calc_fractals(self) -> Optional[FractalPower]:
        df = self.df_1h
        if df is None or df.empty or len(df) < self.pivot_lookback * 2 + 3:
            return None

        highs = df["high"].astype(float).to_numpy()
        lows = df["low"].astype(float).to_numpy()
        closes = df["close"].astype(float).to_numpy()
        n = len(df)

        levels: List[FractalLevel] = []
        left = self.pivot_lookback
        right = self.pivot_lookback
        for i in range(left, n - right):
            high_window = highs[i - left : i + right + 1]
            low_window = lows[i - left : i + right + 1]
            if len(high_window) < left + right + 1:
                continue
            if highs[i] == high_window.max() and (highs[i] > high_window[:-1].max() or highs[i] >= high_window[1:].max()):
                levels.append(FractalLevel(kind="HIGH", price=float(highs[i]), index=i))
            if lows[i] == low_window.min() and (lows[i] < low_window[:-1].min() or lows[i] <= low_window[1:].min()):
                levels.append(FractalLevel(kind="LOW", price=float(lows[i]), index=i))

        levels = sorted(levels, key=lambda x: x.index)
        if len(levels) > self.fractal_limit:
            levels = levels[-self.fractal_limit :]

        bullish_breaks = 0
        bearish_breaks = 0
        for lvl in levels:
            subsequent = closes[lvl.index + 1 :]
            if not len(subsequent):
                continue
            if lvl.kind == "HIGH":
                mask = subsequent > lvl.price
                if mask.any():
                    first_idx = np.argmax(mask)
                    lvl.broken = True
                    lvl.break_index = int(lvl.index + 1 + first_idx)
                    lvl.break_price = float(subsequent[first_idx])
                    bullish_breaks += 1
            else:
                mask = subsequent < lvl.price
                if mask.any():
                    first_idx = np.argmax(mask)
                    lvl.broken = True
                    lvl.break_index = int(lvl.index + 1 + first_idx)
                    lvl.break_price = float(subsequent[first_idx])
                    bearish_breaks += 1

        total_breaks = bullish_breaks + bearish_breaks
        if total_breaks == 0:
            bullish_power = bearish_power = 50.0
        else:
            bullish_power = (bullish_breaks / total_breaks) * 100.0
            bearish_power = (bearish_breaks / total_breaks) * 100.0

        dominant = "NEUTRAL"
        strength = "Neutral"
        dominant_power = max(bullish_power, bearish_power)
        if dominant_power >= 55:
            dominant = "LONG" if bullish_power > bearish_power else "SHORT"
            if dominant_power >= 80:
                strength = "Very Strong"
            elif dominant_power >= 70:
                strength = "Strong"
            elif dominant_power >= 60:
                strength = "Moderate"
            else:
                strength = "Weak"

        return FractalPower(
            levels=levels,
            bullish_breaks=bullish_breaks,
            bearish_breaks=bearish_breaks,
            bullish_power=float(bullish_power),
            bearish_power=float(bearish_power),
            dominant=dominant,
            strength=strength,
        )

    # ---------------------- TIMEFRAME FRACTAL BREAKOUTS ----------------------

    def _calc_latest_fractal_breakout(
        self,
        df: pd.DataFrame,
        *,
        timeframe: str,
        scan_bars: int = 10,
    ) -> Optional[FractalBreakout]:
        """Most recent interaction with a confirmed Williams fractal.

        A fractal at index i (pivot_lookback bars each side) is confirmed only at
        i + pivot_lookback, so a bar j may interact solely with levels where
        i + pivot_lookback < j — no lookahead. Scanning newest-first, the first
        event of either kind wins. TRUE_BREAK requires a close crossing the level,
        rather than merely remaining beyond an already-broken level.
        """
        lb = self.pivot_lookback
        if df is None or df.empty or len(df) < lb * 2 + 3:
            return None

        highs = df["high"].astype(float).to_numpy()
        lows = df["low"].astype(float).to_numpy()
        closes = df["close"].astype(float).to_numpy()
        n = len(df)

        frac_high_idx: List[int] = []
        frac_low_idx: List[int] = []
        for i in range(lb, n - lb):
            hw = highs[i - lb : i + lb + 1]
            lw = lows[i - lb : i + lb + 1]
            if highs[i] == hw.max() and highs[i] > np.delete(hw, lb).max():
                frac_high_idx.append(i)
            if lows[i] == lw.min() and lows[i] < np.delete(lw, lb).min():
                frac_low_idx.append(i)

        if not frac_high_idx and not frac_low_idx:
            return None

        start = max(lb + 1, n - int(scan_bars))
        for j in range(n - 1, start - 1, -1):
            confirmed_highs = [i for i in frac_high_idx if i + lb < j]
            confirmed_lows = [i for i in frac_low_idx if i + lb < j]
            lvl_high = highs[confirmed_highs[-1]] if confirmed_highs else None
            lvl_low = lows[confirmed_lows[-1]] if confirmed_lows else None
            previous_close = closes[j - 1]

            if lvl_high is not None and highs[j] > lvl_high and closes[j] < lvl_high:
                return FractalBreakout(
                    kind="FALSE_BREAK",
                    side="SHORT",
                    level=float(lvl_high),
                    level_kind="HIGH",
                    bar_index=j,
                    bars_ago=n - 1 - j,
                    timeframe=timeframe,
                )
            if lvl_low is not None and lows[j] < lvl_low and closes[j] > lvl_low:
                return FractalBreakout(
                    kind="FALSE_BREAK",
                    side="LONG",
                    level=float(lvl_low),
                    level_kind="LOW",
                    bar_index=j,
                    bars_ago=n - 1 - j,
                    timeframe=timeframe,
                )
            if (
                lvl_high is not None
                and previous_close <= lvl_high
                and closes[j] > lvl_high
            ):
                return FractalBreakout(
                    kind="TRUE_BREAK",
                    side="LONG",
                    level=float(lvl_high),
                    level_kind="HIGH",
                    bar_index=j,
                    bars_ago=n - 1 - j,
                    timeframe=timeframe,
                )
            if (
                lvl_low is not None
                and previous_close >= lvl_low
                and closes[j] < lvl_low
            ):
                return FractalBreakout(
                    kind="TRUE_BREAK",
                    side="SHORT",
                    level=float(lvl_low),
                    level_kind="LOW",
                    bar_index=j,
                    bars_ago=n - 1 - j,
                    timeframe=timeframe,
                )
        return None

    # ---------------------- H1 PREMIUM / DISCOUNT ----------------------

    def _calc_hourly_range(self) -> Optional[HourlyRange]:
        df = self.df_1h
        if df is None or df.empty:
            return None
        last = df.iloc[-1]
        high = float(last["high"])
        low = float(last["low"])
        close = float(last["close"])

        if high <= low:
            return None
        # Localize the current price with M15 while retaining the latest
        # completed H1 candle as the premium/discount reference range.
        if self.df_15m is not None and not self.df_15m.empty:
            close = float(self.df_15m["close"].iloc[-1])
        mid = (high + low) / 2.0
        if close > mid:
            position = "PREMIUM"
        elif close < mid:
            position = "DISCOUNT"
        else:
            position = "EQ"

        return HourlyRange(
            high=high,
            low=low,
            close=close,
            position=position,
        )

    def to_payload(self) -> dict:
        return {
            "hourly_range": self.hourly_range.to_dict() if self.hourly_range else None,
            "fractals": self.fractals.to_dict() if self.fractals else None,
            "order_blocks": [ob.to_dict() for ob in self.order_blocks],
            "rejection_blocks": [rb.to_dict() for rb in self.rejection_blocks],
            "false_breakout_4h": (
                self.false_breakout_4h.to_dict() if self.false_breakout_4h else None
            ),
            "true_breakout_15m": (
                self.true_breakout_15m.to_dict() if self.true_breakout_15m else None
            ),
        }

    @property
    def ready(self) -> bool:
        return (
            self.df_1h is not None
            and not self.df_1h.empty
            and self.df_4h is not None
            and not self.df_4h.empty
            and self.df_15m is not None
            and not self.df_15m.empty
        )
