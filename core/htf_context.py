from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Literal, Mapping, Optional

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
    # Versioned/causal lifecycle metadata. Defaults preserve every legacy
    # constructor while allowing a detector to expose when a block became
    # knowable and which later bar produced its first actionable retest.
    detector_version: str = "legacy"
    available_idx: Optional[int] = None
    pivot_time: Optional[str] = None
    available_time: Optional[str] = None
    atr_at_pivot: Optional[float] = None
    separation_price: Optional[float] = None
    separation_atr: Optional[float] = None
    retested: bool = False
    retest_idx: Optional[int] = None
    broken_idx: Optional[int] = None
    # A bootstrapped rolling window cannot prove whether an older block was
    # touched or broken before the first observation. Such blocks remain in
    # state solely as accepted-pivot separation history and are never eligible
    # for a vote or entry.
    baseline_unverified: bool = False
    retest_time: Optional[str] = None
    broken_time: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "type": "RB",
            "side": self.side,
            "zone_high": self.zone_high,
            "zone_low": self.zone_low,
            "valid": self.valid and not self.broken,
            "detector_version": self.detector_version,
            "available_idx": self.available_idx,
            "pivot_time": self.pivot_time,
            "available_time": self.available_time,
            "atr_at_pivot": self.atr_at_pivot,
            "separation_price": self.separation_price,
            "separation_atr": self.separation_atr,
            "retested": self.retested,
            "retest_idx": self.retest_idx,
            "broken_idx": self.broken_idx,
            "baseline_unverified": self.baseline_unverified,
            "retest_time": self.retest_time,
            "broken_time": self.broken_time,
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

        def body_top(open_price: float, close_price: float) -> float:
            return max(open_price, close_price)

        def body_bottom(open_price: float, close_price: float) -> float:
            return min(open_price, close_price)

        def body_size(open_price: float, close_price: float) -> float:
            return abs(open_price - close_price)

        def upper_wick(
            high_price: float,
            open_price: float,
            close_price: float,
        ) -> float:
            return high_price - body_top(open_price, close_price)

        def lower_wick(
            low_price: float,
            open_price: float,
            close_price: float,
        ) -> float:
            return body_bottom(open_price, close_price) - low_price

        for i in range(self.pivot_left + 2, n):
            i0 = i
            i1 = i - 1
            i2 = i - 2

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


class _PineRejectionBlockDetectionCore:
    """Causal port of the Rejection Blocks ICT indicator for closed H1 bars.

    A pivot at p is not observable until the close of p + 3. The registration
    bar may therefore invalidate a newly known block, but can never count as
    its retest. Retests begin strictly on the following bar.
    """

    VERSION = "pine-v1-causal"
    SWING_STRENGTH = 3
    WICK_TO_BODY_RATIO = 2.0
    ATR_LENGTH = 14
    ATR_SEPARATION_MULTIPLE = 0.5

    def __init__(self, *, min_tick: float = 1e-9) -> None:
        min_tick = float(min_tick)
        if not np.isfinite(min_tick) or min_tick <= 0.0:
            raise ValueError("min_tick must be a finite positive number")
        self.min_tick = min_tick

    @staticmethod
    def _wilder_atr(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = ATR_LENGTH,
    ) -> np.ndarray:
        """Return TradingView-style ATR (true range smoothed with Wilder RMA)."""

        n = len(close)
        atr = np.full(n, np.nan, dtype=float)
        if n == 0 or period <= 0:
            return atr

        true_range = np.empty(n, dtype=float)
        true_range[0] = high[0] - low[0]
        if n > 1:
            true_range[1:] = np.maximum(
                high[1:] - low[1:],
                np.maximum(
                    np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:] - close[:-1]),
                ),
            )

        if n < period or not np.all(np.isfinite(true_range[:period])):
            return atr

        atr[period - 1] = float(np.mean(true_range[:period]))
        for i in range(period, n):
            if not np.isfinite(true_range[i]):
                atr[i] = np.nan
                continue
            previous = atr[i - 1]
            if not np.isfinite(previous):
                continue
            atr[i] = ((previous * (period - 1)) + true_range[i]) / period
        return atr

    @staticmethod
    def _is_strict_pivot(
        values: np.ndarray,
        pivot_idx: int,
        strength: int,
        *,
        high: bool,
    ) -> bool:
        window = values[pivot_idx - strength : pivot_idx + strength + 1]
        if len(window) != strength * 2 + 1 or not np.all(np.isfinite(window)):
            return False
        pivot = values[pivot_idx]
        peers = np.concatenate((window[:strength], window[strength + 1 :]))
        if high:
            return bool(np.all(pivot > peers))
        return bool(np.all(pivot < peers))

    @staticmethod
    def _breaks_block(rb: RejectionBlock, close: float) -> bool:
        if rb.side == "LONG":
            return bool(close < rb.zone_low)
        return bool(close > rb.zone_high)

    @staticmethod
    def _touches_block(
        rb: RejectionBlock,
        *,
        high: float,
        low: float,
        close: float,
    ) -> bool:
        if rb.side == "LONG":
            return bool(
                low <= rb.zone_high
                and high >= rb.zone_low
                and close >= rb.zone_low
            )
        return bool(
            high >= rb.zone_low
            and low <= rb.zone_high
            and close <= rb.zone_high
        )

    def build(self, df: pd.DataFrame) -> List[RejectionBlock]:
        strength = self.SWING_STRENGTH
        if df is None or df.empty or len(df) < strength * 2 + 1:
            return []

        open_ = df["open"].astype(float).to_numpy()
        high = df["high"].astype(float).to_numpy()
        low = df["low"].astype(float).to_numpy()
        close = df["close"].astype(float).to_numpy()
        atr = self._wilder_atr(high, low, close, self.ATR_LENGTH)
        n = len(df)

        accepted: List[RejectionBlock] = []
        last_accepted_price: dict[str, Optional[float]] = {
            "LONG": None,
            "SHORT": None,
        }

        for pivot_idx in range(strength, n - strength):
            pivot_ohlc = (
                open_[pivot_idx],
                high[pivot_idx],
                low[pivot_idx],
                close[pivot_idx],
            )
            if not np.all(np.isfinite(pivot_ohlc)):
                continue

            body_top = max(open_[pivot_idx], close[pivot_idx])
            body_bottom = min(open_[pivot_idx], close[pivot_idx])
            body = abs(close[pivot_idx] - open_[pivot_idx])
            denominator = max(body, self.min_tick)
            upper_wick = high[pivot_idx] - body_top
            lower_wick = body_bottom - low[pivot_idx]
            available_idx = pivot_idx + strength
            atr_value = (
                float(atr[pivot_idx])
                if np.isfinite(atr[pivot_idx])
                else None
            )

            candidates = (
                (
                    "LONG",
                    self._is_strict_pivot(
                        low,
                        pivot_idx,
                        strength,
                        high=False,
                    ),
                    lower_wick,
                    upper_wick,
                    float(low[pivot_idx]),
                    float(body_bottom),
                    float(low[pivot_idx]),
                ),
                (
                    "SHORT",
                    self._is_strict_pivot(
                        high,
                        pivot_idx,
                        strength,
                        high=True,
                    ),
                    upper_wick,
                    lower_wick,
                    float(body_top),
                    float(high[pivot_idx]),
                    float(high[pivot_idx]),
                ),
            )

            for (
                side,
                is_pivot,
                main_wick,
                opposite_wick,
                zone_low,
                zone_high,
                pivot_price,
            ) in candidates:
                if not is_pivot:
                    continue
                if main_wick < self.WICK_TO_BODY_RATIO * denominator:
                    continue
                if main_wick < opposite_wick:
                    continue

                prior_price = last_accepted_price[side]
                separation_price = (
                    abs(pivot_price - prior_price)
                    if prior_price is not None
                    else None
                )
                if prior_price is not None:
                    if atr_value is None:
                        continue
                    if (
                        separation_price
                        < self.ATR_SEPARATION_MULTIPLE * atr_value
                    ):
                        continue

                separation_atr = (
                    separation_price / atr_value
                    if separation_price is not None
                    and atr_value is not None
                    and atr_value > 0.0
                    else None
                )
                last_accepted_price[side] = pivot_price
                accepted.append(
                    RejectionBlock(
                        side=side,
                        zone_high=zone_high,
                        zone_low=zone_low,
                        created_idx=pivot_idx,
                        midline=float((zone_low + zone_high) / 2.0),
                        wick_ratio=float(main_wick / denominator),
                        intrusion_pct=0.0,
                        detector_version=self.VERSION,
                        available_idx=available_idx,
                        pivot_time=str(df.index[pivot_idx]),
                        available_time=str(df.index[available_idx]),
                        atr_at_pivot=atr_value,
                        separation_price=separation_price,
                        separation_atr=separation_atr,
                    )
                )

        # Reconstruct the persistent lifecycle causally. A close break is
        # evaluated before a retest. The registration bar may break a block,
        # but is deliberately excluded from retest eligibility.
        for rb in accepted:
            if rb.available_idx is None:
                continue
            for bar_idx in range(rb.available_idx, n):
                if self._breaks_block(rb, float(close[bar_idx])):
                    rb.broken = True
                    rb.broken_idx = bar_idx
                    break
                if (
                    not rb.retested
                    and bar_idx > rb.available_idx
                    and self._touches_block(
                        rb,
                        high=float(high[bar_idx]),
                        low=float(low[bar_idx]),
                        close=float(close[bar_idx]),
                    )
                ):
                    rb.retested = True
                    rb.retest_idx = bar_idx

        return accepted


class PineRejectionBlockTracker(_PineRejectionBlockDetectionCore):
    """Stateful causal Pine RB tracker for rolling closed-bar windows.

    The first observed H1 window is a fail-closed bootstrap: accepted pivots
    are retained as separation history, but their pre-observation lifecycle is
    unknowable, so they cannot vote or trigger. Only blocks confirmed after
    the bootstrap watermark become eligible.
    """

    STATE_SCHEMA = "pine-rb-state/v1"

    def __init__(
        self,
        *,
        min_tick: float = 1e-9,
        symbol: str = "",
    ) -> None:
        super().__init__(min_tick=min_tick)
        self.symbol = str(symbol or "").strip().upper()
        self._blocks: dict[str, RejectionBlock] = {}
        self._processed_events: set[str] = set()
        self._bootstrap_watermark: Optional[str] = None
        self._h1_observed_through: Optional[str] = None
        self._m1_observed_through: Optional[str] = None
        self._dirty = False

    @property
    def dirty(self) -> bool:
        return bool(self._dirty)

    def mark_clean(self) -> None:
        self._dirty = False

    @staticmethod
    def _utc_timestamp(value: Any) -> Optional[pd.Timestamp]:
        if value is None:
            return None
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(stamp):
            return None
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        else:
            stamp = stamp.tz_convert("UTC")
        return stamp

    @classmethod
    def _iso_time(cls, value: Any) -> Optional[str]:
        stamp = cls._utc_timestamp(value)
        return stamp.isoformat() if stamp is not None else None

    @classmethod
    def _h1_close_time(cls, value: Any) -> Optional[pd.Timestamp]:
        stamp = cls._utc_timestamp(value)
        return stamp + pd.Timedelta(hours=1) if stamp is not None else None

    @classmethod
    def _m1_close_time(cls, value: Any) -> Optional[pd.Timestamp]:
        stamp = cls._utc_timestamp(value)
        return stamp + pd.Timedelta(minutes=1) if stamp is not None else None

    @staticmethod
    def _event_key(side: str, pivot_time: str) -> str:
        return f"{str(side).upper()}|{pivot_time}"

    @staticmethod
    def _pivot_price(block: RejectionBlock) -> float:
        return (
            float(block.zone_low)
            if block.side == "LONG"
            else float(block.zone_high)
        )

    def _latest_prior(
        self,
        side: str,
        pivot_time: pd.Timestamp,
    ) -> Optional[RejectionBlock]:
        eligible: list[tuple[pd.Timestamp, RejectionBlock]] = []
        for block in self._blocks.values():
            if block.side != side:
                continue
            stamp = self._utc_timestamp(block.pivot_time)
            if stamp is not None and stamp < pivot_time:
                eligible.append((stamp, block))
        if not eligible:
            return None
        return max(eligible, key=lambda item: item[0])[1]

    @staticmethod
    def _state_float(
        row: Mapping[str, Any],
        name: str,
        *,
        optional: bool = False,
    ) -> Optional[float]:
        value = row.get(name)
        if value is None and optional:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid RB state field {name}") from exc
        if not np.isfinite(result):
            raise ValueError(f"non-finite RB state field {name}")
        return result

    @staticmethod
    def _state_int(
        row: Mapping[str, Any],
        name: str,
        *,
        optional: bool = False,
    ) -> Optional[int]:
        value = row.get(name)
        if value is None and optional:
            return None
        if isinstance(value, bool):
            raise ValueError(f"invalid RB state field {name}")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid RB state field {name}") from exc

    @staticmethod
    def _state_bool(row: Mapping[str, Any], name: str) -> bool:
        value = row.get(name)
        if not isinstance(value, bool):
            raise ValueError(f"invalid RB state field {name}")
        return value

    def export_state(self) -> dict[str, Any]:
        blocks: list[dict[str, Any]] = []
        for key, block in sorted(self._blocks.items()):
            blocks.append(
                {
                    "event_key": key,
                    "side": str(block.side),
                    "zone_high": float(block.zone_high),
                    "zone_low": float(block.zone_low),
                    "created_idx": int(block.created_idx),
                    "midline": float(block.midline),
                    "wick_ratio": float(block.wick_ratio),
                    "intrusion_pct": float(block.intrusion_pct),
                    "broken": bool(block.broken),
                    "valid": bool(block.valid),
                    "detector_version": str(block.detector_version),
                    "available_idx": (
                        int(block.available_idx)
                        if block.available_idx is not None
                        else None
                    ),
                    "pivot_time": block.pivot_time,
                    "available_time": block.available_time,
                    "atr_at_pivot": (
                        float(block.atr_at_pivot)
                        if block.atr_at_pivot is not None
                        else None
                    ),
                    "separation_price": (
                        float(block.separation_price)
                        if block.separation_price is not None
                        else None
                    ),
                    "separation_atr": (
                        float(block.separation_atr)
                        if block.separation_atr is not None
                        else None
                    ),
                    "retested": bool(block.retested),
                    "retest_idx": (
                        int(block.retest_idx)
                        if block.retest_idx is not None
                        else None
                    ),
                    "broken_idx": (
                        int(block.broken_idx)
                        if block.broken_idx is not None
                        else None
                    ),
                    "baseline_unverified": bool(
                        block.baseline_unverified
                    ),
                    "retest_time": block.retest_time,
                    "broken_time": block.broken_time,
                }
            )
        return {
            "schema": self.STATE_SCHEMA,
            "detector_version": self.VERSION,
            "symbol": self.symbol,
            "min_tick": float(self.min_tick),
            "bootstrap_watermark": self._bootstrap_watermark,
            "h1_observed_through": self._h1_observed_through,
            "m1_observed_through": self._m1_observed_through,
            "processed_events": sorted(self._processed_events),
            "blocks": blocks,
        }

    def import_state(self, payload: Mapping[str, Any]) -> None:
        """Atomically hydrate validated JSON state, leaving self unchanged on error."""

        if not isinstance(payload, Mapping):
            raise ValueError("RB state must be a mapping")
        if payload.get("schema") != self.STATE_SCHEMA:
            raise ValueError("unsupported RB state schema")
        if payload.get("detector_version") != self.VERSION:
            raise ValueError("RB detector version mismatch")
        state_symbol = str(payload.get("symbol") or "").strip().upper()
        if state_symbol != self.symbol:
            raise ValueError("RB state symbol mismatch")
        state_min_tick = self._state_float(payload, "min_tick")
        if state_min_tick is None or not np.isclose(
            state_min_tick,
            self.min_tick,
            rtol=0.0,
            atol=max(1e-15, self.min_tick * 1e-12),
        ):
            raise ValueError("RB state min_tick mismatch")

        watermark_names = (
            "bootstrap_watermark",
            "h1_observed_through",
            "m1_observed_through",
        )
        watermarks: dict[str, Optional[str]] = {}
        for name in watermark_names:
            raw = payload.get(name)
            stamp = self._utc_timestamp(raw)
            if raw is not None and stamp is None:
                raise ValueError(f"invalid RB state watermark {name}")
            watermarks[name] = stamp.isoformat() if stamp is not None else None
        bootstrap_ts = self._utc_timestamp(
            watermarks["bootstrap_watermark"]
        )
        h1_ts = self._utc_timestamp(watermarks["h1_observed_through"])
        if (
            bootstrap_ts is not None
            and h1_ts is not None
            and h1_ts < bootstrap_ts
        ):
            raise ValueError("RB H1 watermark precedes bootstrap")

        processed_raw = payload.get("processed_events")
        if not isinstance(processed_raw, list) or not all(
            isinstance(item, str) and item
            for item in processed_raw
        ):
            raise ValueError("RB processed_events must be a string list")
        processed = set(processed_raw)

        rows = payload.get("blocks")
        if not isinstance(rows, list):
            raise ValueError("RB blocks must be a list")
        hydrated: dict[str, RejectionBlock] = {}
        for raw_row in rows:
            if not isinstance(raw_row, Mapping):
                raise ValueError("RB block state must be a mapping")
            side = str(raw_row.get("side") or "").upper()
            if side not in {"LONG", "SHORT"}:
                raise ValueError("invalid RB state side")
            if raw_row.get("detector_version") != self.VERSION:
                raise ValueError("invalid RB block detector version")
            pivot_time = self._iso_time(raw_row.get("pivot_time"))
            available_time = self._iso_time(
                raw_row.get("available_time")
            )
            if pivot_time is None or available_time is None:
                raise ValueError("RB state requires causal timestamps")
            pivot_ts = self._utc_timestamp(pivot_time)
            available_ts = self._utc_timestamp(available_time)
            if (
                pivot_ts is None
                or available_ts is None
                or available_ts <= pivot_ts
            ):
                raise ValueError("invalid RB causal timestamp ordering")
            if h1_ts is not None and available_ts > h1_ts:
                raise ValueError("RB block is newer than H1 watermark")
            expected_key = self._event_key(side, pivot_time)
            event_key = str(raw_row.get("event_key") or "")
            if event_key != expected_key or event_key in hydrated:
                raise ValueError("invalid or duplicate RB event key")

            zone_high = self._state_float(raw_row, "zone_high")
            zone_low = self._state_float(raw_row, "zone_low")
            if (
                zone_high is None
                or zone_low is None
                or zone_low <= 0.0
                or zone_high <= zone_low
            ):
                raise ValueError("invalid RB zone")
            baseline = self._state_bool(
                raw_row,
                "baseline_unverified",
            )
            valid = self._state_bool(raw_row, "valid")
            retested = self._state_bool(raw_row, "retested")
            if baseline and (valid or not retested):
                raise ValueError(
                    "baseline RB must be invalid and consumed"
                )
            block = RejectionBlock(
                side=side,
                zone_high=zone_high,
                zone_low=zone_low,
                created_idx=int(
                    self._state_int(raw_row, "created_idx")
                ),
                midline=float(self._state_float(raw_row, "midline")),
                wick_ratio=float(
                    self._state_float(raw_row, "wick_ratio")
                ),
                intrusion_pct=float(
                    self._state_float(raw_row, "intrusion_pct")
                ),
                broken=self._state_bool(raw_row, "broken"),
                valid=valid,
                detector_version=self.VERSION,
                available_idx=self._state_int(
                    raw_row,
                    "available_idx",
                    optional=True,
                ),
                pivot_time=pivot_time,
                available_time=available_time,
                atr_at_pivot=self._state_float(
                    raw_row,
                    "atr_at_pivot",
                    optional=True,
                ),
                separation_price=self._state_float(
                    raw_row,
                    "separation_price",
                    optional=True,
                ),
                separation_atr=self._state_float(
                    raw_row,
                    "separation_atr",
                    optional=True,
                ),
                retested=retested,
                retest_idx=self._state_int(
                    raw_row,
                    "retest_idx",
                    optional=True,
                ),
                broken_idx=self._state_int(
                    raw_row,
                    "broken_idx",
                    optional=True,
                ),
                baseline_unverified=baseline,
                retest_time=self._iso_time(
                    raw_row.get("retest_time")
                ),
                broken_time=self._iso_time(
                    raw_row.get("broken_time")
                ),
            )
            hydrated[event_key] = block
            processed.add(event_key)

        if hydrated and bootstrap_ts is None:
            raise ValueError("RB state with blocks requires bootstrap watermark")

        self._blocks = hydrated
        self._processed_events = processed
        self._bootstrap_watermark = watermarks[
            "bootstrap_watermark"
        ]
        self._h1_observed_through = watermarks[
            "h1_observed_through"
        ]
        self._m1_observed_through = watermarks[
            "m1_observed_through"
        ]
        self._dirty = False

    def clone(self) -> "PineRejectionBlockTracker":
        clone = type(self)(min_tick=self.min_tick, symbol=self.symbol)
        clone.import_state(self.export_state())
        clone._dirty = self._dirty
        return clone

    def _register_candidates(
        self,
        df: pd.DataFrame,
        *,
        bootstrap_watermark: pd.Timestamp,
    ) -> None:
        strength = self.SWING_STRENGTH
        open_ = df["open"].astype(float).to_numpy()
        high = df["high"].astype(float).to_numpy()
        low = df["low"].astype(float).to_numpy()
        close = df["close"].astype(float).to_numpy()
        atr = self._wilder_atr(high, low, close, self.ATR_LENGTH)
        n = len(df)

        for pivot_idx in range(strength, n - strength):
            pivot_time_ts = self._utc_timestamp(df.index[pivot_idx])
            available_open_ts = self._utc_timestamp(
                df.index[pivot_idx + strength]
            )
            if pivot_time_ts is None or available_open_ts is None:
                continue
            pivot_time = pivot_time_ts.isoformat()
            available_time_ts = available_open_ts + pd.Timedelta(hours=1)
            pivot_ohlc = (
                open_[pivot_idx],
                high[pivot_idx],
                low[pivot_idx],
                close[pivot_idx],
            )
            if not np.all(np.isfinite(pivot_ohlc)):
                continue

            body_top = max(open_[pivot_idx], close[pivot_idx])
            body_bottom = min(open_[pivot_idx], close[pivot_idx])
            body = abs(close[pivot_idx] - open_[pivot_idx])
            denominator = max(body, self.min_tick)
            upper_wick = high[pivot_idx] - body_top
            lower_wick = body_bottom - low[pivot_idx]
            atr_value = (
                float(atr[pivot_idx])
                if np.isfinite(atr[pivot_idx])
                else None
            )
            candidates = (
                (
                    "LONG",
                    self._is_strict_pivot(
                        low,
                        pivot_idx,
                        strength,
                        high=False,
                    ),
                    lower_wick,
                    upper_wick,
                    float(low[pivot_idx]),
                    float(body_bottom),
                ),
                (
                    "SHORT",
                    self._is_strict_pivot(
                        high,
                        pivot_idx,
                        strength,
                        high=True,
                    ),
                    upper_wick,
                    lower_wick,
                    float(body_top),
                    float(high[pivot_idx]),
                ),
            )
            for (
                side,
                is_pivot,
                main_wick,
                opposite_wick,
                zone_low,
                zone_high,
            ) in candidates:
                if not is_pivot:
                    continue
                event_key = self._event_key(side, pivot_time)
                if event_key in self._processed_events:
                    continue
                self._processed_events.add(event_key)
                self._dirty = True
                if main_wick < self.WICK_TO_BODY_RATIO * denominator:
                    continue
                if main_wick < opposite_wick:
                    continue

                prior = self._latest_prior(side, pivot_time_ts)
                prior_price = (
                    self._pivot_price(prior)
                    if prior is not None
                    else None
                )
                pivot_price = zone_low if side == "LONG" else zone_high
                separation_price = (
                    abs(pivot_price - prior_price)
                    if prior_price is not None
                    else None
                )
                if prior_price is not None:
                    if atr_value is None:
                        continue
                    if (
                        separation_price
                        < self.ATR_SEPARATION_MULTIPLE * atr_value
                    ):
                        continue
                separation_atr = (
                    separation_price / atr_value
                    if separation_price is not None
                    and atr_value is not None
                    and atr_value > 0.0
                    else None
                )
                baseline = available_time_ts <= bootstrap_watermark
                self._blocks[event_key] = RejectionBlock(
                    side=side,
                    zone_high=zone_high,
                    zone_low=zone_low,
                    created_idx=pivot_idx,
                    midline=float((zone_low + zone_high) / 2.0),
                    wick_ratio=float(main_wick / denominator),
                    intrusion_pct=0.0,
                    valid=not baseline,
                    detector_version=self.VERSION,
                    available_idx=pivot_idx + strength,
                    pivot_time=pivot_time,
                    available_time=available_time_ts.isoformat(),
                    atr_at_pivot=atr_value,
                    separation_price=separation_price,
                    separation_atr=separation_atr,
                    retested=baseline,
                    baseline_unverified=baseline,
                )

    def _new_h1_rows(
        self,
        df: pd.DataFrame,
        prior_watermark: Optional[pd.Timestamp],
    ) -> list[tuple[int, pd.Timestamp, pd.Series]]:
        rows: list[tuple[int, pd.Timestamp, pd.Series]] = []
        for idx, (_, row) in enumerate(df.iterrows()):
            close_time = self._h1_close_time(df.index[idx])
            if close_time is None:
                continue
            if prior_watermark is None or close_time > prior_watermark:
                rows.append((idx, close_time, row))
        return rows

    def _process_h1_breaks(
        self,
        rows: list[tuple[int, pd.Timestamp, pd.Series]],
    ) -> None:
        for bar_idx, close_time, row in rows:
            close = float(row["close"])
            if not np.isfinite(close):
                continue
            for block in self._blocks.values():
                if (
                    not block.valid
                    or block.broken
                    or block.baseline_unverified
                ):
                    continue
                available = self._utc_timestamp(block.available_time)
                if available is None or close_time < available:
                    continue
                if self._breaks_block(block, close):
                    block.broken = True
                    block.broken_idx = bar_idx
                    block.broken_time = close_time.isoformat()
                    self._dirty = True

    def _process_m1_touches(
        self,
        df_1m: Optional[pd.DataFrame],
        prior_watermark: Optional[pd.Timestamp],
    ) -> Optional[pd.Timestamp]:
        if df_1m is None or df_1m.empty:
            return prior_watermark
        newest = prior_watermark
        for _, row in df_1m.sort_index(kind="stable").iterrows():
            bar_open = self._utc_timestamp(row.name)
            bar_close = self._m1_close_time(row.name)
            if bar_open is None or bar_close is None:
                continue
            if prior_watermark is not None and bar_close <= prior_watermark:
                continue
            newest = bar_close if newest is None else max(newest, bar_close)
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            if not np.all(np.isfinite((high, low, close))):
                continue
            for block in self._blocks.values():
                if (
                    not block.valid
                    or block.broken
                    or block.retested
                    or block.baseline_unverified
                ):
                    continue
                available = self._utc_timestamp(block.available_time)
                if available is None or bar_open < available:
                    continue
                if self._touches_block(
                    block,
                    high=high,
                    low=low,
                    close=close,
                ):
                    block.retested = True
                    block.retest_idx = None
                    block.retest_time = bar_close.isoformat()
                    self._dirty = True
        return newest

    def _process_h1_retests(
        self,
        rows: list[tuple[int, pd.Timestamp, pd.Series]],
    ) -> None:
        for bar_idx, close_time, row in rows:
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            if not np.all(np.isfinite((high, low, close))):
                continue
            for block in self._blocks.values():
                if (
                    not block.valid
                    or block.broken
                    or block.retested
                    or block.baseline_unverified
                ):
                    continue
                available = self._utc_timestamp(block.available_time)
                # Registration H1 is break-eligible but never retest-eligible.
                if available is None or close_time <= available:
                    continue
                if self._touches_block(
                    block,
                    high=high,
                    low=low,
                    close=close,
                ):
                    block.retested = True
                    block.retest_idx = bar_idx
                    block.retest_time = close_time.isoformat()
                    self._dirty = True

    def _advance_chronology(
        self,
        new_h1: list[tuple[int, pd.Timestamp, pd.Series]],
        df_1m: Optional[pd.DataFrame],
        prior_m1: Optional[pd.Timestamp],
    ) -> Optional[pd.Timestamp]:
        """Replay a batch of new bars in real chronological order.

        A catch-up batch (restart, feed gap, weekend rollover) delivers several
        closed H1 bars in one ``build``.  Sweeping breaks across the whole batch
        before looking at any retest let a break in a *later* bar cancel a touch
        that had already happened in an *earlier* one, so the same candles
        produced a different block state depending only on how many arrived at
        once.  Walking bar by bar removes that path dependence.

        Inside one H1 bar the ordering is unchanged and deliberate: a confirmed
        distal close wins over that same bar's OHLC retest and over the M1
        touches that fall within it.
        """

        m1_sorted: Optional[pd.DataFrame] = None
        if df_1m is not None and not df_1m.empty:
            m1_sorted = df_1m.sort_index(kind="stable")
        total_m1 = 0 if m1_sorted is None else len(m1_sorted)
        m1_pos = 0
        newest_m1 = prior_m1

        for row_entry in new_h1:
            close_time = row_entry[1]
            self._process_h1_breaks((row_entry,))
            if m1_sorted is not None:
                end = m1_pos
                while end < total_m1:
                    m1_close = self._m1_close_time(m1_sorted.index[end])
                    if m1_close is not None and m1_close > close_time:
                        break
                    end += 1
                if end > m1_pos:
                    newest_m1 = self._process_m1_touches(
                        m1_sorted.iloc[m1_pos:end],
                        newest_m1,
                    )
                    m1_pos = end
            self._process_h1_retests((row_entry,))

        # M1 bars of the still-forming H1 candle have no closed H1 bar to be
        # ordered against, so they are applied last.
        if m1_sorted is not None and m1_pos < total_m1:
            newest_m1 = self._process_m1_touches(
                m1_sorted.iloc[m1_pos:],
                newest_m1,
            )
        return newest_m1

    def build(
        self,
        df: pd.DataFrame,
        *,
        df_1m: Optional[pd.DataFrame] = None,
    ) -> List[RejectionBlock]:
        strength = self.SWING_STRENGTH
        if df is None or df.empty:
            return list(self._blocks.values())
        required = {"open", "high", "low", "close"}
        if not required.issubset(df.columns):
            raise ValueError("Pine RB H1 frame lacks OHLC columns")
        observed = self._h1_close_time(df.index[-1])
        if observed is None:
            raise ValueError("Pine RB H1 frame requires datetime index")

        is_bootstrap = self._bootstrap_watermark is None
        if is_bootstrap:
            self._bootstrap_watermark = observed.isoformat()
            self._dirty = True
        bootstrap = self._utc_timestamp(self._bootstrap_watermark)
        if bootstrap is None:
            raise ValueError("Pine RB bootstrap watermark is invalid")

        if len(df) >= strength * 2 + 1:
            self._register_candidates(
                df.sort_index(kind="stable"),
                bootstrap_watermark=bootstrap,
            )

        if is_bootstrap:
            self._h1_observed_through = observed.isoformat()
            newest_m1 = None
            if df_1m is not None and not df_1m.empty:
                newest_m1 = self._m1_close_time(
                    df_1m.sort_index(kind="stable").index[-1]
                )
            self._m1_observed_through = (
                newest_m1.isoformat()
                if newest_m1 is not None
                else observed.isoformat()
            )
            self._dirty = True
        else:
            prior_h1 = self._utc_timestamp(self._h1_observed_through)
            new_h1 = self._new_h1_rows(
                df.sort_index(kind="stable"),
                prior_h1,
            )
            prior_m1 = self._utc_timestamp(self._m1_observed_through)
            if prior_m1 is None:
                prior_m1 = bootstrap
            newest_m1 = self._advance_chronology(new_h1, df_1m, prior_m1)
            if prior_h1 is None or observed > prior_h1:
                self._h1_observed_through = observed.isoformat()
                self._dirty = True
            if newest_m1 is not None and (
                prior_m1 is None or newest_m1 > prior_m1
            ):
                self._m1_observed_through = newest_m1.isoformat()
                self._dirty = True

        return sorted(
            self._blocks.values(),
            key=lambda block: (
                self._utc_timestamp(block.pivot_time)
                or pd.Timestamp.min.tz_localize("UTC"),
                str(block.side),
            ),
        )


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
        rb_detector_version: str = "legacy",
        min_tick: float = 1e-9,
        df_1m: Optional[pd.DataFrame] = None,
        pine_rb_tracker: Optional[PineRejectionBlockTracker] = None,
        rb_lifecycle_enabled: bool = True,
    ) -> None:
        self.df_1h = df_1h
        self.df_4h = df_4h
        self.df_15m = df_15m
        self.df_1m = df_1m
        self.rb_lifecycle_enabled = bool(rb_lifecycle_enabled)

        self.fractal_limit = max(5, int(fractal_limit))
        self.pivot_lookback = max(1, int(pivot_lookback))

        min_tick = float(min_tick)
        if not np.isfinite(min_tick) or min_tick <= 0.0:
            raise ValueError("min_tick must be a finite positive number")
        self.min_tick = min_tick
        detector_version = str(rb_detector_version or "").strip().lower()
        self.rb_detector_version = detector_version

        self.ob_tracker = OrderBlockTracker(swing_lookback=ob_lookback, show_last=ob_show_last)
        if detector_version == PineRejectionBlockTracker.VERSION:
            if pine_rb_tracker is None:
                self.rb_tracker = PineRejectionBlockTracker(
                    min_tick=min_tick
                )
            else:
                if not isinstance(
                    pine_rb_tracker,
                    PineRejectionBlockTracker,
                ):
                    raise ValueError(
                        "pine_rb_tracker must use the Pine causal detector"
                    )
                if not np.isclose(
                    pine_rb_tracker.min_tick,
                    min_tick,
                    rtol=0.0,
                    atol=max(1e-15, min_tick * 1e-12),
                ):
                    raise ValueError(
                        "pine_rb_tracker min_tick does not match context"
                    )
                self.rb_tracker = pine_rb_tracker
        elif detector_version in {"legacy", "legacy-v1"}:
            if pine_rb_tracker is not None:
                raise ValueError(
                    "pine_rb_tracker is invalid for legacy detector"
                )
            self.rb_tracker = RejectionBlockTracker(
                pivot_left=1,
                box_length=rb_box_length,
                use_wick_body_filter=rb_use_wick_filter,
                wick_to_body_ratio=rb_wick_ratio,
                min_intrusion_pct=rb_intrusion_pct,
                body_rule=rb_body_rule,
            )
        else:
            raise ValueError(
                "rb_detector_version must be 'pine-v1-causal' or 'legacy'"
            )

        self.fractals: Optional[FractalPower] = None
        self.hourly_range: Optional[HourlyRange] = None
        self.order_blocks: List[OrderBlock] = []
        self.rejection_blocks: List[RejectionBlock] = []
        self.false_breakout_4h: Optional[FractalBreakout] = None
        self.true_breakout_15m: Optional[FractalBreakout] = None
        self.false_breakout_1h: Optional[FractalBreakout] = None
        self.true_breakout_1h: Optional[FractalBreakout] = None

        if self.df_1h is not None and not self.df_1h.empty:
            self.fractals = self._calc_fractals()
            self.hourly_range = self._calc_hourly_range()
            self.order_blocks = self.ob_tracker.build(self.df_1h)
            if isinstance(
                self.rb_tracker,
                PineRejectionBlockTracker,
            ) and self.rb_lifecycle_enabled:
                self.rejection_blocks = self.rb_tracker.build(
                    self.df_1h,
                    df_1m=self.df_1m,
                )
            elif not isinstance(
                self.rb_tracker,
                PineRejectionBlockTracker,
            ):
                self.rejection_blocks = self.rb_tracker.build(self.df_1h)
            # One scan, routed by kind. ``_calc_latest_fractal_breakout``
            # returns the single most recent interaction of either kind, so
            # the two 1H factors are mutually exclusive by construction --
            # exactly like the 4H and 15M rows, which filter the same single
            # event. They can never both vote on one decision.
            latest_1h_breakout = self._calc_latest_fractal_breakout(
                self.df_1h,
                timeframe="1H",
            )
            if latest_1h_breakout is not None:
                if latest_1h_breakout.kind == "FALSE_BREAK":
                    self.false_breakout_1h = latest_1h_breakout
                elif latest_1h_breakout.kind == "TRUE_BREAK":
                    self.true_breakout_1h = latest_1h_breakout
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
            "false_breakout_1h": (
                self.false_breakout_1h.to_dict() if self.false_breakout_1h else None
            ),
            "true_breakout_1h": (
                self.true_breakout_1h.to_dict() if self.true_breakout_1h else None
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
