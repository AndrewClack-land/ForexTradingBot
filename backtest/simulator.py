"""Deterministic setup-level simulator for the live 3-leg TP policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional, Sequence, Tuple

import pandas as pd


Side = Literal["LONG", "SHORT"]
IntrabarPolicy = Literal["stop-first", "tp-first"]

DEFAULT_WEIGHTS: Tuple[float, float, float] = (0.50, 0.30, 0.20)


@dataclass(frozen=True)
class LegOutcome:
    tp_index: int
    weight: float
    exit_reason: Literal["TP", "STOP", "BE", "OPEN"]
    exit_price: Optional[float]
    r_multiple: float
    exit_time: Optional[pd.Timestamp]


@dataclass(frozen=True)
class SetupOutcome:
    side: Side
    entry: float
    initial_stop: float
    status: Literal["CLOSED", "OPEN"]
    net_r: float
    legs: Tuple[LegOutcome, ...]
    tp_hits: Tuple[int, ...]
    moved_to_be: bool
    ambiguous_bars: int
    bars_processed: int
    exit_time: Optional[pd.Timestamp]

    @property
    def remaining_weight(self) -> float:
        return float(sum(leg.weight for leg in self.legs if leg.exit_reason == "OPEN"))


def _timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _validate_inputs(
    side: str,
    entry: float,
    stop: float,
    tp_prices: Sequence[float],
    weights: Sequence[float],
    intrabar_policy: str,
) -> Tuple[Side, Tuple[float, float, float], Tuple[float, float, float]]:
    normalized_side = str(side).upper()
    if normalized_side not in {"LONG", "SHORT"}:
        raise ValueError("side must be LONG or SHORT")
    if intrabar_policy not in {"stop-first", "tp-first"}:
        raise ValueError("intrabar_policy must be 'stop-first' or 'tp-first'")
    if entry <= 0 or stop <= 0 or entry == stop:
        raise ValueError("entry and stop must be positive and distinct")
    if len(tp_prices) != 3 or len(weights) != 3:
        raise ValueError("the split simulator requires exactly 3 TP prices and 3 weights")

    tps = tuple(float(value) for value in tp_prices)
    normalized_weights = tuple(float(value) for value in weights)
    if any(value <= 0 for value in normalized_weights):
        raise ValueError("all leg weights must be positive")
    if abs(sum(normalized_weights) - 1.0) > 1e-9:
        raise ValueError("leg weights must sum to 1.0")

    if normalized_side == "LONG":
        if not stop < entry or not all(entry < value for value in tps):
            raise ValueError("LONG requires stop < entry < every TP")
        if list(tps) != sorted(tps):
            raise ValueError("LONG TP prices must be nearest-first ascending")
    else:
        if not stop > entry or not all(entry > value for value in tps):
            raise ValueError("SHORT requires stop > entry > every TP")
        if list(tps) != sorted(tps, reverse=True):
            raise ValueError("SHORT TP prices must be nearest-first descending")

    return normalized_side, tps, normalized_weights  # type: ignore[return-value]


def _validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    if bars is None or bars.empty:
        raise ValueError("bars must contain at least one OHLC candle")
    if not required.issubset(bars.columns):
        raise ValueError(f"bars are missing columns: {sorted(required - set(bars.columns))}")
    out = bars.copy().sort_index(kind="stable")
    if out.index.has_duplicates:
        raise ValueError("bar timestamps must be unique")
    for column in required:
        out[column] = pd.to_numeric(out[column], errors="raise")
    invalid = (
        (out["high"] < out["low"])
        | (out["high"] < out[["open", "close"]].max(axis=1))
        | (out["low"] > out[["open", "close"]].min(axis=1))
    )
    if invalid.any():
        raise ValueError("bars contain impossible OHLC candles")
    return out


def simulate_split_outcome(
    *,
    side: Side,
    entry: float,
    stop: float,
    tp_prices: Sequence[float],
    bars: pd.DataFrame,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    intrabar_policy: IntrabarPolicy = "stop-first",
) -> SetupOutcome:
    """Simulate one already-filled split setup over chronological OHLC bars.

    Intrabar ordering is unknowable from OHLC. ``stop-first`` (the default)
    visits the adverse extreme before the favorable extreme; ``tp-first`` does
    the reverse.  A bar is counted as ambiguous whenever reversing that order
    can change leg outcomes, including a TP1 bar whose range also touches the
    newly introduced break-even stop.

    Results are expressed in setup R where the full initial stop is ``-1R``.
    Costs/slippage are intentionally outside this signal-quality simulator.
    """

    normalized_side, tps, normalized_weights = _validate_inputs(
        side,
        float(entry),
        float(stop),
        tp_prices,
        weights,
        intrabar_policy,
    )
    candles = _validate_bars(bars)
    entry = float(entry)
    initial_stop = float(stop)
    risk = abs(entry - initial_stop)

    open_legs = {index for index in range(3)}
    results: dict[int, LegOutcome] = {}
    current_stop = initial_stop
    moved_to_be = False
    ambiguous_bars = 0
    bars_processed = 0
    final_time: Optional[pd.Timestamp] = None

    def stop_touched(row: pd.Series, level: float) -> bool:
        if normalized_side == "LONG":
            return float(row["low"]) <= level
        return float(row["high"]) >= level

    def reached_tps(row: pd.Series) -> list[int]:
        if normalized_side == "LONG":
            return [idx for idx in sorted(open_legs) if float(row["high"]) >= tps[idx]]
        return [idx for idx in sorted(open_legs) if float(row["low"]) <= tps[idx]]

    def leg_r(exit_price: float, weight: float) -> float:
        direction_pnl = (
            exit_price - entry
            if normalized_side == "LONG"
            else entry - exit_price
        )
        return float(direction_pnl / risk * weight)

    def close_at_stop(timestamp: pd.Timestamp) -> None:
        nonlocal final_time
        reason: Literal["STOP", "BE"] = "BE" if moved_to_be else "STOP"
        for idx in sorted(open_legs):
            weight = normalized_weights[idx]
            results[idx] = LegOutcome(
                tp_index=idx + 1,
                weight=weight,
                exit_reason=reason,
                exit_price=current_stop,
                r_multiple=leg_r(current_stop, weight),
                exit_time=timestamp,
            )
        open_legs.clear()
        final_time = timestamp

    def close_targets(indices: Iterable[int], timestamp: pd.Timestamp) -> bool:
        nonlocal moved_to_be, current_stop, final_time
        hit_tp1 = False
        for idx in sorted(indices):
            if idx not in open_legs:
                continue
            weight = normalized_weights[idx]
            results[idx] = LegOutcome(
                tp_index=idx + 1,
                weight=weight,
                exit_reason="TP",
                exit_price=tps[idx],
                r_multiple=leg_r(tps[idx], weight),
                exit_time=timestamp,
            )
            open_legs.remove(idx)
            hit_tp1 = hit_tp1 or idx == 0
        if hit_tp1 and open_legs:
            moved_to_be = True
            current_stop = entry
        if not open_legs:
            final_time = timestamp
        return hit_tp1

    for raw_timestamp, row in candles.iterrows():
        if not open_legs:
            break
        timestamp = _timestamp(raw_timestamp)
        bars_processed += 1
        touched_stop_before = stop_touched(row, current_stop)
        target_indices = reached_tps(row)

        # TP1 can create a BE stop during this candle. If the same range also
        # contains entry, opposite intrabar paths produce different lifecycle.
        creates_be_collision = (
            not moved_to_be
            and 0 in target_indices
            and any(idx not in target_indices for idx in open_legs)
            and stop_touched(row, entry)
        )
        if (touched_stop_before and target_indices) or creates_be_collision:
            ambiguous_bars += 1

        if intrabar_policy == "stop-first":
            if touched_stop_before:
                close_at_stop(timestamp)
                continue
            close_targets(target_indices, timestamp)
            # The adverse extreme was visited before TP1 in this path, so a BE
            # stop introduced afterward cannot trigger retroactively.
        else:
            close_targets(target_indices, timestamp)
            if open_legs and stop_touched(row, current_stop):
                close_at_stop(timestamp)

    legs: list[LegOutcome] = []
    for idx in range(3):
        if idx in results:
            legs.append(results[idx])
        else:
            legs.append(
                LegOutcome(
                    tp_index=idx + 1,
                    weight=normalized_weights[idx],
                    exit_reason="OPEN",
                    exit_price=None,
                    r_multiple=0.0,
                    exit_time=None,
                )
            )

    tp_hits = tuple(leg.tp_index for leg in legs if leg.exit_reason == "TP")
    status: Literal["CLOSED", "OPEN"] = "CLOSED" if not open_legs else "OPEN"
    return SetupOutcome(
        side=normalized_side,
        entry=entry,
        initial_stop=initial_stop,
        status=status,
        net_r=float(sum(leg.r_multiple for leg in legs)),
        legs=tuple(legs),
        tp_hits=tp_hits,
        moved_to_be=moved_to_be,
        ambiguous_bars=ambiguous_bars,
        bars_processed=bars_processed,
        exit_time=final_time,
    )
