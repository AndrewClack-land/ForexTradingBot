"""Causal H1-context to completed-M1 reclaim trigger research.

This module does not alter production signals. It turns a frozen higher-
timeframe entry context into an auditable M1 event and always enters, if at
all, on the next M1 open after the trigger candle has closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Optional

import pandas as pd


H1_M1_TRIGGER_SCHEMA = "h1-context-m1-trigger/v1"


def _hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _timestamp(value: Any, *, name: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return parsed.tz_convert("UTC")


def _price(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


@dataclass(frozen=True)
class M1ReclaimConfig:
    min_penetration_fraction: float = 0.25
    max_entry_extension_r: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_penetration_fraction <= 1.0:
            raise ValueError("min_penetration_fraction must be in [0,1]")
        if self.max_entry_extension_r < 0.0:
            raise ValueError("max_entry_extension_r cannot be negative")


@dataclass(frozen=True)
class H1Context:
    context_id: str
    symbol: str
    side: str
    trigger_kind: str
    decision_time: pd.Timestamp
    expires_at: pd.Timestamp
    entry_min: float
    entry_max: float
    planned_entry: float
    stop: float
    tp_prices: tuple[float, ...]
    signal: Mapping[str, Any]

    @classmethod
    def from_signal(
        cls,
        *,
        symbol: str,
        signal: Mapping[str, Any],
        decision_time: Any,
        expires_at: Any,
    ) -> "H1Context":
        side = str(signal.get("side") or "").strip().upper()
        if side not in {"LONG", "SHORT"}:
            raise ValueError("side must be LONG or SHORT")
        decision = _timestamp(decision_time, name="decision_time")
        expiry = _timestamp(expires_at, name="expires_at")
        if expiry <= decision:
            raise ValueError("expires_at must follow decision_time")
        entry_min = _price(signal.get("entry_min"), name="entry_min")
        entry_max = _price(signal.get("entry_max"), name="entry_max")
        if entry_max <= entry_min:
            raise ValueError("entry range must have positive width")
        planned_entry = _price(
            signal.get("entry_price"),
            name="entry_price",
        )
        stop = _price(signal.get("stop_price"), name="stop_price")
        tps = tuple(
            _price(value, name="tp_price")
            for value in signal.get("tp_prices", ())
        )
        if not tps:
            raise ValueError("tp_prices are required")
        if side == "LONG" and not (
            stop < entry_min <= planned_entry < tps[0]
        ):
            raise ValueError("invalid LONG context geometry")
        if side == "SHORT" and not (
            tps[0] < planned_entry <= entry_max < stop
        ):
            raise ValueError("invalid SHORT context geometry")
        identity = {
            "schema": H1_M1_TRIGGER_SCHEMA,
            "symbol": str(symbol).strip().upper(),
            "side": side,
            "trigger_kind": str(
                signal.get("trigger_kind") or ""
            ).strip().lower(),
            "decision_time": decision.isoformat(),
            "expires_at": expiry.isoformat(),
            "entry_min": entry_min,
            "entry_max": entry_max,
            "planned_entry": planned_entry,
            "stop": stop,
            "tp_prices": list(tps),
        }
        return cls(
            context_id=_hash(identity)[:24],
            symbol=identity["symbol"],
            side=side,
            trigger_kind=identity["trigger_kind"],
            decision_time=decision,
            expires_at=expiry,
            entry_min=entry_min,
            entry_max=entry_max,
            planned_entry=planned_entry,
            stop=stop,
            tp_prices=tps,
            signal=dict(signal),
        )


def _audit(
    context: H1Context,
    status: str,
    **fields: Any,
) -> dict[str, Any]:
    return {
        "schema": H1_M1_TRIGGER_SCHEMA,
        "context_id": context.context_id,
        "symbol": context.symbol,
        "side": context.side,
        "trigger_kind": context.trigger_kind,
        "context_decision_time": context.decision_time.isoformat(),
        "context_expires_at": context.expires_at.isoformat(),
        "status": status,
        **fields,
    }


def detect_m1_reclaim(
    context: H1Context,
    m1: pd.DataFrame,
    *,
    config: M1ReclaimConfig = M1ReclaimConfig(),
) -> dict[str, Any]:
    """Find the first fully post-decision, completed M1 reclaim.

    Timestamps on m1 are bar opens. A bar opening before the H1 decision is
    never used even if it closes afterwards, because part of its path was not
    observable when the context armed.
    """

    required = {"open", "high", "low", "close"}
    if not required.issubset(m1.columns):
        return _audit(context, "DATA_INVALID", reason="missing OHLC")
    frame = m1.sort_index()
    if not frame.index.is_unique:
        return _audit(context, "DATA_INVALID", reason="duplicate M1 index")
    if frame.empty:
        return _audit(context, "NO_TRIGGER")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        return _audit(context, "DATA_INVALID", reason="naive M1 index")
    frame = frame.copy()
    frame.index = index.tz_convert("UTC")
    previous_rows = frame.loc[frame.index < context.decision_time]
    previous_close: Optional[float] = (
        float(previous_rows["close"].iloc[-1])
        if not previous_rows.empty
        else None
    )
    post = frame.loc[
        (frame.index >= context.decision_time)
        & (frame.index + pd.Timedelta(minutes=1) < context.expires_at)
    ]
    if post.empty:
        return _audit(context, "NO_TRIGGER")
    width = context.entry_max - context.entry_min
    risk = abs(context.planned_entry - context.stop)

    for bar_open, row in post.iterrows():
        values = {
            name: float(row[name])
            for name in ("open", "high", "low", "close")
        }
        if not all(math.isfinite(value) for value in values.values()):
            return _audit(
                context,
                "DATA_INVALID",
                reason=f"non-finite M1 bar at {bar_open.isoformat()}",
            )
        bar_close = bar_open + pd.Timedelta(minutes=1)
        if context.side == "LONG":
            if values["low"] <= context.stop:
                return _audit(
                    context,
                    "INVALIDATED_STOP",
                    invalidated_at=bar_close.isoformat(),
                )
            if values["high"] >= context.tp_prices[0]:
                return _audit(
                    context,
                    "INVALIDATED_TP1",
                    invalidated_at=bar_close.isoformat(),
                )
            penetration = max(
                0.0,
                context.entry_max - values["low"],
            ) / width
            approach_ok = (
                previous_close is not None
                and previous_close > context.entry_max
            )
            triggered = (
                approach_ok
                and values["low"] <= context.entry_max
                and penetration >= config.min_penetration_fraction
                and values["close"] > context.entry_max
                and values["close"] > values["open"]
            )
        else:
            if values["high"] >= context.stop:
                return _audit(
                    context,
                    "INVALIDATED_STOP",
                    invalidated_at=bar_close.isoformat(),
                )
            if values["low"] <= context.tp_prices[0]:
                return _audit(
                    context,
                    "INVALIDATED_TP1",
                    invalidated_at=bar_close.isoformat(),
                )
            penetration = max(
                0.0,
                values["high"] - context.entry_min,
            ) / width
            approach_ok = (
                previous_close is not None
                and previous_close < context.entry_min
            )
            triggered = (
                approach_ok
                and values["high"] >= context.entry_min
                and penetration >= config.min_penetration_fraction
                and values["close"] < context.entry_min
                and values["close"] < values["open"]
            )
        if not triggered:
            previous_close = values["close"]
            continue

        next_rows = frame.loc[
            (frame.index >= bar_close)
            & (frame.index < context.expires_at)
        ]
        if next_rows.empty:
            return _audit(
                context,
                "NO_CAUSAL_ENTRY_BAR",
                trigger_bar_open=bar_open.isoformat(),
                trigger_time=bar_close.isoformat(),
            )
        entry_time = next_rows.index[0]
        entry_price = float(next_rows["open"].iloc[0])
        if (
            (context.side == "LONG" and not (
                context.stop < entry_price < context.tp_prices[0]
            ))
            or (
                context.side == "SHORT"
                and not (
                    context.tp_prices[0] < entry_price < context.stop
                )
            )
        ):
            return _audit(
                context,
                "ENTRY_GAP_INVALID",
                trigger_bar_open=bar_open.isoformat(),
                trigger_time=bar_close.isoformat(),
                entry_time=entry_time.isoformat(),
                entry_price=entry_price,
            )
        extension_r = abs(entry_price - context.planned_entry) / risk
        if extension_r > config.max_entry_extension_r:
            return _audit(
                context,
                "ENTRY_EXTENSION_REJECT",
                trigger_bar_open=bar_open.isoformat(),
                trigger_time=bar_close.isoformat(),
                entry_time=entry_time.isoformat(),
                entry_price=entry_price,
                entry_extension_r=extension_r,
            )
        return _audit(
            context,
            "TRIGGERED",
            trigger_bar_open=bar_open.isoformat(),
            trigger_time=bar_close.isoformat(),
            entry_time=entry_time.isoformat(),
            entry_price=entry_price,
            penetration_fraction=penetration,
            entry_extension_r=extension_r,
            signal={
                **dict(context.signal),
                "entry_price": entry_price,
                "entry_min": entry_price,
                "entry_max": entry_price,
                "trigger_kind": (
                    f"{context.trigger_kind}__m1_reclaim"
                ),
                "parent_trigger_kind": context.trigger_kind,
                "h1_context_id": context.context_id,
                "m1_trigger_time": bar_close.isoformat(),
                "m1_entry_time": entry_time.isoformat(),
            },
        )
    return _audit(context, "NO_TRIGGER")


__all__ = [
    "H1_M1_TRIGGER_SCHEMA",
    "H1Context",
    "M1ReclaimConfig",
    "detect_m1_reclaim",
]
