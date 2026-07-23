"""Leakage-resistant rolling walk-forward interval construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {value!r}")
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _duration(value: Any, *, name: str) -> pd.Timedelta:
    try:
        duration = pd.Timedelta(value)
    except Exception as exc:
        raise ValueError(f"Invalid {name}: {value!r}") from exc
    if duration <= pd.Timedelta(0):
        raise ValueError(f"{name} must be positive")
    return duration


@dataclass(frozen=True)
class WalkForwardFold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


def split_walk_forward(
    *,
    start: Any,
    end: Any,
    train: Any,
    test: Any,
    step: Any | None = None,
) -> tuple[WalkForwardFold, ...]:
    """Return half-open ``[start, end)`` rolling train/test folds.

    Test immediately follows train and is never part of that fold's training
    interval. ``step`` defaults to the test duration, yielding non-overlapping
    OOS windows. Parameter selection is intentionally outside this primitive.
    """

    range_start = _utc(start)
    range_end = _utc(end)
    if range_end <= range_start:
        raise ValueError("end must be after start")
    train_delta = _duration(train, name="train")
    test_delta = _duration(test, name="test")
    step_delta = _duration(step if step is not None else test_delta, name="step")

    folds: list[WalkForwardFold] = []
    cursor = range_start
    index = 0
    while True:
        train_end = cursor + train_delta
        test_start = train_end
        test_end = test_start + test_delta
        if test_end > range_end:
            break
        folds.append(
            WalkForwardFold(
                index=index,
                train_start=cursor,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        index += 1
        cursor += step_delta
    return tuple(folds)
