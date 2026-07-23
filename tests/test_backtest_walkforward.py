from __future__ import annotations

import pandas as pd
import pytest

from backtest.walkforward import split_walk_forward


def test_walk_forward_uses_half_open_non_leaking_boundaries():
    folds = split_walk_forward(
        start="2026-01-01T00:00:00Z",
        end="2026-04-01T00:00:00Z",
        train="30D",
        test="10D",
        step="10D",
    )
    assert len(folds) == 6
    for index, fold in enumerate(folds):
        assert fold.index == index
        assert fold.train_end == fold.test_start
        assert fold.train_start < fold.train_end <= fold.test_start < fold.test_end
        assert fold.train_start.tzinfo is not None
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.test_end == later.test_start
    assert folds[-1].test_end == pd.Timestamp("2026-04-01T00:00:00Z")


def test_default_step_equals_test_duration_and_drops_incomplete_tail():
    folds = split_walk_forward(
        start="2026-01-01",
        end="2026-02-15",
        train="20D",
        test="10D",
    )
    assert len(folds) == 2
    assert folds[0].test_end == folds[1].test_start
    assert folds[-1].test_end == pd.Timestamp("2026-02-10T00:00:00Z")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": "2026-01-02", "end": "2026-01-01", "train": "1D", "test": "1D"},
        {"start": "2026-01-01", "end": "2026-02-01", "train": "0D", "test": "1D"},
        {"start": "2026-01-01", "end": "2026-02-01", "train": "1D", "test": "-1D"},
    ],
)
def test_walk_forward_rejects_invalid_ranges(kwargs):
    with pytest.raises(ValueError):
        split_walk_forward(**kwargs)
