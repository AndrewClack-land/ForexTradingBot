"""Tests for the offline quality-profile fit CLI."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.hierarchical_quality_score import HierarchicalQualityProfile
from tools.fit_quality_profile import FitError, fit_profile_from_rows, main


TRAIN_START = datetime(2026, 7, 6, 9, 30, tzinfo=timezone.utc)


def _row(index: int, **overrides) -> dict:
    decision = TRAIN_START + timedelta(hours=8 * index)
    row = {
        "schema_version": "quality-training-row/v1",
        "decision_time_utc": decision.isoformat().replace("+00:00", "Z"),
        "symbol": "EURUSD",
        "side": "LONG" if index % 2 == 0 else "SHORT",
        "trigger_kind": "PIVOT_RECLAIM_H1_M15",
        "stop_atr_h1": 1.0 + 0.05 * (index % 4),
        "fvg_relative_state": "NONE",
        "fvg_age_bars": None,
        "spread_r": None,
        "intended_execution_mode": "MARKET_SPLIT",
        "label_complete": True,
        "label_tp1_hit": int(index % 2 == 0),
        "label_realized_net_r": 0.8 if index % 2 == 0 else -1.1,
        "r_label_basis": "NET_R_AFTER_BROKER_COSTS",
    }
    row.update(overrides)
    return row


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_cli_fits_and_freezes_a_loadable_profile(tmp_path, capsys) -> None:
    rows_path = tmp_path / "rows.jsonl"
    output_path = tmp_path / "profile.json"
    _write_rows(rows_path, [_row(i) for i in range(8)])

    assert main([
        "--rows", str(rows_path),
        "--output", str(output_path),
        "--outside-stop-atr-penalty-r", "0.2",
    ]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["rows_in"] == 8
    assert report["support_global"] == 8
    profile = HierarchicalQualityProfile.load(output_path)
    assert profile.profile_id == report["profile_id"]
    assert profile.profile_sha256 == report["profile_sha256"]
    assert profile.outside_stop_atr_penalty_r == 0.2


def test_partial_labels_readmit_tp_only_rows() -> None:
    rows = [_row(i) for i in range(6)]
    # Rows without the risk breakdown: final TP1 label, censored net R.
    for index in range(6, 12):
        rows.append(_row(
            index,
            label_complete=False,
            label_realized_net_r=None,
        ))

    strict = fit_profile_from_rows(rows)
    assert strict.support_count("global", "*") == 6
    assert strict.tp1_model.sample_count == 6
    assert strict.net_r_model.sample_count == 6

    relaxed = fit_profile_from_rows(rows, allow_partial_labels=True)
    assert relaxed.support_count("global", "*") == 12
    assert relaxed.tp1_model.sample_count == 12
    assert relaxed.net_r_model.sample_count == 6


def test_partial_labels_never_invent_a_label() -> None:
    rows = [_row(i) for i in range(4)]
    rows.append(_row(
        99,
        label_complete=False,
        label_tp1_hit=None,
        label_realized_net_r=None,
    ))
    relaxed = fit_profile_from_rows(rows, allow_partial_labels=True)
    assert relaxed.support_count("global", "*") == 4


def test_cli_rejects_invalid_inputs(tmp_path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="no training rows"):
        main([
            "--rows", str(empty),
            "--output", str(tmp_path / "out.json"),
        ])

    with pytest.raises(FitError, match="nonnegative"):
        fit_profile_from_rows(
            [_row(0)],
            outside_stop_atr_penalty_r=-0.1,
        )
