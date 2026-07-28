from __future__ import annotations

from pathlib import Path

import pytest

from backtest.orderflow_ingest import TapeColumns, TapeIngestError
from backtest.orderflow_inspect import build_parser, inspect_csv_tape, run


BAR = "2024-03-04T10:"

_HEADER = "exec_time,px,qty,taker_side\n"
_ROWS = (
    f"{BAR}00:05Z,1.08000,60,BUY\n"
    f"{BAR}01:00Z,1.08001,40,SELL\n"
    f"{BAR}02:00Z,1.08020,20,BUY\n"
    f"{BAR}20:00Z,1.08050,10,SELL\n"
)

_COLUMNS = TapeColumns(
    timestamp="exec_time",
    price="px",
    size="qty",
    aggressor="taker_side",
)


def _tape(tmp_path, body=_ROWS, header=_HEADER, name="tape.csv"):
    path = tmp_path / name
    path.write_text(header + body, encoding="utf-8")
    return path


def test_inspection_reports_tick_grid_labels_and_coverage(tmp_path):
    report = inspect_csv_tape([_tape(tmp_path)], columns=_COLUMNS)

    assert report["rows_read"] == 4
    assert report["unparseable_rows"] == 0
    assert report["out_of_order_rows"] == 0
    assert report["aggressor_labels"] == {"BUY": 2, "SELL": 2}
    assert report["recommended_tick_size"] == "0.00001"
    assert report["finest_observed_tick_size"] == "0.00001"
    assert report["coverage"]["bars_with_trades"] == 2
    assert report["coverage"]["bars_in_span"] == 2
    assert report["coverage"]["bars_missing"] == 0
    assert report["size"]["min"] == "10"
    assert report["size"]["max"] == "60"
    assert report["warnings"] == []

    suggested = report["suggested_converter_flags"]
    assert suggested["tick_size"] == "0.00001"
    assert suggested["aggressor_labels"] == {"candidates": ["BUY", "SELL"]}
    assert suggested["covers_from"] == "2024-03-04T10:00:00Z"
    assert suggested["covers_through"] == "2024-03-04T10:30:00Z"


def test_tick_grid_reflects_the_finest_observed_price(tmp_path):
    body = _ROWS + f"{BAR}03:00Z,1.080005,5,BUY\n"
    report = inspect_csv_tape([_tape(tmp_path, body=body)], columns=_COLUMNS)
    assert report["recommended_tick_size"] == "0.000005"
    assert report["finest_observed_tick_size"] == "0.000001"


def test_a_coarse_derived_tick_grid_is_flagged_as_sample_dependent(tmp_path):
    body = f"{BAR}00:05Z,1.08000,60,BUY\n" f"{BAR}01:00Z,1.08010,40,SELL\n"
    report = inspect_csv_tape([_tape(tmp_path, body=body)], columns=_COLUMNS)
    assert report["recommended_tick_size"] == "0.0001"
    assert report["finest_observed_tick_size"] == "0.00001"
    assert any("declare the instrument's real tick" in w for w in report["warnings"])


def test_a_third_aggressor_label_is_surfaced_not_swallowed(tmp_path):
    body = _ROWS + f"{BAR}04:00Z,1.08030,5,N\n"
    report = inspect_csv_tape([_tape(tmp_path, body=body)], columns=_COLUMNS)

    assert report["aggressor_labels"]["N"] == 1
    assert "aggressor_labels" not in report["suggested_converter_flags"]
    assert any("more than two aggressor labels" in item for item in report["warnings"])


def test_missing_bars_out_of_order_rows_and_bad_rows_are_reported(tmp_path):
    body = (
        f"{BAR}00:05Z,1.08000,60,BUY\n"
        f"{BAR}00:04Z,1.08000,60,SELL\n"
        "not-a-timestamp,1.08000,60,BUY\n"
        "2024-03-04T11:00:00Z,1.08000,60,BUY\n"
    )
    report = inspect_csv_tape([_tape(tmp_path, body=body)], columns=_COLUMNS)

    assert report["rows_read"] == 4
    assert report["unparseable_rows"] == 1
    assert report["unparseable_samples"][0]["location"].endswith(":4")
    assert report["out_of_order_rows"] == 1
    assert report["coverage"]["bars_in_span"] == 5
    assert report["coverage"]["bars_with_trades"] == 2
    assert report["coverage"]["bars_missing"] == 3
    joined = " ".join(report["warnings"])
    assert "out of chronological order" in joined
    assert "could not be parsed" in joined
    assert "carry no prints" in joined


def test_tz_naive_timestamps_are_counted_and_flagged(tmp_path):
    body = "2024-03-04T10:00:05,1.08000,60,BUY\n"
    report = inspect_csv_tape([_tape(tmp_path, body=body)], columns=_COLUMNS)
    assert report["tz_naive_timestamps"] == 1
    assert report["suggested_converter_flags"]["naive_timestamps_are_utc"] is True


def test_intrabar_gaps_are_measured(tmp_path):
    body = f"{BAR}00:00Z,1.08000,10,BUY\n" f"{BAR}12:00Z,1.08010,10,SELL\n"
    report = inspect_csv_tape([_tape(tmp_path, body=body)], columns=_COLUMNS)
    assert report["intrabar_gap_seconds"]["max"] == pytest.approx(720.0)


def test_limit_rows_truncates_the_scan(tmp_path):
    report = inspect_csv_tape(
        [_tape(tmp_path)],
        columns=_COLUMNS,
        limit_rows=2,
    )
    assert report["rows_read"] == 2


def test_limit_rows_must_be_positive(tmp_path):
    with pytest.raises(TapeIngestError, match="at least 1"):
        inspect_csv_tape(
            [_tape(tmp_path)],
            columns=_COLUMNS,
            limit_rows=0,
        )


def test_non_finite_and_non_positive_price_size_rows_are_profiled(tmp_path):
    body = (
        f"{BAR}00:05Z,NaN,10,BUY\n"
        f"{BAR}01:05Z,1.08000,Infinity,SELL\n"
        f"{BAR}02:05Z,-1.08000,10,BUY\n"
        f"{BAR}03:05Z,1.08000,0,SELL\n"
        f"{BAR}04:05Z,NaN,NaN,BUY\n"
    )
    report = inspect_csv_tape([_tape(tmp_path, body=body)], columns=_COLUMNS)

    assert report["rows_read"] == 5
    assert report["unparseable_rows"] == 5
    assert len(report["unparseable_samples"]) == 5
    assert "price must be finite and positive" in report["unparseable_samples"][0]["error"]
    assert "size must be finite and positive" in report["unparseable_samples"][1]["error"]
    assert "; " in report["unparseable_samples"][4]["error"]
    assert report["coverage"]["bars_with_trades"] == 0
    assert report["recommended_tick_size"] is None


def test_cli_run_rejects_a_missing_tape(tmp_path):
    args = build_parser().parse_args(["--tape", str(tmp_path / "absent.csv")])
    with pytest.raises(TapeIngestError, match="does not exist"):
        run(args)


def test_cli_run_wires_column_overrides(tmp_path):
    tape = _tape(tmp_path)
    args = build_parser().parse_args(
        [
            "--tape",
            str(tape),
            "--column-timestamp",
            "exec_time",
            "--column-price",
            "px",
            "--column-size",
            "qty",
            "--column-aggressor",
            "taker_side",
        ]
    )
    report = run(args)
    assert report["rows_read"] == 4
    assert report["inputs"] == [str(Path(tape).resolve())]
