from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from tools.export_quality_training_rows import (
    ExportError,
    export_quality_training_rows,
)


def _idea(
    number: int,
    *,
    version: str | None = "strategy-v1",
    side: str = "LONG",
    broker_net_usd: float = 50.0,
    decision_time: str = "2026-08-17T10:00:00+00:00",
) -> dict[str, Any]:
    return {
        "idea_number": number,
        "audit_id": f"audit-{number}",
        "opportunity_id": f"opportunity-{number}",
        "setup_id": f"setup-{number}",
        "idea_id": f"idea-{number}",
        "opened_utc": decision_time,
        "symbol": "EURUSD",
        "side": side,
        "trigger_kind": "PIVOT_RECLAIM_H1_M15",
        "strategy_version": version,
        "deployment_id": "deployment-a",
        "broker_net_usd": broker_net_usd,
        "leg_count": 2,
        "entry": {
            "execution_type": "PERSISTENT_LIMIT",
            "planned_risk_entry": 1.1000,
        },
        "stop": {
            "recorded_strategy_stop": 1.0900,
            "technical_risk_atr": 1.25,
        },
        "fvg_regime": side,
        "fvg_age_bars": 7,
    }


def _legs(number: int, *, tp1_reason: str = "TP") -> list[dict[str, Any]]:
    return [
        {
            "idea_number": number,
            "symbol": "EURUSD",
            "side": "LONG",
            "tp_index": 1,
            "close_reason": tp1_reason,
        },
        {
            "idea_number": number,
            "symbol": "EURUSD",
            "side": "LONG",
            "tp_index": 2,
            "close_reason": "SL",
        },
    ]


def _write_jsonl(path: Path, records: list[dict[str, Any]], *, allow_nan: bool = False) -> None:
    path.write_text(
        "".join(json.dumps(record, allow_nan=allow_nan) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_risk_tsv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    key_column: str = "idea_row_id",
) -> None:
    fieldnames = [key_column, "tp_index", "risk_amount", "request_spread"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _export(
    tmp_path: Path,
    ideas: list[dict[str, Any]],
    legs: list[dict[str, Any]],
    *,
    risk_rows: list[dict[str, Any]] | None = None,
    strategy_version: str = "strategy-v1",
    compatibility_group: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ideas_path = tmp_path / "ideas.jsonl"
    legs_path = tmp_path / "legs.jsonl"
    output_path = tmp_path / "training.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_jsonl(ideas_path, ideas)
    _write_jsonl(legs_path, legs)
    risk_path = None
    if risk_rows is not None:
        risk_path = tmp_path / "risk.tsv"
        _write_risk_tsv(risk_path, risk_rows)
    summary = export_quality_training_rows(
        ideas_path=ideas_path,
        legs_path=legs_path,
        risk_legs_tsv=risk_path,
        output_path=output_path,
        summary_path=summary_path,
        strategy_version=strategy_version,
        compatibility_group=compatibility_group,
    )
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
    return rows, summary


def _complete_risk(number: int) -> list[dict[str, Any]]:
    return [
        {
            "idea_row_id": number,
            "tp_index": 1,
            "risk_amount": 60.0,
            "request_spread": 0.0002,
        },
        {
            "idea_row_id": number,
            "tp_index": 2,
            "risk_amount": 40.0,
            "request_spread": 0.0003,
        },
    ]


def test_split_legs_export_exactly_one_causal_idea_row(tmp_path: Path) -> None:
    idea = _idea(1)
    idea["actual_retest_seconds"] = 12.5
    idea["closed_utc"] = "2026-08-17T11:00:00Z"
    idea["future_observation"] = {"maximum_favorable_excursion": 99}

    rows, summary = _export(tmp_path, [idea], _legs(1), risk_rows=_complete_risk(1))

    assert len(rows) == 1
    row = rows[0]
    assert row["source_idea_key"] == "1"
    assert row["label_tp1_hit"] == 1
    assert row["initial_risk_usd"] == pytest.approx(100.0)
    assert row["label_realized_net_r"] == pytest.approx(0.5)
    assert row["label_complete"] is True
    assert row["r_label_basis"] == "NET_R_AFTER_BROKER_COSTS"
    assert row["stop_atr_h1"] == pytest.approx(1.25)
    assert row["fvg_relative_state"] == "ALIGNED"
    assert row["fvg_age_bars"] == 7
    assert row["spread_r"] == pytest.approx(0.02)
    assert row["intended_execution_mode"] == "PERSISTENT_LIMIT"
    assert row["weekday_utc"] == "MONDAY"
    assert all("retest" not in key.lower() for key in row)
    assert "closed_utc" not in row
    assert "future_observation" not in row
    assert summary["written_ideas"] == 1
    assert summary["label_complete"] == 1


def test_tp1_hit_and_net_positive_are_independent_labels(tmp_path: Path) -> None:
    first = _idea(1, broker_net_usd=25.0)
    second = _idea(2, broker_net_usd=-30.0, decision_time="2026-08-17T11:00:00Z")
    rows, _ = _export(
        tmp_path,
        [first, second],
        _legs(1, tp1_reason="SL") + _legs(2, tp1_reason="TP"),
        risk_rows=_complete_risk(1) + _complete_risk(2),
    )

    assert rows[0]["label_tp1_hit"] == 0
    assert rows[0]["label_realized_net_r"] == pytest.approx(0.25)
    assert rows[1]["label_tp1_hit"] == 1
    assert rows[1]["label_realized_net_r"] == pytest.approx(-0.30)


@pytest.mark.parametrize("reason", ["TIME", "EXPERT", "UNKNOWN_REASON", ""])
def test_time_and_unknown_tp1_close_reasons_are_censored(
    tmp_path: Path, reason: str
) -> None:
    rows, summary = _export(
        tmp_path,
        [_idea(1)],
        _legs(1, tp1_reason=reason),
        risk_rows=_complete_risk(1),
    )

    assert rows[0]["label_tp1_hit"] is None
    assert rows[0]["label_realized_net_r"] == pytest.approx(0.5)
    assert rows[0]["label_complete"] is False
    assert summary["label_tp1_hit_censored"] == 1


def test_incomplete_risk_never_emits_realized_net_r(tmp_path: Path) -> None:
    partial_risk = [_complete_risk(1)[0]]
    rows, summary = _export(tmp_path, [_idea(1)], _legs(1), risk_rows=partial_risk)

    assert rows[0]["broker_net_usd"] == pytest.approx(50.0)
    assert rows[0]["initial_risk_usd"] is None
    assert rows[0]["label_realized_net_r"] is None
    assert rows[0]["label_complete"] is False
    assert summary["initial_risk_usd_complete"] == 0


def test_mixed_contracts_fail_without_explicit_compatibility_group(tmp_path: Path) -> None:
    first = _idea(1, version="strategy-v1")
    second = _idea(2, version="strategy-v2", decision_time="2026-08-17T11:00:00Z")
    ideas_path = tmp_path / "ideas.jsonl"
    legs_path = tmp_path / "legs.jsonl"
    _write_jsonl(ideas_path, [first, second])
    _write_jsonl(legs_path, _legs(1) + _legs(2))

    with pytest.raises(ExportError, match="explicit compatibility group"):
        export_quality_training_rows(
            ideas_path=ideas_path,
            legs_path=legs_path,
            output_path=tmp_path / "out.jsonl",
            summary_path=tmp_path / "summary.json",
            strategy_version="strategy-v1",
        )

    rows, summary = _export(
        tmp_path,
        [first, second],
        _legs(1) + _legs(2),
        strategy_version="strategy-v1",
        compatibility_group=("strategy-v2",),
    )
    assert [row["strategy_version"] for row in rows] == ["strategy-v1", "strategy-v2"]
    assert summary["compatibility_group"] == ["strategy-v1", "strategy-v2"]


def test_unset_strategy_contract_is_always_rejected(tmp_path: Path) -> None:
    ideas_path = tmp_path / "ideas.jsonl"
    legs_path = tmp_path / "legs.jsonl"
    _write_jsonl(ideas_path, [_idea(1), _idea(2, version=None)])
    _write_jsonl(legs_path, _legs(1) + _legs(2))

    with pytest.raises(ExportError, match="strategy_version"):
        export_quality_training_rows(
            ideas_path=ideas_path,
            legs_path=legs_path,
            output_path=tmp_path / "out.jsonl",
            summary_path=tmp_path / "summary.json",
            strategy_version="strategy-v1",
            compatibility_group=("legacy",),
        )


def test_duplicate_idea_leg_and_risk_leg_keys_are_rejected(tmp_path: Path) -> None:
    ideas_path = tmp_path / "ideas.jsonl"
    legs_path = tmp_path / "legs.jsonl"
    _write_jsonl(ideas_path, [_idea(1), _idea(1)])
    _write_jsonl(legs_path, _legs(1))
    with pytest.raises(ExportError, match="duplicate idea key"):
        export_quality_training_rows(
            ideas_path=ideas_path,
            legs_path=legs_path,
            output_path=tmp_path / "out.jsonl",
            summary_path=tmp_path / "summary.json",
            strategy_version="strategy-v1",
        )

    _write_jsonl(ideas_path, [_idea(1)])
    duplicate_legs = _legs(1) + [_legs(1)[0]]
    _write_jsonl(legs_path, duplicate_legs)
    with pytest.raises(ExportError, match="duplicate leg key"):
        export_quality_training_rows(
            ideas_path=ideas_path,
            legs_path=legs_path,
            output_path=tmp_path / "out.jsonl",
            summary_path=tmp_path / "summary.json",
            strategy_version="strategy-v1",
        )

    _write_jsonl(legs_path, _legs(1))
    risk_path = tmp_path / "risk.tsv"
    duplicate_risk = _complete_risk(1) + [_complete_risk(1)[0]]
    _write_risk_tsv(risk_path, duplicate_risk)
    with pytest.raises(ExportError, match="duplicate risk leg key"):
        export_quality_training_rows(
            ideas_path=ideas_path,
            legs_path=legs_path,
            risk_legs_tsv=risk_path,
            output_path=tmp_path / "out.jsonl",
            summary_path=tmp_path / "summary.json",
            strategy_version="strategy-v1",
        )


def test_nonfinite_json_and_risk_values_are_rejected(tmp_path: Path) -> None:
    ideas_path = tmp_path / "ideas.jsonl"
    legs_path = tmp_path / "legs.jsonl"
    _write_jsonl(ideas_path, [{**_idea(1), "broker_net_usd": float("nan")}], allow_nan=True)
    _write_jsonl(legs_path, _legs(1))
    with pytest.raises(ExportError, match="nonfinite"):
        export_quality_training_rows(
            ideas_path=ideas_path,
            legs_path=legs_path,
            output_path=tmp_path / "out.jsonl",
            summary_path=tmp_path / "summary.json",
            strategy_version="strategy-v1",
        )

    _write_jsonl(ideas_path, [_idea(1)])
    risk_path = tmp_path / "risk.tsv"
    bad_risk = _complete_risk(1)
    bad_risk[0]["risk_amount"] = "NaN"
    _write_risk_tsv(risk_path, bad_risk)
    with pytest.raises(ExportError, match="nonfinite"):
        export_quality_training_rows(
            ideas_path=ideas_path,
            legs_path=legs_path,
            risk_legs_tsv=risk_path,
            output_path=tmp_path / "out.jsonl",
            summary_path=tmp_path / "summary.json",
            strategy_version="strategy-v1",
        )


def test_validation_failure_preserves_existing_output(tmp_path: Path) -> None:
    ideas_path = tmp_path / "ideas.jsonl"
    legs_path = tmp_path / "legs.jsonl"
    output_path = tmp_path / "training.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_jsonl(ideas_path, [_idea(1), _idea(1)])
    _write_jsonl(legs_path, _legs(1))
    output_path.write_text("previous-output\n", encoding="utf-8")

    with pytest.raises(ExportError):
        export_quality_training_rows(
            ideas_path=ideas_path,
            legs_path=legs_path,
            output_path=output_path,
            summary_path=summary_path,
            strategy_version="strategy-v1",
        )

    assert output_path.read_text(encoding="utf-8") == "previous-output\n"
    assert not summary_path.exists()


def test_utf8_bom_inputs_are_accepted(tmp_path: Path) -> None:
    ideas_path = tmp_path / "ideas.jsonl"
    legs_path = tmp_path / "legs.jsonl"
    ideas_text = "".join(json.dumps(record) + "\n" for record in [_idea(1)])
    legs_text = "".join(json.dumps(record) + "\n" for record in _legs(1))
    ideas_path.write_bytes(b"\xef\xbb\xbf" + ideas_text.encode("utf-8"))
    legs_path.write_bytes(b"\xef\xbb\xbf" + legs_text.encode("utf-8"))

    summary = export_quality_training_rows(
        ideas_path=ideas_path,
        legs_path=legs_path,
        output_path=tmp_path / "training.jsonl",
        summary_path=tmp_path / "summary.json",
        strategy_version="strategy-v1",
    )
    assert summary["written_ideas"] == 1
