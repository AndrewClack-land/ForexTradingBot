"""Export one causal quality-training row per audited trading idea.

The exporter is deliberately offline and read-only with respect to its inputs.  It
does not import the live trading package and it never expands split execution legs
into independent training observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROW_SCHEMA_VERSION = "quality-training-row/v1"
SUMMARY_SCHEMA_VERSION = "quality-training-summary/v1"
R_LABEL_BASIS = "NET_R_AFTER_BROKER_COSTS"


class ExportError(ValueError):
    """Raised when an input violates the quality-training data contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _reject_nonfinite(value: Any, *, location: str) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExportError(f"nonfinite value at {location}")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_nonfinite(nested, location=f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_nonfinite(nested, location=f"{location}[{index}]")


def _read_jsonl(path: Path, *, kind: str) -> list[dict[str, Any]]:
    try:
        # utf-8-sig: audit JSONL produced on Windows may carry a BOM.
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ExportError(f"cannot read {kind} JSONL {path}: {exc}") from exc

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExportError(f"invalid {kind} JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ExportError(f"{kind} row at {path}:{line_number} is not an object")
        _reject_nonfinite(value, location=f"{kind}[{line_number}]")
        records.append(value)
    return records


def _idea_key(record: Mapping[str, Any], *, location: str) -> str:
    number = record.get("idea_number")
    row_id = record.get("idea_row_id")
    if number not in (None, "") and row_id not in (None, ""):
        if str(number).strip() != str(row_id).strip():
            raise ExportError(f"conflicting idea_number and idea_row_id at {location}")
    value = number if number not in (None, "") else row_id
    if value is None or isinstance(value, bool) or not str(value).strip():
        raise ExportError(f"missing idea_number/idea_row_id at {location}")
    return str(value).strip()


def _positive_index(value: Any, *, location: str) -> int:
    if isinstance(value, bool):
        raise ExportError(f"invalid tp_index at {location}")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ExportError(f"invalid tp_index at {location}") from exc
    if number <= 0 or str(value).strip() not in {str(number), f"{number}.0"}:
        raise ExportError(f"invalid tp_index at {location}")
    return number


def _optional_number(value: Any, *, location: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ExportError(f"invalid numeric value at {location}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExportError(f"invalid numeric value at {location}") from exc
    if not math.isfinite(number):
        raise ExportError(f"nonfinite value at {location}")
    return number


def _positive_optional_number(value: Any, *, location: str) -> float | None:
    number = _optional_number(value, location=location)
    if number is not None and number <= 0:
        raise ExportError(f"expected a positive value at {location}")
    return number


def _parse_decision_time(value: Any, *, location: str) -> tuple[datetime, str]:
    if not isinstance(value, str) or not value.strip():
        raise ExportError(f"missing decision timestamp at {location}")
    raw = value.strip()
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
    except ValueError as exc:
        raise ExportError(f"invalid decision timestamp at {location}: {raw}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ExportError(f"decision timestamp must be timezone-aware at {location}")
    utc = parsed.astimezone(timezone.utc)
    normalized = utc.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return utc, normalized


def _required_text(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExportError(f"missing text value at {location}")
    return value.strip()


def _optional_text(value: Any, *, location: str) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not value.strip():
        raise ExportError(f"invalid text value at {location}")
    return value.strip()


def _read_risk_legs(path: Path | None) -> dict[tuple[str, int], dict[str, float | None]]:
    if path is None:
        return {}
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ExportError(f"cannot read risk-legs TSV {path}: {exc}") from exc

    with handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = set(reader.fieldnames or ())
        key_column = "idea_row_id" if "idea_row_id" in fieldnames else "idea_number"
        required = {key_column, "tp_index", "risk_amount"}
        if not required.issubset(fieldnames):
            missing = ", ".join(sorted(required - fieldnames))
            raise ExportError(f"risk-legs TSV is missing columns: {missing}")

        records: dict[tuple[str, int], dict[str, float | None]] = {}
        for line_number, row in enumerate(reader, start=2):
            key = _idea_key(
                {
                    key_column: row.get(key_column),
                },
                location=f"risk_legs[{line_number}]",
            )
            tp_index = _positive_index(
                row.get("tp_index"), location=f"risk_legs[{line_number}].tp_index"
            )
            leg_key = (key, tp_index)
            if leg_key in records:
                raise ExportError(f"duplicate risk leg key {leg_key!r}")
            risk_amount = _positive_optional_number(
                row.get("risk_amount"),
                location=f"risk_legs[{line_number}].risk_amount",
            )
            request_spread = None
            if "request_spread" in fieldnames:
                request_spread = _optional_number(
                    row.get("request_spread"),
                    location=f"risk_legs[{line_number}].request_spread",
                )
                if request_spread is not None and request_spread < 0:
                    raise ExportError(
                        f"expected a non-negative value at risk_legs[{line_number}].request_spread"
                    )
            records[leg_key] = {
                "risk_amount": risk_amount,
                "request_spread": request_spread,
            }
    return records


def _fvg_relative_state(regime: Any, side: str, *, location: str) -> str:
    if regime is None or regime == "":
        return "NONE"
    value = _required_text(regime, location=location).upper()
    if value in {"NONE", "NEUTRAL"}:
        return "NONE"
    if value not in {"LONG", "SHORT"}:
        raise ExportError(f"unsupported FVG state at {location}: {value}")
    return "ALIGNED" if value == side else "OPPOSED"


def _fvg_age(value: Any, *, location: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ExportError(f"invalid FVG age at {location}")
    try:
        age = int(value)
    except (TypeError, ValueError) as exc:
        raise ExportError(f"invalid FVG age at {location}") from exc
    if age < 0 or str(value).strip() not in {str(age), f"{age}.0"}:
        raise ExportError(f"invalid FVG age at {location}")
    return age


def _planned_risk_distance(idea: Mapping[str, Any], *, location: str) -> float | None:
    entry = idea.get("entry")
    stop = idea.get("stop")
    if not isinstance(entry, Mapping) or not isinstance(stop, Mapping):
        return None
    planned_entry = _optional_number(
        entry.get("planned_risk_entry"), location=f"{location}.entry.planned_risk_entry"
    )
    planned_stop = _optional_number(
        stop.get("recorded_strategy_stop"), location=f"{location}.stop.recorded_strategy_stop"
    )
    if planned_entry is None or planned_stop is None:
        return None
    distance = abs(planned_entry - planned_stop)
    return distance if distance > 0 else None


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _assert_distinct_paths(inputs: Iterable[Path], outputs: Sequence[Path]) -> None:
    input_paths = {path.resolve() for path in inputs}
    output_paths = [path.resolve() for path in outputs]
    if len(set(output_paths)) != len(output_paths):
        raise ExportError("output and summary paths must be distinct")
    overlap = input_paths.intersection(output_paths)
    if overlap:
        raise ExportError(f"refusing to overwrite input path: {next(iter(overlap))}")


def export_quality_training_rows(
    *,
    ideas_path: Path,
    legs_path: Path,
    output_path: Path,
    summary_path: Path,
    strategy_version: str,
    risk_legs_tsv: Path | None = None,
    compatibility_group: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate and atomically export a causal, idea-level training dataset."""

    target_version = _required_text(strategy_version, location="strategy_version")
    compatible_versions = {
        _required_text(value, location="compatibility_group") for value in compatibility_group
    }
    allowed_versions = {target_version, *compatible_versions}
    inputs = [ideas_path, legs_path]
    if risk_legs_tsv is not None:
        inputs.append(risk_legs_tsv)
    _assert_distinct_paths(inputs, [output_path, summary_path])

    ideas = _read_jsonl(ideas_path, kind="ideas")
    legs = _read_jsonl(legs_path, kind="legs")
    if not ideas:
        raise ExportError("ideas JSONL is empty")

    ideas_by_key: dict[str, dict[str, Any]] = {}
    observed_versions: set[str] = set()
    for index, idea in enumerate(ideas, start=1):
        key = _idea_key(idea, location=f"ideas[{index}]")
        if key in ideas_by_key:
            raise ExportError(f"duplicate idea key {key!r}")
        version = _required_text(
            idea.get("strategy_version"), location=f"ideas[{index}].strategy_version"
        )
        observed_versions.add(version)
        ideas_by_key[key] = idea

    if target_version not in observed_versions:
        raise ExportError(f"strategy version {target_version!r} is not present in ideas")
    unexpected_versions = observed_versions - allowed_versions
    if unexpected_versions:
        rendered = ", ".join(sorted(unexpected_versions))
        raise ExportError(
            "mixed strategy contracts require an explicit compatibility group; "
            f"unexpected versions: {rendered}"
        )

    legs_by_idea: dict[str, dict[int, dict[str, Any]]] = {key: {} for key in ideas_by_key}
    all_leg_keys: set[tuple[str, int]] = set()
    for index, leg in enumerate(legs, start=1):
        key = _idea_key(leg, location=f"legs[{index}]")
        if key not in ideas_by_key:
            raise ExportError(f"orphan leg references unknown idea key {key!r}")
        tp_index = _positive_index(leg.get("tp_index"), location=f"legs[{index}].tp_index")
        leg_key = (key, tp_index)
        if leg_key in all_leg_keys:
            raise ExportError(f"duplicate leg key {leg_key!r}")
        all_leg_keys.add(leg_key)
        legs_by_idea[key][tp_index] = leg

    risk_legs = _read_risk_legs(risk_legs_tsv)
    orphan_risk_keys = set(risk_legs) - all_leg_keys
    if orphan_risk_keys:
        raise ExportError(f"orphan risk leg key {sorted(orphan_risk_keys)[0]!r}")

    rows_with_sort_keys: list[tuple[datetime, str, dict[str, Any]]] = []
    for key, idea in ideas_by_key.items():
        location = f"idea[{key}]"
        decision_dt, decision_time = _parse_decision_time(
            idea.get("decision_time_utc", idea.get("opened_utc")),
            location=f"{location}.decision_time_utc",
        )
        symbol = _required_text(idea.get("symbol"), location=f"{location}.symbol").upper()
        side = _required_text(idea.get("side"), location=f"{location}.side").upper()
        if side not in {"LONG", "SHORT"}:
            raise ExportError(f"unsupported side at {location}.side: {side}")
        trigger_kind = _required_text(
            idea.get("trigger_kind"), location=f"{location}.trigger_kind"
        )
        version = _required_text(
            idea.get("strategy_version"), location=f"{location}.strategy_version"
        )

        idea_legs = legs_by_idea[key]
        declared_leg_count = idea.get("leg_count")
        if declared_leg_count not in (None, ""):
            expected_count = _positive_index(
                declared_leg_count, location=f"{location}.leg_count"
            )
            if expected_count != len(idea_legs):
                raise ExportError(
                    f"leg_count mismatch for idea {key!r}: {expected_count} != {len(idea_legs)}"
                )
        for tp_index, leg in idea_legs.items():
            leg_symbol = leg.get("symbol")
            leg_side = leg.get("side")
            if leg_symbol not in (None, "") and str(leg_symbol).strip().upper() != symbol:
                raise ExportError(f"symbol mismatch for leg {(key, tp_index)!r}")
            if leg_side not in (None, "") and str(leg_side).strip().upper() != side:
                raise ExportError(f"side mismatch for leg {(key, tp_index)!r}")

        tp1_leg = idea_legs.get(1)
        close_reason = ""
        if tp1_leg is not None:
            raw_reason = tp1_leg.get("close_reason")
            if raw_reason not in (None, ""):
                close_reason = _required_text(
                    raw_reason, location=f"{location}.tp1.close_reason"
                ).upper()
        label_tp1_hit = 1 if close_reason == "TP" else 0 if close_reason == "SL" else None

        broker_net_usd = _optional_number(
            idea.get("broker_net_usd"), location=f"{location}.broker_net_usd"
        )
        risk_values: list[float] = []
        risk_complete = bool(idea_legs) and risk_legs_tsv is not None
        for tp_index in sorted(idea_legs):
            risk_record = risk_legs.get((key, tp_index))
            if risk_record is None or risk_record["risk_amount"] is None:
                risk_complete = False
                continue
            risk_values.append(float(risk_record["risk_amount"]))
        initial_risk_usd = math.fsum(risk_values) if risk_complete else None
        label_realized_net_r = (
            broker_net_usd / initial_risk_usd
            if broker_net_usd is not None
            and initial_risk_usd is not None
            and initial_risk_usd > 0
            else None
        )

        stop = idea.get("stop")
        stop_atr_h1 = None
        if isinstance(stop, Mapping):
            stop_atr_h1 = _positive_optional_number(
                stop.get("technical_risk_atr"),
                location=f"{location}.stop.technical_risk_atr",
            )
        entry = idea.get("entry")
        intended_execution_mode = None
        if isinstance(entry, Mapping):
            intended_execution_mode = _optional_text(
                entry.get("execution_type"),
                location=f"{location}.entry.execution_type",
            )

        spread_r = None
        risk_tp1 = risk_legs.get((key, 1))
        planned_distance = _planned_risk_distance(idea, location=location)
        if risk_tp1 is not None and planned_distance is not None:
            request_spread = risk_tp1["request_spread"]
            if request_spread is not None:
                spread_r = request_spread / planned_distance

        fvg_relative_state = _fvg_relative_state(
            idea.get("fvg_regime"), side, location=f"{location}.fvg_regime"
        )
        fvg_age_bars = _fvg_age(
            idea.get("fvg_age_bars"), location=f"{location}.fvg_age_bars"
        )
        if fvg_relative_state == "NONE":
            fvg_age_bars = None

        row = {
            "schema_version": ROW_SCHEMA_VERSION,
            "source_idea_key": key,
            "opportunity_id": idea.get("opportunity_id"),
            "setup_id": idea.get("setup_id"),
            "idea_id": idea.get("idea_id"),
            "audit_id": idea.get("audit_id"),
            "decision_time_utc": decision_time,
            "weekday_utc": decision_dt.strftime("%A").upper(),
            "symbol": symbol,
            "side": side,
            "trigger_kind": trigger_kind,
            "strategy_version": version,
            "deployment_id": idea.get("deployment_id"),
            "label_complete": label_tp1_hit is not None and label_realized_net_r is not None,
            "label_tp1_hit": label_tp1_hit,
            "broker_net_usd": broker_net_usd,
            "initial_risk_usd": initial_risk_usd,
            "label_realized_net_r": label_realized_net_r,
            "r_label_basis": R_LABEL_BASIS,
            "stop_atr_h1": stop_atr_h1,
            "fvg_relative_state": fvg_relative_state,
            "fvg_age_bars": fvg_age_bars,
            "spread_r": spread_r,
            "intended_execution_mode": intended_execution_mode,
        }
        _reject_nonfinite(row, location=f"output[{key}]")
        rows_with_sort_keys.append((decision_dt, key, row))

    rows_with_sort_keys.sort(key=lambda item: (item[0], item[1]))
    rows = [item[2] for item in rows_with_sort_keys]
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "target_strategy_version": target_version,
        "compatibility_group": sorted(allowed_versions),
        "observed_strategy_versions": sorted(observed_versions),
        "input_ideas": len(ideas),
        "input_legs": len(legs),
        "written_ideas": len(rows),
        "label_complete": sum(bool(row["label_complete"]) for row in rows),
        "label_tp1_hit_complete": sum(row["label_tp1_hit"] is not None for row in rows),
        "label_tp1_hit_censored": sum(row["label_tp1_hit"] is None for row in rows),
        "broker_net_usd_complete": sum(row["broker_net_usd"] is not None for row in rows),
        "initial_risk_usd_complete": sum(row["initial_risk_usd"] is not None for row in rows),
        "label_realized_net_r_complete": sum(
            row["label_realized_net_r"] is not None for row in rows
        ),
        "stop_atr_h1_complete": sum(row["stop_atr_h1"] is not None for row in rows),
        "spread_r_complete": sum(row["spread_r"] is not None for row in rows),
        "risk_legs_tsv_supplied": risk_legs_tsv is not None,
    }

    output_text = "".join(f"{_canonical_json(row)}\n" for row in rows)
    summary_text = f"{_canonical_json(summary)}\n"
    _atomic_write(output_path, output_text)
    _atomic_write(summary_path, summary_text)
    return summary


def _compatibility_versions(values: Sequence[str]) -> tuple[str, ...]:
    versions: list[str] = []
    for value in values:
        versions.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(versions)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ideas", required=True, type=Path, help="canonical ideas.jsonl")
    parser.add_argument("--legs", required=True, type=Path, help="canonical legs.jsonl")
    parser.add_argument("--risk-legs-tsv", type=Path, help="optional causal risk/request TSV")
    parser.add_argument("--output", required=True, type=Path, help="training rows JSONL")
    parser.add_argument("--summary-output", required=True, type=Path, help="coverage summary JSON")
    parser.add_argument("--strategy-version", required=True, help="training contract anchor")
    parser.add_argument(
        "--compatibility-group",
        action="append",
        default=[],
        metavar="VERSION[,VERSION...]",
        help="explicit allow-list for additional compatible strategy versions",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = export_quality_training_rows(
            ideas_path=args.ideas,
            legs_path=args.legs,
            risk_legs_tsv=args.risk_legs_tsv,
            output_path=args.output,
            summary_path=args.summary_output,
            strategy_version=args.strategy_version,
            compatibility_group=_compatibility_versions(args.compatibility_group),
        )
    except ExportError as exc:
        raise SystemExit(f"quality training export failed: {exc}") from exc
    print(_canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
