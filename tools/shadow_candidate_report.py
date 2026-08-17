"""Read-only report for simultaneous-trigger shadow telemetry."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from statistics import mean
from typing import Any
from urllib.parse import quote

from config import SHADOW_CANDIDATE_LEDGER_DB_PATH


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _open_read_only(path: Path) -> sqlite3.Connection:
    encoded = quote(path.resolve().as_posix(), safe="/:")
    connection = sqlite3.connect(
        f"file:{encoded}?mode=ro",
        uri=True,
        timeout=5.0,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def _candidate_metric(rows: list[sqlite3.Row]) -> dict[str, Any]:
    net_values = [
        float(row["expected_net_r"])
        for row in rows
        if row["expected_net_r"] is not None
    ]
    quality_fields = (
        "quality_tp_probability",
        "quality_tp_lcb",
        "quality_tp_ucb",
        "quality_expected_net_r",
        "quality_conservative_net_r",
        "quality_net_uncertainty_r",
        "quality_effective_n",
        "quality_stop_atr_h1",
        "quality_rank_score_r",
    )
    quality_means = {}
    for field in quality_fields:
        values = [float(row[field]) for row in rows if row[field] is not None]
        quality_means[f"mean_{field}"] = mean(values) if values else None

    status_counts = Counter(
        str(row["quality_status"]) for row in rows if row["quality_status"] is not None
    )
    model_counts = Counter(
        str(row["quality_model_id"])
        for row in rows
        if row["quality_model_id"] is not None
    )
    backoff_counts = Counter(
        str(row["quality_backoff_path"])
        for row in rows
        if row["quality_backoff_path"] is not None
    )
    return {
        "candidate_observations": len(rows),
        "scans": len({str(row["scan_id"]) for row in rows}),
        "independent_opportunities": len({str(row["opportunity_id"]) for row in rows}),
        "structural_events": len({str(row["event_id"]) for row in rows}),
        "production_selected_observations": sum(
            int(row["selected_by_production"]) for row in rows
        ),
        "shadow_selected_observations": sum(
            int(row["shadow_selected"] or 0) for row in rows
        ),
        "unselected_independent_opportunities": len(
            {
                str(row["opportunity_id"])
                for row in rows
                if not int(row["selected_by_production"])
            }
        ),
        "net_scored_observations": len(net_values),
        "mean_expected_net_r": mean(net_values) if net_values else None,
        "quality_status_observations": sum(
            row["quality_status"] is not None for row in rows
        ),
        "quality_scored_observations": sum(
            row["quality_conservative_net_r"] is not None for row in rows
        ),
        "quality_selected_observations": sum(
            int(row["quality_selected"] or 0) for row in rows
        ),
        "quality_feature_missing_observations": sum(
            row["quality_feature_missing_json"] not in {None, "[]"} for row in rows
        ),
        "quality_statuses": dict(sorted(status_counts.items())),
        "quality_models": dict(sorted(model_counts.items())),
        "quality_backoff_paths": dict(sorted(backoff_counts.items())),
        **quality_means,
    }


def _trigger_family(trigger_kind: Any) -> str | None:
    normalized = str(trigger_kind or "").strip().lower().replace("-", "_")
    if "pivot" in normalized and "reclaim" in normalized:
        return "pivot_reclaim"
    if "order_block" in normalized or "orderblock" in normalized:
        return "order_block"
    return None


def _winner_disagreement(
    rows_by_scan: dict[str, list[sqlite3.Row]],
) -> dict[str, Any]:
    simultaneous = 0
    ranked = 0
    comparable = 0
    agreements = 0
    disagreements = 0
    for rows in rows_by_scan.values():
        if len(rows) < 2:
            continue
        simultaneous += 1
        if any(row["quality_rank_position"] is not None for row in rows):
            ranked += 1
        production = {
            str(row["opportunity_id"])
            for row in rows
            if int(row["selected_by_production"])
        }
        quality = {
            str(row["opportunity_id"])
            for row in rows
            if int(row["quality_selected"] or 0)
        }
        if not production or not quality:
            continue
        comparable += 1
        if production & quality:
            agreements += 1
        else:
            disagreements += 1
    return {
        "simultaneous_candidate_scans": simultaneous,
        "quality_ranked_simultaneous_scans": ranked,
        "comparable_winner_scans": comparable,
        "agreement_scans": agreements,
        "disagreement_scans": disagreements,
        "disagreement_rate": (disagreements / comparable if comparable else None),
    }


def build_report(
    path: Path,
    *,
    since: datetime | None = None,
    symbol: str | None = None,
    trigger: str | None = None,
    side: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    scan_where: list[str] = []
    scan_params: list[Any] = []
    if since is not None:
        scan_where.append("observed_at_utc >= ?")
        scan_params.append(since.astimezone(timezone.utc).isoformat())
    if symbol:
        scan_where.append("symbol = ?")
        scan_params.append(symbol.strip().upper())
    scan_clause = " WHERE " + " AND ".join(scan_where) if scan_where else ""

    candidate_where: list[str] = []
    candidate_params: list[Any] = []
    if since is not None:
        candidate_where.append("s.observed_at_utc >= ?")
        candidate_params.append(since.astimezone(timezone.utc).isoformat())
    if symbol:
        candidate_where.append("c.symbol = ?")
        candidate_params.append(symbol.strip().upper())
    if trigger:
        candidate_where.append("c.trigger_kind = ?")
        candidate_params.append(trigger.strip())
    if side:
        candidate_where.append("UPPER(COALESCE(c.side,'')) = ?")
        candidate_params.append(side.strip().upper())
    if status:
        candidate_where.append("UPPER(COALESCE(c.quality_status,'')) = ?")
        candidate_params.append(status.strip().upper())
    candidate_clause = (
        " WHERE " + " AND ".join(candidate_where) if candidate_where else ""
    )

    with _open_read_only(path) as connection:
        version = connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
        if version is None or int(version[0]) != 2:
            raise RuntimeError(
                "unsupported shadow candidate schema; "
                "open it once with ShadowCandidateLedger to migrate to v2"
            )
        scans = list(
            connection.execute(
                "SELECT * FROM scans"
                + scan_clause
                + " ORDER BY observed_at_utc, scan_id",
                scan_params,
            )
        )
        candidates = list(
            connection.execute(
                "SELECT c.*,s.observed_at_utc,s.downstream_disposition,"
                "s.watcher_plan_id "
                "FROM candidates c JOIN scans s ON s.scan_id=c.scan_id"
                + candidate_clause
                + " ORDER BY s.observed_at_utc,c.candidate_rank",
                candidate_params,
            )
        )

    groups: dict[str, list[sqlite3.Row]] = defaultdict(list)
    by_scan: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in candidates:
        key = (
            f"{row['symbol']} x {row['trigger_kind'] or 'UNKNOWN'} "
            f"x {row['side'] or 'UNKNOWN'}"
        )
        groups[key].append(row)
        by_scan[str(row["scan_id"])].append(row)

    combinations = Counter()
    for rows in by_scan.values():
        if len(rows) < 2:
            continue
        combination = " + ".join(
            sorted(str(row["trigger_kind"] or "UNKNOWN") for row in rows)
        )
        combinations[combination] += 1

    disposition_counts = Counter(
        str(row["downstream_disposition"] or "UNRECORDED") for row in scans
    )
    atr_stop_buckets = {
        "inside_0.75_1.50": _candidate_metric(
            [
                row
                for row in candidates
                if row["quality_stop_atr_h1"] is not None
                and 0.75 <= float(row["quality_stop_atr_h1"]) <= 1.50
            ]
        ),
        "outside_0.75_1.50": _candidate_metric(
            [
                row
                for row in candidates
                if row["quality_stop_atr_h1"] is not None
                and not 0.75 <= float(row["quality_stop_atr_h1"]) <= 1.50
            ]
        ),
        "missing": _candidate_metric(
            [row for row in candidates if row["quality_stop_atr_h1"] is None]
        ),
    }

    gbpusd_short_rows = [
        row
        for row in candidates
        if str(row["symbol"]).upper() == "GBPUSD"
        and str(row["side"] or "").upper() == "SHORT"
    ]
    gbpusd_short = _candidate_metric(gbpusd_short_rows)
    gbpusd_by_trigger: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in gbpusd_short_rows:
        gbpusd_by_trigger[str(row["trigger_kind"] or "UNKNOWN")].append(row)
    gbpusd_short["by_trigger"] = {
        key: _candidate_metric(rows) for key, rows in sorted(gbpusd_by_trigger.items())
    }

    pivot_ob_comparisons: dict[str, Any] = {}
    for family in ("pivot_reclaim", "order_block"):
        family_rows = [
            row for row in candidates if _trigger_family(row["trigger_kind"]) == family
        ]
        aggregate = {
            side_key: _candidate_metric(
                [
                    row
                    for row in family_rows
                    if str(row["side"] or "").upper() == side_key
                ]
            )
            for side_key in ("LONG", "SHORT")
        }
        by_symbol = {}
        for symbol_key in sorted({str(row["symbol"]).upper() for row in family_rows}):
            by_symbol[symbol_key] = {
                side_key: _candidate_metric(
                    [
                        row
                        for row in family_rows
                        if str(row["symbol"]).upper() == symbol_key
                        and str(row["side"] or "").upper() == side_key
                    ]
                )
                for side_key in ("LONG", "SHORT")
            }
        pivot_ob_comparisons[family] = {
            "aggregate": aggregate,
            "by_symbol": by_symbol,
        }

    watcher_plan_scans = sum(
        bool(str(row["watcher_plan_id"] or "").strip()) for row in scans
    )
    total = _candidate_metric(candidates)
    total.update(
        {
            "decision_scans": len(scans),
            "zero_candidate_scans": sum(
                int(row["candidate_count"]) == 0 for row in scans
            ),
            "simultaneous_candidate_scans": sum(
                int(row["candidate_count"]) > 1 for row in scans
            ),
            "watcher_plan_id_scans": watcher_plan_scans,
            "watcher_plan_id_coverage": (
                watcher_plan_scans / len(scans) if scans else None
            ),
        }
    )
    return {
        "db": str(path.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "unit_definition": {
            "scan": "one deployment x symbol x UTC scheduler minute",
            "structural_event": "one symbol x trigger signature",
            "independent_opportunity": (
                "one symbol x trigger signature x UTC trading day"
            ),
        },
        "filters": {
            "since": since.isoformat() if since else None,
            "symbol": symbol,
            "trigger": trigger,
            "side": side,
            "status": status,
        },
        "total": total,
        "downstream_dispositions": dict(sorted(disposition_counts.items())),
        "candidate_groups": {
            key: _candidate_metric(rows) for key, rows in sorted(groups.items())
        },
        "atr_stop_buckets": atr_stop_buckets,
        "gbpusd_short": gbpusd_short,
        "pivot_ob_side_comparisons": pivot_ob_comparisons,
        "production_vs_quality_winner": _winner_disagreement(by_scan),
        "watcher_plan_coverage": {
            "decision_scans": len(scans),
            "scans_with_watcher_plan_id": watcher_plan_scans,
            "coverage_rate": (watcher_plan_scans / len(scans) if scans else None),
            "scope": (
                "join-key coverage only; exact-retest outcomes are "
                "POST_DECISION diagnostics and are not score inputs"
            ),
        },
        "simultaneous_combinations": dict(
            sorted(
                combinations.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
    }


def _print_text(report: dict[str, Any]) -> None:
    total = report["total"]
    print(f"Shadow candidate DB: {report['db']}")
    print(
        "decision_scans={decision_scans} candidate_observations="
        "{candidate_observations} independent_opportunities="
        "{independent_opportunities} simultaneous_scans="
        "{simultaneous_candidate_scans} unselected_opportunities="
        "{unselected_independent_opportunities}".format(**total)
    )
    print(
        "group | observations | opportunities | production_selected | "
        "quality_selected | unselected_opportunities | "
        "mean_expected_net_r | mean_quality_conservative_net_r"
    )
    for key, row in report["candidate_groups"].items():
        net = row["mean_expected_net_r"]
        net_text = "n/a" if net is None else f"{net:+.4f}"
        quality_net = row["mean_quality_conservative_net_r"]
        quality_net_text = "n/a" if quality_net is None else f"{quality_net:+.4f}"
        print(
            f"{key} | {row['candidate_observations']} | "
            f"{row['independent_opportunities']} | "
            f"{row['production_selected_observations']} | "
            f"{row['quality_selected_observations']} | "
            f"{row['unselected_independent_opportunities']} | {net_text} | "
            f"{quality_net_text}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report shadow multi-candidate population"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=SHADOW_CANDIDATE_LEDGER_DB_PATH,
    )
    parser.add_argument("--since", type=_parse_utc)
    parser.add_argument("--symbol")
    parser.add_argument("--trigger")
    parser.add_argument("--side")
    parser.add_argument("--status")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report(
        args.db,
        since=args.since,
        symbol=args.symbol,
        trigger=args.trigger,
        side=args.side,
        status=args.status,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
