"""Read-only plan-level report for the causal shadow tick watcher."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable
from urllib.parse import quote

from config import SHADOW_TICK_WATCHER_DB_PATH


def _parse_utc(value: str) -> datetime:
    normalized = str(value).strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(percentile)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _open_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.resolve()
    encoded = quote(resolved.as_posix(), safe="/:")
    connection = sqlite3.connect(
        f"file:{encoded}?mode=ro",
        uri=True,
        timeout=5.0,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def _metric(rows: Iterable[sqlite3.Row]) -> dict[str, Any]:
    population = list(rows)
    complete = [row for row in population if int(row["data_complete"])]
    touched = [
        row
        for row in complete
        if row["first_strict_touch_msc"] is not None
    ]
    time_to_touch = [
        max(
            0.0,
            (
                int(row["first_strict_touch_msc"])
                - int(row["activation_tick_msc"])
            )
            / 1000.0,
        )
        for row in touched
    ]
    missed = sum(int(row["missed_by_cadence"] or 0) for row in complete)
    policy_missed = sum(
        int(row["missed_by_policy"] or 0) for row in complete
    )
    recoverable = sum(
        int(row["recoverable_by_limit"] or 0) for row in complete
    )
    return {
        "plans": len(population),
        "finalized": sum(int(row["finalized"]) for row in population),
        "complete": len(complete),
        "unknown_coverage": len(population) - len(complete),
        "tick_touched": len(touched),
        "scan_touched": sum(
            row["first_scan_touch_msc"] is not None for row in complete
        ),
        "policy_touched": sum(
            row["first_policy_touch_msc"] is not None for row in complete
        ),
        "missed_by_cadence": missed,
        "missed_by_policy": policy_missed,
        "limit_recoverable": recoverable,
        "actual_pending_fills": sum(
            str(row["status"] or "").upper() == "FILLED"
            and str(row["terminal_reason"] or "")
            == "broker pending LIMIT filled"
            for row in population
        ),
        "ambiguous_gaps": sum(
            row["first_gap_msc"] is not None for row in complete
        ),
        "cadence_miss_rate": (
            missed / len(touched) if touched else None
        ),
        "policy_miss_rate": (
            policy_missed / len(touched) if touched else None
        ),
        "limit_recovery_rate": (
            recoverable / len(touched) if touched else None
        ),
        "time_to_touch_p50_sec": (
            median(time_to_touch) if time_to_touch else None
        ),
        "time_to_touch_p95_sec": _percentile(time_to_touch, 0.95),
    }


def build_report(
    path: Path,
    *,
    since: datetime | None = None,
    symbol: str | None = None,
    trigger: str | None = None,
    finalized_only: bool = True,
    group_by: str = "symbol",
) -> dict[str, Any]:
    where: list[str] = []
    params: list[Any] = []
    if finalized_only:
        where.append("finalized = 1")
    if since is not None:
        where.append("activated_at_utc >= ?")
        params.append(since.astimezone(timezone.utc).isoformat())
    if symbol:
        where.append("symbol = ?")
        params.append(symbol.strip().upper())
    if trigger:
        where.append("trigger_kind = ?")
        params.append(trigger.strip())
    clause = " WHERE " + " AND ".join(where) if where else ""
    query = (
        "SELECT * FROM plans"
        + clause
        + " ORDER BY activated_at_utc, plan_id"
    )
    with _open_read_only(path) as connection:
        rows = list(connection.execute(query, params))

    field = None if group_by == "none" else group_by
    groups: dict[str, list[sqlite3.Row]] = {}
    if field is not None:
        for row in rows:
            key = str(row[field] or "UNKNOWN")
            groups.setdefault(key, []).append(row)
    return {
        "db": str(path.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "filters": {
            "since": since.isoformat() if since else None,
            "symbol": symbol,
            "trigger": trigger,
            "finalized_only": finalized_only,
            "group_by": group_by,
        },
        "total": _metric(rows),
        "groups": {
            key: _metric(group_rows)
            for key, group_rows in sorted(groups.items())
        },
    }


def _fmt_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _print_text(report: dict[str, Any]) -> None:
    total = report["total"]
    print(f"Shadow tick DB: {report['db']}")
    print(
        "plans={plans} complete={complete} unknown={unknown_coverage} "
        "tick_touched={tick_touched} scan_touched={scan_touched} "
        "cadence_missed={missed_by_cadence} ({miss_rate}) "
        "limit_recoverable={limit_recoverable} ({recovery_rate}) "
        "actual_pending_fills={actual_pending_fills}".format(
            **total,
            miss_rate=_fmt_rate(total["cadence_miss_rate"]),
            recovery_rate=_fmt_rate(total["limit_recovery_rate"]),
        )
    )
    if not report["groups"]:
        return
    print(
        "group | plans | complete | tick | scan | cadence_miss | "
        "limit_recoverable | pending_fills | unknown"
    )
    for key, row in report["groups"].items():
        print(
            f"{key} | {row['plans']} | {row['complete']} | "
            f"{row['tick_touched']} | {row['scan_touched']} | "
            f"{row['missed_by_cadence']} "
            f"({_fmt_rate(row['cadence_miss_rate'])}) | "
            f"{row['limit_recoverable']} "
            f"({_fmt_rate(row['limit_recovery_rate'])}) | "
            f"{row['actual_pending_fills']} | "
            f"{row['unknown_coverage']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure strict tick touches missed by the real minute loop. "
            "Coverage-incomplete plans stay UNKNOWN."
        )
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=SHADOW_TICK_WATCHER_DB_PATH,
    )
    parser.add_argument("--since", type=_parse_utc)
    parser.add_argument("--symbol")
    parser.add_argument("--trigger")
    parser.add_argument(
        "--group-by",
        choices=("symbol", "trigger_kind", "setup_tf", "none"),
        default="symbol",
    )
    parser.add_argument("--include-active", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.db.exists():
        parser.error(f"shadow tick database does not exist: {args.db}")
    report = build_report(
        args.db,
        since=args.since,
        symbol=args.symbol,
        trigger=args.trigger,
        finalized_only=not args.include_active,
        group_by=args.group_by,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
