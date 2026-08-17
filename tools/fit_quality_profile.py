"""Fit and freeze a hierarchical quality profile from training rows.

Offline research CLI.  It reads `quality-training-row/v1` JSONL produced by
`tools/export_quality_training_rows.py`, fits the deterministic
`hierarchical-quality-profile-v1` model, and writes canonical JSON.  The
output is a diagnostic artifact: deploying it can annotate and rank shadow
candidates but can never gate a production entry.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.hierarchical_quality_score import (
    R_BASIS_NET,
    fit_quality_profile,
)


class FitError(ValueError):
    """Raised when the training input violates the fitting contract."""


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        # utf-8-sig: rows produced on Windows may carry a BOM.
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise FitError(f"cannot read training rows {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FitError(
                f"invalid JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise FitError(f"row at {path}:{line_number} is not an object")
        rows.append(value)
    if not rows:
        raise FitError(f"no training rows in {path}")
    return rows


def _apply_partial_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Treat TP1 and net-R labels as independently censored.

    The exporter's strict ``label_complete`` requires both labels.  A closed
    idea whose TP1 leg finished on TP/SL carries a final TP1 label even when
    the per-leg risk breakdown (and therefore net R) is unavailable, so this
    mode re-admits such rows with only the labels they actually have.
    """

    relaxed: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        has_any_label = any(
            copied.get(name) is not None
            for name in (
                "label_tp1_hit",
                "label_realized_net_r",
                "label_exact_retest_seconds",
            )
        )
        copied["label_complete"] = has_any_label
        relaxed.append(copied)
    return relaxed


def fit_profile_from_rows(
    rows: Sequence[dict[str, Any]],
    *,
    allow_partial_labels: bool = False,
    profile_id: str | None = None,
    r_label_basis: str = R_BASIS_NET,
    outside_stop_atr_penalty_r: float = 0.0,
) -> Any:
    prepared = list(rows)
    if allow_partial_labels:
        prepared = _apply_partial_labels(prepared)
    if not math.isfinite(outside_stop_atr_penalty_r) or (
        outside_stop_atr_penalty_r < 0.0
    ):
        raise FitError("outside_stop_atr_penalty_r must be nonnegative")
    try:
        return fit_quality_profile(
            prepared,
            profile_id=profile_id,
            r_label_basis=r_label_basis,
            outside_stop_atr_penalty_r=outside_stop_atr_penalty_r,
        )
    except ValueError as exc:
        raise FitError(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", required=True, type=Path,
        help="quality-training-row/v1 JSONL",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="frozen hierarchical-quality-profile-v1 JSON",
    )
    parser.add_argument(
        "--profile-id", default=None,
        help="optional explicit profile id (default: content hash)",
    )
    parser.add_argument(
        "--r-label-basis", default=R_BASIS_NET,
        help="NET_R_AFTER_BROKER_COSTS or GROSS_R_BEFORE_COSTS",
    )
    parser.add_argument(
        "--outside-stop-atr-penalty-r", type=float, default=0.0,
        help="ranking penalty in R for stops outside the 0.75-1.50 ATR band",
    )
    parser.add_argument(
        "--allow-partial-labels", action="store_true",
        help=(
            "treat TP1 and net-R labels as independently censored instead "
            "of requiring the exporter's strict label_complete flag"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        rows = _read_rows(args.rows)
        profile = fit_profile_from_rows(
            rows,
            allow_partial_labels=args.allow_partial_labels,
            profile_id=args.profile_id,
            r_label_basis=args.r_label_basis,
            outside_stop_atr_penalty_r=args.outside_stop_atr_penalty_r,
        )
    except FitError as exc:
        raise SystemExit(f"quality profile fit failed: {exc}") from exc
    payload = profile.to_mapping()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=1,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    tp_n = profile.tp1_model.sample_count if profile.tp1_model else 0
    net_n = profile.net_r_model.sample_count if profile.net_r_model else 0
    print(
        json.dumps(
            {
                "profile_id": profile.profile_id,
                "profile_sha256": profile.profile_sha256,
                "rows_in": len(rows),
                "support_global": profile.support_count("global", "*"),
                "tp1_samples": tp_n,
                "net_r_samples": net_n,
                "r_label_basis": profile.r_label_basis,
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
