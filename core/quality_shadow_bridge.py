"""Bridge shadow-candidate payloads to v2 quality-ledger diagnostic rows.

The bridge is pure and non-executing.  It copies every candidate before
annotating it, scores each copy independently so one malformed candidate can
never suppress the diagnostics of its siblings, and returns plain rows for
``ShadowCandidateLedger.record_quality_ranking``.  Nothing it returns can
select, veto, resize, or reorder a production entry: the caller runs in the
fail-open shadow worker after the live decision is already final.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from core.hierarchical_quality_score import (
    LiveHierarchicalQualityScorer,
    rank_quality_candidates,
)
from core.shadow_candidate_ledger import (
    stable_candidate_opportunity_id,
    trigger_signature,
)


def _utc(value: Any, *, name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(
                str(value).strip().replace("Z", "+00:00")
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _nonnegative_finite(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0.0:
        return None
    return parsed


# The ledger column on the left is filled from the scorer/ranker key on the
# right.  ``opportunity_id`` and correlation penalty are added separately.
_LEDGER_FIELD_SOURCES = (
    ("quality_status", "quality_status"),
    ("quality_tp_probability", "quality_tp1_probability"),
    ("quality_tp_lcb", "quality_tp1_probability_lcb"),
    ("quality_tp_ucb", "quality_tp1_probability_ucb"),
    ("quality_expected_net_r", "quality_expected_net_r"),
    ("quality_conservative_net_r", "quality_conservative_net_r"),
    ("quality_net_uncertainty_r", "quality_expected_net_r_std_error"),
    ("quality_effective_n", "quality_hierarchy_support"),
    ("quality_backoff_path", "quality_hierarchy_path"),
    ("quality_stop_atr_h1", "quality_stop_atr_h1"),
    ("quality_feature_missing", "quality_feature_missing"),
    ("quality_rank_score_r", "quality_rank_score_r"),
    ("quality_rank_position", "quality_rank_position"),
    ("quality_selected", "quality_selected"),
)


def build_quality_ledger_rows(
    *,
    symbol: str,
    candidates: Sequence[Mapping[str, Any]],
    observed_at_utc: Any,
    decision_bar_close: Any,
    scorer: LiveHierarchicalQualityScorer,
    atr_h1_14: Optional[float] = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Score and rank candidate copies; never mutate or gate the originals.

    ``atr_h1_14`` is the decision-time H1 ATR from the same closed-bars view
    the candidates were generated from; it only backfills the stop-distance
    feature when the candidate itself carries no ATR context.
    """

    observed = _utc(observed_at_utc, name="observed_at_utc")
    decision = (
        _utc(decision_bar_close, name="decision_bar_close")
        if decision_bar_close is not None
        else observed
    )
    injected_atr = _nonnegative_finite(atr_h1_14)

    annotated: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("every candidate must be a mapping")
        payload = dict(candidate)
        signature = str(
            payload.get("shadow_trigger_signature") or ""
        ).strip() or trigger_signature(payload)
        payload["opportunity_id"] = stable_candidate_opportunity_id(
            symbol,
            signature,
            decision.date(),
        )
        if injected_atr is not None and injected_atr > 0.0:
            payload.setdefault("atr_h1_14", injected_atr)
        penalty = _nonnegative_finite(
            payload.get("shadow_correlation_penalty_r")
        )
        if penalty is not None:
            payload["quality_correlation_penalty_r"] = penalty
        try:
            payload.update(
                scorer.score(
                    payload,
                    symbol=symbol,
                    decision_time_utc=decision,
                )
            )
        except Exception as exc:
            # One malformed candidate stays visible as UNSCORED instead of
            # taking down the whole scan's diagnostics.
            payload["quality_status"] = f"UNSCORED:{type(exc).__name__}"
        annotated.append(payload)

    ranked = rank_quality_candidates(annotated)
    rows = [
        {
            "opportunity_id": row["opportunity_id"],
            "quality_correlation_penalty_r": row.get(
                "quality_correlation_penalty_r"
            ),
            **{
                ledger_field: row.get(source_field)
                for ledger_field, source_field in _LEDGER_FIELD_SOURCES
            },
        }
        for row in ranked
    ]
    return scorer.profile.profile_id, rows


__all__ = ["build_quality_ledger_rows"]
