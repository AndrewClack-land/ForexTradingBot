"""Structured conditional attribution for narrative factor vectors.

The tables produced here describe associations inside the candidates selected
by the current production formula.  They deliberately do not claim that a
factor caused an outcome or that changing its weight would preserve the same
candidate population.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from core.narrative_scoring import FACTOR_DEFINITIONS

from .metrics import aggregate_setup_metrics


ATTRIBUTION_SCHEMA = "narrative-factor-attribution/v1"
_PRIOR_STRENGTH = 100


def _event_id(row: Mapping[str, Any]) -> str:
    identity = {
        "symbol": row.get("symbol"),
        "decision_time": row.get("decision_time"),
        "side": row.get("side"),
        "trigger_kind": row.get("trigger_kind"),
        "trigger_reason": row.get("trigger_reason"),
    }
    raw = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def factor_vector_fields(
    factor_vector: Any,
) -> dict[str, Any]:
    """Flatten score-level fields while retaining the complete vector."""

    vector = factor_vector if isinstance(factor_vector, Mapping) else {}
    factors = [
        dict(item)
        for item in vector.get("factors", ())
        if isinstance(item, Mapping)
    ]
    return {
        "factor_schema": vector.get("schema"),
        "factor_bias": vector.get("bias"),
        "factor_score_long": vector.get("score_long"),
        "factor_score_short": vector.get("score_short"),
        "factor_score_delta": vector.get(
            "score_delta_long_minus_short"
        ),
        "factor_base_margin": vector.get("base_margin"),
        "factor_margin_long": vector.get("margin_long"),
        "factor_margin_short": vector.get("margin_short"),
        "factor_fvg_side": vector.get("fvg_side"),
        "factor_aligned_count": sum(
            bool(item.get("present"))
            and item.get("vote_side") == vector.get("bias")
            for item in factors
        ),
        "factor_opposed_count": sum(
            bool(item.get("present"))
            and item.get("vote_side") in {"LONG", "SHORT"}
            and item.get("vote_side") != vector.get("bias")
            for item in factors
        ),
        "factor_pivotal_count": sum(
            bool(item.get("pivotal_without_factor"))
            for item in factors
        ),
        "factor_vector": dict(vector) if vector else None,
    }


def _relation(vote: Mapping[str, Any], side: Any) -> tuple[str, int]:
    if not bool(vote.get("present")):
        return "ABSENT", 0
    vote_side = str(vote.get("vote_side") or "NEUTRAL").upper()
    candidate_side = str(side or "NEUTRAL").upper()
    if vote_side == candidate_side and candidate_side in {"LONG", "SHORT"}:
        return "ALIGNED", 1
    if vote_side in {"LONG", "SHORT"}:
        return "OPPOSED", -1
    return "NEUTRAL", 0


def _factor_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    include_outcome: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        vector = row.get("factor_vector")
        if not isinstance(vector, Mapping):
            continue
        event_id = _event_id(row)
        for vote in vector.get("factors", ()):
            if not isinstance(vote, Mapping) or not vote.get("key"):
                continue
            relation, alignment = _relation(vote, row.get("side"))
            evidence = vote.get("evidence")
            evidence = dict(evidence) if isinstance(evidence, Mapping) else {}
            effective_weight = float(vote.get("effective_weight") or 0.0)
            item = {
                "attribution_schema": ATTRIBUTION_SCHEMA,
                "candidate_id": row.get("candidate_id"),
                "event_id": event_id,
                "fold_index": row.get("fold_index"),
                "symbol": row.get("symbol"),
                "decision_time": row.get("decision_time"),
                "candidate_side": row.get("side"),
                "trigger_kind": row.get("trigger_kind"),
                "factor_schema": vector.get("schema"),
                "factor_key": vote.get("key"),
                "factor_label": vote.get("label"),
                "present": bool(vote.get("present")),
                "vote_side": vote.get("vote_side"),
                "relation": relation,
                "alignment": alignment,
                "configured_weight": vote.get("configured_weight"),
                "effective_weight": effective_weight,
                "long_contribution": vote.get("long_contribution"),
                "short_contribution": vote.get("short_contribution"),
                "candidate_contribution": effective_weight * alignment,
                "pivotal_without_factor": bool(
                    vote.get("pivotal_without_factor")
                ),
                "bias_without_factor": vote.get("bias_without_factor"),
                "age_bars": (
                    evidence.get("age_bars")
                    if "age_bars" in evidence
                    else evidence.get("bars_ago")
                ),
                "evidence": evidence,
                "score_long": vector.get("score_long"),
                "score_short": vector.get("score_short"),
                "score_delta": vector.get(
                    "score_delta_long_minus_short"
                ),
                "selected_bias": vector.get("bias"),
                "base_margin": vector.get("base_margin"),
                "margin_long": vector.get("margin_long"),
                "margin_short": vector.get("margin_short"),
                "fvg_side": vector.get("fvg_side"),
                "vol_r": row.get("vol_r"),
                "vol_regime": row.get("vol_regime"),
                "vol_em_1d": row.get("vol_em_1d"),
                "vol_tp1_em_ratio": row.get("vol_tp1_em_ratio"),
            }
            if include_outcome:
                decision_time = str(row.get("decision_time") or "")
                item.update(
                    {
                        "setup_id": row.get("setup_id"),
                        "policy": row.get("policy"),
                        "entry_time": row.get("entry_time"),
                        "exit_time": row.get("exit_time"),
                        "status": row.get("status"),
                        "net_r": row.get("net_r"),
                        "pnl_amount": row.get("pnl_amount"),
                        "decision_year": decision_time[:4] or None,
                        "decision_quarter": _quarter(decision_time),
                    }
                )
            else:
                item.update(
                    {
                        "gate": row.get("gate"),
                        "gate_reason": row.get("gate_reason"),
                    }
                )
            output.append(item)
    return output


def _quarter(value: str) -> str | None:
    try:
        month = int(value[5:7])
        return f"{value[:4]}-Q{((month - 1) // 3) + 1}"
    except (TypeError, ValueError):
        return None


def candidate_factor_rows(
    candidates: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return _factor_rows(candidates, include_outcome=False)


def setup_factor_rows(
    setups: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return _factor_rows(setups, include_outcome=True)


def _ordered_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return aggregate_setup_metrics(
        sorted(
            rows,
            key=lambda row: (
                row.get("exit_time") is None,
                str(row.get("exit_time") or ""),
                str(row.get("entry_time") or ""),
                str(row.get("setup_id") or ""),
            ),
        )
    )


def _sample_quality(metrics: Mapping[str, Any]) -> str:
    total = int(metrics.get("setups_total") or 0)
    wins = int(metrics.get("wins") or 0)
    losses = int(metrics.get("losses") or 0)
    if total >= 200 and wins >= 30 and losses >= 30:
        return "ROBUST"
    if total >= 50 and wins >= 10 and losses >= 10:
        return "LIMITED"
    return "SPARSE"


def factor_summary_rows(
    attribution: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate factors by requested market dimensions with shrinkage."""

    unique_setups: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in attribution:
        key = (str(row.get("policy")), str(row.get("setup_id")))
        unique_setups.setdefault(key, row)
    policy_priors = {
        policy: _ordered_metrics(
            [
                row
                for (row_policy, _), row in unique_setups.items()
                if row_policy == policy
            ]
        )
        for policy in sorted({key[0] for key in unique_setups})
    }

    groups: dict[
        tuple[str, str, str, str, str],
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    dimensions = (
        ("overall", lambda row: "ALL"),
        ("symbol", lambda row: row.get("symbol")),
        ("side", lambda row: row.get("candidate_side")),
        ("fold", lambda row: row.get("fold_index")),
        ("year", lambda row: row.get("decision_year")),
        ("quarter", lambda row: row.get("decision_quarter")),
        ("trigger", lambda row: row.get("trigger_kind")),
        ("vol_regime", lambda row: row.get("vol_regime")),
        ("fvg_side", lambda row: row.get("fvg_side")),
    )
    for row in attribution:
        for dimension, getter in dimensions:
            value = getter(row)
            if value is None:
                continue
            key = (
                str(row.get("policy")),
                str(row.get("factor_key")),
                str(row.get("relation")),
                dimension,
                str(value),
            )
            groups[key].append(row)

    output: list[dict[str, Any]] = []
    for (
        policy,
        factor_key,
        relation,
        dimension,
        dimension_value,
    ), rows in sorted(groups.items()):
        metrics = _ordered_metrics(rows)
        prior_expectancy = float(
            policy_priors[policy].get("expectancy_r") or 0.0
        )
        count = int(metrics.get("setups_total") or 0)
        expectancy = float(metrics.get("expectancy_r") or 0.0)
        shrunken = (
            count * expectancy + _PRIOR_STRENGTH * prior_expectancy
        ) / (count + _PRIOR_STRENGTH)
        first = rows[0]
        output.append(
            {
                "attribution_schema": ATTRIBUTION_SCHEMA,
                "policy": policy,
                "factor_key": factor_key,
                "factor_label": first.get("factor_label"),
                "configured_weight": first.get("configured_weight"),
                "relation": relation,
                "dimension": dimension,
                "dimension_value": dimension_value,
                "setups": count,
                "net_r": metrics.get("net_r"),
                "expectancy_r": metrics.get("expectancy_r"),
                "shrunken_expectancy_r": shrunken,
                "prior_strength": _PRIOR_STRENGTH,
                "win_rate": metrics.get("win_rate"),
                "profit_factor": metrics.get("profit_factor"),
                "max_drawdown_r": metrics.get("max_drawdown_r"),
                "average_win_r": metrics.get("average_win_r"),
                "average_loss_r": metrics.get("average_loss_r"),
                "wins": metrics.get("wins"),
                "losses": metrics.get("losses"),
                "sample_quality": _sample_quality(metrics),
                "interpretation": "conditional_association_not_causal",
            }
        )
    return output


def attribution_coverage(
    *,
    candidates: Sequence[Mapping[str, Any]],
    setups: Sequence[Mapping[str, Any]],
    candidate_factors: Sequence[Mapping[str, Any]],
    setup_factors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_ids = {
        str(row.get("candidate_id"))
        for row in candidate_factors
        if row.get("candidate_id")
    }
    setup_ids = {
        str(row.get("setup_id"))
        for row in setup_factors
        if row.get("setup_id")
    }
    observed_factor_keys = sorted(
        {
            str(row.get("factor_key"))
            for row in candidate_factors
            if row.get("factor_key")
        }
    )
    expected_factor_keys = sorted(
        definition.key for definition in FACTOR_DEFINITIONS
    )
    return {
        "schema": ATTRIBUTION_SCHEMA,
        "mode": "conditional-production-selection",
        "candidate_population": "raw_enter_bias_plus_detected_trigger",
        "outcome_population": "executed_and_filled_setups",
        "candidate_rows": len(candidates),
        "attributed_candidates": len(candidate_ids),
        "candidate_factor_rows": len(candidate_factors),
        "setup_rows": len(setups),
        "attributed_setups": len(setup_ids),
        "setup_factor_rows": len(setup_factors),
        "factor_keys": observed_factor_keys,
        "expected_factor_keys": expected_factor_keys,
        "complete_candidate_vectors": (
            not candidates
            or (
                observed_factor_keys == expected_factor_keys
                and len(candidate_factors)
                == len(candidates) * len(expected_factor_keys)
            )
        ),
        "complete_setup_vectors": (
            not setups
            or (
                observed_factor_keys == expected_factor_keys
                and len(setup_factors)
                == len(setups) * len(expected_factor_keys)
            )
        ),
        "selection_bias_warning": (
            "Candidate attribution starts only after the baseline score "
            "produces a directional bias and a production trigger is found. "
            "Outcome attribution is narrower again: only setups that survive "
            "execution gates and fill have net R. These rows support "
            "conditional diagnostics, not unbiased replacement weights."
        ),
    }
