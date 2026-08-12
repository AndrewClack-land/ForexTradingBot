"""Non-executing candidate ranking by net R and portfolio correlation.

The ranker is deliberately pure. It receives frozen candidate payloads,
versioned cost/correlation profiles and a frozen shadow scorer. Its output is
telemetry only and cannot place, cancel, resize or veto an order.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from backtest.cost_model import CostModelError, CostProfile, estimate_cost_r
from core.shadow_candidate_ledger import (
    stable_candidate_opportunity_id,
    trigger_signature,
)


CORRELATION_PROFILE_SCHEMA = "portfolio-correlation-profile-v1"


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


def _finite(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class PortfolioCorrelationProfile:
    profile_id: str
    profile_sha256: str
    trained_start_utc: datetime
    trained_end_utc: datetime
    return_timeframe: str
    method: str
    shrinkage: float
    penalty_scale_r: float
    max_penalty_r: float
    correlations: Mapping[str, Mapping[str, float]]

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "PortfolioCorrelationProfile":
        if str(payload.get("schema") or "") != CORRELATION_PROFILE_SCHEMA:
            raise ValueError(
                f"correlation profile schema must be "
                f"{CORRELATION_PROFILE_SCHEMA}"
            )
        profile_id = str(payload.get("profile_id") or "").strip()
        timeframe = str(payload.get("return_timeframe") or "").strip()
        method = str(payload.get("method") or "").strip()
        if not profile_id or not timeframe or not method:
            raise ValueError(
                "profile_id, return_timeframe and method are required"
            )
        start = _utc(
            payload.get("trained_start_utc"),
            name="trained_start_utc",
        )
        end = _utc(
            payload.get("trained_end_utc"),
            name="trained_end_utc",
        )
        if end <= start:
            raise ValueError("trained_end_utc must follow trained_start_utc")
        shrinkage = _finite(payload.get("shrinkage"), name="shrinkage")
        penalty_scale = _finite(
            payload.get("penalty_scale_r"),
            name="penalty_scale_r",
        )
        max_penalty = _finite(
            payload.get("max_penalty_r"),
            name="max_penalty_r",
        )
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0,1]")
        if penalty_scale < 0.0 or max_penalty < 0.0:
            raise ValueError("correlation penalties cannot be negative")
        raw_correlations = payload.get("correlations")
        if not isinstance(raw_correlations, Mapping):
            raise ValueError("correlations must be a mapping")
        correlations: dict[str, dict[str, float]] = {}
        for left, raw_row in raw_correlations.items():
            if not isinstance(raw_row, Mapping):
                raise ValueError("every correlation row must be a mapping")
            left_key = str(left).strip().upper()
            correlations[left_key] = {}
            for right, raw_value in raw_row.items():
                right_key = str(right).strip().upper()
                value = _finite(
                    raw_value,
                    name=f"correlations.{left_key}.{right_key}",
                )
                if not -1.0 <= value <= 1.0:
                    raise ValueError("correlations must be in [-1,1]")
                correlations[left_key][right_key] = value
        for left, row in correlations.items():
            for right, value in row.items():
                reverse = correlations.get(right, {}).get(left)
                if reverse is None or not math.isclose(
                    value,
                    reverse,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"correlation matrix is not symmetric: "
                        f"{left}/{right}"
                    )
        return cls(
            profile_id=profile_id,
            profile_sha256=_canonical_hash(dict(payload)),
            trained_start_utc=start,
            trained_end_utc=end,
            return_timeframe=timeframe,
            method=method,
            shrinkage=shrinkage,
            penalty_scale_r=penalty_scale,
            max_penalty_r=max_penalty,
            correlations=correlations,
        )

    @classmethod
    def load(
        cls,
        path: Path | str,
    ) -> "PortfolioCorrelationProfile":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("correlation profile root must be an object")
        return cls.from_mapping(payload)

    def correlation(self, left: str, right: str) -> float:
        left_key = str(left).strip().upper()
        right_key = str(right).strip().upper()
        if left_key == right_key:
            return 1.0
        try:
            return float(self.correlations[left_key][right_key])
        except KeyError as exc:
            raise ValueError(
                f"correlation profile has no pair "
                f"{left_key}/{right_key}"
            ) from exc


def _direction(side: Any) -> int:
    key = str(side or "").strip().upper()
    if key == "LONG":
        return 1
    if key == "SHORT":
        return -1
    raise ValueError("side must be LONG or SHORT")


def correlation_penalty_r(
    profile: PortfolioCorrelationProfile,
    *,
    symbol: str,
    side: str,
    active_exposures: Mapping[str, str],
    decision_time_utc: datetime,
) -> float:
    decision = _utc(decision_time_utc, name="decision_time_utc")
    if profile.trained_end_utc > decision:
        raise ValueError(
            "correlation profile is future-dated for this decision"
        )
    direction = _direction(side)
    concentration = 0.0
    for active_symbol, active_side in active_exposures.items():
        rho = profile.correlation(symbol, active_symbol)
        aligned = rho * direction * _direction(active_side)
        concentration += max(0.0, aligned)
    return min(
        profile.max_penalty_r,
        profile.penalty_scale_r * concentration,
    )


def build_shadow_rankings(
    *,
    symbol: str,
    candidates: Sequence[Mapping[str, Any]],
    observed_at_utc: datetime,
    decision_bar_close: Optional[datetime],
    scorer: Any,
    cost_profile: CostProfile,
    correlation_profile: PortfolioCorrelationProfile,
    active_exposures: Mapping[str, str],
) -> tuple[str, list[dict[str, Any]]]:
    """Score candidates without returning an execution action."""

    observed = _utc(observed_at_utc, name="observed_at_utc")
    decision = (
        _utc(decision_bar_close, name="decision_bar_close")
        if decision_bar_close is not None
        else observed
    )
    model_id = _canonical_hash({
        "scorer_model_id": str(getattr(scorer, "model_id", "")),
        "cost_profile_sha256": cost_profile.profile_sha256,
        "correlation_profile_sha256": (
            correlation_profile.profile_sha256
        ),
    })[:24]
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        payload = dict(candidate)
        signature = str(
            payload.get("shadow_trigger_signature") or ""
        ).strip() or trigger_signature(payload)
        opportunity_id = stable_candidate_opportunity_id(
            symbol,
            signature,
            decision.date(),
        )
        score = scorer.score(payload, symbol=symbol)
        trigger_in_vocab = bool(score.get("shadow_trigger_in_vocab"))
        row: dict[str, Any] = {
            "opportunity_id": opportunity_id,
            "ranking_status": "UNMODELED_TRIGGER",
            "expected_gross_r": None,
            "estimated_cost_r": None,
            "expected_net_r": None,
            "correlation_penalty_r": None,
            "ranking_score": None,
            "rank_position": None,
            "selected": False,
        }
        if not trigger_in_vocab or score.get("shadow_score") is None:
            rows.append(row)
            continue
        try:
            gross = _finite(
                score["shadow_score"],
                name="shadow_score",
            )
            estimate = estimate_cost_r(
                cost_profile,
                symbol=symbol,
                side=str(payload.get("side") or ""),
                entry_price=payload.get("entry_price"),
                stop_price=payload.get("stop_price"),
                rollover_unit_count=0.0,
            )
            penalty = correlation_penalty_r(
                correlation_profile,
                symbol=symbol,
                side=str(payload.get("side") or ""),
                active_exposures=active_exposures,
                decision_time_utc=decision,
            )
        except (CostModelError, ValueError) as exc:
            row["ranking_status"] = (
                f"UNAVAILABLE:{type(exc).__name__}:{exc}"
            )
            rows.append(row)
            continue
        expected_net = gross - estimate.total_cost_r
        row.update({
            "ranking_status": "SCORED",
            "expected_gross_r": gross,
            "estimated_cost_r": estimate.total_cost_r,
            "expected_net_r": expected_net,
            "correlation_penalty_r": penalty,
            "ranking_score": expected_net - penalty,
        })
        rows.append(row)

    scored = sorted(
        (row for row in rows if row["ranking_status"] == "SCORED"),
        key=lambda row: (
            -float(row["ranking_score"]),
            -float(row["expected_net_r"]),
            str(row["opportunity_id"]),
        ),
    )
    for rank, row in enumerate(scored, start=1):
        row["rank_position"] = rank
        row["selected"] = rank == 1
    return model_id, rows


__all__ = [
    "CORRELATION_PROFILE_SCHEMA",
    "PortfolioCorrelationProfile",
    "build_shadow_rankings",
    "correlation_penalty_r",
]
