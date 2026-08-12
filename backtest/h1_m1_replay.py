"""Paired causal replay for the research-only H1 context to M1 trigger arm.

The callable labels each frozen H1 context twice on the same M1 stream:

* baseline mirrors the existing backtest entry rule: the first M1 open
  strictly after the H1 decision, before the entry deadline, and inside the
  signal's original entry range;
* m1_reclaim delegates entry selection to detect_m1_reclaim and therefore
  enters only on the open after a completed reclaim candle.

Both arms use the same context-level outcome cutoff, intrabar policy, targets,
and stop. A no-entry arm contributes 0R only when its complete entry window was
observed; truncated data is censored rather than silently counted as a rejected
trade. The result is an independent technical-opportunity study, not an
operational portfolio replay: active-position, cooldown, daily limit, and
cross-symbol arbitration gates are intentionally out of scope.

Nothing in this module is imported by live code or changes production config.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import pandas as pd

from .cost_model import CostProfile, estimate_realized_cost_r
from .h1_m1_trigger import H1Context, M1ReclaimConfig, detect_m1_reclaim
from .metrics import aggregate_setup_metrics
from .simulator import (
    DEFAULT_WEIGHTS,
    IntrabarPolicy,
    LegOutcome,
    SetupOutcome,
    simulate_split_outcome,
)


H1_M1_REPLAY_SCHEMA = "h1-context-m1-paired-replay/v2"
_OHLC = ("open", "high", "low", "close")
_ONE_MINUTE = pd.Timedelta(minutes=1)
_ALLOWED_PARENT_TRIGGERS = frozenset(
    {"order_block_1h", "rejection_block_1h"}
)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _duration(value: Any, *, name: str) -> pd.Timedelta:
    try:
        parsed = pd.Timedelta(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a pandas-compatible duration"
        ) from exc
    if parsed <= pd.Timedelta(0):
        raise ValueError(f"{name} must be positive")
    return parsed


@dataclass(frozen=True)
class H1M1ReplayConfig:
    """Explicit research assumptions for the paired arm.

    outcome_horizon is measured from the frozen H1 decision, not from each
    arm's fill. That gives both policies the same information horizon and
    prevents a later M1 entry from receiving extra future bars.
    """

    outcome_horizon: pd.Timedelta
    trigger: M1ReclaimConfig = field(default_factory=M1ReclaimConfig)
    intrabar_policy: IntrabarPolicy = "stop-first"
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outcome_horizon",
            _duration(self.outcome_horizon, name="outcome_horizon"),
        )
        if self.intrabar_policy not in {"stop-first", "tp-first"}:
            raise ValueError(
                "intrabar_policy must be 'stop-first' or 'tp-first'"
            )
        weights = tuple(float(value) for value in self.weights)
        if len(weights) != 3 or any(value <= 0 for value in weights):
            raise ValueError("weights must contain three positive values")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("weights must sum to 1.0")
        object.__setattr__(self, "weights", weights)

    def audit_payload(self) -> dict[str, Any]:
        return {
            "outcome_horizon": self.outcome_horizon.isoformat(),
            "intrabar_policy": self.intrabar_policy,
            "weights": list(self.weights),
            "trigger": {
                "min_penetration_fraction": (
                    self.trigger.min_penetration_fraction
                ),
                "max_entry_extension_r": self.trigger.max_entry_extension_r,
            },
        }


@dataclass(frozen=True)
class H1M1ReplayResult:
    """Deterministic audit rows plus paired aggregate diagnostics."""

    rows: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": H1_M1_REPLAY_SCHEMA,
            "summary": dict(self.summary),
            "rows": [dict(row) for row in self.rows],
        }


@dataclass(frozen=True)
class H1M1FoldWindow:
    """Explicit OOS boundary for one or more frozen H1 contexts."""

    fold_index: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __post_init__(self) -> None:
        if isinstance(self.fold_index, bool) or int(self.fold_index) < 0:
            raise ValueError("fold_index must be a nonnegative integer")
        start = pd.Timestamp(self.test_start)
        end = pd.Timestamp(self.test_end)
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("fold boundaries must be timezone-aware")
        start = start.tz_convert("UTC")
        end = end.tz_convert("UTC")
        if end <= start:
            raise ValueError("fold test_end must follow test_start")
        object.__setattr__(self, "fold_index", int(self.fold_index))
        object.__setattr__(self, "test_start", start)
        object.__setattr__(self, "test_end", end)


def _prepare_m1(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError("M1 data is empty")
    missing = sorted(set(_OHLC) - set(frame.columns))
    if missing:
        raise ValueError(f"M1 data is missing columns: {missing}")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        raise ValueError("M1 index must be timezone-aware")
    prepared = frame.loc[:, list(_OHLC)].copy()
    prepared.index = index.tz_convert("UTC")
    prepared = prepared.sort_index(kind="stable")
    for column in _OHLC:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )
    return prepared


def _validate_bounded_m1(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate only data causally visible to one frozen context."""

    if frame.index.has_duplicates:
        raise ValueError("M1 index contains duplicates in context window")
    for column in _OHLC:
        if not frame[column].map(math.isfinite).all():
            raise ValueError(
                f"M1 {column} contains non-finite values in context window"
            )
    impossible = (
        (frame["high"] < frame["low"])
        | (frame["high"] < frame[["open", "close"]].max(axis=1))
        | (frame["low"] > frame[["open", "close"]].min(axis=1))
    )
    if impossible.any():
        raise ValueError(
            "M1 data contains impossible OHLC candles in context window"
        )
    return frame


def _covers(frame: pd.DataFrame, instant: pd.Timestamp) -> bool:
    return bool(
        not frame.empty
        and pd.Timestamp(frame.index[-1]) + _ONE_MINUTE >= instant
    )


def _is_contiguous_m1(
    frame: pd.DataFrame,
    *,
    first_open: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> bool:
    """Require every M1 open in a causal interval.

    The research arm has no exchange calendar or authoritative outage feed,
    so any missing bar is UNKNOWN rather than an assumed flat/no-touch minute.
    """

    last_open = (
        pd.Timestamp(end_exclusive) - pd.Timedelta(nanoseconds=1)
    ).floor("min")
    first = pd.Timestamp(first_open).ceil("min")
    if last_open < first:
        return True
    window = frame.loc[(frame.index >= first) & (frame.index <= last_open)]
    expected = int((last_open - first) / _ONE_MINUTE) + 1
    if len(window) != expected:
        return False
    return bool(
        pd.Timestamp(window.index[0]) == first
        and pd.Timestamp(window.index[-1]) == last_open
        and (
            len(window) < 2
            or (window.index[1:] - window.index[:-1] == _ONE_MINUTE).all()
        )
    )


def _bounded_context_frame(
    frame: pd.DataFrame,
    *,
    decision_time: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Return one causal pre-decision close plus the bounded label window."""

    left = max(
        0,
        int(frame.index.searchsorted(decision_time, side="left")) - 1,
    )
    right = int(frame.index.searchsorted(cutoff, side="left"))
    return frame.iloc[left:right]


def _baseline_fill(
    context: H1Context,
    frame: pd.DataFrame,
    *,
    deadline: pd.Timestamp,
) -> dict[str, Any]:
    """Mirror strategy_runner._find_fill without a backtest config."""

    post = frame.loc[
        (frame.index > context.decision_time) & (frame.index < deadline)
    ]
    invalid_geometry_seen = False
    for entry_time, row in post.iterrows():
        entry = float(row["open"])
        if entry < context.entry_min or entry > context.entry_max:
            continue
        valid = (
            context.stop < entry
            and all(entry < target for target in context.tp_prices)
            if context.side == "LONG"
            else context.stop > entry
            and all(entry > target for target in context.tp_prices)
        )
        if not valid:
            invalid_geometry_seen = True
            continue
        return {
            "disposition": "FILLED",
            "reason": "first post-decision in-range M1 open",
            "entry_time": pd.Timestamp(entry_time),
            "entry_price": entry,
        }
    if not _covers(frame, deadline):
        return {
            "disposition": "CENSORED_ENTRY_WINDOW",
            "reason": "M1 data ends before the exclusive entry deadline",
        }
    reason = (
        "in-range M1 open had invalid stop/target geometry"
        if invalid_geometry_seen
        else "entry range expired"
    )
    return {"disposition": "NO_FILL", "reason": reason}


def _force_close(
    outcome: SetupOutcome,
    *,
    price: float,
    timestamp: pd.Timestamp,
) -> SetupOutcome:
    if outcome.status == "CLOSED":
        return outcome
    risk = abs(outcome.entry - outcome.initial_stop)
    direction_pnl = (
        price - outcome.entry
        if outcome.side == "LONG"
        else outcome.entry - price
    )
    legs = tuple(
        leg
        if leg.exit_reason != "OPEN"
        else LegOutcome(
            tp_index=leg.tp_index,
            weight=leg.weight,
            exit_reason="TIME",
            exit_price=float(price),
            r_multiple=float(direction_pnl / risk * leg.weight),
            exit_time=timestamp,
        )
        for leg in outcome.legs
    )
    return SetupOutcome(
        side=outcome.side,
        entry=outcome.entry,
        initial_stop=outcome.initial_stop,
        status="CLOSED",
        net_r=float(sum(leg.r_multiple for leg in legs)),
        legs=legs,
        tp_hits=outcome.tp_hits,
        moved_to_be=outcome.moved_to_be,
        ambiguous_bars=outcome.ambiguous_bars,
        bars_processed=outcome.bars_processed,
        exit_time=timestamp,
    )


def _label_fill(
    *,
    context: H1Context,
    entry_time: pd.Timestamp,
    entry_price: float,
    frame: pd.DataFrame,
    cutoff: pd.Timestamp,
    config: H1M1ReplayConfig,
) -> dict[str, Any]:
    # Only fully completed candles at or before the common cutoff are allowed.
    bars = frame.loc[
        (frame.index >= entry_time)
        & (frame.index + _ONE_MINUTE <= cutoff)
    ]
    if bars.empty:
        return {
            "label_status": "CENSORED",
            "label_reason": "no completed M1 outcome bar before cutoff",
        }
    try:
        outcome = simulate_split_outcome(
            side=context.side,
            entry=entry_price,
            stop=context.stop,
            tp_prices=context.tp_prices,
            bars=bars,
            weights=config.weights,
            intrabar_policy=config.intrabar_policy,
        )
    except (TypeError, ValueError) as exc:
        return {
            "label_status": "INVALID",
            "label_reason": f"outcome simulation failed: {exc}",
        }
    continuity_end = (
        pd.Timestamp(outcome.exit_time) + _ONE_MINUTE
        if outcome.status == "CLOSED" and outcome.exit_time is not None
        else cutoff
    )
    if not _is_contiguous_m1(
        frame,
        first_open=entry_time,
        end_exclusive=continuity_end,
    ):
        return {
            "label_status": "CENSORED",
            "label_reason": "missing M1 bar inside the observed outcome path",
        }
    forced = False
    if outcome.status == "OPEN":
        if not _covers(frame, cutoff):
            return {
                "label_status": "CENSORED",
                "label_reason": (
                    "M1 data ends before the common outcome cutoff"
                ),
            }
        last_open = pd.Timestamp(bars.index[-1])
        outcome = _force_close(
            outcome,
            price=float(bars.iloc[-1]["close"]),
            timestamp=last_open + _ONE_MINUTE,
        )
        forced = True
    if outcome.exit_time is None or not math.isfinite(float(outcome.net_r)):
        return {
            "label_status": "INVALID",
            "label_reason": "closed outcome has no finite exit label",
        }
    return {
        "label_status": "FORCED_TIME" if forced else "CLOSED",
        "label_reason": (
            "forced at common context cutoff"
            if forced
            else "terminal stop/target outcome"
        ),
        "net_r": float(outcome.net_r),
        "exit_time": outcome.exit_time.isoformat(),
        "tp_hits": list(outcome.tp_hits),
        "moved_to_be": bool(outcome.moved_to_be),
        "ambiguous_bars": int(outcome.ambiguous_bars),
        "bars_processed": int(outcome.bars_processed),
    }


def _strict_entry_after_trigger(
    *,
    context: H1Context,
    frame: pd.DataFrame,
    deadline: pd.Timestamp,
    trigger_time: pd.Timestamp,
    trigger_audit: Mapping[str, Any],
    config: H1M1ReplayConfig,
) -> dict[str, Any]:
    """Resolve entry strictly after the completed trigger timestamp."""

    strict_rows = frame.loc[
        (frame.index > trigger_time) & (frame.index < deadline)
    ]
    base = {
        **dict(trigger_audit),
        "entry_rule": "first_m1_open_strictly_after_trigger_time",
    }
    if strict_rows.empty:
        return {
            **base,
            "status": "NO_CAUSAL_ENTRY_BAR",
            "entry_time": None,
            "entry_price": None,
        }
    entry_time = pd.Timestamp(strict_rows.index[0])
    intervening = frame.loc[
        (frame.index >= trigger_time) & (frame.index < entry_time)
    ]
    for bar_open, row in intervening.iterrows():
        bar_close = pd.Timestamp(bar_open) + _ONE_MINUTE
        if context.side == "LONG":
            if float(row["low"]) <= context.stop:
                return {
                    **base,
                    "status": "INVALIDATED_STOP",
                    "invalidated_at": bar_close.isoformat(),
                }
            if float(row["high"]) >= context.tp_prices[0]:
                return {
                    **base,
                    "status": "INVALIDATED_TP1",
                    "invalidated_at": bar_close.isoformat(),
                }
        else:
            if float(row["high"]) >= context.stop:
                return {
                    **base,
                    "status": "INVALIDATED_STOP",
                    "invalidated_at": bar_close.isoformat(),
                }
            if float(row["low"]) <= context.tp_prices[0]:
                return {
                    **base,
                    "status": "INVALIDATED_TP1",
                    "invalidated_at": bar_close.isoformat(),
                }

    entry_price = float(strict_rows.iloc[0]["open"])
    geometry_valid = (
        context.stop < entry_price < context.tp_prices[0]
        if context.side == "LONG"
        else context.tp_prices[0] < entry_price < context.stop
    )
    if not geometry_valid:
        return {
            **base,
            "status": "ENTRY_GAP_INVALID",
            "entry_time": entry_time.isoformat(),
            "entry_price": entry_price,
        }
    risk = abs(context.planned_entry - context.stop)
    extension_r = abs(entry_price - context.planned_entry) / risk
    if extension_r > config.trigger.max_entry_extension_r:
        return {
            **base,
            "status": "ENTRY_EXTENSION_REJECT",
            "entry_time": entry_time.isoformat(),
            "entry_price": entry_price,
            "entry_extension_r": extension_r,
        }
    return {
        **base,
        "status": "TRIGGERED",
        "entry_time": entry_time.isoformat(),
        "entry_price": entry_price,
        "entry_extension_r": extension_r,
    }


def _m1_entry(
    context: H1Context,
    frame: pd.DataFrame,
    *,
    deadline: pd.Timestamp,
    config: H1M1ReplayConfig,
) -> dict[str, Any]:
    # The effective context prevents the detector from inspecting bars outside
    # the common outcome horizon when a caller supplies a longer entry TTL.
    effective_context = replace(context, expires_at=deadline)
    visible = frame.loc[frame.index < deadline]
    audit = detect_m1_reclaim(
        effective_context,
        visible,
        config=config.trigger,
    )
    status = str(audit.get("status") or "DATA_INVALID")
    trigger_time_raw = audit.get("trigger_time")
    if trigger_time_raw is not None and status in {
        "TRIGGERED",
        "ENTRY_GAP_INVALID",
        "ENTRY_EXTENSION_REJECT",
        "NO_CAUSAL_ENTRY_BAR",
    }:
        audit = _strict_entry_after_trigger(
            context=context,
            frame=visible,
            deadline=deadline,
            trigger_time=pd.Timestamp(trigger_time_raw),
            trigger_audit=audit,
            config=config,
        )
        status = str(audit["status"])
    if status == "TRIGGERED":
        return {
            "disposition": "FILLED",
            "reason": "completed M1 reclaim; entry on next M1 open",
            "entry_time": pd.Timestamp(audit["entry_time"]),
            "entry_price": float(audit["entry_price"]),
            "trigger_status": status,
            "trigger_time": audit.get("trigger_time"),
            "trigger_bar_open": audit.get("trigger_bar_open"),
            "penetration_fraction": audit.get("penetration_fraction"),
            "entry_extension_r": audit.get("entry_extension_r"),
            "entry_rule": audit.get("entry_rule"),
        }
    if status == "DATA_INVALID":
        disposition = "INVALID"
    elif status in {"NO_TRIGGER", "NO_CAUSAL_ENTRY_BAR"} and not _covers(
        frame,
        deadline,
    ):
        disposition = "CENSORED_ENTRY_WINDOW"
    else:
        disposition = "NO_FILL"
    return {
        "disposition": disposition,
        "reason": str(audit.get("reason") or status),
        "trigger_status": status,
        "trigger_time": audit.get("trigger_time"),
        "trigger_bar_open": audit.get("trigger_bar_open"),
        "penetration_fraction": audit.get("penetration_fraction"),
        "entry_extension_r": audit.get("entry_extension_r"),
        "entry_rule": audit.get("entry_rule"),
        "invalidated_at": audit.get("invalidated_at"),
    }


def _finish_arm(
    *,
    entry: Mapping[str, Any],
    context: H1Context,
    frame: pd.DataFrame,
    cutoff: pd.Timestamp,
    config: H1M1ReplayConfig,
) -> dict[str, Any]:
    result = dict(entry)
    if result.get("disposition") == "FILLED":
        result.update(
            _label_fill(
                context=context,
                entry_time=pd.Timestamp(result["entry_time"]),
                entry_price=float(result["entry_price"]),
                frame=frame,
                cutoff=cutoff,
                config=config,
            )
        )
    elif result.get("disposition") == "NO_FILL":
        result.update(
            label_status="NO_TRADE",
            label_reason="complete entry window observed without an entry",
            net_r=0.0,
        )
    else:
        result.update(
            label_status="CENSORED",
            label_reason=str(result.get("reason") or "unresolved arm"),
        )
    return result


def _apply_cost_basis(
    arm: Mapping[str, Any],
    *,
    context: H1Context,
    cost_profile: CostProfile | None,
) -> dict[str, Any]:
    """Attach an explicit gross/net basis without charging non-trades."""

    result = dict(arm)
    if not _is_policy_resolved(result):
        result.update(
            gross_r=None,
            transaction_cost_r=None,
            net_after_cost_r=None,
            policy_r=None,
            cost_status="UNRESOLVED",
        )
        return result

    gross_r = float(result["net_r"])
    result["gross_r"] = gross_r
    if result.get("disposition") != "FILLED":
        result.update(
            transaction_cost_r=0.0,
            net_after_cost_r=0.0,
            policy_r=0.0,
            cost_status="NO_TRADE",
        )
        if cost_profile is not None:
            result.update(
                cost_profile_id=cost_profile.profile_id,
                cost_profile_sha256=cost_profile.profile_sha256,
            )
        return result

    if cost_profile is None:
        result.update(
            transaction_cost_r=None,
            net_after_cost_r=None,
            policy_r=gross_r,
            cost_status="NOT_SUPPLIED_GROSS_ONLY",
        )
        return result

    estimate = estimate_realized_cost_r(
        cost_profile,
        {
            "symbol": context.symbol,
            "side": context.side,
            "entry": result["entry_price"],
            "stop": context.stop,
            "entry_time": result["entry_time"],
            "exit_time": result["exit_time"],
        },
    )
    net_after_cost_r = gross_r - estimate.total_cost_r
    result.update(estimate.to_dict())
    result.update(
        net_after_cost_r=float(net_after_cost_r),
        policy_r=float(net_after_cost_r),
        cost_status="APPLIED_REALIZED_HOLDING_TIME",
    )
    return result


def _is_policy_resolved(arm: Mapping[str, Any]) -> bool:
    return str(arm.get("label_status")) in {
        "NO_TRADE",
        "CLOSED",
        "FORCED_TIME",
    }


def _entered_and_labeled(arm: Mapping[str, Any]) -> bool:
    return (
        arm.get("disposition") == "FILLED"
        and _is_policy_resolved(arm)
    )


def _pair_status(
    baseline: Mapping[str, Any],
    m1_arm: Mapping[str, Any],
) -> str:
    if not _is_policy_resolved(baseline) or not _is_policy_resolved(m1_arm):
        return "UNRESOLVED"
    baseline_filled = baseline.get("disposition") == "FILLED"
    m1_filled = m1_arm.get("disposition") == "FILLED"
    if baseline_filled and m1_filled:
        return "BOTH_ENTERED"
    if baseline_filled:
        return "BASELINE_ONLY"
    if m1_filled:
        return "M1_RECLAIM_ONLY"
    return "NEITHER_ENTERED"


def _prefix(
    target: dict[str, Any],
    prefix: str,
    arm: Mapping[str, Any],
) -> None:
    for key, value in arm.items():
        if isinstance(value, pd.Timestamp):
            value = value.isoformat()
        target[f"{prefix}_{key}"] = value


def _summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: H1M1ReplayConfig,
    cost_profile: CostProfile | None,
    fold_aware: bool,
) -> dict[str, Any]:
    complete = [row for row in rows if row["policy_complete"]]
    common = [row for row in complete if row["common_fill"]]
    baseline_policy_net = float(
        sum(float(row["baseline_policy_r"]) for row in complete)
    )
    m1_policy_net = float(
        sum(float(row["m1_reclaim_policy_r"]) for row in complete)
    )
    baseline_gross = float(
        sum(float(row["baseline_gross_r"]) for row in complete)
    )
    m1_gross = float(
        sum(float(row["m1_reclaim_gross_r"]) for row in complete)
    )
    common_baseline = float(
        sum(float(row["baseline_policy_r"]) for row in common)
    )
    common_m1 = float(
        sum(float(row["m1_reclaim_policy_r"]) for row in common)
    )
    metric_basis = (
        "net_after_transaction_costs"
        if cost_profile is not None
        else "gross_price_path_r"
    )
    profile_created_before_all_decisions = (
        all(
            pd.Timestamp(cost_profile.created_at_utc)
            <= pd.Timestamp(row["context_decision_time"])
            for row in rows
        )
        if cost_profile is not None
        else None
    )
    fold_starts = sorted({
        str(row["fold_test_start"])
        for row in rows
        if row.get("fold_test_start") is not None
    })
    point_in_time_audit = (
        cost_profile.point_in_time_audit(fold_starts)
        if cost_profile is not None and fold_starts
        else None
    )
    point_in_time_cost_evidence = bool(
        point_in_time_audit
        and point_in_time_audit["causal_for_all_fold_test_starts"]
    )

    def entry_metrics(prefix: str, *, field: str) -> dict[str, Any]:
        outcomes = []
        for row in complete:
            if row.get(f"{prefix}_disposition") != "FILLED":
                continue
            outcomes.append(
                {
                    "status": "CLOSED",
                    "net_r": float(row[f"{prefix}_{field}"]),
                    "tp_hits": row.get(f"{prefix}_tp_hits") or (),
                    "moved_to_be": bool(
                        row.get(f"{prefix}_moved_to_be", False)
                    ),
                    "ambiguous_bars": int(
                        row.get(f"{prefix}_ambiguous_bars") or 0
                    ),
                }
            )
        return aggregate_setup_metrics(outcomes)

    result = {
        "schema": H1_M1_REPLAY_SCHEMA,
        "config": config.audit_payload(),
        "metric_basis": metric_basis,
        "fold_boundary_mode": (
            "explicit_per_context" if fold_aware else "not_supplied"
        ),
        "oos_boundary_safe": bool(fold_aware),
        "cost_profile": (
            {
                "profile_id": cost_profile.profile_id,
                "profile_sha256": cost_profile.profile_sha256,
                "measured_from": cost_profile.measured_from,
                "created_at_utc": cost_profile.created_at_utc.isoformat(),
            }
            if cost_profile is not None
            else None
        ),
        "cost_profile_created_before_all_decisions": (
            profile_created_before_all_decisions
        ),
        "cost_profile_point_in_time_audit": point_in_time_audit,
        "cost_evidence_scope": (
            (
                "POINT_IN_TIME_PROFILE_POST_HOC_OUTCOME_COSTS"
                if point_in_time_cost_evidence
                else "STATIC_STRESS_SCENARIO_NOT_POINT_IN_TIME"
            )
            if cost_profile is not None
            else "GROSS_ONLY"
        ),
        "population_semantics": (
            "independent paired technical opportunities; no portfolio gates"
        ),
        "baseline_semantics": (
            "first in-range M1 open strictly after the H1 decision and "
            "before the exclusive context deadline"
        ),
        "m1_reclaim_semantics": (
            "completed reclaim candle followed by entry on the next M1 open"
        ),
        "outcome_semantics": (
            "same decision-anchored cutoff and intrabar policy for both arms"
        ),
        "contexts": len(rows),
        "complete_pairs": len(complete),
        "excluded_pairs": len(rows) - len(complete),
        "baseline_entries": sum(
            row.get("baseline_disposition") == "FILLED"
            for row in complete
        ),
        "m1_reclaim_entries": sum(
            row.get("m1_reclaim_disposition") == "FILLED"
            for row in complete
        ),
        "both_entered": sum(
            row["pair_status"] == "BOTH_ENTERED" for row in rows
        ),
        "baseline_only": sum(
            row["pair_status"] == "BASELINE_ONLY" for row in rows
        ),
        "m1_reclaim_only": sum(
            row["pair_status"] == "M1_RECLAIM_ONLY" for row in rows
        ),
        "neither_entered": sum(
            row["pair_status"] == "NEITHER_ENTERED" for row in rows
        ),
        "baseline_policy_net_r": baseline_policy_net,
        "m1_reclaim_policy_net_r": m1_policy_net,
        "policy_delta_net_r": m1_policy_net - baseline_policy_net,
        "baseline_policy_expectancy_per_context_r": (
            baseline_policy_net / len(complete) if complete else 0.0
        ),
        "m1_reclaim_policy_expectancy_per_context_r": (
            m1_policy_net / len(complete) if complete else 0.0
        ),
        "baseline_gross_net_r": baseline_gross,
        "m1_reclaim_gross_net_r": m1_gross,
        "gross_delta_net_r": m1_gross - baseline_gross,
        "common_fill_pairs": len(common),
        "baseline_common_fill_net_r": common_baseline,
        "m1_reclaim_common_fill_net_r": common_m1,
        "common_fill_delta_net_r": common_m1 - common_baseline,
        "baseline_entry_metrics": entry_metrics(
            "baseline",
            field="policy_r",
        ),
        "m1_reclaim_entry_metrics": entry_metrics(
            "m1_reclaim",
            field="policy_r",
        ),
        "baseline_gross_entry_metrics": entry_metrics(
            "baseline",
            field="gross_r",
        ),
        "m1_reclaim_gross_entry_metrics": entry_metrics(
            "m1_reclaim",
            field="gross_r",
        ),
    }
    return result


def run_h1_m1_paired_replay(
    *,
    contexts: Sequence[H1Context],
    m1_by_symbol: Mapping[str, pd.DataFrame],
    config: H1M1ReplayConfig,
    cost_profile: CostProfile | None = None,
    fold_by_context: Mapping[str, H1M1FoldWindow] | None = None,
) -> H1M1ReplayResult:
    """Replay an exact paired population without touching production state.

    Contexts are independent observations. Outcomes never select, suppress, or
    reorder another context, so scanning the fixed post-decision label window
    cannot feed future information into a later decision.
    """

    ordered = sorted(
        contexts,
        key=lambda item: (
            item.decision_time,
            item.symbol,
            item.context_id,
        ),
    )
    ids = [context.context_id for context in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError(
            "context_id values must be unique for exact pairing"
        )
    if fold_by_context is not None:
        missing_fold = sorted(set(ids) - set(fold_by_context))
        extra_fold = sorted(set(fold_by_context) - set(ids))
        if missing_fold or extra_fold:
            raise ValueError(
                "fold_by_context must match context ids exactly; "
                f"missing={missing_fold}, extra={extra_fold}"
            )
    unsupported = sorted(
        {
            context.trigger_kind
            for context in ordered
            if context.trigger_kind not in _ALLOWED_PARENT_TRIGGERS
        }
    )
    if unsupported:
        raise ValueError(
            "H1 to M1 research arm only accepts order_block_1h and "
            f"rejection_block_1h parents; received {unsupported}"
        )

    prepared: dict[str, pd.DataFrame] = {}
    data_errors: dict[str, str] = {}
    for symbol in sorted({context.symbol for context in ordered}):
        raw = m1_by_symbol.get(symbol)
        try:
            prepared[symbol] = _prepare_m1(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            data_errors[symbol] = str(exc)

    rows: list[dict[str, Any]] = []
    for context in ordered:
        fold = (
            fold_by_context[context.context_id]
            if fold_by_context is not None
            else None
        )
        if fold is not None and not (
            fold.test_start <= context.decision_time < fold.test_end
        ):
            raise ValueError(
                f"{context.context_id}: decision_time is outside fold "
                f"{fold.fold_index}"
            )
        intended_cutoff = context.decision_time + config.outcome_horizon
        cutoff = (
            min(intended_cutoff, fold.test_end)
            if fold is not None
            else intended_cutoff
        )
        deadline = min(context.expires_at, cutoff)
        fold_clipped = bool(
            fold is not None
            and (
                intended_cutoff > fold.test_end
                or context.expires_at > fold.test_end
            )
        )
        base: dict[str, Any] = {
            "schema": H1_M1_REPLAY_SCHEMA,
            "context_id": context.context_id,
            "symbol": context.symbol,
            "side": context.side,
            "trigger_kind": context.trigger_kind,
            "context_decision_time": (
                context.decision_time.isoformat()
            ),
            "context_expires_at": context.expires_at.isoformat(),
            "effective_entry_deadline": deadline.isoformat(),
            "outcome_cutoff": cutoff.isoformat(),
            "intended_outcome_cutoff": intended_cutoff.isoformat(),
            "fold_index": fold.fold_index if fold is not None else None,
            "fold_test_start": (
                fold.test_start.isoformat() if fold is not None else None
            ),
            "fold_test_end": (
                fold.test_end.isoformat() if fold is not None else None
            ),
            "fold_clipped": fold_clipped,
            "parent_arbitration_group_id": context.signal.get(
                "arbitration_group_id"
            ),
            "parent_trigger_event_id": context.signal.get(
                "trigger_event_id"
            ),
        }
        if context.symbol in data_errors:
            reason = data_errors[context.symbol]
            baseline = {
                "disposition": "INVALID",
                "reason": reason,
                "label_status": "CENSORED",
                "label_reason": reason,
            }
            m1_arm = dict(baseline)
            m1_arm["trigger_status"] = "DATA_INVALID"
        else:
            try:
                frame = _validate_bounded_m1(
                    _bounded_context_frame(
                        prepared[context.symbol],
                        decision_time=context.decision_time,
                        cutoff=cutoff,
                    )
                )
            except ValueError as exc:
                reason = str(exc)
                baseline = {
                    "disposition": "INVALID",
                    "reason": reason,
                    "label_status": "CENSORED",
                    "label_reason": reason,
                }
                m1_arm = {
                    **baseline,
                    "trigger_status": "DATA_INVALID",
                }
                baseline = _apply_cost_basis(
                    baseline,
                    context=context,
                    cost_profile=cost_profile,
                )
                m1_arm = _apply_cost_basis(
                    m1_arm,
                    context=context,
                    cost_profile=cost_profile,
                )
                row = dict(base)
                _prefix(row, "baseline", baseline)
                _prefix(row, "m1_reclaim", m1_arm)
                row.update(
                    pair_status="UNRESOLVED",
                    policy_complete=False,
                    common_fill=False,
                    baseline_policy_r=None,
                    m1_reclaim_policy_r=None,
                    baseline_gross_r=None,
                    m1_reclaim_gross_r=None,
                    policy_delta_r=None,
                    common_fill_delta_r=None,
                )
                row["row_id"] = _canonical_hash(row)[:24]
                rows.append(row)
                continue
            first_entry_open = (
                context.decision_time.floor("min") + _ONE_MINUTE
            )
            baseline_contiguous = _is_contiguous_m1(
                frame,
                first_open=first_entry_open,
                end_exclusive=deadline,
            )
            first_trigger_open = context.decision_time.ceil("min")
            trigger_reference_open = first_trigger_open - _ONE_MINUTE
            m1_contiguous = _is_contiguous_m1(
                frame,
                first_open=trigger_reference_open,
                end_exclusive=deadline,
            )
            if not baseline_contiguous:
                baseline = {
                    "disposition": "CENSORED_ENTRY_WINDOW",
                    "reason": "missing M1 bar inside the entry window",
                    "label_status": "CENSORED",
                    "label_reason": (
                        "missing M1 bar inside the entry window"
                    ),
                }
            else:
                baseline = _finish_arm(
                    entry=_baseline_fill(
                        context,
                        frame,
                        deadline=deadline,
                    ),
                    context=context,
                    frame=frame,
                    cutoff=cutoff,
                    config=config,
                )
            if not m1_contiguous:
                m1_arm = {
                    "disposition": "CENSORED_ENTRY_WINDOW",
                    "reason": (
                        "missing M1 trigger/reference bar inside the "
                        "entry window"
                    ),
                    "label_status": "CENSORED",
                    "label_reason": (
                        "missing M1 trigger/reference bar inside the "
                        "entry window"
                    ),
                    "trigger_status": "DATA_GAP",
                }
            else:
                m1_arm = _finish_arm(
                    entry=_m1_entry(
                        context,
                        frame,
                        deadline=deadline,
                        config=config,
                    ),
                    context=context,
                    frame=frame,
                    cutoff=cutoff,
                    config=config,
                )
            baseline = _apply_cost_basis(
                baseline,
                context=context,
                cost_profile=cost_profile,
            )
            m1_arm = _apply_cost_basis(
                m1_arm,
                context=context,
                cost_profile=cost_profile,
            )

        row = dict(base)
        _prefix(row, "baseline", baseline)
        _prefix(row, "m1_reclaim", m1_arm)
        policy_complete = (
            _is_policy_resolved(baseline)
            and _is_policy_resolved(m1_arm)
        )
        common_fill = (
            _entered_and_labeled(baseline)
            and _entered_and_labeled(m1_arm)
        )
        row["pair_status"] = _pair_status(baseline, m1_arm)
        row["policy_complete"] = policy_complete
        row["common_fill"] = common_fill
        row["baseline_policy_r"] = (
            float(baseline["policy_r"]) if policy_complete else None
        )
        row["m1_reclaim_policy_r"] = (
            float(m1_arm["policy_r"]) if policy_complete else None
        )
        row["baseline_gross_r"] = (
            float(baseline["gross_r"]) if policy_complete else None
        )
        row["m1_reclaim_gross_r"] = (
            float(m1_arm["gross_r"]) if policy_complete else None
        )
        row["policy_delta_r"] = (
            row["m1_reclaim_policy_r"] - row["baseline_policy_r"]
            if policy_complete
            else None
        )
        row["common_fill_delta_r"] = (
            float(m1_arm["policy_r"]) - float(baseline["policy_r"])
            if common_fill
            else None
        )
        row["row_id"] = _canonical_hash(row)[:24]
        rows.append(row)

    summary = _summary(
        rows,
        config=config,
        cost_profile=cost_profile,
        fold_aware=fold_by_context is not None,
    )
    summary["result_id"] = _canonical_hash(
        {
            "schema": H1_M1_REPLAY_SCHEMA,
            "config": config.audit_payload(),
            "row_ids": [row["row_id"] for row in rows],
        }
    )[:24]
    return H1M1ReplayResult(rows=tuple(rows), summary=summary)


__all__ = [
    "H1_M1_REPLAY_SCHEMA",
    "H1M1ReplayConfig",
    "H1M1FoldWindow",
    "H1M1ReplayResult",
    "run_h1_m1_paired_replay",
]
