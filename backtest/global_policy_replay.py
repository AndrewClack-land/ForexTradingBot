"""Research-only chronological replay for the paired entry-policy arms.

``paired_entry_arms`` deliberately evaluates one decision in isolation.  This
module adds the missing policy state: one unified UTC event queue across
symbols, pending orders, active positions, fold resets and decision-time
portfolio-aware ranking.  It is not imported by the live bot.

The event order is part of the experiment contract.  At an equal timestamp a
fold boundary is applied first, then pending invalidation/expiry, position
exit, fill, and finally a new decision.  A candidate is selected before its
future M1 fill is observed and is never replaced by an alternative that later
fills more conveniently.

``planned_limit_touch`` remains an M1-range sensitivity, not proof of an exact
tick fill.  Entry/stop collisions are terminal ambiguous observations and are
not counted as executions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import heapq
import json
import math
from typing import Any, Literal, Mapping, Optional, Sequence

import pandas as pd

from backtest.cost_model import CostProfile, estimate_realized_cost_r
from backtest.paired_entry_arms import (
    ArmSpec,
    DecisionCandidate,
    FoldWindow,
    PairedEntryArmError,
    four_arm_specs,
    observe_entry,
    select_candidate,
)
from backtest.simulator import DEFAULT_WEIGHTS, SetupOutcome, simulate_split_outcome
from core.shadow_candidate_ranker import (
    PortfolioCorrelationProfile,
    correlation_penalty_r,
)


GLOBAL_POLICY_REPLAY_SCHEMA = "global-policy-replay/v1"

EVENT_PRIORITY: Mapping[str, int] = {
    "FOLD_END": 0,
    "PENDING_TERMINAL": 10,
    "POSITION_EXIT": 20,
    "FILL": 30,
    "DECISION": 40,
}

DuplicateKeyPolicy = Literal[
    "none",
    "symbol_trigger_side",
    "parent_opportunity_id",
]
DuplicateRecordOn = Literal["arm", "fill"]
DailyCapBasis = Literal["gross_r", "net_after_cost_r"]
CommonCostProfileUsage = Literal[
    "POINT_IN_TIME_COMMON",
    "POST_HOC_STATIC_STRESS",
]
CommonCorrelationProfileUsage = Literal["POINT_IN_TIME_COMMON"]


class GlobalReplayError(ValueError):
    """Raised when a global replay cannot be evaluated causally."""


class UnsupportedReplayFeature(GlobalReplayError):
    """Raised instead of silently approximating an unsupported live feature."""


def _utc(value: Any, *, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise GlobalReplayError(f"{name} must be a timestamp") from exc
    if timestamp.tzinfo is None:
        raise GlobalReplayError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _duration(value: Any, *, name: str) -> pd.Timedelta:
    try:
        duration = pd.Timedelta(value)
    except (TypeError, ValueError) as exc:
        raise GlobalReplayError(f"{name} must be a duration") from exc
    if duration < pd.Timedelta(0):
        raise GlobalReplayError(f"{name} cannot be negative")
    return duration


def _finite(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GlobalReplayError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise GlobalReplayError(f"{name} must be finite")
    return number


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _iso(value: Optional[pd.Timestamp]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def _candidate_lineage(candidate: DecisionCandidate) -> dict[str, Any]:
    """Return every decision-time candidate field in canonical JSON form."""

    return {
        "parent_opportunity_id": candidate.parent_opportunity_id,
        "decision_event_id": candidate.decision_event_id,
        "arbitration_group_id": candidate.arbitration_group_id,
        "fold_index": candidate.fold_index,
        "symbol": candidate.symbol,
        "trigger_kind": candidate.trigger_kind,
        "side": candidate.side,
        "bar_close_time": candidate.bar_close_time.isoformat(),
        "decision_time": candidate.decision_time.isoformat(),
        "expires_at": candidate.expires_at.isoformat(),
        "entry_min": candidate.entry_min,
        "entry_max": candidate.entry_max,
        "planned_entry": candidate.planned_entry,
        "stop": candidate.stop,
        "tp_prices": list(candidate.tp_prices),
        "production_priority": candidate.production_priority,
        "expected_gross_r": candidate.expected_gross_r,
        "estimated_cost_r": candidate.estimated_cost_r,
        "expected_net_r": candidate.expected_net_r,
        "ranking_basis": candidate.ranking_basis,
        "optimizer_rank": candidate.optimizer_rank,
        "model_id": candidate.model_id,
        "model_train_end": _iso(candidate.model_train_end),
        "known_at": _iso(candidate.known_at),
        "cost_profile_id": candidate.cost_profile_id,
        "cost_profile_sha256": candidate.cost_profile_sha256,
        "cost_profile_created_at_utc": _iso(
            candidate.cost_profile_created_at_utc
        ),
        "cost_profile_measured_through": _iso(
            candidate.cost_profile_measured_through
        ),
    }


@dataclass(frozen=True)
class ReplayPolicyConfig:
    """Stateful gates applied identically to every experimental arm.

    A symbol can never have more than one pending or active idea.  Global
    position capacity, cooldown, daily loss and duplicate semantics are
    configurable so a report cannot inherit hidden live defaults.
    """

    intrabar_policy: Literal["stop-first", "tp-first"] = "stop-first"
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS
    max_active_positions: Optional[int] = None
    reserve_active_slot_for_pending: bool = True
    cooldown_after_exit: pd.Timedelta = pd.Timedelta(0)
    cooldown_after_pending_terminal: pd.Timedelta = pd.Timedelta(0)
    daily_loss_cap_r: Optional[float] = None
    daily_cap_basis: DailyCapBasis = "gross_r"
    duplicate_window: pd.Timedelta = pd.Timedelta(0)
    duplicate_key_policy: DuplicateKeyPolicy = "none"
    duplicate_record_on: DuplicateRecordOn = "arm"
    correlation_include_pending: bool = False
    carry_across_folds: bool = False
    allow_position_adding: bool = False

    def __post_init__(self) -> None:
        if self.intrabar_policy not in {"stop-first", "tp-first"}:
            raise GlobalReplayError("unsupported intrabar_policy")
        weights = tuple(_finite(value, name="weight") for value in self.weights)
        if len(weights) != 3 or any(value <= 0.0 for value in weights):
            raise GlobalReplayError("weights must contain three positive values")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise GlobalReplayError("weights must sum to 1")
        object.__setattr__(self, "weights", weights)
        if self.max_active_positions is not None:
            if (
                isinstance(self.max_active_positions, bool)
                or int(self.max_active_positions) < 1
            ):
                raise GlobalReplayError("max_active_positions must be positive")
            object.__setattr__(
                self,
                "max_active_positions",
                int(self.max_active_positions),
            )
            if not self.reserve_active_slot_for_pending:
                raise UnsupportedReplayFeature(
                    "global capacity without pending slot reservation is "
                    "unsupported because an accepted pending order cannot be "
                    "causally rejected at its later fill"
                )
        object.__setattr__(
            self,
            "cooldown_after_exit",
            _duration(self.cooldown_after_exit, name="cooldown_after_exit"),
        )
        object.__setattr__(
            self,
            "cooldown_after_pending_terminal",
            _duration(
                self.cooldown_after_pending_terminal,
                name="cooldown_after_pending_terminal",
            ),
        )
        object.__setattr__(
            self,
            "duplicate_window",
            _duration(self.duplicate_window, name="duplicate_window"),
        )
        if self.daily_loss_cap_r is not None:
            cap = _finite(self.daily_loss_cap_r, name="daily_loss_cap_r")
            if cap <= 0.0:
                raise GlobalReplayError("daily_loss_cap_r must be positive")
            object.__setattr__(self, "daily_loss_cap_r", cap)
        if self.daily_cap_basis not in {"gross_r", "net_after_cost_r"}:
            raise GlobalReplayError("unsupported daily_cap_basis")
        if self.duplicate_key_policy not in {
            "none",
            "symbol_trigger_side",
            "parent_opportunity_id",
        }:
            raise GlobalReplayError("unsupported duplicate_key_policy")
        if self.duplicate_record_on not in {"arm", "fill"}:
            raise GlobalReplayError("unsupported duplicate_record_on")
        if self.duplicate_key_policy == "none" and self.duplicate_window:
            raise GlobalReplayError(
                "duplicate_window requires a duplicate_key_policy"
            )
        if self.carry_across_folds:
            raise UnsupportedReplayFeature(
                "carry_across_folds is unsupported; folds reset by default"
            )
        if self.allow_position_adding:
            raise UnsupportedReplayFeature(
                "position adding is unsupported in the rank-one experiment"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intrabar_policy": self.intrabar_policy,
            "weights": list(self.weights),
            "max_active_positions": self.max_active_positions,
            "reserve_active_slot_for_pending": (
                self.reserve_active_slot_for_pending
            ),
            "cooldown_after_exit_seconds": self.cooldown_after_exit.total_seconds(),
            "cooldown_after_pending_terminal_seconds": (
                self.cooldown_after_pending_terminal.total_seconds()
            ),
            "daily_loss_cap_r": self.daily_loss_cap_r,
            "daily_cap_basis": self.daily_cap_basis,
            "duplicate_window_seconds": self.duplicate_window.total_seconds(),
            "duplicate_key_policy": self.duplicate_key_policy,
            "duplicate_record_on": self.duplicate_record_on,
            "correlation_include_pending": self.correlation_include_pending,
            "carry_across_folds": self.carry_across_folds,
            "allow_position_adding": self.allow_position_adding,
        }


@dataclass(frozen=True)
class PendingInvalidation:
    """Causal, externally observed invalidation for a pending candidate."""

    fold_index: int
    parent_opportunity_id: str
    observed_at: pd.Timestamp
    reason: str

    def __post_init__(self) -> None:
        if isinstance(self.fold_index, bool) or int(self.fold_index) < 0:
            raise GlobalReplayError("fold_index must be nonnegative")
        parent = str(self.parent_opportunity_id or "").strip()
        reason = str(self.reason or "").strip()
        if not parent or not reason:
            raise GlobalReplayError("invalidation parent and reason are required")
        object.__setattr__(self, "fold_index", int(self.fold_index))
        object.__setattr__(self, "parent_opportunity_id", parent)
        object.__setattr__(
            self,
            "observed_at",
            _utc(self.observed_at, name="invalidation.observed_at"),
        )
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True)
class GlobalReplayResult:
    experiment_input_id: str
    replay_id: str
    arm: str
    events: tuple[Mapping[str, Any], ...]
    setups: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, Any]
    event_order: Mapping[str, int] = field(default_factory=lambda: EVENT_PRIORITY)
    unsupported_features: tuple[str, ...] = (
        "position_adding",
        "partial_fills",
        "tick_path_inside_planned_limit_m1_range",
        "cross_fold_position_carry",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GLOBAL_POLICY_REPLAY_SCHEMA,
            "experiment_input_id": self.experiment_input_id,
            "replay_id": self.replay_id,
            "arm": self.arm,
            "event_order": dict(self.event_order),
            "unsupported_features": list(self.unsupported_features),
            "metrics": dict(self.metrics),
            "events": [dict(row) for row in self.events],
            "setups": [dict(row) for row in self.setups],
        }


@dataclass
class _Idea:
    setup_id: str
    candidate: DecisionCandidate
    spec: ArmSpec
    selected_score: Optional[float]
    correlation_penalty_r: Optional[float]
    alternatives: tuple[Mapping[str, Any], ...]
    status: str = "PENDING"
    terminal_reason: Optional[str] = None
    fill_time: Optional[pd.Timestamp] = None
    entry_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    gross_r: Optional[float] = None
    net_after_cost_r: Optional[float] = None
    transaction_cost_r: Optional[float] = None
    cost_fields: dict[str, Any] = field(default_factory=dict)
    outcome: Optional[SetupOutcome] = None
    partial_gross_r: Optional[float] = None
    observation_status: Optional[str] = None
    observation_reason: Optional[str] = None

    def to_row(self, replay_id: str) -> dict[str, Any]:
        candidate = self.candidate
        legs: list[dict[str, Any]] = []
        if self.outcome is not None:
            for leg in self.outcome.legs:
                row = asdict(leg)
                row["exit_time"] = _iso(leg.exit_time)
                legs.append(row)
        return {
            "schema": GLOBAL_POLICY_REPLAY_SCHEMA,
            "replay_id": replay_id,
            "setup_id": self.setup_id,
            "arm": self.spec.arm,
            "candidate_policy": self.spec.candidate_policy,
            "entry_policy": self.spec.entry_policy,
            "sensitivity_only": self.spec.sensitivity_only,
            "fold_index": candidate.fold_index,
            "arbitration_group_id": candidate.arbitration_group_id,
            "decision_event_id": candidate.decision_event_id,
            "parent_opportunity_id": candidate.parent_opportunity_id,
            "symbol": candidate.symbol,
            "trigger_kind": candidate.trigger_kind,
            "side": candidate.side,
            "decision_time": candidate.decision_time.isoformat(),
            "expires_at": candidate.expires_at.isoformat(),
            "status": self.status,
            "terminal_reason": self.terminal_reason,
            "observation_status": self.observation_status,
            "observation_reason": self.observation_reason,
            "fill_time": _iso(self.fill_time),
            "entry": self.entry_price,
            "stop": candidate.stop,
            "tp_prices": list(candidate.tp_prices),
            "exit_time": _iso(self.exit_time),
            "gross_net_r": self.gross_r,
            "transaction_cost_r": self.transaction_cost_r,
            "net_after_cost_r": self.net_after_cost_r,
            "realized_partial_gross_r_at_censor": self.partial_gross_r,
            "selected_ranking_score": self.selected_score,
            "correlation_penalty_r": self.correlation_penalty_r,
            "alternatives": [dict(row) for row in self.alternatives],
            "tp_hits": (
                list(self.outcome.tp_hits) if self.outcome is not None else []
            ),
            "ambiguous_bars": (
                self.outcome.ambiguous_bars
                if self.outcome is not None
                else None
            ),
            "legs": legs,
            **self.cost_fields,
        }


@dataclass(frozen=True)
class _ScheduledEvent:
    timestamp: pd.Timestamp
    priority: int
    stable_key: str
    serial: int
    kind: str = field(compare=False)
    payload: Any = field(compare=False)

    def heap_key(self) -> tuple[int, int, str, int, "_ScheduledEvent"]:
        return (
            int(self.timestamp.value),
            self.priority,
            self.stable_key,
            self.serial,
            self,
        )


def _normalize_m1_frames(
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    normalized: dict[str, pd.DataFrame] = {}
    for raw_symbol, source in frames.items():
        symbol = str(raw_symbol).strip().upper()
        required = {"open", "high", "low", "close"}
        if not required.issubset(source.columns):
            missing = sorted(required - set(source.columns))
            raise GlobalReplayError(f"{symbol}: missing M1 columns {missing}")
        frame = source.copy().sort_index(kind="stable")
        if frame.index.has_duplicates:
            raise GlobalReplayError(f"{symbol}: duplicate M1 timestamps")
        index = pd.DatetimeIndex(frame.index)
        if index.tz is None:
            raise GlobalReplayError(f"{symbol}: M1 index must be timezone-aware")
        frame.index = index.tz_convert("UTC")
        for column in required:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if not frame[column].map(math.isfinite).all():
                raise GlobalReplayError(f"{symbol}: non-finite M1 prices")
        impossible = (
            (frame["high"] < frame["low"])
            | (frame["high"] < frame[["open", "close"]].max(axis=1))
            | (frame["low"] > frame[["open", "close"]].min(axis=1))
        )
        if impossible.any():
            raise GlobalReplayError(f"{symbol}: impossible M1 candle")
        normalized[symbol] = frame
    return normalized


def _validate_folds(folds: Sequence[FoldWindow]) -> tuple[FoldWindow, ...]:
    ordered = tuple(sorted(folds, key=lambda fold: fold.test_start))
    if not ordered:
        raise GlobalReplayError("at least one fold is required")
    if len({fold.fold_index for fold in ordered}) != len(ordered):
        raise GlobalReplayError("fold_index values must be unique")
    for previous, current in zip(ordered, ordered[1:]):
        if current.test_start < previous.test_end:
            raise GlobalReplayError("OOS folds cannot overlap")
    return ordered


def _validate_groups(
    decision_groups: Sequence[Sequence[DecisionCandidate]],
    folds: Mapping[int, FoldWindow],
    *,
    net_policy: bool,
) -> dict[int, list[tuple[DecisionCandidate, ...]]]:
    grouped: dict[int, list[tuple[DecisionCandidate, ...]]] = {
        fold_index: [] for fold_index in folds
    }
    seen_groups: set[tuple[int, str]] = set()
    seen_symbol_times: dict[tuple[int, str, int], str] = {}
    seen_parents: set[tuple[int, str]] = set()
    for source in decision_groups:
        group = tuple(source)
        if not group:
            raise GlobalReplayError("decision groups cannot be empty")
        fold_index = group[0].fold_index
        if fold_index not in folds:
            raise GlobalReplayError(f"unknown candidate fold {fold_index}")
        fold = folds[fold_index]
        try:
            select_candidate(
                group,
                policy="net_rank_one" if net_policy else "production_priority_one",
                fold=fold,
            )
        except PairedEntryArmError as exc:
            raise GlobalReplayError(str(exc)) from exc
        first = group[0]
        group_key = (fold_index, first.arbitration_group_id)
        if group_key in seen_groups:
            raise GlobalReplayError("arbitration_group_id must be unique per fold")
        seen_groups.add(group_key)
        symbol_time = (fold_index, first.symbol, int(first.decision_time.value))
        previous_group = seen_symbol_times.get(symbol_time)
        if previous_group is not None and previous_group != first.arbitration_group_id:
            raise GlobalReplayError(
                "one symbol/timestamp must use one arbitration_group_id"
            )
        seen_symbol_times[symbol_time] = first.arbitration_group_id
        for candidate in group:
            parent_key = (fold_index, candidate.parent_opportunity_id)
            if parent_key in seen_parents:
                raise GlobalReplayError(
                    "parent_opportunity_id must be unique within a fold"
                )
            seen_parents.add(parent_key)
            if len(candidate.tp_prices) != 3:
                raise UnsupportedReplayFeature(
                    "global replay currently requires exactly three TP prices"
                )
        grouped[fold_index].append(group)
    for groups in grouped.values():
        groups.sort(
            key=lambda group: (
                group[0].decision_time,
                group[0].symbol,
                group[0].arbitration_group_id,
            )
        )
    return grouped


def _resolve_fold_profiles(
    *,
    folds: Sequence[FoldWindow],
    net_policy: bool,
    config: ReplayPolicyConfig,
    cost_profile: Optional[CostProfile],
    correlation_profile: Optional[PortfolioCorrelationProfile],
    cost_profiles_by_fold: Optional[Mapping[int, CostProfile]],
    correlation_profiles_by_fold: Optional[
        Mapping[int, PortfolioCorrelationProfile]
    ],
    common_cost_profile_usage: Optional[CommonCostProfileUsage],
    common_correlation_profile_usage: Optional[
        CommonCorrelationProfileUsage
    ],
) -> tuple[
    dict[int, CostProfile],
    dict[int, PortfolioCorrelationProfile],
    dict[int, Mapping[str, Any]],
    dict[int, str],
    dict[int, Mapping[str, Any]],
]:
    """Bind profiles to folds without hiding common-profile assumptions."""

    fold_map = {fold.fold_index: fold for fold in folds}
    fold_keys = set(fold_map)
    if cost_profile is not None and cost_profiles_by_fold is not None:
        raise GlobalReplayError(
            "use either cost_profile or cost_profiles_by_fold, not both"
        )
    if (
        correlation_profile is not None
        and correlation_profiles_by_fold is not None
    ):
        raise GlobalReplayError(
            "use either correlation_profile or "
            "correlation_profiles_by_fold, not both"
        )
    if cost_profile is not None and common_cost_profile_usage is None:
        raise GlobalReplayError(
            "common_cost_profile_usage explicitly labels a shared profile"
        )
    if cost_profile is None and common_cost_profile_usage is not None:
        raise GlobalReplayError(
            "common_cost_profile_usage requires cost_profile"
        )
    if (
        correlation_profile is not None
        and common_correlation_profile_usage is None
    ):
        raise GlobalReplayError(
            "common_correlation_profile_usage explicitly labels a "
            "shared profile"
        )
    if (
        correlation_profile is None
        and common_correlation_profile_usage is not None
    ):
        raise GlobalReplayError(
            "common_correlation_profile_usage requires correlation_profile"
        )

    cost_by_fold: dict[int, CostProfile] = {}
    if cost_profile is not None:
        cost_by_fold = {key: cost_profile for key in fold_keys}
    elif cost_profiles_by_fold is not None:
        cost_by_fold = {int(key): value for key, value in cost_profiles_by_fold.items()}
        if set(cost_by_fold) != fold_keys:
            raise GlobalReplayError(
                "cost_profiles_by_fold must have exactly one profile per fold"
            )

    correlation_by_fold: dict[int, PortfolioCorrelationProfile] = {}
    if correlation_profile is not None:
        correlation_by_fold = {
            key: correlation_profile for key in fold_keys
        }
    elif correlation_profiles_by_fold is not None:
        correlation_by_fold = {
            int(key): value
            for key, value in correlation_profiles_by_fold.items()
        }
        if set(correlation_by_fold) != fold_keys:
            raise GlobalReplayError(
                "correlation_profiles_by_fold must have exactly one profile "
                "per fold"
            )

    cost_changes_policy = (
        net_policy or config.daily_cap_basis == "net_after_cost_r"
    )
    if cost_changes_policy and set(cost_by_fold) != fold_keys:
        raise GlobalReplayError(
            "policy-changing costs require a causal profile for every fold"
        )
    if net_policy and set(correlation_by_fold) != fold_keys:
        raise GlobalReplayError(
            "net_rank_one requires a causal correlation profile for every fold"
        )
    if (
        common_cost_profile_usage == "POST_HOC_STATIC_STRESS"
        and cost_changes_policy
    ):
        raise GlobalReplayError(
            "POST_HOC_STATIC_STRESS cannot change ranking or entry gates"
        )

    cost_audits: dict[int, Mapping[str, Any]] = {}
    cost_usages: dict[int, str] = {}
    for fold_index, profile in cost_by_fold.items():
        fold = fold_map[fold_index]
        audit = profile.point_in_time_audit(
            [fold.test_start.to_pydatetime()]
        )
        cost_audits[fold_index] = audit
        if cost_profile is not None:
            usage = str(common_cost_profile_usage)
        else:
            usage = (
                "POINT_IN_TIME_FOLD_POLICY_INPUT"
                if cost_changes_policy
                else "POST_HOC_FOLD_STATIC_STRESS"
            )
        cost_usages[fold_index] = usage
        requires_causal = (
            cost_changes_policy
            or common_cost_profile_usage == "POINT_IN_TIME_COMMON"
        )
        if requires_causal and not audit["causal_for_all_fold_test_starts"]:
            raise GlobalReplayError(
                f"cost profile is not causal for fold {fold_index}"
            )

    correlation_audits: dict[int, Mapping[str, Any]] = {}
    for fold_index, profile in correlation_by_fold.items():
        fold = fold_map[fold_index]
        trained_end = _utc(
            profile.trained_end_utc,
            name="correlation_profile.trained_end_utc",
        )
        causal = trained_end <= fold.test_start
        audit = {
            "profile_id": profile.profile_id,
            "profile_sha256": profile.profile_sha256,
            "trained_end_utc": trained_end.isoformat(),
            "fold_test_start": fold.test_start.isoformat(),
            "causal_for_fold": causal,
            "usage": (
                str(common_correlation_profile_usage)
                if correlation_profile is not None
                else "POINT_IN_TIME_FOLD_POLICY_INPUT"
            ),
        }
        correlation_audits[fold_index] = audit
        if not causal:
            raise GlobalReplayError(
                f"correlation profile is not causal for fold {fold_index}"
            )

    return (
        cost_by_fold,
        correlation_by_fold,
        cost_audits,
        cost_usages,
        correlation_audits,
    )


def _duplicate_key(
    candidate: DecisionCandidate,
    policy: DuplicateKeyPolicy,
) -> Optional[str]:
    if policy == "none":
        return None
    if policy == "parent_opportunity_id":
        return candidate.parent_opportunity_id
    return f"{candidate.symbol}|{candidate.trigger_kind}|{candidate.side}"


def _selection(
    group: tuple[DecisionCandidate, ...],
    *,
    spec: ArmSpec,
    fold: FoldWindow,
    cost_profile: Optional[CostProfile],
    correlation_profile: Optional[PortfolioCorrelationProfile],
    exposures: Mapping[str, str],
) -> tuple[
    DecisionCandidate,
    Optional[float],
    Optional[float],
    tuple[Mapping[str, Any], ...],
]:
    if spec.candidate_policy == "production_priority_one":
        selected = select_candidate(
            group,
            policy="production_priority_one",
            fold=fold,
        )
        alternatives = tuple(
            {
                "parent_opportunity_id": candidate.parent_opportunity_id,
                "production_priority": candidate.production_priority,
                "expected_net_r": candidate.expected_net_r,
                "correlation_penalty_r": None,
                "ranking_score": None,
                "selected": candidate is selected,
            }
            for candidate in sorted(
                group,
                key=lambda item: (
                    item.production_priority,
                    item.parent_opportunity_id,
                ),
            )
        )
        return selected, None, None, alternatives

    if cost_profile is None or correlation_profile is None:
        raise GlobalReplayError(
            "net_rank_one requires causal cost and correlation profiles"
        )
    # This call performs the strict model/cost score provenance validation.  We
    # intentionally discard its winner and add the current causal exposures.
    select_candidate(group, policy="net_rank_one", fold=fold)
    scored: list[tuple[DecisionCandidate, float, float]] = []
    for candidate in group:
        if (
            candidate.cost_profile_id != cost_profile.profile_id
            or candidate.cost_profile_sha256 != cost_profile.profile_sha256
        ):
            raise GlobalReplayError(
                "candidate score and realized-cost profile identities differ"
            )
        penalty = correlation_penalty_r(
            correlation_profile,
            symbol=candidate.symbol,
            side=candidate.side,
            active_exposures=exposures,
            decision_time_utc=candidate.decision_time.to_pydatetime(),
        )
        score = float(candidate.expected_net_r) - penalty
        scored.append((candidate, penalty, score))
    scored.sort(
        key=lambda item: (
            -item[2],
            -float(item[0].expected_net_r),
            item[0].production_priority,
            item[0].parent_opportunity_id,
        )
    )
    selected, selected_penalty, selected_score = scored[0]
    alternatives = tuple(
        {
            "parent_opportunity_id": candidate.parent_opportunity_id,
            "production_priority": candidate.production_priority,
            "expected_net_r": candidate.expected_net_r,
            "correlation_penalty_r": penalty,
            "ranking_score": score,
            "selected": candidate is selected,
        }
        for candidate, penalty, score in scored
    )
    return selected, selected_score, selected_penalty, alternatives


def _metric_block(
    setups: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
) -> Optional[dict[str, Any]]:
    closed = [row for row in setups if row["status"] == "CLOSED"]
    if not closed or any(row.get(value_key) is None for row in closed):
        return None
    values = [float(row[value_key]) for row in closed]
    wins = [value for value in values if value > 0.0]
    losses = [value for value in values if value < 0.0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    by_exit: dict[str, float] = {}
    for row, value in zip(closed, values):
        exit_time = str(row["exit_time"])
        by_exit[exit_time] = by_exit.get(exit_time, 0.0) + value
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for timestamp in sorted(by_exit):
        equity += by_exit[timestamp]
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return {
        "closed_setups": len(values),
        "net_r": float(sum(values)),
        "expectancy_r": float(sum(values) / len(values)),
        "win_rate": float(len(wins) / len(values)),
        "profit_factor": (
            float(gross_profit / gross_loss) if gross_loss > 0.0 else None
        ),
        "max_drawdown_r": float(max_drawdown),
    }


def _summarize(setups: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for row in setups:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    rejected = sum(1 for row in events if row["kind"] == "DECISION_REJECTED")
    return {
        "decision_groups": sum(
            1
            for row in events
            if row["kind"] in {"DECISION_ARMED", "DECISION_REJECTED"}
        ),
        "armed_setups": len(setups),
        "rejected_decisions": rejected,
        "filled_setups": sum(row.get("fill_time") is not None for row in setups),
        "status_counts": status_counts,
        "gross": _metric_block(setups, value_key="gross_net_r"),
        "net_after_cost": _metric_block(
            setups,
            value_key="net_after_cost_r",
        ),
    }


def _run_fold(
    *,
    replay_id: str,
    fold: FoldWindow,
    groups: Sequence[tuple[DecisionCandidate, ...]],
    frames: Mapping[str, pd.DataFrame],
    spec: ArmSpec,
    config: ReplayPolicyConfig,
    cost_profile: Optional[CostProfile],
    correlation_profile: Optional[PortfolioCorrelationProfile],
    invalidations: Mapping[tuple[int, str], PendingInvalidation],
) -> tuple[list[dict[str, Any]], list[_Idea]]:
    heap: list[tuple[int, int, str, int, _ScheduledEvent]] = []
    serial = 0

    def schedule(
        timestamp: pd.Timestamp,
        kind: str,
        stable_key: str,
        payload: Any,
    ) -> None:
        nonlocal serial
        event = _ScheduledEvent(
            timestamp=_utc(timestamp, name=f"{kind}.timestamp"),
            priority=EVENT_PRIORITY[kind],
            stable_key=stable_key,
            serial=serial,
            kind=kind,
            payload=payload,
        )
        serial += 1
        heapq.heappush(heap, event.heap_key())

    decision_batches: dict[
        pd.Timestamp,
        list[tuple[DecisionCandidate, ...]],
    ] = {}
    for group in groups:
        decision_batches.setdefault(group[0].decision_time, []).append(group)
    for decision_time, batch in sorted(decision_batches.items()):
        ordered_batch = tuple(
            sorted(
                batch,
                key=lambda group: (
                    group[0].symbol,
                    group[0].arbitration_group_id,
                ),
            )
        )
        schedule(
            decision_time,
            "DECISION",
            f"batch|{decision_time.isoformat()}",
            ordered_batch,
        )
    schedule(fold.test_end, "FOLD_END", f"fold-{fold.fold_index}", None)

    pending: dict[str, _Idea] = {}
    active: dict[str, _Idea] = {}
    cooldown_until: dict[str, pd.Timestamp] = {}
    duplicate_seen: dict[str, pd.Timestamp] = {}
    daily_realized: dict[str, float] = {}
    ideas: list[_Idea] = []
    events: list[dict[str, Any]] = []

    def emit(
        at: pd.Timestamp,
        priority: int,
        kind: str,
        *,
        candidate: Optional[DecisionCandidate] = None,
        reason: str,
        state_before: Optional[str] = None,
        state_after: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        row: dict[str, Any] = {
            "schema": GLOBAL_POLICY_REPLAY_SCHEMA,
            "replay_id": replay_id,
            "arm": spec.arm,
            "timestamp": at.isoformat(),
            "event_priority": priority,
            "kind": kind,
            "fold_index": fold.fold_index,
            "symbol": candidate.symbol if candidate is not None else None,
            "arbitration_group_id": (
                candidate.arbitration_group_id if candidate is not None else None
            ),
            "decision_event_id": (
                candidate.decision_event_id if candidate is not None else None
            ),
            "parent_opportunity_id": (
                candidate.parent_opportunity_id if candidate is not None else None
            ),
            "reason": reason,
            "state_before": state_before,
            "state_after": state_after,
        }
        if details:
            row.update(details)
        events.append(row)

    def record_duplicate(candidate: DecisionCandidate, at: pd.Timestamp) -> None:
        key = _duplicate_key(candidate, config.duplicate_key_policy)
        if key is not None:
            duplicate_seen[key] = at

    def exposures() -> dict[str, str]:
        result = {
            symbol: idea.candidate.side for symbol, idea in active.items()
        }
        if config.correlation_include_pending:
            result.update(
                {
                    symbol: idea.candidate.side
                    for symbol, idea in pending.items()
                }
            )
        return result

    def finalize_pending(
        idea: _Idea,
        at: pd.Timestamp,
        *,
        status: str,
        reason: str,
    ) -> None:
        candidate = idea.candidate
        if pending.get(candidate.symbol) is not idea:
            return
        pending.pop(candidate.symbol)
        idea.status = status
        idea.terminal_reason = reason
        if config.cooldown_after_pending_terminal:
            cooldown_until[candidate.symbol] = (
                at + config.cooldown_after_pending_terminal
            )
        emit(
            at,
            EVENT_PRIORITY["PENDING_TERMINAL"],
            "PENDING_TERMINAL",
            candidate=candidate,
            reason=reason,
            state_before="PENDING",
            state_after=status,
        )

    while heap:
        _, priority, _, _, scheduled = heapq.heappop(heap)
        at = scheduled.timestamp
        kind = scheduled.kind

        if kind == "FOLD_END":
            for symbol in sorted(pending):
                idea = pending[symbol]
                idea.status = "CENSORED_PENDING_FOLD_END"
                idea.terminal_reason = "fold boundary; no pending carry"
                emit(
                    at,
                    priority,
                    "FOLD_END_PENDING_CENSOR",
                    candidate=idea.candidate,
                    reason=idea.terminal_reason,
                    state_before="PENDING",
                    state_after=idea.status,
                )
            for symbol in sorted(active):
                idea = active[symbol]
                idea.status = "CENSORED_ACTIVE_FOLD_END"
                idea.terminal_reason = "fold boundary; no active carry"
                if idea.outcome is not None:
                    idea.partial_gross_r = idea.outcome.net_r
                emit(
                    at,
                    priority,
                    "FOLD_END_ACTIVE_CENSOR",
                    candidate=idea.candidate,
                    reason=idea.terminal_reason,
                    state_before="ACTIVE",
                    state_after=idea.status,
                )
            pending.clear()
            active.clear()
            break

        if kind == "PENDING_TERMINAL":
            idea, status, reason = scheduled.payload
            finalize_pending(idea, at, status=status, reason=reason)
            continue

        if kind == "POSITION_EXIT":
            idea, outcome = scheduled.payload
            candidate = idea.candidate
            if active.get(candidate.symbol) is not idea:
                continue
            active.pop(candidate.symbol)
            idea.status = "CLOSED"
            idea.terminal_reason = "split simulator reached a terminal outcome"
            idea.exit_time = at
            idea.gross_r = float(outcome.net_r)
            idea.outcome = outcome
            if cost_profile is not None:
                estimate = estimate_realized_cost_r(
                    cost_profile,
                    {
                        "symbol": candidate.symbol,
                        "side": candidate.side,
                        "entry": idea.entry_price,
                        "stop": candidate.stop,
                        "entry_time": idea.fill_time,
                        "exit_time": at,
                    },
                )
                idea.transaction_cost_r = estimate.total_cost_r
                idea.net_after_cost_r = idea.gross_r - estimate.total_cost_r
                idea.cost_fields.update(estimate.to_dict())
            basis_value = (
                idea.gross_r
                if config.daily_cap_basis == "gross_r"
                else idea.net_after_cost_r
            )
            if basis_value is not None:
                day = at.date().isoformat()
                daily_realized[day] = daily_realized.get(day, 0.0) + basis_value
            if config.cooldown_after_exit:
                cooldown_until[candidate.symbol] = at + config.cooldown_after_exit
            emit(
                at,
                priority,
                "POSITION_EXIT",
                candidate=candidate,
                reason=idea.terminal_reason,
                state_before="ACTIVE",
                state_after="CLOSED",
                details={
                    "gross_net_r": idea.gross_r,
                    "net_after_cost_r": idea.net_after_cost_r,
                },
            )
            continue

        if kind == "FILL":
            idea, observation = scheduled.payload
            candidate = idea.candidate
            if pending.get(candidate.symbol) is not idea:
                continue
            pending.pop(candidate.symbol)
            idea.status = "ACTIVE"
            idea.observation_status = observation.status
            idea.observation_reason = observation.reason
            idea.fill_time = at
            idea.entry_price = observation.entry_price
            active[candidate.symbol] = idea
            if config.duplicate_record_on == "fill":
                record_duplicate(candidate, at)
            emit(
                at,
                priority,
                "FILL",
                candidate=candidate,
                reason=observation.reason,
                state_before="PENDING",
                state_after="ACTIVE",
                details={"entry_price": idea.entry_price},
            )
            frame = frames[candidate.symbol]
            minute = pd.Timedelta(minutes=1)
            start = at
            outcome_bars = frame.loc[
                (frame.index >= start)
                & ((frame.index + minute) < fold.test_end)
            ]
            if not outcome_bars.empty:
                outcome = simulate_split_outcome(
                    side=candidate.side,
                    entry=float(idea.entry_price),
                    stop=candidate.stop,
                    tp_prices=candidate.tp_prices,
                    bars=outcome_bars,
                    weights=config.weights,
                    intrabar_policy=config.intrabar_policy,
                )
                idea.outcome = outcome
                if outcome.status == "CLOSED" and outcome.exit_time is not None:
                    exit_at = _utc(
                        outcome.exit_time,
                        name="outcome.exit_time",
                    ) + minute
                    schedule(
                        exit_at,
                        "POSITION_EXIT",
                        idea.setup_id,
                        (idea, outcome),
                    )
            continue

        if kind != "DECISION":
            raise GlobalReplayError(f"unknown event kind {kind}")

        batch_groups = tuple(scheduled.payload)
        exposure_snapshot = exposures()
        decision_batch_id = _canonical_hash(
            {
                "replay_id": replay_id,
                "fold_index": fold.fold_index,
                "decision_time": at.isoformat(),
                "arbitration_group_ids": sorted(
                    group[0].arbitration_group_id
                    for group in batch_groups
                ),
            }
        )[:24]
        proposals: list[dict[str, Any]] = []
        for group in batch_groups:
            (
                selected,
                selected_score,
                selected_penalty,
                alternatives,
            ) = _selection(
                group,
                spec=spec,
                fold=fold,
                cost_profile=cost_profile,
                correlation_profile=correlation_profile,
                exposures=exposure_snapshot,
            )
            rejection: Optional[str] = None
            if selected.symbol in pending:
                rejection = "SYMBOL_HAS_PENDING_IDEA"
            elif selected.symbol in active:
                rejection = "SYMBOL_HAS_ACTIVE_IDEA"
            elif at < cooldown_until.get(selected.symbol, at):
                rejection = "SYMBOL_COOLDOWN"
            elif config.daily_loss_cap_r is not None:
                realized = daily_realized.get(at.date().isoformat(), 0.0)
                if realized <= -config.daily_loss_cap_r:
                    rejection = "DAILY_LOSS_CAP"
            duplicate_key = _duplicate_key(
                selected,
                config.duplicate_key_policy,
            )
            if rejection is None and duplicate_key is not None:
                previous = duplicate_seen.get(duplicate_key)
                if (
                    previous is not None
                    and at < previous + config.duplicate_window
                ):
                    rejection = "DUPLICATE_TRIGGER_WINDOW"
            external = invalidations.get(
                (selected.fold_index, selected.parent_opportunity_id)
            )
            if (
                rejection is None
                and external is not None
                and external.observed_at <= at
            ):
                rejection = "ALREADY_INVALIDATED"
            portfolio_score = (
                -float(selected.production_priority)
                if spec.candidate_policy == "production_priority_one"
                else float(selected_score)
            )
            proposals.append(
                {
                    "selected": selected,
                    "selected_score": selected_score,
                    "selected_penalty": selected_penalty,
                    "alternatives": alternatives,
                    "external": external,
                    "rejection": rejection,
                    "portfolio_score": portfolio_score,
                }
            )

        def portfolio_key(proposal: Mapping[str, Any]) -> tuple[Any, ...]:
            candidate = proposal["selected"]
            if spec.candidate_policy == "production_priority_one":
                return (
                    candidate.production_priority,
                    candidate.parent_opportunity_id,
                    candidate.symbol,
                    candidate.arbitration_group_id,
                )
            return (
                -float(proposal["portfolio_score"]),
                -float(candidate.expected_net_r),
                candidate.production_priority,
                candidate.parent_opportunity_id,
                candidate.symbol,
                candidate.arbitration_group_id,
            )

        proposals.sort(key=portfolio_key)
        available_slots: Optional[int] = None
        if config.max_active_positions is not None:
            occupied = len(active)
            if config.reserve_active_slot_for_pending:
                occupied += len(pending)
            available_slots = max(
                0,
                config.max_active_positions - occupied,
            )

        for portfolio_rank, proposal in enumerate(proposals, start=1):
            selected = proposal["selected"]
            selected_score = proposal["selected_score"]
            selected_penalty = proposal["selected_penalty"]
            alternatives = proposal["alternatives"]
            external = proposal["external"]
            rejection = proposal["rejection"]
            if (
                rejection is None
                and available_slots is not None
                and available_slots <= 0
            ):
                rejection = "GLOBAL_POSITION_CAPACITY"
            common_details = {
                "decision_batch_id": decision_batch_id,
                "decision_batch_size": len(proposals),
                "decision_batch_rank": portfolio_rank,
                "portfolio_policy_score": proposal["portfolio_score"],
                "portfolio_order_basis": (
                    "production_priority_then_stable_id"
                    if spec.candidate_policy
                    == "production_priority_one"
                    else "expected_net_minus_causal_correlation"
                ),
                "exposure_snapshot": dict(sorted(exposure_snapshot.items())),
                "alternatives": [dict(row) for row in alternatives],
            }
            if rejection is not None:
                emit(
                    at,
                    priority,
                    "DECISION_REJECTED",
                    candidate=selected,
                    reason=rejection,
                    state_before="NONE",
                    state_after="REJECTED",
                    details=common_details,
                )
                continue
            if available_slots is not None:
                available_slots -= 1

            setup_identity = {
                "replay_id": replay_id,
                "arm": spec.arm,
                "fold_index": selected.fold_index,
                "parent_opportunity_id": selected.parent_opportunity_id,
                "arbitration_group_id": selected.arbitration_group_id,
            }
            idea = _Idea(
                setup_id=_canonical_hash(setup_identity)[:24],
                candidate=selected,
                spec=spec,
                selected_score=selected_score,
                correlation_penalty_r=selected_penalty,
                alternatives=alternatives,
            )
            ideas.append(idea)
            pending[selected.symbol] = idea
            if config.duplicate_record_on == "arm":
                record_duplicate(selected, at)
            emit(
                at,
                priority,
                "DECISION_ARMED",
                candidate=selected,
                reason=(
                    "within-group rank-one winner accepted by the "
                    "same-timestamp portfolio batch"
                ),
                state_before="NONE",
                state_after="PENDING",
                details={
                    **common_details,
                    "selected_ranking_score": selected_score,
                    "correlation_penalty_r": selected_penalty,
                },
            )

            observation = observe_entry(
                selected,
                frames[selected.symbol],
                policy=spec.entry_policy,
                fold=fold,
            )
            idea.observation_status = observation.status
            idea.observation_reason = observation.reason
            event_at = (
                observation.observed_at
                or observation.effective_expires_at
            )
            terminal_status = {
                "NO_FILL": "NO_FILL_EXPIRED",
                "CENSORED": "CENSORED_PENDING_DATA",
                "STOP_INVALIDATED": "INVALIDATED_STOP",
                "TP1_INVALIDATED": "INVALIDATED_TP1",
                "AMBIGUOUS_FILL_STOP": "AMBIGUOUS_ENTRY_STOP",
            }.get(observation.status)
            if external is not None and external.observed_at <= event_at:
                schedule(
                    external.observed_at,
                    "PENDING_TERMINAL",
                    idea.setup_id,
                    (idea, "INVALIDATED_EXTERNAL", external.reason),
                )
            elif observation.status == "FILLED":
                schedule(
                    event_at,
                    "FILL",
                    idea.setup_id,
                    (idea, observation),
                )
            elif terminal_status is not None:
                schedule(
                    event_at,
                    "PENDING_TERMINAL",
                    idea.setup_id,
                    (idea, terminal_status, observation.reason),
                )
            else:
                raise GlobalReplayError(
                    "unsupported entry observation status "
                    f"{observation.status}"
                )

    return events, ideas


def run_global_policy_replay(
    *,
    decision_groups: Sequence[Sequence[DecisionCandidate]],
    folds: Sequence[FoldWindow],
    m1_by_symbol: Mapping[str, pd.DataFrame],
    data_snapshot_id: str,
    spec: ArmSpec,
    config: ReplayPolicyConfig = ReplayPolicyConfig(),
    cost_profile: Optional[CostProfile] = None,
    correlation_profile: Optional[PortfolioCorrelationProfile] = None,
    invalidations: Sequence[PendingInvalidation] = (),
) -> GlobalReplayResult:
    """Replay one candidate/entry policy over a global chronological clock."""

    snapshot_id = str(data_snapshot_id or "").strip()
    if not snapshot_id:
        raise GlobalReplayError("data_snapshot_id is required")
    ordered_folds = _validate_folds(folds)
    fold_map = {fold.fold_index: fold for fold in ordered_folds}
    net_policy = spec.candidate_policy == "net_rank_one"
    groups_by_fold = _validate_groups(
        decision_groups,
        fold_map,
        net_policy=net_policy,
    )
    frames = _normalize_m1_frames(m1_by_symbol)
    needed_symbols = {
        candidate.symbol
        for group in decision_groups
        for candidate in group
    }
    missing_symbols = sorted(needed_symbols - set(frames))
    if missing_symbols:
        raise GlobalReplayError(f"missing M1 frames for {missing_symbols}")
    if config.daily_cap_basis == "net_after_cost_r" and cost_profile is None:
        raise GlobalReplayError("net daily cap requires a cost profile")
    if net_policy and (cost_profile is None or correlation_profile is None):
        raise GlobalReplayError(
            "net_rank_one requires causal cost and correlation profiles"
        )
    cost_audit: Optional[Mapping[str, Any]] = None
    cost_usage: Optional[str] = None
    if cost_profile is not None:
        cost_audit = cost_profile.point_in_time_audit(
            [fold.test_start.to_pydatetime() for fold in ordered_folds]
        )
        cost_changes_policy = (
            net_policy or config.daily_cap_basis == "net_after_cost_r"
        )
        cost_usage = (
            "POINT_IN_TIME_POLICY_INPUT"
            if cost_changes_policy
            else "POST_HOC_STATIC_STRESS"
        )
        if (
            cost_changes_policy
            and not cost_audit["causal_for_all_fold_test_starts"]
        ):
            raise GlobalReplayError("cost profile is not causal for every fold")
    if correlation_profile is not None:
        earliest_start = ordered_folds[0].test_start
        trained_end = _utc(
            correlation_profile.trained_end_utc,
            name="correlation_profile.trained_end_utc",
        )
        if trained_end > earliest_start:
            raise GlobalReplayError(
                "correlation profile is not causal for every fold"
            )

    invalidation_map: dict[tuple[int, str], PendingInvalidation] = {}
    for invalidation in invalidations:
        key = (invalidation.fold_index, invalidation.parent_opportunity_id)
        if key in invalidation_map:
            raise GlobalReplayError("duplicate pending invalidation")
        invalidation_map[key] = invalidation

    experiment_payload = {
        "schema": GLOBAL_POLICY_REPLAY_SCHEMA,
        "data_snapshot_id": snapshot_id,
        "config": config.to_dict(),
        "folds": [
            {
                "fold_index": fold.fold_index,
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
            }
            for fold in ordered_folds
        ],
        "groups": sorted(
            (
                {
                    "fold_index": group[0].fold_index,
                    "arbitration_group_id": group[0].arbitration_group_id,
                    "parents": sorted(
                        candidate.parent_opportunity_id for candidate in group
                    ),
                }
                for group in decision_groups
            ),
            key=lambda row: (
                int(row["fold_index"]),
                str(row["arbitration_group_id"]),
            ),
        ),
        "invalidations": sorted(
            (
                {
                    "fold_index": invalidation.fold_index,
                    "parent_opportunity_id": (
                        invalidation.parent_opportunity_id
                    ),
                    "observed_at": invalidation.observed_at.isoformat(),
                    "reason": invalidation.reason,
                }
                for invalidation in invalidation_map.values()
            ),
            key=lambda row: (
                int(row["fold_index"]),
                str(row["parent_opportunity_id"]),
            ),
        ),
        "cost_profile_sha256": (
            cost_profile.profile_sha256 if cost_profile is not None else None
        ),
        "correlation_profile_sha256": (
            correlation_profile.profile_sha256
            if correlation_profile is not None
            else None
        ),
    }
    experiment_input_id = _canonical_hash(experiment_payload)[:24]
    replay_id = _canonical_hash(
        {**experiment_payload, "arm": spec.arm}
    )[:24]
    all_events: list[dict[str, Any]] = []
    all_ideas: list[_Idea] = []
    for fold in ordered_folds:
        events, ideas = _run_fold(
            replay_id=replay_id,
            fold=fold,
            groups=groups_by_fold[fold.fold_index],
            frames=frames,
            spec=spec,
            config=config,
            cost_profile=cost_profile,
            correlation_profile=correlation_profile,
            invalidations=invalidation_map,
        )
        all_events.extend(events)
        all_ideas.extend(ideas)
    all_events.sort(
        key=lambda row: (
            pd.Timestamp(row["timestamp"]),
            int(row["event_priority"]),
            str(row.get("symbol") or ""),
            str(row.get("arbitration_group_id") or ""),
            str(row.get("parent_opportunity_id") or ""),
            str(row["kind"]),
        )
    )
    for event_sequence, row in enumerate(all_events, start=1):
        row["event_sequence"] = event_sequence
    setup_rows = [idea.to_row(replay_id) for idea in all_ideas]
    setup_rows.sort(
        key=lambda row: (
            int(row["fold_index"]),
            str(row["decision_time"]),
            str(row["symbol"]),
            str(row["setup_id"]),
        )
    )
    metrics = _summarize(setup_rows, all_events)
    metrics["experiment_input_id"] = experiment_input_id
    metrics["data_snapshot_id"] = snapshot_id
    metrics["cost_profile_point_in_time"] = cost_audit
    metrics["cost_profile_usage"] = cost_usage
    metrics["correlation_profile_id"] = (
        correlation_profile.profile_id
        if correlation_profile is not None
        else None
    )
    return GlobalReplayResult(
        experiment_input_id=experiment_input_id,
        replay_id=replay_id,
        arm=spec.arm,
        events=tuple(all_events),
        setups=tuple(setup_rows),
        metrics=metrics,
    )


def run_four_arm_global_replay(
    *,
    decision_groups: Sequence[Sequence[DecisionCandidate]],
    folds: Sequence[FoldWindow],
    m1_by_symbol: Mapping[str, pd.DataFrame],
    data_snapshot_id: str,
    config: ReplayPolicyConfig = ReplayPolicyConfig(),
    cost_profile: CostProfile,
    correlation_profile: PortfolioCorrelationProfile,
    invalidations: Sequence[PendingInvalidation] = (),
) -> tuple[GlobalReplayResult, ...]:
    """Run the fixed two-by-two factorial with independent state per arm."""

    results = tuple(
        run_global_policy_replay(
            decision_groups=decision_groups,
            folds=folds,
            m1_by_symbol=m1_by_symbol,
            data_snapshot_id=data_snapshot_id,
            spec=spec,
            config=config,
            cost_profile=cost_profile,
            correlation_profile=correlation_profile,
            invalidations=invalidations,
        )
        for spec in four_arm_specs()
    )
    # The factorial only isolates the policy change when every arm consumed
    # exactly one identical input population.
    input_ids = {result.experiment_input_id for result in results}
    if len(input_ids) != 1:
        raise GlobalReplayError(
            "four-arm factorial arms do not share one experiment input id"
        )
    if len({result.replay_id for result in results}) != len(results):
        raise GlobalReplayError("four-arm factorial replay ids are not distinct")
    return results


__all__ = [
    "EVENT_PRIORITY",
    "GLOBAL_POLICY_REPLAY_SCHEMA",
    "GlobalReplayError",
    "GlobalReplayResult",
    "PendingInvalidation",
    "ReplayPolicyConfig",
    "UnsupportedReplayFeature",
    "run_four_arm_global_replay",
    "run_global_policy_replay",
]
