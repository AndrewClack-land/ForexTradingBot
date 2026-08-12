"""Research-only paired candidate and entry-policy arms.

The module deliberately sits outside the live path and the current
counterfactual runner.  It freezes one candidate at the higher-timeframe
decision timestamp and only then observes that candidate under one of two M1
entry policies.  In particular, a candidate that would have filled in the
future can never replace the candidate chosen at decision time.

Each call covers one within-symbol decision group only.  The module carries
no pending-order or active-position state across calls; chronological
one-pending enforcement belongs to a future event replay.

``planned_limit_touch`` is an OHLC sensitivity, not evidence of an exact
limit fill.  Its audit rows retain that qualification and surface a same-bar
entry/stop collision instead of silently imposing an intrabar path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Literal, Mapping, Optional, Sequence

import pandas as pd


PAIRED_ENTRY_ARMS_SCHEMA = "paired-entry-arms/v1"

CandidatePolicy = Literal[
    "production_priority_one",
    "net_rank_one",
]
EntryPolicy = Literal[
    "m1_open_in_range",
    "planned_limit_touch",
]
Side = Literal["LONG", "SHORT"]

CANDIDATE_POLICIES: tuple[CandidatePolicy, ...] = (
    "production_priority_one",
    "net_rank_one",
)
ENTRY_POLICIES: tuple[EntryPolicy, ...] = (
    "m1_open_in_range",
    "planned_limit_touch",
)


class PairedEntryArmError(ValueError):
    """Raised when a paired arm cannot be evaluated causally."""


def _utc(value: Any, *, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise PairedEntryArmError(f"{name} must be a timestamp") from exc
    if timestamp.tzinfo is None:
        raise PairedEntryArmError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _finite(value: Any, *, name: str, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PairedEntryArmError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise PairedEntryArmError(f"{name} must be {qualifier}")
    return number


def _optional_finite(value: Any, *, name: str) -> Optional[float]:
    if value is None:
        return None
    return _finite(value, name=name)


def _iso(value: Optional[pd.Timestamp]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FoldWindow:
    """OOS interval whose end is an exclusive entry-policy cutoff."""

    fold_index: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __post_init__(self) -> None:
        if isinstance(self.fold_index, bool) or int(self.fold_index) < 0:
            raise PairedEntryArmError("fold_index must be a nonnegative integer")
        start = _utc(self.test_start, name="test_start")
        end = _utc(self.test_end, name="test_end")
        if end <= start:
            raise PairedEntryArmError("test_end must follow test_start")
        object.__setattr__(self, "fold_index", int(self.fold_index))
        object.__setattr__(self, "test_start", start)
        object.__setattr__(self, "test_end", end)


@dataclass(frozen=True)
class DecisionCandidate:
    """Immutable information available at one context decision.

    There is intentionally no fill, outcome, or realized-R field.  Ranking
    consumes only ``expected_net_r`` plus causal provenance timestamps.
    """

    parent_opportunity_id: str
    decision_event_id: str
    arbitration_group_id: str
    fold_index: int
    symbol: str
    trigger_kind: str
    side: Side
    bar_close_time: pd.Timestamp
    decision_time: pd.Timestamp
    expires_at: pd.Timestamp
    entry_min: float
    entry_max: float
    planned_entry: float
    stop: float
    tp_prices: tuple[float, ...]
    production_priority: int
    expected_gross_r: Optional[float] = None
    estimated_cost_r: Optional[float] = None
    expected_net_r: Optional[float] = None
    ranking_basis: Optional[str] = None
    optimizer_rank: Optional[int] = None
    model_id: Optional[str] = None
    model_train_end: Optional[pd.Timestamp] = None
    known_at: Optional[pd.Timestamp] = None
    cost_profile_id: Optional[str] = None
    cost_profile_sha256: Optional[str] = None
    cost_profile_created_at_utc: Optional[pd.Timestamp] = None
    cost_profile_measured_through: Optional[pd.Timestamp] = None

    def __post_init__(self) -> None:
        for field_name in (
            "parent_opportunity_id",
            "decision_event_id",
            "arbitration_group_id",
            "symbol",
            "trigger_kind",
        ):
            value = str(getattr(self, field_name) or "").strip()
            if not value:
                raise PairedEntryArmError(f"{field_name} is required")
            if field_name == "symbol":
                value = value.upper()
            if field_name == "trigger_kind":
                value = value.lower()
            object.__setattr__(self, field_name, value)

        if isinstance(self.fold_index, bool) or int(self.fold_index) < 0:
            raise PairedEntryArmError("fold_index must be a nonnegative integer")
        object.__setattr__(self, "fold_index", int(self.fold_index))

        side = str(self.side or "").strip().upper()
        if side not in {"LONG", "SHORT"}:
            raise PairedEntryArmError("side must be LONG or SHORT")
        object.__setattr__(self, "side", side)

        bar_close = _utc(self.bar_close_time, name="bar_close_time")
        decision = _utc(self.decision_time, name="decision_time")
        expires = _utc(self.expires_at, name="expires_at")
        if bar_close > decision:
            raise PairedEntryArmError("bar_close_time cannot follow decision_time")
        if expires <= decision:
            raise PairedEntryArmError("expires_at must follow decision_time")
        object.__setattr__(self, "bar_close_time", bar_close)
        object.__setattr__(self, "decision_time", decision)
        object.__setattr__(self, "expires_at", expires)

        entry_min = _finite(self.entry_min, name="entry_min", positive=True)
        entry_max = _finite(self.entry_max, name="entry_max", positive=True)
        planned = _finite(
            self.planned_entry,
            name="planned_entry",
            positive=True,
        )
        stop = _finite(self.stop, name="stop", positive=True)
        if entry_max < entry_min:
            raise PairedEntryArmError("entry_max cannot be below entry_min")
        if not entry_min <= planned <= entry_max:
            raise PairedEntryArmError("planned_entry must be inside entry range")

        targets = tuple(
            _finite(value, name="tp_price", positive=True) for value in self.tp_prices
        )
        if not targets:
            raise PairedEntryArmError("tp_prices cannot be empty")
        if side == "LONG":
            if not stop < entry_min or not all(entry_max < tp for tp in targets):
                raise PairedEntryArmError(
                    "LONG requires stop < entry range < every target"
                )
            if tuple(sorted(targets)) != targets:
                raise PairedEntryArmError("LONG targets must be ascending")
        else:
            if not stop > entry_max or not all(entry_min > tp for tp in targets):
                raise PairedEntryArmError(
                    "SHORT requires every target < entry range < stop"
                )
            if tuple(sorted(targets, reverse=True)) != targets:
                raise PairedEntryArmError("SHORT targets must be descending")
        object.__setattr__(self, "entry_min", entry_min)
        object.__setattr__(self, "entry_max", entry_max)
        object.__setattr__(self, "planned_entry", planned)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "tp_prices", targets)

        if (
            isinstance(self.production_priority, bool)
            or int(self.production_priority) < 0
        ):
            raise PairedEntryArmError(
                "production_priority must be a nonnegative integer"
            )
        object.__setattr__(
            self,
            "production_priority",
            int(self.production_priority),
        )
        if self.optimizer_rank is not None:
            if isinstance(self.optimizer_rank, bool) or int(self.optimizer_rank) < 1:
                raise PairedEntryArmError("optimizer_rank must be a positive integer")
            object.__setattr__(self, "optimizer_rank", int(self.optimizer_rank))

        for field_name in (
            "expected_gross_r",
            "estimated_cost_r",
            "expected_net_r",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_finite(getattr(self, field_name), name=field_name),
            )

        train_end = (
            _utc(self.model_train_end, name="model_train_end")
            if self.model_train_end is not None
            else None
        )
        known_at = (
            _utc(self.known_at, name="known_at") if self.known_at is not None else None
        )
        if train_end is not None and train_end > decision:
            raise PairedEntryArmError(
                "model_train_end cannot follow context decision_time"
            )
        if known_at is not None and known_at > decision:
            raise PairedEntryArmError("known_at cannot follow context decision_time")
        object.__setattr__(self, "model_train_end", train_end)
        object.__setattr__(self, "known_at", known_at)

        profile_created = (
            _utc(
                self.cost_profile_created_at_utc,
                name="cost_profile_created_at_utc",
            )
            if self.cost_profile_created_at_utc is not None
            else None
        )
        profile_measured_through = (
            _utc(
                self.cost_profile_measured_through,
                name="cost_profile_measured_through",
            )
            if self.cost_profile_measured_through is not None
            else None
        )
        if (
            profile_created is not None
            and profile_measured_through is not None
            and profile_measured_through > profile_created
        ):
            raise PairedEntryArmError(
                "cost_profile_measured_through cannot follow "
                "cost_profile_created_at_utc"
            )
        object.__setattr__(
            self,
            "cost_profile_created_at_utc",
            profile_created,
        )
        object.__setattr__(
            self,
            "cost_profile_measured_through",
            profile_measured_through,
        )


@dataclass(frozen=True)
class ArmSpec:
    candidate_policy: CandidatePolicy
    entry_policy: EntryPolicy

    @property
    def arm(self) -> str:
        return f"{self.candidate_policy}__{self.entry_policy}"

    @property
    def sensitivity_only(self) -> bool:
        return self.entry_policy == "planned_limit_touch"


@dataclass(frozen=True)
class EntryObservation:
    status: str
    reason: str
    effective_expires_at: pd.Timestamp
    entry_time: Optional[pd.Timestamp] = None
    entry_price: Optional[float] = None
    observed_at: Optional[pd.Timestamp] = None
    touch_bar_open: Optional[pd.Timestamp] = None
    same_bar_stop_ambiguous: bool = False
    same_bar_tp1_ambiguous: bool = False
    fold_clipped: bool = False


@dataclass(frozen=True)
class ArmDecision:
    spec: ArmSpec
    candidate: DecisionCandidate
    observation: EntryObservation
    computed_optimizer_rank: Optional[int]

    @property
    def arm(self) -> str:
        return self.spec.arm

    @property
    def status(self) -> str:
        return self.observation.status

    def to_audit_row(self) -> dict[str, Any]:
        candidate = self.candidate
        observation = self.observation
        entry_time_precision: Optional[str] = None
        if observation.entry_time is not None:
            entry_time_precision = (
                "M1_RANGE_WINDOW" if self.spec.sensitivity_only else "M1_OPEN_EXACT"
            )
        identity = {
            "schema": PAIRED_ENTRY_ARMS_SCHEMA,
            "arm": self.arm,
            "arbitration_group_id": candidate.arbitration_group_id,
            "parent_opportunity_id": candidate.parent_opportunity_id,
        }
        return {
            **identity,
            "arm_decision_id": _canonical_hash(identity)[:24],
            "candidate_policy": self.spec.candidate_policy,
            "entry_policy": self.spec.entry_policy,
            "arbitration_scope": "single_decision_only_no_timeline_state",
            "sensitivity_only": self.spec.sensitivity_only,
            "decision_event_id": candidate.decision_event_id,
            "fold_index": candidate.fold_index,
            "symbol": candidate.symbol,
            "parent_trigger_kind": candidate.trigger_kind,
            "side": candidate.side,
            "bar_close_time": candidate.bar_close_time.isoformat(),
            "context_decision_time": candidate.decision_time.isoformat(),
            "context_expires_at": candidate.expires_at.isoformat(),
            # These policies arm directly from the parent context.  They do
            # not claim a completed M1 confirmation trigger.
            "m1_trigger_bar_open": None,
            "m1_trigger_time": None,
            "execution_eligible_at": candidate.decision_time.isoformat(),
            "entry_time": _iso(observation.entry_time),
            "entry_time_precision": entry_time_precision,
            "entry_price": observation.entry_price,
            "touch_bar_open": _iso(observation.touch_bar_open),
            "observed_at": _iso(observation.observed_at),
            "effective_expires_at": observation.effective_expires_at.isoformat(),
            "status": observation.status,
            "reason": observation.reason,
            "same_bar_stop_ambiguous": observation.same_bar_stop_ambiguous,
            "same_bar_tp1_ambiguous": observation.same_bar_tp1_ambiguous,
            "fold_clipped": observation.fold_clipped,
            "production_priority": candidate.production_priority,
            "optimizer_rank": (
                self.computed_optimizer_rank
                if self.computed_optimizer_rank is not None
                else candidate.optimizer_rank
            ),
            "expected_gross_r": candidate.expected_gross_r,
            "estimated_cost_r": candidate.estimated_cost_r,
            "expected_net_r": candidate.expected_net_r,
            "ranking_basis": (
                "production_priority"
                if self.spec.candidate_policy == "production_priority_one"
                else candidate.ranking_basis or "expected_net_r"
            ),
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


def four_arm_specs() -> tuple[ArmSpec, ...]:
    """Return the fixed two-by-two paired research design."""

    return tuple(
        ArmSpec(candidate_policy, entry_policy)
        for candidate_policy in CANDIDATE_POLICIES
        for entry_policy in ENTRY_POLICIES
    )


def _validate_decision_group(
    candidates: Sequence[DecisionCandidate],
    fold: FoldWindow,
) -> tuple[DecisionCandidate, ...]:
    group = tuple(candidates)
    if not group:
        raise PairedEntryArmError("at least one candidate is required")
    if len({candidate.parent_opportunity_id for candidate in group}) != len(group):
        raise PairedEntryArmError("parent_opportunity_id values must be unique")
    expected = group[0]
    for candidate in group:
        if candidate.fold_index != fold.fold_index:
            raise PairedEntryArmError("candidate fold does not match FoldWindow")
        if not fold.test_start <= candidate.decision_time < fold.test_end:
            raise PairedEntryArmError("context decision must start inside OOS fold")
        for name in (
            "arbitration_group_id",
            "decision_event_id",
            "symbol",
            "decision_time",
        ):
            if getattr(candidate, name) != getattr(expected, name):
                raise PairedEntryArmError(f"decision candidates disagree on {name}")
    return group


def _validate_net_rank_provenance(
    candidate: DecisionCandidate,
    *,
    fold: FoldWindow,
) -> None:
    missing_scores = [
        name
        for name in (
            "expected_gross_r",
            "estimated_cost_r",
            "expected_net_r",
        )
        if getattr(candidate, name) is None
    ]
    if missing_scores:
        raise PairedEntryArmError(
            f"{candidate.parent_opportunity_id}: net rank requires "
            f"{', '.join(missing_scores)}"
        )
    if candidate.model_train_end is None or candidate.known_at is None:
        raise PairedEntryArmError(
            f"{candidate.parent_opportunity_id}: net rank requires "
            "model_train_end and known_at"
        )
    if candidate.model_train_end > fold.test_start:
        raise PairedEntryArmError(
            f"{candidate.parent_opportunity_id}: model_train_end must not "
            "enter the OOS fold"
        )
    if not str(candidate.model_id or "").strip():
        raise PairedEntryArmError(
            f"{candidate.parent_opportunity_id}: model_id is required"
        )
    ranking_basis = str(candidate.ranking_basis or "").strip()
    if ranking_basis != "expected_net_r":
        raise PairedEntryArmError(
            f"{candidate.parent_opportunity_id}: ranking_basis must be "
            "expected_net_r"
        )
    if not str(candidate.cost_profile_id or "").strip():
        raise PairedEntryArmError(
            f"{candidate.parent_opportunity_id}: cost_profile_id is required"
        )
    profile_hash = str(candidate.cost_profile_sha256 or "").strip().lower()
    if len(profile_hash) != 64 or any(
        character not in "0123456789abcdef" for character in profile_hash
    ):
        raise PairedEntryArmError(
            f"{candidate.parent_opportunity_id}: cost_profile_sha256 must "
            "be a 64-character hexadecimal digest"
        )
    if (
        candidate.cost_profile_created_at_utc is None
        or candidate.cost_profile_measured_through is None
    ):
        raise PairedEntryArmError(
            f"{candidate.parent_opportunity_id}: net rank requires "
            "cost_profile_created_at_utc and cost_profile_measured_through"
        )
    if candidate.cost_profile_created_at_utc > fold.test_start:
        raise PairedEntryArmError(
            f"{candidate.parent_opportunity_id}: cost profile must be "
            "created before the OOS fold"
        )
    if candidate.cost_profile_measured_through > fold.test_start:
        raise PairedEntryArmError(
            f"{candidate.parent_opportunity_id}: cost profile measurements "
            "must not enter the OOS fold"
        )
    if candidate.cost_profile_created_at_utc > candidate.known_at:
        raise PairedEntryArmError(
            f"{candidate.parent_opportunity_id}: cost profile cannot be "
            "created after the ranking score is known"
        )
    estimated_cost = float(candidate.estimated_cost_r)
    if estimated_cost < 0.0:
        raise PairedEntryArmError(
            f"{candidate.parent_opportunity_id}: estimated_cost_r cannot "
            "be negative"
        )
    expected = float(candidate.expected_gross_r) - estimated_cost
    if not math.isclose(
        float(candidate.expected_net_r),
        expected,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise PairedEntryArmError(
            f"{candidate.parent_opportunity_id}: expected_net_r must equal "
            "expected_gross_r - estimated_cost_r"
        )


def _net_order_key(candidate: DecisionCandidate) -> tuple[float, int, str]:
    if candidate.expected_net_r is None:
        raise PairedEntryArmError(
            f"{candidate.parent_opportunity_id}: expected_net_r is required"
        )
    return (
        -candidate.expected_net_r,
        candidate.production_priority,
        candidate.parent_opportunity_id,
    )


def select_candidate(
    candidates: Sequence[DecisionCandidate],
    *,
    policy: CandidatePolicy,
    fold: FoldWindow,
) -> DecisionCandidate:
    """Select exactly one candidate using decision-time fields only.

    No M1 bars or fill observations are accepted by this function.  The net
    arm fails closed unless every alternative carries a score whose model,
    cost profile, and publication timestamps are causal for the OOS fold.
    """

    group = _validate_decision_group(candidates, fold)
    if policy == "production_priority_one":
        return min(
            group,
            key=lambda candidate: (
                candidate.production_priority,
                candidate.parent_opportunity_id,
            ),
        )
    if policy != "net_rank_one":
        raise PairedEntryArmError(f"unsupported candidate policy: {policy}")
    for candidate in group:
        _validate_net_rank_provenance(candidate, fold=fold)
    profile_identities = {
        (
            str(candidate.cost_profile_id).strip(),
            str(candidate.cost_profile_sha256).strip().lower(),
            candidate.cost_profile_created_at_utc,
            candidate.cost_profile_measured_through,
        )
        for candidate in group
    }
    if len(profile_identities) != 1:
        raise PairedEntryArmError(
            "net-rank alternatives must use one identical cost profile"
        )
    return min(group, key=_net_order_key)


def _computed_net_ranks(
    candidates: Sequence[DecisionCandidate],
) -> Mapping[str, int]:
    if any(candidate.expected_net_r is None for candidate in candidates):
        return {}
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate.expected_net_r),
            candidate.production_priority,
            candidate.parent_opportunity_id,
        ),
    )
    return {
        candidate.parent_opportunity_id: index
        for index, candidate in enumerate(ordered, start=1)
    }


def _normalize_m1(m1: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    if not required.issubset(m1.columns):
        missing = sorted(required - set(m1.columns))
        raise PairedEntryArmError(f"M1 bars are missing columns: {missing}")
    frame = m1.copy().sort_index(kind="stable")
    if frame.index.has_duplicates:
        raise PairedEntryArmError("M1 timestamps must be unique")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        raise PairedEntryArmError("M1 index must be timezone-aware")
    frame.index = index.tz_convert("UTC")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    values = frame[list(required)].astype(float)
    if not values.map(math.isfinite).all(axis=None):
        raise PairedEntryArmError("M1 bars contain non-finite prices")
    invalid = (
        (frame["high"] < frame["low"])
        | (frame["high"] < frame[["open", "close"]].max(axis=1))
        | (frame["low"] > frame[["open", "close"]].min(axis=1))
    )
    if invalid.any():
        raise PairedEntryArmError("M1 bars contain impossible OHLC candles")
    return frame


def _first_expected_open(decision_time: pd.Timestamp) -> pd.Timestamp:
    # The M1 candle stamped exactly at the context decision has already
    # opened.  This mirrors the production backtest's searchsorted(...,
    # side="right") rule.
    return decision_time.floor("min") + pd.Timedelta(minutes=1)


def _stop_touched(candidate: DecisionCandidate, row: pd.Series) -> bool:
    if candidate.side == "LONG":
        return float(row["low"]) <= candidate.stop
    return float(row["high"]) >= candidate.stop


def _tp1_touched(candidate: DecisionCandidate, row: pd.Series) -> bool:
    if candidate.side == "LONG":
        return float(row["high"]) >= candidate.tp_prices[0]
    return float(row["low"]) <= candidate.tp_prices[0]


def _planned_entry_touched(
    candidate: DecisionCandidate,
    row: pd.Series,
) -> bool:
    return float(row["low"]) <= candidate.planned_entry <= float(row["high"])


def _censored_or_expired(
    *,
    candidate: DecisionCandidate,
    fold: FoldWindow,
    deadline: pd.Timestamp,
    complete_horizon: bool,
    reason: str,
) -> EntryObservation:
    fold_clipped = fold.test_end < candidate.expires_at
    if fold_clipped:
        return EntryObservation(
            status="CENSORED",
            reason="fold end precedes context expiry",
            effective_expires_at=deadline,
            fold_clipped=True,
        )
    if not complete_horizon:
        return EntryObservation(
            status="CENSORED",
            reason=reason,
            effective_expires_at=deadline,
        )
    return EntryObservation(
        status="NO_FILL",
        reason="entry TTL expired",
        effective_expires_at=deadline,
    )


def observe_entry(
    candidate: DecisionCandidate,
    m1: pd.DataFrame,
    *,
    policy: EntryPolicy,
    fold: FoldWindow,
) -> EntryObservation:
    """Observe one already-selected candidate under an M1 entry policy."""

    _validate_decision_group((candidate,), fold)
    if policy not in ENTRY_POLICIES:
        raise PairedEntryArmError(f"unsupported entry policy: {policy}")
    frame = _normalize_m1(m1)
    deadline = min(candidate.expires_at, fold.test_end)
    fold_clipped = fold.test_end < candidate.expires_at
    one_minute = pd.Timedelta(minutes=1)
    expected_open = _first_expected_open(candidate.decision_time)
    if (
        policy == "planned_limit_touch"
        and candidate.decision_time != candidate.decision_time.floor("min")
    ):
        return EntryObservation(
            status="CENSORED",
            reason="context decision starts inside an unobservable M1 range",
            effective_expires_at=deadline,
            fold_clipped=fold_clipped,
        )

    post = frame.loc[(frame.index > candidate.decision_time) & (frame.index < deadline)]
    for bar_open, row in post.iterrows():
        if bar_open != expected_open:
            return EntryObservation(
                status="CENSORED",
                reason=(f"M1 horizon gap before {pd.Timestamp(bar_open).isoformat()}"),
                effective_expires_at=deadline,
                fold_clipped=fold_clipped,
            )

        if policy == "m1_open_in_range":
            price = float(row["open"])
            if candidate.entry_min <= price <= candidate.entry_max:
                return EntryObservation(
                    status="FILLED",
                    reason="first post-decision M1 open inside entry range",
                    effective_expires_at=deadline,
                    entry_time=pd.Timestamp(bar_open),
                    entry_price=price,
                    observed_at=pd.Timestamp(bar_open),
                    touch_bar_open=pd.Timestamp(bar_open),
                    fold_clipped=fold_clipped,
                )
        else:
            bar_close = pd.Timestamp(bar_open) + one_minute
            if bar_close > deadline:
                break
            touched_entry = _planned_entry_touched(candidate, row)
            touched_stop = _stop_touched(candidate, row)
            touched_tp1 = _tp1_touched(candidate, row)
            if touched_entry and touched_stop:
                return EntryObservation(
                    status="AMBIGUOUS_FILL_STOP",
                    reason=(
                        "planned entry and stop occur in one M1 range; "
                        "intrabar order is unknown"
                    ),
                    effective_expires_at=deadline,
                    entry_time=pd.Timestamp(bar_open),
                    entry_price=candidate.planned_entry,
                    observed_at=bar_close,
                    touch_bar_open=pd.Timestamp(bar_open),
                    same_bar_stop_ambiguous=True,
                    same_bar_tp1_ambiguous=touched_tp1,
                    fold_clipped=fold_clipped,
                )
            if touched_entry:
                return EntryObservation(
                    status="FILLED",
                    reason="planned limit contained in completed M1 range",
                    effective_expires_at=deadline,
                    entry_time=pd.Timestamp(bar_open),
                    entry_price=candidate.planned_entry,
                    observed_at=bar_close,
                    touch_bar_open=pd.Timestamp(bar_open),
                    same_bar_tp1_ambiguous=touched_tp1,
                    fold_clipped=fold_clipped,
                )
            if touched_stop:
                return EntryObservation(
                    status="STOP_INVALIDATED",
                    reason="stop touched before any observed planned-limit touch",
                    effective_expires_at=deadline,
                    observed_at=bar_close,
                    touch_bar_open=pd.Timestamp(bar_open),
                    fold_clipped=fold_clipped,
                )
            if touched_tp1:
                return EntryObservation(
                    status="TP1_INVALIDATED",
                    reason="TP1 touched before any observed planned-limit touch",
                    effective_expires_at=deadline,
                    observed_at=bar_close,
                    touch_bar_open=pd.Timestamp(bar_open),
                    fold_clipped=fold_clipped,
                )
        expected_open += one_minute

    if policy == "planned_limit_touch" and deadline != deadline.floor("min"):
        return _censored_or_expired(
            candidate=candidate,
            fold=fold,
            deadline=deadline,
            complete_horizon=False,
            reason="entry TTL ends inside an unobservable M1 range",
        )

    # Latest minute open strictly before the exclusive deadline.  Subtracting
    # one nanosecond also handles a non-minute-aligned baseline TTL: a 00:13:30
    # cutoff still requires the 00:13 M1 open to be present.
    last_required_open = (deadline - pd.Timedelta(nanoseconds=1)).floor("min")
    complete_horizon = (
        last_required_open < _first_expected_open(candidate.decision_time)
        or expected_open > last_required_open
    )
    return _censored_or_expired(
        candidate=candidate,
        fold=fold,
        deadline=deadline,
        complete_horizon=complete_horizon,
        reason="M1 data ends before the complete entry TTL",
    )


def evaluate_four_arms(
    candidates: Sequence[DecisionCandidate],
    m1: pd.DataFrame,
    *,
    fold: FoldWindow,
) -> tuple[ArmDecision, ...]:
    """Evaluate the fixed four-arm design for one within-symbol decision.

    Selection is completed for both candidate policies before the M1 frame is
    normalized or inspected.  The chosen candidate is never replaced when it
    later fails to fill.  This function does not arbitrate against candidates
    from earlier or later decisions.
    """

    group = _validate_decision_group(candidates, fold)
    selected = {
        policy: select_candidate(group, policy=policy, fold=fold)
        for policy in CANDIDATE_POLICIES
    }
    ranks = _computed_net_ranks(group)
    decisions: list[ArmDecision] = []
    for spec in four_arm_specs():
        candidate = selected[spec.candidate_policy]
        observation = observe_entry(
            candidate,
            m1,
            policy=spec.entry_policy,
            fold=fold,
        )
        decisions.append(
            ArmDecision(
                spec=spec,
                candidate=candidate,
                observation=observation,
                computed_optimizer_rank=ranks.get(candidate.parent_opportunity_id),
            )
        )
    return tuple(decisions)


__all__ = [
    "ArmDecision",
    "ArmSpec",
    "CANDIDATE_POLICIES",
    "DecisionCandidate",
    "ENTRY_POLICIES",
    "EntryObservation",
    "FoldWindow",
    "PAIRED_ENTRY_ARMS_SCHEMA",
    "PairedEntryArmError",
    "evaluate_four_arms",
    "four_arm_specs",
    "observe_entry",
    "select_candidate",
]
