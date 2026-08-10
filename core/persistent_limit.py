"""Durable, broker-agnostic state for persistent LIMIT entry plans.

The live MT5 adapter is intentionally not imported here.  This module owns the
small piece of state that must survive the unsafe windows around ``order_send``:

1. persist a ``placing`` plan with its stable ``plan_id``;
2. submit broker order(s) carrying :attr:`PendingLimitPlan.broker_comment`;
3. attach returned tickets and transition to ``placed``;
4. reconcile fills/cancellation and only then retire the plan.

The model is deliberately strict.  Corrupt or newer state raises instead of
being interpreted as an empty order book, because that could lead the caller to
submit a duplicate order after a restart.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


PENDING_LIMIT_SCHEMA_VERSION = 1
BROKER_COMMENT_PREFIX = "FBPL:"


class PendingLimitValidationError(ValueError):
    """Persistent LIMIT state is malformed or violates a safety invariant."""


class PendingLimitState(str, Enum):
    PLACING = "placing"
    PLACED = "placed"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"


_TERMINAL_STATES = frozenset(
    {
        PendingLimitState.FILLED,
        PendingLimitState.CANCELLED,
        PendingLimitState.EXPIRED,
        PendingLimitState.FAILED,
    }
)

_ALLOWED_TRANSITIONS = {
    PendingLimitState.PLACING: frozenset(
        {
            PendingLimitState.PLACED,
            PendingLimitState.PARTIAL,
            PendingLimitState.FILLED,
            PendingLimitState.CANCELLING,
            PendingLimitState.CANCELLED,
            PendingLimitState.EXPIRED,
            PendingLimitState.FAILED,
        }
    ),
    PendingLimitState.PLACED: frozenset(
        {
            PendingLimitState.PARTIAL,
            PendingLimitState.FILLED,
            PendingLimitState.CANCELLING,
            PendingLimitState.CANCELLED,
            PendingLimitState.EXPIRED,
        }
    ),
    PendingLimitState.PARTIAL: frozenset(
        {
            PendingLimitState.FILLED,
            PendingLimitState.CANCELLING,
        }
    ),
    PendingLimitState.CANCELLING: frozenset(
        {
            PendingLimitState.PARTIAL,
            PendingLimitState.FILLED,
            PendingLimitState.CANCELLED,
            PendingLimitState.EXPIRED,
        }
    ),
    PendingLimitState.FILLED: frozenset(),
    PendingLimitState.CANCELLED: frozenset(),
    PendingLimitState.EXPIRED: frozenset(),
    PendingLimitState.FAILED: frozenset(),
}


def _finite_positive(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PendingLimitValidationError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise PendingLimitValidationError(f"{name} must be a finite positive number")
    return number


def _finite_nonnegative(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PendingLimitValidationError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(number) or number < 0.0:
        raise PendingLimitValidationError(f"{name} must be a finite nonnegative number")
    return number


def _positive_ticket(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PendingLimitValidationError(f"{name} must be a positive integer")
    ticket = int(value)
    if ticket <= 0:
        raise PendingLimitValidationError(f"{name} must be a positive integer")
    return ticket


def _json_safe(value: Any, *, path: str, seen: set[int]) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PendingLimitValidationError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            raise PendingLimitValidationError(f"{path} contains a reference cycle")
        seen.add(identity)
        try:
            result: Dict[str, Any] = {}
            for key, child in value.items():
                if not isinstance(key, str):
                    raise PendingLimitValidationError(
                        f"{path} contains a non-string mapping key"
                    )
                result[key] = _json_safe(
                    child,
                    path=f"{path}.{key}",
                    seen=seen,
                )
            return result
        finally:
            seen.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in seen:
            raise PendingLimitValidationError(f"{path} contains a reference cycle")
        seen.add(identity)
        try:
            return [
                _json_safe(child, path=f"{path}[{index}]", seen=seen)
                for index, child in enumerate(value)
            ]
        finally:
            seen.remove(identity)
    raise PendingLimitValidationError(
        f"{path} contains unsupported value type {type(value).__name__}"
    )


def make_json_safe_signal(signal: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a detached JSON-safe copy of a signal.

    Tuples are normalized to lists.  Unsupported objects, non-string mapping
    keys, cycles and NaN/Infinity fail closed instead of being stringified.
    """

    if not isinstance(signal, Mapping):
        raise PendingLimitValidationError("signal must be a mapping")
    payload = _json_safe(signal, path="signal", seen=set())
    if not isinstance(payload, dict):  # defensive; a mapping always becomes dict
        raise PendingLimitValidationError("signal must serialize to an object")
    # Keep json's own encoder as a final compatibility assertion.
    try:
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:  # pragma: no cover - guarded above
        raise PendingLimitValidationError("signal is not JSON-safe") from exc
    return payload


@dataclass(frozen=True)
class PendingLimitLeg:
    """One broker order/position leg belonging to a pending entry plan."""

    index: int
    target_price: float
    requested_volume: float
    submission_attempted: bool = False
    broker_order_ticket: Optional[int] = None
    broker_position_id: Optional[int] = None
    filled_volume: float = 0.0
    average_fill_price: Optional[float] = None
    deal_tickets: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or int(self.index) != self.index or self.index < 1:
            raise PendingLimitValidationError("leg index must be a positive integer")
        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(
            self,
            "target_price",
            _finite_positive(self.target_price, name="leg target_price"),
        )
        requested = _finite_positive(
            self.requested_volume,
            name="leg requested_volume",
        )
        filled = _finite_nonnegative(self.filled_volume, name="leg filled_volume")
        if filled > requested + 1e-9:
            raise PendingLimitValidationError(
                "leg filled_volume cannot exceed requested_volume"
            )
        object.__setattr__(self, "requested_volume", requested)
        object.__setattr__(self, "filled_volume", filled)

        if not isinstance(self.submission_attempted, bool):
            raise PendingLimitValidationError(
                "leg submission_attempted must be a boolean"
            )

        if self.broker_order_ticket is not None:
            if not self.submission_attempted:
                raise PendingLimitValidationError(
                    "broker_order_ticket requires submission_attempted"
                )
            object.__setattr__(
                self,
                "broker_order_ticket",
                _positive_ticket(
                    self.broker_order_ticket,
                    name="broker_order_ticket",
                ),
            )
        if self.broker_position_id is not None:
            object.__setattr__(
                self,
                "broker_position_id",
                _positive_ticket(
                    self.broker_position_id,
                    name="broker_position_id",
                ),
            )
            if filled <= 0.0:
                raise PendingLimitValidationError(
                    "broker_position_id requires a positive filled_volume"
                )

        if filled > 0.0:
            if not self.submission_attempted:
                raise PendingLimitValidationError(
                    "filled_volume requires submission_attempted"
                )
            average = _finite_positive(
                self.average_fill_price,
                name="average_fill_price",
            )
            object.__setattr__(self, "average_fill_price", average)
        elif self.average_fill_price is not None:
            raise PendingLimitValidationError(
                "average_fill_price requires a positive filled_volume"
            )

        tickets = tuple(
            _positive_ticket(ticket, name="deal_ticket")
            for ticket in self.deal_tickets
        )
        if len(tickets) != len(set(tickets)):
            raise PendingLimitValidationError("deal_tickets must be unique")
        if tickets and filled <= 0.0:
            raise PendingLimitValidationError(
                "deal_tickets require a positive filled_volume"
            )
        object.__setattr__(self, "deal_tickets", tickets)

    @property
    def remaining_volume(self) -> float:
        return max(0.0, self.requested_volume - self.filled_volume)

    def with_order(self, ticket: int) -> "PendingLimitLeg":
        """Attach the broker order ticket; repeated observation is idempotent."""

        normalized = _positive_ticket(ticket, name="broker_order_ticket")
        if self.broker_order_ticket is not None:
            if self.broker_order_ticket != normalized:
                raise PendingLimitValidationError(
                    "cannot replace a leg's broker_order_ticket"
                )
            return self
        return replace(
            self,
            submission_attempted=True,
            broker_order_ticket=normalized,
        )

    def with_submission_attempted(self) -> "PendingLimitLeg":
        """Durably mark the unsafe broker submission window as entered."""

        if self.submission_attempted:
            return self
        if self.broker_order_ticket is not None or self.filled_volume > 0.0:
            raise PendingLimitValidationError(
                "broker evidence cannot precede submission_attempted"
            )
        return replace(self, submission_attempted=True)

    def with_fill(
        self,
        volume: float,
        price: float,
        *,
        deal_ticket: Optional[int] = None,
        position_id: Optional[int] = None,
    ) -> "PendingLimitLeg":
        """Record one incremental broker fill and update its weighted price.

        A repeated ``deal_ticket`` is ignored, making deal-history reconciliation
        idempotent.  Callers should always provide deal tickets when available.
        """

        normalized_deal = (
            _positive_ticket(deal_ticket, name="deal_ticket")
            if deal_ticket is not None
            else None
        )
        if normalized_deal is not None and normalized_deal in self.deal_tickets:
            return self

        increment = _finite_positive(volume, name="fill volume")
        fill_price = _finite_positive(price, name="fill price")
        new_volume = self.filled_volume + increment
        if new_volume > self.requested_volume + 1e-9:
            raise PendingLimitValidationError(
                "incremental fill would exceed requested_volume"
            )
        old_notional = self.filled_volume * float(self.average_fill_price or 0.0)
        average = (old_notional + increment * fill_price) / new_volume

        new_position = self.broker_position_id
        if position_id is not None:
            normalized_position = _positive_ticket(
                position_id,
                name="broker_position_id",
            )
            if new_position is not None and new_position != normalized_position:
                raise PendingLimitValidationError(
                    "cannot replace a leg's broker_position_id"
                )
            new_position = normalized_position

        deals = self.deal_tickets
        if normalized_deal is not None:
            deals = deals + (normalized_deal,)
        return replace(
            self,
            submission_attempted=True,
            filled_volume=new_volume,
            average_fill_price=average,
            broker_position_id=new_position,
            deal_tickets=deals,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "target_price": self.target_price,
            "requested_volume": self.requested_volume,
            "submission_attempted": self.submission_attempted,
            "broker_order_ticket": self.broker_order_ticket,
            "broker_position_id": self.broker_position_id,
            "filled_volume": self.filled_volume,
            "average_fill_price": self.average_fill_price,
            "deal_tickets": list(self.deal_tickets),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PendingLimitLeg":
        if not isinstance(data, Mapping):
            raise PendingLimitValidationError("leg must be an object")
        expected = {
            "index",
            "target_price",
            "requested_volume",
            "submission_attempted",
            "broker_order_ticket",
            "broker_position_id",
            "filled_volume",
            "average_fill_price",
            "deal_tickets",
        }
        if set(data) != expected:
            raise PendingLimitValidationError(
                f"leg fields mismatch: expected {sorted(expected)}, got {sorted(data)}"
            )
        return cls(
            index=data["index"],
            target_price=data["target_price"],
            requested_volume=data["requested_volume"],
            submission_attempted=data["submission_attempted"],
            broker_order_ticket=data["broker_order_ticket"],
            broker_position_id=data["broker_position_id"],
            filled_volume=data["filled_volume"],
            average_fill_price=data["average_fill_price"],
            deal_tickets=tuple(data["deal_tickets"] or ()),
        )


@dataclass(frozen=True)
class PendingLimitPlan:
    """Versioned, durable state of one symbol's persistent LIMIT intent."""

    schema_version: int
    plan_id: str
    symbol: str
    side: str
    trigger_signature: str
    magic: int
    signal_payload: Dict[str, Any]
    limit_price: float
    entry_min: float
    entry_max: float
    stop_price: float
    tp_prices: Tuple[float, ...]
    legs: Tuple[PendingLimitLeg, ...]
    created_at: float
    expires_at: float
    updated_at: float
    entry_announced_at: Optional[float] = None
    state: PendingLimitState = PendingLimitState.PLACING
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != PENDING_LIMIT_SCHEMA_VERSION
        ):
            raise PendingLimitValidationError(
                f"unsupported plan schema_version {self.schema_version!r}"
            )
        object.__setattr__(self, "schema_version", PENDING_LIMIT_SCHEMA_VERSION)

        plan_id = str(self.plan_id or "").strip()
        if not (8 <= len(plan_id) <= 64) or any(
            not (char.isalnum() or char in "_-") for char in plan_id
        ):
            raise PendingLimitValidationError(
                "plan_id must contain 8-64 letters, digits, '_' or '-'"
            )
        object.__setattr__(self, "plan_id", plan_id)

        symbol = str(self.symbol or "").strip().upper()
        if not symbol:
            raise PendingLimitValidationError("symbol is required")
        object.__setattr__(self, "symbol", symbol)

        side = str(self.side or "").strip().upper()
        if side not in {"LONG", "SHORT"}:
            raise PendingLimitValidationError("side must be LONG or SHORT")
        object.__setattr__(self, "side", side)

        signature = str(self.trigger_signature or "").strip()
        if not signature:
            raise PendingLimitValidationError("trigger_signature is required")
        object.__setattr__(self, "trigger_signature", signature)

        if isinstance(self.magic, bool) or not isinstance(self.magic, int):
            raise PendingLimitValidationError("magic must be an integer")
        magic = int(self.magic)
        if magic < 0:
            raise PendingLimitValidationError("magic must be nonnegative")
        object.__setattr__(self, "magic", magic)

        limit_price = _finite_positive(self.limit_price, name="limit_price")
        entry_min = _finite_positive(self.entry_min, name="entry_min")
        entry_max = _finite_positive(self.entry_max, name="entry_max")
        stop_price = _finite_positive(self.stop_price, name="stop_price")
        if entry_min > entry_max:
            raise PendingLimitValidationError("entry_min cannot exceed entry_max")
        if limit_price < entry_min - 1e-12 or limit_price > entry_max + 1e-12:
            raise PendingLimitValidationError(
                "limit_price must be inside [entry_min, entry_max]"
            )
        object.__setattr__(self, "limit_price", limit_price)
        object.__setattr__(self, "entry_min", entry_min)
        object.__setattr__(self, "entry_max", entry_max)
        object.__setattr__(self, "stop_price", stop_price)

        targets = tuple(
            _finite_positive(value, name="tp_price") for value in self.tp_prices
        )
        if not targets:
            raise PendingLimitValidationError("at least one tp_price is required")
        if side == "LONG":
            if stop_price >= limit_price or any(target <= limit_price for target in targets):
                raise PendingLimitValidationError(
                    "LONG geometry requires stop < limit < every target"
                )
            if list(targets) != sorted(targets):
                raise PendingLimitValidationError("LONG tp_prices must be ascending")
        else:
            if stop_price <= limit_price or any(target >= limit_price for target in targets):
                raise PendingLimitValidationError(
                    "SHORT geometry requires stop > limit > every target"
                )
            if list(targets) != sorted(targets, reverse=True):
                raise PendingLimitValidationError("SHORT tp_prices must be descending")
        object.__setattr__(self, "tp_prices", targets)

        legs = tuple(self.legs)
        if not legs or not all(isinstance(leg, PendingLimitLeg) for leg in legs):
            raise PendingLimitValidationError(
                "legs must contain at least one PendingLimitLeg"
            )
        if len({leg.index for leg in legs}) != len(legs):
            raise PendingLimitValidationError("leg indexes must be unique")
        if tuple(sorted(legs, key=lambda leg: leg.index)) != legs:
            raise PendingLimitValidationError("legs must be ordered by index")
        for leg in legs:
            if leg.index > len(targets):
                raise PendingLimitValidationError(
                    f"leg index {leg.index} has no matching tp_price"
                )
            expected_target = targets[leg.index - 1]
            if not math.isclose(
                leg.target_price,
                expected_target,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise PendingLimitValidationError(
                    f"leg {leg.index} target does not match tp_prices"
                )
        object.__setattr__(self, "legs", legs)

        created = _finite_positive(self.created_at, name="created_at")
        expires = _finite_positive(self.expires_at, name="expires_at")
        updated = _finite_positive(self.updated_at, name="updated_at")
        if expires <= created:
            raise PendingLimitValidationError("expires_at must be after created_at")
        if updated < created:
            raise PendingLimitValidationError("updated_at cannot precede created_at")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(self, "updated_at", updated)
        announced = self.entry_announced_at
        if announced is not None:
            announced = _finite_positive(
                announced,
                name="entry_announced_at",
            )
            if announced < created or announced > updated + 1e-9:
                raise PendingLimitValidationError(
                    "entry_announced_at must be between created_at and updated_at"
                )
        object.__setattr__(self, "entry_announced_at", announced)

        try:
            state = (
                self.state
                if isinstance(self.state, PendingLimitState)
                else PendingLimitState(str(self.state))
            )
        except ValueError as exc:
            raise PendingLimitValidationError(f"unknown pending state {self.state!r}") from exc
        object.__setattr__(self, "state", state)

        payload = make_json_safe_signal(self.signal_payload)
        payload_side = str(payload.get("side") or "").upper()
        if payload_side != side:
            raise PendingLimitValidationError("signal side does not match plan side")
        for key, expected in (
            ("entry_price", limit_price),
            ("stop_price", stop_price),
        ):
            if key not in payload or not math.isclose(
                float(payload[key]), expected, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise PendingLimitValidationError(
                    f"signal {key} does not match plan geometry"
                )
        raw_entry_min = payload.get("entry_min")
        raw_entry_max = payload.get("entry_max")
        if raw_entry_min is not None and not math.isclose(
            float(raw_entry_min), entry_min, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise PendingLimitValidationError("signal entry_min does not match plan")
        if raw_entry_max is not None and not math.isclose(
            float(raw_entry_max), entry_max, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise PendingLimitValidationError("signal entry_max does not match plan")
        payload_targets = tuple(float(value) for value in (payload.get("tp_prices") or ()))
        if payload_targets != targets:
            raise PendingLimitValidationError("signal tp_prices do not match plan")
        if str(payload.get("setup_id") or "") != plan_id:
            raise PendingLimitValidationError("signal setup_id must equal plan_id")
        object.__setattr__(self, "signal_payload", payload)

        reason = None if self.reason is None else str(self.reason).strip() or None
        object.__setattr__(self, "reason", reason)
        self._validate_state_consistency()

    def _validate_state_consistency(self) -> None:
        total_requested = sum(leg.requested_volume for leg in self.legs)
        total_filled = sum(leg.filled_volume for leg in self.legs)
        all_tickets = all(leg.broker_order_ticket is not None for leg in self.legs)
        any_ticket = any(leg.broker_order_ticket is not None for leg in self.legs)
        any_attempted = any(leg.submission_attempted for leg in self.legs)

        if self.state == PendingLimitState.PLACING and total_filled > 0.0:
            raise PendingLimitValidationError("placing plan cannot already contain fills")
        if self.state == PendingLimitState.PLACED:
            if not all_tickets or total_filled > 0.0:
                raise PendingLimitValidationError(
                    "placed plan requires every order ticket and zero fills"
                )
        if self.state == PendingLimitState.PARTIAL:
            if total_filled <= 0.0 or total_filled >= total_requested - 1e-9:
                raise PendingLimitValidationError(
                    "partial plan requires a nonzero incomplete fill"
                )
        if self.state == PendingLimitState.FILLED and total_filled <= 0.0:
            raise PendingLimitValidationError("filled plan requires a positive fill")
        if self.state == PendingLimitState.CANCELLING and not (
            any_attempted or any_ticket or total_filled > 0.0
        ):
            raise PendingLimitValidationError(
                "cancelling plan requires a submission attempt, order ticket or fill"
            )
        if self.state in {
            PendingLimitState.CANCELLED,
            PendingLimitState.EXPIRED,
        } and total_filled > 0.0:
            raise PendingLimitValidationError(
                f"{self.state.value} plan cannot hide a broker fill"
            )
        if self.state == PendingLimitState.FAILED and (
            any_ticket or total_filled > 0.0
        ):
            raise PendingLimitValidationError(
                "failed plan cannot own an order ticket or broker fill"
            )
        if self.entry_announced_at is not None and total_filled <= 0.0:
            raise PendingLimitValidationError(
                "entry_announced_at requires a broker fill"
            )

    @classmethod
    def create(
        cls,
        *,
        symbol: str,
        signal: Mapping[str, Any],
        trigger_signature: str,
        expires_at: float,
        magic: int,
        legs: Sequence[PendingLimitLeg],
        now: Optional[float] = None,
        plan_id: Optional[str] = None,
    ) -> "PendingLimitPlan":
        payload = make_json_safe_signal(signal)
        created = time.time() if now is None else float(now)
        stable_id = str(plan_id or payload.get("setup_id") or uuid.uuid4().hex)
        payload["setup_id"] = stable_id

        try:
            limit_price = float(payload["entry_price"])
            stop_price = float(payload["stop_price"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PendingLimitValidationError(
                "signal requires numeric entry_price and stop_price"
            ) from exc
        raw_targets = payload.get("tp_prices") or ()
        targets = tuple(float(value) for value in raw_targets)
        raw_min = payload.get("entry_min")
        raw_max = payload.get("entry_max")
        entry_min = limit_price if raw_min is None else float(raw_min)
        entry_max = limit_price if raw_max is None else float(raw_max)

        return cls(
            schema_version=PENDING_LIMIT_SCHEMA_VERSION,
            plan_id=stable_id,
            symbol=symbol,
            side=str(payload.get("side") or ""),
            trigger_signature=trigger_signature,
            magic=magic,
            signal_payload=payload,
            limit_price=limit_price,
            entry_min=entry_min,
            entry_max=entry_max,
            stop_price=stop_price,
            tp_prices=targets,
            legs=tuple(legs),
            created_at=created,
            expires_at=float(expires_at),
            updated_at=created,
            state=PendingLimitState.PLACING,
        )

    @property
    def broker_comment(self) -> str:
        """Stable collision-resistant ownership prefix for every plan leg."""

        token = hashlib.sha256(self.plan_id.encode("utf-8")).hexdigest()[:20]
        return f"{BROKER_COMMENT_PREFIX}{token}"

    def broker_comment_for_leg(self, index: int) -> str:
        """Stable per-leg ownership comment with a shared plan prefix."""

        if isinstance(index, bool) or not isinstance(index, int) or index < 1:
            raise PendingLimitValidationError("leg index must be a positive integer")
        if not any(leg.index == index for leg in self.legs):
            raise PendingLimitValidationError(f"unknown leg index {index}")
        comment = f"{self.broker_comment}:T{index}"
        if len(comment) > 31:
            raise PendingLimitValidationError("broker comment exceeds 31 characters")
        return comment

    @property
    def is_terminal(self) -> bool:
        return self.state in _TERMINAL_STATES

    @property
    def has_fills(self) -> bool:
        return any(leg.filled_volume > 0.0 for leg in self.legs)

    @property
    def entry_announcement_pending(self) -> bool:
        """Whether a broker fill still needs its durable ENTER acknowledgement."""

        return self.has_fills and self.entry_announced_at is None

    @property
    def working_order_tickets(self) -> Tuple[int, ...]:
        if self.is_terminal:
            return ()
        return tuple(
            int(leg.broker_order_ticket)
            for leg in self.legs
            if leg.broker_order_ticket is not None
            and leg.remaining_volume > 1e-9
        )

    def is_due(self, now: Optional[float] = None) -> bool:
        moment = time.time() if now is None else float(now)
        if not math.isfinite(moment):
            raise PendingLimitValidationError("now must be finite")
        return moment >= self.expires_at

    def replace_leg(
        self,
        leg: PendingLimitLeg,
        *,
        now: Optional[float] = None,
    ) -> "PendingLimitPlan":
        if self.is_terminal:
            raise PendingLimitValidationError("cannot modify a terminal plan")
        if not isinstance(leg, PendingLimitLeg):
            raise PendingLimitValidationError("replacement must be PendingLimitLeg")
        old = next((item for item in self.legs if item.index == leg.index), None)
        if old is None:
            raise PendingLimitValidationError(f"unknown leg index {leg.index}")
        if not math.isclose(
            old.target_price, leg.target_price, rel_tol=1e-12, abs_tol=1e-12
        ) or not math.isclose(
            old.requested_volume,
            leg.requested_volume,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise PendingLimitValidationError(
                "leg target and requested volume are immutable"
            )
        moment = self._validated_transition_time(now)
        legs = tuple(leg if item.index == leg.index else item for item in self.legs)
        return replace(self, legs=legs, updated_at=moment)

    def attach_order(
        self,
        index: int,
        ticket: int,
        *,
        now: Optional[float] = None,
    ) -> "PendingLimitPlan":
        """Atomically attach a ticket and become placed after the last leg."""

        old = next((leg for leg in self.legs if leg.index == index), None)
        if old is None:
            raise PendingLimitValidationError(f"unknown leg index {index}")
        normalized_ticket = _positive_ticket(ticket, name="broker_order_ticket")
        if self.state == PendingLimitState.PLACED:
            if old.broker_order_ticket == normalized_ticket:
                return self
            raise PendingLimitValidationError("cannot replace a placed order ticket")
        if self.state != PendingLimitState.PLACING:
            raise PendingLimitValidationError(
                "broker order tickets can only be attached while placing"
            )
        moment = self._validated_transition_time(now)
        updated_leg = old.with_order(normalized_ticket)
        legs = tuple(
            updated_leg if leg.index == updated_leg.index else leg for leg in self.legs
        )
        state = (
            PendingLimitState.PLACED
            if all(leg.broker_order_ticket is not None for leg in legs)
            else PendingLimitState.PLACING
        )
        return replace(self, legs=legs, state=state, updated_at=moment)

    def mark_submission_attempted(
        self,
        index: int,
        *,
        now: Optional[float] = None,
    ) -> "PendingLimitPlan":
        """Persist the unsafe intent immediately before one broker order_send.

        Re-observing the marker is idempotent. Callers must never invoke
        order_send for a leg whose marker was already true on entry; broker
        reconciliation, not resubmission, owns that ambiguous state.
        """

        old = next((leg for leg in self.legs if leg.index == index), None)
        if old is None:
            raise PendingLimitValidationError(f"unknown leg index {index}")
        if old.submission_attempted:
            return self
        if self.state != PendingLimitState.PLACING:
            raise PendingLimitValidationError(
                "submission can only begin while a plan is placing"
            )
        moment = self._validated_transition_time(now)
        updated_leg = old.with_submission_attempted()
        legs = tuple(
            updated_leg if leg.index == updated_leg.index else leg
            for leg in self.legs
        )
        return replace(self, legs=legs, updated_at=moment)

    def record_fill(
        self,
        index: int,
        volume: float,
        price: float,
        *,
        deal_ticket: Optional[int] = None,
        position_id: Optional[int] = None,
        now: Optional[float] = None,
    ) -> "PendingLimitPlan":
        """Atomically record a fill and choose a state that can contain it.

        A fill racing an outstanding cancellation keeps the plan cancelling
        until the broker confirms the remainder gone. A complete basket is
        promoted directly to terminal filled.
        """

        if self.is_terminal:
            raise PendingLimitValidationError("cannot fill a terminal plan")
        old = next((leg for leg in self.legs if leg.index == index), None)
        if old is None:
            raise PendingLimitValidationError(f"unknown leg index {index}")
        moment = self._validated_transition_time(now)
        updated_leg = old.with_fill(
            volume,
            price,
            deal_ticket=deal_ticket,
            position_id=position_id,
        )
        if updated_leg is old:
            return self
        legs = tuple(
            updated_leg if leg.index == updated_leg.index else leg for leg in self.legs
        )
        requested = sum(leg.requested_volume for leg in legs)
        filled = sum(leg.filled_volume for leg in legs)
        if filled >= requested - 1e-9:
            state = PendingLimitState.FILLED
        elif self.state == PendingLimitState.CANCELLING:
            state = PendingLimitState.CANCELLING
        else:
            state = PendingLimitState.PARTIAL
        return replace(self, legs=legs, state=state, updated_at=moment)

    def mark_entry_announced(
        self,
        *,
        now: Optional[float] = None,
    ) -> "PendingLimitPlan":
        """Acknowledge idempotent delivery of the durable ENTER outbox item."""

        if self.entry_announced_at is not None:
            return self
        if not self.has_fills:
            raise PendingLimitValidationError(
                "cannot announce an entry before a broker fill"
            )
        moment = self._validated_transition_time(now)
        return replace(
            self,
            entry_announced_at=moment,
            updated_at=moment,
        )

    def _validated_transition_time(self, now: Optional[float]) -> float:
        moment = time.time() if now is None else float(now)
        if not math.isfinite(moment) or moment < self.updated_at:
            raise PendingLimitValidationError(
                "transition time must be finite and monotonic"
            )
        return moment

    def transition(
        self,
        new_state: PendingLimitState | str,
        *,
        now: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> "PendingLimitPlan":
        try:
            target = (
                new_state
                if isinstance(new_state, PendingLimitState)
                else PendingLimitState(str(new_state))
            )
        except ValueError as exc:
            raise PendingLimitValidationError(f"unknown pending state {new_state!r}") from exc
        if target == self.state:
            return self
        if target not in _ALLOWED_TRANSITIONS[self.state]:
            raise PendingLimitValidationError(
                f"invalid transition {self.state.value} -> {target.value}"
            )
        moment = self._validated_transition_time(now)
        if target == PendingLimitState.EXPIRED and moment < self.expires_at:
            raise PendingLimitValidationError(
                "plan cannot expire before its immutable expires_at"
            )
        return replace(
            self,
            state=target,
            updated_at=moment,
            reason=None if reason is None else str(reason),
        )

    def expire_if_due(self, now: Optional[float] = None) -> "PendingLimitPlan":
        """Start safe TTL retirement without claiming broker cancellation.

        An unsubmitted plan can become ``expired`` immediately.  Once any
        broker ticket or fill exists, elapsed TTL means ``cancelling``; only a
        later broker reconciliation may mark it ``expired``/``cancelled`` or
        promote a racing fill to ``filled``.
        """

        if self.is_terminal:
            return self
        moment = time.time() if now is None else float(now)
        if not self.is_due(moment):
            return self
        if (
            any(leg.submission_attempted for leg in self.legs)
            or any(leg.broker_order_ticket is not None for leg in self.legs)
            or self.has_fills
        ):
            if self.state == PendingLimitState.CANCELLING:
                return self
            return self.transition(
                PendingLimitState.CANCELLING,
                now=moment,
                reason="entry TTL elapsed; broker cancellation required",
            )
        return self.transition(
            PendingLimitState.EXPIRED,
            now=moment,
            reason="entry TTL elapsed before broker placement",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "symbol": self.symbol,
            "side": self.side,
            "trigger_signature": self.trigger_signature,
            "magic": self.magic,
            "signal_payload": make_json_safe_signal(self.signal_payload),
            "limit_price": self.limit_price,
            "entry_min": self.entry_min,
            "entry_max": self.entry_max,
            "stop_price": self.stop_price,
            "tp_prices": list(self.tp_prices),
            "legs": [leg.to_dict() for leg in self.legs],
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "updated_at": self.updated_at,
            "entry_announced_at": self.entry_announced_at,
            "state": self.state.value,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PendingLimitPlan":
        if not isinstance(data, Mapping):
            raise PendingLimitValidationError("plan must be an object")
        expected = {
            "schema_version",
            "plan_id",
            "symbol",
            "side",
            "trigger_signature",
            "magic",
            "signal_payload",
            "limit_price",
            "entry_min",
            "entry_max",
            "stop_price",
            "tp_prices",
            "legs",
            "created_at",
            "expires_at",
            "updated_at",
            "entry_announced_at",
            "state",
            "reason",
        }
        if set(data) != expected:
            raise PendingLimitValidationError(
                f"plan fields mismatch: expected {sorted(expected)}, got {sorted(data)}"
            )
        raw_legs = data["legs"]
        if not isinstance(raw_legs, list):
            raise PendingLimitValidationError("plan legs must be a list")
        raw_targets = data["tp_prices"]
        if not isinstance(raw_targets, list):
            raise PendingLimitValidationError("plan tp_prices must be a list")
        return cls(
            schema_version=data["schema_version"],
            plan_id=data["plan_id"],
            symbol=data["symbol"],
            side=data["side"],
            trigger_signature=data["trigger_signature"],
            magic=data["magic"],
            signal_payload=data["signal_payload"],
            limit_price=data["limit_price"],
            entry_min=data["entry_min"],
            entry_max=data["entry_max"],
            stop_price=data["stop_price"],
            tp_prices=tuple(raw_targets),
            legs=tuple(PendingLimitLeg.from_dict(item) for item in raw_legs),
            created_at=data["created_at"],
            expires_at=data["expires_at"],
            updated_at=data["updated_at"],
            entry_announced_at=data["entry_announced_at"],
            state=data["state"],
            reason=data["reason"],
        )


def save_pending_limits(
    plans: Mapping[str, PendingLimitPlan],
    path: Path,
) -> None:
    """Atomically persist a symbol-keyed pending LIMIT order book."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized: Dict[str, Dict[str, Any]] = {}
    plan_ids: set[str] = set()
    normalized_symbols: set[str] = set()
    for key, plan in plans.items():
        if not isinstance(plan, PendingLimitPlan):
            raise PendingLimitValidationError(
                f"pending plan for {key!r} is not PendingLimitPlan"
            )
        symbol = str(key).upper()
        if symbol != plan.symbol:
            raise PendingLimitValidationError(
                f"pending map key {key!r} does not match plan symbol {plan.symbol!r}"
            )
        if symbol in normalized_symbols:
            raise PendingLimitValidationError(
                f"duplicate normalized pending symbol {symbol!r}"
            )
        normalized_symbols.add(symbol)
        if plan.plan_id in plan_ids:
            raise PendingLimitValidationError(
                f"duplicate pending plan_id {plan.plan_id!r}"
            )
        plan_ids.add(plan.plan_id)
        serialized[symbol] = plan.to_dict()

    envelope = {
        "schema_version": PENDING_LIMIT_SCHEMA_VERSION,
        "plans": serialized,
    }
    encoded = json.dumps(
        envelope,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_pending_limits(path: Path) -> Dict[str, PendingLimitPlan]:
    """Load and validate a pending LIMIT order book.

    Missing state is an empty book.  Existing but malformed state raises; it is
    never silently downgraded to empty because broker orders may still exist.
    """

    source = Path(path)
    if not source.exists():
        return {}
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PendingLimitValidationError(
            f"cannot read pending LIMIT state {source}: {exc}"
        ) from exc
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "plans"}:
        raise PendingLimitValidationError("pending LIMIT store envelope is malformed")
    if raw["schema_version"] != PENDING_LIMIT_SCHEMA_VERSION:
        raise PendingLimitValidationError(
            f"unsupported pending LIMIT store schema_version {raw['schema_version']!r}"
        )
    raw_plans = raw["plans"]
    if not isinstance(raw_plans, dict):
        raise PendingLimitValidationError("pending LIMIT plans must be an object")

    restored: Dict[str, PendingLimitPlan] = {}
    plan_ids: set[str] = set()
    normalized_symbols: set[str] = set()
    for key, value in raw_plans.items():
        if not isinstance(key, str):  # JSON objects already guarantee this
            raise PendingLimitValidationError("pending LIMIT symbol key must be a string")
        plan = PendingLimitPlan.from_dict(value)
        symbol = key.upper()
        if symbol != plan.symbol:
            raise PendingLimitValidationError(
                f"pending map key {key!r} does not match plan symbol {plan.symbol!r}"
            )
        if symbol in normalized_symbols:
            raise PendingLimitValidationError(
                f"duplicate normalized pending symbol {symbol!r}"
            )
        normalized_symbols.add(symbol)
        if plan.plan_id in plan_ids:
            raise PendingLimitValidationError(
                f"duplicate pending plan_id {plan.plan_id!r}"
            )
        plan_ids.add(plan.plan_id)
        restored[symbol] = plan
    return restored


__all__ = [
    "BROKER_COMMENT_PREFIX",
    "PENDING_LIMIT_SCHEMA_VERSION",
    "PendingLimitLeg",
    "PendingLimitPlan",
    "PendingLimitState",
    "PendingLimitValidationError",
    "load_pending_limits",
    "make_json_safe_signal",
    "save_pending_limits",
]
