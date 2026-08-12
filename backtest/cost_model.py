"""Versioned, explicit transaction-cost model for setup-level R metrics.

Historical LSE OHLCV has no broker bid/ask, commissions, swaps or slippage.
This module therefore has no silent defaults: callers must supply a complete
profile with provenance. Missing geometry or conversion data produces an
unknown estimate rather than a fabricated zero-cost trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo


COST_PROFILE_SCHEMA = "fx-cost-profile-v1"


class CostModelError(ValueError):
    pass


def _finite(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CostModelError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise CostModelError(f"{name} must be finite")
    return result


def _utc(value: Any, *, name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(
                str(value).strip().replace("Z", "+00:00")
            )
        except (TypeError, ValueError) as exc:
            raise CostModelError(f"{name} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CostModelError(f"{name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SymbolCostSpec:
    symbol: str
    base_currency: str
    quote_currency: str
    contract_size: float
    spread_price: float
    slippage_price_per_side: float
    commission_round_turn_per_lot: float
    swap_long_per_lot_rollover: float
    swap_short_per_lot_rollover: float
    quote_to_account_rate: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "base_currency": self.base_currency,
            "quote_currency": self.quote_currency,
            "contract_size": self.contract_size,
            "spread_price": self.spread_price,
            "slippage_price_per_side": self.slippage_price_per_side,
            "commission_round_turn_per_lot": (
                self.commission_round_turn_per_lot
            ),
            "swap_long_per_lot_rollover": (
                self.swap_long_per_lot_rollover
            ),
            "swap_short_per_lot_rollover": (
                self.swap_short_per_lot_rollover
            ),
            "quote_to_account_rate": self.quote_to_account_rate,
        }

    @classmethod
    def from_mapping(
        cls,
        symbol: str,
        payload: Mapping[str, Any],
    ) -> "SymbolCostSpec":
        spec = cls(
            symbol=str(symbol).strip().upper(),
            base_currency=str(
                payload.get("base_currency") or ""
            ).strip().upper(),
            quote_currency=str(
                payload.get("quote_currency") or ""
            ).strip().upper(),
            contract_size=_finite(
                payload.get("contract_size"),
                name=f"{symbol}.contract_size",
            ),
            spread_price=_finite(
                payload.get("spread_price"),
                name=f"{symbol}.spread_price",
            ),
            slippage_price_per_side=_finite(
                payload.get("slippage_price_per_side"),
                name=f"{symbol}.slippage_price_per_side",
            ),
            commission_round_turn_per_lot=_finite(
                payload.get("commission_round_turn_per_lot"),
                name=f"{symbol}.commission_round_turn_per_lot",
            ),
            swap_long_per_lot_rollover=_finite(
                payload.get("swap_long_per_lot_rollover"),
                name=f"{symbol}.swap_long_per_lot_rollover",
            ),
            swap_short_per_lot_rollover=_finite(
                payload.get("swap_short_per_lot_rollover"),
                name=f"{symbol}.swap_short_per_lot_rollover",
            ),
            quote_to_account_rate=(
                _finite(
                    payload.get("quote_to_account_rate"),
                    name=f"{symbol}.quote_to_account_rate",
                )
                if payload.get("quote_to_account_rate") is not None
                else None
            ),
        )
        if not spec.symbol or not spec.base_currency or not spec.quote_currency:
            raise CostModelError(f"{symbol}: currencies are required")
        for name in (
            "contract_size",
            "spread_price",
            "slippage_price_per_side",
            "commission_round_turn_per_lot",
        ):
            value = float(getattr(spec, name))
            if value <= 0 and name == "contract_size":
                raise CostModelError(f"{symbol}.{name} must be positive")
            if value < 0:
                raise CostModelError(f"{symbol}.{name} cannot be negative")
        if (
            spec.quote_to_account_rate is not None
            and spec.quote_to_account_rate <= 0
        ):
            raise CostModelError(
                f"{symbol}.quote_to_account_rate must be positive"
            )
        return spec


@dataclass(frozen=True)
class CostProfile:
    profile_id: str
    profile_sha256: str
    account_currency: str
    measured_from: str
    created_at_utc: datetime
    rollover_timezone: str
    rollover_hour: int
    triple_swap_weekday: int
    symbols: Mapping[str, SymbolCostSpec]
    source_payload: Mapping[str, Any]
    measured_through_utc: Optional[datetime] = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CostProfile":
        if str(payload.get("schema") or "") != COST_PROFILE_SCHEMA:
            raise CostModelError(
                f"cost profile schema must be {COST_PROFILE_SCHEMA}"
            )
        profile_id = str(payload.get("profile_id") or "").strip()
        measured_from = str(payload.get("measured_from") or "").strip()
        account_currency = str(
            payload.get("account_currency") or ""
        ).strip().upper()
        if not profile_id or not measured_from or not account_currency:
            raise CostModelError(
                "profile_id, measured_from and account_currency are required"
            )
        timezone_name = str(
            payload.get("rollover_timezone") or ""
        ).strip()
        try:
            ZoneInfo(timezone_name)
        except Exception as exc:
            raise CostModelError(
                f"unknown rollover_timezone: {timezone_name}"
            ) from exc
        rollover_hour = int(payload.get("rollover_hour", 0))
        triple_weekday = int(payload.get("triple_swap_weekday", 2))
        if not 0 <= rollover_hour <= 23:
            raise CostModelError("rollover_hour must be 0..23")
        if not 0 <= triple_weekday <= 4:
            raise CostModelError("triple_swap_weekday must be 0..4")
        raw_symbols = payload.get("symbols")
        if not isinstance(raw_symbols, Mapping) or not raw_symbols:
            raise CostModelError("symbols must be a non-empty mapping")
        symbols = {
            str(symbol).strip().upper(): SymbolCostSpec.from_mapping(
                str(symbol),
                spec,
            )
            for symbol, spec in raw_symbols.items()
            if isinstance(spec, Mapping)
        }
        if len(symbols) != len(raw_symbols):
            raise CostModelError("every symbol spec must be a mapping")
        source_payload = dict(payload)
        source_payload.pop("profile_sha256", None)
        canonical_source = json.loads(json.dumps(
            source_payload,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ))
        return cls(
            profile_id=profile_id,
            profile_sha256=_canonical_hash(canonical_source),
            account_currency=account_currency,
            measured_from=measured_from,
            created_at_utc=_utc(
                payload.get("created_at_utc"),
                name="created_at_utc",
            ),
            rollover_timezone=timezone_name,
            rollover_hour=rollover_hour,
            triple_swap_weekday=triple_weekday,
            symbols=symbols,
            source_payload=canonical_source,
            measured_through_utc=(
                _utc(
                    payload.get("measured_through_utc"),
                    name="measured_through_utc",
                )
                if payload.get("measured_through_utc") is not None
                else None
            ),
        )

    @classmethod
    def load(cls, path: Path | str) -> "CostProfile":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CostModelError(
                f"cannot read cost profile {source}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise CostModelError("cost profile root must be an object")
        return cls.from_mapping(payload)

    def symbol(self, symbol: str) -> SymbolCostSpec:
        key = str(symbol).strip().upper()
        try:
            return self.symbols[key]
        except KeyError as exc:
            raise CostModelError(
                f"cost profile has no symbol {key}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        payload = json.loads(json.dumps(
            self.source_payload,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ))
        payload["profile_sha256"] = self.profile_sha256
        return payload

    def point_in_time_audit(
        self,
        fold_test_starts: Sequence[Any],
    ) -> dict[str, Any]:
        """Audit whether profile provenance precedes every OOS fold.

        ``measured_from`` remains human-readable provenance for legacy
        profiles. Only the structured ``measured_through_utc`` timestamp can
        establish that the measured inputs were available at a fold boundary.
        """

        starts = tuple(
            sorted(
                _utc(value, name="fold_test_start")
                for value in fold_test_starts
            )
        )
        if not starts:
            raise CostModelError(
                "point-in-time audit requires at least one fold test_start"
            )
        created_before_all = all(
            self.created_at_utc <= test_start
            for test_start in starts
        )
        measured_before_all = (
            all(
                self.measured_through_utc <= test_start
                for test_start in starts
            )
            if self.measured_through_utc is not None
            else False
        )
        issues: list[str] = []
        if self.measured_through_utc is None:
            issues.append("MISSING_MEASURED_THROUGH_UTC")
        if not created_before_all:
            issues.append("CREATED_AT_AFTER_FOLD_TEST_START")
        if (
            self.measured_through_utc is not None
            and not measured_before_all
        ):
            issues.append("MEASURED_THROUGH_AFTER_FOLD_TEST_START")
        causal = not issues
        return {
            "status": (
                "CAUSAL_FOR_ALL_FOLDS"
                if causal
                else "NOT_POINT_IN_TIME"
            ),
            "causal_for_all_fold_test_starts": causal,
            "issues": issues,
            "created_at_utc": self.created_at_utc.isoformat(),
            "measured_through_utc": (
                self.measured_through_utc.isoformat()
                if self.measured_through_utc is not None
                else None
            ),
            "created_at_before_all_fold_test_starts": created_before_all,
            "measured_through_before_all_fold_test_starts": (
                measured_before_all
                if self.measured_through_utc is not None
                else None
            ),
            "earliest_fold_test_start": starts[0].isoformat(),
            "latest_fold_test_start": starts[-1].isoformat(),
            "fold_test_start_count": len(starts),
        }


@dataclass(frozen=True)
class CostEstimate:
    profile_id: str
    profile_sha256: str
    symbol: str
    spread_r: float
    slippage_r: float
    commission_r: float
    swap_r: float
    rollover_units: float
    total_cost_r: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost_profile_id": self.profile_id,
            "cost_profile_sha256": self.profile_sha256,
            "cost_symbol": self.symbol,
            "spread_cost_r": self.spread_r,
            "slippage_cost_r": self.slippage_r,
            "commission_cost_r": self.commission_r,
            "swap_cost_r": self.swap_r,
            "rollover_units": self.rollover_units,
            "transaction_cost_r": self.total_cost_r,
        }


def rollover_units(
    entry_time: Any,
    exit_time: Any,
    *,
    timezone_name: str,
    rollover_hour: int,
    triple_swap_weekday: int,
) -> int:
    entry = _utc(entry_time, name="entry_time")
    exit_ = _utc(exit_time, name="exit_time")
    if exit_ < entry:
        raise CostModelError("exit_time cannot precede entry_time")
    zone = ZoneInfo(timezone_name)
    local_entry = entry.astimezone(zone)
    local_exit = exit_.astimezone(zone)
    day = local_entry.date()
    end_day = local_exit.date()
    units = 0
    while day <= end_day:
        cutoff = datetime.combine(
            day,
            dt_time(hour=int(rollover_hour)),
            tzinfo=zone,
        )
        if (
            local_entry < cutoff <= local_exit
            and cutoff.weekday() < 5
        ):
            units += 3 if cutoff.weekday() == triple_swap_weekday else 1
        day += timedelta(days=1)
    return units


def _quote_to_account(
    profile: CostProfile,
    spec: SymbolCostSpec,
    *,
    entry_price: float,
) -> float:
    if spec.quote_currency == profile.account_currency:
        return 1.0
    if spec.base_currency == profile.account_currency:
        return 1.0 / entry_price
    if spec.quote_to_account_rate is not None:
        return float(spec.quote_to_account_rate)
    raise CostModelError(
        f"{spec.symbol}: quote-to-{profile.account_currency} conversion "
        "is not derivable and quote_to_account_rate is absent"
    )


def estimate_cost_r(
    profile: CostProfile,
    *,
    symbol: str,
    side: str,
    entry_price: Any,
    stop_price: Any,
    rollover_unit_count: float = 0.0,
) -> CostEstimate:
    spec = profile.symbol(symbol)
    entry = _finite(entry_price, name="entry_price")
    stop = _finite(stop_price, name="stop_price")
    risk_distance = abs(entry - stop)
    if entry <= 0 or risk_distance <= 0:
        raise CostModelError(
            "entry_price must be positive and stop distance non-zero"
        )
    units = _finite(
        rollover_unit_count,
        name="rollover_unit_count",
    )
    if units < 0:
        raise CostModelError("rollover_unit_count cannot be negative")
    conversion = _quote_to_account(
        profile,
        spec,
        entry_price=entry,
    )
    cash_per_price_unit = spec.contract_size * conversion
    commission_price = (
        spec.commission_round_turn_per_lot / cash_per_price_unit
    )
    side_key = str(side).strip().upper()
    if side_key not in {"LONG", "SHORT"}:
        raise CostModelError("side must be LONG or SHORT")
    swap_cash = (
        spec.swap_long_per_lot_rollover
        if side_key == "LONG"
        else spec.swap_short_per_lot_rollover
    ) * units
    # Broker swap is signed P&L: a negative debit is a positive cost.
    swap_price_cost = -swap_cash / cash_per_price_unit
    spread_r = spec.spread_price / risk_distance
    slippage_r = (
        2.0 * spec.slippage_price_per_side / risk_distance
    )
    commission_r = commission_price / risk_distance
    swap_r = swap_price_cost / risk_distance
    return CostEstimate(
        profile_id=profile.profile_id,
        profile_sha256=profile.profile_sha256,
        symbol=spec.symbol,
        spread_r=float(spread_r),
        slippage_r=float(slippage_r),
        commission_r=float(commission_r),
        swap_r=float(swap_r),
        rollover_units=float(units),
        total_cost_r=float(
            spread_r + slippage_r + commission_r + swap_r
        ),
    )


def estimate_realized_cost_r(
    profile: CostProfile,
    setup: Mapping[str, Any],
) -> CostEstimate:
    units = rollover_units(
        setup.get("entry_time"),
        setup.get("exit_time"),
        timezone_name=profile.rollover_timezone,
        rollover_hour=profile.rollover_hour,
        triple_swap_weekday=profile.triple_swap_weekday,
    )
    return estimate_cost_r(
        profile,
        symbol=str(setup.get("symbol") or ""),
        side=str(setup.get("side") or ""),
        entry_price=setup.get("entry"),
        stop_price=setup.get("stop"),
        rollover_unit_count=units,
    )


def apply_costs_to_setups(
    setups: Sequence[Mapping[str, Any]],
    profile: CostProfile,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in setups:
        row = dict(source)
        gross = _finite(row.get("net_r"), name="net_r")
        row["gross_net_r"] = gross
        raw_risk_amount = row.get("risk_amount")
        risk_amount = (
            _finite(raw_risk_amount, name="risk_amount")
            if raw_risk_amount is not None
            else None
        )
        row["gross_pnl_amount"] = (
            gross * risk_amount if risk_amount is not None else None
        )
        if (
            str(row.get("status") or "CLOSED").upper() != "CLOSED"
            or row.get("exit_time") is None
        ):
            row.update({
                "cost_profile_id": profile.profile_id,
                "cost_profile_sha256": profile.profile_sha256,
                "cost_symbol": str(row.get("symbol") or "").upper(),
                "cost_status": "UNKNOWN_OPEN_SETUP",
                "spread_cost_r": None,
                "slippage_cost_r": None,
                "commission_cost_r": None,
                "swap_cost_r": None,
                "rollover_units": None,
                "transaction_cost_r": None,
                "net_after_cost_r": None,
            })
            output.append(row)
            continue
        estimate = estimate_realized_cost_r(profile, row)
        row.update(estimate.to_dict())
        row["cost_status"] = "APPLIED_REALIZED_HOLDING_TIME"
        row["net_after_cost_r"] = gross - estimate.total_cost_r
        # Existing metrics consume net_r. Preserve gross explicitly and make
        # the profile-applied population unambiguously net of all components.
        row["net_r"] = row["net_after_cost_r"]
        if risk_amount is not None:
            row["pnl_amount"] = row["net_r"] * risk_amount
        output.append(row)
    return output


__all__ = [
    "COST_PROFILE_SCHEMA",
    "CostEstimate",
    "CostModelError",
    "CostProfile",
    "SymbolCostSpec",
    "apply_costs_to_setups",
    "estimate_cost_r",
    "estimate_realized_cost_r",
    "rollover_units",
]
