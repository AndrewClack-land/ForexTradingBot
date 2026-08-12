"""Causal spread, rollover and point-in-time news entry filters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo


EXECUTION_QUALITY_SCHEMA = "execution-quality-profile-v1"
NEWS_CALENDAR_SCHEMA = "point-in-time-news-calendar-v1"


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


def _hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class SymbolExecutionQuality:
    symbol: str
    currencies: tuple[str, ...]
    max_spread_price: float
    max_spread_r: float

    @classmethod
    def from_mapping(
        cls,
        symbol: str,
        payload: Mapping[str, Any],
    ) -> "SymbolExecutionQuality":
        currencies = tuple(
            str(item).strip().upper()
            for item in payload.get("currencies", ())
            if str(item).strip()
        )
        result = cls(
            symbol=str(symbol).strip().upper(),
            currencies=currencies,
            max_spread_price=_finite(
                payload.get("max_spread_price"),
                name=f"{symbol}.max_spread_price",
            ),
            max_spread_r=_finite(
                payload.get("max_spread_r"),
                name=f"{symbol}.max_spread_r",
            ),
        )
        if not result.symbol or not result.currencies:
            raise ValueError(f"{symbol}: symbol and currencies are required")
        if result.max_spread_price <= 0 or result.max_spread_r <= 0:
            raise ValueError(f"{symbol}: spread thresholds must be positive")
        return result


@dataclass(frozen=True)
class ExecutionQualityProfile:
    profile_id: str
    profile_sha256: str
    measured_from: str
    created_at_utc: datetime
    rollover_timezone: str
    rollover_hour: int
    rollover_minute: int
    rollover_before_minutes: int
    rollover_after_minutes: int
    news_filter_enabled: bool
    news_before_minutes: int
    news_after_minutes: int
    news_min_impact: int
    symbols: Mapping[str, SymbolExecutionQuality]

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "ExecutionQualityProfile":
        if str(payload.get("schema") or "") != EXECUTION_QUALITY_SCHEMA:
            raise ValueError(
                f"execution-quality schema must be "
                f"{EXECUTION_QUALITY_SCHEMA}"
            )
        profile_id = str(payload.get("profile_id") or "").strip()
        measured_from = str(payload.get("measured_from") or "").strip()
        if not profile_id or not measured_from:
            raise ValueError("profile_id and measured_from are required")
        timezone_name = str(
            payload.get("rollover_timezone") or ""
        ).strip()
        try:
            ZoneInfo(timezone_name)
        except Exception as exc:
            raise ValueError(
                f"unknown rollover_timezone: {timezone_name}"
            ) from exc
        raw_symbols = payload.get("symbols")
        if not isinstance(raw_symbols, Mapping) or not raw_symbols:
            raise ValueError("symbols must be a non-empty mapping")
        symbols = {
            str(symbol).strip().upper(): (
                SymbolExecutionQuality.from_mapping(str(symbol), spec)
            )
            for symbol, spec in raw_symbols.items()
            if isinstance(spec, Mapping)
        }
        if len(symbols) != len(raw_symbols):
            raise ValueError("every symbol spec must be a mapping")
        result = cls(
            profile_id=profile_id,
            profile_sha256=_hash(dict(payload)),
            measured_from=measured_from,
            created_at_utc=_utc(
                payload.get("created_at_utc"),
                name="created_at_utc",
            ),
            rollover_timezone=timezone_name,
            rollover_hour=int(payload.get("rollover_hour", 0)),
            rollover_minute=int(payload.get("rollover_minute", 0)),
            rollover_before_minutes=int(
                payload.get("rollover_before_minutes", 0)
            ),
            rollover_after_minutes=int(
                payload.get("rollover_after_minutes", 0)
            ),
            news_filter_enabled=bool(
                payload.get("news_filter_enabled", False)
            ),
            news_before_minutes=int(
                payload.get("news_before_minutes", 0)
            ),
            news_after_minutes=int(
                payload.get("news_after_minutes", 0)
            ),
            news_min_impact=int(payload.get("news_min_impact", 3)),
            symbols=symbols,
        )
        if not 0 <= result.rollover_hour <= 23:
            raise ValueError("rollover_hour must be 0..23")
        if not 0 <= result.rollover_minute <= 59:
            raise ValueError("rollover_minute must be 0..59")
        for name in (
            "rollover_before_minutes",
            "rollover_after_minutes",
            "news_before_minutes",
            "news_after_minutes",
        ):
            if getattr(result, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if not 1 <= result.news_min_impact <= 3:
            raise ValueError("news_min_impact must be 1..3")
        return result

    @classmethod
    def load(cls, path: Path | str) -> "ExecutionQualityProfile":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("execution-quality profile must be an object")
        return cls.from_mapping(payload)

    def symbol(self, symbol: str) -> SymbolExecutionQuality:
        key = str(symbol).strip().upper()
        try:
            return self.symbols[key]
        except KeyError as exc:
            raise ValueError(
                f"execution-quality profile has no symbol {key}"
            ) from exc


@dataclass(frozen=True)
class NewsRevision:
    event_id: str
    revision_id: str
    known_at_utc: datetime
    scheduled_at_utc: datetime
    currencies: tuple[str, ...]
    impact: int
    title: str


@dataclass(frozen=True)
class PointInTimeNewsCalendar:
    calendar_id: str
    calendar_sha256: str
    source: str
    captured_from: str
    revisions: tuple[NewsRevision, ...]

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "PointInTimeNewsCalendar":
        if str(payload.get("schema") or "") != NEWS_CALENDAR_SCHEMA:
            raise ValueError(
                f"news calendar schema must be {NEWS_CALENDAR_SCHEMA}"
            )
        calendar_id = str(payload.get("calendar_id") or "").strip()
        source = str(payload.get("source") or "").strip()
        captured_from = str(payload.get("captured_from") or "").strip()
        if not calendar_id or not source or not captured_from:
            raise ValueError(
                "calendar_id, source and captured_from are required"
            )
        raw_revisions = payload.get("revisions")
        if not isinstance(raw_revisions, Sequence):
            raise ValueError("revisions must be a sequence")
        revisions: list[NewsRevision] = []
        identities: set[tuple[str, str]] = set()
        for raw in raw_revisions:
            if not isinstance(raw, Mapping):
                raise ValueError("every news revision must be an object")
            event_id = str(raw.get("event_id") or "").strip()
            revision_id = str(raw.get("revision_id") or "").strip()
            currencies = tuple(
                str(item).strip().upper()
                for item in raw.get("currencies", ())
                if str(item).strip()
            )
            impact = int(raw.get("impact", 0))
            if (
                not event_id
                or not revision_id
                or not currencies
                or not 1 <= impact <= 3
            ):
                raise ValueError("invalid news revision identity/impact")
            identity = (event_id, revision_id)
            if identity in identities:
                raise ValueError("duplicate news revision identity")
            identities.add(identity)
            revisions.append(NewsRevision(
                event_id=event_id,
                revision_id=revision_id,
                known_at_utc=_utc(
                    raw.get("known_at_utc"),
                    name="known_at_utc",
                ),
                scheduled_at_utc=_utc(
                    raw.get("scheduled_at_utc"),
                    name="scheduled_at_utc",
                ),
                currencies=currencies,
                impact=impact,
                title=str(raw.get("title") or ""),
            ))
        return cls(
            calendar_id=calendar_id,
            calendar_sha256=_hash(dict(payload)),
            source=source,
            captured_from=captured_from,
            revisions=tuple(revisions),
        )

    @classmethod
    def load(cls, path: Path | str) -> "PointInTimeNewsCalendar":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("news calendar root must be an object")
        return cls.from_mapping(payload)

    def blocking_events(
        self,
        *,
        as_of_utc: datetime,
        currencies: Sequence[str],
        before_minutes: int,
        after_minutes: int,
        min_impact: int,
    ) -> list[NewsRevision]:
        as_of = _utc(as_of_utc, name="as_of_utc")
        currency_set = {
            str(item).strip().upper()
            for item in currencies
            if str(item).strip()
        }
        latest: dict[str, NewsRevision] = {}
        for revision in self.revisions:
            if revision.known_at_utc > as_of:
                continue
            previous = latest.get(revision.event_id)
            if (
                previous is None
                or revision.known_at_utc > previous.known_at_utc
                or (
                    revision.known_at_utc == previous.known_at_utc
                    and revision.revision_id > previous.revision_id
                )
            ):
                latest[revision.event_id] = revision
        before = timedelta(minutes=int(before_minutes))
        after = timedelta(minutes=int(after_minutes))
        return sorted(
            (
                revision
                for revision in latest.values()
                if revision.impact >= int(min_impact)
                and currency_set.intersection(revision.currencies)
                and revision.scheduled_at_utc - before
                <= as_of
                < revision.scheduled_at_utc + after
            ),
            key=lambda item: (
                item.scheduled_at_utc,
                item.event_id,
            ),
        )


def _inside_rollover_window(
    now_utc: datetime,
    profile: ExecutionQualityProfile,
) -> bool:
    zone = ZoneInfo(profile.rollover_timezone)
    local_now = _utc(now_utc, name="now_utc").astimezone(zone)
    for offset in (-1, 0, 1):
        day = local_now.date() + timedelta(days=offset)
        cutoff = datetime(
            day.year,
            day.month,
            day.day,
            profile.rollover_hour,
            profile.rollover_minute,
            tzinfo=zone,
        )
        start = cutoff - timedelta(
            minutes=profile.rollover_before_minutes
        )
        end = cutoff + timedelta(
            minutes=profile.rollover_after_minutes
        )
        if start <= local_now < end:
            return True
    return False


def assess_execution_quality(
    profile: ExecutionQualityProfile,
    *,
    symbol: str,
    signal: Mapping[str, Any],
    bid: Any,
    ask: Any,
    observed_at_utc: datetime,
    news_calendar: Optional[PointInTimeNewsCalendar],
) -> dict[str, Any]:
    spec = profile.symbol(symbol)
    bid_value = _finite(bid, name="bid")
    ask_value = _finite(ask, name="ask")
    if bid_value <= 0 or ask_value < bid_value:
        raise ValueError("invalid causal bid/ask quote")
    entry = _finite(signal.get("entry_price"), name="entry_price")
    stop = _finite(signal.get("stop_price"), name="stop_price")
    risk_distance = abs(entry - stop)
    if risk_distance <= 0:
        raise ValueError("entry/stop risk distance must be positive")
    spread = ask_value - bid_value
    spread_r = spread / risk_distance
    base = {
        "profile_id": profile.profile_id,
        "profile_sha256": profile.profile_sha256,
        "observed_at_utc": _utc(
            observed_at_utc,
            name="observed_at_utc",
        ).isoformat(),
        "bid": bid_value,
        "ask": ask_value,
        "spread_price": spread,
        "spread_r": spread_r,
        "max_spread_price": spec.max_spread_price,
        "max_spread_r": spec.max_spread_r,
    }
    if (
        spread > spec.max_spread_price
        or spread_r > spec.max_spread_r
    ):
        return {**base, "allowed": False, "reason": "SPREAD"}
    if _inside_rollover_window(observed_at_utc, profile):
        return {**base, "allowed": False, "reason": "ROLLOVER"}
    if profile.news_filter_enabled:
        if news_calendar is None:
            return {
                **base,
                "allowed": False,
                "reason": "NEWS_DATA_UNAVAILABLE",
            }
        events = news_calendar.blocking_events(
            as_of_utc=observed_at_utc,
            currencies=spec.currencies,
            before_minutes=profile.news_before_minutes,
            after_minutes=profile.news_after_minutes,
            min_impact=profile.news_min_impact,
        )
        if events:
            return {
                **base,
                "allowed": False,
                "reason": "NEWS",
                "news_calendar_id": news_calendar.calendar_id,
                "news_calendar_sha256": news_calendar.calendar_sha256,
                "blocking_events": [
                    {
                        "event_id": event.event_id,
                        "revision_id": event.revision_id,
                        "scheduled_at_utc": (
                            event.scheduled_at_utc.isoformat()
                        ),
                        "impact": event.impact,
                        "title": event.title,
                    }
                    for event in events
                ],
            }
    return {**base, "allowed": True, "reason": "ALLOW"}


__all__ = [
    "EXECUTION_QUALITY_SCHEMA",
    "NEWS_CALENDAR_SCHEMA",
    "ExecutionQualityProfile",
    "PointInTimeNewsCalendar",
    "SymbolExecutionQuality",
    "assess_execution_quality",
]
