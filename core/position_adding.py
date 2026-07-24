# core/position_adding.py
"""Position Adding — staged entries into one trading idea.

The methodology: entry 1 opens on the idea with 1% risk. Each later confirmation
of the same direction adds another 1% entry, and before the aggregate risk of the
idea would exceed its cap (2%), the oldest still-risking entries are moved to
break-even. A failed idea therefore costs the cap, not the sum of its entries,
while a working idea carries several times the base volume into the targets.

This module owns the decision only. Execution, break-even orders and lifecycle
management stay in the caller (main.Core) and the MT5 executor, so the rules here
stay testable without a broker connection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class PyramidSettings:
    enabled: bool = False
    max_entries: int = 3
    max_idea_risk_pct: float = 0.02
    min_progress_r: float = 0.5
    max_progress_pct: float = 0.5


@dataclass
class AddonDecision:
    """Verdict on adding one more entry to an open idea."""

    allowed: bool
    reason: str
    # Entries that must be at break-even before the add-on may be opened, in the
    # order they should be moved (oldest first).
    entries_to_breakeven: List[Any] = field(default_factory=list)
    # Risk of the idea as it stands, and as it would stand after the add-on is
    # opened and the break-even moves above have succeeded — both as fractions
    # of the risk capital base.
    idea_risk_pct: float = 0.0
    projected_risk_pct: float = 0.0
    entry_index: int = 0

    def as_signal_info(self) -> str:
        return (
            f"{self.reason} | риск идеи {self.idea_risk_pct:.2%}"
            f" → {self.projected_risk_pct:.2%}"
        )


class PyramidRiskError(RuntimeError):
    """Idea risk could not be measured, so no add-on may be opened."""


class PyramidManager:
    """Applies the Position Adding rules to an open idea."""

    def __init__(self, settings: PyramidSettings):
        self.settings = settings

    # ---------------- risk ----------------

    def entry_risk_amount(self, executor: Any, entry: Any) -> float:
        """Money still at risk on one entry (0.0 once it sits at break-even)."""
        volume = float(getattr(entry, "volume_remaining", 0.0) or 0.0)
        if volume <= 0:
            volume = float(getattr(entry, "volume", 0.0) or 0.0)
        if volume <= 0:
            return 0.0
        try:
            return float(
                executor.open_risk_amount(
                    str(getattr(entry, "symbol", "") or ""),
                    side=str(getattr(entry, "side", "") or ""),
                    entry_price=float(getattr(entry, "entry", 0.0) or 0.0),
                    stop_price=float(getattr(entry, "stop", 0.0) or 0.0),
                    volume=volume,
                )
            )
        except Exception as exc:
            raise PyramidRiskError(f"open risk unavailable: {exc}") from exc

    def idea_risk_pct(self, executor: Any, trade: Any) -> float:
        """Aggregate open risk of every entry of the idea, as a capital fraction."""
        capital = self._capital_base(executor)
        total = sum(
            self.entry_risk_amount(executor, entry) for entry in _entries(trade)
        )
        return total / capital

    def _capital_base(self, executor: Any) -> float:
        try:
            capital = float(executor.risk_capital_base())
        except Exception as exc:
            raise PyramidRiskError(f"risk capital unavailable: {exc}") from exc
        if not capital > 0:
            raise PyramidRiskError(f"invalid risk capital base {capital!r}")
        return capital

    # ---------------- decision ----------------

    def evaluate(
        self,
        trade: Any,
        sig: Dict[str, Any],
        *,
        executor: Any,
        last_price: float,
        trigger_signature: str,
    ) -> AddonDecision:
        """Decide whether ``sig`` may be opened as the idea's next entry."""
        cfg = self.settings
        entries = _entries(trade)
        decision_index = len(entries) + 1

        def block(reason: str, **kw: Any) -> AddonDecision:
            return AddonDecision(
                allowed=False, reason=reason, entry_index=decision_index, **kw
            )

        if not cfg.enabled:
            return block("Position Adding выключен")
        if executor is None:
            return block("Нет подключения к MT5 — риск идеи неизмерим")
        if len(entries) >= max(1, int(cfg.max_entries)):
            return block(f"Достигнут лимит {cfg.max_entries} входов на идею")

        side = str(sig.get("side") or "").upper()
        if not side or side != str(getattr(trade, "side", "") or "").upper():
            return block("Сигнал не совпадает с направлением идеи")

        used = set(getattr(trade, "idea_trigger_signatures", []) or [])
        if trigger_signature in used:
            return block("Подтверждение повторяет триггер уже открытого входа")

        progress_reason = self._check_progress(trade, entries, last_price)
        if progress_reason:
            return block(progress_reason)

        # Risk: the add-on is sized to the standard per-trade fraction, so the
        # idea must have room for it — freeing room by moving the oldest still
        # risking entries to break-even, exactly as the methodology prescribes.
        capital = self._capital_base(executor)
        risks = [self.entry_risk_amount(executor, entry) for entry in entries]
        try:
            planned = float(executor.planned_entry_risk_fraction()) * capital
        except Exception as exc:
            raise PyramidRiskError(f"planned entry risk unavailable: {exc}") from exc

        budget = float(cfg.max_idea_risk_pct) * capital
        idea_risk = sum(risks)
        if planned > budget:
            return block(
                "Риск одного входа превышает лимит идеи",
                idea_risk_pct=idea_risk / capital,
                projected_risk_pct=(idea_risk + planned) / capital,
            )

        to_breakeven: List[Any] = []
        projected = idea_risk + planned
        # The newest entry keeps its stop — it is the one still proving the idea.
        for entry, risk in list(zip(entries, risks))[:-1]:
            if projected <= budget:
                break
            if risk <= 0:
                continue
            to_breakeven.append(entry)
            projected -= risk
        if projected > budget:
            return block(
                "Совокупный риск идеи не помещается в лимит даже после безубытка",
                idea_risk_pct=idea_risk / capital,
                projected_risk_pct=projected / capital,
            )

        return AddonDecision(
            allowed=True,
            reason=f"Подтверждение идеи: вход {decision_index}/{cfg.max_entries}",
            entries_to_breakeven=to_breakeven,
            idea_risk_pct=idea_risk / capital,
            projected_risk_pct=projected / capital,
            entry_index=decision_index,
        )

    # ---------------- progress rules ----------------

    def _check_progress(
        self, trade: Any, entries: Sequence[Any], last_price: float
    ) -> Optional[str]:
        """Empty when price confirms the idea; otherwise the blocking reason."""
        cfg = self.settings
        try:
            price = float(last_price)
        except (TypeError, ValueError):
            return "Нет цены для проверки прогресса идеи"
        if not price > 0:
            return "Нет цены для проверки прогресса идеи"

        side = str(getattr(trade, "side", "") or "").upper()
        primary = entries[0]
        newest = entries[-1]
        sign = 1.0 if side == "LONG" else -1.0

        unit_r = _risk_distance(newest) or _risk_distance(primary)
        if not unit_r:
            return "Не удалось определить 1R идеи"
        progress_r = sign * (price - float(newest.entry)) / unit_r
        if progress_r < cfg.min_progress_r:
            return (
                f"Идея ещё не подтверждена движением: {progress_r:.2f}R"
                f" < {cfg.min_progress_r:.2f}R от входа {len(entries)}"
            )

        # Kolachi rule: the closer price is to the target, the worse the reward
        # left for a new entry's risk.
        tps = [float(x) for x in (getattr(primary, "tp_prices", []) or [])]
        if tps:
            final_tp = max(tps) if side == "LONG" else min(tps)
            span = sign * (final_tp - float(primary.entry))
            if span > 0:
                travelled = sign * (price - float(primary.entry)) / span
                if travelled > cfg.max_progress_pct:
                    return (
                        f"Пройдено {travelled:.0%} пути до финального TP"
                        f" (> {cfg.max_progress_pct:.0%}) — добавление запрещено"
                    )
        return None


def _entries(trade: Any) -> List[Any]:
    entries = getattr(trade, "entries", None)
    if callable(entries):
        return entries()
    return [trade]


def _risk_distance(entry: Any) -> float:
    """|entry - stop| of one entry; 0.0 when it is already at break-even."""
    try:
        return abs(float(entry.entry) - float(entry.stop))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def build_manager() -> PyramidManager:
    """PyramidManager configured from the project config module."""
    try:
        import config as _cfg
    except Exception:
        _cfg = None
    return PyramidManager(
        PyramidSettings(
            enabled=bool(getattr(_cfg, "POSITION_ADDING_ENABLED", False)),
            max_entries=int(getattr(_cfg, "IDEA_MAX_ENTRIES", 3)),
            max_idea_risk_pct=float(getattr(_cfg, "IDEA_MAX_RISK_PCT", 0.02)),
            min_progress_r=float(getattr(_cfg, "ADDON_MIN_PROGRESS_R", 0.5)),
            max_progress_pct=float(getattr(_cfg, "ADDON_MAX_PROGRESS_PCT", 0.5)),
        )
    )
