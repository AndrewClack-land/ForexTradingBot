"""Setup-level performance metrics.

One split setup is one statistical observation regardless of how many TP legs
the broker closed.  This avoids inflating sample size and win rate.
"""

from __future__ import annotations

from statistics import median
from typing import Any, Iterable, Mapping, Optional

from .simulator import SetupOutcome


def _field(item: SetupOutcome | Mapping[str, Any], name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _max_drawdown(values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return float(max_drawdown)


def _longest_loss_streak(values: list[float], epsilon: float) -> int:
    longest = 0
    current = 0
    for value in values:
        if value < -epsilon:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def aggregate_setup_metrics(
    outcomes: Iterable[SetupOutcome | Mapping[str, Any]],
    *,
    epsilon: float = 1e-9,
) -> dict[str, Any]:
    """Aggregate closed setup outcomes in chronological input order.

    Open setups are reported but excluded from expectancy/win-rate metrics.
    ``profit_factor`` is ``None`` when no losing setup exists.
    """

    items = list(outcomes)
    closed = [item for item in items if str(_field(item, "status", "CLOSED")).upper() == "CLOSED"]
    values = [float(_field(item, "net_r", 0.0)) for item in closed]
    wins = [value for value in values if value > epsilon]
    losses = [value for value in values if value < -epsilon]
    breakeven = [value for value in values if abs(value) <= epsilon]
    gross_profit = float(sum(wins))
    gross_loss = float(abs(sum(losses)))
    profit_factor: Optional[float] = (
        gross_profit / gross_loss if gross_loss > epsilon else None
    )

    tp_counts = {1: 0, 2: 0, 3: 0}
    moved_to_be = 0
    ambiguous_bars = 0
    for item in closed:
        hits = {int(value) for value in (_field(item, "tp_hits", ()) or ())}
        for tp_index in tp_counts:
            if tp_index in hits:
                tp_counts[tp_index] += 1
        moved_to_be += int(bool(_field(item, "moved_to_be", False)))
        ambiguous_bars += int(_field(item, "ambiguous_bars", 0) or 0)

    count = len(values)
    denominator = float(count) if count else 1.0
    return {
        "setups_total": len(items),
        "setups_closed": count,
        "setups_open": len(items) - count,
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(breakeven),
        "win_rate": len(wins) / denominator if count else 0.0,
        "loss_rate": len(losses) / denominator if count else 0.0,
        "breakeven_rate": len(breakeven) / denominator if count else 0.0,
        "net_r": float(sum(values)),
        "expectancy_r": float(sum(values) / count) if count else 0.0,
        "median_r": float(median(values)) if values else 0.0,
        "average_win_r": float(sum(wins) / len(wins)) if wins else 0.0,
        "average_loss_r": float(sum(losses) / len(losses)) if losses else 0.0,
        "gross_profit_r": gross_profit,
        "gross_loss_r": gross_loss,
        "profit_factor": profit_factor,
        "max_drawdown_r": _max_drawdown(values),
        "longest_loss_streak": _longest_loss_streak(values, epsilon),
        "tp1_reach_rate": tp_counts[1] / denominator if count else 0.0,
        "tp2_reach_rate": tp_counts[2] / denominator if count else 0.0,
        "tp3_reach_rate": tp_counts[3] / denominator if count else 0.0,
        "moved_to_be_rate": moved_to_be / denominator if count else 0.0,
        "ambiguous_bars": ambiguous_bars,
    }
