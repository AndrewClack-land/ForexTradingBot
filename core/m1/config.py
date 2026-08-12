# core/m1/config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key, "")
    if not v:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class AIConfig:
    """
    AI/ML filter on top of strategy:
    - evaluates p(TP) from per-symbol historical stats
    - rejects entries when quality is below threshold
    """
    enabled: bool = field(default_factory=lambda: _env_bool("AI_ENABLED", True))

    # Legacy binary TP/SL by trigger is not an execution-quality model for
    # split/BE/time exits. Keep both its capture and gate dormant until a
    # durable idea-level realized-net-R calibration replaces it.
    trigger_calibration_enabled: bool = field(
        default_factory=lambda: _env_bool(
            "AI_TRIGGER_CALIBRATION_ENABLED",
            False,
        )
    )

    # Legacy aggregate threshold retained for diagnostics.
    min_closed_per_symbol: int = field(default_factory=lambda: _env_int("AI_MIN_CLOSED", 30))

    # Entry quality is calibrated on the independent-idea source, not pooled
    # across unrelated trigger families on the same symbol.
    min_closed_per_trigger: int = field(
        default_factory=lambda: _env_int("AI_MIN_CLOSED_TRIGGER", 30)
    )

    # Absolute minimum p(TP) threshold — fallback when rr_numeric unavailable
    min_p_tp: float = field(default_factory=lambda: _env_float("AI_MIN_P_TP", 0.20))

    # Required edge above break-even winrate (1 / (1 + RR))
    # 0.0 = block only if win-rate is below break-even (negative EV)
    min_edge_above_be: float = field(default_factory=lambda: _env_float("AI_MIN_EDGE_ABOVE_BE", 0.0))

    # Minimum RR — immediate reject if strategy gives less
    min_rr: float = field(default_factory=lambda: _env_float("AI_MIN_RR", 1.3))

    # Beta prior smoothing: p = (tp + alpha) / (tp + sl + alpha + beta)
    alpha: float = field(default_factory=lambda: _env_float("AI_ALPHA", 1.0))
    beta: float = field(default_factory=lambda: _env_float("AI_BETA", 1.0))

    # SQLite stats file location
    db_filename: str = "ai_stats.db"
