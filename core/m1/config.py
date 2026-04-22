# core/m1/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AIConfig:
    """
    Простой AI/ML “фильтр” поверх стратегии:
    - оценивает p(TP) по исторической статистике (по символу)
    - может отклонять входы, если качество низкое
    """
    enabled: bool = True

    # Минимум закрытых сделок по символу, чтобы доверять статистике
    min_closed_per_symbol: int = 20

    # Порог вероятности для пропуска ENTER
    min_p_tp: float = 0.52

    # Минимальный RR (если стратегия дала меньше — сразу reject)
    min_rr: float = 1.3

    # Сглаживание (beta prior): p=(tp+alpha)/(tp+sl+alpha+beta)
    alpha: float = 1.0
    beta: float = 1.0

    # где хранить ai статистику (sqlite)
    db_filename: str = "ai_stats.db"
