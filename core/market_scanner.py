# core/market_scanner.py
from __future__ import annotations

from typing import Dict, List


class MarketScanner:
    """
    Для FX/металлов/индексов (CFD) нам НЕ нужен динамический сканер как на Binance.
    Мы работаем с фиксированным universe из config.UNIVERSE.

    universe формат:
      {
        "XAUUSD": "OANDA:XAUUSD",
        ...
      }

    scan() возвращает список "внутренних" символов:
      ["EURUSD", "GBPUSD", ...]
    """

    def __init__(self, universe: Dict[str, str]):
        self.universe = universe or {}

    def scan(self) -> List[str]:
        # Возвращаем ключи universe в стабильном порядке добавления
        return list(self.universe.keys())
