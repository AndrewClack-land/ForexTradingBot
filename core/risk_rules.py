from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Any, Optional

from core.strategy_narrative import ActiveTrade


@dataclass
class RiskRules:
    max_hold_minutes: int = 180
    force_exit_grace_minutes: int = 30

    def check_trade(self, trade: ActiveTrade, *, last_price: float) -> Optional[Dict[str, Any]]:
        now = time.time()
        elapsed_minutes = (now - float(trade.ts_open or now)) / 60.0

        if elapsed_minutes > (self.max_hold_minutes + self.force_exit_grace_minutes):
            return {
                "signal": "EXIT_TIME",
                "exit_price": float(last_price),
                "info": "Forced exit by max_hold",
            }

        return None
