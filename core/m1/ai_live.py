# core/m1/ai_live.py   (у тебя было "ai_liv" — файл лучше назвать ai_live.py)
from __future__ import annotations

from typing import Any, Dict

from core.m1.config import AIConfig
from core.m1.store import TradeStore


class AILive:
    """
    Лёгкий AI слой:
      - для ENTER: оцениваем p(TP) по истории (по символу) и можем отклонить
      - для EXIT_*: обновляем статистику в TradeStore
    """

    def __init__(self, cfg: AIConfig, store: TradeStore, strategy: Any):
        self.cfg = cfg
        self.store = store
        self.strategy = strategy

    def on_signal(self, symbol: str, sig: Dict[str, Any], data: Dict[str, Any], active_trades: Dict[str, Any]) -> Dict[str, Any]:
        if not self.cfg.enabled:
            return sig
        if not isinstance(sig, dict):
            return sig

        st = sig.get("signal")

        # ---- update on close ----
        if st in ("EXIT_TP", "EXIT_SL"):
            outcome = "TP" if st == "EXIT_TP" else "SL"

            # rr_numeric можно попытаться восстановить из active_trades (если есть)
            rr_numeric = None
            tr = active_trades.get(symbol)
            if tr is not None:
                try:
                    reward = abs(float(tr.tp) - float(tr.entry))
                    risk = abs(float(tr.entry) - float(tr.stop))
                    rr_numeric = (reward / risk) if risk > 0 else None
                except Exception:
                    rr_numeric = None

            self.store.update_on_close(symbol, outcome, rr_numeric=rr_numeric)

            # можно добавить немного инфы в сообщение
            stats = self.store.get_symbol_stats(symbol)
            sig["ai_stats_closed"] = stats["closed"]
            sig["ai_p_tp"] = round(float(stats["p_tp"]), 3)
            return sig

        # ---- filter on enter ----
        if st == "ENTER":
            rr_numeric = sig.get("rr_numeric")
            try:
                rr_numeric_f = float(rr_numeric) if rr_numeric is not None else 0.0
            except Exception:
                rr_numeric_f = 0.0

            if rr_numeric_f < float(self.cfg.min_rr):
                return {
                    "signal": "AI_REJECT",
                    "reason": f"RR<{self.cfg.min_rr:.2f}",
                    "ai_p_tp": None,
                }

            stats = self.store.get_symbol_stats(symbol)
            p_tp = float(stats["p_tp"])
            closed = int(stats["closed"])

            # если мало сделок — НЕ режем жёстко, но даём p(TP) в текст
            sig["ai_p_tp"] = round(p_tp, 3)
            sig["ai_stats_closed"] = closed

            # Порог = break-even winrate + запас.
            # BE winrate = 1 / (1 + RR): минимальный винрейт при котором сетап в 0.
            # Отклоняем только если статистика достаточна и p(TP) ниже порога.
            if rr_numeric_f > 0:
                be_p = 1.0 / (1.0 + rr_numeric_f)
                effective_threshold = be_p + float(self.cfg.min_edge_above_be)
            else:
                effective_threshold = float(self.cfg.min_p_tp)

            if closed >= int(self.cfg.min_closed_per_symbol) and p_tp < effective_threshold:
                return {
                    "signal": "AI_REJECT",
                    "reason": (
                        f"pTP={p_tp:.2f} < BE+edge={effective_threshold:.2f}"
                        f" (RR={rr_numeric_f:.2f}, n={closed})"
                    ),
                    "ai_p_tp": round(p_tp, 3),
                }

            return sig

        return sig
