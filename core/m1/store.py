# core/m1/store.py
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

from config import AI_DATA_DIR
from core.m1.config import AIConfig


class TradeStore:
    """
    Мини-стор для AI статистики (по символу):
      - сколько TP/SL
      - средний RR (опционально)

    Храним в SQLite: ai_data/ai_stats.db
    """

    def __init__(self, cfg: AIConfig):
        self.cfg = cfg
        self.db_path = Path(AI_DATA_DIR) / cfg.db_filename
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        try:
            self._conn.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            pass

        self._ensure_schema()

    def close(self):
        with self._lock:
            try:
                self._conn.commit()
            finally:
                self._conn.close()

    def _ensure_schema(self):
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS symbol_stats (
                    symbol TEXT PRIMARY KEY,
                    tp INTEGER NOT NULL DEFAULT 0,
                    sl INTEGER NOT NULL DEFAULT 0,
                    rr_sum REAL NOT NULL DEFAULT 0.0,
                    rr_n INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            self._conn.commit()

    # ---------- API ----------
    def update_on_close(self, symbol: str, outcome: str, rr_numeric: Optional[float] = None) -> None:
        """
        outcome: "TP" | "SL"
        rr_numeric: можно передать, но не обязателен
        """
        outcome = (outcome or "").upper()
        if outcome not in ("TP", "SL"):
            return

        with self._lock:
            row = self._conn.execute(
                "SELECT symbol, tp, sl, rr_sum, rr_n FROM symbol_stats WHERE symbol=?;",
                (symbol,),
            ).fetchone()

            if row is None:
                tp = 1 if outcome == "TP" else 0
                sl = 1 if outcome == "SL" else 0
                rr_sum = float(rr_numeric) if rr_numeric is not None else 0.0
                rr_n = 1 if rr_numeric is not None else 0

                self._conn.execute(
                    "INSERT INTO symbol_stats(symbol,tp,sl,rr_sum,rr_n) VALUES(?,?,?,?,?);",
                    (symbol, tp, sl, rr_sum, rr_n),
                )
            else:
                tp = int(row["tp"])
                sl = int(row["sl"])
                rr_sum = float(row["rr_sum"])
                rr_n = int(row["rr_n"])

                if outcome == "TP":
                    tp += 1
                else:
                    sl += 1

                if rr_numeric is not None:
                    rr_sum += float(rr_numeric)
                    rr_n += 1

                self._conn.execute(
                    "UPDATE symbol_stats SET tp=?, sl=?, rr_sum=?, rr_n=? WHERE symbol=?;",
                    (tp, sl, rr_sum, rr_n, symbol),
                )

            self._conn.commit()

    def get_symbol_stats(self, symbol: str) -> Dict[str, float]:
        with self._lock:
            row = self._conn.execute(
                "SELECT tp, sl, rr_sum, rr_n FROM symbol_stats WHERE symbol=?;",
                (symbol,),
            ).fetchone()

        if row is None:
            return {"tp": 0, "sl": 0, "closed": 0, "p_tp": 0.5, "rr_avg": 0.0}

        tp = int(row["tp"])
        sl = int(row["sl"])
        closed = tp + sl

        p_tp = self.estimate_p_tp(tp, sl)
        rr_avg = (float(row["rr_sum"]) / int(row["rr_n"])) if int(row["rr_n"]) > 0 else 0.0

        return {"tp": tp, "sl": sl, "closed": closed, "p_tp": p_tp, "rr_avg": rr_avg}

    def estimate_p_tp(self, tp: int, sl: int) -> float:
        # beta prior smoothing
        a = float(self.cfg.alpha)
        b = float(self.cfg.beta)
        denom = (tp + sl + a + b)
        return float((tp + a) / denom) if denom > 0 else 0.5
