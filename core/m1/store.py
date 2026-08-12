# core/m1/store.py
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Optional

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
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS symbol_trigger_stats (
                    symbol TEXT NOT NULL,
                    trigger_kind TEXT NOT NULL,
                    tp INTEGER NOT NULL DEFAULT 0,
                    sl INTEGER NOT NULL DEFAULT 0,
                    rr_sum REAL NOT NULL DEFAULT 0.0,
                    rr_n INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(symbol, trigger_kind)
                );
                """
            )
            self._conn.commit()

    # ---------- API ----------
    @staticmethod
    def _key(value: Any) -> str:
        return str(value or "").strip().upper()

    @staticmethod
    def _trigger_key(value: Any) -> str:
        return str(value or "").strip().lower()

    def _upsert_stats(
        self,
        *,
        table: str,
        symbol: str,
        outcome: str,
        rr_numeric: Optional[float],
        trigger_kind: Optional[str] = None,
    ) -> None:
        tp = 1 if outcome == "TP" else 0
        sl = 1 if outcome == "SL" else 0
        rr_sum = float(rr_numeric) if rr_numeric is not None else 0.0
        rr_n = 1 if rr_numeric is not None else 0
        if table == "symbol_stats":
            self._conn.execute(
                """
                INSERT INTO symbol_stats(symbol,tp,sl,rr_sum,rr_n)
                VALUES(?,?,?,?,?)
                ON CONFLICT(symbol) DO UPDATE SET
                    tp=tp+excluded.tp,
                    sl=sl+excluded.sl,
                    rr_sum=rr_sum+excluded.rr_sum,
                    rr_n=rr_n+excluded.rr_n;
                """,
                (symbol, tp, sl, rr_sum, rr_n),
            )
            return
        if table != "symbol_trigger_stats" or not trigger_kind:
            raise ValueError("invalid stats bucket")
        self._conn.execute(
            """
            INSERT INTO symbol_trigger_stats(
                symbol,trigger_kind,tp,sl,rr_sum,rr_n
            )
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(symbol,trigger_kind) DO UPDATE SET
                tp=tp+excluded.tp,
                sl=sl+excluded.sl,
                rr_sum=rr_sum+excluded.rr_sum,
                rr_n=rr_n+excluded.rr_n;
            """,
            (symbol, trigger_kind, tp, sl, rr_sum, rr_n),
        )

    def update_on_close(
        self,
        symbol: str,
        outcome: str,
        rr_numeric: Optional[float] = None,
        *,
        trigger_kind: Optional[str] = None,
    ) -> None:
        """
        outcome: "TP" | "SL"
        rr_numeric: можно передать, но не обязателен
        """
        outcome = (outcome or "").upper()
        if outcome not in ("TP", "SL"):
            return
        symbol_key = self._key(symbol)
        trigger_key = self._trigger_key(trigger_kind)
        if not symbol_key:
            return

        with self._lock, self._conn:
            self._upsert_stats(
                table="symbol_stats",
                symbol=symbol_key,
                outcome=outcome,
                rr_numeric=rr_numeric,
            )
            if trigger_key:
                self._upsert_stats(
                    table="symbol_trigger_stats",
                    symbol=symbol_key,
                    trigger_kind=trigger_key,
                    outcome=outcome,
                    rr_numeric=rr_numeric,
                )

    def _stats(
        self,
        row: Optional[sqlite3.Row],
        *,
        symbol: str,
        trigger_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        if row is None:
            tp = sl = 0
            rr_sum = 0.0
            rr_n = 0
        else:
            tp = int(row["tp"])
            sl = int(row["sl"])
            rr_sum = float(row["rr_sum"])
            rr_n = int(row["rr_n"])
        closed = tp + sl
        return {
            "symbol": symbol,
            "trigger_kind": trigger_kind,
            "tp": tp,
            "sl": sl,
            "closed": closed,
            "p_tp": self.estimate_p_tp(tp, sl),
            "rr_avg": rr_sum / rr_n if rr_n > 0 else 0.0,
        }

    def get_symbol_stats(self, symbol: str) -> Dict[str, Any]:
        symbol_key = self._key(symbol)
        with self._lock:
            row = self._conn.execute(
                "SELECT tp, sl, rr_sum, rr_n FROM symbol_stats WHERE symbol=?;",
                (symbol_key,),
            ).fetchone()
        return self._stats(row, symbol=symbol_key)

    def get_symbol_trigger_stats(
        self,
        symbol: str,
        trigger_kind: str,
    ) -> Dict[str, Any]:
        symbol_key = self._key(symbol)
        trigger_key = self._trigger_key(trigger_kind)
        if not trigger_key:
            return self._stats(
                None,
                symbol=symbol_key,
                trigger_kind=None,
            )
        with self._lock:
            row = self._conn.execute(
                """
                SELECT tp, sl, rr_sum, rr_n
                FROM symbol_trigger_stats
                WHERE symbol=? AND trigger_kind=?;
                """,
                (symbol_key, trigger_key),
            ).fetchone()
        return self._stats(
            row,
            symbol=symbol_key,
            trigger_kind=trigger_key,
        )

    def estimate_p_tp(self, tp: int, sl: int) -> float:
        # beta prior smoothing
        a = float(self.cfg.alpha)
        b = float(self.cfg.beta)
        denom = (tp + sl + a + b)
        return float((tp + a) / denom) if denom > 0 else 0.5
