from __future__ import annotations

import csv
import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps_safe(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        return "{}"


def _json_loads_safe(s: Any) -> Any:
    if s is None:
        return None
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(str(s))
    except Exception:
        return None


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class TradeRow:
    ts_open: str
    ts_close: Optional[str]
    symbol: str
    side: str
    entry: float
    stop: float
    tp: float
    exit: Optional[float]
    outcome: Optional[str]
    rr: Optional[float]
    rr_text: Optional[str]
    tf: Optional[str]
    narrative: Optional[str]
    trigger_reason: Optional[str]
    vc: Optional[str]
    meta_json: str
    features_json: str
    telegram_chat_id_open: Optional[int]
    telegram_message_id_open: Optional[int]
    telegram_chat_id_close: Optional[int]
    telegram_message_id_close: Optional[int]

    # NEW (persistence of active trade state)
    tp_prices_json: Optional[str] = None
    tp_hit: int = 0
    moved_to_be: int = 0
    stop_current: Optional[float] = None
    ts_update: Optional[str] = None

    # setup-level attribution and broker-authoritative result
    setup_id: Optional[str] = None
    strategy_version: Optional[str] = None
    experiment_id: Optional[str] = None
    experiment_variant: Optional[str] = None
    deployment_id: Optional[str] = None
    planned_entry: Optional[float] = None
    execution_mode: Optional[str] = None
    total_volume: Optional[float] = None
    weighted_rr_planned: Optional[float] = None
    exit_signal: Optional[str] = None
    realized_net: Optional[float] = None
    pnl_complete: bool = False
    close_meta_json: Optional[str] = None


class TradeJournal:
    """
    SQLite trade journal + CSV/Parquet export.

    Главное обновление:
      - сохраняем активные сделки (tp_prices, tp_hit, moved_to_be, актуальный stop)
      - умеем восстановить активные сделки после перезапуска
      - умеем дать отчет /report по сделкам, которые публиковались до рестарта
    """

    def __init__(
        self,
        db_path: str,
        csv_path: Optional[str] = None,
        parquet_path: Optional[str] = None,
        export_on_each_event: bool = True,
        strategy_version: Optional[str] = None,
        experiment_id: Optional[str] = None,
        experiment_variant: Optional[str] = None,
        deployment_id: Optional[str] = None,
    ):
        self.db_path = Path(db_path)
        self.csv_path = Path(csv_path) if csv_path else None
        self.parquet_path = Path(parquet_path) if parquet_path else None
        self.export_on_each_event = export_on_each_event
        # Explicit release tags are deliberately not derived from git: the VPS
        # can run a verified but uncommitted deployment. Old rows stay NULL;
        # only setups opened by this journal instance receive current tags.
        self.strategy_version = str(
            strategy_version or os.getenv("STRATEGY_VERSION", "unversioned")
        ).strip() or "unversioned"
        self.experiment_id = str(
            experiment_id or os.getenv("STRATEGY_EXPERIMENT_ID", "")
        ).strip()
        self.experiment_variant = str(
            experiment_variant or os.getenv("STRATEGY_EXPERIMENT_VARIANT", "")
        ).strip()
        self.deployment_id = str(
            deployment_id or os.getenv("DEPLOYMENT_ID", "")
        ).strip()

        _ensure_dir(self.db_path)
        if self.csv_path:
            _ensure_dir(self.csv_path)
        if self.parquet_path:
            _ensure_dir(self.parquet_path)

        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        try:
            self._conn.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            pass

        self._ensure_schema()
        self._migrate_columns()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
            finally:
                self._conn.close()

    # ----------------- schema / migrations -----------------
    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_open TEXT NOT NULL,
                    ts_close TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,              -- LONG/SHORT
                    entry REAL NOT NULL,
                    stop REAL NOT NULL,
                    tp REAL NOT NULL,
                    exit REAL,
                    outcome TEXT,                    -- TP/SL
                    rr REAL,                         -- numeric rr
                    rr_text TEXT,                    -- display rr "1:1.50"
                    tf TEXT,
                    narrative TEXT,
                    trigger_reason TEXT,
                    vc TEXT,
                    meta_json TEXT,
                    features_json TEXT,
                    telegram_chat_id_open INTEGER,
                    telegram_message_id_open INTEGER,
                    telegram_chat_id_close INTEGER,
                    telegram_message_id_close INTEGER,

                    -- NEW state fields
                    tp_prices_json TEXT,
                    tp_hit INTEGER DEFAULT 0,
                    moved_to_be INTEGER DEFAULT 0,
                    stop_current REAL,
                    ts_update TEXT,

                    -- One row is one executed setup (never one split leg).
                    setup_id TEXT,
                    strategy_version TEXT,
                    experiment_id TEXT,
                    experiment_variant TEXT,
                    deployment_id TEXT,
                    planned_entry REAL,
                    execution_mode TEXT,
                    total_volume REAL,
                    weighted_rr_planned REAL,
                    exit_signal TEXT,
                    realized_net REAL,
                    pnl_complete INTEGER DEFAULT 0,
                    close_meta_json TEXT
                );
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_symbol_open ON trades(symbol, ts_open);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_symbol_close ON trades(symbol, ts_close);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_open ON trades(symbol, ts_close);"
            )
            self._conn.commit()

    def _column_exists(self, table: str, col: str) -> bool:
        cur = self._conn.execute(f"PRAGMA table_info({table});")
        return any(row["name"] == col for row in cur.fetchall())

    def _migrate_columns(self) -> None:
        """
        Добавляем новые колонки, если база старая.
        """
        with self._lock:
            needed = {
                "tf": "TEXT",
                "narrative": "TEXT",
                "trigger_reason": "TEXT",
                "vc": "TEXT",
                "meta_json": "TEXT",
                "features_json": "TEXT",
                "telegram_chat_id_open": "INTEGER",
                "telegram_message_id_open": "INTEGER",
                "telegram_chat_id_close": "INTEGER",
                "telegram_message_id_close": "INTEGER",
                "rr_text": "TEXT",

                "tp_prices_json": "TEXT",
                "tp_hit": "INTEGER DEFAULT 0",
                "moved_to_be": "INTEGER DEFAULT 0",
                "stop_current": "REAL",
                "ts_update": "TEXT",

                "setup_id": "TEXT",
                "strategy_version": "TEXT",
                "experiment_id": "TEXT",
                "experiment_variant": "TEXT",
                "deployment_id": "TEXT",
                "planned_entry": "REAL",
                "execution_mode": "TEXT",
                "total_volume": "REAL",
                "weighted_rr_planned": "REAL",
                "exit_signal": "TEXT",
                "realized_net": "REAL",
                "pnl_complete": "INTEGER DEFAULT 0",
                "close_meta_json": "TEXT",
            }
            for col, ddl in needed.items():
                if not self._column_exists("trades", col):
                    self._conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {ddl};")
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_setup_id ON trades(setup_id);"
            )
            self._conn.commit()

    # ----------------- helpers -----------------
    def _find_last_open_trade_id(self, symbol: str) -> Optional[int]:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT id
                FROM trades
                WHERE symbol = ?
                  AND ts_close IS NULL
                ORDER BY id DESC
                LIMIT 1;
                """,
                (symbol,),
            )
            row = cur.fetchone()
            return int(row["id"]) if row else None

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        # NaN/inf must not poison aggregate metrics or exported datasets.
        if number != number or number in (float("inf"), float("-inf")):
            return None
        return number

    @staticmethod
    def _tp_prices_from_sig(sig: Dict[str, Any]) -> List[float]:
        tps = sig.get("tp_prices")
        if isinstance(tps, list) and tps:
            out = []
            for x in tps:
                try:
                    out.append(float(x))
                except Exception:
                    pass
            if out:
                return out

        if sig.get("tp_price") is not None:
            try:
                return [float(sig.get("tp_price"))]
            except Exception:
                pass
        return []

    # ----------------- core API -----------------
    def ingest_signal(
        self,
        symbol: str,
        sig: Dict[str, Any],
        telegram_chat_id: Optional[int] = None,
        telegram_message_id: Optional[int] = None,
        features: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        ENTER: открываем сделку и сохраняем tp_prices + текущий стоп (stop_current)
        EXIT_TP/EXIT_SL: закрываем сделку
        """
        if not isinstance(sig, dict):
            return

        st = sig.get("signal")
        if st == "ENTER":
            rr_num = sig.get("rr_numeric")
            rr_text = sig.get("rr")
            tp_prices = self._tp_prices_from_sig(sig)

            self._open_trade(
                symbol=symbol,
                side=str(sig.get("side", "")).upper(),
                entry=float(sig.get("entry_price")),
                stop=float(sig.get("stop_price")),
                tp=float(sig.get("tp_price", tp_prices[-1] if tp_prices else sig.get("stop_price"))),
                rr=float(rr_num) if rr_num is not None else None,
                rr_text=str(rr_text) if rr_text is not None else None,
                tf=str(sig.get("tf")) if sig.get("tf") is not None else None,
                narrative=str(sig.get("narrative")) if sig.get("narrative") is not None else None,
                trigger_reason=str(sig.get("trigger_reason")) if sig.get("trigger_reason") is not None else None,
                vc=str(sig.get("vc")) if sig.get("vc") is not None else None,
                meta=sig,
                features=features,
                tp_prices=tp_prices,
                tp_hit=0,
                moved_to_be=0,
                stop_current=float(sig.get("stop_price")),
                telegram_chat_id_open=telegram_chat_id,
                telegram_message_id_open=telegram_message_id,
            )
            if self.export_on_each_event:
                self.export_all()

        elif st in ("EXIT_TP", "EXIT_SL", "EXIT_BROKER", "EXIT_TIME"):
            # EXIT_BROKER carries the TP/SL outcome inferred from MT5 deal history
            # (split mode closes every trade this way); EXIT_TIME is a forced
            # time-based close. Both MUST close the DB row — a dangling open row
            # blocks journaling of every following trade on the symbol (duplicate
            # guard in _open_trade).
            if st == "EXIT_TP":
                outcome = "TP"
            elif st == "EXIT_SL":
                outcome = "SL"
            elif st == "EXIT_TIME":
                outcome = str(sig.get("outcome") or "TIME")
            else:
                outcome = str(sig.get("outcome") or "BROKER")
            exit_price = sig.get("exit_price")
            if exit_price is None:
                return

            self._close_trade(
                symbol=symbol,
                exit=float(exit_price),
                outcome=outcome,
                exit_signal=str(st),
                close_meta=sig,
                telegram_chat_id_close=telegram_chat_id,
                telegram_message_id_close=telegram_message_id,
            )
            if self.export_on_each_event:
                self.export_all()

    # ----------------- trade operations -----------------
    def _open_trade(
        self,
        symbol: str,
        side: str,
        entry: float,
        stop: float,
        tp: float,
        rr: Optional[float],
        rr_text: Optional[str],
        tf: Optional[str],
        narrative: Optional[str],
        trigger_reason: Optional[str],
        vc: Optional[str],
        meta: Optional[Dict[str, Any]],
        features: Optional[Dict[str, Any]],
        tp_prices: Optional[List[float]],
        tp_hit: int,
        moved_to_be: int,
        stop_current: Optional[float],
        telegram_chat_id_open: Optional[int],
        telegram_message_id_open: Optional[int],
    ) -> None:
        ts_open = _utc_now_iso()
        with self._lock:
            # защита от дубля
            meta = dict(meta or {})
            requested_setup_id = str(meta.get("setup_id") or "").strip()
            if requested_setup_id:
                duplicate = self._conn.execute(
                    "SELECT id FROM trades WHERE setup_id = ? LIMIT 1;",
                    (requested_setup_id,),
                ).fetchone()
                if duplicate is not None:
                    return
            else:
                # Legacy callers have no stable identity; retain their original
                # symbol-level duplicate guard. Current Core always supplies id.
                existing_id = self._find_last_open_trade_id(symbol)
                if existing_id is not None:
                    return

            execution = meta.get("execution")
            execution = execution if isinstance(execution, dict) else {}
            legs = execution.get("legs")
            legs = legs if isinstance(legs, list) else []
            total_volume = self._optional_float(execution.get("volume"))
            if total_volume is None and legs:
                leg_volumes = [
                    self._optional_float(leg.get("volume"))
                    for leg in legs
                    if isinstance(leg, dict)
                ]
                known_volumes = [value for value in leg_volumes if value is not None]
                total_volume = sum(known_volumes) if known_volumes else None
            execution_mode = str(execution.get("mode") or "").strip()
            if not execution_mode and execution:
                execution_mode = "split" if legs else "monitor"

            setup_id = requested_setup_id or uuid.uuid4().hex
            strategy_version = str(
                meta.get("strategy_version") or self.strategy_version
            ).strip() or "unversioned"
            experiment_id = str(
                meta.get("experiment_id") or self.experiment_id
            ).strip()
            experiment_variant = str(
                meta.get("experiment_variant") or self.experiment_variant
            ).strip()
            deployment_id = str(
                meta.get("deployment_id") or self.deployment_id
            ).strip()
            planned_entry = self._optional_float(
                meta.get("planned_entry_price", meta.get("entry_price"))
            )
            weighted_rr_planned = self._optional_float(
                meta.get("weighted_rr_numeric", meta.get("weighted_rr_planned"))
            )

            self._conn.execute(
                """
                INSERT INTO trades (
                    ts_open, ts_close, symbol, side,
                    entry, stop, tp, exit, outcome, rr, rr_text,
                    tf, narrative, trigger_reason, vc,
                    meta_json, features_json,
                    telegram_chat_id_open, telegram_message_id_open,
                    tp_prices_json, tp_hit, moved_to_be, stop_current, ts_update,
                    setup_id, strategy_version, experiment_id, experiment_variant,
                    deployment_id, planned_entry, execution_mode, total_volume,
                    weighted_rr_planned
                ) VALUES (
                    ?, NULL, ?, ?,
                    ?, ?, ?, NULL, NULL, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?
                );
                """,
                (
                    ts_open, symbol, side,
                    entry, stop, tp, rr, rr_text,
                    tf, narrative, trigger_reason, vc,
                    _json_dumps_safe(meta or {}),
                    _json_dumps_safe(features or {}),
                    telegram_chat_id_open, telegram_message_id_open,
                    _json_dumps_safe(tp_prices or []),
                    int(tp_hit or 0),
                    int(moved_to_be or 0),
                    float(stop_current) if stop_current is not None else float(stop),
                    _utc_now_iso(),
                    setup_id, strategy_version, experiment_id, experiment_variant,
                    deployment_id, planned_entry, execution_mode or None,
                    total_volume, weighted_rr_planned,
                ),
            )
            self._conn.commit()

    def _close_trade(
        self,
        symbol: str,
        exit: float,
        outcome: str,
        exit_signal: Optional[str],
        close_meta: Optional[Dict[str, Any]],
        telegram_chat_id_close: Optional[int],
        telegram_message_id_close: Optional[int],
    ) -> None:
        ts_close = _utc_now_iso()
        with self._lock:
            trade_id = self._find_last_open_trade_id(symbol)
            if trade_id is None:
                return

            close_meta = dict(close_meta or {})
            realized_net = self._optional_float(close_meta.get("realized_net"))
            # Only explicitly complete broker history is authoritative for W/L,
            # PF, and expectancy. A partial split history can otherwise turn a
            # losing setup into an apparent winner (or vice versa).
            pnl_complete = bool(close_meta.get("pnl_complete")) and realized_net is not None

            self._conn.execute(
                """
                UPDATE trades
                SET ts_close = ?,
                    exit = ?,
                    outcome = ?,
                    exit_signal = ?,
                    realized_net = ?,
                    pnl_complete = ?,
                    close_meta_json = ?,
                    telegram_chat_id_close = ?,
                    telegram_message_id_close = ?,
                    ts_update = ?
                WHERE id = ?;
                """,
                (
                    ts_close, exit, outcome,
                    exit_signal, realized_net, 1 if pnl_complete else 0,
                    _json_dumps_safe(close_meta),
                    telegram_chat_id_close, telegram_message_id_close,
                    _utc_now_iso(),
                    trade_id,
                ),
            )
            self._conn.commit()

    # ----------------- NEW: update active trade state -----------------
    def update_open_trade_state(
        self,
        symbol: str,
        *,
        stop_current: Optional[float] = None,
        tp_hit: Optional[int] = None,
        moved_to_be: Optional[bool] = None,
        tp_prices: Optional[List[float]] = None,
        telegram_chat_id_open: Optional[int] = None,
        telegram_message_id_open: Optional[int] = None,
    ) -> None:
        with self._lock:
            trade_id = self._find_last_open_trade_id(symbol)
            if trade_id is None:
                return

            cols = []
            vals = []

            if stop_current is not None:
                cols.append("stop_current = ?")
                vals.append(float(stop_current))

            if tp_hit is not None:
                cols.append("tp_hit = ?")
                vals.append(int(tp_hit))

            if moved_to_be is not None:
                cols.append("moved_to_be = ?")
                vals.append(1 if bool(moved_to_be) else 0)

            if tp_prices is not None:
                cols.append("tp_prices_json = ?")
                vals.append(_json_dumps_safe([float(x) for x in tp_prices]))

            if telegram_chat_id_open is not None:
                cols.append("telegram_chat_id_open = ?")
                vals.append(int(telegram_chat_id_open))

            if telegram_message_id_open is not None:
                cols.append("telegram_message_id_open = ?")
                vals.append(int(telegram_message_id_open))

            if not cols:
                return

            cols.append("ts_update = ?")
            vals.append(_utc_now_iso())
            vals.append(int(trade_id))

            sql = f"UPDATE trades SET {', '.join(cols)} WHERE id = ?;"
            self._conn.execute(sql, tuple(vals))
            self._conn.commit()

            if self.export_on_each_event:
                self.export_all()

    # ----------------- NEW: restore / reports -----------------
    def load_open_trades(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    id, ts_open, symbol, side, entry, stop, tp, tf, narrative,
                    telegram_chat_id_open, telegram_message_id_open,
                    tp_prices_json, tp_hit, moved_to_be, stop_current,
                    setup_id, strategy_version, experiment_id,
                    experiment_variant, deployment_id
                FROM trades
                WHERE ts_close IS NULL
                ORDER BY id ASC;
                """
            ).fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            tp_prices = _json_loads_safe(r["tp_prices_json"])
            tp_prices = [float(x) for x in (tp_prices or []) if x is not None]
            if not tp_prices and r["tp"] is not None:
                tp_prices = [float(r["tp"])]

            stop_current = r["stop_current"]
            if stop_current is None:
                stop_current = r["stop"]

            out.append({
                "id": int(r["id"]),
                "ts_open": str(r["ts_open"]),
                "symbol": str(r["symbol"]),
                "side": str(r["side"]),
                "entry": float(r["entry"]),
                "stop_current": float(stop_current),
                "tp_prices": tp_prices,
                "tp_hit": int(r["tp_hit"] or 0),
                "moved_to_be": bool(int(r["moved_to_be"] or 0)),
                "tf": str(r["tf"] or ""),
                "narrative": str(r["narrative"] or ""),
                "telegram_chat_id": int(r["telegram_chat_id_open"]) if r["telegram_chat_id_open"] is not None else None,
                "telegram_message_id": int(r["telegram_message_id_open"]) if r["telegram_message_id_open"] is not None else None,
                "setup_id": str(r["setup_id"]) if r["setup_id"] is not None else None,
                "strategy_version": str(r["strategy_version"]) if r["strategy_version"] is not None else None,
                "experiment_id": str(r["experiment_id"] or ""),
                "experiment_variant": str(r["experiment_variant"] or ""),
                "deployment_id": str(r["deployment_id"] or ""),
            })
        return out

    def recent_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    id, ts_open, ts_close, symbol, side, entry, stop, tp, stop_current,
                    tp_prices_json, tp_hit, moved_to_be,
                    exit, outcome, rr_text, tf, telegram_message_id_open,
                    setup_id, strategy_version, experiment_id,
                    experiment_variant, deployment_id, planned_entry,
                    execution_mode, total_volume, weighted_rr_planned,
                    exit_signal, realized_net, pnl_complete
                FROM trades
                ORDER BY id DESC
                LIMIT ?;
                """,
                (limit,),
            ).fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            tp_prices = _json_loads_safe(r["tp_prices_json"])
            if not isinstance(tp_prices, list) or not tp_prices:
                tp_prices = [r["tp"]] if r["tp"] is not None else []
            tp_prices = [float(x) for x in tp_prices if x is not None]
            out.append({
                "id": int(r["id"]),
                "ts_open": str(r["ts_open"]),
                "ts_close": str(r["ts_close"]) if r["ts_close"] is not None else None,
                "symbol": str(r["symbol"]),
                "side": str(r["side"]),
                "entry": float(r["entry"]),
                "stop": float(r["stop"]),
                "stop_current": float(r["stop_current"]) if r["stop_current"] is not None else None,
                "tp_prices": tp_prices,
                "tp_hit": int(r["tp_hit"] or 0),
                "moved_to_be": bool(int(r["moved_to_be"] or 0)),
                "exit": float(r["exit"]) if r["exit"] is not None else None,
                "outcome": str(r["outcome"]) if r["outcome"] is not None else None,
                "rr_text": str(r["rr_text"] or ""),
                "tf": str(r["tf"] or ""),
                "telegram_message_id_open": int(r["telegram_message_id_open"]) if r["telegram_message_id_open"] is not None else None,
                "setup_id": str(r["setup_id"]) if r["setup_id"] is not None else None,
                "strategy_version": str(r["strategy_version"]) if r["strategy_version"] is not None else None,
                "experiment_id": str(r["experiment_id"] or ""),
                "experiment_variant": str(r["experiment_variant"] or ""),
                "deployment_id": str(r["deployment_id"] or ""),
                "planned_entry": float(r["planned_entry"]) if r["planned_entry"] is not None else None,
                "execution_mode": str(r["execution_mode"] or ""),
                "total_volume": float(r["total_volume"]) if r["total_volume"] is not None else None,
                "weighted_rr_planned": float(r["weighted_rr_planned"]) if r["weighted_rr_planned"] is not None else None,
                "exit_signal": str(r["exit_signal"] or ""),
                "realized_net": float(r["realized_net"]) if r["realized_net"] is not None else None,
                "pnl_complete": bool(int(r["pnl_complete"] or 0)),
                "result": self._classify_setup_row(r),
            })
        return out

    # ----------------- exports -----------------
    def export_csv(self) -> None:
        if not self.csv_path:
            return

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    ts_open, ts_close, symbol, side, entry, stop, tp, stop_current,
                    tp_prices_json, tp_hit, moved_to_be,
                    exit, outcome, rr, rr_text, tf, narrative, trigger_reason, vc,
                    meta_json, features_json,
                    telegram_chat_id_open, telegram_message_id_open,
                    telegram_chat_id_close, telegram_message_id_close,
                    ts_update,
                    setup_id, strategy_version, experiment_id, experiment_variant,
                    deployment_id, planned_entry, execution_mode, total_volume,
                    weighted_rr_planned, exit_signal, realized_net, pnl_complete,
                    close_meta_json
                FROM trades
                ORDER BY ts_open ASC;
                """
            ).fetchall()

        tmp = self.csv_path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "ts_open", "ts_close", "symbol", "side", "entry", "stop", "tp", "stop_current",
                "tp_prices_json", "tp_hit", "moved_to_be",
                "exit", "outcome", "rr", "rr_text", "tf", "narrative", "trigger_reason", "vc",
                "meta_json", "features_json",
                "telegram_chat_id_open", "telegram_message_id_open",
                "telegram_chat_id_close", "telegram_message_id_close",
                "ts_update",
                "setup_id", "strategy_version", "experiment_id", "experiment_variant",
                "deployment_id", "planned_entry", "execution_mode", "total_volume",
                "weighted_rr_planned", "exit_signal", "realized_net", "pnl_complete",
                "close_meta_json",
            ])
            for r in rows:
                writer.writerow([r[k] for k in r.keys()])

        os.replace(tmp, self.csv_path)

    def export_parquet(self) -> None:
        if not self.parquet_path:
            return

        try:
            import pandas as pd  # noqa
        except Exception:
            return

        engine_ok = False
        try:
            import pyarrow  # noqa
            engine_ok = True
        except Exception:
            pass
        if not engine_ok:
            try:
                import fastparquet  # noqa
                engine_ok = True
            except Exception:
                pass
        if not engine_ok:
            return

        with self._lock:
            df = pd.read_sql_query(
                """
                SELECT
                    ts_open, ts_close, symbol, side, entry, stop, tp, stop_current,
                    tp_prices_json, tp_hit, moved_to_be,
                    exit, outcome, rr, rr_text, tf, narrative, trigger_reason, vc,
                    meta_json, features_json,
                    telegram_chat_id_open, telegram_message_id_open,
                    telegram_chat_id_close, telegram_message_id_close,
                    ts_update,
                    setup_id, strategy_version, experiment_id, experiment_variant,
                    deployment_id, planned_entry, execution_mode, total_volume,
                    weighted_rr_planned, exit_signal, realized_net, pnl_complete,
                    close_meta_json
                FROM trades
                ORDER BY ts_open ASC;
                """,
                self._conn,
            )

        tmp = self.parquet_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, self.parquet_path)

    def export_all(self) -> None:
        self.export_csv()
        self.export_parquet()

    @staticmethod
    def _classify_setup_row(row: Any) -> str:
        """Classify one closed setup, never an individual TP leg.

        Complete broker P&L is authoritative. Legacy rows fall back to their
        terminal outcome, while TIME/BROKER without P&L remain UNKNOWN instead
        of silently improving or depressing the win rate.
        """
        try:
            keys = set(row.keys())
        except Exception:
            keys = set()

        is_closed = (row["ts_close"] if "ts_close" in keys else None) is not None
        if not is_closed:
            return "OPEN"

        pnl_complete = bool(int(row["pnl_complete"] or 0)) if "pnl_complete" in keys else False
        realized_net = row["realized_net"] if "realized_net" in keys else None
        if pnl_complete and realized_net is not None:
            value = float(realized_net)
            if value > 0.005:
                return "WIN"
            if value < -0.005:
                return "LOSS"
            return "BE"

        outcome = str(row["outcome"] or "").strip().upper() if "outcome" in keys else ""
        if outcome in {"TP", "WIN", "PROFIT", "MANUAL_TP", "PARTIAL_TP"}:
            return "WIN"
        if outcome in {"SL", "LOSS", "MANUAL_SL", "PARTIAL_SL"}:
            return "LOSS"
        if outcome in {"BE", "BREAKEVEN", "BREAK_EVEN", "FLAT"}:
            return "BE"
        return "UNKNOWN"

    @classmethod
    def _aggregate_setup_rows(cls, rows: List[Any]) -> Dict[str, Any]:
        total = len(rows)
        closed_rows = [row for row in rows if row["ts_close"] is not None]
        results = [cls._classify_setup_row(row) for row in closed_rows]
        wins = results.count("WIN")
        losses = results.count("LOSS")
        breakeven = results.count("BE")
        unknown = results.count("UNKNOWN")
        evaluated = wins + losses

        pnl_rows = [
            row for row in closed_rows
            if bool(int(row["pnl_complete"] or 0)) and row["realized_net"] is not None
        ]
        pnl_values = [float(row["realized_net"]) for row in pnl_rows]
        gross_profit = sum(value for value in pnl_values if value > 0.0)
        gross_loss = abs(sum(value for value in pnl_values if value < 0.0))
        if gross_loss > 1e-12:
            profit_factor: Optional[float] = gross_profit / gross_loss
        elif gross_profit > 1e-12:
            profit_factor = float("inf")
        else:
            profit_factor = None

        closed = len(closed_rows)
        tp1_hits = sum(1 for row in closed_rows if int(row["tp_hit"] or 0) >= 1)
        be_moves = sum(1 for row in closed_rows if bool(int(row["moved_to_be"] or 0)))
        return {
            "total_setups": total,
            "open_setups": total - closed,
            "closed_setups": closed,
            "evaluated_setups": evaluated,
            "wins": wins,
            "losses": losses,
            "breakeven": breakeven,
            "unknown": unknown,
            "win_rate": (wins / evaluated) if evaluated else 0.0,
            "all_closed_win_rate": (wins / closed) if closed else 0.0,
            "classification_coverage": ((evaluated + breakeven) / closed) if closed else 0.0,
            "net_known_setups": len(pnl_rows),
            "pnl_coverage": (len(pnl_rows) / closed) if closed else 0.0,
            "net_realized": sum(pnl_values),
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "expectancy_net": (sum(pnl_values) / len(pnl_values)) if pnl_values else None,
            "tp1_hits": tp1_hits,
            "tp1_rate": (tp1_hits / closed) if closed else 0.0,
            "be_moves": be_moves,
            "be_move_rate": (be_moves / closed) if closed else 0.0,
        }

    def setup_metrics(
        self,
        limit: Optional[int] = None,
        *,
        strategy_version: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return setup-level metrics with explicit P&L coverage.

        ``limit`` selects the newest N setups before aggregation. Optional
        release filters make before/after comparisons reproducible.
        """
        conditions: List[str] = []
        params: List[Any] = []
        if strategy_version is not None:
            conditions.append("strategy_version = ?")
            params.append(str(strategy_version))
        if experiment_id is not None:
            conditions.append("experiment_id = ?")
            params.append(str(experiment_id))

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_sql = ""
        if limit is not None:
            safe_limit = max(1, min(int(limit), 10000))
            limit_sql = "LIMIT ?"
            params.append(safe_limit)

        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT
                    id, ts_close, symbol, side, outcome, tp_hit, moved_to_be,
                    strategy_version, experiment_id, experiment_variant,
                    realized_net, pnl_complete
                FROM trades
                {where}
                ORDER BY id DESC
                {limit_sql};
                """,
                tuple(params),
            ).fetchall()

        metrics = self._aggregate_setup_rows(list(rows))

        symbol_side: Dict[Tuple[str, str], List[Any]] = {}
        strategy_groups: Dict[Tuple[str, str, str], List[Any]] = {}
        for row in rows:
            symbol_side.setdefault((str(row["symbol"]), str(row["side"])), []).append(row)
            strategy_key = (
                str(row["strategy_version"] or "legacy"),
                str(row["experiment_id"] or ""),
                str(row["experiment_variant"] or ""),
            )
            strategy_groups.setdefault(strategy_key, []).append(row)

        metrics["by_symbol_side"] = [
            {
                "symbol": key[0],
                "side": key[1],
                **self._aggregate_setup_rows(group_rows),
            }
            for key, group_rows in sorted(symbol_side.items())
        ]
        metrics["by_strategy"] = [
            {
                "strategy_version": key[0],
                "experiment_id": key[1],
                "experiment_variant": key[2],
                **self._aggregate_setup_rows(group_rows),
            }
            for key, group_rows in sorted(strategy_groups.items())
        ]
        return metrics

    def winrate(self) -> Tuple[float, int, int, int]:
        metrics = self.setup_metrics()
        wins = int(metrics["wins"])
        losses = int(metrics["losses"])
        evaluated = int(metrics["evaluated_setups"])
        return float(metrics["win_rate"]), evaluated, wins, losses
