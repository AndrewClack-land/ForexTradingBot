# core/state_store.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


def save_active_trades(active_trades: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, dict] = {}
    for symbol, tr in (active_trades or {}).items():
        if tr is None:
            continue
        payload[symbol] = {
            "side": getattr(tr, "side", None),
            "entry": float(getattr(tr, "entry", 0.0)),
            "stop": float(getattr(tr, "stop", 0.0)),
            "tp_prices": [float(x) for x in (getattr(tr, "tp_prices", []) or [])],
            "tf": str(getattr(tr, "tf", "")),
            "narrative": str(getattr(tr, "narrative", "")),
            "symbol": str(getattr(tr, "symbol", symbol)),

            "tp_hit": int(getattr(tr, "tp_hit", 0) or 0),
            "ts_open": float(getattr(tr, "ts_open", 0.0)),
            "last_price_ts": float(getattr(tr, "last_price_ts", 0.0)),

            # чтобы reply продолжал работать после рестарта:
            "telegram_chat_id": getattr(tr, "telegram_chat_id", None),
            "telegram_message_id": getattr(tr, "telegram_message_id", None),

            # MT5 position tracking (needed to resume partial closes after restart)
            "volume": float(getattr(tr, "volume", 0.0) or 0.0),
            "mt5_ticket": getattr(tr, "mt5_ticket", None),
            "mt5_position_id": getattr(tr, "mt5_position_id", None),

            # Partial-close state — persisted so restarts don't re-close already-closed slices
            "volume_per_tp": [float(v) for v in (getattr(tr, "volume_per_tp", []) or [])],
            "volume_remaining": float(getattr(tr, "volume_remaining", 0.0) or 0.0),

            # Split-mode legs — CRITICAL: without this, bot loses split mode on restart
            # and tries to manage SL/TP manually instead of letting the broker handle them.
            "split_position_ids": list(getattr(tr, "split_position_ids", []) or []),
        }

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_active_trades(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8") or "{}")

    # импорт внутри, чтобы не словить циклические импорты
    from core.strategy_narrative import ActiveTrade

    restored: Dict[str, Any] = {}
    for symbol, d in (data or {}).items():
        try:
            tr = ActiveTrade(
                side=d.get("side"),
                entry=float(d.get("entry", 0.0)),
                stop=float(d.get("stop", 0.0)),
                tp_prices=[float(x) for x in (d.get("tp_prices") or [])],
                tf=str(d.get("tf", "")),
                narrative=str(d.get("narrative", "")),
                symbol=str(d.get("symbol", symbol)),
            )
            tr.tp_hit = int(d.get("tp_hit", 0) or 0)
            tr.ts_open = float(d.get("ts_open", 0.0) or time.time())
            tr.last_price_ts = float(d.get("last_price_ts", 0.0) or tr.ts_open)
            tr.telegram_chat_id = d.get("telegram_chat_id")
            tr.telegram_message_id = d.get("telegram_message_id")
            tr.volume = float(d.get("volume") or 0.0)
            tr.mt5_ticket = d.get("mt5_ticket")
            tr.mt5_position_id = d.get("mt5_position_id")
            tr.volume_per_tp = [float(v) for v in (d.get("volume_per_tp") or [])]
            tr.volume_remaining = float(d.get("volume_remaining") or 0.0)
            tr.split_position_ids = [int(x) for x in (d.get("split_position_ids") or [])]
            restored[symbol] = tr
        except Exception:
            continue

    return restored
