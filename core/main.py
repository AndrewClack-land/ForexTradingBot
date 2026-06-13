# main.py
from __future__ import annotations

import time
import traceback
from datetime import datetime, time as dt_time
from typing import Dict, Any, List
from zoneinfo import ZoneInfo

from config import (
    TELEGRAM_TOKEN,
    UNIVERSE,
    CONTEXT_SYMBOLS,
    AI_DATA_DIR,
    LOG_TICK,
    SESSION_WINDOWS,
    ALLOWED_SESSIONS,
    SESSION_TIMEZONE,
    MT5_CACHE_DIR,
    MT5_EXECUTION_ENABLED,
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_SERVER,
    MT5_MAGIC,
    MT5_RISK_PER_TRADE,
    MT5_SLIPPAGE,
)
from core.data_cache import DataCache

from core.data_feed import DataFeed
from core.market_scanner import MarketScanner
from core.strategy_narrative import NarrativeStrategy, ActiveTrade
from bot.telegram_bot import TelegramBot

from core.m1.config import AIConfig
from core.m1.store import TradeStore
from core.m1.ai_live import AILive

from core.state_store import save_active_trades, load_active_trades
from core.profiler import TickProfiler
from core.risk_rules import RiskRules
from executors.mt5_executor import MT5Executor, MT5Settings


def _compute_tp_volumes(total_volume: float, n_tps: int) -> List[float]:
    """
    Split total_volume into per-TP partial-close amounts.

    Allocation rules (user-defined):
      1 TP  → [100%]
      2 TPs → [50%, remainder]
      3 TPs → [50%, 25%, remainder]
      4 TPs → [25%, 30%, 30%, remainder]
      >4 TPs → first three get 25%/30%/30%, last entry is always remainder

    The *last* element always captures any rounding residual so that
    sum(result) == total_volume exactly.
    """
    if n_tps <= 0 or total_volume <= 0:
        return []
    if n_tps == 1:
        return [total_volume]

    if n_tps == 2:
        pre_ratios: List[float] = [0.50]
    elif n_tps == 3:
        pre_ratios = [0.50, 0.25]
    else:  # 4+
        pre_ratios = [0.25, 0.30, 0.30]
        # extra TPs beyond 4 receive nothing here; remainder goes to last
        pre_ratios += [0.0] * (n_tps - 4)

    vols: List[float] = []
    allocated = 0.0
    for ratio in pre_ratios:
        vol = round(total_volume * ratio, 2)
        vols.append(vol)
        allocated = round(allocated + vol, 10)

    # Last TP = whatever is left (ensures sum == total_volume)
    vols.append(round(total_volume - allocated, 2))
    return vols


class Core:
    def __init__(self):
        self.universe = dict(UNIVERSE)
        self.context_symbols = dict(CONTEXT_SYMBOLS)
        feed_universe = dict(self.universe)
        feed_universe.update(self.context_symbols)
        self.feed = DataFeed(universe=feed_universe, mt5_cache_dir=MT5_CACHE_DIR)
        self.data_cache = DataCache(self.feed)
        self.data_cache.start()
        self.strategy = NarrativeStrategy()
        self.scanner = MarketScanner(self.universe)

        self.session_tz = ZoneInfo(SESSION_TIMEZONE)
        self.allowed_session_windows = []
        for name in (ALLOWED_SESSIONS or []):
            window = SESSION_WINDOWS.get(name.upper())
            if not window:
                continue
            self.allowed_session_windows.append(
                (
                    name.upper(),
                    self._parse_time_str(window[0]),
                    self._parse_time_str(window[1]),
                )
            )

        # ✅ restore open trades after restart
        self.active_trades: dict[str, ActiveTrade] = load_active_trades(AI_DATA_DIR / "active_trades.json")
        print(f"[Core] restored active_trades={len(self.active_trades)}")

        # AI layer
        self.ai_cfg = AIConfig()
        self.ai_store = TradeStore(self.ai_cfg)
        self.ai = AILive(self.ai_cfg, self.ai_store, self.strategy)

        self.TIME_BUDGET_SEC = 35.0
        self.N_BARS = 300

        self.profiler = TickProfiler()
        self.global_context: Dict[str, Any] = {"session": "ALL", "session_allowed": True}
        self.log_tick = LOG_TICK
        self.risk_rules = RiskRules()

        # Grace period after startup: block all MT5 closes for 90s to let state sync
        self._startup_time: float = time.time()
        self._startup_grace_sec: float = 90.0

        # cooldown per symbol after a failed entry attempt (prevents infinite retries)
        self._entry_cooldowns: Dict[str, float] = {}
        self._entry_cooldown_sec: float = 300.0   # 5 min — generic execution error
        self._stale_cooldown_sec: float = 1800.0  # 30 min — price already at/past SL

        self.mt5_executor: MT5Executor | None = None
        if MT5_EXECUTION_ENABLED:
            try:
                if MT5_LOGIN is None or not MT5_PASSWORD:
                    raise RuntimeError("MT5 credentials are missing")
                settings = MT5Settings(
                    login=MT5_LOGIN,
                    password=MT5_PASSWORD,
                    server=MT5_SERVER,
                    risk_pct=MT5_RISK_PER_TRADE,
                    magic=MT5_MAGIC,
                    slippage=MT5_SLIPPAGE,
                )
                self.mt5_executor = MT5Executor(settings)
                print(f"[MT5] Execution enabled (risk={MT5_RISK_PER_TRADE:.2%})")
            except Exception as exc:
                print(f"[MT5] Executor disabled: {exc}")

        # Reconcile active_trades with real MT5 positions on startup (handles PC shutdown/restart)
        if self.mt5_executor and self.active_trades:
            self._reconcile_with_mt5()

    def _reconcile_with_mt5(self) -> None:
        """On startup: remove active_trades whose MT5 positions no longer exist."""
        to_remove = []
        for sym, trade in self.active_trades.items():
            pos_id = getattr(trade, "mt5_position_id", None) or getattr(trade, "mt5_ticket", None)
            if pos_id is None:
                # If MT5 execution is enabled and the trade has no IDs + zero volume,
                # the entry was never actually executed — treat it as a ghost and remove it.
                if self.mt5_executor and not (getattr(trade, "volume", 0.0) or 0.0):
                    print(f"[Core] {sym} has no MT5 position ID and volume=0 — removing ghost trade")
                    to_remove.append(sym)
                continue  # paper trade (or ghost handled above) — keep it
            try:
                pos = self.mt5_executor.get_position(sym, pos_id)
                if pos is None:
                    print(f"[Core] MT5 position {pos_id} for {sym} not found — removing from active_trades")
                    to_remove.append(sym)
                else:
                    # Sync actual volume from MT5 in case it was partially closed
                    actual_vol = float(getattr(pos, "volume", trade.volume or 0.0))
                    if actual_vol != (trade.volume or 0.0):
                        print(f"[Core] Volume mismatch for {sym}: bot={trade.volume} MT5={actual_vol} — syncing")
                        trade.volume = actual_vol
            except Exception as exc:
                print(f"[Core] Reconcile error for {sym}: {exc}")
        if to_remove:
            for sym in to_remove:
                del self.active_trades[sym]
            save_active_trades(self.active_trades, AI_DATA_DIR / "active_trades.json")
            print(f"[Core] Reconcile: removed {to_remove}")

    @staticmethod
    def _parse_time_str(value: str) -> dt_time:
        hour, minute = [int(x) for x in value.split(":", 1)]
        return dt_time(hour=hour, minute=minute)

    @staticmethod
    def _time_in_window(now: dt_time, start: dt_time, end: dt_time) -> bool:
        if start <= end:
            return start <= now < end
        return now >= start or now < end

    def _session_allowance(self) -> tuple[bool, str]:
        if not self.allowed_session_windows:
            return True, "ALL"
        now = datetime.now(self.session_tz).time()
        for name, start, end in self.allowed_session_windows:
            if self._time_in_window(now, start, end):
                return True, name
        return False, "OFF"

    @staticmethod
    def _is_friday_weekend_close() -> bool:
        """True on Friday at or after 22:00 UTC+3 (Europe/Moscow, no DST)."""
        now = datetime.now(ZoneInfo("Europe/Moscow"))
        return now.weekday() == 4 and now.hour >= 22

    def _get_symbols(self) -> list[str]:
        return self.scanner.scan()

    def _build_tf_data(self, symbol_key: str) -> dict:
        # ✅ include 1M for faster execution / management
        def _get(tf: str):
            df = self.data_cache.request(symbol_key, tf, limit=self.N_BARS)
            if df is None or df.empty:
                return self.feed.get_klines(symbol_key, tf, limit=self.N_BARS)
            return df

        return {
            "D": None,
            "4H": _get("4h"),
            "1H": _get("1h"),
            "15M": _get("15m"),
            "5M": _get("5m"),
            "1M": _get("1m"),
        }

    def _update_global_context(self) -> None:
        allowed, session_name = self._session_allowance()
        self.global_context["session"] = session_name
        self.global_context["session_allowed"] = allowed
        self.global_context["friday_close"] = self._is_friday_weekend_close()

    def _apply_session_filter(self, symbol: str, sig: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(sig, dict):
            return sig
        if sig.get("signal") != "ENTER":
            return sig
        if self.global_context.get("session_allowed", True):
            return sig
        new_sig = dict(sig)
        new_sig["signal"] = "WAIT_SESSION"
        reason = f"Blocked by session ({self.global_context.get('session')})"
        narrative = str(new_sig.get("narrative", ""))
        if reason not in narrative:
            new_sig["narrative"] = (narrative + " | " + reason).strip(" |")
        new_sig["session_blocked"] = self.global_context.get("session")
        return new_sig

    def _apply_global_filters(self, symbol: str, sig: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(sig, dict) and sig.get("signal") == "ENTER":
            if self.global_context.get("friday_close"):
                new_sig = dict(sig)
                new_sig["signal"] = "WAIT_SESSION"
                new_sig["info"] = "Заблокировано: закрытие перед выходными (пятница 22:00 UTC+3)"
                return new_sig
        return sig

    def _log_signal(self, symbol: str, sig: Dict[str, Any]) -> None:
        if not isinstance(sig, dict):
            return
        important = sig.get("signal") in {"ENTER", "EXIT_SL", "EXIT_TP"}
        if not (self.log_tick or important):
            return
        m15_raw = sig.get("m15_trend_raw")
        m15_part = f" | m15_ema={m15_raw}" if m15_raw else ""
        vc = sig.get("vc")
        vc_part = f" | block=VC({vc})" if vc else ""
        info = (
            f"[Ticker] {symbol} {sig.get('signal')}"
            f" | side={sig.get('side')}"
            f" | session={self.global_context.get('session')}"
            f" | trigger={sig.get('trigger_reason')}"
            f"{m15_part}"
            f"{vc_part}"
            f" | narrative={sig.get('narrative')}"
        )
        print(info)

    def _check_active_trade(self, symbol: str, last_price: float, trade: ActiveTrade,
                            mt5_managed: bool = False) -> dict:
        side = trade.side
        tps = trade.tp_prices or []
        n_tps = len(tps)

        # 1) stop — skip when MT5 holds a real SL order; let the broker handle it.
        # EXIT_BROKER (position disappears from MT5) is the exit signal in that case.
        if not mt5_managed:
            if side == "LONG":
                if last_price <= trade.stop:
                    return {"signal": "EXIT_SL", "side": side, "exit_price": last_price, "info": "Стоп-лосс"}
            else:
                if last_price >= trade.stop:
                    return {"signal": "EXIT_SL", "side": side, "exit_price": last_price, "info": "Стоп-лосс"}

        # 2) tps (may jump multiple)
        events: List[Dict[str, Any]] = []
        next_idx = trade.tp_hit + 1

        def _tp_hit_condition(idx: int) -> bool:
            tp = float(tps[idx - 1])
            return (last_price >= tp) if side == "LONG" else (last_price <= tp)

        while next_idx <= n_tps and _tp_hit_condition(next_idx):
            tp_price = float(tps[next_idx - 1])
            events.append({
                "type": "TP",
                "tp_index": next_idx,
                "tp_price": tp_price,
                "hit_price": float(last_price),
            })
            trade.tp_hit = next_idx
            next_idx += 1

        if events:
            if trade.tp_hit >= n_tps and n_tps > 0:
                return {
                    "signal": "EXIT_TP",
                    "side": side,
                    "exit_price": float(last_price),
                    "info": f"TP{n_tps} (final)",
                    "events": events,
                }

            return {
                "signal": "HOLD",
                "side": side,
                "entry_price": float(trade.entry),
                "stop_price": float(trade.stop),
                "tp_prices": [float(x) for x in tps],
                "tp_hit": int(trade.tp_hit),
                "events": events,
                "tf": trade.tf,
                "narrative": trade.narrative,
            }

        return {
            "signal": "HOLD",
            "side": side,
            "entry_price": float(trade.entry),
            "stop_price": float(trade.stop),
            "tp_prices": [float(x) for x in tps],
            "tp_hit": int(trade.tp_hit),
            "tf": trade.tf,
            "narrative": trade.narrative,
        }

    def get_signals(self) -> Dict[str, dict]:
        results: Dict[str, dict] = {}
        symbols = self._get_symbols()
        t_start = time.time()
        prof = self.profiler

        with prof.section("context_update"):
            self._update_global_context()

        dirty = False  # ✅ if active trades changed — persist

        for symbol in symbols:
            if time.time() - t_start > self.TIME_BUDGET_SEC:
                results[symbol] = {"signal": "SKIP_BUDGET", "info": "Time budget exceeded, continue next tick"}
                continue

            try:
                with prof.section("data_fetch"):
                    data = self._build_tf_data(symbol)
                df_1H = data.get("1H")

                if df_1H is None or df_1H.empty:
                    results[symbol] = {"signal": "NO_DATA"}
                    continue

                # ✅ Faster management price from 1M if available (fallback 5M -> 15M -> 1H)
                df_1M = data.get("1M")
                df_5M = data.get("5M")
                df_15M = data.get("15M")

                if df_1M is not None and not df_1M.empty:
                    last_price = float(df_1M["close"].iloc[-1])
                elif df_5M is not None and not df_5M.empty:
                    last_price = float(df_5M["close"].iloc[-1])
                elif df_15M is not None and not df_15M.empty:
                    last_price = float(df_15M["close"].iloc[-1])
                else:
                    last_price = float(df_1H["close"].iloc[-1])

                # manage open trade
                if symbol in self.active_trades:
                    trade = self.active_trades[symbol]

                    if self.mt5_executor and (getattr(trade, "mt5_position_id", None) or getattr(trade, "mt5_ticket", None)):
                        pos_id = trade.mt5_position_id or trade.mt5_ticket
                        position = self.mt5_executor.get_position(symbol, pos_id)
                        if position is None:
                            manage = {
                                "signal": "EXIT_BROKER",
                                "side": trade.side,
                                "exit_price": float(last_price),
                                "info": "Position closed externally",
                                "tf": trade.tf,
                                "narrative": trade.narrative,
                                "events": [
                                    {"type": "BROKER_CLOSE", "info": "Position disappeared from MT5"}
                                ],
                            }
                            results[symbol] = manage
                            del self.active_trades[symbol]
                            dirty = True
                            self._log_signal(symbol, manage)
                            continue

                        # Use real-time MT5 tick price for SL/TP monitoring instead of
                        # potentially stale TV candle data. This eliminates false exits
                        # caused by data-source divergence (TV vs broker real prices).
                        mt5_price = self.mt5_executor.get_current_price(symbol, trade.side)
                        if mt5_price is not None:
                            last_price = mt5_price

                    with prof.section("trade_manage"):
                        mt5_managed = bool(
                            self.mt5_executor
                            and (getattr(trade, "mt5_position_id", None) or getattr(trade, "mt5_ticket", None))
                        )
                        raw_manage = self._check_active_trade(symbol, last_price, trade, mt5_managed=mt5_managed)
                    with prof.section("ai_filter"):
                        manage = self.ai.on_signal(symbol, raw_manage, data, self.active_trades)

                    risk_action = self.risk_rules.check_trade(trade, last_price=last_price)
                    if risk_action:
                        manage = dict(manage)
                        manage.update(risk_action)

                    if raw_manage.get("events"):
                        dirty = True

                    # preserve TP/BE events if AI layer dropped them
                    if raw_manage.get("events") and not manage.get("events"):
                        manage["events"] = raw_manage["events"]
                    if raw_manage.get("signal") in ("EXIT_SL", "EXIT_TP"):
                        manage["signal"] = raw_manage["signal"]
                        manage["exit_price"] = raw_manage.get("exit_price", manage.get("exit_price"))

                    # Block all MT5 closes during startup grace period.
                    _in_grace = (time.time() - self._startup_time) < self._startup_grace_sec
                    if _in_grace and self.mt5_executor:
                        manage.pop("events", None)
                        if manage.get("signal") in ("EXIT_SL", "EXIT_TP", "EXIT_TIME"):
                            print(f"[Core] {symbol} startup grace ({self._startup_grace_sec:.0f}s) — skipping close")
                            manage["signal"] = "HOLD"

                    # Process all trade events: TP partial closes.
                    # Iterate the full list — do NOT break early so every event is handled.
                    for ev in (manage.get("events") or []):
                        ev_type = ev.get("type")

                        if ev_type == "TP":
                            # Execute partial close for this TP level.
                            tp_idx = int(ev.get("tp_index", 0))
                            vol_per_tp = getattr(trade, "volume_per_tp", [])
                            has_executor = self.mt5_executor and getattr(trade, "mt5_position_id", None)
                            if has_executor and tp_idx > 0 and tp_idx <= len(vol_per_tp):
                                partial_vol = vol_per_tp[tp_idx - 1]
                                remaining = getattr(trade, "volume_remaining", 0.0) or trade.volume
                                close_vol = round(min(partial_vol, remaining), 2)
                                if close_vol > 0:
                                    try:
                                        self.mt5_executor.close_trade(
                                            symbol,
                                            position_id=trade.mt5_position_id,
                                            volume=close_vol,
                                        )
                                        trade.volume_remaining = round(max(0.0, remaining - close_vol), 2)
                                        ev["partial_close_vol"] = close_vol
                                        ev["volume_remaining"] = trade.volume_remaining
                                        dirty = True
                                        print(
                                            f"[Core] {symbol} TP{tp_idx} partial close "
                                            f"{close_vol:.2f} lots → remaining: {trade.volume_remaining:.2f}"
                                        )
                                    except Exception as exc:
                                        manage.setdefault("execution_error", str(exc))

                    # Friday weekend close: hard rule — force exit all positions at 22:00 UTC+3.
                    # Placed after TP partial-close events and after grace period so it always fires.
                    if self.global_context.get("friday_close") and manage.get("signal") not in ("EXIT_BROKER",):
                        manage = dict(manage)
                        manage["signal"] = "EXIT_TIME"
                        manage["info"] = "Закрытие перед выходными (пятница 22:00 UTC+3)"
                        manage["exit_price"] = float(last_price)
                        print(f"[Core] {symbol} пятница 22:00 UTC+3 — принудительное закрытие позиции")

                    if manage.get("signal") in ("EXIT_SL", "EXIT_TP", "EXIT_TIME"):
                        manage.setdefault("telegram_chat_id", getattr(trade, "telegram_chat_id", None))
                        manage.setdefault("telegram_message_id", getattr(trade, "telegram_message_id", None))
                        if self.mt5_executor and getattr(trade, "mt5_position_id", None):
                            try:
                                # Use volume_remaining (reduced by prior partial closes).
                                # Fall back to full volume if not initialised (legacy trades).
                                close_vol = getattr(trade, "volume_remaining", 0.0) or trade.volume
                                closed = self.mt5_executor.close_trade(
                                    symbol,
                                    position_id=trade.mt5_position_id,
                                    volume=close_vol,
                                )
                                exec_block = manage.setdefault("execution", {})
                                if not isinstance(exec_block, dict):
                                    exec_block = {}
                                    manage["execution"] = exec_block
                                exec_block["mt5_closed"] = closed
                            except Exception as exc:
                                manage.setdefault("execution_error", str(exc))
                        results[symbol] = manage
                        del self.active_trades[symbol]
                        dirty = True
                        self._log_signal(symbol, manage)
                        continue

                    trade.last_price_ts = time.time()
                    results[symbol] = manage
                    self._log_signal(symbol, manage)
                    continue

                # new signal
                with prof.section("strategy"):
                    raw_sig = self.strategy.generate_signal(data, symbol=symbol)
                with prof.section("ai_filter"):
                    sig = self.ai.on_signal(symbol, raw_sig, data, self.active_trades)

                # preserve tp_prices if AI layer dropped them
                if raw_sig.get("tp_prices") and not sig.get("tp_prices"):
                    sig["tp_prices"] = raw_sig["tp_prices"]
                # Do NOT force ENTER when AI explicitly rejected — that bypasses the filter
                # and causes KeyError (AI_REJECT dict has no side/entry_price/stop_price).

                sig = self._apply_global_filters(symbol, sig)
                sig = self._apply_session_filter(symbol, sig)

                if sig.get("signal") == "ENTER":
                    # Skip if this symbol is in cooldown after a failed execution
                    cooldown_until = self._entry_cooldowns.get(symbol, 0.0)
                    if time.time() < cooldown_until:
                        sig["signal"] = "WAIT_COOLDOWN"
                        sig["info"] = f"Entry cooldown after failed execution"
                        results[symbol] = sig
                        continue

                    tp_prices = sig.get("tp_prices") or [sig.get("tp_price")]
                    tp_prices = [float(x) for x in tp_prices if x is not None]

                    new_trade = ActiveTrade(
                        side=sig["side"],
                        entry=float(sig["entry_price"]),
                        stop=float(sig["stop_price"]),
                        tp_prices=tp_prices,
                        tf=str(sig.get("tf", "")),
                        narrative=str(sig.get("narrative", "")),
                        symbol=symbol,
                        ts_open=time.time(),
                        last_price_ts=time.time(),
                    )

                    if self.mt5_executor:
                        try:
                            # Pass the *final* TP as MT5 hard TP so that the broker closes
                            # any remaining volume at TP_LAST even if the bot is offline.
                            # Intermediate TPs are managed by the bot via partial market closes.
                            execution_payload = self.mt5_executor.execute_entry(
                                symbol,
                                side=new_trade.side,
                                entry_price=new_trade.entry,
                                stop_price=new_trade.stop,
                                tp_price=tp_prices[-1] if tp_prices else None,
                                comment=new_trade.narrative[:28] if new_trade.narrative else None,
                            )
                            new_trade.volume = execution_payload.get("volume", 0.0)
                            new_trade.volume_remaining = new_trade.volume
                            new_trade.volume_per_tp = _compute_tp_volumes(new_trade.volume, len(tp_prices))
                            print(
                                f"[Core] {symbol} volume_per_tp={new_trade.volume_per_tp} "
                                f"(total={new_trade.volume:.2f}, n_tps={len(tp_prices)})"
                            )
                            new_trade.mt5_ticket = execution_payload.get("ticket")
                            new_trade.mt5_position_id = execution_payload.get("position_id")
                            new_trade.execution_comment = execution_payload.get("comment")
                            sig["execution"] = execution_payload
                            # Clear cooldown on success
                            self._entry_cooldowns.pop(symbol, None)

                            # ── Sync entry price to actual MT5 fill price ───────────────
                            # MT5 sets SL/TP at the original signal prices (absolute technical
                            # levels). Only update entry to the real fill so that Telegram
                            # shows the broker's actual open price. SL/TP stay unchanged so
                            # they match exactly what MT5 has on the position.
                            actual_price = execution_payload.get("price")
                            if actual_price and abs(actual_price - new_trade.entry) > 1e-9:
                                offset = actual_price - new_trade.entry
                                new_trade.entry = actual_price
                                sig["entry_price"] = actual_price
                                print(
                                    f"[Core] {symbol} fill offset {offset:+.5f}: "
                                    f"entry={actual_price}, stop={new_trade.stop:.5f} (unchanged)"
                                )

                            # Strip TPs already passed by the fill price (slippage).
                            # Without this, bot immediately marks near TPs as hit even
                            # though MT5 never triggered them.
                            _e = new_trade.entry
                            if new_trade.side == "LONG":
                                new_trade.tp_prices = [t for t in new_trade.tp_prices if t > _e]
                            else:
                                new_trade.tp_prices = [t for t in new_trade.tp_prices if t < _e]
                            sig["tp_prices"] = [round(t, 6) for t in new_trade.tp_prices]
                            if new_trade.tp_prices:
                                sig["tp_price"] = round(new_trade.tp_prices[-1], 6)
                            print(
                                f"[Core] {symbol} after fill={new_trade.entry:.5f}: "
                                f"active tp_prices={[round(t, 5) for t in new_trade.tp_prices]}"
                            )
                        except Exception as exc:
                            # Stale-rejection means price has already moved through the setup's SL.
                            # Use a much longer cooldown so the signal isn't retried every 5 min
                            # while price continues trending through the invalidated level.
                            err_str = str(exc)
                            is_stale = "Stale signal rejected" in err_str
                            cooldown_sec = self._stale_cooldown_sec if is_stale else self._entry_cooldown_sec
                            self._entry_cooldowns[symbol] = time.time() + cooldown_sec
                            sig["signal"] = "EXECUTION_ERROR"
                            sig["execution_error"] = err_str
                            print(f"[Core] Execution failed for {symbol}: {exc}. Cooldown {cooldown_sec:.0f}s")
                            results[symbol] = sig
                            self._log_signal(symbol, sig)
                            continue

                    self.active_trades[symbol] = new_trade
                    dirty = True

                results[symbol] = sig
                self._log_signal(symbol, sig)

            except Exception:
                traceback.print_exc()
                results[symbol] = {"signal": "ERROR", "info": "Exception in get_signals() (see logs)"}

        if dirty:
            with prof.section("save_state"):
                save_active_trades(self.active_trades, AI_DATA_DIR / "active_trades.json")

        prof.dump(prefix="[Profiler:core]")
        return results

if __name__ == "__main__":
    core = Core()
    bot = TelegramBot(TELEGRAM_TOKEN, core)
    bot.run()
