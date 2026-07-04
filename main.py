# main.py
from __future__ import annotations

import math
import os
import threading
import time
import traceback
from pathlib import Path
from datetime import datetime, time as dt_time
from typing import Dict, Any, List, Tuple, Optional
from zoneinfo import ZoneInfo

from config import (
    TELEGRAM_TOKEN,
    UNIVERSE,
    CONTEXT_SYMBOLS,
    AI_DATA_DIR,
    LOG_TICK,
    DEBUG_RAW_SIGNALS,
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
    MT5_BRIDGE_SYMBOLS,
    MT5_BRIDGE_TIMEFRAMES,
    MT5_BRIDGE_LOOKBACK_DAYS,
    MT5_BRIDGE_INTERVAL,
    PARTIAL_TP_MODE,
    MOVE_BE_AFTER_TP1,
    SIGNAL_ON_CLOSED_BARS,
)
from core.mt5_guard import install as _install_mt5_guard

# Must run before any thread touches the MetaTrader5 API — wraps every mt5.*
# call with a shared lock (tick loop, DataCacheLoop and MT5Bridge all use it).
_install_mt5_guard()

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
from mt5_bridge.mt5_native_bridge import MT5NativeBridge, parse_symbol_spec, parse_timeframes


def _compute_tp_volumes(total_volume: float, n_tps: int, step: float = 0.01) -> List[float]:
    """
    Split total_volume into per-TP partial-close amounts.

      1 TP  → [100%]
      2 TPs → [50%, remainder]
      3 TPs → [50%, 25%, remainder]
      4 TPs → [25%, 30%, 30%, remainder]
      >4 TPs → first three get 25%/30%/30%, last entry is always remainder

    Volumes are floored to the broker's volume_step — legs that are not
    step-aligned get rejected with "Invalid volume" on some brokers.
    """
    if n_tps <= 0 or total_volume <= 0:
        return []
    if not step or step <= 0:
        step = 0.01

    def _quantize(v: float) -> float:
        return round(math.floor(v / step + 1e-9) * step, 8)

    if n_tps == 1:
        return [_quantize(total_volume)]

    if n_tps == 2:
        pre_ratios: List[float] = [0.50]
    elif n_tps == 3:
        pre_ratios = [0.50, 0.25]
    else:
        pre_ratios = [0.25, 0.30, 0.30]
        pre_ratios += [0.0] * (n_tps - 4)

    vols: List[float] = []
    allocated = 0.0
    for ratio in pre_ratios:
        vol = _quantize(total_volume * ratio)
        vols.append(vol)
        allocated = round(allocated + vol, 10)

    remainder = _quantize(total_volume - allocated)
    # Floating-point rounding can produce 0.00 or tiny negatives for the
    # last leg.  Clamp to 0.0 — the split executor will skip zero-volume legs.
    vols.append(max(0.0, remainder))
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

        self.active_trades: dict[str, ActiveTrade] = load_active_trades(AI_DATA_DIR / "active_trades.json")
        print(f"[Core] restored active_trades={len(self.active_trades)}")

        self.ai_cfg = AIConfig()
        self.ai_store = TradeStore(self.ai_cfg)
        self.ai = AILive(self.ai_cfg, self.ai_store, self.strategy)

        self.TIME_BUDGET_SEC = 35.0
        self.N_BARS = 300

        self.profiler = TickProfiler()
        self.global_context: Dict[str, Any] = {"session": "ALL", "session_allowed": True}
        self.log_tick = LOG_TICK
        self.risk_rules = RiskRules()

        # Grace period after startup: block MT5 closes for 90s to let state sync
        self._startup_time: float = time.time()
        self._startup_grace_sec: float = 90.0

        # cooldown per symbol after a failed entry attempt (prevents infinite retries)
        self._entry_cooldowns: Dict[str, float] = {}
        self._entry_cooldown_sec: float = 300.0  # 5 minutes
        self._stale_cooldown_sec: float = 60.0   # shorter cooldown for stale-price rejections

        # A broker position "disappearing" must be confirmed on N consecutive ticks
        # with a healthy MT5 connection before the trade is treated as closed.
        # Otherwise a dropped terminal link (positions_get() → None) produces a
        # false EXIT_BROKER and the bot forgets live positions.
        self._broker_missing_counts: Dict[str, int] = {}
        self._broker_missing_confirm: int = 2
        self._last_reconnect_ts: float = 0.0

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

        if self.mt5_executor:
            self._hydrate_active_trades_from_mt5()

    def _hydrate_active_trades_from_mt5(self) -> None:
        try:
            positions = self.mt5_executor.list_positions() if self.mt5_executor else []
        except Exception as exc:
            print(f"[Core] MT5 hydration skipped: {exc}")
            return

        if not positions:
            return

        # Group positions by symbol so we can properly rebuild split_position_ids
        from collections import defaultdict
        by_symbol: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for pos in positions:
            symbol = pos.get("symbol")
            if symbol and symbol in self.universe:
                by_symbol[symbol].append(pos)

        hydrated = 0
        relinked = 0
        for symbol, sym_positions in by_symbol.items():
            first = sym_positions[0]
            all_tickets = [int(p["ticket"]) for p in sym_positions if p.get("ticket")]
            total_volume = round(sum(float(p.get("volume", 0.0) or 0.0) for p in sym_positions), 2)
            is_split = len(all_tickets) > 1

            trade = self.active_trades.get(symbol)
            if trade:
                updated = False
                existing_split = list(getattr(trade, "split_position_ids", []) or [])
                if is_split:
                    # Rebuild split_position_ids — critical after restart so bot tracks all legs
                    if set(all_tickets) != set(existing_split):
                        trade.split_position_ids = all_tickets
                        trade.mt5_position_id = all_tickets[0]
                        trade.mt5_ticket = all_tickets[0]
                        updated = True
                else:
                    ticket = all_tickets[0] if all_tickets else None
                    if ticket and trade.mt5_position_id != ticket:
                        trade.mt5_position_id = ticket
                        trade.mt5_ticket = ticket
                        updated = True
                if abs(float(trade.volume or 0.0) - total_volume) > 1e-6:
                    trade.volume = total_volume
                    updated = True
                if updated:
                    relinked += 1
                    print(
                        f"[Core] relinked {symbol}: split_ids={all_tickets}, vol={total_volume}"
                        if is_split else
                        f"[Core] relinked {symbol}: ticket={all_tickets[0] if all_tickets else None}, vol={total_volume}"
                    )
                continue

            # Position(s) not in active_trades — hydrate from MT5
            tp_prices = sorted({float(p.get("tp", 0.0) or 0.0) for p in sym_positions} - {0.0})
            narrative = first.get("comment") or "Hydrated from MT5"
            trade = ActiveTrade(
                side=str(first.get("side", "LONG")).upper(),
                entry=float(first.get("entry_price", 0.0) or 0.0),
                stop=float(first.get("stop", 0.0) or 0.0),
                tp_prices=tp_prices,
                tf="15m",
                narrative=narrative,
                symbol=symbol,
            )
            trade.volume = total_volume
            trade.volume_remaining = total_volume
            trade.mt5_ticket = all_tickets[0] if all_tickets else None
            trade.mt5_position_id = all_tickets[0] if all_tickets else None
            if is_split:
                trade.split_position_ids = all_tickets
            trade.ts_open = float(first.get("time", time.time()) or time.time())
            trade.last_price_ts = time.time()
            self.active_trades[symbol] = trade
            hydrated += 1
            print(
                f"[Core] hydrated {symbol} (split {len(all_tickets)} legs, vol={total_volume}, tps={tp_prices})"
                if is_split else
                f"[Core] hydrated {symbol} (vol={total_volume}, tp={tp_prices})"
            )

        if hydrated or relinked:
            save_active_trades(self.active_trades, AI_DATA_DIR / "active_trades.json")
            print(f"[Core] hydration done: {hydrated} new, {relinked} relinked")


    def _position_gone_confirmed(self, symbol: str, query_failed: bool) -> bool:
        """True only after N consecutive ticks where the position is absent AND the
        MT5 connection is verifiably alive. Any failed/untrusted query resets nothing
        and confirms nothing — better to hold a closed trade one extra tick than to
        forget a live position."""
        if query_failed:
            return False
        if not (self.mt5_executor and self.mt5_executor.connection_alive()):
            return False
        count = self._broker_missing_counts.get(symbol, 0) + 1
        self._broker_missing_counts[symbol] = count
        return count >= self._broker_missing_confirm

    @staticmethod
    def _trade_rr(trade: ActiveTrade) -> Optional[float]:
        try:
            tps = list(trade.tp_prices or [])
            risk = abs(float(trade.entry) - float(trade.stop))
            if not tps or risk <= 0:
                return None
            return abs(float(tps[-1]) - float(trade.entry)) / risk
        except Exception:
            return None

    @staticmethod
    def _closed_bars_view(data: Dict[str, Any]) -> Dict[str, Any]:
        """Strategy input with the still-forming last candle removed per timeframe.

        MT5 copy_rates returns the current forming bar as the last row while the
        market is open; triggers computed on it can appear mid-bar and vanish by
        the close (repaint). Dropping the last row makes signals deterministic
        per closed bar. Price/SLTP checks keep using the live tick, not this view.
        """
        out: Dict[str, Any] = {}
        for tf, df in (data or {}).items():
            if df is not None and getattr(df, "empty", True) is False and len(df) > 1:
                out[tf] = df.iloc[:-1]
            else:
                out[tf] = df
        return out

    def _move_to_breakeven(self, symbol: str, trade: ActiveTrade) -> bool:
        """Move SL to the entry (fill) price after TP1. Returns True on success.

        Split mode: updates every remaining leg. Monitor mode: updates the single
        position. move_stop() itself clamps the level to the broker's minimum
        stop distance, so this never produces [Invalid stops].
        """
        if not (MOVE_BE_AFTER_TP1 and self.mt5_executor):
            return False
        if getattr(trade, "moved_to_be", False):
            return False
        be_price = float(trade.entry)
        if be_price <= 0:
            return False
        try:
            split_ids = list(getattr(trade, "split_position_ids", []) or [])
            if split_ids:
                updated = self.mt5_executor.move_stop_all(symbol, position_ids=split_ids, new_stop=be_price)
                ok = updated > 0
            else:
                pos_id = getattr(trade, "mt5_position_id", None) or getattr(trade, "mt5_ticket", None)
                if not pos_id:
                    return False
                ok = self.mt5_executor.move_stop(symbol, position_id=pos_id, new_stop=be_price)
        except Exception as exc:
            print(f"[Core] {symbol} BE move failed: {exc}")
            return False
        if ok:
            trade.moved_to_be = True
            trade.stop = be_price
            print(f"[Core] {symbol} TP1 hit — SL moved to break-even {be_price:.5f}")
        return ok

    def _register_broker_close(self, symbol: str, trade: ActiveTrade, manage: Dict[str, Any]) -> None:
        """Infer TP/SL outcome of a broker-side close from MT5 deal history and
        feed it to the AI stats store. Split mode ends every trade via EXIT_BROKER,
        so without this neither the journal nor the AI filter ever sees outcomes."""
        if not self.mt5_executor:
            return
        ids = list(getattr(trade, "split_position_ids", []) or [])
        if not ids:
            pid = getattr(trade, "mt5_position_id", None) or getattr(trade, "mt5_ticket", None)
            ids = [pid] if pid else []
        outcome: Optional[str] = None
        for pid in ids:
            reason = self.mt5_executor.get_position_close_reason(pid)
            if reason == "SL":
                outcome = "SL"
                break
            if reason == "TP":
                outcome = "TP"
        if outcome:
            manage["outcome"] = outcome
            try:
                self.ai_store.update_on_close(symbol, outcome, rr_numeric=self._trade_rr(trade))
            except Exception:
                traceback.print_exc()

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
        signal_type = sig.get("signal")
        # Always log anything that isn't a silent HOLD
        important = signal_type not in {"HOLD", None}
        if not (self.log_tick or important):
            return
        m15_raw = sig.get("m15_trend_raw")
        m15_part = f" | m15_ema={m15_raw}" if m15_raw else ""
        vc = sig.get("vc")
        vc_part = f" | block=VC({vc})" if vc else ""
        info = (
            f"[Ticker] {symbol} {signal_type}"
            f" | side={sig.get('side')}"
            f" | session={self.global_context.get('session')}"
            f" | trigger={sig.get('trigger_reason')}"
            f"{m15_part}"
            f"{vc_part}"
            f" | narrative={sig.get('narrative')}"
        )
        if signal_type == "EXECUTION_ERROR" and sig.get("execution_error"):
            info += f" | error={sig.get('execution_error')}"
        if signal_type == "WAIT_SESSION":
            info += f" | blocked_by={sig.get('session_blocked')}"
        if signal_type == "SKIP_BUDGET":
            info += f" | reason=time_budget_exceeded"
        if signal_type == "NO_DATA":
            info += f" | reason=no_data_from_feed"
        if signal_type == "EXIT_BROKER":
            info += f" | reason=position_closed_externally_in_MT5"
        try:
            print(info)
        except UnicodeEncodeError:
            print(info.encode("utf-8", errors="replace").decode("ascii", errors="replace"))

    def _check_active_trade(self, symbol: str, last_price: float, trade: ActiveTrade) -> dict:
        side = trade.side
        tps = trade.tp_prices or []
        n_tps = len(tps)

        stop = float(trade.stop or 0.0)
        # stop == 0 means "no SL known" (e.g. a position hydrated from MT5 without SL).
        # A zero stop must never trigger an exit — for SHORT `price >= 0` is always true.
        if stop > 0:
            if side == "LONG":
                if last_price <= stop:
                    return {"signal": "EXIT_SL", "side": side, "exit_price": last_price, "info": "Стоп-лосс"}
            else:
                if last_price >= stop:
                    return {"signal": "EXIT_SL", "side": side, "exit_price": last_price, "info": "Стоп-лосс"}

        # Final TP was already reached earlier but the close failed (trade kept for
        # retry) → re-emit EXIT_TP so the close is retried instead of holding forever.
        if n_tps > 0 and trade.tp_hit >= n_tps:
            return {
                "signal": "EXIT_TP",
                "side": side,
                "exit_price": float(last_price),
                "info": f"TP{n_tps} (final, retry)",
            }

        events: List[Dict[str, Any]] = []
        next_idx = trade.tp_hit + 1

        def _tp_hit_condition(idx: int) -> bool:
            tp = float(tps[idx - 1])
            return (last_price >= tp) if side == "LONG" else (last_price <= tp)

        while next_idx <= n_tps and _tp_hit_condition(next_idx):
            tp_price = float(tps[next_idx - 1])
            events.append({"type": "TP", "tp_index": next_idx, "tp_price": tp_price, "hit_price": float(last_price)})
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

        # MT5 health check: the executor connects once at startup and the link can
        # silently die (terminal restart, network). Try to re-initialize, throttled.
        if self.mt5_executor and not self.mt5_executor.connection_alive():
            now = time.time()
            if now - self._last_reconnect_ts >= 60.0:
                self._last_reconnect_ts = now
                ok = self.mt5_executor.reconnect()
                print(f"[MT5] connection lost — reconnect {'ok' if ok else 'failed'}")

        dirty = False

        for symbol in symbols:
            if time.time() - t_start > self.TIME_BUDGET_SEC:
                results[symbol] = {"signal": "SKIP_BUDGET", "info": "Time budget exceeded, continue next tick"}
                continue

            try:
                with prof.section("data_fetch"):
                    data = self._build_tf_data(symbol)
                df_1H = data.get("1H")

                if df_1H is None or df_1H.empty:
                    print(f"[Ticker] {symbol} NO_DATA | 1H frame is empty — check data source / MT5 bridge")
                    results[symbol] = {"signal": "NO_DATA"}
                    continue

                df_1M = data.get("1M")
                df_5M = data.get("5M")
                df_15M = data.get("15M")

                # MT5 live tick is primary — bridge candles are fallback only.
                # Pass the open trade's side so SHORT positions are monitored on
                # ASK (the price MT5 uses for their SL/TP) and LONG on BID.
                _trade_side = getattr(self.active_trades.get(symbol), "side", None)
                _mt5_live = self.mt5_executor.get_current_price(symbol, _trade_side) if self.mt5_executor else None
                if _mt5_live is not None:
                    last_price = _mt5_live
                elif df_1M is not None and not df_1M.empty:
                    last_price = float(df_1M["close"].iloc[-1])
                elif df_5M is not None and not df_5M.empty:
                    last_price = float(df_5M["close"].iloc[-1])
                elif df_15M is not None and not df_15M.empty:
                    last_price = float(df_15M["close"].iloc[-1])
                else:
                    last_price = float(df_1H["close"].iloc[-1])

                if symbol in self.active_trades:
                    trade = self.active_trades[symbol]

                    if self.mt5_executor and (getattr(trade, "mt5_position_id", None) or getattr(trade, "mt5_ticket", None)):
                        split_ids = getattr(trade, "split_position_ids", [])
                        if split_ids:
                            # Split mode: check how many legs are still open.
                            # positions_get() returns None on an API/connection failure and
                            # an EMPTY tuple when there really are no positions — never
                            # conflate the two, or a dropped terminal link masquerades as
                            # "all legs closed" and live positions get forgotten.
                            import MetaTrader5 as _mt5
                            open_positions = _mt5.positions_get(symbol=symbol)
                            query_failed = open_positions is None
                            open_ids = {p.ticket for p in (open_positions or [])}
                            still_open = [pid for pid in split_ids if pid in open_ids]
                            if still_open:
                                self._broker_missing_counts.pop(symbol, None)
                                # Update remaining legs list
                                if len(still_open) < len(split_ids):
                                    trade.split_position_ids = still_open
                                    dirty = True
                            elif self._position_gone_confirmed(symbol, query_failed):
                                # All legs closed (all TPs hit or SL fired)
                                self._broker_missing_counts.pop(symbol, None)
                                manage = {
                                    "signal": "EXIT_BROKER",
                                    "side": trade.side,
                                    "exit_price": float(last_price),
                                    "info": "All split legs closed",
                                    "tf": trade.tf,
                                    "narrative": trade.narrative,
                                    "events": [{"type": "BROKER_CLOSE", "info": "All split TP legs closed in MT5"}],
                                    "telegram_chat_id": getattr(trade, "telegram_chat_id", None),
                                    "telegram_message_id": getattr(trade, "telegram_message_id", None),
                                }
                                self._register_broker_close(symbol, trade, manage)
                                results[symbol] = manage
                                del self.active_trades[symbol]
                                dirty = True
                                self._log_signal(symbol, manage)
                                continue
                            else:
                                info = (
                                    "MT5 query failed — keeping trade"
                                    if query_failed
                                    else "split legs not visible — awaiting confirmation"
                                )
                                print(f"[Core] {symbol} broker check inconclusive: {info}")
                                results[symbol] = {"signal": "HOLD", "info": info}
                                continue
                        else:
                            pos_id = trade.mt5_position_id or trade.mt5_ticket
                            position = self.mt5_executor.get_position(symbol, pos_id)
                            if position is not None:
                                self._broker_missing_counts.pop(symbol, None)
                            elif self._position_gone_confirmed(symbol, query_failed=False):
                                self._broker_missing_counts.pop(symbol, None)
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
                                    "telegram_chat_id": getattr(trade, "telegram_chat_id", None),
                                    "telegram_message_id": getattr(trade, "telegram_message_id", None),
                                }
                                self._register_broker_close(symbol, trade, manage)
                                results[symbol] = manage
                                del self.active_trades[symbol]
                                dirty = True
                                self._log_signal(symbol, manage)
                                continue
                            else:
                                print(f"[Core] {symbol} position not visible — awaiting confirmation before EXIT_BROKER")
                                results[symbol] = {"signal": "HOLD", "info": "position not visible — awaiting confirmation"}
                                continue

                    with prof.section("trade_manage"):
                        raw_manage = self._check_active_trade(symbol, last_price, trade)

                    # In split mode the broker owns each leg's SL and TP.
                    # EXIT_SL, EXIT_TP, and EXIT_TIME are all suppressed — the bot waits
                    # for EXIT_BROKER (all split_position_ids disappear from MT5) instead
                    # of sending redundant close orders that would fill at market price
                    # rather than the exact TP/SL levels set on each leg.
                    _is_split_active = bool(getattr(trade, "split_position_ids", []))
                    if _is_split_active and raw_manage.get("signal") in ("EXIT_SL", "EXIT_TP"):
                        raw_manage = dict(raw_manage)
                        raw_manage["signal"] = "HOLD"
                        raw_manage["info"] = "split_mode: SL/TP managed by broker"

                    with prof.section("ai_filter"):
                        manage = self.ai.on_signal(symbol, raw_manage, data, self.active_trades)

                    risk_action = self.risk_rules.check_trade(trade, last_price=last_price)
                    # In split mode the broker manages each leg's TP/SL automatically.
                    # EXIT_TIME would force-close still-open legs that haven't reached TP yet,
                    # turning a winning trade into a loss. Let broker handle the exit instead.
                    if risk_action and not _is_split_active:
                        manage = dict(manage)
                        manage.update(risk_action)

                    if raw_manage.get("events"):
                        dirty = True

                    if raw_manage.get("events") and not manage.get("events"):
                        manage["events"] = raw_manage["events"]
                    if raw_manage.get("signal") in ("EXIT_SL", "EXIT_TP"):
                        manage["signal"] = raw_manage["signal"]
                        manage["exit_price"] = raw_manage.get("exit_price", manage.get("exit_price"))

                    # Block all MT5 closes during startup grace period.
                    _in_grace = (time.time() - self._startup_time) < self._startup_grace_sec
                    if _in_grace and self.mt5_executor:
                        dropped_events = manage.pop("events", None) or []
                        # _check_active_trade already advanced tp_hit for these events;
                        # roll it back, otherwise the partial closes (and Telegram
                        # notifications) for those TPs are swallowed forever.
                        n_tp_events = sum(1 for e in dropped_events if e.get("type") == "TP")
                        if n_tp_events:
                            trade.tp_hit = max(0, int(trade.tp_hit) - n_tp_events)
                            dirty = True
                        if manage.get("signal") in ("EXIT_SL", "EXIT_TP", "EXIT_TIME"):
                            print(f"[Core] {symbol} startup grace ({self._startup_grace_sec:.0f}s) — skipping close")
                            manage["signal"] = "HOLD"

                    # Process TP partial closes
                    for ev in (manage.get("events") or []):
                        ev_type = ev.get("type")

                        if ev_type == "TP":
                            # In split mode each leg has its own broker TP — the broker
                            # closes it automatically at the exact price.  Sending an
                            # additional close_trade() here would fill at the current
                            # market price, which is always worse than the broker TP.
                            if getattr(trade, "split_position_ids", []):
                                continue
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

                    # Move SL to break-even once TP1 is reached. tp_hit is persisted, so
                    # a failed modify retries every tick until it succeeds. Works for both
                    # modes: split (all remaining legs) and monitor (single position).
                    if (
                        not getattr(trade, "moved_to_be", False)
                        and int(getattr(trade, "tp_hit", 0) or 0) >= 1
                        and manage.get("signal") not in ("EXIT_SL", "EXIT_TP", "EXIT_TIME", "EXIT_BROKER")
                    ):
                        if self._move_to_breakeven(symbol, trade):
                            dirty = True
                            manage.setdefault("events", []).append(
                                {"type": "BE", "price": float(trade.entry)}
                            )

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
                        close_failed = False
                        if self.mt5_executor:
                            exec_block = manage.setdefault("execution", {})
                            if not isinstance(exec_block, dict):
                                exec_block = {}
                                manage["execution"] = exec_block
                            conn_ok = self.mt5_executor.connection_alive()
                            split_ids = getattr(trade, "split_position_ids", [])
                            if split_ids:
                                # Close any remaining split legs (already-hit TPs are gone)
                                closed_count = 0
                                remaining_legs: List[int] = []
                                for pid in split_ids:
                                    try:
                                        ok = self.mt5_executor.close_trade(symbol, position_id=pid, volume=None)
                                        if ok:
                                            closed_count += 1
                                        elif not conn_ok:
                                            # "not found" is not trustworthy on a dead connection
                                            remaining_legs.append(pid)
                                    except Exception as exc:
                                        manage.setdefault("execution_error", str(exc))
                                        remaining_legs.append(pid)
                                exec_block["mt5_closed_split"] = closed_count
                                if remaining_legs:
                                    close_failed = True
                                    trade.split_position_ids = remaining_legs
                            elif getattr(trade, "mt5_position_id", None):
                                try:
                                    close_vol = getattr(trade, "volume_remaining", 0.0) or trade.volume
                                    closed = self.mt5_executor.close_trade(
                                        symbol,
                                        position_id=trade.mt5_position_id,
                                        volume=close_vol,
                                    )
                                    exec_block["mt5_closed"] = closed
                                    if not closed and not conn_ok:
                                        close_failed = True
                                except Exception as exc:
                                    manage.setdefault("execution_error", str(exc))
                                    close_failed = True
                        if close_failed:
                            # Keep the trade tracked and retry next tick — deleting it
                            # here would leave a live position unmanaged in the market.
                            print(
                                f"[Core] {symbol} close failed ({manage.get('execution_error', 'position not confirmed')})"
                                f" — keeping trade, retrying next tick"
                            )
                            manage = dict(manage)
                            manage["signal"] = "HOLD"
                            manage["info"] = "MT5 close failed — retrying next tick"
                            trade.last_price_ts = time.time()
                            results[symbol] = manage
                            dirty = True
                            self._log_signal(symbol, manage)
                            continue
                        results[symbol] = manage
                        del self.active_trades[symbol]
                        dirty = True
                        self._log_signal(symbol, manage)
                        continue

                    trade.last_price_ts = time.time()
                    results[symbol] = manage
                    self._log_signal(symbol, manage)
                    continue

                strategy_data = self._closed_bars_view(data) if SIGNAL_ON_CLOSED_BARS else data
                with prof.section("strategy"):
                    raw_sig = self.strategy.generate_signal(strategy_data, symbol=symbol)
                if DEBUG_RAW_SIGNALS and raw_sig.get("signal") != "NO_TRIGGER":
                    print(f"[RAW_SIG] {symbol} {raw_sig}")
                with prof.section("ai_filter"):
                    sig = self.ai.on_signal(symbol, raw_sig, data, self.active_trades)

                if raw_sig.get("tp_prices") and not sig.get("tp_prices"):
                    sig["tp_prices"] = raw_sig["tp_prices"]
                # Do NOT force ENTER when AI explicitly rejected — AI_REJECT dict has no
                # side/entry_price/stop_price and bypassing the filter causes SKIP_NO_SIDE every tick.

                sig = self._apply_global_filters(symbol, sig)
                sig = self._apply_session_filter(symbol, sig)

                if sig.get("signal") == "ENTER":
                    side = sig.get("side")
                    if not side:
                        sig["signal"] = "SKIP_NO_SIDE"
                        results[symbol] = sig
                        self._log_signal(symbol, sig)
                        continue

                    # Skip if this symbol is in cooldown after a failed execution
                    cooldown_until = self._entry_cooldowns.get(symbol, 0.0)
                    if time.time() < cooldown_until:
                        sig["signal"] = "WAIT_COOLDOWN"
                        sig["info"] = "Entry cooldown after failed execution"
                        results[symbol] = sig
                        self._log_signal(symbol, sig)
                        continue

                    # Anti-hedging guard: block entry if opposite MT5 position is open
                    if self.mt5_executor:
                        import MetaTrader5 as _mt5
                        _open_pos = _mt5.positions_get(symbol=symbol) or []
                        _expected_type = _mt5.POSITION_TYPE_BUY if side == "LONG" else _mt5.POSITION_TYPE_SELL
                        _opposite = [p for p in _open_pos if p.magic == self.mt5_executor.settings.magic and p.type != _expected_type]
                        if _opposite:
                            sig["signal"] = "SKIP_HEDGE"
                            sig["info"] = f"Opposite MT5 position still open ({len(_opposite)} legs)"
                            results[symbol] = sig
                            self._log_signal(symbol, sig)
                            print(f"[Core] {symbol} anti-hedge block: {len(_opposite)} opposite leg(s) still open in MT5")
                            continue

                    tp_prices = sig.get("tp_prices") or [sig.get("tp_price")]
                    tp_prices = [float(x) for x in tp_prices if x is not None]

                    new_trade = ActiveTrade(
                        side=side,
                        entry=float(sig.get("entry_price", 0.0)),
                        stop=float(sig.get("stop_price", 0.0)),
                        tp_prices=tp_prices,
                        tf=str(sig.get("tf", "")),
                        narrative=str(sig.get("narrative", "")),
                        symbol=symbol,
                        ts_open=time.time(),
                        last_price_ts=time.time(),
                    )

                    if self.mt5_executor:
                        try:
                            # Auto-select mode:
                            #   split  → 2+ TPs and PARTIAL_TP_MODE != "monitor"
                            #   monitor → 1 TP, or forced via PARTIAL_TP_MODE=monitor
                            use_split = (
                                len(tp_prices) > 1
                                and PARTIAL_TP_MODE != "monitor"
                            )
                            if use_split:
                                # MK-style: calculate total volume once, then open N legs
                                import MetaTrader5 as _mt5
                                tick = _mt5.symbol_info_tick(symbol)
                                if tick is None:
                                    raise RuntimeError(f"No tick for {symbol}")
                                actual_entry = float(tick.ask if new_trade.side == "LONG" else tick.bid)
                                total_vol = self.mt5_executor._calc_volume(symbol, actual_entry, new_trade.stop)
                                # Sort TPs nearest-first: leg-0 (largest volume at 4+ TPs)
                                # targets the nearest TP, not the furthest.
                                tp_prices = (
                                    sorted(tp_prices)
                                    if new_trade.side == "LONG"
                                    else sorted(tp_prices, reverse=True)
                                )
                                new_trade.tp_prices = tp_prices
                                _info = _mt5.symbol_info(symbol)
                                _step = float(getattr(_info, "volume_step", 0.0) or 0.0) if _info else 0.0
                                vols = _compute_tp_volumes(total_vol, len(tp_prices), step=_step or 0.01)
                                legs = self.mt5_executor.execute_split_entry(
                                    symbol,
                                    side=new_trade.side,
                                    entry_price=new_trade.entry,
                                    stop_price=new_trade.stop,
                                    tp_prices=tp_prices,
                                    volumes_per_tp=vols,
                                    comment=new_trade.narrative[:20] if new_trade.narrative else None,
                                )
                                if not legs:
                                    raise RuntimeError(
                                        "Split entry opened no legs (all volumes below broker minimum)"
                                    )
                                new_trade.volume = round(sum(l["volume"] for l in legs), 2)
                                new_trade.volume_remaining = new_trade.volume
                                new_trade.volume_per_tp = vols
                                new_trade.split_position_ids = [
                                    l["position_id"] for l in legs if l.get("position_id")
                                ]
                                # anchor mt5_position_id to first leg for backward compat
                                if new_trade.split_position_ids:
                                    new_trade.mt5_position_id = new_trade.split_position_ids[0]
                                new_trade.mt5_ticket = legs[0]["ticket"] if legs else None
                                new_trade.execution_comment = legs[0].get("comment") if legs else None
                                sig["execution"] = {"legs": legs, "mode": "split"}
                                # Update entry to actual volume-weighted fill price
                                _fill_vols = [l["volume"] for l in legs]
                                _fill_prices = [l["price"] for l in legs]
                                _total_vol = sum(_fill_vols)
                                if _total_vol > 0:
                                    _avg_fill = sum(p * v for p, v in zip(_fill_prices, _fill_vols)) / _total_vol
                                    new_trade.entry = round(_avg_fill, 6)
                                    sig["entry_price"] = new_trade.entry
                                print(
                                    f"[Core] {symbol} SPLIT entry: {len(legs)} legs, "
                                    f"vols={vols}, total={new_trade.volume:.2f}, "
                                    f"position_ids={new_trade.split_position_ids}"
                                )
                            else:
                                execution_payload = self.mt5_executor.execute_entry(
                                    symbol,
                                    side=new_trade.side,
                                    entry_price=new_trade.entry,
                                    stop_price=new_trade.stop,
                                    # Pass LAST TP as hard broker TP so intermediate TPs
                                    # are handled by the bot via partial market closes.
                                    tp_price=tp_prices[-1] if tp_prices else None,
                                    comment=new_trade.narrative[:28] if new_trade.narrative else None,
                                )
                                new_trade.volume = execution_payload.get("volume", 0.0)
                                new_trade.volume_remaining = new_trade.volume
                                new_trade.volume_per_tp = _compute_tp_volumes(new_trade.volume, len(tp_prices))
                                print(
                                    f"[Core] {symbol} MONITOR entry: volume_per_tp={new_trade.volume_per_tp} "
                                    f"(total={new_trade.volume:.2f}, n_tps={len(tp_prices)})"
                                )
                                new_trade.mt5_ticket = execution_payload.get("ticket")
                                new_trade.mt5_position_id = execution_payload.get("position_id")
                                new_trade.execution_comment = execution_payload.get("comment")
                                sig["execution"] = execution_payload
                                # Update entry to actual broker fill price
                                _fill = execution_payload.get("price")
                                if _fill:
                                    new_trade.entry = round(float(_fill), 6)
                                    sig["entry_price"] = new_trade.entry

                            # Strip TPs that the fill price has already passed.
                            # Slippage can push the fill beyond near TPs, causing the bot
                            # to immediately mark them as hit even though MT5 never triggered.
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
                                f"active tp_prices={[round(t,5) for t in new_trade.tp_prices]}"
                            )
                            self._entry_cooldowns.pop(symbol, None)
                        except Exception as exc:
                            err_str = str(exc)
                            is_stale = "Stale signal rejected" in err_str
                            cooldown_sec = self._stale_cooldown_sec if is_stale else self._entry_cooldown_sec
                            self._entry_cooldowns[symbol] = time.time() + cooldown_sec
                            sig["signal"] = "EXECUTION_ERROR"
                            sig["execution_error"] = err_str
                            results[symbol] = sig
                            self._log_signal(symbol, sig)
                            print(f"[Core] Execution failed for {symbol}: {exc}. Cooldown {cooldown_sec:.0f}s")
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


def _should_start_bridge() -> bool:
    # Bridge writes cache files used as DataFeed fallback — start whenever
    # MT5 credentials are available, regardless of execution mode.
    return bool(MT5_LOGIN and MT5_PASSWORD and MT5_SERVER)


def _start_bridge_thread(manage_connection: bool) -> Tuple[Optional[threading.Event], Optional[threading.Thread]]:
    try:
        mappings = parse_symbol_spec(MT5_BRIDGE_SYMBOLS)
        timeframes = parse_timeframes(MT5_BRIDGE_TIMEFRAMES)
    except Exception as exc:
        print(f"[MT5 Bridge] config error: {exc}")
        return None, None

    stop_event = threading.Event()

    def _runner():
        bridge = MT5NativeBridge(
            mappings=mappings,
            timeframes=timeframes,
            lookback_days=MT5_BRIDGE_LOOKBACK_DAYS,
            poll_interval=MT5_BRIDGE_INTERVAL,
            cache_dir=MT5_CACHE_DIR,
        )
        try:
            bridge.run(
                once=False,
                stop_event=stop_event,
                manage_connection=manage_connection,
                login=MT5_LOGIN,
                password=MT5_PASSWORD,
                server=MT5_SERVER,
            )
        except Exception as exc:
            print(f"[MT5 Bridge] stopped: {exc}")

    thread = threading.Thread(target=_runner, name="MT5Bridge", daemon=True)
    thread.start()
    print(
        f"[MT5 Bridge] started (symbols={MT5_BRIDGE_SYMBOLS}, tfs={MT5_BRIDGE_TIMEFRAMES}, interval={MT5_BRIDGE_INTERVAL}s)"
    )
    return stop_event, thread


def _acquire_pid_lock(path: Path) -> bool:
    """Return True if this is the only running instance, False if another is alive."""
    if path.exists():
        try:
            old_pid = int(path.read_text().strip())
            import psutil
            if psutil.pid_exists(old_pid):
                proc = psutil.Process(old_pid)
                if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                    print(f"[PID Lock] Another instance is already running (PID {old_pid}). Exiting.")
                    return False
        except Exception:
            pass  # stale lock — overwrite it
    path.write_text(str(os.getpid()))
    return True


def _release_pid_lock(path: Path) -> None:
    try:
        if path.exists() and path.read_text().strip() == str(os.getpid()):
            path.unlink()
    except Exception:
        pass


if __name__ == "__main__":
    _LOCK_PATH = AI_DATA_DIR / "bot.pid"
    if not _acquire_pid_lock(_LOCK_PATH):
        raise SystemExit(1)

    bridge_stop: Optional[threading.Event] = None
    bridge_thread: Optional[threading.Thread] = None
    try:
        core = Core()
        if _should_start_bridge():
            manage_conn = not MT5_EXECUTION_ENABLED
            bridge_stop, bridge_thread = _start_bridge_thread(manage_conn)
        bot = TelegramBot(TELEGRAM_TOKEN, core)
        bot.run()
    except KeyboardInterrupt:
        pass
    finally:
        if bridge_stop:
            bridge_stop.set()
        if bridge_thread:
            bridge_thread.join(timeout=5)
        _release_pid_lock(_LOCK_PATH)
