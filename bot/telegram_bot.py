from __future__ import annotations

import asyncio
import time
import traceback
from typing import Any, Dict, Optional, List

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from telegram.request import HTTPXRequest
from telegram.error import NetworkError, TimedOut, RetryAfter

from config import (
    AI_DATA_DIR,
    TELEGRAM_CHANNEL_ID,
    TELEGRAM_ADMIN_IDS,
    SYMBOL_DECIMALS,
    POST_STARTUP_REPORT,
    REPORT_DEFAULT_LIMIT,
)
from core.profiler import TickProfiler
from core.trade_journal import TradeJournal


def _guess_decimals(symbol: str) -> int:
    if symbol in SYMBOL_DECIMALS:
        return int(SYMBOL_DECIMALS[symbol])
    if symbol.endswith("JPY"):
        return 3
    if symbol.startswith("XAU") or symbol == "GOLD":
        return 2
    if symbol.startswith("XAG"):
        return 3
    return 5


def _fmt_price(symbol: str, price: float) -> str:
    d = _guess_decimals(symbol)
    return f"{float(price):.{d}f}"


def _side_to_text(side: str) -> tuple[str, str]:
    s = (side or "").upper()
    if s == "LONG":
        return "⬆️", "Покупка"
    return "⬇️", "Продажа"


class TelegramBot:
    def __init__(self, token: str, core: Any, journal: Optional[TradeJournal] = None):
        self.token = token
        self.core = core
        self.channel_id = int(TELEGRAM_CHANNEL_ID)

        self.journal = journal or TradeJournal(
            db_path=str(AI_DATA_DIR / "trades.db"),
            csv_path=str(AI_DATA_DIR / "trades.csv"),
            parquet_path=str(AI_DATA_DIR / "trades.parquet"),
            export_on_each_event=True,
        )

        # if you had "skipped max_instances" often — set 90/120
        self.poll_sec = 60

        self._tick_lock = asyncio.Lock()

        # send retries
        self._send_retries = 6
        self._send_base_delay = 1.5  # seconds

        self.profiler = TickProfiler()

    # ----------------- formatting -----------------
    def _format_signal(self, symbol: str, sig: Dict[str, Any]) -> str:
        side = str(sig.get("side", ""))
        arrow, action = _side_to_text(side)

        # entry can be range (tuple/list) or single float
        entry_val = sig.get("entry_price")
        entry_text = ""
        if isinstance(entry_val, (list, tuple)) and len(entry_val) >= 2:
            entry_text = f"{_fmt_price(symbol, float(entry_val[0]))}–{_fmt_price(symbol, float(entry_val[1]))}"
        else:
            entry_text = _fmt_price(symbol, float(entry_val))

        stop = _fmt_price(symbol, float(sig["stop_price"]))

        tps = sig.get("tp_prices") or [sig.get("tp_price")]
        tps = [float(x) for x in tps if x is not None]
        # show up to 4 tps if provided
        while len(tps) < 3:
            tps.append(tps[-1] if tps else float(sig.get("tp_price", sig["entry_price"])))
        tps = tps[:4] if len(tps) >= 4 else tps[:3]

        lines = [
            f"{symbol} {arrow} {action}",
            "",
            f"{entry_text}",
            "",
        ]
        for i, tp in enumerate(tps, start=1):
            lines.append(f"✅Тп {i}: {_fmt_price(symbol, tp)}")
        lines += ["", f"Стоп лос: {stop}"]
        return "\n".join(lines)

    # ----------------- telegram helpers -----------------
    async def _safe_send_message(self, app, **kwargs):
        """
        Robust sender with retry/backoff for temporary network problems.
        """
        last_exc: Optional[Exception] = None
        start = time.perf_counter() if self.profiler.enabled else None
        try:
            for attempt in range(1, self._send_retries + 1):
                try:
                    return await app.bot.send_message(**kwargs)
                except RetryAfter as e:
                    # Telegram rate limit
                    wait_s = float(getattr(e, "retry_after", 2.0))
                    await asyncio.sleep(min(10.0, wait_s + 0.5))
                    last_exc = e
                except (NetworkError, TimedOut) as e:
                    # includes httpx.ReadError wrapped as NetworkError
                    delay = min(15.0, self._send_base_delay * (2 ** (attempt - 1)))
                    print(f"[TelegramBot] send_message retry {attempt}/{self._send_retries} after {delay:.1f}s | {e}")
                    await asyncio.sleep(delay)
                    last_exc = e
                except Exception as e:
                    # non-retryable or unknown
                    last_exc = e
                    break
        finally:
            if start is not None:
                self.profiler.add("telegram_send", time.perf_counter() - start)

        if last_exc:
            print("[TelegramBot] send_message failed окончательно:")
            traceback.print_exc()
        return None

    async def _reply(self, app, symbol: str, text: str, *, meta: Optional[Dict[str, Any]] = None) -> None:
        chat_id = self.channel_id
        reply_to: Optional[int] = None

        if meta is not None:
            chat_id = int(meta.get("telegram_chat_id", self.channel_id) or self.channel_id)
            reply_id = meta.get("telegram_message_id")
            reply_to = int(reply_id) if reply_id is not None else None
        else:
            tr = self.core.active_trades.get(symbol)
            if tr is not None:
                chat_id = int(getattr(tr, "telegram_chat_id", self.channel_id) or self.channel_id)
                reply_id = getattr(tr, "telegram_message_id", None)
                reply_to = int(reply_id) if reply_id is not None else None

        if reply_to is None:
            await self._safe_send_message(
                app,
                chat_id=chat_id,
                text=text,
                disable_web_page_preview=True,
            )
            return

        await self._safe_send_message(
            app,
            chat_id=chat_id,
            text=text,
            reply_to_message_id=reply_to,
            disable_web_page_preview=True,
        )

    async def _handle_events(self, app, symbol: str, sig: Dict[str, Any], *,
                             meta: Optional[Dict[str, Any]] = None) -> None:
        events: List[Dict[str, Any]] = sig.get("events") or []
        if not events:
            return

        for e in events:
            et = e.get("type")
            if et == "TP":
                idx = int(e.get("tp_index", 0))
                tp_price = e.get("tp_price")
                if tp_price is None:
                    continue
                await self._reply(app, symbol, f"✅Тп {idx}: {_fmt_price(symbol, float(tp_price))}", meta=meta)
            elif et == "BE":
                price = e.get("price")
                if price is None:
                    continue
                await self._reply(
                    app, symbol,
                    f"🔒 Стоп перенесён в безубыток: {_fmt_price(symbol, float(price))}",
                    meta=meta,
                )

    async def _handle_exit(self, app, symbol: str, sig: Dict[str, Any]) -> None:
        st = sig.get("signal")

        if st == "EXIT_SL":
            px = sig.get("exit_price")
            if px is not None:
                await self._reply(
                    app,
                    symbol,
                    f"🛑 Стоп лос: {_fmt_price(symbol, float(px))}",
                    meta=sig,
                )

            try:
                with self.profiler.section("journal_ingest"):
                    self.journal.ingest_signal(
                        symbol=symbol,
                        sig=sig,
                        telegram_chat_id=self.channel_id,
                        telegram_message_id=None,
                        features=None,
                    )
            except Exception:
                traceback.print_exc()

        elif st == "EXIT_TP":
            await self._handle_events(app, symbol, sig, meta=sig)

            px = sig.get("exit_price")
            if px is not None:
                await self._reply(
                    app,
                    symbol,
                    f"✅ Тейк-профит: {_fmt_price(symbol, float(px))}",
                    meta=sig,
                )

            try:
                with self.profiler.section("journal_ingest"):
                    self.journal.ingest_signal(
                        symbol=symbol,
                        sig=sig,
                        telegram_chat_id=self.channel_id,
                        telegram_message_id=None,
                        features=None,
                    )
            except Exception:
                traceback.print_exc()

    async def _post_enter(self, app, symbol: str, sig: Dict[str, Any]) -> None:
        text = self._format_signal(symbol, sig)

        msg = await self._safe_send_message(
            app,
            chat_id=self.channel_id,
            text=text,
            disable_web_page_preview=True,
        )
        if msg is None:
            return

        # bind for reply + persistence
        tr = self.core.active_trades.get(symbol)
        if tr is not None:
            tr.telegram_chat_id = self.channel_id
            tr.telegram_message_id = msg.message_id

        try:
            with self.profiler.section("journal_update"):
                self.journal.update_open_trade_state(
                    symbol,
                    telegram_chat_id_open=self.channel_id,
                    telegram_message_id_open=msg.message_id,
                )
        except Exception:
            pass

        try:
            with self.profiler.section("journal_ingest"):
                self.journal.ingest_signal(
                    symbol=symbol,
                    sig=sig,
                    telegram_chat_id=self.channel_id,
                    telegram_message_id=msg.message_id,
                    features=None,
                )
        except Exception:
            traceback.print_exc()

    # ----------------- main tick loop -----------------
    async def _tick(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        if self._tick_lock.locked():
            return

        async with self._tick_lock:
            try:
                with self.profiler.section("core_get_signals"):
                    signals = self.core.get_signals()
            except Exception:
                traceback.print_exc()
                return

            app = context.application

            for symbol, sig in (signals or {}).items():
                try:
                    st = sig.get("signal")

                    if st == "ENTER":
                        with self.profiler.section("post_enter"):
                            await self._post_enter(app, symbol, sig)
                        continue

                    if st == "HOLD" and sig.get("events"):
                        with self.profiler.section("handle_events"):
                            await self._handle_events(app, symbol, sig)
                        continue

                    if st in ("EXIT_SL", "EXIT_TP"):
                        with self.profiler.section("handle_exit"):
                            await self._handle_exit(app, symbol, sig)
                        continue

                    if st == "EXIT_BROKER":
                        px = sig.get("exit_price")
                        px_str = f" @ {_fmt_price(symbol, float(px))}" if px is not None else ""
                        await self._reply(
                            app,
                            symbol,
                            f"♻️ {symbol}: позиция закрыта брокером{px_str}",
                            meta=sig,
                        )
                        try:
                            with self.profiler.section("journal_ingest"):
                                self.journal.ingest_signal(
                                    symbol=symbol,
                                    sig=sig,
                                    telegram_chat_id=self.channel_id,
                                    telegram_message_id=None,
                                    features=None,
                                )
                        except Exception:
                            traceback.print_exc()
                        continue

                    if st == "EXIT_TIME":
                        px = sig.get("exit_price")
                        px_str = f" @ {_fmt_price(symbol, float(px))}" if px is not None else ""
                        await self._reply(
                            app,
                            symbol,
                            f"⏱ {symbol}: позиция закрыта по времени{px_str}",
                            meta=sig,
                        )
                        try:
                            with self.profiler.section("journal_ingest"):
                                self.journal.ingest_signal(
                                    symbol=symbol,
                                    sig=sig,
                                    telegram_chat_id=self.channel_id,
                                    telegram_message_id=None,
                                    features=None,
                                )
                        except Exception:
                            traceback.print_exc()
                        continue

                except Exception:
                    traceback.print_exc()

            self.profiler.dump(prefix="[Profiler:bot]")

    # ----------------- commands -----------------
    def _authorized(self, update: Update) -> bool:
        """Commands expose open positions/history — restrict to the channel and
        explicitly whitelisted ids (TELEGRAM_ADMIN_IDS). Everyone else is ignored."""
        chat = update.effective_chat
        user = update.effective_user
        if chat is not None and int(chat.id) == self.channel_id:
            return True
        if user is not None and int(user.id) in TELEGRAM_ADMIN_IDS:
            return True
        if chat is not None and int(chat.id) in TELEGRAM_ADMIN_IDS:
            return True
        print(
            f"[TelegramBot] unauthorized command ignored: "
            f"chat={getattr(chat, 'id', None)} user={getattr(user, 'id', None)}"
        )
        return False

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        open_trades = len(getattr(self.core, "active_trades", {}) or {})
        await update.message.reply_text(f"Open trades: {open_trades}")

    async def cmd_open(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        try:
            rows = self.journal.load_open_trades()
        except Exception:
            await update.message.reply_text("DB error while loading open trades (see logs).")
            traceback.print_exc()
            return

        if not rows:
            await update.message.reply_text("Open trades: 0")
            return

        lines = ["Open trades:\n"]
        for r in rows:
            sym = r["symbol"]
            arrow, act = _side_to_text(r["side"])
            entry = _fmt_price(sym, r["entry"])
            stop = _fmt_price(sym, r["stop_current"])
            tps = r.get("tp_prices") or []
            tps = [float(x) for x in tps][:3]
            while len(tps) < 3:
                tps.append(tps[-1] if tps else float(r["entry"]))
            tp1, tp2, tp3 = (_fmt_price(sym, tps[0]), _fmt_price(sym, tps[1]), _fmt_price(sym, tps[2]))
            lines.append(
                f"{sym} {arrow} {act} | entry {entry} | stop {stop} | TP1 {tp1} TP2 {tp2} TP3 {tp3} | hit={r.get('tp_hit',0)}"
            )

        msg = "\n".join(lines)
        if len(msg) <= 3900:
            await update.message.reply_text(msg)
        else:
            chunk = ""
            for line in lines:
                if len(chunk) + len(line) + 1 > 3900:
                    await update.message.reply_text(chunk)
                    chunk = line
                else:
                    chunk = chunk + ("\n" if chunk else "") + line
            if chunk:
                await update.message.reply_text(chunk)

    async def cmd_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        limit = REPORT_DEFAULT_LIMIT
        if context.args:
            try:
                limit = int(context.args[0])
            except Exception:
                limit = REPORT_DEFAULT_LIMIT

        try:
            rows = self.journal.recent_trades(limit=limit)
        except Exception:
            await update.message.reply_text("DB error while loading report (see logs).")
            traceback.print_exc()
            return

        if not rows:
            await update.message.reply_text("No trades in DB yet.")
            return

        lines = [f"Last {len(rows)} trades (newest first):\n"]
        for r in rows:
            sym = r["symbol"]
            arrow, act = _side_to_text(r["side"])
            entry = _fmt_price(sym, r["entry"])
            stop_cur = r.get("stop_current") if r.get("stop_current") is not None else r.get("stop")
            stop = _fmt_price(sym, float(stop_cur))
            outcome = r.get("outcome") or "OPEN"
            exit_px = f" exit={_fmt_price(sym, r['exit'])}" if r.get("exit") is not None else ""
            lines.append(
                f"#{r['id']} {sym} {arrow} {act} | entry {entry} stop {stop} | {outcome}{exit_px} | tp_hit={r.get('tp_hit',0)}"
            )

        msg = "\n".join(lines)
        if len(msg) <= 3900:
            await update.message.reply_text(msg)
        else:
            chunk = ""
            for line in lines:
                if len(chunk) + len(line) + 1 > 3900:
                    await update.message.reply_text(chunk)
                    chunk = line
                else:
                    chunk = chunk + ("\n" if chunk else "") + line
            if chunk:
                await update.message.reply_text(chunk)

    async def cmd_universe(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        uni = list(getattr(self.core, "universe", {}).keys())
        await update.message.reply_text("Universe:\n" + ", ".join(uni))

    # ----------------- app error handler -----------------
    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        # prevents silent failures
        print("[TelegramBot] Application error:", repr(context.error))
        traceback.print_exception(type(context.error), context.error, context.error.__traceback__)

    # ----------------- run -----------------
    async def _startup_report(self, app) -> None:
        if not POST_STARTUP_REPORT:
            return
        try:
            rows = self.journal.load_open_trades()
        except Exception:
            traceback.print_exc()
            return
        if not rows:
            return
        syms = ", ".join([r["symbol"] for r in rows])
        await self._safe_send_message(
            app,
            chat_id=self.channel_id,
            text=f"♻️ Bot restarted. Restored open trades: {len(rows)} ({syms})",
        )

    def run(self) -> None:
        # slightly higher timeouts to reduce httpx.ReadError chance
        request = HTTPXRequest(connect_timeout=30, read_timeout=30, write_timeout=30, pool_timeout=30)
        app = ApplicationBuilder().token(self.token).request(request).build()

        app.add_handler(CommandHandler("status", self.cmd_status))
        app.add_handler(CommandHandler("open", self.cmd_open))
        app.add_handler(CommandHandler("report", self.cmd_report))
        app.add_handler(CommandHandler("universe", self.cmd_universe))

        # ✅ global error handler (no more "No error handlers are registered")
        app.add_error_handler(self._on_error)

        app.job_queue.run_repeating(
            self._tick,
            interval=self.poll_sec,
            first=3,
            job_kwargs={"max_instances": 1, "coalesce": True, "misfire_grace_time": 30},
        )

        async def _post_startup(_: Any) -> None:
            await self._startup_report(app)

        app.post_init = _post_startup  # type: ignore

        if not TELEGRAM_ADMIN_IDS:
            print(
                "[TelegramBot] WARNING: TELEGRAM_ADMIN_IDS is empty — bot commands "
                "(/status /open /report /universe) are only accepted from the channel. "
                "Add your user id to TELEGRAM_ADMIN_IDS in .env to use them in DM."
            )
        print(f"[TelegramBot] posting to channel_id={self.channel_id} | poll_sec={self.poll_sec}")
        app.run_polling(close_loop=False)
