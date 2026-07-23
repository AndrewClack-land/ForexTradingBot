from __future__ import annotations

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

from bot.telegram_bot import TelegramBot
from core.trade_journal import TradeJournal


class _Profiler:
    def section(self, _name):
        return nullcontext()


def test_failed_telegram_send_does_not_drop_executed_setup(tmp_path):
    journal = TradeJournal(str(tmp_path / "trades.db"), export_on_each_event=False)
    bot = TelegramBot.__new__(TelegramBot)
    bot.journal = journal
    bot.profiler = _Profiler()
    bot.channel_id = -100123
    bot.core = SimpleNamespace(active_trades={})
    bot._format_signal = lambda symbol, sig: f"{symbol} ENTER"

    async def failed_send(*args, **kwargs):
        return None

    bot._safe_send_message = failed_send
    signal = {
        "signal": "ENTER",
        "side": "LONG",
        "entry_price": 1.0,
        "stop_price": 0.99,
        "tp_price": 1.03,
        "tp_prices": [1.01, 1.02, 1.03],
        "tf": "15M",
    }

    try:
        asyncio.run(bot._post_enter(object(), "EURUSD", signal))
        rows = journal.recent_trades(10)
        assert len(rows) == 1
        assert rows[0]["symbol"] == "EURUSD"
        assert rows[0]["telegram_message_id_open"] is None

        # A retried notification cannot duplicate the setup row.
        asyncio.run(bot._post_enter(object(), "EURUSD", signal))
        assert journal.setup_metrics()["total_setups"] == 1
    finally:
        journal.close()


def test_report_labels_setup_metrics_and_pnl_coverage(tmp_path):
    journal = TradeJournal(str(tmp_path / "trades.db"), export_on_each_event=False)
    journal.ingest_signal(
        "EURUSD",
        {
            "signal": "ENTER",
            "side": "LONG",
            "entry_price": 1.0,
            "stop_price": 0.99,
            "tp_price": 1.03,
            "tp_prices": [1.01, 1.02, 1.03],
        },
    )
    journal.ingest_signal(
        "EURUSD",
        {
            "signal": "EXIT_BROKER",
            "exit_price": 1.02,
            "outcome": "TP",
            "realized_net": 12.5,
            "pnl_complete": True,
        },
    )

    replies = []

    class _Message:
        async def reply_text(self, value):
            replies.append(value)

    bot = TelegramBot.__new__(TelegramBot)
    bot.journal = journal
    bot._authorized = lambda update: True
    update = SimpleNamespace(message=_Message())
    context = SimpleNamespace(args=[])

    try:
        asyncio.run(bot.cmd_report(update, context))
    finally:
        journal.close()

    rendered = "\n".join(replies)
    assert "one setup, not MT5 legs" in rendered
    assert "W/L/BE/?=1/0/0/0" in rendered
    assert "WR=100.0%" in rendered
    assert "P&L coverage=1/1" in rendered
    assert "net=+12.50" in rendered
