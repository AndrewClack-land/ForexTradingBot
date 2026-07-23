from __future__ import annotations

import csv
import sqlite3

import pytest

from core.trade_journal import TradeJournal


def _enter(
    journal: TradeJournal,
    symbol: str,
    *,
    entry: float = 1.0,
    execution: dict | None = None,
) -> None:
    journal.ingest_signal(
        symbol,
        {
            "signal": "ENTER",
            "side": "LONG",
            "entry_price": entry,
            "planned_entry_price": entry - 0.001,
            "stop_price": entry - 0.01,
            "tp_price": entry + 0.03,
            "tp_prices": [entry + 0.01, entry + 0.02, entry + 0.03],
            "rr_numeric": 3.0,
            "weighted_rr_numeric": 1.7,
            "rr": "1:3.00",
            "tf": "15M",
            "execution": execution or {"mode": "monitor", "volume": 0.3},
        },
    )


def _close(
    journal: TradeJournal,
    symbol: str,
    *,
    signal: str = "EXIT_BROKER",
    outcome: str | None = None,
    realized_net: float | None = None,
    pnl_complete: bool = False,
) -> None:
    payload = {
        "signal": signal,
        "exit_price": 1.01,
        "pnl_complete": pnl_complete,
    }
    if outcome is not None:
        payload["outcome"] = outcome
    if realized_net is not None:
        payload["realized_net"] = realized_net
    journal.ingest_signal(symbol, payload)


def test_setup_metrics_count_setups_not_split_legs(tmp_path):
    journal = TradeJournal(
        str(tmp_path / "trades.db"),
        export_on_each_event=False,
        strategy_version="narrative-1r2r3r-v1",
        experiment_id="wr-20260716",
        experiment_variant="live",
        deployment_id="deploy-a",
    )
    try:
        _enter(
            journal,
            "EURUSD",
            execution={
                "mode": "split",
                "legs": [
                    {"position_id": 11, "volume": 0.15},
                    {"position_id": 12, "volume": 0.09},
                    {"position_id": 13, "volume": 0.06},
                ],
            },
        )
        journal.update_open_trade_state("EURUSD", tp_hit=1, moved_to_be=True)
        _close(
            journal,
            "EURUSD",
            outcome="TP",  # net P&L must override a misleading terminal label
            realized_net=-11.0,
            pnl_complete=True,
        )

        _enter(journal, "GBPUSD")
        _close(journal, "GBPUSD", outcome="SL", realized_net=20.0, pnl_complete=True)

        _enter(journal, "USDJPY")
        _close(journal, "USDJPY", outcome="TP", realized_net=0.0, pnl_complete=True)

        _enter(journal, "GOLD")
        _close(journal, "GOLD", signal="EXIT_TIME")

        _enter(journal, "USDCAD")
        _close(journal, "USDCAD", signal="EXIT_TP")

        _enter(journal, "AUDUSD")  # remains open

        metrics = journal.setup_metrics()

        assert metrics["total_setups"] == 6
        assert metrics["open_setups"] == 1
        assert metrics["closed_setups"] == 5
        assert metrics["wins"] == 2
        assert metrics["losses"] == 1
        assert metrics["breakeven"] == 1
        assert metrics["unknown"] == 1
        assert metrics["evaluated_setups"] == 3
        assert metrics["win_rate"] == pytest.approx(2 / 3)
        assert metrics["net_known_setups"] == 3
        assert metrics["pnl_coverage"] == pytest.approx(3 / 5)
        assert metrics["net_realized"] == pytest.approx(9.0)
        assert metrics["profit_factor"] == pytest.approx(20.0 / 11.0)
        assert metrics["tp1_hits"] == 1
        assert metrics["be_moves"] == 1

        wr, evaluated, wins, losses = journal.winrate()
        assert (evaluated, wins, losses) == (3, 2, 1)
        assert wr == pytest.approx(2 / 3)

        eurusd = next(row for row in journal.recent_trades(20) if row["symbol"] == "EURUSD")
        assert eurusd["result"] == "LOSS"
        assert eurusd["execution_mode"] == "split"
        assert eurusd["total_volume"] == pytest.approx(0.3)
        assert eurusd["planned_entry"] == pytest.approx(0.999)
        assert eurusd["weighted_rr_planned"] == pytest.approx(1.7)
        assert eurusd["strategy_version"] == "narrative-1r2r3r-v1"
        assert eurusd["experiment_id"] == "wr-20260716"
        assert eurusd["setup_id"]

        # A repeated terminal event is idempotent: there is still one setup row.
        _close(journal, "EURUSD", outcome="TP", realized_net=999.0, pnl_complete=True)
        assert journal.setup_metrics()["total_setups"] == 6
        assert next(row for row in journal.recent_trades(20) if row["symbol"] == "EURUSD")[
            "realized_net"
        ] == pytest.approx(-11.0)
    finally:
        journal.close()


def test_legacy_schema_migrates_without_relabeling_old_rows(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_open TEXT NOT NULL,
            ts_close TEXT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry REAL NOT NULL,
            stop REAL NOT NULL,
            tp REAL NOT NULL,
            exit REAL,
            outcome TEXT,
            rr REAL
        );
        """
    )
    conn.execute(
        """
        INSERT INTO trades
            (id, ts_open, ts_close, symbol, side, entry, stop, tp, exit, outcome, rr)
        VALUES
            (7, '2026-01-01T00:00:00+00:00', '2026-01-01T01:00:00+00:00',
             'EURUSD', 'LONG', 1.0, 0.99, 1.03, 1.03, 'TP', 3.0);
        """
    )
    conn.commit()
    conn.close()

    journal = TradeJournal(
        str(db_path),
        export_on_each_event=False,
        strategy_version="must-not-backfill",
    )
    try:
        row = journal._conn.execute(
            "SELECT id, setup_id, strategy_version, realized_net, pnl_complete FROM trades"
        ).fetchone()
        assert row["id"] == 7
        assert row["setup_id"] is None
        assert row["strategy_version"] is None
        assert row["realized_net"] is None
        assert row["pnl_complete"] == 0
        assert journal._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

        metrics = journal.setup_metrics()
        assert metrics["wins"] == 1
        assert metrics["net_known_setups"] == 0
        assert metrics["pnl_coverage"] == 0.0
    finally:
        journal.close()


def test_csv_export_appends_setup_audit_columns(tmp_path):
    csv_path = tmp_path / "trades.csv"
    journal = TradeJournal(
        str(tmp_path / "trades.db"),
        csv_path=str(csv_path),
        export_on_each_event=False,
        strategy_version="v-test",
    )
    try:
        _enter(journal, "EURUSD")
        _close(journal, "EURUSD", outcome="TP", realized_net=5.0, pnl_complete=True)
        journal.export_csv()
    finally:
        journal.close()

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["strategy_version"] == "v-test"
    assert rows[0]["realized_net"] == "5.0"
    assert rows[0]["pnl_complete"] == "1"
    assert rows[0]["setup_id"]


def test_stable_setup_id_prevents_stale_row_from_hiding_new_execution(tmp_path):
    journal = TradeJournal(str(tmp_path / "trades.db"), export_on_each_event=False)
    try:
        # Simulate a legacy row left open after a state mismatch.
        _enter(journal, "EURUSD")
        new_signal = {
            "signal": "ENTER",
            "setup_id": "setup-new-1",
            "side": "LONG",
            "entry_price": 1.1,
            "stop_price": 1.09,
            "tp_price": 1.13,
            "tp_prices": [1.11, 1.12, 1.13],
        }
        journal.ingest_signal("EURUSD", new_signal)

        rows = journal._conn.execute(
            "SELECT id, setup_id, ts_close FROM trades ORDER BY id"
        ).fetchall()
        assert len(rows) == 2
        assert rows[-1]["setup_id"] == "setup-new-1"

        _close(journal, "EURUSD", signal="EXIT_TP")
        metrics = journal.setup_metrics()
        assert metrics["total_setups"] == 2
        assert metrics["open_setups"] == 1
        assert metrics["closed_setups"] == 1

        # Replayed ENTER for the same setup stays idempotent even after close.
        journal.ingest_signal("EURUSD", new_signal)
        assert journal.setup_metrics()["total_setups"] == 2
    finally:
        journal.close()
