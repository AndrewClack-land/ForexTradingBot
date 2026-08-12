from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import core.m1.store as store_module
from core.m1.ai_live import AILive, primary_idea_trigger_kind
from core.m1.config import AIConfig
from core.m1.store import TradeStore


def _config(*, trigger_calibration_enabled=True) -> AIConfig:
    return AIConfig(
        enabled=True,
        trigger_calibration_enabled=trigger_calibration_enabled,
        min_closed_per_symbol=2,
        min_closed_per_trigger=2,
        min_p_tp=0.2,
        min_edge_above_be=0.0,
        min_rr=1.3,
        alpha=1.0,
        beta=1.0,
        db_filename="ai_stats_test.db",
    )


def _store(tmp_path, monkeypatch) -> TradeStore:
    monkeypatch.setattr(store_module, "AI_DATA_DIR", tmp_path)
    return TradeStore(_config())


def test_schema_migration_preserves_legacy_symbol_stats(tmp_path, monkeypatch):
    database = tmp_path / "ai_stats_test.db"
    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TABLE symbol_stats (
            symbol TEXT PRIMARY KEY,
            tp INTEGER NOT NULL DEFAULT 0,
            sl INTEGER NOT NULL DEFAULT 0,
            rr_sum REAL NOT NULL DEFAULT 0.0,
            rr_n INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    connection.execute(
        "INSERT INTO symbol_stats VALUES('EURUSD', 3, 1, 6.0, 4)"
    )
    connection.commit()
    connection.close()

    store = _store(tmp_path, monkeypatch)
    try:
        assert store.get_symbol_stats("eurusd")["closed"] == 4
        assert store.get_symbol_trigger_stats(
            "EURUSD",
            "rejection_block_1h",
        )["closed"] == 0
    finally:
        store.close()


def test_updates_aggregate_and_trigger_bucket_atomically(
    tmp_path,
    monkeypatch,
):
    store = _store(tmp_path, monkeypatch)
    try:
        store.update_on_close(
            "eurusd",
            "TP",
            rr_numeric=1.5,
            trigger_kind="rejection_block_1h",
        )
        store.update_on_close(
            "EURUSD",
            "SL",
            rr_numeric=1.5,
            trigger_kind="fxpro_cluster_rejection_15m",
        )
        store.update_on_close("EURUSD", "TP", rr_numeric=1.5)

        assert store.get_symbol_stats("EURUSD")["closed"] == 3
        assert store.get_symbol_trigger_stats(
            "EURUSD",
            "rejection_block_1h",
        )["tp"] == 1
        assert store.get_symbol_trigger_stats(
            "EURUSD",
            "fxpro_cluster_rejection_15m",
        )["sl"] == 1
    finally:
        store.close()


def test_ai_gate_never_uses_another_trigger_family_as_fallback(
    tmp_path,
    monkeypatch,
):
    store = _store(tmp_path, monkeypatch)
    try:
        for _ in range(2):
            store.update_on_close(
                "EURUSD",
                "TP",
                rr_numeric=1.5,
                trigger_kind="rejection_block_1h",
            )
            store.update_on_close(
                "EURUSD",
                "SL",
                rr_numeric=1.5,
                trigger_kind="fxpro_cluster_rejection_15m",
            )
        ai = AILive(_config(), store, strategy=None)

        rb = ai.on_signal(
            "EURUSD",
            {
                "signal": "ENTER",
                "rr_numeric": 1.5,
                "trigger_kind": "rejection_block_1h",
            },
            {},
            {},
        )
        cluster = ai.on_signal(
            "EURUSD",
            {
                "signal": "ENTER",
                "rr_numeric": 1.5,
                "trigger_kind": "fxpro_cluster_rejection_15m",
            },
            {},
            {},
        )
        unseen = ai.on_signal(
            "EURUSD",
            {
                "signal": "ENTER",
                "rr_numeric": 1.5,
                "trigger_kind": "order_block_1h",
            },
            {},
            {},
        )

        assert rb["signal"] == "ENTER"
        assert cluster["signal"] == "AI_REJECT"
        assert cluster["ai_calibration_key"] == (
            "EURUSDxfxpro_cluster_rejection_15m"
        )
        assert unseen["signal"] == "ENTER"
        assert unseen["ai_trigger_stats_closed"] == 0
        assert unseen["ai_symbol_stats_closed"] == 4
    finally:
        store.close()


def test_exit_is_attributed_to_primary_idea_trigger_not_addon(
    tmp_path,
    monkeypatch,
):
    store = _store(tmp_path, monkeypatch)
    try:
        ai = AILive(_config(), store, strategy=None)
        trade = SimpleNamespace(
            entry=1.10,
            stop=1.09,
            tp_prices=[1.11, 1.12, 1.13],
            idea_trigger_signatures=[
                "LONG|rejection_block_1h|z1",
                "LONG|order_block_1h|z2",
            ],
        )

        assert primary_idea_trigger_kind(trade) == "rejection_block_1h"
        ai.on_signal(
            "EURUSD",
            {"signal": "EXIT_TP"},
            {},
            {"EURUSD": trade},
        )

        assert store.get_symbol_trigger_stats(
            "EURUSD",
            "rejection_block_1h",
        )["tp"] == 1
        assert store.get_symbol_trigger_stats(
            "EURUSD",
            "order_block_1h",
        )["closed"] == 0
    finally:
        store.close()


def test_trigger_calibration_is_diagnostic_only_when_disabled(
    tmp_path,
    monkeypatch,
):
    cfg = _config(trigger_calibration_enabled=False)
    monkeypatch.setattr(store_module, "AI_DATA_DIR", tmp_path)
    store = TradeStore(cfg)
    try:
        for _ in range(2):
            store.update_on_close(
                "EURUSD",
                "SL",
                rr_numeric=1.5,
                trigger_kind="fxpro_cluster_rejection_15m",
            )
        ai = AILive(cfg, store, strategy=None)
        signal = ai.on_signal(
            "EURUSD",
            {
                "signal": "ENTER",
                "rr_numeric": 1.5,
                "trigger_kind": "fxpro_cluster_rejection_15m",
            },
            {},
            {},
        )
        assert signal["signal"] == "ENTER"
        assert signal["ai_trigger_calibration_enabled"] is False

        trade = SimpleNamespace(
            entry=1.10,
            stop=1.09,
            tp_prices=[1.11, 1.12, 1.13],
            idea_trigger_signatures=[
                "LONG|rejection_block_1h|z1",
            ],
        )
        ai.on_signal(
            "EURUSD",
            {"signal": "EXIT_TP"},
            {},
            {"EURUSD": trade},
        )
        assert store.get_symbol_stats("EURUSD")["tp"] == 1
        assert store.get_symbol_trigger_stats(
            "EURUSD",
            "rejection_block_1h",
        )["closed"] == 0
    finally:
        store.close()
