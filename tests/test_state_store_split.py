from __future__ import annotations

import json

from core.state_store import load_active_trades, save_active_trades
from core.strategy_narrative import ActiveTrade


def _trade() -> ActiveTrade:
    trade = ActiveTrade(
        side="LONG",
        entry=1.0,
        stop=0.9,
        tp_prices=[1.1, 1.2, 1.3],
        tf="15m",
        narrative="state test",
        symbol="EURUSD",
    )
    trade.split_position_ids = [22, 33]
    trade.split_legs = {
        11: {
            "tp_index": 1,
            "tp": 1.1,
            "volume": 0.2,
            "status": "closed",
            "close_reason": "TP",
        },
        22: {"tp_index": 2, "tp": 1.2, "volume": 0.1, "status": "open"},
        33: {"tp_index": 3, "tp": 1.3, "volume": 0.1, "status": "open"},
    }
    trade.tp_hit = 1
    return trade


def test_split_leg_mapping_round_trips_with_integer_tickets(tmp_path):
    path = tmp_path / "active.json"
    save_active_trades({"EURUSD": _trade()}, path)

    restored = load_active_trades(path)["EURUSD"]

    assert set(restored.split_legs) == {11, 22, 33}
    assert restored.split_legs[11]["close_reason"] == "TP"
    assert restored.split_legs[22]["tp_index"] == 2
    assert restored.split_position_ids == [22, 33]


def test_legacy_split_ids_backfill_from_tp_hit(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "EURUSD": {
                    "side": "LONG",
                    "entry": 1.0,
                    "stop": 0.9,
                    "tp_prices": [1.1, 1.2, 1.3],
                    "tf": "15m",
                    "narrative": "legacy",
                    "symbol": "EURUSD",
                    "tp_hit": 1,
                    "volume_per_tp": [0.2, 0.1, 0.1],
                    "split_position_ids": [222, 333],
                }
            }
        ),
        encoding="utf-8",
    )

    restored = load_active_trades(path)["EURUSD"]

    assert restored.split_legs[222] == {
        "tp_index": 2,
        "tp": 1.2,
        "volume": 0.1,
        "status": "open",
        "legacy_inferred": True,
    }
    assert restored.split_legs[333]["tp_index"] == 3
