from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

import main
from core.shadow_tick_watcher import QuoteTick


class _Watcher:
    enabled = True

    def __init__(self) -> None:
        self.spec = None
        self.activation_quote = None
        self.scans = []

    def register_plan(self, spec, *, activation_quote):
        self.spec = spec
        self.activation_quote = activation_quote
        return True

    def record_scan(self, plan_id, quote, **fields):
        self.scans.append((plan_id, quote, dict(fields)))
        return True


def test_candidate_activation_uses_utc_observation_not_server_quote_clock(
    monkeypatch,
) -> None:
    core = main.Core.__new__(main.Core)
    watcher = _Watcher()
    core.shadow_tick_watcher = watcher
    core.universe = {"EURUSD": "EURUSD"}
    core.global_context = {"session": "LONDON"}
    source_shift_msc = 3 * 60 * 60 * 1000
    source_quote = QuoteTick(
        time_msc=int(datetime.now(timezone.utc).timestamp() * 1000)
        + source_shift_msc,
        bid=1.1000,
        ask=1.1002,
    )
    core._capture_quote = lambda _symbol: source_quote
    core._closed_m15_decision_time = (
        lambda _data: pd.Timestamp("2026-08-11T12:00:00Z")
    )
    monkeypatch.setattr(
        main,
        "stable_plan_id",
        lambda **_fields: "clock-domain-plan",
    )

    plan_id, quote, _expires_at, observed_at = (
        core._register_shadow_candidate(
            "EURUSD",
            {
                "side": "LONG",
                "entry_min": 1.0998,
                "entry_max": 1.1002,
                "entry_price": 1.1000,
                "stop_price": 1.0950,
                "trigger_kind": "rejection_block_1h",
                "setup_tf": "1H",
            },
            trigger_signature="LONG|rb1h|closed",
            data=None,
        )
    )

    assert plan_id == "clock-domain-plan"
    assert quote is source_quote
    assert watcher.spec is not None
    assert watcher.spec.activation_tick_msc == int(
        observed_at.timestamp() * 1000
    )
    assert watcher.spec.activation_source_tick_msc == source_quote.time_msc
    assert watcher.spec.activation_tick_msc != source_quote.time_msc
    assert watcher.scans[0][2]["observed_at_utc"] == observed_at
