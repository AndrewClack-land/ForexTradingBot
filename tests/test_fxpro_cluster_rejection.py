from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import backtest.fxpro_cluster_data as cluster_data
from backtest.fxpro_cluster_data import (
    FxProClusterEventDataset,
    seal_quantower_diagnostics,
)
from core.fxpro_cluster_rejection import (
    CLUSTER_CLASSIFICATION,
    CLUSTER_DATA_KIND,
    CLUSTER_EVENT_SCHEMA_VERSION,
    CLUSTER_MARKET_TYPE,
    CLUSTER_SOURCE,
    CLUSTER_TIMEFRAME,
    CLUSTER_VENUE,
    ClusterRejectionThresholds,
    detect_fxpro_cluster_rejection,
    seal_cluster_event,
    validate_cluster_event,
)
from core.strategy_narrative import NarrativeStrategy


BAR_OPEN = pd.Timestamp("2026-07-24T10:00:00Z")
BAR_CLOSE = BAR_OPEN + pd.Timedelta(minutes=15)


def _level(
    ticks: int,
    *,
    buy: float,
    sell: float,
) -> dict:
    volume = buy + sell
    return {
        "price": f"{ticks / 10_000:.4f}",
        "price_ticks": ticks,
        "volume": str(volume),
        "trades": int(volume),
        "buy_volume": str(buy),
        "sell_volume": str(sell),
        "buy_trades": int(buy),
        "sell_trades": int(sell),
        "delta": str(buy - sell),
    }


def _levels() -> list[dict]:
    return [
        _level(10990, buy=5, sell=35),
        _level(10995, buy=4, sell=6),
        _level(11000, buy=15, sell=15),
        _level(11005, buy=6, sell=4),
        _level(11010, buy=5, sell=5),
    ]


def _total(levels: list[dict]) -> dict:
    return {
        "volume": str(sum(float(level["volume"]) for level in levels)),
        "trades": sum(level["trades"] for level in levels),
        "buy_volume": str(
            sum(float(level["buy_volume"]) for level in levels)
        ),
        "sell_volume": str(
            sum(float(level["sell_volume"]) for level in levels)
        ),
        "buy_trades": sum(level["buy_trades"] for level in levels),
        "sell_trades": sum(level["sell_trades"] for level in levels),
        "delta": str(sum(float(level["delta"]) for level in levels)),
    }


def _event(
    *,
    availability_mode: str = "research_assumption",
    available_at: pd.Timestamp = BAR_CLOSE,
) -> dict:
    levels = _levels()
    return seal_cluster_event(
        {
            "schema_version": CLUSTER_EVENT_SCHEMA_VERSION,
            "symbol": "EURUSD",
            "timeframe": CLUSTER_TIMEFRAME,
            "bar_open": BAR_OPEN.isoformat(),
            "bar_close": BAR_CLOSE.isoformat(),
            "available_at": available_at.isoformat(),
            "availability_mode": availability_mode,
            "availability_evidence": (
                "unit-test forward timestamp"
                if availability_mode == "forward_observed"
                else "unit-test research delay assumption"
            ),
            "source": CLUSTER_SOURCE,
            "venue": CLUSTER_VENUE,
            "market_type": CLUSTER_MARKET_TYPE,
            "data_kind": CLUSTER_DATA_KIND,
            "classification": CLUSTER_CLASSIFICATION,
            "execution_proof": False,
            "aggressor_polarity_proven": False,
            "finalized": True,
            "tick_size": "0.0001",
            "ohlc": {
                "open": "1.1000",
                "high": "1.1010",
                "low": "1.0990",
                "close": "1.1005",
            },
            "total": _total(levels),
            "price_levels": levels,
            "source_bar_hash": "a" * 64,
            "source_manifest_sha256": "b" * 64,
        }
    )


def _candle() -> dict:
    return {
        "open": 1.1000,
        "high": 1.1010,
        "low": 1.0990,
        "close": 1.1005,
    }


def test_cluster_proxy_detects_long_rejection_without_execution_claim():
    signal = detect_fxpro_cluster_rejection(
        candle=_candle(),
        candle_open_time=BAR_OPEN,
        event=_event(availability_mode="forward_observed"),
        symbol="EURUSD",
        side="LONG",
        decision_time=BAR_CLOSE,
    )

    assert signal is not None
    assert signal.side == "LONG"
    assert signal.edge_imbalance_ratio == pytest.approx(7.0)
    assert signal.edge_volume_share == pytest.approx(0.4)
    assert signal.rejection_fraction == pytest.approx(0.75)
    assert signal.wick_fraction == pytest.approx(0.5)
    assert signal.to_dict()["execution_proof_claimed"] is False
    assert signal.to_dict()["aggressor_polarity_claimed"] is False
    assert (
        detect_fxpro_cluster_rejection(
            candle=_candle(),
            candle_open_time=BAR_OPEN,
            event=_event(availability_mode="forward_observed"),
            symbol="EURUSD",
            side="SHORT",
            decision_time=BAR_CLOSE,
        )
        is None
    )


def test_research_assumption_is_refused_by_live_default():
    event = _event()
    assert (
        detect_fxpro_cluster_rejection(
            candle=_candle(),
            candle_open_time=BAR_OPEN,
            event=event,
            symbol="EURUSD",
            side="LONG",
            decision_time=BAR_CLOSE,
        )
        is None
    )
    assert (
        detect_fxpro_cluster_rejection(
            candle=_candle(),
            candle_open_time=BAR_OPEN,
            event=event,
            symbol="EURUSD",
            side="LONG",
            decision_time=BAR_CLOSE,
            allow_research_assumption=True,
        )
        is not None
    )


def test_cluster_event_fails_closed_for_tamper_delay_and_candle_mismatch():
    tampered = _event()
    tampered["total"]["sell_volume"] = "66"
    assert validate_cluster_event(tampered) is None

    delayed = _event(available_at=BAR_CLOSE + pd.Timedelta(seconds=1))
    assert (
        detect_fxpro_cluster_rejection(
            candle=_candle(),
            candle_open_time=BAR_OPEN,
            event=delayed,
            symbol="EURUSD",
            side="LONG",
            decision_time=BAR_CLOSE,
            allow_research_assumption=True,
        )
        is None
    )

    mismatch = _candle()
    mismatch["low"] -= 0.001
    assert (
        detect_fxpro_cluster_rejection(
            candle=mismatch,
            candle_open_time=BAR_OPEN,
            event=_event(availability_mode="forward_observed"),
            symbol="EURUSD",
            side="LONG",
            decision_time=BAR_CLOSE,
        )
        is None
    )


def test_strategy_trigger_is_structured_and_research_guarded():
    frame = pd.DataFrame([_candle()], index=[BAR_OPEN])
    strategy = NarrativeStrategy()
    strategy.cluster_rejection_15m_entry_enabled = True
    strategy.cluster_rejection_allow_research_assumption = True

    entry = strategy.trigger_15m_cluster_rejection(
        frame,
        "LONG",
        _event(),
        symbol="EURUSD",
    )

    assert entry is not None
    assert entry.trigger_kind == "fxpro_cluster_rejection_15m"
    assert entry.trigger_event_id == _event()["checksum"]
    assert entry.trigger_meta["cluster_score"] >= 0.6
    assert "Cluster Rejection" in entry.reason


def _write_minimal_diagnostic(root: Path) -> None:
    levels = _levels()
    manifest = {
        "semantics": {
            "classification": (
                "quantower_tick_reconstructed_bidask_history_available"
            )
        },
        "instrument": {"tick_size": "0.0001"},
    }
    bars = {
        "bars": [
            {
                "symbol": "EURUSD",
                "bar_open": BAR_OPEN.isoformat(),
                "bar_close": BAR_CLOSE.isoformat(),
                "ohlc": {
                    "open": "1.1000",
                    "high": "1.1010",
                    "low": "1.0990",
                    "close": "1.1005",
                },
                "total": _total(levels),
                "price_levels": levels,
            }
        ]
    }
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (root / "bars.json").write_text(json.dumps(bars), encoding="utf-8")


def test_diagnostic_conversion_seals_research_delay_and_asof(
    tmp_path,
    monkeypatch,
):
    diagnostic = tmp_path / "diagnostic"
    _write_minimal_diagnostic(diagnostic)
    monkeypatch.setattr(
        cluster_data,
        "_validate_diagnostic_export",
        lambda *_args, **_kwargs: SimpleNamespace(
            export_id="11111111-2222-3333-4444-555555555555",
            day_utc="2026-07-24",
            bars=1,
            price_levels=5,
            missing_slots=95,
            content_sha256="c" * 64,
            exporter_binary_sha256="d" * 64,
        ),
    )

    sidecar = seal_quantower_diagnostics(
        [diagnostic],
        tmp_path / "sidecar",
        availability_delay_ms=1_000,
        availability_evidence=(
            "research assumption: live exporter target delay is one second"
        ),
    )
    dataset = FxProClusterEventDataset.load(sidecar)

    assert dataset.research_only is True
    assert dataset.event_asof("EURUSD", BAR_OPEN, BAR_CLOSE) is None
    event = dataset.event_asof(
        "EURUSD",
        BAR_OPEN,
        BAR_CLOSE + pd.Timedelta(seconds=1),
    )
    assert event is not None
    assert event["availability_mode"] == "research_assumption"
    assert validate_cluster_event(event) is not None

    event_path = sidecar / "events.jsonl"
    event_path.write_text(
        event_path.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(cluster_data.ClusterDataValidationError):
        FxProClusterEventDataset.load(sidecar)


def test_threshold_contract_rejects_non_directional_configuration():
    with pytest.raises(ValueError, match="imbalance"):
        ClusterRejectionThresholds(min_edge_imbalance_ratio=1.0)
