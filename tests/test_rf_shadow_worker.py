"""Safety tests for post-decision RF candidate enrichment.

The production ``main.py`` wiring is exercised through its fail-open shape:
enrich copied candidates, catch diagnostics, then persist the original scan.
No RF result is routed back into entry selection or execution.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import sqlite3

import pandas as pd
import pytest

from backtest.cost_model import CostProfile
import core.rf_shadow_bridge as bridge_module
from core.rf_shadow_bridge import (
    RFShadowBridge,
    compute_orca_snapshot_hash,
    load_orca_rf_snapshot,
)
from core.shadow_candidate_ledger import ShadowCandidateLedger


DECISION = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
OBSERVED = DECISION + timedelta(seconds=8)


def _profile() -> CostProfile:
    return CostProfile.from_mapping(
        {
            "schema": "fx-cost-profile-v1",
            "profile_id": "rf-shadow-cost-v1",
            "account_currency": "USD",
            "measured_from": "causal forward broker observations",
            "created_at_utc": "2026-08-20T00:00:00Z",
            "measured_through_utc": "2026-08-19T23:59:00Z",
            "rollover_timezone": "UTC",
            "rollover_hour": 0,
            "triple_swap_weekday": 2,
            "symbols": {
                "EURUSD": {
                    "base_currency": "EUR",
                    "quote_currency": "USD",
                    "contract_size": 100000,
                    "spread_price": 0.00010,
                    "slippage_price_per_side": 0.00002,
                    "commission_round_turn_per_lot": 7.0,
                    "swap_long_per_lot_rollover": -2.0,
                    "swap_short_per_lot_rollover": 1.0,
                }
            },
        }
    )


def _frame(frequency: str, rows: int = 30) -> pd.DataFrame:
    index = pd.date_range(
        end=DECISION,
        periods=rows,
        freq=frequency,
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "open": [1.1000 + index * 0.0001 for index in range(rows)],
            "high": [1.1020 + index * 0.0001 for index in range(rows)],
            "low": [1.0980 + index * 0.0001 for index in range(rows)],
            "close": [1.1010 + index * 0.0001 for index in range(rows)],
        },
        index=index,
    )


def _strategy_data() -> dict[str, pd.DataFrame]:
    return {"1H": _frame("1h"), "15M": _frame("15min")}


def _candidates() -> list[dict[str, object]]:
    return [
        {
            "signal": "ENTER",
            "side": "LONG",
            "entry_price": 1.1010,
            "entry_min": 1.1008,
            "entry_max": 1.1012,
            "stop_price": 1.0960,
            "tp_prices": [1.1060, 1.1110, 1.1160],
            "trigger_kind": "pivot_reclaim",
            "trigger_event_id": "pivot-1",
            "shadow_trigger_signature": "pivot:long:1",
            "risk_pct": 0.01,
            "factor_vector": {"bias": "LONG"},
        },
        {
            "signal": "ENTER",
            "side": "SHORT",
            "entry_price": 1.1000,
            "entry_min": 1.0998,
            "entry_max": 1.1002,
            "stop_price": 1.1050,
            "tp_prices": [1.0950, 1.0900, 1.0850],
            "trigger_kind": "order_block_1h",
            "trigger_event_id": "ob-1",
            "shadow_trigger_signature": "ob:short:1",
            "risk_pct": 0.01,
            "factor_vector": {"bias": "SHORT"},
        },
    ]


class RecordingScorer:
    model_id = "recording-rf"

    def __init__(self) -> None:
        self.calls: list[tuple[dict, dict]] = []

    def score(self, candidate, **kwargs):
        self.calls.append((deepcopy(candidate), deepcopy(kwargs)))
        is_long = candidate["side"] == "LONG"
        return {
            "rf_executing": False,
            "rf_causal_cost_r": kwargs["causal_cost_r"],
            "rf_tp_probability": 0.65 if is_long else 0.55,
            "rf_expected_net_r": 0.30 if is_long else 0.20,
            "rf_conservative_intrinsic_score_r": 0.25 if is_long else 0.15,
        }


def _snapshot(
    *,
    generated: datetime = DECISION - timedelta(hours=1),
    market_as_of: datetime = DECISION - timedelta(hours=14),
) -> dict[str, object]:
    model_known = generated - timedelta(minutes=30)
    payload: dict[str, object] = {
        "schema_version": "orca-monitor-snapshot-v1",
        "generated_at_utc": generated.isoformat().replace("+00:00", "Z"),
        "as_of_market_utc": market_as_of.isoformat().replace("+00:00", "Z"),
        "calculation_as_of_utc": generated.isoformat().replace("+00:00", "Z"),
        "returns_start_utc": (market_as_of - timedelta(days=120))
        .isoformat()
        .replace("+00:00", "Z"),
        "freshness": "FRESH",
        "age_seconds": (generated - market_as_of).total_seconds(),
        "expected_assets": 3,
        "asset_count": 3,
        "full_universe": True,
        "symbols": ["SPY", "TLT", "GLD"],
        "spectral_schema": "orca-spectral-v1",
        "spectral_config_id": "spectral-test-v1",
        "feature_registry_hash": "a" * 64,
        "feature_values_hash": "b" * 64,
        "features": {"ewm_hl30d__lambda_01": 1.8},
        "networks": [
            {
                "estimator": "ewm_hl30d",
                "observation_count": 120,
                "eigenvalues": [1.8, 0.8, 0.4],
                "absorption_ratios": {"1": 0.60, "3": 1.0},
                "lambda1_lambda2": 2.25,
                "marchenko_pastur_lambda1_excess": 0.35,
            }
        ],
        "model": {
            "status": "OOS_VALIDATED",
            "model_id": "orca-rf-1",
            "model_artifact_hash": "c" * 64,
            "prediction_hash": "d" * 64,
            "as_of_market_utc": market_as_of.isoformat().replace("+00:00", "Z"),
            "known_at_utc": model_known.isoformat().replace("+00:00", "Z"),
            "trained_through_utc": (market_as_of - timedelta(days=1))
            .isoformat()
            .replace("+00:00", "Z"),
            "p_rally": 0.30,
            "p_crash": 0.20,
            "rally_rank": 0.50,
            "crash_rank": 0.35,
            "validation": {"bcd_auc": 0.71},
            "calibration_status": "UNCALIBRATED",
        },
        "regime": "NORMAL",
        "risk_mode": "NEUTRAL",
        "suggested_equity_exposure": None,
        "exposure_status": "INTERMEDIATE_MAP_NOT_PUBLISHED",
        "regime_eligible": True,
        "executing": False,
    }
    payload["snapshot_hash"] = compute_orca_snapshot_hash(payload)
    return payload


def _write_snapshot(path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _non_rf(candidate: dict) -> dict:
    return {
        key: value for key, value in candidate.items() if not str(key).startswith("rf_")
    }


def test_scores_copies_cost_once_and_preserves_all_trade_geometry(
    monkeypatch,
) -> None:
    scorer = RecordingScorer()
    rf_bridge = RFShadowBridge(scorer=scorer, cost_profile=_profile())
    candidates = _candidates()
    frozen = deepcopy(candidates)
    original_estimator = bridge_module.estimate_cost_r
    cost_calls = []

    def counted_estimator(*args, **kwargs):
        cost_calls.append((args, kwargs))
        return original_estimator(*args, **kwargs)

    monkeypatch.setattr(bridge_module, "estimate_cost_r", counted_estimator)
    ranked = rf_bridge.enrich_candidates(
        symbol="EURUSD",
        candidates=candidates,
        observed_at_utc=OBSERVED,
        decision_bar_close=DECISION,
        strategy_data=_strategy_data(),
    )

    assert candidates == frozen
    assert len(cost_calls) == len(candidates)
    assert len(scorer.calls) == len(candidates)
    assert [_non_rf(row) for row in ranked] == frozen
    assert [row["rf_rank_position"] for row in ranked] == [1, 2]
    assert [row["rf_selected"] for row in ranked] == [True, False]
    assert all(row["rf_executing"] is False for row in ranked)
    assert all(call[0]["spread_r"] > 0.0 for call in scorer.calls)
    assert all(call[1]["atr_h1_14"] > 0.0 for call in scorer.calls)
    assert all(call[1]["atr_m15_14"] > 0.0 for call in scorer.calls)


def test_malicious_non_rf_output_and_nested_mutation_are_rejected() -> None:
    class MaliciousScorer:
        model_id = "malicious"

        def score(self, candidate, **kwargs):
            del kwargs
            candidate["side"] = "SHORT"
            candidate["tp_prices"].append(9.999)
            return {"side": "SHORT", "rf_expected_net_r": 99.0}

    candidate = _candidates()[:1]
    frozen = deepcopy(candidate)
    rf_bridge = RFShadowBridge(
        scorer=MaliciousScorer(),
        cost_profile=_profile(),
    )
    with pytest.raises(ValueError, match="non-rf"):
        rf_bridge.enrich_candidates(
            symbol="EURUSD",
            candidates=candidate,
            observed_at_utc=OBSERVED,
            decision_bar_close=DECISION,
            strategy_data=_strategy_data(),
        )
    assert candidate == frozen


def test_valid_orca_is_adapted_but_stale_and_future_are_unavailable(
    tmp_path,
) -> None:
    path = tmp_path / "orca.json"
    valid = _snapshot()
    _write_snapshot(path, valid)
    adapted = load_orca_rf_snapshot(path, decision_time_utc=DECISION)
    assert adapted["known_at_utc"] == valid["generated_at_utc"]
    assert adapted["market_mode"] == "TRANSITION"
    assert adapted["absorption_ratio"] == pytest.approx(1.0)
    assert adapted["largest_eigenvalue"] == pytest.approx(1.8)
    assert adapted["eigenvalue_ratio"] == pytest.approx(0.6)
    assert adapted["crisis_probability"] == pytest.approx(0.2)

    for name, payload in (
        (
            "stale",
            _snapshot(
                generated=DECISION - timedelta(hours=1),
                market_as_of=DECISION - timedelta(days=4),
            ),
        ),
        (
            "future",
            _snapshot(
                generated=DECISION + timedelta(seconds=1),
                market_as_of=DECISION - timedelta(hours=12),
            ),
        ),
    ):
        _write_snapshot(path, payload)
        scorer = RecordingScorer()
        rf_bridge = RFShadowBridge(
            scorer=scorer,
            cost_profile=_profile(),
            orca_snapshot_path=path,
        )
        result = rf_bridge.enrich_candidates(
            symbol="EURUSD",
            candidates=_candidates()[:1],
            observed_at_utc=OBSERVED,
            decision_bar_close=DECISION,
            strategy_data=_strategy_data(),
        )
        assert scorer.calls[0][1]["orca_snapshot"] is None, name
        assert result[0]["rf_orca_snapshot_status"].startswith("UNAVAILABLE:")


def test_bridge_reads_snapshot_once_and_gates_status_and_universe(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "orca.json"
    valid = _snapshot()
    _write_snapshot(path, valid)
    original_read_text = type(path).read_text
    reads: list[object] = []

    def counted_read_text(self, *args, **kwargs):
        if self == path:
            reads.append(self)
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(type(path), "read_text", counted_read_text)
    scorer = RecordingScorer()
    rf_bridge = RFShadowBridge(
        scorer=scorer,
        cost_profile=_profile(),
        orca_snapshot_path=path,
    )
    result = rf_bridge.enrich_candidates(
        symbol="EURUSD",
        candidates=_candidates()[:1],
        observed_at_utc=OBSERVED,
        decision_bar_close=DECISION,
        strategy_data=_strategy_data(),
    )
    assert len(reads) == 1
    assert scorer.calls[0][1]["orca_snapshot"] is not None
    assert result[0]["rf_orca_snapshot_status"] == "AVAILABLE"
    assert result[0]["rf_orca_snapshot_hash"] == valid["snapshot_hash"]

    for name, mutate, error in (
        (
            "partial-model",
            lambda row: row["model"].update({"status": "OOS_PARTIAL"}),
            "model status",
        ),
        (
            "incomplete-universe",
            lambda row: row.update({"full_universe": False}),
            "complete expected universe",
        ),
    ):
        invalid = deepcopy(valid)
        mutate(invalid)
        invalid["snapshot_hash"] = compute_orca_snapshot_hash(invalid)
        _write_snapshot(path, invalid)
        with pytest.raises(ValueError, match=error):
            load_orca_rf_snapshot(path, decision_time_utc=DECISION)

        local_scorer = RecordingScorer()
        local_bridge = RFShadowBridge(
            scorer=local_scorer,
            cost_profile=_profile(),
            orca_snapshot_path=path,
        )
        annotated = local_bridge.enrich_candidates(
            symbol="EURUSD",
            candidates=_candidates()[:1],
            observed_at_utc=OBSERVED,
            decision_bar_close=DECISION,
            strategy_data=_strategy_data(),
        )
        assert local_scorer.calls[0][1]["orca_snapshot"] is None, name
        assert annotated[0]["rf_orca_snapshot_status"].startswith("UNAVAILABLE:")


def test_future_bar_mutation_cannot_change_h1_or_m15_atr() -> None:
    base = _strategy_data()
    mutated = {name: frame.copy(deep=True) for name, frame in base.items()}
    for name, frame in mutated.items():
        future = frame.iloc[-1].copy()
        future["high"] = 9.0
        future["low"] = 0.1
        future["close"] = 8.0
        frame.loc[DECISION + timedelta(days=1)] = future

    scorer = RecordingScorer()
    rf_bridge = RFShadowBridge(scorer=scorer, cost_profile=_profile())
    for frames in (base, mutated):
        rf_bridge.enrich_candidates(
            symbol="EURUSD",
            candidates=_candidates()[:1],
            observed_at_utc=OBSERVED,
            decision_bar_close=DECISION,
            strategy_data=frames,
        )
    first = scorer.calls[0][1]
    second = scorer.calls[1][1]
    assert first["atr_h1_14"] == pytest.approx(second["atr_h1_14"])
    assert first["atr_m15_14"] == pytest.approx(second["atr_m15_14"])


def test_missing_cost_profile_marks_rf_unavailable_without_scoring() -> None:
    scorer = RecordingScorer()
    rf_bridge = RFShadowBridge(scorer=scorer, cost_profile=None)
    candidates = _candidates()
    frozen = deepcopy(candidates)
    result = rf_bridge.enrich_candidates(
        symbol="EURUSD",
        candidates=candidates,
        observed_at_utc=OBSERVED,
        decision_bar_close=DECISION,
        strategy_data=_strategy_data(),
    )
    assert candidates == frozen
    assert scorer.calls == []
    assert [_non_rf(row) for row in result] == frozen
    assert all(row["rf_status"] == "UNAVAILABLE_COST_PROFILE" for row in result)


def test_scorer_failure_preserves_scan_and_production_signal_bytes(
    tmp_path,
) -> None:
    class BrokenScorer:
        model_id = "broken"

        def score(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("injected scorer failure")

    ledger = ShadowCandidateLedger(tmp_path / "shadow.db")
    rf_bridge = RFShadowBridge(
        scorer=BrokenScorer(),
        cost_profile=_profile(),
    )
    candidates = _candidates()
    production_signal = deepcopy(candidates[0])
    production_bytes = json.dumps(
        production_signal,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    errors = []

    def diagnostic_worker() -> None:
        try:
            diagnostic_candidates = rf_bridge.enrich_candidates(
                symbol="EURUSD",
                candidates=candidates,
                observed_at_utc=OBSERVED,
                decision_bar_close=DECISION,
                strategy_data=_strategy_data(),
            )
        except Exception as exc:
            errors.append(str(exc))
            diagnostic_candidates = deepcopy(candidates)
        ledger.record_scan(
            deployment_id="rf-shadow-test",
            symbol="EURUSD",
            observed_at_utc=OBSERVED,
            decision_bar_close=DECISION,
            production_signal=production_signal,
            candidates=diagnostic_candidates,
        )

    result = diagnostic_worker()
    current_bytes = json.dumps(
        production_signal,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert result is None
    assert current_bytes == production_bytes
    assert errors == ["injected scorer failure"]
    assert ledger.enabled, ledger.disabled_reason
    ledger.close()

    with sqlite3.connect(tmp_path / "shadow.db") as connection:
        rows = connection.execute(
            "SELECT payload_json FROM candidates ORDER BY candidate_rank"
        ).fetchall()
    assert len(rows) == len(candidates)
    persisted = [json.loads(row[0]) for row in rows]
    assert persisted == candidates
    assert not any(key.startswith("rf_") for row in persisted for key in row)


@pytest.mark.parametrize("bridge_fails", [False, True])
def test_main_shadow_worker_rf_wiring_is_post_decision_and_fail_open(
    tmp_path,
    bridge_fails,
) -> None:
    import main

    source_candidates = _candidates()
    production_signal = deepcopy(source_candidates[0])
    production_bytes = json.dumps(
        production_signal,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    class Strategy:
        _last_candidate_errors = []

        def generate_candidate_signals(self, strategy_data, *, symbol):
            assert strategy_data == {"closed": True}
            assert symbol == "EURUSD"
            return deepcopy(source_candidates)

    class Bridge:
        def enrich_candidates(self, **kwargs):
            assert kwargs["symbol"] == "EURUSD"
            assert kwargs["decision_bar_close"] == DECISION
            assert kwargs["observed_at_utc"] == OBSERVED
            if bridge_fails:
                raise RuntimeError("injected bridge failure")
            rows = deepcopy(kwargs["candidates"])
            for rank, row in enumerate(rows, start=1):
                row["rf_executing"] = False
                row["rf_rank_position"] = rank
            return rows

    ledger = ShadowCandidateLedger(tmp_path / "main-shadow.db")
    main.Core._write_shadow_candidate_batch(
        ledger,
        Strategy(),
        (
            {
                "deployment_id": "rf-main-wiring-test",
                "symbol": "EURUSD",
                "observed_at_utc": OBSERVED,
                "decision_bar_close": DECISION,
                "production_signal": production_signal,
                "strategy_data": {"closed": True},
            },
        ),
        {"EURUSD": {"signal": "HOLD"}},
        None,
        None,
        None,
        None,
        Bridge(),
    )
    ledger.close()

    with sqlite3.connect(tmp_path / "main-shadow.db") as connection:
        rows = connection.execute(
            "SELECT payload_json FROM candidates ORDER BY candidate_rank"
        ).fetchall()
    persisted = [json.loads(row[0]) for row in rows]
    persisted_geometry = [_non_rf(row) for row in persisted]
    source_geometry = deepcopy(source_candidates)
    for row in persisted_geometry:
        row.pop("shadow_trigger_signature", None)
    for row in source_geometry:
        row.pop("shadow_trigger_signature", None)
    assert persisted_geometry == source_geometry
    if bridge_fails:
        assert not any(key.startswith("rf_") for row in persisted for key in row)
    else:
        assert [row["rf_rank_position"] for row in persisted] == [1, 2]
        assert all(row["rf_executing"] is False for row in persisted)
    assert (
        json.dumps(
            production_signal,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        == production_bytes
    )
