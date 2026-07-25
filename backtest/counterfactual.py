"""Counterfactual all-M15 universe and frozen-weight OOS replay.

This module is intentionally separate from the fixed production backtest.
For every completed M15 decision it freezes the raw five-factor vector, then
detects each technically available trigger for LONG and SHORT independently
of the production bias or live trigger switches. Technical opportunities are
labelled before stateful
execution gates, fitted inside the outer training window only, and replayed
chronologically with one frozen model in the following OOS window.

Turtle Soup is not part of the trigger manifest.  Absorption is available only
when a sealed, causal order-flow event is supplied; OHLCV is never used as a
proxy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import pandas as pd

from .data import HistoricalDataset
from .orderflow_data import AbsorptionEventDataset
from .simulator import simulate_split_outcome
from .strategy_runner import (
    NarrativeBacktestConfig,
    StrategyBacktestError,
    StrategyFactory,
    _BAR_CLOSE_COLUMN,
    _EMPTY_CSV_FIELDS,
    _Candidate,
    _Period,
    _apply_deterministic_gates,
    _causal_force_price,
    _configure_strategy,
    _default_strategy_factory,
    _emit,
    _execute_candidates,
    _find_fill,
    _force_close_outcome,
    _metrics_for,
    _next_friday_close,
    _periods,
    _prepare_symbol,
    _safe_json_value,
    _strategy_data,
    _validate_factor_vector,
    infer_common_strategy_range,
)
from .weight_optimizer import (
    DEFAULT_MIN_TRAIN_OPPORTUNITIES,
    DEFAULT_RIDGE_ALPHA,
    build_counterfactual_weight_scores,
)


COUNTERFACTUAL_SCHEMA = "narrative-counterfactual-wfo/v1"
TRIGGER_MANIFEST = (
    "rejection_block_15m",
    "absorption_15m",
    "h1_pivot_reclaim_15m",
    "order_block_1h",
)
_TRIGGER_PRIORITY = {
    trigger: index for index, trigger in enumerate(TRIGGER_MANIFEST)
}
_PRIMARY_POLICY = "stop-first"
_METRIC_FIELDS = (
    "setups_total",
    "setups_closed",
    "setups_open",
    "wins",
    "losses",
    "breakeven",
    "win_rate",
    "loss_rate",
    "breakeven_rate",
    "net_r",
    "expectancy_r",
    "median_r",
    "average_win_r",
    "average_loss_r",
    "gross_profit_r",
    "gross_loss_r",
    "profit_factor",
    "max_drawdown_r",
    "longest_loss_streak",
    "tp1_reach_rate",
    "tp2_reach_rate",
    "tp3_reach_rate",
    "moved_to_be_rate",
    "ambiguous_bars",
)
_REPORT_FIELDS: Mapping[str, tuple[str, ...]] = {
    "decision_events.csv": (
        "schema",
        "decision_event_id",
        "symbol",
        "decision_time",
        "production_bias_ignored_for_generation",
        "factor_vector",
        "fvg_side",
        "orderflow_status",
        "technical_triggers",
    ),
    "technical_opportunities.csv": (
        "opportunity_id",
        "decision_event_id",
        "symbol",
        "decision_time",
        "side",
        "trigger_kind",
        "trigger_tags",
        "trigger_reason",
        "entry_min",
        "entry_max",
        "planned_entry",
        "stop",
        "tp_prices",
        "gate",
        "gate_reason",
        "factor_vector",
        "fvg_regime",
        "vol_r",
        "vol_regime",
        "vol_em_1d",
        "vol_tp1_em_ratio",
        "trigger_event_id",
        "trigger_meta",
    ),
    "opportunity_labels.csv": (
        "schema",
        "opportunity_id",
        "decision_event_id",
        "symbol",
        "decision_time",
        "side",
        "trigger_kind",
        "gate",
        "gate_reason",
        "factor_vector",
        "fill_time",
        "fill_price",
        "exit_time",
        "label_status",
        "opportunity_r",
        "net_r_conditional_fill",
        "label_reason",
        "sample_weight",
    ),
    "optimizer_predictions.csv": (
        "schema",
        "opportunity_id",
        "decision_event_id",
        "fold_index",
        "symbol",
        "decision_time",
        "side",
        "trigger_kind",
        "model_status",
        "model_id",
        "train_opportunities",
        "train_decision_side_clusters",
        "train_effective_sample_size",
        "factor_alignment",
        "weighted_factor_score",
        "normalized_factor_score",
        "predicted_opportunity_r",
        "actual_label_status",
        "actual_label_known_before_test_end",
        "actual_opportunity_r",
        "prediction_error_r",
        "score_threshold_applied",
    ),
    "optimizer_coefficients.csv": (
        "schema",
        "model_id",
        "fold_index",
        "model_status",
        "feature",
        "reference_weight",
        "fitted_weight",
        "delta_from_reference",
    ),
    "optimizer_metrics.csv": (
        "schema",
        "dimension",
        "dimension_value",
        "oos_opportunities",
        "opportunities_scored",
        "labeled_opportunities",
        "filled_labels",
        "no_fill_labels",
        "actual_mean_r",
        "predicted_mean_r",
        "mae_r",
        "rmse_r",
        "rank_ic",
    ),
    "oos_selections.csv": (
        "schema",
        "replay_id",
        "fold_index",
        "test_start",
        "test_end",
        "symbol",
        "decision_time",
        "opportunity_id",
        "side",
        "trigger_kind",
        "model_id",
        "model_status",
        "score",
        "rank",
        "hard_score_threshold_applied",
        "eligible_pool",
        "alternatives",
        "selection_status",
    ),
    "executions.csv": tuple(_EMPTY_CSV_FIELDS["executions"]),
    "setups.csv": tuple(_EMPTY_CSV_FIELDS["setups"]),
    "legs.csv": tuple(_EMPTY_CSV_FIELDS["legs"]),
    "folds.csv": (
        "fold_index",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "policy",
        *_METRIC_FIELDS,
    ),
}


@dataclass(frozen=True)
class _Opportunity:
    opportunity_id: str
    decision_event_id: str
    symbol: str
    decision_time: pd.Timestamp
    trigger_kind: str
    trigger_tags: tuple[str, ...]
    gate: str
    gate_reason: str
    signal: Mapping[str, Any]

    def report_row(self) -> dict[str, Any]:
        return {
            "opportunity_id": self.opportunity_id,
            "decision_event_id": self.decision_event_id,
            "symbol": self.symbol,
            "decision_time": self.decision_time.isoformat(),
            "side": self.signal.get("side"),
            "trigger_kind": self.trigger_kind,
            "trigger_tags": list(self.trigger_tags),
            "trigger_reason": self.signal.get("trigger_reason"),
            "entry_min": self.signal.get("entry_min"),
            "entry_max": self.signal.get("entry_max"),
            "planned_entry": self.signal.get("entry_price"),
            "stop": self.signal.get("stop_price"),
            "tp_prices": list(self.signal.get("tp_prices") or ()),
            "gate": self.gate,
            "gate_reason": self.gate_reason,
            "factor_vector": self.signal.get("factor_vector"),
            "fvg_regime": self.signal.get("fvg_regime"),
            "vol_r": self.signal.get("vol_R"),
            "vol_regime": self.signal.get("vol_regime"),
            "vol_em_1d": self.signal.get("vol_em_1d"),
            "vol_tp1_em_ratio": self.signal.get("vol_tp1_em_ratio"),
            "trigger_event_id": self.signal.get("trigger_event_id"),
            "trigger_meta": self.signal.get("trigger_meta"),
        }


@dataclass(frozen=True)
class CounterfactualBacktestResult:
    summary: Mapping[str, Any]
    config: Mapping[str, Any]
    decision_events: tuple[Mapping[str, Any], ...]
    opportunities: tuple[Mapping[str, Any], ...]
    labels: tuple[Mapping[str, Any], ...]
    models: Mapping[str, Any]
    predictions: tuple[Mapping[str, Any], ...]
    coefficients: tuple[Mapping[str, Any], ...]
    optimizer_metrics: tuple[Mapping[str, Any], ...]
    selections: tuple[Mapping[str, Any], ...]
    executions: tuple[Mapping[str, Any], ...]
    setups: tuple[Mapping[str, Any], ...]
    legs: tuple[Mapping[str, Any], ...]
    folds: tuple[Mapping[str, Any], ...]

    def write(self, output: str | Path) -> Path:
        return write_counterfactual_report(self, output)


def _canonical_hash(payload: Any) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            _safe_json_value(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _global_decision_range(
    periods: Sequence[_Period],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    train_starts = [
        period.train_start
        for period in periods
        if period.train_start is not None
    ]
    if len(train_starts) != len(periods):
        raise StrategyBacktestError(
            "Counterfactual optimizer requires explicit walk-forward "
            "train/test windows"
        )
    return min(train_starts), max(period.test_end for period in periods)


def _decision_event_id(
    *,
    manifest_sha256: str,
    symbol: str,
    decision_time: pd.Timestamp,
) -> str:
    return _canonical_hash(
        {
            "schema": COUNTERFACTUAL_SCHEMA,
            "snapshot_manifest": manifest_sha256,
            "symbol": symbol,
            "decision_time": decision_time.isoformat(),
        }
    )[:24]


def _entry_geometry(signal: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "side": signal.get("side"),
        "entry_min": signal.get("entry_min"),
        "entry_max": signal.get("entry_max"),
        "entry_price": signal.get("entry_price"),
        "stop_price": signal.get("stop_price"),
        "tp_prices": list(signal.get("tp_prices") or ()),
    }


def _opportunity_id(
    *,
    decision_event_id: str,
    trigger_kind: str,
    signal: Mapping[str, Any],
) -> str:
    return _canonical_hash(
        {
            "schema": COUNTERFACTUAL_SCHEMA,
            "decision_event_id": decision_event_id,
            "trigger_kind": trigger_kind,
            "trigger_event_id": signal.get("trigger_event_id"),
            "geometry": _entry_geometry(signal),
        }
    )[:24]


def _materialize_entry(
    *,
    strategy: Any,
    entry: Any,
    trigger_kind: str,
    data: Mapping[str, pd.DataFrame],
    symbol: str,
    narrative: str,
    factor_vector: Mapping[str, Any],
    fvg_side: str,
    fvg_text: str,
) -> dict[str, Any]:
    df_4h = data["4H"]
    df_1h = data["1H"]
    df_15m = data["15M"]
    side = str(getattr(entry, "side", "")).upper()
    if side not in {"LONG", "SHORT"}:
        raise StrategyBacktestError(
            f"{symbol}: counterfactual trigger returned invalid side {side!r}"
        )

    if not bool(getattr(entry, "lock_entry_range", False)):
        entry = strategy._build_entry_range(entry, df_15m, side)
    if side == "LONG":
        entry_for_risk = float(
            entry.entry_max
            if entry.entry_max is not None
            else entry.entry_price
        )
    else:
        entry_for_risk = float(
            entry.entry_min
            if entry.entry_min is not None
            else entry.entry_price
        )
    stop, targets = strategy.calc_stop_and_tps(
        entry_for_risk,
        side,
        df_1h,
        df_4h,
        custom_stop=getattr(entry, "stop_override", None),
        symbol=symbol,
    )
    if len(targets) != 3:
        raise StrategyBacktestError(
            f"{symbol}: counterfactual opportunity must have three targets"
        )
    payload: dict[str, Any] = {
        "signal": "ENTER",
        "side": side,
        "entry_min": (
            round(float(entry.entry_min), 6)
            if entry.entry_min is not None
            else None
        ),
        "entry_max": (
            round(float(entry.entry_max), 6)
            if entry.entry_max is not None
            else None
        ),
        "entry_price": round(entry_for_risk, 6),
        "stop_price": round(float(stop), 6),
        "tp_price": round(float(targets[-1]), 6),
        "tp_prices": [round(float(value), 6) for value in targets],
        "risk_percent": f"{float(strategy.risk_per_trade) * 100:.2f}%",
        "tf": "15M",
        "setup_tf": "15M",
        "narrative": narrative,
        "factor_vector": dict(factor_vector),
        "vc": fvg_text,
        "fvg_regime": fvg_side,
        "trigger_reason": str(getattr(entry, "reason", trigger_kind)),
        "trigger_kind": (
            str(getattr(entry, "trigger_kind", "") or trigger_kind)
        ),
        "trigger_event_id": getattr(entry, "trigger_event_id", None),
        "trigger_meta": getattr(entry, "trigger_meta", None),
        "counterfactual_population": True,
        "m1_localized": False,
    }
    if entry.zone_low is not None and entry.zone_high is not None:
        payload["zone_low"] = round(
            float(min(entry.zone_low, entry.zone_high)),
            6,
        )
        payload["zone_high"] = round(
            float(max(entry.zone_low, entry.zone_high)),
            6,
        )
    return payload


def _detect_entries(
    *,
    strategy: Any,
    data: Mapping[str, Any],
    symbol: str,
) -> list[tuple[str, Any]]:
    df_1h = data["1H"]
    df_15m = data["15M"]
    context = getattr(strategy, "_last_htf_context", None)
    orderflow_event = data.get("ORDERFLOW_15M")
    detected: list[tuple[str, Any]] = []

    for side in ("LONG", "SHORT"):
        entry = strategy.trigger_15m_rejection_block(df_15m, side)
        if entry is not None:
            detected.append(("rejection_block_15m", entry))

        entry = strategy.trigger_15m_absorption(
            df_15m,
            side,
            orderflow_event,
            symbol=symbol,
        )
        if entry is not None:
            detected.append(("absorption_15m", entry))

        entry = strategy.trigger_h1_pivot_reclaim_on_15m(
            df_1h,
            df_15m,
            side,
        )
        if entry is not None:
            detected.append(("h1_pivot_reclaim_15m", entry))

        entry = strategy.trigger_orderblock_touch(df_1h, context, side)
        if entry is not None:
            detected.append(("order_block_1h", entry))
    return detected


def _deduplicate_opportunities(
    rows: Sequence[_Opportunity],
) -> list[_Opportunity]:
    by_id: dict[str, _Opportunity] = {}
    for row in rows:
        if row.opportunity_id in by_id:
            raise StrategyBacktestError(
                "duplicate counterfactual opportunity identity: "
                f"{row.opportunity_id}"
            )
        by_id[row.opportunity_id] = row
    return sorted(
        by_id.values(),
        key=lambda item: (
            item.decision_time,
            item.symbol,
            str(item.signal.get("side")),
            _TRIGGER_PRIORITY.get(item.trigger_kind, 999),
            item.opportunity_id,
        ),
    )


def _generate_symbol_universe(
    *,
    dataset: HistoricalDataset,
    symbol: str,
    prepared: Mapping[str, Any],
    periods: Sequence[_Period],
    config: NarrativeBacktestConfig,
    orderflow: Optional[AbsorptionEventDataset],
    strategy_factory: StrategyFactory,
    progress: Optional[Callable[[Mapping[str, Any]], None]],
) -> tuple[
    list[dict[str, Any]],
    list[_Opportunity],
    Counter[str],
]:
    range_start, range_end = _global_decision_range(periods)
    closes = prepared["15m"].close_times
    left = int(closes.searchsorted(range_start, side="left"))
    right = int(closes.searchsorted(range_end, side="left"))
    total = max(0, right - left)
    strategy = _configure_strategy(strategy_factory(), config)
    # Counterfactual detection must not inherit production trigger switches.
    # Operationally disabled families are still labelled, then remain blocked
    # in OOS replay unless the research command explicitly enables them.
    strategy.orderblock_entry_enabled = True
    if orderflow is not None:
        # Research-only activation is tied to a sealed sidecar.  Production
        # remains controlled by ABSORPTION_15M_ENTRY_ENABLED and currently has
        # no order-flow adapter on the VPS.
        strategy.absorption_15m_entry_enabled = True
    decisions: list[dict[str, Any]] = []
    opportunities: list[_Opportunity] = []
    counters: Counter[str] = Counter()
    next_progress = 0

    _emit(
        progress,
        phase="counterfactual-signals",
        state="start",
        symbol=symbol,
        decisions=total,
    )
    for completed, raw_time in enumerate(closes[left:right], start=1):
        decision_time = pd.Timestamp(raw_time)
        counters["decisions"] += 1
        data = _strategy_data(prepared, decision_time, config)
        if data is None:
            counters["warmup_skipped"] += 1
            continue

        orderflow_status = "DATA_UNAVAILABLE"
        event = None
        if orderflow is not None:
            event = orderflow.event_asof(
                symbol,
                data["15M"].index[-1],
                decision_time,
            )
            orderflow_status = (
                "AVAILABLE" if event is not None else "MISSING_OR_DELAYED"
            )
        data = dict(data)
        data["ORDERFLOW_15M"] = event

        try:
            baseline_bias, narrative = strategy.calc_narrative(
                data["4H"],
                data["1H"],
                data["15M"],
            )
            factor_vector = getattr(strategy, "_last_factor_vector", None)
            fvg_side, fvg_text = strategy.calc_fvg_regime_1h(data["1H"])
        except Exception as exc:
            raise StrategyBacktestError(
                f"{symbol} {decision_time.isoformat()}: "
                f"counterfactual context failed: {exc}"
            ) from exc
        frozen = {
            "signal": "NO_TRIGGER",
            "factor_vector": factor_vector,
        }
        _validate_factor_vector(
            frozen,
            status="NO_TRIGGER",
            required=(config.profile == "production-deterministic"),
        )
        if not isinstance(factor_vector, Mapping):
            counters["factor_vector_unavailable"] += 1
            continue

        decision_event_id = _decision_event_id(
            manifest_sha256=dataset.manifest_sha256,
            symbol=symbol,
            decision_time=decision_time,
        )
        detected = _detect_entries(
            strategy=strategy,
            data=data,
            symbol=symbol,
        )
        decision_row = {
            "schema": COUNTERFACTUAL_SCHEMA,
            "decision_event_id": decision_event_id,
            "symbol": symbol,
            "decision_time": decision_time.isoformat(),
            "production_bias_ignored_for_generation": baseline_bias,
            "factor_vector": dict(factor_vector),
            "fvg_side": fvg_side,
            "orderflow_status": orderflow_status,
            "technical_triggers": len(detected),
        }
        decisions.append(decision_row)
        counters[f"orderflow_{orderflow_status.lower()}"] += 1

        for trigger_kind, entry in detected:
            signal = _materialize_entry(
                strategy=strategy,
                entry=entry,
                trigger_kind=trigger_kind,
                data=data,
                symbol=symbol,
                narrative=narrative,
                factor_vector=factor_vector,
                fvg_side=fvg_side,
                fvg_text=fvg_text,
            )
            _validate_factor_vector(
                signal,
                status="OPPORTUNITY",
                required=(config.profile == "production-deterministic"),
            )
            gate, reason, enriched = _apply_deterministic_gates(
                symbol=symbol,
                decision_time=decision_time,
                signal=signal,
                data=data,
                config=config,
            )
            if (
                trigger_kind == "rejection_block_15m"
                and not config.rejection_block_entry_enabled
            ):
                gate = "BLOCK_TRIGGER_DISABLED"
                reason = (
                    "rejection_block_15m is included in counterfactual "
                    "training but disabled for OOS execution"
                )
            elif (
                trigger_kind == "order_block_1h"
                and not config.orderblock_entry_enabled
            ):
                gate = "BLOCK_TRIGGER_DISABLED"
                reason = (
                    "order_block_1h is included in counterfactual training "
                    "but disabled for OOS execution"
                )
            opportunity_id = _opportunity_id(
                decision_event_id=decision_event_id,
                trigger_kind=trigger_kind,
                signal=enriched,
            )
            opportunities.append(
                _Opportunity(
                    opportunity_id=opportunity_id,
                    decision_event_id=decision_event_id,
                    symbol=symbol,
                    decision_time=decision_time,
                    trigger_kind=trigger_kind,
                    trigger_tags=(trigger_kind,),
                    gate=gate,
                    gate_reason=str(reason),
                    signal=enriched,
                )
            )
            counters[f"trigger_{trigger_kind}"] += 1

        percent = int(completed * 100 / total) if total else 100
        if percent >= next_progress:
            _emit(
                progress,
                phase="counterfactual-signals",
                state="progress",
                symbol=symbol,
                completed=completed,
                decisions=total,
                candidates=len(opportunities),
                percent=percent,
            )
            next_progress = min(100, percent + 1)

    validated = _deduplicate_opportunities(opportunities)
    counters["opportunities_before_identity_validation"] = len(
        opportunities
    )
    counters["opportunities"] = len(validated)
    _emit(
        progress,
        phase="counterfactual-signals",
        state="complete",
        symbol=symbol,
        completed=total,
        decisions=total,
        candidates=len(validated),
    )
    return decisions, validated, counters


def _label_opportunity(
    *,
    opportunity: _Opportunity,
    m1: pd.DataFrame,
    config: NarrativeBacktestConfig,
    available_end: pd.Timestamp,
) -> dict[str, Any]:
    signal = opportunity.signal
    base = {
        "schema": COUNTERFACTUAL_SCHEMA,
        "opportunity_id": opportunity.opportunity_id,
        "decision_event_id": opportunity.decision_event_id,
        "symbol": opportunity.symbol,
        "decision_time": opportunity.decision_time.isoformat(),
        "side": signal.get("side"),
        "trigger_kind": opportunity.trigger_kind,
        "gate": opportunity.gate,
        "gate_reason": opportunity.gate_reason,
        "factor_vector": signal.get("factor_vector"),
        "fill_time": None,
        "fill_price": None,
        "exit_time": None,
        "label_status": "PENDING",
        "opportunity_r": None,
        "net_r_conditional_fill": None,
        "label_reason": None,
        "sample_weight": 1.0,
    }
    ttl_end = opportunity.decision_time + config.entry_ttl
    if ttl_end > available_end:
        return {
            **base,
            "label_status": "CENSORED",
            "label_reason": "entry TTL extends beyond sealed M1 coverage",
        }

    fill_position, fill_time, fill_price, fill_reason = _find_fill(
        m1=m1,
        decision_time=opportunity.decision_time,
        period_end=ttl_end,
        signal=signal,
        config=config,
    )
    if fill_position is None or fill_time is None or fill_price is None:
        if "invalid" in fill_reason:
            return {
                **base,
                "exit_time": ttl_end.isoformat(),
                "label_status": "INVALID",
                "label_reason": fill_reason,
            }
        return {
            **base,
            "exit_time": ttl_end.isoformat(),
            "label_status": "NO_FILL",
            "opportunity_r": 0.0,
            "label_reason": fill_reason,
        }

    horizon_end = fill_time + config.max_holding
    if config.profile == "production-deterministic":
        horizon_end = min(
            horizon_end,
            _next_friday_close(fill_time, config),
        )
    if horizon_end > available_end:
        return {
            **base,
            "fill_time": fill_time.isoformat(),
            "fill_price": fill_price,
            "label_status": "CENSORED",
            "label_reason": "outcome horizon extends beyond sealed M1 coverage",
        }
    simulation_right = int(
        m1.index.searchsorted(horizon_end, side="left")
    )
    bars = m1.iloc[fill_position:simulation_right]
    if bars.empty:
        return {
            **base,
            "fill_time": fill_time.isoformat(),
            "fill_price": fill_price,
            "label_status": "INVALID",
            "label_reason": "fill has no executable outcome bar",
        }
    try:
        outcome = simulate_split_outcome(
            side=str(signal.get("side") or "").upper(),
            entry=float(fill_price),
            stop=float(signal["stop_price"]),
            tp_prices=[
                float(value) for value in signal.get("tp_prices") or ()
            ],
            bars=bars,
            intrabar_policy=_PRIMARY_POLICY,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return {
            **base,
            "fill_time": fill_time.isoformat(),
            "fill_price": fill_price,
            "label_status": "INVALID",
            "label_reason": f"outcome simulation failed: {exc}",
        }
    if outcome.status == "OPEN":
        force_time, force_price = _causal_force_price(m1, horizon_end)
        if force_time is None or force_price is None:
            return {
                **base,
                "fill_time": fill_time.isoformat(),
                "fill_price": fill_price,
                "label_status": "CENSORED",
                "label_reason": "no causal force-close price",
            }
        outcome = _force_close_outcome(
            outcome,
            price=force_price,
            timestamp=force_time,
        )
    if outcome.exit_time is None or not math.isfinite(float(outcome.net_r)):
        return {
            **base,
            "fill_time": fill_time.isoformat(),
            "fill_price": fill_price,
            "label_status": "INVALID",
            "label_reason": "closed outcome has no finite exit label",
        }
    return {
        **base,
        "fill_time": fill_time.isoformat(),
        "fill_price": float(fill_price),
        "exit_time": outcome.exit_time.isoformat(),
        "label_status": "FILLED",
        "opportunity_r": float(outcome.net_r),
        "net_r_conditional_fill": float(outcome.net_r),
        "label_reason": "independent technical opportunity label",
    }


def _select_oos_candidates(
    *,
    predictions: Sequence[Mapping[str, Any]],
    opportunity_by_id: Mapping[str, _Opportunity],
    periods: Sequence[_Period],
) -> tuple[list[_Candidate], list[dict[str, Any]]]:
    by_decision: dict[
        tuple[int, str, str],
        list[tuple[Mapping[str, Any], _Opportunity]],
    ] = defaultdict(list)
    for prediction in predictions:
        opportunity = opportunity_by_id.get(
            str(prediction.get("opportunity_id") or "")
        )
        if opportunity is None:
            raise StrategyBacktestError(
                "optimizer prediction references unknown opportunity"
            )
        key = (
            int(prediction["fold_index"]),
            opportunity.symbol,
            opportunity.decision_time.isoformat(),
        )
        by_decision[key].append((prediction, opportunity))

    selected: list[_Candidate] = []
    audit: list[dict[str, Any]] = []
    period_by_index = {period.fold_index: period for period in periods}
    for key, candidates in sorted(by_decision.items()):
        fold_index, symbol, _ = key
        period = period_by_index[fold_index]
        statuses = {
            str(prediction.get("model_status") or "")
            for prediction, _ in candidates
        }
        if len(statuses) != 1:
            raise StrategyBacktestError(
                f"fold {fold_index}: inconsistent model statuses "
                f"inside one decision: {sorted(statuses)}"
            )
        model_status = next(iter(statuses))
        if model_status != "FIT":
            audit.append(
                {
                    "schema": COUNTERFACTUAL_SCHEMA,
                    "replay_id": None,
                    "fold_index": fold_index,
                    "test_start": period.test_start.isoformat(),
                    "test_end": period.test_end.isoformat(),
                    "symbol": symbol,
                    "decision_time": candidates[0][
                        1
                    ].decision_time.isoformat(),
                    "opportunity_id": None,
                    "side": None,
                    "trigger_kind": None,
                    "model_id": candidates[0][0].get("model_id"),
                    "model_status": model_status,
                    "score": None,
                    "hard_score_threshold_applied": False,
                    "eligible_pool": "none",
                    "alternatives": len(candidates),
                    "selection_status": "SKIPPED_NOT_FIT",
                }
            )
            continue
        def rank(
            item: tuple[Mapping[str, Any], _Opportunity],
        ) -> tuple[float, int, str]:
            prediction, opportunity = item
            raw_score = prediction.get("predicted_opportunity_r")
            if raw_score is None:
                raise StrategyBacktestError(
                    f"fold {fold_index}: FIT model emitted no prediction"
                )
            score = float(raw_score or 0.0)
            return (
                -score,
                _TRIGGER_PRIORITY.get(opportunity.trigger_kind, 999),
                opportunity.opportunity_id,
            )

        ordered = sorted(candidates, key=rank)
        for rank_index, (prediction, opportunity) in enumerate(
            ordered,
            start=1,
        ):
            replay_id: Optional[str] = None
            signal = dict(opportunity.signal)
            signal["optimizer_model_id"] = prediction.get("model_id")
            signal["optimizer_model_status"] = prediction.get(
                "model_status"
            )
            signal["optimizer_score"] = prediction.get(
                "predicted_opportunity_r"
            )
            signal["optimizer_rank"] = rank_index
            selection_status = "SKIPPED_PRE_GATE"
            if opportunity.gate == "ENTER":
                replay_id = _canonical_hash(
                    {
                        "schema": COUNTERFACTUAL_SCHEMA,
                        "arm": "optimized_weights",
                        "fold_index": fold_index,
                        "opportunity_id": opportunity.opportunity_id,
                        "model_id": prediction.get("model_id"),
                    }
                )[:24]
                selected.append(
                    _Candidate(
                        candidate_id=replay_id,
                        fold_index=fold_index,
                        symbol=symbol,
                        decision_time=opportunity.decision_time,
                        gate=opportunity.gate,
                        gate_reason=opportunity.gate_reason,
                        signal=signal,
                    )
                )
                selection_status = "RANKED_FOR_REPLAY"
            audit.append(
                {
                    "schema": COUNTERFACTUAL_SCHEMA,
                    "replay_id": replay_id,
                    "fold_index": fold_index,
                    "test_start": period.test_start.isoformat(),
                    "test_end": period.test_end.isoformat(),
                    "symbol": symbol,
                    "decision_time": opportunity.decision_time.isoformat(),
                    "opportunity_id": opportunity.opportunity_id,
                    "side": signal.get("side"),
                    "trigger_kind": opportunity.trigger_kind,
                    "model_id": prediction.get("model_id"),
                    "model_status": prediction.get("model_status"),
                    "score": signal.get("optimizer_score"),
                    "rank": rank_index,
                    "hard_score_threshold_applied": False,
                    "eligible_pool": (
                        "gate_enter"
                        if opportunity.gate == "ENTER"
                        else "pre_gate_blocked"
                    ),
                    "alternatives": len(candidates),
                    "selection_status": selection_status,
                }
            )
    return selected, audit


def _run_selected_replay(
    *,
    selected: Sequence[_Candidate],
    prepared_by_symbol: Mapping[str, Mapping[str, Any]],
    periods: Sequence[_Period],
    config: NarrativeBacktestConfig,
    progress: Optional[Callable[[Mapping[str, Any]], None]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    all_setups: list[dict[str, Any]] = []
    all_legs: list[dict[str, Any]] = []
    all_executions: list[dict[str, Any]] = []
    by_symbol: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in selected:
        by_symbol[candidate.symbol].append(candidate)

    for symbol in config.symbols:
        candidates = by_symbol.get(symbol, [])
        for policy in config.intrabar_policies:
            setups, legs, executions, _ = _execute_candidates(
                candidates=candidates,
                prepared=prepared_by_symbol[symbol],
                periods=periods,
                config=config,
                policy=policy,
                progress=progress,
                carry_positions_across_folds=True,
            )
            all_setups.extend(setups)
            all_legs.extend(legs)
            all_executions.extend(executions)

    fold_rows: list[dict[str, Any]] = []
    for policy in config.intrabar_policies:
        for period in periods:
            rows = [
                row
                for row in all_setups
                if row["policy"] == policy
                and row["fold_index"] == period.fold_index
            ]
            fold_rows.append(
                {
                    **period.to_dict(),
                    "policy": policy,
                    **_metrics_for(rows),
                }
            )
    return all_setups, all_legs, all_executions, fold_rows


def run_counterfactual_backtest(
    dataset: HistoricalDataset,
    config: NarrativeBacktestConfig,
    *,
    orderflow: Optional[AbsorptionEventDataset] = None,
    strategy_factory: StrategyFactory = _default_strategy_factory,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    min_train_opportunities: int = DEFAULT_MIN_TRAIN_OPPORTUNITIES,
    progress: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> CounterfactualBacktestResult:
    """Build all-side opportunities, fit train-only weights, replay frozen OOS."""

    if _PRIMARY_POLICY not in config.intrabar_policies:
        raise StrategyBacktestError(
            "Counterfactual optimizer requires stop-first as the primary "
            "causal label policy; use --intrabar-policy stop-first or both"
        )
    if config.train is None or config.test is None:
        raise StrategyBacktestError(
            "Counterfactual optimizer requires --train and --test"
        )
    unknown = sorted(set(config.symbols) - set(dataset.symbols))
    if unknown:
        raise StrategyBacktestError(
            f"Symbols are absent from snapshot: {', '.join(unknown)}"
        )
    coverage_start, coverage_end = infer_common_strategy_range(
        dataset,
        config.symbols,
    )
    if config.start < coverage_start or config.end > coverage_end:
        raise StrategyBacktestError(
            "Requested decision range is outside common M15 coverage"
        )
    periods = _periods(config)
    prepared_by_symbol = {
        symbol: _prepare_symbol(dataset, symbol)
        for symbol in config.symbols
    }
    decision_rows: list[dict[str, Any]] = []
    opportunities: list[_Opportunity] = []
    signal_counters: Counter[str] = Counter()
    for symbol in config.symbols:
        decisions, symbol_opportunities, counters = (
            _generate_symbol_universe(
                dataset=dataset,
                symbol=symbol,
                prepared=prepared_by_symbol[symbol],
                periods=periods,
                config=config,
                orderflow=orderflow,
                strategy_factory=strategy_factory,
                progress=progress,
            )
        )
        decision_rows.extend(decisions)
        opportunities.extend(symbol_opportunities)
        signal_counters.update(counters)

    labels: list[dict[str, Any]] = []
    for symbol in config.symbols:
        prepared = prepared_by_symbol[symbol]
        m1 = prepared["1m"].frame
        m1_execution = m1.drop(
            columns=[_BAR_CLOSE_COLUMN],
            errors="ignore",
        )
        available_end = min(
            config.end,
            pd.Timestamp(m1.index[-1]) + pd.Timedelta(minutes=1),
        )
        symbol_opportunities = [
            item for item in opportunities if item.symbol == symbol
        ]
        total = len(symbol_opportunities)
        _emit(
            progress,
            phase="counterfactual-labels",
            state="start",
            symbol=symbol,
            candidates=total,
        )
        for completed, opportunity in enumerate(
            symbol_opportunities,
            start=1,
        ):
            labels.append(
                _label_opportunity(
                    opportunity=opportunity,
                    m1=m1_execution,
                    config=config,
                    available_end=available_end,
                )
            )
            if total and (
                completed == total
                or completed == 1
                or completed % max(1, total // 100) == 0
            ):
                _emit(
                    progress,
                    phase="counterfactual-labels",
                    state="progress",
                    symbol=symbol,
                    completed=completed,
                    candidates=total,
                    percent=int(completed * 100 / total),
                )
        _emit(
            progress,
            phase="counterfactual-labels",
            state="complete",
            symbol=symbol,
            completed=total,
            candidates=total,
        )
    optimizer = build_counterfactual_weight_scores(
        opportunities=labels,
        periods=[period.to_dict() for period in periods],
        entry_ttl=config.entry_ttl,
        max_holding=config.max_holding,
        ridge_alpha=ridge_alpha,
        min_train_opportunities=min_train_opportunities,
    )
    opportunity_by_id = {
        opportunity.opportunity_id: opportunity
        for opportunity in opportunities
    }
    selected, selection_rows = _select_oos_candidates(
        predictions=optimizer["predictions"],
        opportunity_by_id=opportunity_by_id,
        periods=periods,
    )
    setups, legs, executions, fold_rows = _run_selected_replay(
        selected=selected,
        prepared_by_symbol=prepared_by_symbol,
        periods=periods,
        config=config,
        progress=progress,
    )

    primary_setups = [
        row for row in setups if row["policy"] == _PRIMARY_POLICY
    ]
    force_exit_counts = Counter(
        str(row.get("forced_exit_reason"))
        for row in primary_setups
        if row.get("forced_exit_reason")
    )
    label_counts = Counter(str(row["label_status"]) for row in labels)
    enabled_triggers = ["h1_pivot_reclaim_15m"]
    if config.orderblock_entry_enabled:
        enabled_triggers.append("order_block_1h")
    if config.rejection_block_entry_enabled:
        enabled_triggers.append("rejection_block_15m")
    if orderflow is not None:
        enabled_triggers.append("absorption_15m")
    enabled_triggers = sorted(
        set(enabled_triggers),
        key=lambda item: _TRIGGER_PRIORITY.get(item, 999),
    )
    generated_triggers = sorted(
        {item.trigger_kind for item in opportunities},
        key=lambda item: _TRIGGER_PRIORITY.get(item, 999),
    )
    config_payload = {
        **config.to_dict(),
        "counterfactual_optimizer": {
            "schema": COUNTERFACTUAL_SCHEMA,
            "ridge_alpha": float(ridge_alpha),
            "min_train_opportunities": int(min_train_opportunities),
            "trigger_manifest": list(TRIGGER_MANIFEST),
            "enabled_trigger_manifest": enabled_triggers,
            "turtle_soup_enabled": False,
            "score_threshold": None,
            "primary_label_policy": _PRIMARY_POLICY,
        },
        "orderflow_manifest_sha256": (
            orderflow.manifest_sha256 if orderflow is not None else None
        ),
    }
    summary = {
        "schema": COUNTERFACTUAL_SCHEMA,
        "dataset": str(dataset.root),
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "config_sha256": _canonical_hash(config_payload),
        "decision_events": len(decision_rows),
        "technical_opportunities": len(opportunities),
        "label_status_counts": dict(sorted(label_counts.items())),
        "signal_counters": dict(sorted(signal_counters.items())),
        "optimizer": optimizer["summary"],
        "oos_selected_candidates": len(selected),
        "oos_ranked_candidates": len(selected),
        "oos_ranked_decisions": len(
            {
                (
                    row["fold_index"],
                    row["symbol"],
                    row["decision_time"],
                )
                for row in selection_rows
                if row.get("selection_status") == "RANKED_FOR_REPLAY"
            }
        ),
        "oos_decisions_skipped_not_fit": sum(
            row.get("selection_status") == "SKIPPED_NOT_FIT"
            for row in selection_rows
        ),
        "oos_primary_metrics": _metrics_for(primary_setups),
        "oos_primary_force_exit_counts": dict(
            sorted(force_exit_counts.items())
        ),
        "risk": {
            "basis": "initial_capital_fixed",
            "initial_capital": config.initial_capital,
            "risk_fraction": config.risk_fraction,
            "risk_amount_per_setup": (
                config.initial_capital * config.risk_fraction
            ),
            "maximum_fraction": 0.01,
            "changed_by_optimizer": False,
        },
        "trigger_manifest": list(TRIGGER_MANIFEST),
        "supported_trigger_manifest": list(TRIGGER_MANIFEST),
        "enabled_trigger_manifest": enabled_triggers,
        "generated_trigger_manifest": generated_triggers,
        "turtle_soup_called": False,
        "absorption": {
            "enabled": orderflow is not None,
            "orderflow_sidecar_loaded": orderflow is not None,
            "orderflow_manifest_sha256": (
                orderflow.manifest_sha256 if orderflow is not None else None
            ),
            "ohlcv_proxy_used": False,
        },
        "assumptions": {
            "generation": (
                "every completed M15 decision across every outer train/test "
                "window; LONG and SHORT detectors called independently of "
                "the production bias and live trigger switches; operationally "
                "disabled triggers remain blocked only in OOS replay"
            ),
            "labels": (
                "independent technical opportunities before active-setup, "
                "cooldown, daily-cap and duplicate-trigger state; fixed "
                "operational gates retained for audit/replay but do not "
                "exclude technically valid training labels"
            ),
            "no_fill": "explicit 0R opportunity label",
            "purge": (
                "entry_ttl + max_holding, with exit_time before train_end"
            ),
            "selection": (
                "continuous frozen OOS score; no score threshold; all "
                "pre-gate-eligible alternatives are replayed in rank order "
                "until one fills; fixed trigger priority breaks score ties; "
                "NOT_FIT folds do not trade"
            ),
            "trigger_optimization": (
                "factor weights optimize direction alignment only; trigger "
                "families remain separate observations and trigger "
                "arbitration is fixed, not fitted"
            ),
            "oos_metrics": (
                "integrated replay is authoritative; optimizer label metrics "
                "exclude outcomes whose exit is not known before test_end"
            ),
            "fold_boundaries": (
                "entries and open positions carry across artificial OOS fold "
                "boundaries; only Friday, max holding, stop/target, or final "
                "dataset end closes exposure"
            ),
            "costs_applied": False,
        },
        "warnings": [
            (
                "This is a gross OHLCV research replay. Spread, commission, "
                "swap, slippage and historical tick ordering are unavailable."
            ),
            (
                "Absorption is DATA_UNAVAILABLE unless a sealed footprint "
                "sidecar supplies executed Bid/Ask volume at price. LSE OHLCV "
                "is never converted into a proxy."
            ),
            (
                "Removing Turtle Soup was decided after observing historical "
                "performance; final confirmation requires a new holdout or "
                "forward shadow/paper period."
            ),
        ],
    }
    return CounterfactualBacktestResult(
        summary=_safe_json_value(summary),
        config=_safe_json_value(config_payload),
        decision_events=tuple(
            _safe_json_value(row)
            for row in sorted(
                decision_rows,
                key=lambda item: (
                    item["decision_time"],
                    item["symbol"],
                ),
            )
        ),
        opportunities=tuple(
            _safe_json_value(item.report_row())
            for item in opportunities
        ),
        labels=tuple(_safe_json_value(row) for row in labels),
        models=_safe_json_value(
            {
                "summary": optimizer["summary"],
                "models": optimizer["models"],
            }
        ),
        predictions=tuple(
            _safe_json_value(row) for row in optimizer["predictions"]
        ),
        coefficients=tuple(
            _safe_json_value(row) for row in optimizer["coefficients"]
        ),
        optimizer_metrics=tuple(
            _safe_json_value(row) for row in optimizer["metrics"]
        ),
        selections=tuple(_safe_json_value(row) for row in selection_rows),
        executions=tuple(
            _safe_json_value(row)
            for row in sorted(
                executions,
                key=lambda item: (
                    item["policy"],
                    item["decision_time"],
                    item["symbol"],
                ),
            )
        ),
        setups=tuple(
            _safe_json_value(row)
            for row in sorted(
                setups,
                key=lambda item: (
                    item["policy"],
                    item["decision_time"],
                    item["symbol"],
                ),
            )
        ),
        legs=tuple(_safe_json_value(row) for row in legs),
        folds=tuple(_safe_json_value(row) for row in fold_rows),
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str],
) -> None:
    import csv

    declared = tuple(fields)
    unexpected = sorted(
        {
            str(field)
            for row in rows
            for field in row
            if field not in declared
        }
    )
    if unexpected:
        raise StrategyBacktestError(
            f"{path.name} contains undeclared columns: {unexpected}"
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=declared)
        writer.writeheader()
        for row in rows:
            encoded: dict[str, Any] = {}
            for field in declared:
                value = _safe_json_value(row.get(field))
                if isinstance(value, (dict, list)):
                    value = json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                encoded[field] = value
            writer.writerow(encoded)


def write_counterfactual_report(
    result: CounterfactualBacktestResult,
    output: str | Path,
) -> Path:
    """Atomically publish a deterministic counterfactual WFO report."""

    target = Path(output).expanduser().resolve()
    if target.exists():
        raise StrategyBacktestError(f"Output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.partial-",
            dir=target.parent,
        )
    )
    try:
        (staging / "config.json").write_bytes(_json_bytes(result.config))
        (staging / "summary.json").write_bytes(_json_bytes(result.summary))
        (staging / "frozen_weight_models.json").write_bytes(
            _json_bytes(result.models)
        )
        tables = {
            "decision_events.csv": result.decision_events,
            "technical_opportunities.csv": result.opportunities,
            "opportunity_labels.csv": result.labels,
            "optimizer_predictions.csv": result.predictions,
            "optimizer_coefficients.csv": result.coefficients,
            "optimizer_metrics.csv": result.optimizer_metrics,
            "oos_selections.csv": result.selections,
            "executions.csv": result.executions,
            "setups.csv": result.setups,
            "legs.csv": result.legs,
            "folds.csv": result.folds,
        }
        for filename, rows in tables.items():
            _write_csv(
                staging / filename,
                rows,
                fields=_REPORT_FIELDS[filename],
            )
        files = [
            {
                "path": path.name,
                "size": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
            for path in sorted(staging.iterdir(), key=lambda item: item.name)
            if path.is_file()
        ]
        manifest = {
            "schema": COUNTERFACTUAL_SCHEMA,
            "dataset_manifest_sha256": result.summary[
                "dataset_manifest_sha256"
            ],
            "config_sha256": result.summary["config_sha256"],
            "files": files,
        }
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


__all__ = [
    "COUNTERFACTUAL_SCHEMA",
    "CounterfactualBacktestResult",
    "TRIGGER_MANIFEST",
    "run_counterfactual_backtest",
    "write_counterfactual_report",
]
