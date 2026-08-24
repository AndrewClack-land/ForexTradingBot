"""Causal offline runner for the production narrative strategy.

The runner deliberately stays separate from ``main.Core`` and MetaTrader 5.
Signals are calculated only from candles closed at the decision timestamp,
then an entry can be filled only from later M1 opens.  Execution results are
reported in setup R and can be scaled by a fixed percentage of the explicitly
provided starting capital.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import tempfile
import types
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from core.htf_context import PineRejectionBlockTracker
from core.narrative_scoring import (
    FACTOR_CONTRACT_NO_H1PD,
    FACTOR_VECTOR_SCHEMA,
    FACTOR_WEIGHTS,
    build_factor_vector,
    resolve_factor_contract,
)

from .attribution import (
    attribution_coverage,
    candidate_factor_rows,
    factor_summary_rows,
    factor_vector_fields,
    setup_factor_rows,
)
from .data import HistoricalDataset, LIVE_CLOSED_BAR_LIMIT
from .metrics import aggregate_setup_metrics
from .optimizer import build_shadow_scores
from .simulator import LegOutcome, SetupOutcome, simulate_split_outcome
from .walkforward import split_walk_forward


ProgressCallback = Callable[[Mapping[str, Any]], None]
StrategyFactory = Callable[[], Any]

REPORT_SCHEMA = "narrative-backtest/v2"
RELEASE_MANIFEST_SCHEMA = "forexbot-backtest-release/v1"
_BAR_CLOSE_COLUMN = "bar_close_time"
_REQUIRED_TIMEFRAMES = ("1m", "15m", "1h", "4h", "1d")
_TIMEFRAME_DURATION = {
    "1m": pd.Timedelta(minutes=1),
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}
_STRATEGY_KEY = {
    "1d": "D",
    "4h": "4H",
    "1h": "1H",
    "15m": "15M",
}
_SESSION_WINDOWS = {
    "ASIA": ("00:00", "08:00"),
    "LONDON": ("06:30", "15:30"),
    "NY": ("12:00", "21:00"),
}
_REQUIRED_RELEASE_FILES = {
    "backtest/__init__.py",
    "backtest/__main__.py",
    "backtest/attribution.py",
    "backtest/counterfactual.py",
    "backtest/data.py",
    "backtest/lse_ingest.py",
    "backtest/fxpro_cluster_data.py",
    "backtest/fxpro_quote_pressure_data.py",
    "backtest/metrics.py",
    "backtest/optimizer.py",
    "backtest/orderflow_data.py",
    "backtest/simulator.py",
    "backtest/strategy_runner.py",
    "backtest/walkforward.py",
    "backtest/weight_optimizer.py",
    "core/__init__.py",
    "core/absorption.py",
    "core/htf_context.py",
    "core/fxpro_cluster_rejection.py",
    "core/fxpro_quote_pressure.py",
    "core/narrative_scoring.py",
    "core/pivot_trigger.py",
    "core/strategy_narrative.py",
    "core/vol_regime.py",
    "deploy/build_backtest_release_manifest.py",
    "requirements-backtest.txt",
}
_EMPTY_CSV_FIELDS = {
    "candidates": (
        "candidate_id",
        "fold_index",
        "symbol",
        "decision_time",
        "side",
        "trigger_kind",
        "trigger_reason",
        "entry_min",
        "entry_max",
        "planned_entry",
        "stop",
        "tp_prices",
        "gate",
        "gate_reason",
        "vol_r",
        "vol_regime",
        "vol_em_1d",
        "vol_tp1_em_ratio",
        "fvg_regime",
        "narrative",
        "factor_schema",
        "factor_bias",
        "factor_score_long",
        "factor_score_short",
        "factor_score_delta",
        "factor_base_margin",
        "factor_margin_long",
        "factor_margin_short",
        "factor_fvg_side",
        "factor_aligned_count",
        "factor_opposed_count",
        "factor_pivotal_count",
        "factor_vector",
    ),
    "executions": (
        "candidate_id",
        "policy",
        "fold_index",
        "symbol",
        "decision_time",
        "trigger_kind",
        "pre_gate",
        "pre_gate_reason",
        "disposition",
        "disposition_reason",
        "fill_time",
        "fill_price",
        "setup_id",
        "blocked_until",
        "blocking_setup_id",
    ),
    "setups": (
        "setup_id",
        "candidate_id",
        "policy",
        "fold_index",
        "symbol",
        "decision_time",
        "entry_time",
        "exit_time",
        "side",
        "entry",
        "planned_entry",
        "entry_min",
        "entry_max",
        "stop",
        "tp_prices",
        "status",
        "net_r",
        "risk_fraction",
        "risk_amount",
        "pnl_amount",
        "tp_hits",
        "moved_to_be",
        "ambiguous_bars",
        "bars_processed",
        "trigger_kind",
        "trigger_reason",
        "fvg_regime",
        "vol_r",
        "vol_regime",
        "vol_em_1d",
        "vol_tp1_em_ratio",
        "narrative",
        "factor_schema",
        "factor_bias",
        "factor_score_long",
        "factor_score_short",
        "factor_score_delta",
        "factor_base_margin",
        "factor_margin_long",
        "factor_margin_short",
        "factor_fvg_side",
        "factor_aligned_count",
        "factor_opposed_count",
        "factor_pivotal_count",
        "factor_vector",
        "forced_exit_reason",
    ),
    "legs": (
        "setup_id",
        "candidate_id",
        "policy",
        "tp_index",
        "weight",
        "exit_reason",
        "exit_price",
        "r_multiple",
        "exit_time",
    ),
    "candidate_factors": (
        "attribution_schema",
        "candidate_id",
        "event_id",
        "fold_index",
        "symbol",
        "decision_time",
        "candidate_side",
        "trigger_kind",
        "factor_schema",
        "factor_key",
        "factor_label",
        "present",
        "vote_side",
        "relation",
        "alignment",
        "configured_weight",
        "effective_weight",
        "long_contribution",
        "short_contribution",
        "candidate_contribution",
        "pivotal_without_factor",
        "bias_without_factor",
        "age_bars",
        "score_long",
        "score_short",
        "score_delta",
        "selected_bias",
        "base_margin",
        "margin_long",
        "margin_short",
        "fvg_side",
        "vol_r",
        "vol_regime",
        "vol_em_1d",
        "vol_tp1_em_ratio",
        "evidence",
        "gate",
        "gate_reason",
    ),
    "setup_factors": (
        "attribution_schema",
        "candidate_id",
        "event_id",
        "setup_id",
        "policy",
        "fold_index",
        "symbol",
        "decision_time",
        "candidate_side",
        "trigger_kind",
        "factor_schema",
        "factor_key",
        "factor_label",
        "present",
        "vote_side",
        "relation",
        "alignment",
        "configured_weight",
        "effective_weight",
        "long_contribution",
        "short_contribution",
        "candidate_contribution",
        "pivotal_without_factor",
        "bias_without_factor",
        "age_bars",
        "score_long",
        "score_short",
        "score_delta",
        "selected_bias",
        "base_margin",
        "margin_long",
        "margin_short",
        "fvg_side",
        "entry_time",
        "exit_time",
        "status",
        "net_r",
        "pnl_amount",
        "decision_year",
        "decision_quarter",
        "vol_r",
        "vol_regime",
        "vol_em_1d",
        "vol_tp1_em_ratio",
        "evidence",
    ),
    "factor_summary": (
        "attribution_schema",
        "policy",
        "factor_key",
        "factor_label",
        "configured_weight",
        "relation",
        "dimension",
        "dimension_value",
        "setups",
        "net_r",
        "expectancy_r",
        "shrunken_expectancy_r",
        "prior_strength",
        "win_rate",
        "profit_factor",
        "max_drawdown_r",
        "average_win_r",
        "average_loss_r",
        "wins",
        "losses",
        "sample_quality",
        "interpretation",
    ),
    "shadow_predictions": (
        "schema",
        "setup_id",
        "candidate_id",
        "fold_index",
        "symbol",
        "decision_time",
        "side",
        "trigger_kind",
        "policy",
        "model_status",
        "model_id",
        "train_setups",
        "predicted_expected_r",
        "actual_net_r",
        "prediction_error_r",
    ),
    "shadow_coefficients": (
        "schema",
        "model_id",
        "fold_index",
        "train_start",
        "purge_cutoff",
        "test_start",
        "train_setups",
        "ridge_alpha",
        "feature",
        "coefficient",
    ),
    "shadow_metrics": (
        "schema",
        "dimension",
        "dimension_value",
        "setups",
        "actual_expectancy_r",
        "predicted_mean_r",
        "mae_r",
        "rmse_r",
        "rank_ic",
        "top_minus_bottom_quintile_r",
    ),
}


class StrategyBacktestError(ValueError):
    """Raised when a strategy run cannot be completed safely."""


def _utc(value: Any, *, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise StrategyBacktestError(f"Invalid {name}: {value!r}") from exc
    if pd.isna(timestamp):
        raise StrategyBacktestError(f"Invalid {name}: {value!r}")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _duration(value: Any, *, name: str) -> pd.Timedelta:
    try:
        duration = pd.Timedelta(value)
    except Exception as exc:
        raise StrategyBacktestError(f"Invalid {name}: {value!r}") from exc
    if pd.isna(duration) or duration <= pd.Timedelta(0):
        raise StrategyBacktestError(f"{name} must be positive")
    return duration


def _parse_clock(value: str) -> time:
    try:
        hour, minute = (int(part) for part in value.split(":", maxsplit=1))
        return time(hour=hour, minute=minute)
    except (TypeError, ValueError) as exc:
        raise StrategyBacktestError(f"Invalid session clock: {value!r}") from exc


def _time_in_window(now: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= now < end
    return now >= start or now < end


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {
            str(key): _safe_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe_json_value(item) for item in value]
    return value


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


def _canonical_hash(payload: Any) -> str:
    raw = json.dumps(
        _safe_json_value(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_release_manifest(
    path: str | Path,
    *,
    expected_commit: str,
) -> str:
    """Verify root-owned release file hashes against the running app tree."""

    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StrategyBacktestError(
            f"{manifest_path}: invalid release manifest"
        ) from exc
    if payload.get("schema") != RELEASE_MANIFEST_SCHEMA:
        raise StrategyBacktestError(
            f"{manifest_path}: unexpected release manifest schema"
        )
    if str(payload.get("release_commit") or "").lower() != expected_commit.lower():
        raise StrategyBacktestError(
            f"{manifest_path}: release commit does not match RELEASE_COMMIT"
        )
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise StrategyBacktestError(
            f"{manifest_path}: release files must be an object"
        )
    declared = set(str(name) for name in files)
    missing = sorted(_REQUIRED_RELEASE_FILES - declared)
    if missing:
        raise StrategyBacktestError(
            f"{manifest_path}: required release files are missing: "
            f"{', '.join(missing)}"
        )

    app_root = Path(__file__).resolve().parent.parent
    for relative_name, expected_sha in sorted(files.items()):
        relative = Path(str(relative_name))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
        ):
            raise StrategyBacktestError(
                f"{manifest_path}: unsafe release path {relative_name!r}"
            )
        expected = str(expected_sha).lower()
        if not _is_sha256(expected):
            raise StrategyBacktestError(
                f"{manifest_path}: invalid SHA-256 for {relative_name}"
            )
        lexical_path = app_root / relative
        if lexical_path.is_symlink():
            raise StrategyBacktestError(
                f"{manifest_path}: release file cannot be a symlink: "
                f"{relative_name}"
            )
        source_path = lexical_path.resolve()
        try:
            source_path.relative_to(app_root)
        except ValueError as exc:
            raise StrategyBacktestError(
                f"{manifest_path}: release path escapes app root"
            ) from exc
        if not source_path.is_file():
            raise StrategyBacktestError(
                f"{manifest_path}: missing regular file {relative_name}"
            )
        actual = _sha256_file(source_path)
        if actual != expected:
            raise StrategyBacktestError(
                f"{manifest_path}: hash mismatch for {relative_name}"
            )
    return _sha256_file(manifest_path)


def _is_sha256(value: str) -> bool:
    return (
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class NarrativeBacktestConfig:
    symbols: tuple[str, ...]
    start: pd.Timestamp
    end: pd.Timestamp
    initial_capital: float
    risk_fraction: float = 0.01
    history_limit: int = LIVE_CLOSED_BAR_LIMIT
    min_context_bars: int = LIVE_CLOSED_BAR_LIMIT
    min_daily_bars: int = 22
    entry_ttl: pd.Timedelta = pd.Timedelta(minutes=15)
    max_holding: pd.Timedelta = pd.Timedelta(days=30)
    intrabar_policies: tuple[str, ...] = ("stop-first",)
    profile: str = "production-deterministic"
    train: Optional[pd.Timedelta] = None
    test: Optional[pd.Timedelta] = None
    step: Optional[pd.Timedelta] = None
    session_timezone: str = "UTC"
    allowed_sessions: tuple[str, ...] = ("LONDON", "NY")
    friday_close_hour_moscow: int = 21
    vol_filter_enabled: bool = True
    vol_max_r: float = 60.0
    em_tp_ratio: float = 1.0
    max_setups_per_symbol_day: int = 3
    post_loss_cooldown: pd.Timedelta = pd.Timedelta(minutes=60)
    rejection_block_entry_enabled: bool = False
    # Research-only OOS permission for the live H1 RB detector. This is
    # intentionally independent from the retired M15 RB switch above.
    rejection_block_h1_oos_enabled: bool = False
    rejection_block_h1_detector_version: str = (
        PineRejectionBlockTracker.VERSION
    )
    orderblock_entry_enabled: bool = True
    orderblock_touch_atr_k: float = 0.15
    orderblock_touch_min_abs: float = 0.0005
    orderblock_max_age_bars: int = 80
    htf_score_margin: int = 2
    fvg_regime_max_age_bars: int = 0
    fvg_event_invalidation_mode: str = "none"
    factor_contract: str = FACTOR_CONTRACT_NO_H1PD
    fvg_veto_enabled: bool = False
    release_commit: Optional[str] = None
    release_manifest_sha256: Optional[str] = None
    environment_lock_sha256: Optional[str] = None

    @classmethod
    def build(
        cls,
        *,
        symbols: Sequence[str],
        start: Any,
        end: Any,
        initial_capital: float,
        risk_fraction: float = 0.01,
        history_limit: int = LIVE_CLOSED_BAR_LIMIT,
        min_context_bars: int = LIVE_CLOSED_BAR_LIMIT,
        min_daily_bars: int = 22,
        entry_ttl: Any = "15min",
        max_holding: Any = "30D",
        intrabar_policy: str = "stop-first",
        profile: str = "production-deterministic",
        train: Any | None = None,
        test: Any | None = None,
        step: Any | None = None,
        session_timezone: str = "UTC",
        allowed_sessions: Sequence[str] = ("LONDON", "NY"),
        friday_close_hour_moscow: int = 21,
        vol_filter_enabled: bool = True,
        vol_max_r: float = 60.0,
        em_tp_ratio: float = 1.0,
        max_setups_per_symbol_day: int = 3,
        post_loss_cooldown: Any = "60min",
        rejection_block_entry_enabled: bool = False,
        rejection_block_h1_oos_enabled: bool = False,
        rejection_block_h1_detector_version: str = (
            PineRejectionBlockTracker.VERSION
        ),
        orderblock_entry_enabled: bool = True,
        orderblock_touch_atr_k: float = 0.15,
        orderblock_touch_min_abs: float = 0.0005,
        orderblock_max_age_bars: int = 80,
        htf_score_margin: int = 2,
        fvg_regime_max_age_bars: int = 0,
        fvg_event_invalidation_mode: str = "none",
        factor_contract: str = FACTOR_CONTRACT_NO_H1PD,
        fvg_veto_enabled: bool = False,
        release_commit: Optional[str] = None,
        release_manifest_sha256: Optional[str] = None,
        environment_lock_sha256: Optional[str] = None,
    ) -> "NarrativeBacktestConfig":
        normalized_symbols = tuple(
            dict.fromkeys(str(symbol).strip().upper() for symbol in symbols)
        )
        if not normalized_symbols or any(not symbol for symbol in normalized_symbols):
            raise StrategyBacktestError("At least one non-empty symbol is required")

        range_start = _utc(start, name="start")
        range_end = _utc(end, name="end")
        if range_end <= range_start:
            raise StrategyBacktestError("end must be after start")

        capital = float(initial_capital)
        if not math.isfinite(capital) or capital <= 0:
            raise StrategyBacktestError("initial_capital must be positive")
        risk = float(risk_fraction)
        if not math.isfinite(risk) or not 0 < risk <= 0.01:
            raise StrategyBacktestError(
                "risk_fraction must be positive and cannot exceed 0.01 (1%)"
            )
        if history_limit <= 0 or min_context_bars <= 0 or min_daily_bars <= 0:
            raise StrategyBacktestError("history limits must be positive")
        if min_context_bars > history_limit:
            raise StrategyBacktestError(
                "min_context_bars cannot exceed history_limit"
            )
        if min_daily_bars > history_limit:
            raise StrategyBacktestError(
                "min_daily_bars cannot exceed history_limit"
            )

        policy = str(intrabar_policy).lower()
        if policy == "both":
            policies = ("stop-first", "tp-first")
        elif policy in {"stop-first", "tp-first"}:
            policies = (policy,)
        else:
            raise StrategyBacktestError(
                "intrabar_policy must be stop-first, tp-first, or both"
            )

        normalized_profile = str(profile).strip().lower()
        if normalized_profile not in {"signal-quality", "production-deterministic"}:
            raise StrategyBacktestError(
                "profile must be signal-quality or production-deterministic"
            )

        train_delta = _duration(train, name="train") if train is not None else None
        test_delta = _duration(test, name="test") if test is not None else None
        if (train_delta is None) != (test_delta is None):
            raise StrategyBacktestError("train and test must be provided together")
        step_delta = _duration(step, name="step") if step is not None else None
        if step_delta is not None and test_delta is None:
            raise StrategyBacktestError("step requires train and test")
        if (
            test_delta is not None
            and step_delta is not None
            and step_delta < test_delta
        ):
            raise StrategyBacktestError(
                "step cannot be shorter than test; overlapping OOS windows "
                "would double-count setups"
            )

        sessions = tuple(
            dict.fromkeys(str(name).strip().upper() for name in allowed_sessions)
        )
        invalid_sessions = [
            name for name in sessions if name not in _SESSION_WINDOWS and name != "ALL"
        ]
        if invalid_sessions:
            raise StrategyBacktestError(
                f"Unknown sessions: {', '.join(invalid_sessions)}"
            )
        try:
            ZoneInfo(session_timezone)
        except Exception as exc:
            raise StrategyBacktestError(
                f"Unknown session timezone: {session_timezone}"
            ) from exc
        if not 0 <= int(friday_close_hour_moscow) <= 23:
            raise StrategyBacktestError(
                "friday_close_hour_moscow must be between 0 and 23"
            )
        if max_setups_per_symbol_day <= 0:
            raise StrategyBacktestError(
                "max_setups_per_symbol_day must be positive"
            )
        normalized_fvg_event_mode = str(
            fvg_event_invalidation_mode
        ).strip().lower()
        if normalized_fvg_event_mode not in {
            "none",
            "zone_break",
            "structure_change",
            "zone_or_structure",
        }:
            raise StrategyBacktestError(
                "fvg_event_invalidation_mode must be none, zone_break, "
                "structure_change, or zone_or_structure"
            )
        normalized_rb_detector = str(
            rejection_block_h1_detector_version
        ).strip().lower()
        if normalized_rb_detector != PineRejectionBlockTracker.VERSION:
            raise StrategyBacktestError(
                "rejection_block_h1_detector_version must be "
                f"{PineRejectionBlockTracker.VERSION!r}"
            )
        numeric_settings = {
            "vol_max_r": float(vol_max_r),
            "em_tp_ratio": float(em_tp_ratio),
            "orderblock_touch_atr_k": float(orderblock_touch_atr_k),
            "orderblock_touch_min_abs": float(orderblock_touch_min_abs),
        }
        non_finite = [
            name
            for name, value in numeric_settings.items()
            if not math.isfinite(value)
        ]
        if non_finite:
            raise StrategyBacktestError(
                f"Strategy settings must be finite: {', '.join(non_finite)}"
            )
        if numeric_settings["vol_max_r"] < 0:
            raise StrategyBacktestError("vol_max_r cannot be negative")
        if numeric_settings["em_tp_ratio"] < 0:
            raise StrategyBacktestError("em_tp_ratio cannot be negative")
        if numeric_settings["orderblock_touch_atr_k"] < 0:
            raise StrategyBacktestError(
                "orderblock_touch_atr_k cannot be negative"
            )
        if numeric_settings["orderblock_touch_min_abs"] < 0:
            raise StrategyBacktestError(
                "orderblock_touch_min_abs cannot be negative"
            )
        if int(orderblock_max_age_bars) <= 0:
            raise StrategyBacktestError(
                "orderblock_max_age_bars must be positive"
            )
        normalized_release_manifest = (
            str(release_manifest_sha256).strip().lower()
            if release_manifest_sha256
            else None
        )
        if (
            normalized_release_manifest is not None
            and not _is_sha256(normalized_release_manifest)
        ):
            raise StrategyBacktestError(
                "release_manifest_sha256 must be a 64-hex SHA-256"
            )
        normalized_environment_lock = (
            str(environment_lock_sha256).strip().lower()
            if environment_lock_sha256
            else None
        )
        if (
            normalized_environment_lock is not None
            and not _is_sha256(normalized_environment_lock)
        ):
            raise StrategyBacktestError(
                "environment_lock_sha256 must be a 64-hex SHA-256"
            )

        return cls(
            symbols=normalized_symbols,
            start=range_start,
            end=range_end,
            initial_capital=capital,
            risk_fraction=risk,
            history_limit=int(history_limit),
            min_context_bars=int(min_context_bars),
            min_daily_bars=int(min_daily_bars),
            entry_ttl=_duration(entry_ttl, name="entry_ttl"),
            max_holding=_duration(max_holding, name="max_holding"),
            intrabar_policies=policies,
            profile=normalized_profile,
            train=train_delta,
            test=test_delta,
            step=step_delta,
            session_timezone=str(session_timezone),
            allowed_sessions=sessions,
            friday_close_hour_moscow=int(friday_close_hour_moscow),
            vol_filter_enabled=bool(vol_filter_enabled),
            vol_max_r=numeric_settings["vol_max_r"],
            em_tp_ratio=numeric_settings["em_tp_ratio"],
            max_setups_per_symbol_day=int(max_setups_per_symbol_day),
            post_loss_cooldown=_duration(
                post_loss_cooldown,
                name="post_loss_cooldown",
            ),
            rejection_block_entry_enabled=bool(
                rejection_block_entry_enabled
            ),
            rejection_block_h1_oos_enabled=bool(
                rejection_block_h1_oos_enabled
            ),
            rejection_block_h1_detector_version=normalized_rb_detector,
            orderblock_entry_enabled=bool(orderblock_entry_enabled),
            orderblock_touch_atr_k=numeric_settings[
                "orderblock_touch_atr_k"
            ],
            orderblock_touch_min_abs=numeric_settings[
                "orderblock_touch_min_abs"
            ],
            orderblock_max_age_bars=int(orderblock_max_age_bars),
            htf_score_margin=max(1, int(htf_score_margin)),
            fvg_regime_max_age_bars=max(0, int(fvg_regime_max_age_bars)),
            fvg_event_invalidation_mode=(
                normalized_fvg_event_mode
            ),
            factor_contract=resolve_factor_contract(factor_contract)["name"],
            fvg_veto_enabled=bool(fvg_veto_enabled),
            release_commit=str(release_commit).strip() if release_commit else None,
            release_manifest_sha256=normalized_release_manifest,
            environment_lock_sha256=normalized_environment_lock,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbols": list(self.symbols),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "initial_capital": self.initial_capital,
            "risk_fraction": self.risk_fraction,
            "risk_basis": "initial_capital_fixed",
            "history_limit": self.history_limit,
            "min_context_bars": self.min_context_bars,
            "min_daily_bars": self.min_daily_bars,
            "entry_ttl": self.entry_ttl.isoformat(),
            "max_holding": self.max_holding.isoformat(),
            "intrabar_policies": list(self.intrabar_policies),
            "profile": self.profile,
            "walk_forward": {
                "train": self.train.isoformat() if self.train is not None else None,
                "test": self.test.isoformat() if self.test is not None else None,
                "step": self.step.isoformat() if self.step is not None else None,
            },
            "session_timezone": self.session_timezone,
            "allowed_sessions": list(self.allowed_sessions),
            "friday_close_hour_moscow": self.friday_close_hour_moscow,
            "vol_filter_enabled": self.vol_filter_enabled,
            "vol_max_r": self.vol_max_r,
            "em_tp_ratio": self.em_tp_ratio,
            "max_setups_per_symbol_day": self.max_setups_per_symbol_day,
            "post_loss_cooldown": self.post_loss_cooldown.isoformat(),
            "strategy_settings": {
                "rejection_block_entry_enabled": (
                    self.rejection_block_entry_enabled
                ),
                "rejection_block_h1_oos_enabled": (
                    self.rejection_block_h1_oos_enabled
                ),
                "rejection_block_h1_detector_version": (
                    self.rejection_block_h1_detector_version
                ),
                "orderblock_entry_enabled": self.orderblock_entry_enabled,
                "orderblock_touch_atr_k": self.orderblock_touch_atr_k,
                "orderblock_touch_min_abs": self.orderblock_touch_min_abs,
                "orderblock_max_age_bars": self.orderblock_max_age_bars,
                "htf_score_margin": self.htf_score_margin,
                "fvg_regime_max_age_bars": self.fvg_regime_max_age_bars,
                "fvg_event_invalidation_mode": (
                    self.fvg_event_invalidation_mode
                ),
                "factor_contract": self.factor_contract,
                "fvg_veto_enabled": self.fvg_veto_enabled,
            },
            "release_commit": self.release_commit,
            "release_manifest_sha256": self.release_manifest_sha256,
            "environment_lock_sha256": self.environment_lock_sha256,
        }


@dataclass(frozen=True)
class StrategyBacktestResult:
    summary: Mapping[str, Any]
    config: Mapping[str, Any]
    candidates: tuple[Mapping[str, Any], ...]
    executions: tuple[Mapping[str, Any], ...]
    setups: tuple[Mapping[str, Any], ...]
    legs: tuple[Mapping[str, Any], ...]
    folds: tuple[Mapping[str, Any], ...]
    candidate_factors: tuple[Mapping[str, Any], ...]
    setup_factors: tuple[Mapping[str, Any], ...]
    factor_summary: tuple[Mapping[str, Any], ...]
    shadow_models: Mapping[str, Any]
    shadow_predictions: tuple[Mapping[str, Any], ...]
    shadow_coefficients: tuple[Mapping[str, Any], ...]
    shadow_metrics: tuple[Mapping[str, Any], ...]

    def write(self, output: str | Path) -> Path:
        return write_strategy_report(self, output)


@dataclass(frozen=True)
class _Period:
    fold_index: int
    train_start: Optional[pd.Timestamp]
    train_end: Optional[pd.Timestamp]
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "train_start": (
                self.train_start.isoformat()
                if self.train_start is not None
                else None
            ),
            "train_end": (
                self.train_end.isoformat()
                if self.train_end is not None
                else None
            ),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


@dataclass
class _PreparedSeries:
    timeframe: str
    frame: pd.DataFrame
    close_times: pd.DatetimeIndex

    def asof(self, at: pd.Timestamp, *, limit: int) -> pd.DataFrame:
        right = int(self.close_times.searchsorted(at, side="right"))
        left = max(0, right - int(limit))
        return (
            self.frame.iloc[left:right]
            .drop(columns=[_BAR_CLOSE_COLUMN], errors="ignore")
            .copy()
        )


@dataclass(frozen=True)
class _Candidate:
    candidate_id: str
    fold_index: int
    symbol: str
    decision_time: pd.Timestamp
    gate: str
    gate_reason: str
    signal: Mapping[str, Any]

    def report_row(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "fold_index": self.fold_index,
            "symbol": self.symbol,
            "decision_time": self.decision_time.isoformat(),
            "side": self.signal.get("side"),
            "trigger_kind": _trigger_kind(self.signal),
            "trigger_reason": self.signal.get("trigger_reason"),
            "entry_min": self.signal.get("entry_min"),
            "entry_max": self.signal.get("entry_max"),
            "planned_entry": self.signal.get("entry_price"),
            "stop": self.signal.get("stop_price"),
            "tp_prices": list(self.signal.get("tp_prices") or ()),
            "gate": self.gate,
            "gate_reason": self.gate_reason,
            "vol_r": self.signal.get("vol_R"),
            "vol_regime": self.signal.get("vol_regime"),
            "vol_em_1d": self.signal.get("vol_em_1d"),
            "vol_tp1_em_ratio": self.signal.get(
                "vol_tp1_em_ratio"
            ),
            "fvg_regime": self.signal.get("fvg_regime"),
            "narrative": self.signal.get("narrative"),
            **factor_vector_fields(self.signal.get("factor_vector")),
        }


def _close_times(frame: pd.DataFrame, timeframe: str) -> pd.DatetimeIndex:
    if _BAR_CLOSE_COLUMN in frame.columns:
        parsed = pd.to_datetime(frame[_BAR_CLOSE_COLUMN], utc=True, errors="coerce")
        close_times = pd.DatetimeIndex(parsed)
        if close_times.hasnans:
            raise StrategyBacktestError(
                f"{timeframe}: invalid explicit bar close timestamp"
            )
        return close_times
    return pd.DatetimeIndex(frame.index + _TIMEFRAME_DURATION[timeframe])


def _prepare_symbol(
    dataset: HistoricalDataset,
    symbol: str,
) -> dict[str, _PreparedSeries]:
    available = {
        timeframe
        for candidate, timeframe in dataset.keys
        if candidate == symbol
    }
    missing = sorted(set(_REQUIRED_TIMEFRAMES) - available)
    if missing:
        raise StrategyBacktestError(
            f"{symbol}: missing required timeframes: {', '.join(missing)}"
        )

    prepared: dict[str, _PreparedSeries] = {}
    for timeframe in _REQUIRED_TIMEFRAMES:
        frame = dataset.get_frame(symbol, timeframe)
        if frame.empty:
            raise StrategyBacktestError(f"{symbol} {timeframe}: empty series")
        if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
            raise StrategyBacktestError(
                f"{symbol} {timeframe}: timestamps are not sorted and unique"
            )
        prepared[timeframe] = _PreparedSeries(
            timeframe=timeframe,
            frame=frame,
            close_times=_close_times(frame, timeframe),
        )
    return prepared


def infer_common_strategy_range(
    dataset: HistoricalDataset,
    symbols: Sequence[str],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    normalized_symbols = tuple(
        dict.fromkeys(str(symbol).strip().upper() for symbol in symbols)
    )
    if not normalized_symbols or any(not symbol for symbol in normalized_symbols):
        raise StrategyBacktestError("At least one non-empty symbol is required")
    unknown = sorted(set(normalized_symbols) - set(dataset.symbols))
    if unknown:
        raise StrategyBacktestError(
            f"Symbols are absent from snapshot: {', '.join(unknown)}"
        )

    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for normalized in normalized_symbols:
        available = {
            timeframe
            for candidate, timeframe in dataset.keys
            if candidate == normalized
        }
        if "15m" not in available:
            raise StrategyBacktestError(
                f"{normalized}: missing required timeframe: 15m"
            )
        frame = dataset.get_frame(normalized, "15m")
        closes = _close_times(frame, "15m")
        if not len(closes):
            raise StrategyBacktestError(f"{normalized}: 15m series is empty")
        starts.append(pd.Timestamp(frame.index[0]))
        ends.append(closes[-1] + pd.Timedelta(nanoseconds=1))
    start = max(starts)
    end = min(ends)
    if end <= start:
        raise StrategyBacktestError("Symbols have no common 15m coverage")
    return start, end


def _periods(config: NarrativeBacktestConfig) -> tuple[_Period, ...]:
    if config.train is None or config.test is None:
        return (
            _Period(
                fold_index=0,
                train_start=None,
                train_end=None,
                test_start=config.start,
                test_end=config.end,
            ),
        )
    folds = split_walk_forward(
        start=config.start,
        end=config.end,
        train=config.train,
        test=config.test,
        step=config.step,
    )
    if not folds:
        raise StrategyBacktestError(
            "The selected range is too short for one complete walk-forward fold"
        )
    return tuple(
        _Period(
            fold_index=fold.index,
            train_start=fold.train_start,
            train_end=fold.train_end,
            test_start=fold.test_start,
            test_end=fold.test_end,
        )
        for fold in folds
    )


def _assert_local_module(module_name: str) -> Any:
    module = sys.modules.get(module_name)
    if module is None:
        module = importlib.import_module(module_name)
    raw_path = getattr(module, "__file__", None)
    if not raw_path:
        raise StrategyBacktestError(
            f"{module_name}: module has no verifiable source path"
        )
    app_root = Path(__file__).resolve().parent.parent
    module_path = Path(raw_path).resolve()
    try:
        module_path.relative_to(app_root)
    except ValueError as exc:
        raise StrategyBacktestError(
            f"{module_name}: resolved outside isolated app root: {module_path}"
        ) from exc
    return module


def _default_strategy_factory() -> Any:
    try:
        module_name = "core.strategy_narrative"
        if module_name not in sys.modules:
            shim = types.ModuleType("config")
            shim.ORDERBLOCK_ENTRY_ENABLED = True
            shim.REJECTION_BLOCK_ENTRY_ENABLED = False
            shim.REJECTION_BLOCK_H1_ENTRY_ENABLED = False
            shim.FXPRO_CLUSTER_REJECTION_ENTRY_ENABLED = False
            shim.FXPRO_CLUSTER_ALLOW_RESEARCH_ASSUMPTION = False
            shim.FXPRO_QUOTE_PRESSURE_REJECTION_ENTRY_ENABLED = False
            shim.ORDERBLOCK_TOUCH_ATR_K = 0.15
            shim.ORDERBLOCK_TOUCH_MIN_ABS = 0.0005
            shim.ORDERBLOCK_MAX_AGE_BARS = 80
            shim.HTF_SCORE_MARGIN = 2
            missing = object()
            previous = sys.modules.get("config", missing)
            sys.modules["config"] = shim
            try:
                importlib.import_module(module_name)
            finally:
                if previous is missing:
                    sys.modules.pop("config", None)
                else:
                    sys.modules["config"] = previous
        module = _assert_local_module(module_name)
        _assert_local_module("core.htf_context")
        _assert_local_module("core.pivot_trigger")
        NarrativeStrategy = getattr(module, "NarrativeStrategy")
    except Exception as exc:
        raise StrategyBacktestError(
            "NarrativeStrategy is unavailable in the isolated backtest release"
        ) from exc
    return NarrativeStrategy()


def _configure_strategy(strategy: Any, config: NarrativeBacktestConfig) -> Any:
    strategy.risk_per_trade = config.risk_fraction
    # The sealed research contract keeps exactly three 1R/2R/3R targets with
    # 50/30/20 weights regardless of the live single-TP production flag.
    # Migrating the backtest to the single-TP contract is a separate,
    # explicitly approved change with its own frozen OOS evidence.
    strategy.single_tp_mode = False
    strategy.tp_rr_levels = [1.0, 2.0, 3.0]
    strategy.rr_min = 1.5
    strategy.rejection_block_entry_enabled = (
        config.rejection_block_entry_enabled
    )
    strategy.rejection_block_h1_entry_enabled = (
        config.rejection_block_h1_oos_enabled
    )
    strategy.rejection_block_h1_detector_version = (
        config.rejection_block_h1_detector_version
    )
    strategy.orderblock_entry_enabled = config.orderblock_entry_enabled
    strategy.orderblock_touch_atr_k = config.orderblock_touch_atr_k
    strategy.orderblock_touch_min_abs = config.orderblock_touch_min_abs
    strategy.orderblock_max_age_bars = config.orderblock_max_age_bars
    strategy.htf_score_margin = config.htf_score_margin
    strategy.fvg_regime_max_age_bars = config.fvg_regime_max_age_bars
    strategy.fvg_event_invalidation_mode = (
        config.fvg_event_invalidation_mode
    )
    strategy.factor_contract = config.factor_contract
    strategy.fvg_veto_enabled = config.fvg_veto_enabled
    expected = {
        "risk_per_trade": config.risk_fraction,
        "single_tp_mode": False,
        "tp_rr_levels": [1.0, 2.0, 3.0],
        "rr_min": 1.5,
        "rejection_block_entry_enabled": (
            config.rejection_block_entry_enabled
        ),
        "rejection_block_h1_entry_enabled": (
            config.rejection_block_h1_oos_enabled
        ),
        "rejection_block_h1_detector_version": (
            config.rejection_block_h1_detector_version
        ),
        "orderblock_entry_enabled": config.orderblock_entry_enabled,
        "orderblock_touch_atr_k": config.orderblock_touch_atr_k,
        "orderblock_touch_min_abs": config.orderblock_touch_min_abs,
        "orderblock_max_age_bars": config.orderblock_max_age_bars,
        "htf_score_margin": config.htf_score_margin,
        "fvg_regime_max_age_bars": config.fvg_regime_max_age_bars,
        "fvg_event_invalidation_mode": (
            config.fvg_event_invalidation_mode
        ),
        "factor_contract": config.factor_contract,
        "fvg_veto_enabled": config.fvg_veto_enabled,
    }
    mismatches = {
        name: (getattr(strategy, name, None), value)
        for name, value in expected.items()
        if getattr(strategy, name, None) != value
    }
    if mismatches:
        raise StrategyBacktestError(
            f"Strategy settings failed closed: {mismatches}"
        )
    return strategy


def _strategy_data(
    prepared: Mapping[str, _PreparedSeries],
    decision_time: pd.Timestamp,
    config: NarrativeBacktestConfig,
) -> Optional[dict[str, pd.DataFrame]]:
    data = {
        _STRATEGY_KEY[timeframe]: prepared[timeframe].asof(
            decision_time,
            limit=config.history_limit,
        )
        for timeframe in ("1d", "4h", "1h", "15m")
    }
    # The stateful first-touch trackers may inspect intrahour history, but only
    # M1 bars whose close is known at the decision timestamp are exposed.
    # M1 depth is deliberately not part of the HTF minimum-context gate.
    data["1M"] = prepared["1m"].asof(
        decision_time,
        limit=config.history_limit,
    )
    if any(
        len(data[key]) < config.min_context_bars
        for key in ("4H", "1H", "15M")
    ):
        return None
    if len(data["D"]) < config.min_daily_bars:
        return None
    return data


def _session_allowed(
    decision_time: pd.Timestamp,
    config: NarrativeBacktestConfig,
) -> bool:
    if not config.allowed_sessions or "ALL" in config.allowed_sessions:
        return True
    local_time = decision_time.tz_convert(
        ZoneInfo(config.session_timezone)
    ).time()
    for name in config.allowed_sessions:
        start_raw, end_raw = _SESSION_WINDOWS[name]
        if _time_in_window(
            local_time,
            _parse_clock(start_raw),
            _parse_clock(end_raw),
        ):
            return True
    return False


def _friday_closed(
    decision_time: pd.Timestamp,
    config: NarrativeBacktestConfig,
) -> bool:
    local = decision_time.tz_convert(ZoneInfo("Europe/Moscow"))
    return (
        local.weekday() == 4
        and local.hour >= config.friday_close_hour_moscow
    )


def _next_friday_close(
    after: pd.Timestamp,
    config: NarrativeBacktestConfig,
) -> pd.Timestamp:
    local = after.tz_convert(ZoneInfo("Europe/Moscow"))
    days_ahead = (4 - local.weekday()) % 7
    cutoff = (
        local.normalize()
        + pd.Timedelta(days=days_ahead)
        + pd.Timedelta(hours=config.friday_close_hour_moscow)
    )
    if cutoff <= local:
        cutoff += pd.Timedelta(days=7)
    return cutoff.tz_convert("UTC")


def _force_close_outcome(
    outcome: SetupOutcome,
    *,
    price: float,
    timestamp: pd.Timestamp,
) -> SetupOutcome:
    if outcome.status == "CLOSED":
        return outcome
    risk = abs(outcome.entry - outcome.initial_stop)
    if risk <= 0:
        raise StrategyBacktestError("Cannot force-close an outcome with zero risk")
    direction_pnl = (
        price - outcome.entry
        if outcome.side == "LONG"
        else outcome.entry - price
    )
    legs = tuple(
        leg
        if leg.exit_reason != "OPEN"
        else LegOutcome(
            tp_index=leg.tp_index,
            weight=leg.weight,
            exit_reason="TIME",
            exit_price=float(price),
            r_multiple=float(direction_pnl / risk * leg.weight),
            exit_time=timestamp,
        )
        for leg in outcome.legs
    )
    return SetupOutcome(
        side=outcome.side,
        entry=outcome.entry,
        initial_stop=outcome.initial_stop,
        status="CLOSED",
        net_r=float(sum(leg.r_multiple for leg in legs)),
        legs=legs,
        tp_hits=outcome.tp_hits,
        moved_to_be=outcome.moved_to_be,
        ambiguous_bars=outcome.ambiguous_bars,
        bars_processed=outcome.bars_processed,
        exit_time=timestamp,
    )


def _causal_force_price(
    m1: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[Optional[pd.Timestamp], Optional[float]]:
    position = int(m1.index.searchsorted(cutoff, side="left"))
    if position < len(m1):
        timestamp = pd.Timestamp(m1.index[position])
        if timestamp <= cutoff + pd.Timedelta(minutes=1):
            return timestamp, float(m1.iloc[position]["open"])
    previous = position - 1
    if previous >= 0:
        timestamp = pd.Timestamp(m1.index[previous]) + pd.Timedelta(minutes=1)
        if cutoff - pd.Timedelta(days=4) <= timestamp <= cutoff:
            return timestamp, float(m1.iloc[previous]["close"])
    return None, None


def _apply_deterministic_gates(
    *,
    symbol: str,
    decision_time: pd.Timestamp,
    signal: Mapping[str, Any],
    data: Mapping[str, pd.DataFrame],
    config: NarrativeBacktestConfig,
) -> tuple[str, str, dict[str, Any]]:
    enriched = dict(signal)
    if config.profile == "signal-quality":
        return "ENTER", "raw signal-quality profile", enriched
    if _friday_closed(decision_time, config):
        return "BLOCK_FRIDAY", "Friday weekend close", enriched
    if not _session_allowed(decision_time, config):
        return "BLOCK_SESSION", "outside configured session", enriched
    if not config.vol_filter_enabled:
        return "ENTER", "volatility filter disabled", enriched

    try:
        vol_module = _assert_local_module("core.vol_regime")
        build_vol_context = getattr(vol_module, "build_vol_context")
        entry_gate = getattr(vol_module, "entry_gate")
    except Exception as exc:
        raise StrategyBacktestError(
            "Volatility filter is unavailable in the isolated backtest release"
        ) from exc

    daily = data["D"]
    m15 = data["15M"]
    context = build_vol_context(
        symbol,
        daily["close"].to_list(),
        spot=float(m15["close"].iloc[-1]),
        df_15m=m15,
    )
    if context is None:
        return "ENTER", "volatility context unavailable; fail open", enriched
    enriched["vol_R"] = round(float(context.r_t), 1)
    enriched["vol_regime"] = context.regime
    enriched["vol_em_1d"] = round(float(context.em_1d), 6)
    targets = [
        float(value)
        for value in (enriched.get("tp_prices") or ())
        if value is not None
    ]
    entry_value = enriched.get("entry_price")
    if context.em_1d > 0 and entry_value is not None and targets:
        tp1_distance = min(
            abs(float(target) - float(entry_value))
            for target in targets
        )
        enriched["vol_tp1_em_ratio"] = round(
            float(tp1_distance / context.em_1d),
            6,
        )
    allowed, reason = entry_gate(
        context,
        enriched.get("entry_price"),
        targets,
        max_r=config.vol_max_r,
        em_tp_ratio=config.em_tp_ratio,
    )
    if allowed:
        return "ENTER", reason, enriched
    gate = (
        "BLOCK_VOL_REGIME"
        if config.vol_max_r > 0 and context.r_t >= config.vol_max_r
        else "BLOCK_EM_TP"
    )
    return gate, reason, enriched


def _trigger_signature(signal: Mapping[str, Any]) -> str:
    trigger = str(signal.get("trigger_kind") or "").strip().lower()
    if not trigger:
        trigger = (
            str(signal.get("trigger_reason") or "")
            .split("|")[0]
            .strip()
            .split(" ")[0]
            .lower()
        )
    event_id = str(signal.get("trigger_event_id") or "").strip()
    zone_low = signal.get("zone_low")
    zone_high = signal.get("zone_high")
    if event_id:
        anchor = f"e{event_id}"
    elif zone_low is not None and zone_high is not None:
        anchor = f"z{float(zone_low):.5f}-{float(zone_high):.5f}"
    else:
        anchor = f"s{float(signal.get('stop_price') or 0.0):.5f}"
    return f"{signal.get('side')}|{trigger}|{anchor}"


def _validate_factor_vector(
    signal: Mapping[str, Any],
    *,
    status: str,
    required: bool,
    factor_contract: Optional[str] = None,
) -> None:
    vector = signal.get("factor_vector")
    if not isinstance(vector, Mapping):
        if required and status in {"ENTER", "NO_TREND", "NO_TRIGGER"}:
            raise StrategyBacktestError(
                f"{status} signal is missing its structured factor vector"
            )
        return
    raw_factors = vector.get("factors", ())
    factors = [
        row
        for row in raw_factors
        if isinstance(row, Mapping)
    ]
    contract = resolve_factor_contract(factor_contract)
    if required:
        actual_key_list = [str(row.get("key")) for row in factors]
        actual_keys = set(actual_key_list)
        if vector.get("schema") != FACTOR_VECTOR_SCHEMA:
            raise StrategyBacktestError("unexpected factor-vector schema")
        if vector.get("factor_contract") != contract["name"]:
            raise StrategyBacktestError(
                "factor vector identity does not match the "
                f"{contract['name']} contract"
            )
        if (
            len(factors) != len(FACTOR_WEIGHTS)
            or len(actual_key_list) != len(actual_keys)
            or actual_keys != set(FACTOR_WEIGHTS)
        ):
            raise StrategyBacktestError(
                f"factor vector must contain exactly {len(FACTOR_WEIGHTS)} "
                "unique configured factors"
            )
        # ``configured_weight`` is the definition's own weight and stays
        # constant across contracts; the arm is carried by
        # ``effective_weight``, which is what actually scores.
        configured = {
            str(row.get("key")): row.get("configured_weight")
            for row in factors
        }
        if configured != FACTOR_WEIGHTS:
            raise StrategyBacktestError(
                "factor vector configured weights do not match the "
                "legacy/reference definitions"
            )
        effective = {
            str(row.get("key")): float(row.get("effective_weight") or 0.0)
            for row in factors
        }
        expected_effective = {
            key: float(value) for key, value in contract["weights"].items()
        }
        if effective != expected_effective:
            raise StrategyBacktestError(
                "factor vector effective weights do not match the "
                f"{contract['name']} contract"
            )
        if bool(vector.get("fvg_margin_enabled", True)) != contract[
            "fvg_margin_enabled"
        ]:
            raise StrategyBacktestError(
                "factor vector FVG margin rule does not match the "
                f"{contract['name']} contract"
            )
    score_long = sum(
        float(row.get("long_contribution") or 0.0)
        for row in factors
    )
    score_short = sum(
        float(row.get("short_contribution") or 0.0)
        for row in factors
    )
    if not math.isclose(
        score_long,
        float(vector.get("score_long") or 0.0),
        abs_tol=1e-12,
    ) or not math.isclose(
        score_short,
        float(vector.get("score_short") or 0.0),
        abs_tol=1e-12,
    ):
        raise StrategyBacktestError(
            "factor contributions do not reproduce narrative scores"
        )
    if required:
        try:
            rebuilt = build_factor_vector(
                {
                    str(row["key"]): {
                        "present": row.get("present"),
                        "side": row.get("vote_side"),
                        "evidence": row.get("evidence") or {},
                    }
                    for row in factors
                },
                base_margin=int(vector["base_margin"]),
                fvg_side=str(vector.get("fvg_side") or "NEUTRAL"),
                weights=contract["weights"],
                fvg_margin_enabled=contract["fvg_margin_enabled"],
                factor_contract=contract["name"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StrategyBacktestError(
                "factor vector cannot be reconstructed safely"
            ) from exc

        def require_equal(field: str, actual: Any, expected: Any) -> None:
            if actual != expected:
                raise StrategyBacktestError(
                    f"factor-vector derived field {field} does not match "
                    "the selected contract"
                )

        def require_number(field: str, actual: Any, expected: Any) -> None:
            try:
                actual_value = float(actual)
                expected_value = float(expected)
            except (TypeError, ValueError) as exc:
                raise StrategyBacktestError(
                    f"factor-vector numeric field {field} is invalid"
                ) from exc
            if (
                not math.isfinite(actual_value)
                or not math.isclose(
                    actual_value,
                    expected_value,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise StrategyBacktestError(
                    f"factor-vector derived field {field} does not match "
                    "the selected contract"
                )

        for field in ("bias", "fvg_side"):
            require_equal(field, vector.get(field), rebuilt[field])
        for field in (
            "score_long",
            "score_short",
            "score_delta_long_minus_short",
            "base_margin",
            "margin_long",
            "margin_short",
        ):
            require_number(field, vector.get(field), rebuilt[field])

        actual_by_key = {
            str(row["key"]): row
            for row in factors
        }
        for expected_row in rebuilt["factors"]:
            key = str(expected_row["key"])
            actual_row = actual_by_key[key]
            for field in (
                "label",
                "vote_side",
                "bias_without_factor",
            ):
                require_equal(
                    f"{key}.{field}",
                    actual_row.get(field),
                    expected_row[field],
                )
            for field in ("present", "pivotal_without_factor"):
                if not isinstance(actual_row.get(field), bool):
                    raise StrategyBacktestError(
                        f"factor-vector boolean field {key}.{field} is invalid"
                    )
                require_equal(
                    f"{key}.{field}",
                    actual_row.get(field),
                    expected_row[field],
                )
            for field in (
                "configured_weight",
                "effective_weight",
                "long_contribution",
                "short_contribution",
                "selected_side_contribution",
            ):
                require_number(
                    f"{key}.{field}",
                    actual_row.get(field),
                    expected_row[field],
                )
    if status == "ENTER" and str(vector.get("bias")) != str(
        signal.get("side")
    ):
        raise StrategyBacktestError(
            "factor-vector bias does not match ENTER side"
        )


def _trigger_kind(signal: Mapping[str, Any]) -> str:
    structured = str(signal.get("trigger_kind") or "").strip().lower()
    if structured in {
        "rejection_block_15m",
        "rejection_block_1h",
        "absorption_15m",
        "fxpro_cluster_rejection_15m",
        "fxpro_quote_pressure_rejection_15m",
        "h1_pivot_reclaim_15m",
        "order_block_1h",
        "turtle_soup_15m",
    }:
        return structured
    reason = str(signal.get("trigger_reason") or "").strip().lower()
    if reason.startswith("rejectionblock 15m"):
        return "rejection_block_15m"
    if reason.startswith(("rejectionblock 1h", "rejectionblock h1")):
        return "rejection_block_1h"
    if reason.startswith("absorption 15m"):
        return "absorption_15m"
    if reason.startswith("fxpro cluster rejection 15m"):
        return "fxpro_cluster_rejection_15m"
    if reason.startswith("fxpro quote pressure rejection 15m"):
        return "fxpro_quote_pressure_rejection_15m"
    if reason.startswith("turtlesoup 15m"):
        return "turtle_soup_15m"
    if reason.startswith(("h1 pivothigh reclaim", "h1 pivotlow reclaim")):
        return "h1_pivot_reclaim_15m"
    if reason.startswith("orderblock touch"):
        return "order_block_1h"
    return "unknown"


def _execution_row(
    *,
    candidate: _Candidate,
    policy: str,
    disposition: str,
    reason: str,
    fill_time: Optional[pd.Timestamp] = None,
    fill_price: Optional[float] = None,
    setup_id: Optional[str] = None,
    blocked_until: Optional[pd.Timestamp] = None,
    blocking_setup_id: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "policy": policy,
        "fold_index": candidate.fold_index,
        "symbol": candidate.symbol,
        "decision_time": candidate.decision_time.isoformat(),
        "trigger_kind": _trigger_kind(candidate.signal),
        "pre_gate": candidate.gate,
        "pre_gate_reason": candidate.gate_reason,
        "disposition": disposition,
        "disposition_reason": reason,
        "fill_time": fill_time.isoformat() if fill_time is not None else None,
        "fill_price": fill_price,
        "setup_id": setup_id,
        "blocked_until": (
            blocked_until.isoformat() if blocked_until is not None else None
        ),
        "blocking_setup_id": blocking_setup_id,
    }


def _candidate_id(
    *,
    manifest_sha256: str,
    fold_index: int,
    symbol: str,
    decision_time: pd.Timestamp,
    signal: Mapping[str, Any],
) -> str:
    identity = {
        "manifest": manifest_sha256,
        "fold": fold_index,
        "symbol": symbol,
        "decision_time": decision_time.isoformat(),
        "trigger": _trigger_signature(signal),
    }
    return _canonical_hash(identity)[:24]


def _emit(
    progress: Optional[ProgressCallback],
    **event: Any,
) -> None:
    if progress is not None:
        progress(event)


def _generate_candidates(
    *,
    dataset: HistoricalDataset,
    symbol: str,
    prepared: Mapping[str, _PreparedSeries],
    periods: Sequence[_Period],
    config: NarrativeBacktestConfig,
    strategy_factory: StrategyFactory,
    progress: Optional[ProgressCallback],
) -> tuple[list[_Candidate], Counter[str]]:
    counters: Counter[str] = Counter()
    candidates: list[_Candidate] = []
    m15_closes = prepared["15m"].close_times
    total = sum(
        max(
            0,
            int(m15_closes.searchsorted(period.test_end, side="left"))
            - int(m15_closes.searchsorted(period.test_start, side="left")),
        )
        for period in periods
    )
    completed = 0
    next_progress = 0

    _emit(
        progress,
        phase="signals",
        state="start",
        symbol=symbol,
        decisions=total,
    )
    for period in periods:
        strategy = _configure_strategy(strategy_factory(), config)
        left = int(m15_closes.searchsorted(period.test_start, side="left"))
        right = int(m15_closes.searchsorted(period.test_end, side="left"))
        for decision_time in m15_closes[left:right]:
            decision_time = pd.Timestamp(decision_time)
            counters["decisions"] += 1
            completed += 1
            data = _strategy_data(prepared, decision_time, config)
            if data is None:
                counters["warmup_skipped"] += 1
            else:
                try:
                    raw_signal = strategy.generate_signal(data, symbol=symbol)
                except Exception as exc:
                    raise StrategyBacktestError(
                        f"{symbol} {decision_time.isoformat()}: "
                        f"strategy failed: {exc}"
                    ) from exc
                status = str(raw_signal.get("signal") or "UNKNOWN").upper()
                _validate_factor_vector(
                    raw_signal,
                    status=status,
                    required=(
                        config.profile == "production-deterministic"
                    ),
                    factor_contract=config.factor_contract,
                )
                counters[f"signal_{status.lower()}"] += 1
                if status == "ENTER":
                    counters["raw_candidates"] += 1
                    gate, reason, enriched = _apply_deterministic_gates(
                        symbol=symbol,
                        decision_time=decision_time,
                        signal=raw_signal,
                        data=data,
                        config=config,
                    )
                    counters[f"gate_{gate.lower()}"] += 1
                    candidates.append(
                        _Candidate(
                            candidate_id=_candidate_id(
                                manifest_sha256=dataset.manifest_sha256,
                                fold_index=period.fold_index,
                                symbol=symbol,
                                decision_time=decision_time,
                                signal=enriched,
                            ),
                            fold_index=period.fold_index,
                            symbol=symbol,
                            decision_time=decision_time,
                            gate=gate,
                            gate_reason=str(reason),
                            signal=enriched,
                        )
                    )

            percent = int(completed * 100 / total) if total else 100
            if percent >= next_progress:
                _emit(
                    progress,
                    phase="signals",
                    state="progress",
                    symbol=symbol,
                    completed=completed,
                    decisions=total,
                    percent=percent,
                    candidates=len(candidates),
                )
                next_progress = min(100, percent + 1)

    _emit(
        progress,
        phase="signals",
        state="complete",
        symbol=symbol,
        completed=completed,
        decisions=total,
        candidates=len(candidates),
    )
    return candidates, counters


def _find_fill(
    *,
    m1: pd.DataFrame,
    decision_time: pd.Timestamp,
    period_end: pd.Timestamp,
    signal: Mapping[str, Any],
    config: NarrativeBacktestConfig,
) -> tuple[Optional[int], Optional[pd.Timestamp], Optional[float], str]:
    entry_ttl = config.entry_ttl
    signal_ttl_raw = signal.get("entry_ttl_min")
    if signal_ttl_raw is not None:
        try:
            signal_ttl_min = float(signal_ttl_raw)
        except (TypeError, ValueError):
            return None, None, None, "invalid signal-specific entry TTL"
        if (
            not math.isfinite(signal_ttl_min)
            or signal_ttl_min < 1.0
            or signal_ttl_min > 240.0
        ):
            return None, None, None, "invalid signal-specific entry TTL"
        entry_ttl = pd.Timedelta(minutes=signal_ttl_min)
    deadline = min(decision_time + entry_ttl, period_end)
    # The M15 signal only exists after its close.  The M1 candle stamped at
    # exactly that close has already opened, so its open/high/low cannot be
    # used as an executable post-signal price.
    left = int(m1.index.searchsorted(decision_time, side="right"))
    right = int(m1.index.searchsorted(deadline, side="left"))
    if right <= left:
        return None, None, None, "no M1 opens inside entry TTL"

    lower_raw = signal.get("entry_min")
    upper_raw = signal.get("entry_max")
    lower = float(lower_raw) if lower_raw is not None else float("-inf")
    upper = float(upper_raw) if upper_raw is not None else float("inf")
    if lower > upper:
        return None, None, None, "invalid entry range"

    side = str(signal.get("side") or "").upper()
    entry_order_type = str(
        signal.get("entry_order_type") or ""
    ).upper()
    is_limit_retest = entry_order_type == "LIMIT_RETEST"
    stop = float(signal.get("stop_price") or 0.0)
    planned = float(signal.get("entry_price") or 0.0)
    targets = [
        float(value)
        for value in (signal.get("tp_prices") or ())
        if value is not None
    ]
    if side not in {"LONG", "SHORT"} or stop <= 0 or planned <= 0:
        return None, None, None, "invalid side/entry/stop"
    if len(targets) != 3:
        return None, None, None, "strategy must provide exactly three targets"
    if is_limit_retest and not lower <= planned <= upper:
        return None, None, None, "invalid LIMIT_RETEST planned-entry geometry"
    if is_limit_retest:
        target_valid = (
            stop < planned
            and all(planned < target for target in targets)
            if side == "LONG"
            else stop > planned
            and all(planned > target for target in targets)
        )
        if not target_valid:
            return None, None, None, "invalid LIMIT_RETEST stop/target geometry"

    invalid_geometry_seen = False
    for position in range(left, right):
        row = m1.iloc[position]
        if is_limit_retest:
            low = float(row["low"])
            high = float(row["high"])
            touched_entry = low <= planned <= high
            touched_stop = (
                low <= stop if side == "LONG" else high >= stop
            )
            touched_tp1 = (
                high >= targets[0]
                if side == "LONG"
                else low <= targets[0]
            )
            if touched_entry and touched_stop:
                return (
                    position,
                    pd.Timestamp(m1.index[position]),
                    planned,
                    "filled LIMIT_RETEST; entry and stop share M1 bar, stop-first",
                )
            if touched_entry:
                if touched_tp1:
                    return (
                        None,
                        None,
                        None,
                        "ambiguous LIMIT_RETEST entry and TP1 in one M1 bar",
                    )
                return (
                    position,
                    pd.Timestamp(m1.index[position]),
                    planned,
                    "filled LIMIT_RETEST at planned price",
                )
            if touched_stop:
                return (
                    None,
                    None,
                    None,
                    "LIMIT_RETEST invalidated: stop touched before entry",
                )
            if touched_tp1:
                return (
                    None,
                    None,
                    None,
                    "LIMIT_RETEST invalidated: TP1 touched before entry",
                )
            continue

        price = float(row["open"])
        if price < lower or price > upper:
            continue
        target_valid = (
            stop < price and all(price < target for target in targets)
            if side == "LONG"
            else stop > price and all(price > target for target in targets)
        )
        if not target_valid:
            invalid_geometry_seen = True
            continue
        return position, pd.Timestamp(m1.index[position]), price, "filled"
    reason = (
        (
            "LIMIT_RETEST touch had invalid stop/target geometry"
            if is_limit_retest
            else "in-range M1 open had invalid stop/target geometry"
        )
        if invalid_geometry_seen
        else "entry range expired"
    )
    return None, None, None, reason


def _outcome_rows(
    *,
    outcome: SetupOutcome,
    setup_id: str,
    candidate: _Candidate,
    policy: str,
    entry_time: pd.Timestamp,
    entry: float,
    signal: Mapping[str, Any],
    risk_amount: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    exit_time = outcome.exit_time
    setup = {
        "setup_id": setup_id,
        "candidate_id": candidate.candidate_id,
        "policy": policy,
        "fold_index": candidate.fold_index,
        "symbol": candidate.symbol,
        "decision_time": candidate.decision_time.isoformat(),
        "entry_time": entry_time.isoformat(),
        "exit_time": exit_time.isoformat() if exit_time is not None else None,
        "side": outcome.side,
        "entry": entry,
        "planned_entry": signal.get("entry_price"),
        "entry_min": signal.get("entry_min"),
        "entry_max": signal.get("entry_max"),
        "stop": outcome.initial_stop,
        "tp_prices": list(signal.get("tp_prices") or ()),
        "status": outcome.status,
        "net_r": outcome.net_r,
        "risk_fraction": None,
        "risk_amount": risk_amount,
        "pnl_amount": outcome.net_r * risk_amount,
        "tp_hits": list(outcome.tp_hits),
        "moved_to_be": outcome.moved_to_be,
        "ambiguous_bars": outcome.ambiguous_bars,
        "bars_processed": outcome.bars_processed,
        "trigger_kind": _trigger_kind(signal),
        "trigger_reason": signal.get("trigger_reason"),
        "fvg_regime": signal.get("fvg_regime"),
        "vol_r": signal.get("vol_R"),
        "vol_regime": signal.get("vol_regime"),
        "vol_em_1d": signal.get("vol_em_1d"),
        "vol_tp1_em_ratio": signal.get("vol_tp1_em_ratio"),
        "narrative": signal.get("narrative"),
        **factor_vector_fields(signal.get("factor_vector")),
    }
    legs = [
        {
            "setup_id": setup_id,
            "candidate_id": candidate.candidate_id,
            "policy": policy,
            "tp_index": leg.tp_index,
            "weight": leg.weight,
            "exit_reason": leg.exit_reason,
            "exit_price": leg.exit_price,
            "r_multiple": leg.r_multiple,
            "exit_time": (
                leg.exit_time.isoformat()
                if leg.exit_time is not None
                else None
            ),
        }
        for leg in outcome.legs
    ]
    return setup, legs


def _execute_candidates(
    *,
    candidates: Sequence[_Candidate],
    prepared: Mapping[str, _PreparedSeries],
    periods: Sequence[_Period],
    config: NarrativeBacktestConfig,
    policy: str,
    progress: Optional[ProgressCallback],
    carry_positions_across_folds: bool = False,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    Counter[str],
]:
    counters: Counter[str] = Counter()
    setups: list[dict[str, Any]] = []
    legs: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    m1 = prepared["1m"].frame.drop(
        columns=[_BAR_CLOSE_COLUMN],
        errors="ignore",
    )
    risk_amount = config.initial_capital * config.risk_fraction
    by_fold: dict[int, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_fold[candidate.fold_index].append(candidate)

    total = len(candidates)
    completed = 0
    next_progress = 0
    symbol = candidates[0].symbol if candidates else ""
    _emit(
        progress,
        phase="execution",
        state="start",
        symbol=symbol,
        policy=policy,
        candidates=total,
    )

    active_until: Optional[pd.Timestamp] = None
    active_setup_id: Optional[str] = None
    cooldown_until: Optional[pd.Timestamp] = None
    cooldown_setup_id: Optional[str] = None
    current_day: Any = None
    entries_today = 0
    trigger_signatures: set[str] = set()
    # Exact structural retests are single-attempt events in live execution.
    # Unlike the generic daily signature guard, this reservation survives UTC
    # day and walk-forward fold boundaries for the whole replay.
    consumed_exact_events: set[tuple[str, str]] = set()
    filled_ranked_decisions: set[tuple[int, pd.Timestamp]] = set()
    production_gates = config.profile == "production-deterministic"
    for period in periods:
        for candidate in sorted(
            by_fold.get(period.fold_index, ()),
            key=lambda item: item.decision_time,
        ):
            completed += 1
            percent = int(completed * 100 / total) if total else 100
            if percent >= next_progress:
                _emit(
                    progress,
                    phase="execution",
                    state="progress",
                    symbol=symbol,
                    policy=policy,
                    completed=completed,
                    candidates=total,
                    percent=percent,
                    setups=len(setups),
                )
                next_progress = min(100, percent + 1)
            decision_time = candidate.decision_time
            ranked_decision = (
                (candidate.fold_index, decision_time)
                if candidate.signal.get("optimizer_rank") is not None
                else None
            )
            if (
                ranked_decision is not None
                and ranked_decision in filled_ranked_decisions
            ):
                counters["blocked_lower_ranked_alternative"] += 1
                executions.append(
                    _execution_row(
                        candidate=candidate,
                        policy=policy,
                        disposition="BLOCK_ALTERNATIVE",
                        reason=(
                            "another opportunity from this M15 decision "
                            "filled first"
                        ),
                    )
                )
                continue
            day = decision_time.date()
            if day != current_day:
                current_day = day
                entries_today = 0
                trigger_signatures = set()

            if (
                production_gates
                and active_until is not None
                and decision_time < active_until
            ):
                counters["blocked_active_setup"] += 1
                executions.append(
                    _execution_row(
                        candidate=candidate,
                        policy=policy,
                        disposition="BLOCK_ACTIVE",
                        reason="another setup is still active",
                        blocked_until=active_until,
                        blocking_setup_id=active_setup_id,
                    )
                )
                continue
            if candidate.gate != "ENTER":
                counters[f"blocked_{candidate.gate.lower()}"] += 1
                executions.append(
                    _execution_row(
                        candidate=candidate,
                        policy=policy,
                        disposition="BLOCK_GATE",
                        reason=candidate.gate_reason,
                    )
                )
                continue
            if (
                production_gates
                and cooldown_until is not None
                and decision_time < cooldown_until
            ):
                counters["blocked_post_loss_cooldown"] += 1
                executions.append(
                    _execution_row(
                        candidate=candidate,
                        policy=policy,
                        disposition="BLOCK_COOLDOWN",
                        reason="post-loss cooldown is active",
                        blocked_until=cooldown_until,
                        blocking_setup_id=cooldown_setup_id,
                    )
                )
                continue
            if (
                production_gates
                and entries_today >= config.max_setups_per_symbol_day
            ):
                counters["blocked_daily_setup_limit"] += 1
                executions.append(
                    _execution_row(
                        candidate=candidate,
                        policy=policy,
                        disposition="BLOCK_DAILY_LIMIT",
                        reason="daily setup limit reached",
                    )
                )
                continue
            signature = _trigger_signature(candidate.signal)
            if production_gates and signature in trigger_signatures:
                counters["blocked_duplicate_trigger"] += 1
                executions.append(
                    _execution_row(
                        candidate=candidate,
                        policy=policy,
                        disposition="BLOCK_DUPLICATE",
                        reason="duplicate trigger signature for UTC day",
                    )
                )
                continue

            entry_order_type = str(
                candidate.signal.get("entry_order_type") or ""
            ).upper()
            trigger_event_id = str(
                candidate.signal.get("trigger_event_id") or ""
            ).strip()
            exact_event_key = (
                str(candidate.symbol).upper(),
                trigger_event_id,
            )
            if (
                entry_order_type == "LIMIT_RETEST"
                and trigger_event_id
                and exact_event_key in consumed_exact_events
            ):
                counters["blocked_duplicate_exact_event"] += 1
                executions.append(
                    _execution_row(
                        candidate=candidate,
                        policy=policy,
                        disposition="BLOCK_DUPLICATE",
                        reason=(
                            "LIMIT_RETEST trigger event was already consumed "
                            "by an earlier entry attempt"
                        ),
                    )
                )
                continue
            if entry_order_type == "LIMIT_RETEST" and trigger_event_id:
                consumed_exact_events.add(exact_event_key)

            replay_end = (
                config.end
                if carry_positions_across_folds
                else period.test_end
            )
            fill_end = replay_end
            if production_gates:
                fill_end = min(
                    fill_end,
                    _next_friday_close(decision_time, config),
                )
            fill_position, entry_time, entry, fill_reason = _find_fill(
                m1=m1,
                decision_time=decision_time,
                period_end=fill_end,
                signal=candidate.signal,
                config=config,
            )
            if fill_position is None or entry_time is None or entry is None:
                counters[f"fill_reason_{fill_reason}"] += 1
                if fill_reason == (
                    "ambiguous LIMIT_RETEST entry and TP1 in one M1 bar"
                ):
                    counters["fill_censored_intrabar"] += 1
                    disposition = "CENSORED_INTRABAR"
                else:
                    counters["fill_expired"] += 1
                    disposition = (
                        "REJECT_INVALID_GEOMETRY"
                        if "invalid" in fill_reason
                        else "EXPIRED_ENTRY_RANGE"
                    )
                executions.append(
                    _execution_row(
                        candidate=candidate,
                        policy=policy,
                        disposition=disposition,
                        reason=fill_reason,
                    )
                )
                continue

            horizon_end = min(
                replay_end,
                entry_time + config.max_holding,
            )
            friday_cutoff: Optional[pd.Timestamp] = None
            if production_gates:
                candidate_cutoff = _next_friday_close(entry_time, config)
                if candidate_cutoff <= horizon_end:
                    friday_cutoff = candidate_cutoff
            simulation_end = friday_cutoff or horizon_end
            simulation_right = int(
                m1.index.searchsorted(simulation_end, side="left")
            )
            bars = m1.iloc[fill_position:simulation_right]
            if bars.empty:
                counters["fill_without_future_bar"] += 1
                executions.append(
                    _execution_row(
                        candidate=candidate,
                        policy=policy,
                        disposition="REJECT_NO_FUTURE_BAR",
                        reason="entry filled without an executable outcome bar",
                        fill_time=entry_time,
                        fill_price=entry,
                    )
                )
                continue

            try:
                outcome = simulate_split_outcome(
                    side=str(candidate.signal.get("side")).upper(),
                    entry=float(entry),
                    stop=float(candidate.signal["stop_price"]),
                    tp_prices=[
                        float(value)
                        for value in candidate.signal["tp_prices"]
                    ],
                    bars=bars,
                    intrabar_policy=policy,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise StrategyBacktestError(
                    f"{candidate.symbol} {entry_time.isoformat()}: "
                    f"outcome simulation failed: {exc}"
                ) from exc
            forced_exit_reason: Optional[str] = None
            if outcome.status == "OPEN":
                if friday_cutoff is not None:
                    force_cutoff = friday_cutoff
                    forced_exit_reason = "FRIDAY_CLOSE"
                elif simulation_end == replay_end:
                    force_cutoff = replay_end
                    forced_exit_reason = (
                        "DATASET_END"
                        if carry_positions_across_folds
                        else "FOLD_END"
                    )
                else:
                    force_cutoff = simulation_end
                    forced_exit_reason = "MAX_HOLDING"
                force_time, force_price = _causal_force_price(
                    m1,
                    force_cutoff,
                )
                if force_time is None or force_price is None:
                    raise StrategyBacktestError(
                        f"{candidate.symbol} {force_cutoff.isoformat()}: "
                        "no causal M1 price for the required time exit"
                    )
                outcome = _force_close_outcome(
                    outcome,
                    price=force_price,
                    timestamp=force_time,
                )
                counters[
                    f"forced_close_{forced_exit_reason.lower()}"
                ] += 1

            setup_id = _canonical_hash(
                {
                    "candidate_id": candidate.candidate_id,
                    "policy": policy,
                    "entry_time": entry_time.isoformat(),
                }
            )[:24]
            setup, setup_legs = _outcome_rows(
                outcome=outcome,
                setup_id=setup_id,
                candidate=candidate,
                policy=policy,
                entry_time=entry_time,
                entry=entry,
                signal=candidate.signal,
                risk_amount=risk_amount,
            )
            setup["risk_fraction"] = config.risk_fraction
            setup["forced_exit_reason"] = forced_exit_reason
            setups.append(setup)
            legs.extend(setup_legs)
            executions.append(
                _execution_row(
                    candidate=candidate,
                    policy=policy,
                    disposition="FILLED",
                    reason=(
                        f"setup simulated with status {outcome.status}"
                    ),
                    fill_time=entry_time,
                    fill_price=entry,
                    setup_id=setup_id,
                )
            )
            counters["setups_filled"] += 1
            counters[f"setups_{outcome.status.lower()}"] += 1
            if ranked_decision is not None:
                filled_ranked_decisions.add(ranked_decision)
            if production_gates:
                entries_today += 1
                trigger_signatures.add(signature)

            if not production_gates:
                continue
            if outcome.exit_time is None:
                active_until = replay_end
            else:
                exit_bar_close = outcome.exit_time + pd.Timedelta(minutes=1)
                active_until = min(exit_bar_close, replay_end)
                active_setup_id = setup_id
                if any(
                    leg.exit_reason == "STOP"
                    for leg in outcome.legs
                ):
                    cooldown_until = (
                        exit_bar_close + config.post_loss_cooldown
                    )
                    cooldown_setup_id = setup_id

    _emit(
        progress,
        phase="execution",
        state="complete",
        symbol=symbol,
        policy=policy,
        completed=completed,
        candidates=total,
        setups=len(setups),
    )
    return setups, legs, executions, counters


def _metrics_for(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("exit_time") is None,
            str(row.get("exit_time") or ""),
            str(row.get("entry_time") or ""),
            str(row.get("symbol") or ""),
            str(row.get("setup_id") or ""),
        ),
    )
    return aggregate_setup_metrics(ordered)


def _breakdowns(
    setups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    dimensions = {
        "by_symbol": "symbol",
        "by_trigger": "trigger_kind",
        "by_side": "side",
        "by_fold": "fold_index",
    }
    for output_name, field in dimensions.items():
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for setup in setups:
            grouped[str(setup.get(field))].append(setup)
        output[output_name] = {
            key: _metrics_for(items)
            for key, items in sorted(grouped.items())
        }
    return output


def _policy_summary(
    *,
    setups: Sequence[Mapping[str, Any]],
    execution_counters: Mapping[str, int],
    config: NarrativeBacktestConfig,
) -> dict[str, Any]:
    metrics = _metrics_for(setups)
    net_r = float(metrics["net_r"])
    risk_amount = config.initial_capital * config.risk_fraction
    ending_capital = config.initial_capital + net_r * risk_amount
    return {
        "metrics": metrics,
        "fixed_risk_equity": {
            "initial_capital": config.initial_capital,
            "risk_per_setup_amount": risk_amount,
            "ending_capital": ending_capital,
            "net_pnl": ending_capital - config.initial_capital,
            "return_pct": (
                (ending_capital / config.initial_capital - 1.0) * 100.0
            ),
            "max_drawdown_amount": (
                float(metrics["max_drawdown_r"]) * risk_amount
            ),
            "max_drawdown_pct_of_initial": (
                float(metrics["max_drawdown_r"])
                * config.risk_fraction
                * 100.0
            ),
        },
        "execution_counters": dict(sorted(execution_counters.items())),
        **_breakdowns(setups),
    }


def _intrabar_sensitivity(
    *,
    policy_summaries: Mapping[str, Mapping[str, Any]],
    setups: Sequence[Mapping[str, Any]],
) -> Optional[dict[str, Any]]:
    if not {"stop-first", "tp-first"}.issubset(policy_summaries):
        return None
    stop_metrics = policy_summaries["stop-first"]["metrics"]
    tp_metrics = policy_summaries["tp-first"]["metrics"]
    stop_by_candidate = {
        str(row["candidate_id"]): row
        for row in setups
        if row["policy"] == "stop-first"
    }
    tp_by_candidate = {
        str(row["candidate_id"]): row
        for row in setups
        if row["policy"] == "tp-first"
    }
    matched = sorted(set(stop_by_candidate) & set(tp_by_candidate))
    changed = sum(
        not math.isclose(
            float(stop_by_candidate[key]["net_r"]),
            float(tp_by_candidate[key]["net_r"]),
            abs_tol=1e-12,
        )
        for key in matched
    )
    return {
        "primary": "stop-first",
        "net_r_delta_tp_minus_stop": (
            float(tp_metrics["net_r"]) - float(stop_metrics["net_r"])
        ),
        "expectancy_r_delta_tp_minus_stop": (
            float(tp_metrics["expectancy_r"])
            - float(stop_metrics["expectancy_r"])
        ),
        "setups_total_delta_tp_minus_stop": (
            int(tp_metrics["setups_total"])
            - int(stop_metrics["setups_total"])
        ),
        "matched_filled_candidates": len(matched),
        "matched_outcome_changed": changed,
        "ambiguous_bars_stop_first": int(stop_metrics["ambiguous_bars"]),
        "ambiguous_bars_tp_first": int(tp_metrics["ambiguous_bars"]),
    }


def run_narrative_backtest(
    dataset: HistoricalDataset,
    config: NarrativeBacktestConfig,
    *,
    strategy_factory: StrategyFactory = _default_strategy_factory,
    progress: Optional[ProgressCallback] = None,
) -> StrategyBacktestResult:
    """Run causal signal generation and M1 outcome simulation."""

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
            "Requested decision range is outside common M15 coverage: "
            f"{config.start.isoformat()} -> {config.end.isoformat()}; "
            f"available {coverage_start.isoformat()} -> "
            f"{coverage_end.isoformat()}"
        )
    periods = _periods(config)
    all_candidates: list[_Candidate] = []
    all_executions: list[dict[str, Any]] = []
    all_setups: list[dict[str, Any]] = []
    all_legs: list[dict[str, Any]] = []
    signal_counters: Counter[str] = Counter()
    execution_counters: dict[str, Counter[str]] = {
        policy: Counter() for policy in config.intrabar_policies
    }

    for symbol in config.symbols:
        prepared = _prepare_symbol(dataset, symbol)
        candidates, counters = _generate_candidates(
            dataset=dataset,
            symbol=symbol,
            prepared=prepared,
            periods=periods,
            config=config,
            strategy_factory=strategy_factory,
            progress=progress,
        )
        all_candidates.extend(candidates)
        signal_counters.update(counters)
        for policy in config.intrabar_policies:
            setups, legs, executions, policy_counters = _execute_candidates(
                candidates=candidates,
                prepared=prepared,
                periods=periods,
                config=config,
                policy=policy,
                progress=progress,
            )
            all_setups.extend(setups)
            all_legs.extend(legs)
            all_executions.extend(executions)
            execution_counters[policy].update(policy_counters)

    policy_summaries = {}
    for policy in config.intrabar_policies:
        policy_setups = [
            setup for setup in all_setups if setup["policy"] == policy
        ]
        policy_summaries[policy] = _policy_summary(
            setups=policy_setups,
            execution_counters=execution_counters[policy],
            config=config,
        )

    fold_rows: list[dict[str, Any]] = []
    for policy in config.intrabar_policies:
        for period in periods:
            rows = [
                setup
                for setup in all_setups
                if setup["policy"] == policy
                and setup["fold_index"] == period.fold_index
            ]
            fold_rows.append(
                {
                    **period.to_dict(),
                    "policy": policy,
                    **_metrics_for(rows),
                }
            )

    candidate_rows = tuple(
        _safe_json_value(candidate.report_row())
        for candidate in sorted(
            all_candidates,
            key=lambda item: (
                item.decision_time,
                item.symbol,
                item.fold_index,
            ),
        )
    )
    setup_rows = tuple(
        sorted(
            (_safe_json_value(setup) for setup in all_setups),
            key=lambda item: (
                item["policy"],
                item["decision_time"],
                item["symbol"],
            ),
        )
    )
    candidate_factor_output = tuple(
        _safe_json_value(row)
        for row in candidate_factor_rows(candidate_rows)
    )
    setup_factor_output = tuple(
        _safe_json_value(row)
        for row in setup_factor_rows(setup_rows)
    )
    factor_summary_output = tuple(
        _safe_json_value(row)
        for row in factor_summary_rows(setup_factor_output)
    )
    factor_coverage = attribution_coverage(
        candidates=candidate_rows,
        setups=setup_rows,
        candidate_factors=candidate_factor_output,
        setup_factors=setup_factor_output,
    )
    shadow = build_shadow_scores(
        setups=setup_rows,
        periods=[period.to_dict() for period in periods],
        symbols=config.symbols,
        entry_ttl=config.entry_ttl,
        max_holding=config.max_holding,
    )
    config_payload = config.to_dict()
    intrabar_sensitivity = _intrabar_sensitivity(
        policy_summaries=policy_summaries,
        setups=all_setups,
    )
    summary = {
        "schema": REPORT_SCHEMA,
        "dataset": str(dataset.root),
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "config_sha256": _canonical_hash(config_payload),
        "release_commit": config.release_commit,
        "release_manifest_sha256": config.release_manifest_sha256,
        "environment_lock_sha256": config.environment_lock_sha256,
        "periods": [period.to_dict() for period in periods],
        "signal_counters": dict(sorted(signal_counters.items())),
        "policies": policy_summaries,
        "intrabar_sensitivity": intrabar_sensitivity,
        "factor_attribution": factor_coverage,
        "shadow_score": shadow["summary"],
        "assumptions": {
            "decision": "each completed M15 candle",
            "context": (
                f"up to {config.history_limit} bars closed at decision time; "
                "no forming candle is dropped twice"
            ),
            "fill": (
                "first subsequent M1 open inside the strategy entry range "
                f"within {config.entry_ttl}"
            ),
            "friday_close": (
                f"force remaining exposure at Friday "
                f"{config.friday_close_hour_moscow:02d}:00 Europe/Moscow "
                "in production-deterministic profile"
            ),
            "time_exits": (
                "remaining exposure is marked at a causal M1 price at "
                "Friday close, max holding, or each OOS fold boundary"
            ),
            "risk": (
                f"{config.risk_fraction:.2%} of initial capital per setup; "
                "fixed, non-compounding"
            ),
            "costs_applied": False,
            "spread_applied": False,
            "swap_applied": False,
            "ai_filter_applied": False,
            "pyramiding_applied": False,
            "historical_tick_order_available": False,
        },
        "warnings": [
            (
                "Results are gross signal-quality diagnostics. LSE OHLCV does "
                "not contain historical bid/ask spread, broker commission, "
                "swap, slippage, or tick ordering."
            ),
            (
                "The production-deterministic profile includes session, Friday, "
                "volatility, daily setup-count, trigger-dedupe, and post-loss "
                "cooldown gates. Live AI state, broker constraints, anti-hedge, "
                "the bot-wide daily equity-loss brake, portfolio correlation, "
                "and pyramiding are intentionally excluded."
            ),
            (
                "Stops and targets execute at their configured level. M1 gaps "
                "through a level and aggregate portfolio open risk are not yet "
                "broker-accurate."
            ),
            (
                "Candidate-factor attribution is conditional on raw ENTER "
                "signals (baseline directional bias plus a detected trigger). "
                "Outcome summaries are conditional again on execution gates "
                "and a fill. They do not identify unbiased replacement weights "
                "or counterfactual LONG/SHORT outcomes."
            ),
            (
                "The shadow score is diagnostic only and leaves candidate IDs, "
                "execution dispositions, setup count, and fixed risk unchanged."
            ),
        ],
    }
    return StrategyBacktestResult(
        summary=_safe_json_value(summary),
        config=_safe_json_value(config_payload),
        candidates=candidate_rows,
        executions=tuple(
            sorted(
                (
                    _safe_json_value(execution)
                    for execution in all_executions
                ),
                key=lambda item: (
                    item["policy"],
                    item["decision_time"],
                    item["symbol"],
                ),
            )
        ),
        setups=setup_rows,
        legs=tuple(_safe_json_value(leg) for leg in all_legs),
        folds=tuple(_safe_json_value(row) for row in fold_rows),
        candidate_factors=candidate_factor_output,
        setup_factors=setup_factor_output,
        factor_summary=factor_summary_output,
        shadow_models=_safe_json_value(
            {
                "summary": shadow["summary"],
                "models": shadow["models"],
            }
        ),
        shadow_predictions=tuple(
            _safe_json_value(row) for row in shadow["predictions"]
        ),
        shadow_coefficients=tuple(
            _safe_json_value(row) for row in shadow["coefficients"]
        ),
        shadow_metrics=tuple(
            _safe_json_value(row) for row in shadow["metrics"]
        ),
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    empty_fields: Sequence[str] = (),
) -> None:
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as stream:
            if empty_fields:
                csv.DictWriter(stream, fieldnames=empty_fields).writeheader()
        return
    fields = list(empty_fields)
    seen = set(fields)
    if fields:
        unexpected = sorted(
            {
                str(field)
                for row in rows
                for field in row
                if field not in seen
            }
        )
        if unexpected:
            raise StrategyBacktestError(
                f"{path.name} contains undeclared columns: {unexpected}"
            )
    else:
        for row in rows:
            for field in row:
                if field not in seen:
                    seen.add(field)
                    fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = {}
            for field in fields:
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


def write_strategy_report(
    result: StrategyBacktestResult,
    output: str | Path,
) -> Path:
    """Atomically publish a deterministic JSON/CSV strategy report."""

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
        _write_csv(
            staging / "candidates.csv",
            result.candidates,
            empty_fields=_EMPTY_CSV_FIELDS["candidates"],
        )
        _write_csv(
            staging / "executions.csv",
            result.executions,
            empty_fields=_EMPTY_CSV_FIELDS["executions"],
        )
        _write_csv(
            staging / "setups.csv",
            result.setups,
            empty_fields=_EMPTY_CSV_FIELDS["setups"],
        )
        _write_csv(
            staging / "legs.csv",
            result.legs,
            empty_fields=_EMPTY_CSV_FIELDS["legs"],
        )
        _write_csv(
            staging / "candidate_factors.csv",
            result.candidate_factors,
            empty_fields=_EMPTY_CSV_FIELDS["candidate_factors"],
        )
        _write_csv(
            staging / "setup_factor_attribution.csv",
            result.setup_factors,
            empty_fields=_EMPTY_CSV_FIELDS["setup_factors"],
        )
        _write_csv(
            staging / "factor_summary.csv",
            result.factor_summary,
            empty_fields=_EMPTY_CSV_FIELDS["factor_summary"],
        )
        (staging / "shadow_models.json").write_bytes(
            _json_bytes(result.shadow_models)
        )
        _write_csv(
            staging / "shadow_scores.csv",
            result.shadow_predictions,
            empty_fields=_EMPTY_CSV_FIELDS["shadow_predictions"],
        )
        _write_csv(
            staging / "shadow_coefficients.csv",
            result.shadow_coefficients,
            empty_fields=_EMPTY_CSV_FIELDS["shadow_coefficients"],
        )
        _write_csv(
            staging / "shadow_score_metrics.csv",
            result.shadow_metrics,
            empty_fields=_EMPTY_CSV_FIELDS["shadow_metrics"],
        )
        _write_csv(staging / "folds.csv", result.folds)
        files = []
        for path in sorted(staging.iterdir(), key=lambda item: item.name):
            if path.is_file():
                files.append(
                    {
                        "path": path.name,
                        "size": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                )
        manifest = {
            "schema": REPORT_SCHEMA,
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
