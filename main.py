# main.py
from __future__ import annotations

import math
import os
import re
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
import copy
from dataclasses import replace as dataclass_replace
from pathlib import Path
from datetime import datetime, date, time as dt_time, timedelta, timezone
from typing import Dict, Any, List, Tuple, Optional
from zoneinfo import ZoneInfo

from config import (
    TELEGRAM_TOKEN,
    UNIVERSE,
    CONTEXT_SYMBOLS,
    AI_DATA_DIR,
    LOG_TICK,
    DEBUG_RAW_SIGNALS,
    SESSION_WINDOWS,
    ALLOWED_SESSIONS,
    SESSION_TIMEZONE,
    MT5_CACHE_DIR,
    MT5_EXECUTION_ENABLED,
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_SERVER,
    MT5_MAGIC,
    MT5_RISK_PER_TRADE,
    MT5_INITIAL_CAPITAL,
    MT5_SLIPPAGE,
    PERSISTENT_LIMIT_ENABLED,
    PERSISTENT_LIMIT_CONFLICT_AUTOCLOSE_ENABLED,
    PERSISTENT_LIMIT_SYMBOLS,
    PERSISTENT_LIMIT_TTL_MIN,
    PERSISTENT_LIMIT_STATE_PATH,
    RB_H1_CONSUMED_EVENTS_PATH,
    SHADOW_TICK_WATCHER_ENABLED,
    SHADOW_TICK_WATCHER_POLL_SEC,
    SHADOW_TICK_WATCHER_MAX_BACKFILL_SEC,
    SHADOW_TICK_WATCHER_DB_PATH,
    SHADOW_CANDIDATE_LEDGER_ENABLED,
    SHADOW_CANDIDATE_LEDGER_DB_PATH,
    SHADOW_CANDIDATE_COST_PROFILE_PATH,
    SHADOW_CANDIDATE_CORRELATION_PROFILE_PATH,
    SHADOW_RF_PROFILE_PATH,
    SHADOW_RF_ORCA_SNAPSHOT_PATH,
    SHADOW_QUALITY_PROFILE_PATH,
    EXECUTION_QUALITY_FILTER_ENABLED,
    EXECUTION_QUALITY_PROFILE_PATH,
    NEWS_CALENDAR_PATH,
    MT5_BRIDGE_SYMBOLS,
    MT5_BRIDGE_TIMEFRAMES,
    MT5_BRIDGE_LOOKBACK_DAYS,
    MT5_BRIDGE_INTERVAL,
    PARTIAL_TP_MODE,
    MOVE_BE_AFTER_TP1,
    SIGNAL_ON_CLOSED_BARS,
    FRIDAY_CLOSE_HOUR,
    DAILY_FLAT_ENABLED,
    DAILY_CLOSE_HOUR,
    DAILY_CLOSE_BUFFER_MIN,
    CORRELATED_GROUPS,
    POST_SL_COOLDOWN_MIN,
    MAX_SETUPS_PER_SYMBOL_PER_DAY,
    DAILY_MAX_LOSS_PCT,
    MT5_MAX_VOLUME,
    MT5_COMMISSION_PER_LOT,
    VOL_REGIME_FILTER_ENABLED,
    VOL_REGIME_SYMBOLS,
    VOL_REGIME_MAX_R,
    EM_TP_MAX_RATIO,
    VOL_REGIME_REFRESH_MIN,
    SHADOW_SCORE_MODEL_PATH,
    POSITION_ADDING_ENABLED,
    IDEA_MAX_ENTRIES,
    FXPRO_DOM_CAPTURE_ENABLED,
    FXPRO_DOM_SYMBOLS,
    FXPRO_DOM_DATA_DIR,
    FXPRO_DOM_POLL_MS,
    FXPRO_DOM_MAX_LEVELS,
    FXPRO_DOM_HEARTBEAT_SEC,
    FXPRO_CLUSTER_LIVE_SIDECAR_DIR,
    FXPRO_TICK_CLUSTER_CAPTURE_ENABLED,
    FXPRO_TICK_CLUSTER_SYMBOLS,
    FXPRO_TICK_CLUSTER_DATA_DIR,
    FXPRO_TICK_CLUSTER_SIDECAR_DIR,
    FXPRO_TICK_CLUSTER_RETENTION_DAYS,
    FXPRO_TICK_CLUSTER_SETTLE_SEC,
    FXPRO_TICK_CLUSTER_POLL_SEC,
    FXPRO_TICK_CLUSTER_MAX_CATCHUP_BARS,
    FXPRO_TICK_CLUSTER_ARCHIVE_RAW,
)
from core.mt5_guard import install as _install_mt5_guard

# Must run before any thread touches the MetaTrader5 API — wraps every mt5.*
# call with a shared lock (tick loop, DataCacheLoop and MT5Bridge all use it).
_install_mt5_guard()

from backtest.cost_model import CostProfile
from backtest.fxpro_cluster_data import FxProClusterEventDataset
from core.data_cache import DataCache
from core.data_feed import DataFeed
from core.fxpro_dom import FxProDomRecorder
from core.execution_quality import (
    ExecutionQualityProfile,
    PointInTimeNewsCalendar,
    assess_execution_quality,
)
from core.fxpro_tick_cluster import (
    FxProTickClusterRecorder,
    TickClusterSymbol,
)
from core.market_scanner import MarketScanner
from core.shadow_score import LiveShadowScorer, tp1_em_ratio
from core.strategy_narrative import NarrativeStrategy, ActiveTrade
from bot.telegram_bot import TelegramBot
from core.m1.config import AIConfig
from core.m1.store import TradeStore
from core.m1.ai_live import AILive, primary_idea_trigger_kind
from core.state_store import save_active_trades, load_active_trades
from core.vol_regime import VolContext, build_vol_context, entry_gate
from core.profiler import TickProfiler
from core.risk_rules import RiskRules
from core.position_adding import PyramidRiskError, build_manager as build_pyramid_manager
from core.persistent_limit import (
    BROKER_COMMENT_PREFIX,
    PendingLimitLeg,
    PendingLimitPlan,
    PendingLimitState,
    PendingLimitValidationError,
    load_pending_limits,
    save_pending_limits,
)
from core.rb_consumed_events import (
    RBConsumedEventStore,
    RBConsumedEventsValidationError,
)
from core.shadow_tick_watcher import (
    MT5ReadOnlyFacade,
    QuoteTick,
    ShadowPlanSpec,
    ShadowTickWatcher,
    stable_plan_id,
)
from core.shadow_candidate_ledger import (
    ShadowCandidateLedger,
    trigger_signature as shadow_trigger_signature,
)
from core.shadow_candidate_ranker import (
    PortfolioCorrelationProfile,
    build_shadow_rankings,
)
from core.hierarchical_quality_score import (
    HierarchicalQualityProfile,
    LiveHierarchicalQualityScorer,
)
from core.quality_shadow_bridge import build_quality_ledger_rows
from core.rf_candidate_contract import SINGLE_TP_TARGET_CONTRACT
from core.rf_shadow_bridge import RFShadowBridge
from executors.mt5_executor import (
    MT5Executor,
    MT5Settings,
    PendingOrderRejected,
    RiskCapacityError,
)
from mt5_bridge.mt5_native_bridge import MT5NativeBridge, parse_symbol_spec, parse_timeframes


_PENDING_CONFLICT_REASON = (
    "incompatible active/broker position on symbol"
)
_PENDING_CONFLICT_CLOSE_REQUESTED_REASON = (
    _PENDING_CONFLICT_REASON
    + "; emergency auto-close requested"
)
_PENDING_CONFLICT_CLOSE_FAILED_REASON = (
    _PENDING_CONFLICT_REASON
    + "; emergency auto-close failed; retry pending"
)
_PENDING_CONFLICT_CLOSE_RESOLVED_REASON = (
    _PENDING_CONFLICT_REASON
    + "; emergency auto-close exposure absent; manual recovery required"
)
_PENDING_CONFLICT_ABSENCE_GRACE_SEC = 30.0
_PENDING_DEFINITIVE_REJECTION_PREFIX = "definitive pending rejection: "


def _compute_tp_volumes(total_volume: float, n_tps: int, step: float = 0.01) -> List[float]:
    """
    Split total_volume into per-TP partial-close amounts.

      1 TP  → [100%]
      2 TPs → [50%, remainder]
      3 TPs → [50%, 30%, remainder]  (50/30/20 @ TP1/TP2/TP3)
      4 TPs → [25%, 30%, 30%, remainder]
      >4 TPs → first three get 25%/30%/30%, last entry is always remainder

    Allocation is performed in integer broker-step units (largest-remainder
    method). This preserves the total and, for tiny setups, assigns the first
    available unit to TP1 instead of accidentally leaving only the farthest TP.
    """
    if n_tps <= 0 or total_volume <= 0:
        return []
    if not step or step <= 0:
        step = 0.01

    total_units = max(0, int(math.floor(total_volume / step + 1e-9)))
    if n_tps == 1:
        return [round(total_units * step, 8)]

    if n_tps == 2:
        ratios: List[float] = [0.50, 0.50]
    elif n_tps == 3:
        ratios = [0.50, 0.30, 0.20]
    else:
        ratios = [0.25, 0.30, 0.30] + [0.0] * (n_tps - 4) + [0.15]

    targets = [total_units * ratio for ratio in ratios]
    units = [int(math.floor(target + 1e-12)) for target in targets]
    remainder_units = total_units - sum(units)
    order = sorted(
        range(n_tps),
        key=lambda idx: (targets[idx] - units[idx], -idx),
        reverse=True,
    )
    for idx in order[:remainder_units]:
        units[idx] += 1
    return [round(value * step, 8) for value in units]


def _entry_label(entry: ActiveTrade) -> str:
    """' entry#N' for a position-adding add-on, empty for a plain setup."""
    index = int(getattr(entry, "entry_index", 1) or 1)
    return f" entry#{index}" if index > 1 else ""


class Core:
    def __init__(self):
        self.universe = dict(UNIVERSE)
        self.context_symbols = dict(CONTEXT_SYMBOLS)
        feed_universe = dict(self.universe)
        feed_universe.update(self.context_symbols)
        self.feed = DataFeed(universe=feed_universe, mt5_cache_dir=MT5_CACHE_DIR)
        self.data_cache = DataCache(self.feed)
        self.data_cache.start()
        self.strategy = NarrativeStrategy()
        factor_contract = self.strategy._factor_contract()
        print(
            "[Narrative] factor_contract="
            f"{factor_contract['name']} effective_weights="
            f"{factor_contract['weights']}"
        )
        print(
            "[RB] M15 entry hard-disabled; H1 entry "
            f"{'enabled' if self.strategy.rejection_block_h1_entry_enabled else 'disabled'}; "
            "detector="
            f"{self.strategy.rejection_block_h1_detector_version}"
        )
        self.scanner = MarketScanner(self.universe)

        self.session_tz = ZoneInfo(SESSION_TIMEZONE)
        self.allowed_session_windows = []
        for name in (ALLOWED_SESSIONS or []):
            window = SESSION_WINDOWS.get(name.upper())
            if not window:
                continue
            self.allowed_session_windows.append(
                (
                    name.upper(),
                    self._parse_time_str(window[0]),
                    self._parse_time_str(window[1]),
                )
            )

        self.active_trades: dict[str, ActiveTrade] = load_active_trades(AI_DATA_DIR / "active_trades.json")
        print(f"[Core] restored active_trades={len(self.active_trades)}")
        self.pending_limits: Dict[str, PendingLimitPlan] = {}
        self._pending_state_safe = True
        self._pending_entry_events: Dict[str, Dict[str, Any]] = {}
        try:
            self.pending_limits = load_pending_limits(
                PERSISTENT_LIMIT_STATE_PATH
            )
            print(
                f"[Persistent LIMIT] restored plans={len(self.pending_limits)}"
            )
        except PendingLimitValidationError as exc:
            # Never interpret a corrupt book as empty. Broker orders may still
            # exist and a fresh entry could duplicate them.
            self._pending_state_safe = False
            print(
                "[Persistent LIMIT] STATE CORRUPT — all new entries blocked "
                f"until recovery: {exc}"
            )

        self._rb_consumed_events = RBConsumedEventStore()
        self._rb_consumed_events_safe = True
        self._rb_consumed_events_lock = threading.RLock()
        try:
            self._rb_consumed_events = RBConsumedEventStore.load(
                RB_H1_CONSUMED_EVENTS_PATH
            )
            if self._pending_state_safe:
                self._recover_consumed_rb_events_from_pending()
            print(
                "[RB Event Ledger] restored consumed_events="
                f"{len(self._rb_consumed_events)}"
            )
        except (RBConsumedEventsValidationError, OSError) as exc:
            # A corrupt/unwritable RB ledger blocks exact H1-RB entries only.
            # Other trigger families remain available, while interpreting the
            # state as empty could duplicate a first-touch trade after restart.
            self._rb_consumed_events_safe = False
            print(
                "[RB Event Ledger] STATE UNSAFE — exact H1-RB entries "
                f"blocked until recovery: {exc}"
            )

        self.ai_cfg = AIConfig()
        self.ai_store = TradeStore(self.ai_cfg)
        self.ai = AILive(self.ai_cfg, self.ai_store, self.strategy)

        self.TIME_BUDGET_SEC = 35.0
        self.N_BARS = 300

        self.profiler = TickProfiler()
        self.global_context: Dict[str, Any] = {"session": "ALL", "session_allowed": True}
        self.log_tick = LOG_TICK
        self.risk_rules = RiskRules()
        # Position Adding: staged entries into one idea under a shared risk cap.
        self.pyramid = build_pyramid_manager()
        if POSITION_ADDING_ENABLED:
            print(
                f"[Pyramid] Position Adding enabled: max {IDEA_MAX_ENTRIES} entries/idea, "
                f"idea risk cap {self.pyramid.settings.max_idea_risk_pct:.2%}"
            )

        # Grace period after startup: block MT5 closes for 90s to let state sync
        self._startup_time: float = time.time()
        self._startup_grace_sec: float = 90.0

        # cooldown per symbol after a failed entry attempt (prevents infinite retries)
        self._entry_cooldowns: Dict[str, float] = {}
        self._entry_cooldown_sec: float = 300.0  # 5 minutes
        self._stale_cooldown_sec: float = 60.0   # shorter cooldown for stale-price rejections
        # cooldown after a stop-loss close — the same still-valid M15 trigger
        # otherwise re-enters 2-3 minutes after the stop (2026-07-10 pattern)
        self._post_sl_cooldown_sec: float = float(POST_SL_COOLDOWN_MIN) * 60.0

        # Daily entry-frequency state (reset at UTC midnight, in-memory):
        #   _entries_today       — executed setups per symbol
        #   _trigger_signatures  — one-shot trigger dedupe (same zone/stop never re-traded)
        #   _day_baseline_balance / _daily_loss_stop — bot-wide daily loss brake
        self._counters_day: Optional[date] = None
        self._entries_today: Dict[str, int] = {}
        # Vol-regime contexts per symbol: (computed_at_ts, VolContext | None).
        # Recomputed at most every VOL_REGIME_REFRESH_MIN minutes — the 21-day
        # realized vol underneath moves on daily candles, not on ticks.
        self._vol_contexts: Dict[str, Tuple[float, Optional[VolContext]]] = {}
        self._trigger_signatures: Dict[str, set] = {}
        self._day_baseline_balance: Optional[float] = None
        self._daily_loss_stop: bool = False

        # A broker position "disappearing" must be confirmed on N consecutive ticks
        # with a healthy MT5 connection before the trade is treated as closed.
        # Otherwise a dropped terminal link (positions_get() → None) produces a
        # false EXIT_BROKER and the bot forgets live positions.
        self._broker_missing_counts: Dict[str, int] = {}
        self._broker_missing_confirm: int = 2
        self._last_reconnect_ts: float = 0.0
        # Serializes the fast broker-management job with any direct lifecycle
        # polling. TelegramBot also shares one asyncio lock between its 3-second
        # management job and 60-second signal job.
        self._management_lock = threading.RLock()

        self.mt5_executor: MT5Executor | None = None
        self._executor_retry_ts: float = 0.0
        self._pending_limit_capability_cache: Dict[
            str, Dict[str, Any]
        ] = {}
        if MT5_EXECUTION_ENABLED:
            if self._try_create_executor():
                self._refresh_pending_limit_capabilities()

        self.shadow_tick_watcher: Optional[ShadowTickWatcher] = None
        if SHADOW_TICK_WATCHER_ENABLED:
            try:
                import MetaTrader5 as _mt5

                max_chunks = max(
                    1,
                    int(math.ceil(
                        SHADOW_TICK_WATCHER_MAX_BACKFILL_SEC / 60.0
                    )),
                )
                self.shadow_tick_watcher = ShadowTickWatcher(
                    mt5=MT5ReadOnlyFacade.from_module(_mt5),
                    db_path=SHADOW_TICK_WATCHER_DB_PATH,
                    poll_seconds=SHADOW_TICK_WATCHER_POLL_SEC,
                    max_query_span_seconds=60.0,
                    max_chunks_per_poll=max_chunks,
                )
                if self.shadow_tick_watcher.start():
                    print(
                        "[Shadow Tick] diagnostic watcher started "
                        f"(poll={SHADOW_TICK_WATCHER_POLL_SEC:g}s, "
                        f"db={SHADOW_TICK_WATCHER_DB_PATH})"
                    )
                else:
                    print(
                        "[Shadow Tick] watcher disabled after fail-open init: "
                        f"{self.shadow_tick_watcher.disabled_reason}"
                    )
            except Exception as exc:
                self.shadow_tick_watcher = None
                print(f"[Shadow Tick] watcher unavailable (fail-open): {exc}")

        self.shadow_candidate_ledger: Optional[
            ShadowCandidateLedger
        ] = None
        self._shadow_candidate_executor: Optional[
            ThreadPoolExecutor
        ] = None
        self._shadow_candidate_future: Optional[Future[None]] = None
        self._shadow_candidate_dropped_jobs = 0
        self.shadow_candidate_cost_profile: Optional[CostProfile] = None
        self.shadow_candidate_correlation_profile: Optional[
            PortfolioCorrelationProfile
        ] = None
        self.shadow_quality_scorer: Optional[
            LiveHierarchicalQualityScorer
        ] = None
        self.shadow_rf_bridge: Optional[RFShadowBridge] = None
        if SHADOW_CANDIDATE_LEDGER_ENABLED:
            try:
                self.shadow_candidate_ledger = ShadowCandidateLedger(
                    SHADOW_CANDIDATE_LEDGER_DB_PATH
                )
                self._shadow_candidate_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="shadow-candidates",
                )
                print(
                    "[Shadow Candidates] diagnostic ledger started "
                    f"(db={SHADOW_CANDIDATE_LEDGER_DB_PATH})"
                )
            except Exception as exc:
                self.shadow_candidate_ledger = None
                self._shadow_candidate_executor = None
                print(
                    "[Shadow Candidates] ledger unavailable "
                    f"(fail-open): {exc}"
                )
            if SHADOW_CANDIDATE_COST_PROFILE_PATH is not None:
                try:
                    self.shadow_candidate_cost_profile = CostProfile.load(
                        SHADOW_CANDIDATE_COST_PROFILE_PATH
                    )
                    print(
                        "[Shadow Candidates] cost profile loaded: "
                        f"{self.shadow_candidate_cost_profile.profile_id}"
                    )
                except Exception as exc:
                    print(
                        "[Shadow Candidates] cost profile unavailable "
                        f"(ranking disabled, capture continues): {exc}"
                    )
            if SHADOW_CANDIDATE_CORRELATION_PROFILE_PATH is not None:
                try:
                    self.shadow_candidate_correlation_profile = (
                        PortfolioCorrelationProfile.load(
                            SHADOW_CANDIDATE_CORRELATION_PROFILE_PATH
                        )
                    )
                    print(
                        "[Shadow Candidates] correlation profile loaded: "
                        f"{self.shadow_candidate_correlation_profile.profile_id}"
                    )
                except Exception as exc:
                    print(
                        "[Shadow Candidates] correlation profile unavailable "
                        f"(ranking disabled, capture continues): {exc}"
                    )
            if SHADOW_RF_PROFILE_PATH is not None:
                if self.shadow_candidate_cost_profile is None:
                    print(
                        "[Shadow RF] profile configured without a valid cost "
                        "profile; RF diagnostics unavailable"
                    )
                else:
                    try:
                        self.shadow_rf_bridge = RFShadowBridge.load(
                            SHADOW_RF_PROFILE_PATH,
                            cost_profile=self.shadow_candidate_cost_profile,
                            expected_strategy_version=(
                                os.getenv(
                                    "STRATEGY_VERSION",
                                    "unversioned",
                                ).strip()
                                or "unversioned"
                            ),
                            expected_factor_contract=(
                                self.strategy.factor_contract
                            ),
                            expected_target_contract=(
                                SINGLE_TP_TARGET_CONTRACT
                            ),
                            orca_snapshot_path=(
                                SHADOW_RF_ORCA_SNAPSHOT_PATH
                            ),
                        )
                        print(
                            "[Shadow RF] diagnostic candidate scorer loaded: "
                            f"{self.shadow_rf_bridge.model_id} "
                            "(non-executing)"
                        )
                    except Exception as exc:
                        self.shadow_rf_bridge = None
                        print(
                            "[Shadow RF] profile unavailable "
                            f"(execution unaffected): {exc}"
                        )
            if SHADOW_QUALITY_PROFILE_PATH is not None:
                try:
                    self.shadow_quality_scorer = (
                        LiveHierarchicalQualityScorer(
                            HierarchicalQualityProfile.load(
                                SHADOW_QUALITY_PROFILE_PATH
                            )
                        )
                    )
                    print(
                        "[Shadow Candidates] quality profile loaded "
                        "(diagnostic-only): "
                        f"{self.shadow_quality_scorer.profile.profile_id}"
                    )
                except Exception as exc:
                    print(
                        "[Shadow Candidates] quality profile unavailable "
                        f"(quality diagnostics disabled, capture "
                        f"continues): {exc}"
                    )

        self.execution_quality_profile: Optional[
            ExecutionQualityProfile
        ] = None
        self.news_calendar: Optional[PointInTimeNewsCalendar] = None
        if EXECUTION_QUALITY_FILTER_ENABLED:
            if EXECUTION_QUALITY_PROFILE_PATH is None:
                print(
                    "[Execution Quality] enabled without profile; "
                    "new entries fail closed"
                )
            else:
                try:
                    self.execution_quality_profile = (
                        ExecutionQualityProfile.load(
                            EXECUTION_QUALITY_PROFILE_PATH
                        )
                    )
                    print(
                        "[Execution Quality] profile loaded: "
                        f"{self.execution_quality_profile.profile_id}"
                    )
                except Exception as exc:
                    print(
                        "[Execution Quality] profile unavailable; "
                        f"new entries fail closed: {exc}"
                    )
            if NEWS_CALENDAR_PATH is not None:
                try:
                    self.news_calendar = PointInTimeNewsCalendar.load(
                        NEWS_CALENDAR_PATH
                    )
                    print(
                        "[Execution Quality] point-in-time news calendar "
                        f"loaded: {self.news_calendar.calendar_id}"
                    )
                except Exception as exc:
                    print(
                        "[Execution Quality] news calendar unavailable; "
                        f"news-enabled profile fails closed: {exc}"
                    )

        print(
            "[Persistent LIMIT] "
            f"{'enabled' if PERSISTENT_LIMIT_ENABLED else 'disabled'} "
            f"(exact retest, TTL={PERSISTENT_LIMIT_TTL_MIN}m, "
            f"symbols={','.join(sorted(PERSISTENT_LIMIT_SYMBOLS)) or 'none'})"
        )
        if self.mt5_executor:
            # Reconcile the pre-fill order book before generic position
            # hydration. A pending fill carries richer setup/TP identity than
            # broker rows alone and must win that race. The diagnostic watcher
            # starts first so a restart-time fill also receives a terminal fact.
            self._manage_pending_limits(startup=True)
            self._hydrate_active_trades_from_mt5()

        self.fxpro_dom_recorder: Optional[FxProDomRecorder] = None
        if FXPRO_DOM_CAPTURE_ENABLED:
            try:
                import MetaTrader5 as _mt5

                capture_symbols = [
                    self.universe.get(symbol, symbol)
                    for symbol in FXPRO_DOM_SYMBOLS
                ]
                self.fxpro_dom_recorder = FxProDomRecorder(
                    mt5_module=_mt5,
                    symbols=capture_symbols,
                    output_dir=FXPRO_DOM_DATA_DIR,
                    poll_interval_ms=FXPRO_DOM_POLL_MS,
                    max_levels=FXPRO_DOM_MAX_LEVELS,
                    heartbeat_seconds=FXPRO_DOM_HEARTBEAT_SEC,
                )
                self.fxpro_dom_recorder.start()
                print(
                    "[FxPro DOM] capture started; "
                    "Quote Pressure Rejection entry remains independently gated"
                )
            except Exception as exc:
                self.fxpro_dom_recorder = None
                print(f"[FxPro DOM] capture unavailable: {exc}")

        self.fxpro_tick_cluster_recorder: Optional[
            FxProTickClusterRecorder
        ] = None
        if FXPRO_TICK_CLUSTER_CAPTURE_ENABLED:
            try:
                import MetaTrader5 as _mt5

                capture_symbols = [
                    TickClusterSymbol(
                        symbol=symbol,
                        source_symbol=self.universe.get(symbol, symbol),
                    )
                    for symbol in FXPRO_TICK_CLUSTER_SYMBOLS
                ]
                self.fxpro_tick_cluster_recorder = FxProTickClusterRecorder(
                    mt5_module=_mt5,
                    symbols=capture_symbols,
                    output_dir=FXPRO_TICK_CLUSTER_DATA_DIR,
                    sidecar_dir=FXPRO_TICK_CLUSTER_SIDECAR_DIR,
                    retention_days=FXPRO_TICK_CLUSTER_RETENTION_DAYS,
                    settle_seconds=FXPRO_TICK_CLUSTER_SETTLE_SEC,
                    poll_seconds=FXPRO_TICK_CLUSTER_POLL_SEC,
                    max_catchup_bars=FXPRO_TICK_CLUSTER_MAX_CATCHUP_BARS,
                    archive_raw_ticks=FXPRO_TICK_CLUSTER_ARCHIVE_RAW,
                )
                self.fxpro_tick_cluster_recorder.start()
                print(
                    "[FxPro Cluster] MT5 bid-tick capture started -> "
                    f"{FXPRO_TICK_CLUSTER_SIDECAR_DIR}; Cluster Rejection "
                    "entry remains independently gated"
                )
            except Exception as exc:
                self.fxpro_tick_cluster_recorder = None
                print(f"[FxPro Cluster] tick capture unavailable: {exc}")

        # Diagnostic only. A frozen WFO shadow model annotates live ENTER
        # signals with a predicted expected R; it never gates or resizes them.
        self.shadow_scorer = LiveShadowScorer.load(SHADOW_SCORE_MODEL_PATH)
        if self.shadow_scorer is not None:
            print(
                "[Shadow] diagnostic model loaded: "
                f"id={self.shadow_scorer.model_id} "
                f"fold={self.shadow_scorer.fold_index} (non-executing)"
            )
        elif SHADOW_SCORE_MODEL_PATH is not None:
            print(
                "[Shadow] no fitted model at "
                f"{SHADOW_SCORE_MODEL_PATH} — annotation disabled"
            )

        self.fxpro_cluster_dataset: Optional[
            FxProClusterEventDataset
        ] = None
        self._fxpro_cluster_manifest_mtime_ns: Optional[int] = None
        self._fxpro_cluster_reload_error_ts = 0.0
        if FXPRO_CLUSTER_LIVE_SIDECAR_DIR is not None:
            self._refresh_fxpro_cluster_dataset(force=True)

    def _refresh_fxpro_cluster_dataset(
        self,
        *,
        force: bool = False,
    ) -> Optional[FxProClusterEventDataset]:
        """Reload an atomically published forward-observed cluster sidecar."""

        root = FXPRO_CLUSTER_LIVE_SIDECAR_DIR
        if root is None:
            return None
        manifest = Path(root) / "manifest.json"
        try:
            mtime_ns = manifest.stat().st_mtime_ns
            if (
                not force
                and self.fxpro_cluster_dataset is not None
                and self._fxpro_cluster_manifest_mtime_ns == mtime_ns
            ):
                return self.fxpro_cluster_dataset
            dataset = FxProClusterEventDataset.load(root)
            if dataset.research_only:
                raise ValueError(
                    "research_only cluster sidecar is refused by live Core"
                )
            self.fxpro_cluster_dataset = dataset
            self._fxpro_cluster_manifest_mtime_ns = mtime_ns
            print(
                "[FxPro Cluster] forward-observed sidecar loaded: "
                f"events={len(dataset.events)} "
                f"manifest={dataset.manifest_sha256[:12]}"
            )
        except Exception as exc:
            now = time.time()
            if now - self._fxpro_cluster_reload_error_ts >= 60.0:
                print(f"[FxPro Cluster] live sidecar unavailable: {exc}")
                self._fxpro_cluster_reload_error_ts = now
        return self.fxpro_cluster_dataset

    def close(self) -> None:
        candidate_executor = getattr(
            self, "_shadow_candidate_executor", None
        )
        if candidate_executor is not None:
            candidate_executor.shutdown(
                wait=True,
                cancel_futures=False,
            )
        candidate_ledger = getattr(
            self, "shadow_candidate_ledger", None
        )
        if candidate_ledger is not None:
            candidate_ledger.close()
        watcher = getattr(self, "shadow_tick_watcher", None)
        if watcher is not None:
            watcher.close()
        for attribute in (
            "fxpro_dom_recorder",
            "fxpro_tick_cluster_recorder",
        ):
            recorder = getattr(self, attribute, None)
            if recorder is not None:
                recorder.stop()

    @staticmethod
    def _write_shadow_candidate_batch(
        ledger: ShadowCandidateLedger,
        strategy: NarrativeStrategy,
        jobs: Tuple[Dict[str, Any], ...],
        outcomes: Dict[str, Dict[str, Any]],
        scorer: Optional[LiveShadowScorer],
        cost_profile: Optional[CostProfile],
        correlation_profile: Optional[PortfolioCorrelationProfile],
        quality_scorer: Optional[LiveHierarchicalQualityScorer] = None,
        rf_bridge: Optional[RFShadowBridge] = None,
    ) -> None:
        """Evaluate lower-priority candidates after live decisions are final.

        This runs on a cloned strategy in a single diagnostic worker. No value
        is returned to Core, so its population cannot select or veto an order.
        """

        for job in jobs:
            symbol = str(job["symbol"])
            try:
                candidates = strategy.generate_candidate_signals(
                    job["strategy_data"],
                    symbol=symbol,
                )
                for candidate in candidates:
                    candidate["shadow_trigger_signature"] = (
                        shadow_trigger_signature(candidate)
                    )
                errors = list(
                    getattr(strategy, "_last_candidate_errors", [])
                    or []
                )
                outcome = dict(outcomes.get(symbol) or {})
                ranking_model_id: Optional[str] = None
                rankings: List[Dict[str, Any]] = []
                if (
                    scorer is not None
                    and cost_profile is not None
                    and correlation_profile is not None
                ):
                    ranking_model_id, rankings = build_shadow_rankings(
                        symbol=symbol,
                        candidates=candidates,
                        observed_at_utc=job["observed_at_utc"],
                        decision_bar_close=job["decision_bar_close"],
                        scorer=scorer,
                        cost_profile=cost_profile,
                        correlation_profile=correlation_profile,
                        active_exposures=dict(
                            job.get("active_exposures") or {}
                        ),
                    )
                    for candidate, ranking in zip(
                        candidates,
                        rankings,
                    ):
                        candidate["shadow_ranking_status"] = ranking[
                            "ranking_status"
                        ]
                        candidate["shadow_expected_gross_r"] = ranking[
                            "expected_gross_r"
                        ]
                        candidate["shadow_estimated_cost_r"] = ranking[
                            "estimated_cost_r"
                        ]
                        candidate["shadow_expected_net_r"] = ranking[
                            "expected_net_r"
                        ]
                        candidate["shadow_correlation_penalty_r"] = (
                            ranking["correlation_penalty_r"]
                        )
                        candidate["shadow_portfolio_score"] = ranking[
                            "ranking_score"
                        ]
                if rf_bridge is not None and candidates:
                    try:
                        candidates = rf_bridge.enrich_candidates(
                            symbol=symbol,
                            candidates=candidates,
                            observed_at_utc=job["observed_at_utc"],
                            decision_bar_close=job[
                                "decision_bar_close"
                            ],
                            strategy_data=job.get("strategy_data"),
                        )
                    except Exception as exc:
                        print(
                            f"[Shadow RF] {symbol} scoring failed "
                            f"(execution unaffected): {exc}"
                        )
                scan_id = ledger.record_scan(
                    deployment_id=str(job["deployment_id"]),
                    symbol=symbol,
                    observed_at_utc=job["observed_at_utc"],
                    decision_bar_close=job["decision_bar_close"],
                    production_signal=job["production_signal"],
                    candidates=candidates,
                    detector_errors=errors,
                    downstream_disposition=str(
                        outcome.get("signal") or ""
                    )
                    or None,
                    downstream_details=outcome,
                )
                if (
                    scan_id is not None
                    and ranking_model_id is not None
                    and rankings
                ):
                    ledger.record_ranking(
                        scan_id,
                        rankings,
                        model_id=ranking_model_id,
                    )
                if (
                    scan_id is not None
                    and quality_scorer is not None
                    and candidates
                ):
                    try:
                        atr_h1: Optional[float] = None
                        frame_1h = (
                            job.get("strategy_data") or {}
                        ).get("1H")
                        if frame_1h is not None:
                            raw_atr = float(
                                NarrativeStrategy._atr(frame_1h, 14)
                            )
                            if math.isfinite(raw_atr) and raw_atr > 0.0:
                                atr_h1 = raw_atr
                        (
                            quality_model_id,
                            quality_rows,
                        ) = build_quality_ledger_rows(
                            symbol=symbol,
                            candidates=candidates,
                            observed_at_utc=job["observed_at_utc"],
                            decision_bar_close=job["decision_bar_close"],
                            scorer=quality_scorer,
                            atr_h1_14=atr_h1,
                        )
                        ledger.record_quality_ranking(
                            scan_id,
                            quality_rows,
                            model_id=quality_model_id,
                        )
                    except Exception as exc:
                        # Quality diagnostics are fail-open on top of an
                        # already fail-open batch: the scan row survives.
                        print(
                            f"[Shadow Candidates] {symbol} quality "
                            f"scoring failed (execution unaffected): {exc}"
                        )
            except Exception as exc:
                # Diagnostic work is deliberately fail-open.
                print(
                    f"[Shadow Candidates] {symbol} batch failed "
                    f"(execution unaffected): {exc}"
                )

    def _submit_shadow_candidate_jobs(
        self,
        jobs: List[Dict[str, Any]],
        outcomes: Dict[str, dict],
    ) -> None:
        ledger = getattr(self, "shadow_candidate_ledger", None)
        executor = getattr(self, "_shadow_candidate_executor", None)
        if (
            not jobs
            or ledger is None
            or not ledger.enabled
            or executor is None
        ):
            return
        previous = getattr(self, "_shadow_candidate_future", None)
        if previous is not None and not previous.done():
            self._shadow_candidate_dropped_jobs += len(jobs)
            dropped = self._shadow_candidate_dropped_jobs
            if dropped == len(jobs) or dropped % 100 == 0:
                print(
                    "[Shadow Candidates] worker backlog; "
                    f"dropped_jobs={dropped}"
                )
            return
        strategy_snapshot = copy.copy(self.strategy)
        outcome_snapshot = {
            symbol: dict(outcomes.get(symbol) or {})
            for symbol in {str(job["symbol"]) for job in jobs}
        }
        try:
            self._shadow_candidate_future = executor.submit(
                self._write_shadow_candidate_batch,
                ledger,
                strategy_snapshot,
                tuple(jobs),
                outcome_snapshot,
                getattr(self, "shadow_scorer", None),
                getattr(
                    self,
                    "shadow_candidate_cost_profile",
                    None,
                ),
                getattr(
                    self,
                    "shadow_candidate_correlation_profile",
                    None,
                ),
                getattr(self, "shadow_quality_scorer", None),
                getattr(self, "shadow_rf_bridge", None),
            )
        except Exception as exc:
            print(
                "[Shadow Candidates] submit failed "
                f"(execution unaffected): {exc}"
            )

    def _try_create_executor(self) -> bool:
        """Create the MT5 executor. Safe to call repeatedly — used both at startup
        and as a periodic retry when the terminal wasn't ready yet (a cold MT5
        start on a VPS can take longer than the IPC timeout)."""
        try:
            if MT5_LOGIN is None or not MT5_PASSWORD:
                raise RuntimeError("MT5 credentials are missing")
            settings = MT5Settings(
                login=MT5_LOGIN,
                password=MT5_PASSWORD,
                server=MT5_SERVER,
                risk_pct=MT5_RISK_PER_TRADE,
                initial_capital=MT5_INITIAL_CAPITAL,
                risk_state_path=str(AI_DATA_DIR / "risk_capital.json"),
                magic=MT5_MAGIC,
                slippage=MT5_SLIPPAGE,
                max_volume=MT5_MAX_VOLUME,
                commission_per_lot=MT5_COMMISSION_PER_LOT,
            )
            self.mt5_executor = MT5Executor(settings)
            print(
                f"[MT5] Execution enabled (risk={MT5_RISK_PER_TRADE:.2%}, "
                f"initial_capital={self.mt5_executor.initial_capital:.2f})"
            )
            return True
        except Exception as exc:
            self.mt5_executor = None
            print(f"[MT5] Executor unavailable: {exc} — will retry")
            return False

    def _save_pending_limit_state(self) -> None:
        save_pending_limits(
            self.pending_limits,
            PERSISTENT_LIMIT_STATE_PATH,
        )

    @staticmethod
    def _is_exact_h1_rb_signal(sig: Dict[str, Any]) -> bool:
        return (
            str(sig.get("entry_order_type") or "").strip().upper()
            == "LIMIT_RETEST"
            and str(sig.get("trigger_kind") or "").strip().lower()
            == "rejection_block_1h"
        )

    @classmethod
    def _exact_h1_rb_event_id(cls, sig: Dict[str, Any]) -> Optional[str]:
        if not cls._is_exact_h1_rb_signal(sig):
            return None
        event_id = str(sig.get("trigger_event_id") or "").strip()
        if not event_id:
            raise RBConsumedEventsValidationError(
                "exact H1-RB signal is missing trigger_event_id"
            )
        return event_id

    def _rb_event_is_consumed(
        self,
        symbol: str,
        sig: Dict[str, Any],
        *,
        trigger_signature: str,
    ) -> bool:
        event_id = self._exact_h1_rb_event_id(sig)
        if event_id is None:
            return False
        if not getattr(self, "_rb_consumed_events_safe", False):
            raise RBConsumedEventsValidationError(
                "consumed H1-RB event ledger is unsafe"
            )
        store = getattr(self, "_rb_consumed_events", None)
        if not isinstance(store, RBConsumedEventStore):
            raise RBConsumedEventsValidationError(
                "consumed H1-RB event ledger is unavailable"
            )
        existing = store.events.get(event_id)
        if existing is None:
            return False
        if (
            existing.symbol != str(symbol).upper()
            or existing.trigger_signature != trigger_signature
        ):
            self._rb_consumed_events_safe = False
            raise RBConsumedEventsValidationError(
                f"event_id identity collision for {event_id!r}"
            )
        return True

    def _consume_exact_h1_rb_event(
        self,
        symbol: str,
        sig: Dict[str, Any],
        *,
        trigger_signature: str,
        reason: str,
        consumed_at_utc: Optional[datetime] = None,
        require_new: bool = False,
    ) -> bool:
        """Persist a one-shot exact-RB event before any broker side effect."""

        event_id = self._exact_h1_rb_event_id(sig)
        if event_id is None:
            return False
        lock = getattr(self, "_rb_consumed_events_lock", None)
        if lock is None:
            raise RBConsumedEventsValidationError(
                "consumed H1-RB event ledger lock is unavailable"
            )
        with lock:
            already_consumed = self._rb_event_is_consumed(
                symbol,
                sig,
                trigger_signature=trigger_signature,
            )
            if already_consumed:
                if require_new:
                    raise RBConsumedEventsValidationError(
                        f"exact H1-RB event already consumed: {event_id}"
                    )
                return False
            updated = self._rb_consumed_events.consume(
                event_id=event_id,
                symbol=str(symbol).upper(),
                trigger_signature=trigger_signature,
                reason=reason,
                consumed_at_utc=consumed_at_utc,
            )
            try:
                updated.save(RB_H1_CONSUMED_EVENTS_PATH)
            except Exception as exc:
                self._rb_consumed_events_safe = False
                raise RBConsumedEventsValidationError(
                    "cannot durably persist consumed H1-RB event"
                ) from exc
            self._rb_consumed_events = updated
        print(
            f"[RB Event Ledger] consumed {symbol} event={event_id} "
            f"reason={reason}"
        )
        return True

    def _recover_consumed_rb_events_from_pending(self) -> None:
        """Backfill the ledger from durable pending intent after a crash."""

        updated = self._rb_consumed_events
        for symbol, plan in sorted(self.pending_limits.items()):
            sig = dict(plan.signal_payload)
            event_id = self._exact_h1_rb_event_id(sig)
            if event_id is None:
                continue
            updated = updated.consume(
                event_id=event_id,
                symbol=str(symbol).upper(),
                trigger_signature=str(plan.trigger_signature),
                reason="startup-pending-plan-recovery",
                consumed_at_utc=datetime.fromtimestamp(
                    float(plan.created_at), tz=timezone.utc
                ),
            )
        if updated is self._rb_consumed_events:
            return
        updated.save(RB_H1_CONSUMED_EVENTS_PATH)
        self._rb_consumed_events = updated

    def _get_pending_limit_capabilities(
        self,
        symbol: str,
        *,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        """Cache the executor's strictly read-only LIMIT capability probe."""

        key = str(symbol).upper()
        cache = getattr(
            self,
            "_pending_limit_capability_cache",
            None,
        )
        if cache is None:
            cache = {}
            self._pending_limit_capability_cache = cache
        if not refresh and key in cache:
            return dict(cache[key])

        executor = self.mt5_executor
        if executor is None:
            result: Dict[str, Any] = {
                "limit_allowed": False,
                "specified_expiration": False,
                "account_hedging": False,
                "ready": False,
                "error": "executor unavailable",
            }
        else:
            source_symbol = self.universe.get(key, key)
            try:
                raw = executor.pending_limit_capabilities(
                    source_symbol
                )
                if not isinstance(raw, dict):
                    raise RuntimeError(
                        "capability probe returned a non-mapping"
                    )
                limit_allowed = raw.get("limit_allowed") is True
                specified_expiration = (
                    raw.get("specified_expiration") is True
                )
                account_hedging = (
                    raw.get("account_hedging") is True
                )
                result = {
                    "limit_allowed": limit_allowed,
                    "specified_expiration": specified_expiration,
                    "account_hedging": account_hedging,
                    "ready": bool(
                        raw.get("ready") is True
                        and limit_allowed
                        and specified_expiration
                        and account_hedging
                    ),
                }
            except Exception as exc:
                result = {
                    "limit_allowed": False,
                    "specified_expiration": False,
                    "account_hedging": False,
                    "ready": False,
                    "error": str(exc),
                }
        cache[key] = dict(result)
        print(
            f"[Persistent LIMIT] capabilities {key}: "
            f"ready={result['ready']} "
            f"limit={result['limit_allowed']} "
            f"expiry={result['specified_expiration']} "
            f"hedging={result['account_hedging']}"
            + (
                f" error={result['error']}"
                if result.get("error")
                else ""
            )
        )
        return dict(result)

    def _refresh_pending_limit_capabilities(self) -> None:
        """Probe at startup/reconnect even if persistent LIMITs are off."""

        symbols = set(PERSISTENT_LIMIT_SYMBOLS)
        symbols.add("EURUSD")
        for symbol in sorted(symbols):
            self._get_pending_limit_capabilities(
                symbol,
                refresh=True,
            )

    def _capture_quote(self, symbol: str) -> Optional[QuoteTick]:
        """Read one broker quote for pending/shadow instrumentation."""
        watcher = getattr(self, "shadow_tick_watcher", None)
        source_symbol = self.universe.get(symbol, symbol)
        if watcher is not None:
            quote = watcher.capture_activation_quote(source_symbol)
            if quote is not None:
                return quote
        try:
            import MetaTrader5 as _mt5

            raw = _mt5.symbol_info_tick(source_symbol)
            if raw is None:
                return None
            return QuoteTick(
                time_msc=int(
                    getattr(raw, "time_msc", 0)
                    or int(time.time() * 1000)
                ),
                bid=float(raw.bid),
                ask=float(raw.ask),
                flags=int(getattr(raw, "flags", 0) or 0),
            )
        except Exception:
            return None

    @staticmethod
    def _closed_m15_decision_time(
        data: Optional[Dict[str, Any]],
    ) -> datetime:
        """Stable identity timestamp for repeated scans of one closed M15."""
        try:
            frame = (data or {}).get("15M")
            raw = frame.index[-1]
            value = (
                raw.to_pydatetime()
                if hasattr(raw, "to_pydatetime")
                else datetime.fromisoformat(str(raw))
            )
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            else:
                value = value.astimezone(timezone.utc)
            return value + timedelta(minutes=15)
        except Exception:
            now = datetime.now(timezone.utc)
            minute = (now.minute // 15) * 15
            return now.replace(
                minute=minute,
                second=0,
                microsecond=0,
            )

    def _record_shadow_scan_raw(
        self,
        symbol: str,
    ) -> Optional[QuoteTick]:
        """Record the quote visible to the actual 60-second loop.

        It is recorded for every still-live shadow plan, even if the strategy
        no longer emits the candidate. That is what separates a cadence miss
        from a policy/trigger disappearance.
        """
        watcher = getattr(self, "shadow_tick_watcher", None)
        if watcher is None or not watcher.enabled:
            return None
        quote = self._capture_quote(symbol)
        if quote is None:
            return None
        quote_observed_at = datetime.now(timezone.utc)
        for plan_id in watcher.active_plan_ids(symbol):
            watcher.record_scan(
                plan_id,
                quote,
                observed_at_utc=quote_observed_at,
                candidate_matches=False,
                policy_executable=False,
                disposition="RAW_60S_SCAN",
            )
        return quote

    def _register_shadow_candidate(
        self,
        symbol: str,
        sig: Dict[str, Any],
        *,
        trigger_signature: str,
        data: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[QuoteTick], float, datetime]:
        """Register one fully filtered candidate without influencing it."""
        watcher = getattr(self, "shadow_tick_watcher", None)
        # Activation must be captured after every policy gate has passed.
        # Reusing the quote from the start of the symbol scan would count ticks
        # that happened before the entry decision actually existed.
        quote = self._capture_quote(symbol)
        activation_observed_at = datetime.now(timezone.utc)
        entry_ttl_min = float(PERSISTENT_LIMIT_TTL_MIN)
        signal_ttl_raw = sig.get("entry_ttl_min")
        if signal_ttl_raw is not None:
            try:
                entry_ttl_min = float(signal_ttl_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "signal-specific entry_ttl_min must be numeric"
                ) from exc
            if (
                not math.isfinite(entry_ttl_min)
                or entry_ttl_min < 1.0
                or entry_ttl_min > 240.0
            ):
                raise ValueError(
                    "signal-specific entry_ttl_min must be within 1..240"
                )
        expires_at = (
            activation_observed_at
            + timedelta(minutes=entry_ttl_min)
        ).timestamp()
        if watcher is None or not watcher.enabled:
            return None, quote, expires_at, activation_observed_at

        if quote is None:
            return None, None, expires_at, activation_observed_at
        decision_time = self._closed_m15_decision_time(data)
        deployment_id = (
            os.getenv("DEPLOYMENT_ID", "").strip()
            or os.getenv("STRATEGY_VERSION", "unversioned").strip()
            or "unversioned"
        )
        plan_id = stable_plan_id(
            deployment_id=deployment_id,
            symbol=symbol,
            candidate_signature=trigger_signature,
            decision_bar_close=decision_time,
        )
        planned = float(sig.get("entry_price") or 0.0)
        lower = float(
            sig.get("entry_min")
            if sig.get("entry_min") is not None
            else planned
        )
        upper = float(
            sig.get("entry_max")
            if sig.get("entry_max") is not None
            else planned
        )
        point = 0.0
        try:
            import MetaTrader5 as _mt5

            info = _mt5.symbol_info(self.universe.get(symbol, symbol))
            point = float(
                getattr(info, "point", 0.0)
                or getattr(info, "trade_tick_size", 0.0)
                or 0.0
            )
        except Exception:
            pass
        spec = ShadowPlanSpec(
            plan_id=plan_id,
            symbol=symbol,
            source_symbol=self.universe.get(symbol, symbol),
            side=str(sig.get("side") or ""),
            entry_min=min(lower, upper),
            entry_max=max(lower, upper),
            planned_entry=planned,
            stop_price=float(sig.get("stop_price") or 0.0),
            activated_at_utc=activation_observed_at,
            activation_tick_msc=int(
                activation_observed_at.timestamp() * 1000
            ),
            activation_source_tick_msc=int(quote.time_msc),
            expires_at_utc=datetime.fromtimestamp(
                expires_at, tz=timezone.utc
            ),
            point=point,
            candidate_signature=trigger_signature,
            trigger_kind=str(sig.get("trigger_kind") or ""),
            trigger_event_id=sig.get("trigger_event_id"),
            setup_tf=str(sig.get("setup_tf") or sig.get("tf") or ""),
            strategy_version=os.getenv(
                "STRATEGY_VERSION", "unversioned"
            ),
            deployment_id=deployment_id,
            metadata={
                "session": self.global_context.get("session"),
                "entry_order_type": sig.get("entry_order_type"),
                "entry_ttl_min": entry_ttl_min,
                "entry_range_widened": bool(
                    sig.get("entry_range_widened")
                ),
            },
        )
        if watcher.register_plan(spec, activation_quote=quote):
            print(
                f"[Shadow Tick] {symbol} plan={plan_id[:12]} "
                f"expires={entry_ttl_min:g}m"
            )
        watcher.record_scan(
            plan_id,
            quote,
            observed_at_utc=activation_observed_at,
            candidate_matches=True,
            policy_executable=True,
            disposition="ELIGIBLE_60S_SCAN",
        )
        return plan_id, quote, expires_at, activation_observed_at

    def _arm_pending_limit(
        self,
        symbol: str,
        sig: Dict[str, Any],
        *,
        trigger_signature: str,
        expires_at: float,
        shadow_plan_id: Optional[str],
        observed_at: datetime,
    ) -> PendingLimitPlan:
        """Persist one-shot reservation and intent before broker LIMIT legs."""
        if not self._pending_state_safe:
            raise RuntimeError(
                "Persistent LIMIT state is unsafe; new entries are blocked"
            )
        if not self.mt5_executor:
            raise RuntimeError("Persistent LIMIT requires an MT5 executor")
        if symbol in self.pending_limits:
            return self.pending_limits[symbol]

        side = str(sig.get("side") or "").upper()
        limit_price = float(sig.get("entry_price") or 0.0)
        stop_price = float(sig.get("stop_price") or 0.0)
        tp_prices = [
            float(value)
            for value in (sig.get("tp_prices") or [sig.get("tp_price")])
            if value is not None
        ]
        tp_prices = (
            sorted(tp_prices)
            if side == "LONG"
            else sorted(tp_prices, reverse=True)
        )
        sig["tp_prices"] = tp_prices
        if tp_prices:
            sig["tp_price"] = tp_prices[-1]

        prepared = self.mt5_executor.prepare_limit_entry(
            symbol,
            side=side,
            limit_price=limit_price,
            stop_price=stop_price,
        )
        total_volume = float(prepared["volume"])
        volume_step = float(prepared.get("volume_step") or 0.01)
        volume_min = float(prepared.get("volume_min") or volume_step)
        use_split = (
            len(tp_prices) > 1
            and PARTIAL_TP_MODE != "monitor"
            and self.mt5_executor.account_is_hedging()
        )
        if use_split:
            volumes = _compute_tp_volumes(
                total_volume,
                len(tp_prices),
                step=volume_step,
            )
            # Preserve the current split-entry policy: sub-minimum far legs
            # merge toward TP1 instead of leaving only a distant target.
            for index in range(len(volumes) - 1, 0, -1):
                if 0 < volumes[index] < volume_min:
                    volumes[index - 1] = round(
                        volumes[index - 1] + volumes[index], 8
                    )
                    volumes[index] = 0.0
        else:
            volumes = [0.0] * max(0, len(tp_prices) - 1) + [
                total_volume
            ]
        legs = tuple(
            PendingLimitLeg(
                index=index + 1,
                target_price=tp_prices[index],
                requested_volume=float(volume),
            )
            for index, volume in enumerate(volumes)
            if float(volume) > 0
        )
        if not legs:
            raise RuntimeError(
                f"Persistent LIMIT sizing produced no broker-valid leg for {symbol}"
            )
        allocated = sum(leg.requested_volume for leg in legs)
        allocation_tolerance = max(1e-8, volume_step * 1e-6)
        if abs(allocated - total_volume) > allocation_tolerance:
            raise RuntimeError(
                f"Persistent LIMIT allocation mismatch for {symbol}: "
                f"{allocated:g} != {total_volume:g}"
            )

        plan_id = shadow_plan_id or uuid.uuid4().hex
        sig.setdefault("planned_entry_price", limit_price)
        sig["setup_id"] = plan_id
        sig.setdefault("idea_id", uuid.uuid4().hex)
        sig["entry_index"] = 1
        # Reserve an exact RB event before the pending book and before the
        # first order_send. A crash in the following tiny window can sacrifice
        # one opportunity, but it can never duplicate a first-touch trade.
        self._consume_exact_h1_rb_event(
            symbol,
            sig,
            trigger_signature=trigger_signature,
            reason="native-limit-arming",
            consumed_at_utc=observed_at,
            require_new=True,
        )
        plan = PendingLimitPlan.create(
            symbol=symbol,
            signal=sig,
            trigger_signature=trigger_signature,
            expires_at=expires_at,
            magic=int(self.mt5_executor.settings.magic),
            legs=legs,
            now=observed_at.timestamp(),
            plan_id=plan_id,
        )
        self.pending_limits[symbol] = plan
        self._save_pending_limit_state()
        # Arming is one causal attempt for this trigger. Reserve it before the
        # unsafe broker-call window so a restart cannot arm the same setup twice.
        self._trigger_signatures.setdefault(symbol, set()).add(
            trigger_signature
        )

        order_rows: List[Dict[str, Any]] = []
        risk_used = 0.0
        try:
            for leg in plan.legs:
                current = self.pending_limits[symbol]
                current_leg = next(
                    item
                    for item in current.legs
                    if item.index == leg.index
                )
                if current_leg.submission_attempted:
                    raise RuntimeError(
                        f"Refusing duplicate order_send for {symbol} "
                        f"leg {leg.index}; reconciliation owns it"
                    )
                current = current.mark_submission_attempted(
                    leg.index,
                    now=max(time.time(), current.updated_at),
                )
                self.pending_limits[symbol] = current
                self._save_pending_limit_state()
                plan = current
                result = self.mt5_executor.place_limit_leg(
                    symbol,
                    side=plan.side,
                    volume=leg.requested_volume,
                    limit_price=plan.limit_price,
                    stop_price=plan.stop_price,
                    tp_price=leg.target_price,
                    expires_at=plan.expires_at,
                    comment=plan.broker_comment_for_leg(leg.index),
                    risk_budget_amount=float(
                        prepared["risk_budget_amount"]
                    ),
                    risk_used_amount=risk_used,
                )
                order_rows.append(
                    {
                        **result,
                        "tp_index": leg.index,
                    }
                )
                ticket = int(result.get("ticket") or 0)
                if ticket > 0:
                    plan = plan.attach_order(
                        leg.index,
                        ticket,
                        now=max(time.time(), plan.updated_at),
                    )
                    self.pending_limits[symbol] = plan
                    self._save_pending_limit_state()
                risk_used += float(result.get("risk_amount") or 0.0)
                # A placement race may execute immediately. Stop submitting
                # siblings; the fast reconciliation pass will materialize the
                # fill and cancel every still-working remainder.
                if int(result.get("deal") or 0) > 0:
                    break
        except PendingOrderRejected as exc:
            current = self.pending_limits.get(symbol, plan)
            accepted_broker_evidence = bool(
                current.has_fills
                or any(
                    leg.broker_order_ticket is not None
                    for leg in current.legs
                )
            )
            target = (
                PendingLimitState.CANCELLING
                if accepted_broker_evidence
                else PendingLimitState.FAILED
            )
            current = current.transition(
                target,
                now=max(time.time(), current.updated_at),
                reason=(
                    _PENDING_DEFINITIVE_REJECTION_PREFIX + str(exc)
                ),
            )
            self.pending_limits[symbol] = current
            self._save_pending_limit_state()
            print(
                f"[Persistent LIMIT] {symbol} definitively rejected; "
                f"state={current.state.value}, retcode={exc.retcode}"
            )
            raise
        except Exception:
            current = self.pending_limits.get(symbol, plan)
            if any(
                leg.submission_attempted for leg in current.legs
            ):
                try:
                    current = current.transition(
                        PendingLimitState.CANCELLING,
                        now=max(time.time(), current.updated_at),
                        reason="placement failed; cancel submitted legs",
                    )
                    self.pending_limits[symbol] = current
                    self._save_pending_limit_state()
                except Exception:
                    pass
            # order_send can time out after the server accepted a request.
            # The attempted marker makes that ambiguity durable even when no
            # ticket was returned.
            if not order_rows:
                print(
                    f"[Persistent LIMIT] {symbol} placement outcome "
                    "ambiguous; durable plan retained for reconciliation"
                )
            raise

        plan = self.pending_limits[symbol]
        print(
            f"[Persistent LIMIT] {symbol} {plan.side} "
            f"@ {plan.limit_price:.6f}: state={plan.state.value}, "
            f"legs={len(order_rows)}, expires={plan.expires_at:.0f}"
        )
        return plan

    def _materialize_pending_trade(
        self,
        plan: PendingLimitPlan,
        *,
        fill_time_msc: Optional[int] = None,
        queue_entry_event: bool = False,
    ) -> Optional[ActiveTrade]:
        """Promote broker-confirmed pending fills into managed trade state.

        Active state is persisted before the pending plan is ever retired.
        """
        filled_legs = [
            leg for leg in plan.legs if leg.filled_volume > 0
        ]
        if not filled_legs:
            return None
        total_volume = sum(leg.filled_volume for leg in filled_legs)
        average_fill = sum(
            leg.filled_volume * float(leg.average_fill_price or 0.0)
            for leg in filled_legs
        ) / total_volume
        signal = dict(plan.signal_payload)
        idea_id = str(signal.get("idea_id") or plan.plan_id)
        trade = self.active_trades.get(plan.symbol)
        created = trade is None
        if trade is not None and str(
            getattr(trade, "idea_id", "") or ""
        ) != idea_id:
            self._pending_state_safe = False
            raise RuntimeError(
                f"{plan.symbol} pending fill overlaps a different active idea"
            )

        if any(
            int(leg.broker_position_id or 0) <= 0
            for leg in filled_legs
        ):
            # Deal history can briefly precede publication of position_id.
            # Keep the plan durable and retry; never create an unmanageable
            # ActiveTrade with no broker identity.
            return None
        if trade is None:
            opened_at = (
                int(fill_time_msc) / 1000.0
                if fill_time_msc is not None
                and int(fill_time_msc) > 0
                else min(float(plan.updated_at), time.time())
            )
            trade = ActiveTrade(
                side=plan.side,
                entry=round(float(average_fill), 6),
                stop=float(plan.stop_price),
                tp_prices=[float(value) for value in plan.tp_prices],
                tf=str(signal.get("tf") or ""),
                narrative=str(signal.get("narrative") or ""),
                symbol=plan.symbol,
                ts_open=opened_at,
                last_price_ts=time.time(),
            )
            trade.idea_id = idea_id
            trade.entry_index = 1
            trade.idea_trigger_signatures = [plan.trigger_signature]

        split_mode = len(plan.legs) > 1
        execution_legs: List[Dict[str, Any]] = []
        if split_mode:
            for leg in filled_legs:
                position_id = int(leg.broker_position_id or 0)
                if position_id <= 0:
                    continue
                execution_leg = {
                    "ticket": int(leg.broker_order_ticket or 0),
                    "position_id": position_id,
                    "volume": float(leg.filled_volume),
                    "price": float(leg.average_fill_price or average_fill),
                    "stop_price": float(plan.stop_price),
                    "tp": float(leg.target_price),
                    "tp_index": int(leg.index),
                    "comment": plan.broker_comment_for_leg(leg.index),
                }
                execution_legs.append(execution_leg)
                if position_id not in trade.split_legs:
                    trade.split_legs[position_id] = {
                        "tp_index": int(leg.index),
                        "tp": float(leg.target_price),
                        "volume": float(leg.filled_volume),
                        "order_ticket": int(
                            leg.broker_order_ticket or 0
                        ),
                        "status": "open",
                    }
                else:
                    # Deal publication may be incremental for one broker
                    # position. Keep the managed/journal volume monotonic with
                    # the durable leg rather than freezing its first fragment.
                    meta = trade.split_legs[position_id]
                    meta["tp_index"] = int(leg.index)
                    meta["tp"] = float(leg.target_price)
                    meta["volume"] = float(leg.filled_volume)
                    meta["order_ticket"] = int(
                        leg.broker_order_ticket or 0
                    )
                    meta["status"] = "open"
            trade.split_position_ids = [
                ticket
                for ticket, meta in sorted(
                    trade.split_legs.items(),
                    key=lambda item: (
                        int(item[1].get("tp_index") or 10**6),
                        int(item[0]),
                    ),
                )
                if str(meta.get("status") or "open") != "closed"
            ]
            trade.mt5_position_id = (
                trade.split_position_ids[0]
                if trade.split_position_ids
                else trade.mt5_position_id
            )
            trade.volume_per_tp = [0.0] * len(plan.tp_prices)
            for leg in filled_legs:
                idx = int(leg.index) - 1
                if 0 <= idx < len(trade.volume_per_tp):
                    trade.volume_per_tp[idx] = float(leg.filled_volume)
        else:
            leg = filled_legs[0]
            position_id = int(leg.broker_position_id or 0)
            if position_id <= 0:
                return None
            trade.mt5_position_id = position_id
            trade.mt5_ticket = int(leg.broker_order_ticket or 0) or None
            trade.execution_comment = plan.broker_comment_for_leg(
                leg.index
            )
            trade.volume_per_tp = _compute_tp_volumes(
                total_volume,
                len(plan.tp_prices),
            )
            execution_legs.append(
                {
                    "ticket": int(leg.broker_order_ticket or 0),
                    "position_id": position_id,
                    "volume": float(leg.filled_volume),
                    "price": float(leg.average_fill_price or average_fill),
                    "stop_price": float(plan.stop_price),
                    "tp": float(leg.target_price),
                    "tp_index": int(leg.index),
                    "comment": trade.execution_comment,
                }
            )

        if not getattr(trade, "moved_to_be", False):
            trade.entry = round(float(average_fill), 6)
        trade.volume = round(float(total_volume), 8)
        if split_mode:
            trade.volume_remaining = round(
                sum(
                    float(meta.get("volume") or 0.0)
                    for meta in trade.split_legs.values()
                    if str(meta.get("status") or "open") != "closed"
                ),
                8,
            )
        else:
            trade.volume_remaining = round(float(total_volume), 8)
        trade.last_price_ts = time.time()
        self.active_trades[plan.symbol] = trade
        save_active_trades(
            self.active_trades,
            AI_DATA_DIR / "active_trades.json",
        )

        if created:
            self._roll_daily_counters()
            self._entries_today[plan.symbol] = (
                self._entries_today.get(plan.symbol, 0) + 1
            )
            self._trigger_signatures.setdefault(
                plan.symbol, set()
            ).add(plan.trigger_signature)
        elif (
            plan.entry_announcement_pending
            and plan.trigger_signature
            not in self._trigger_signatures.get(plan.symbol, set())
        ):
            # Crash recovery: ActiveTrade may have reached disk before the
            # in-memory daily counters and durable ENTER outbox were delivered.
            self._roll_daily_counters()
            self._entries_today[plan.symbol] = (
                self._entries_today.get(plan.symbol, 0) + 1
            )
            self._trigger_signatures.setdefault(
                plan.symbol, set()
            ).add(plan.trigger_signature)

        if queue_entry_event and plan.entry_announcement_pending:
            signal.update(
                {
                    "signal": "ENTER",
                    "entry_price": round(float(average_fill), 6),
                    "idea_id": idea_id,
                    "entry_index": 1,
                    "execution": {
                        "mode": (
                            "pending_limit_split"
                            if split_mode
                            else "pending_limit_monitor"
                        ),
                        "volume": round(float(total_volume), 8),
                        "legs": execution_legs,
                    },
                    "pending_limit": {
                        "plan_id": plan.plan_id,
                        "limit_price": plan.limit_price,
                        "entry_min": plan.entry_min,
                        "entry_max": plan.entry_max,
                        "created_at": plan.created_at,
                        "expires_at": plan.expires_at,
                        "gap_beyond_zone": bool(
                            (
                                plan.side == "LONG"
                                and average_fill < plan.entry_min
                            )
                            or (
                                plan.side == "SHORT"
                                and average_fill > plan.entry_max
                            )
                        ),
                    },
                }
            )
            self._pending_entry_events[plan.symbol] = signal
        self._record_pending_fill_terminal(
            plan,
            fill_time_msc=fill_time_msc,
        )
        return trade

    def _record_pending_fill_terminal(
        self,
        plan: PendingLimitPlan,
        *,
        fill_time_msc: Optional[int],
    ) -> None:
        watcher = getattr(self, "shadow_tick_watcher", None)
        if watcher is None:
            return
        terminal_at = (
            datetime.fromtimestamp(
                int(fill_time_msc) / 1000.0,
                tz=timezone.utc,
            )
            if fill_time_msc is not None
            and int(fill_time_msc) > 0
            else datetime.fromtimestamp(
                float(plan.updated_at),
                tz=timezone.utc,
            )
        )
        watcher.record_terminal(
            plan.plan_id,
            "FILLED",
            at_utc=terminal_at,
            tick_msc=fill_time_msc,
            reason="broker pending LIMIT filled",
            inclusive=True,
        )

    @staticmethod
    def _is_pending_conflict_plan(plan: PendingLimitPlan) -> bool:
        return str(plan.reason or "").startswith(
            _PENDING_CONFLICT_REASON
        )

    @staticmethod
    def _with_pending_conflict_reason(
        plan: PendingLimitPlan,
        reason: str,
        *,
        now: Optional[float] = None,
    ) -> PendingLimitPlan:
        if plan.reason == reason:
            return plan
        moment = time.time() if now is None else float(now)
        return dataclass_replace(
            plan,
            reason=reason,
            updated_at=max(moment, float(plan.updated_at)),
        )

    @staticmethod
    def _mark_pending_conflict(
        plan: PendingLimitPlan,
    ) -> PendingLimitPlan:
        """Persist that this plan must never merge into an active idea."""

        if Core._is_pending_conflict_plan(plan):
            return plan
        return Core._with_pending_conflict_reason(
            plan,
            _PENDING_CONFLICT_REASON,
        )

    @staticmethod
    def _pending_plan_owned_position_ids(
        plan: PendingLimitPlan,
        positions: List[Dict[str, Any]],
    ) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
        """Classify exact plan-owned positions without changing broker state."""

        symbol = plan.symbol.upper()
        symbol_rows = [
            row
            for row in positions
            if str(row.get("symbol") or "").upper() == symbol
        ]
        rows_by_ticket = {
            int(row.get("ticket") or 0): row
            for row in symbol_rows
            if int(row.get("ticket") or 0) > 0
        }
        rows_by_comment: Dict[str, List[Dict[str, Any]]] = {}
        for row in symbol_rows:
            comment = str(row.get("comment") or "")
            if comment:
                rows_by_comment.setdefault(comment, []).append(row)

        owned: set[int] = set()
        ambiguous: List[str] = []
        for leg in plan.legs:
            comment = plan.broker_comment_for_leg(leg.index)
            matches = rows_by_comment.get(comment, [])
            recorded_id = int(leg.broker_position_id or 0)
            if recorded_id > 0:
                recorded_row = rows_by_ticket.get(recorded_id)
                if recorded_row is not None:
                    actual_comment = str(
                        recorded_row.get("comment") or ""
                    )
                    if actual_comment != comment:
                        ambiguous.append(
                            f"position {recorded_id} comment "
                            f"{actual_comment!r} != {comment!r}"
                        )
                    else:
                        owned.add(recorded_id)
                unexpected = sorted(
                    int(row.get("ticket") or 0)
                    for row in matches
                    if int(row.get("ticket") or 0) != recorded_id
                )
                if unexpected:
                    ambiguous.append(
                        f"leg {leg.index} maps to unexpected "
                        f"positions {unexpected}"
                    )
                continue

            if len(matches) > 1:
                ambiguous.append(
                    f"leg {leg.index} maps to multiple live positions"
                )
            elif matches:
                ticket = int(matches[0].get("ticket") or 0)
                if ticket <= 0:
                    ambiguous.append(
                        f"leg {leg.index} live position has no ticket"
                    )
                else:
                    owned.add(ticket)
        return tuple(sorted(owned)), tuple(ambiguous)

    @staticmethod
    def _pending_plan_close_targets(
        plan: PendingLimitPlan,
        positions: List[Dict[str, Any]],
        *,
        executor_magic: Any,
    ) -> Tuple[Tuple[Tuple[int, int, str], ...], Tuple[str, ...]]:
        """Return only positions whose durable ownership is fully proven.

        A close target requires a positive position id captured from the
        opening deal plus the exact symbol and full per-leg broker comment.
        ``list_positions`` is already executor-magic filtered; matching the
        plan's magic to that executor closes the remaining provenance gap.
        """

        errors: List[str] = []
        try:
            normalized_magic = int(executor_magic)
        except (TypeError, ValueError):
            normalized_magic = -1
        if normalized_magic != int(plan.magic):
            errors.append(
                f"plan magic {plan.magic} != executor magic "
                f"{normalized_magic}"
            )

        rows_by_ticket: Dict[int, List[Dict[str, Any]]] = {}
        for row in positions:
            ticket = int(row.get("ticket") or 0)
            if ticket > 0:
                rows_by_ticket.setdefault(ticket, []).append(row)

        leg_by_comment = {
            plan.broker_comment_for_leg(leg.index): leg
            for leg in plan.legs
        }
        recorded_ids: Dict[int, int] = {}
        targets: List[Tuple[int, int, str]] = []
        for leg in plan.legs:
            if leg.filled_volume <= 1e-9:
                continue
            position_id = int(leg.broker_position_id or 0)
            if position_id <= 0:
                errors.append(
                    f"filled leg {leg.index} has no recorded "
                    "broker_position_id"
                )
                continue
            other_leg = recorded_ids.get(position_id)
            if other_leg is not None and other_leg != leg.index:
                errors.append(
                    f"position {position_id} is recorded for legs "
                    f"{other_leg} and {leg.index}"
                )
                continue
            recorded_ids[position_id] = leg.index
            matches = rows_by_ticket.get(position_id, [])
            if len(matches) > 1:
                errors.append(
                    f"position {position_id} appears multiple times"
                )
                continue
            if not matches:
                continue
            row = matches[0]
            expected_comment = plan.broker_comment_for_leg(leg.index)
            actual_symbol = str(row.get("symbol") or "").upper()
            actual_comment = str(row.get("comment") or "")
            if actual_symbol != plan.symbol.upper():
                errors.append(
                    f"position {position_id} symbol {actual_symbol!r} "
                    f"!= {plan.symbol.upper()!r}"
                )
            if actual_comment != expected_comment:
                errors.append(
                    f"position {position_id} comment "
                    f"{actual_comment!r} != {expected_comment!r}"
                )
            row_magic = row.get("magic")
            row_magic_matches = True
            if row_magic is not None:
                try:
                    row_magic_matches = (
                        int(row_magic) == int(plan.magic)
                    )
                except (TypeError, ValueError):
                    row_magic_matches = False
                if not row_magic_matches:
                    errors.append(
                        f"position {position_id} magic "
                        f"{row_magic!r} != {plan.magic}"
                    )
            if (
                actual_symbol == plan.symbol.upper()
                and actual_comment == expected_comment
                and row_magic_matches
            ):
                targets.append(
                    (leg.index, position_id, expected_comment)
                )

        # A plan-looking comment on any other id/symbol is evidence of a race
        # or corrupt identity, never a comment-only fallback close target.
        for row in positions:
            comment = str(row.get("comment") or "")
            leg = leg_by_comment.get(comment)
            if leg is None:
                continue
            ticket = int(row.get("ticket") or 0)
            expected_id = int(leg.broker_position_id or 0)
            if expected_id <= 0:
                errors.append(
                    f"leg {leg.index} live position lacks a recorded id"
                )
            elif ticket != expected_id:
                errors.append(
                    f"leg {leg.index} maps to unexpected position "
                    f"{ticket} instead of {expected_id}"
                )
            if str(row.get("symbol") or "").upper() != plan.symbol.upper():
                errors.append(
                    f"leg {leg.index} exact comment appears on wrong symbol"
                )

        return (
            tuple(sorted(targets)),
            tuple(dict.fromkeys(errors)),
        )

    def _pending_cancel_reason(
        self,
        plan: PendingLimitPlan,
        *,
        now: float,
        quote: Optional[QuoteTick],
    ) -> Optional[str]:
        """Return a causal reason why an unfilled remainder must be removed."""
        if plan.has_fills:
            return "first pending leg filled; cancel every sibling"
        if plan.is_due(now):
            return "entry TTL elapsed"
        if (
            not PERSISTENT_LIMIT_ENABLED
            or plan.symbol not in PERSISTENT_LIMIT_SYMBOLS
        ):
            return "persistent LIMIT feature disabled for symbol"
        session_allowed, session_name = self._session_allowance()
        if not session_allowed:
            return f"session closed ({session_name})"
        if self._is_friday_weekend_close():
            return "Friday weekend close"
        if self._is_daily_flat_close():
            return "daily flat close"
        if self._is_daily_entry_cutoff():
            return "daily entry cutoff"
        self._roll_daily_counters()
        self._check_daily_loss_stop()
        if self._daily_loss_stop:
            return "daily loss stop"
        if quote is None:
            return None

        # Pending-entry invalidation uses the broker's closing side of spread:
        # BUY positions stop/target on BID, SELL positions on ASK.
        nearest_target = float(plan.tp_prices[0])
        if plan.side == "LONG":
            if float(quote.bid) <= float(plan.stop_price):
                return "LONG path crossed stop before retest"
            if float(quote.bid) >= nearest_target:
                return "LONG path crossed TP1 before retest"
        else:
            if float(quote.ask) >= float(plan.stop_price):
                return "SHORT path crossed stop before retest"
            if float(quote.ask) <= nearest_target:
                return "SHORT path crossed TP1 before retest"
        return None

    def _manage_pending_limits(self, *, startup: bool = False) -> None:
        """Reconcile durable entry intents with MT5 orders, deals and positions.

        No plan is retired until either its broker fill has first been persisted
        as an ActiveTrade, or every submitted remainder is authoritatively
        terminal in order history. Any ambiguous broker query fails closed.
        """
        executor = self.mt5_executor
        if executor is None or not hasattr(self, "pending_limits"):
            return
        lock = getattr(self, "_management_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._management_lock = lock

        with lock:
            try:
                live_orders = executor.list_pending_orders()
                positions = executor.list_positions()
            except Exception as exc:
                now = time.time()
                last = float(
                    getattr(self, "_pending_reconcile_error_ts", 0.0)
                )
                if now - last >= 30.0:
                    phase = "startup" if startup else "live"
                    print(
                        f"[Persistent LIMIT] {phase} reconciliation "
                        f"deferred (fail-closed): {exc}"
                    )
                    self._pending_reconcile_error_ts = now
                return

            plans = list(self.pending_limits.values())
            known_comments = {
                plan.broker_comment_for_leg(leg.index)
                for plan in plans
                for leg in plan.legs
            }
            orphan_orders = [
                row
                for row in live_orders
                if str(row.get("comment") or "").startswith(
                    BROKER_COMMENT_PREFIX
                )
                and str(row.get("comment") or "") not in known_comments
            ]
            if orphan_orders:
                self._pending_state_safe = False
                tickets = sorted(
                    int(row.get("ticket") or 0) for row in orphan_orders
                )
                print(
                    "[Persistent LIMIT] orphan broker orders detected; "
                    f"new entries blocked: {tickets}"
                )
                # Admission is now fail-closed, but an unrelated orphan must
                # never freeze cancellation/reconciliation of plans whose
                # ownership is still known exactly.
            if not plans:
                return

            live_by_ticket = {
                int(row["ticket"]): row
                for row in live_orders
                if int(row.get("ticket") or 0) > 0
            }
            live_by_comment: Dict[str, List[Dict[str, Any]]] = {}
            for row in live_orders:
                comment = str(row.get("comment") or "")
                if comment in known_comments:
                    live_by_comment.setdefault(comment, []).append(row)
            positions_by_comment: Dict[str, List[Dict[str, Any]]] = {}
            for row in positions:
                comment = str(row.get("comment") or "")
                if comment in known_comments:
                    positions_by_comment.setdefault(comment, []).append(row)

            duplicate_comments = {
                comment
                for comment, rows in (
                    list(live_by_comment.items())
                    + list(positions_by_comment.items())
                )
                if len(rows) > 1
            }
            if duplicate_comments:
                self._pending_state_safe = False
                print(
                    "[Persistent LIMIT] duplicate per-leg broker identity; "
                    "new entries blocked: "
                    + ", ".join(sorted(duplicate_comments))
                )
                return

            attached_tickets = [
                int(leg.broker_order_ticket)
                for plan in plans
                for leg in plan.legs
                if leg.broker_order_ticket is not None
            ]
            oldest_created_at = min(
                float(plan.created_at) for plan in plans
            )
            lookback_days = max(
                1,
                int(
                    math.ceil(
                        max(0.0, time.time() - oldest_created_at)
                        / 86_400.0
                    )
                )
                + 1,
            )
            try:
                fills_by_order = executor.get_pending_order_fills(
                    attached_tickets,
                    comments=sorted(known_comments),
                    lookback_days=lookback_days,
                )
                if hasattr(executor, "get_pending_order_histories"):
                    histories_by_comment = (
                        executor.get_pending_order_histories(
                            sorted(known_comments),
                            lookback_days=lookback_days,
                        )
                    )
                else:
                    histories_by_comment = {}
            except Exception as exc:
                now = time.time()
                last = float(
                    getattr(self, "_pending_reconcile_error_ts", 0.0)
                )
                if now - last >= 30.0:
                    print(
                        "[Persistent LIMIT] deal reconciliation deferred "
                        f"(fail-closed): {exc}"
                    )
                    self._pending_reconcile_error_ts = now
                return

            history_by_ticket = {
                int(row["ticket"]): row
                for rows in histories_by_comment.values()
                for row in rows
                if int(row.get("ticket") or 0) > 0
            }

            fills_by_comment: Dict[str, List[Dict[str, Any]]] = {}
            for group in fills_by_order.values():
                comments = {
                    str(row.get("comment") or "")
                    for row in group.get("deals") or []
                    if str(row.get("comment") or "") in known_comments
                }
                for comment in comments:
                    fills_by_comment.setdefault(comment, []).append(group)
            duplicated_fill_comments = {
                comment
                for comment, groups in fills_by_comment.items()
                if len(
                    {
                        int(group.get("order_ticket") or 0)
                        for group in groups
                    }
                )
                > 1
            }
            if duplicated_fill_comments:
                self._pending_state_safe = False
                print(
                    "[Persistent LIMIT] multiple broker orders share one "
                    "leg identity; new entries blocked: "
                    + ", ".join(sorted(duplicated_fill_comments))
                )
                return

            removed: List[str] = []
            book_dirty = False
            now = time.time()
            for symbol, original in list(self.pending_limits.items()):
                plan = original
                plan_changed = False
                fill_time_msc: Optional[int] = None
                position_fill_visible = False

                # Recover the crash window after broker acceptance but before
                # the returned order ticket reached the durable state file.
                if not plan.is_terminal:
                    for leg in plan.legs:
                        comment = plan.broker_comment_for_leg(leg.index)
                        evidence = {
                            int(row.get("ticket") or 0)
                            for row in live_by_comment.get(comment, [])
                            if int(row.get("ticket") or 0) > 0
                        }
                        evidence.update(
                            int(group.get("order_ticket") or 0)
                            for group in fills_by_comment.get(comment, [])
                            if int(group.get("order_ticket") or 0) > 0
                        )
                        evidence.update(
                            int(row.get("ticket") or 0)
                            for row in histories_by_comment.get(
                                comment, []
                            )
                            if int(row.get("ticket") or 0) > 0
                        )
                        if leg.broker_order_ticket is not None:
                            evidence.add(int(leg.broker_order_ticket))
                        if len(evidence) > 1:
                            self._pending_state_safe = False
                            print(
                                f"[Persistent LIMIT] {symbol} leg "
                                f"{leg.index} maps to multiple tickets "
                                f"{sorted(evidence)}; new entries blocked"
                            )
                            return
                        if leg.broker_order_ticket is None and evidence:
                            ticket = next(iter(evidence))
                            updated_leg = leg.with_order(ticket)
                            plan = plan.replace_leg(
                                updated_leg,
                                now=max(time.time(), plan.updated_at),
                            )
                            plan_changed = True
                    if (
                        plan.state == PendingLimitState.PLACING
                        and all(
                            leg.broker_order_ticket is not None
                            for leg in plan.legs
                        )
                    ):
                        plan = plan.transition(
                            PendingLimitState.PLACED,
                            now=max(time.time(), plan.updated_at),
                        )
                        plan_changed = True

                # Deal tickets are immutable dedupe keys. Record every opening
                # fill incrementally, never feed an aggregate back to the model.
                if not plan.is_terminal:
                    for leg in plan.legs:
                        comment = plan.broker_comment_for_leg(leg.index)
                        group = None
                        if leg.broker_order_ticket is not None:
                            group = fills_by_order.get(
                                int(leg.broker_order_ticket)
                            )
                        if group is None:
                            groups = fills_by_comment.get(comment, [])
                            group = groups[0] if groups else None
                        for row in (group or {}).get("deals") or []:
                            deal_ticket = int(
                                row.get("deal_ticket") or 0
                            )
                            position_id = int(
                                row.get("position_id") or 0
                            )
                            row_time_msc = int(
                                row.get("time_msc") or 0
                            )
                            if row_time_msc > 0:
                                fill_time_msc = (
                                    row_time_msc
                                    if fill_time_msc is None
                                    else min(
                                        fill_time_msc,
                                        row_time_msc,
                                    )
                                )
                            if deal_ticket <= 0 or position_id <= 0:
                                continue
                            before = plan
                            plan = plan.record_fill(
                                leg.index,
                                float(row["volume"]),
                                float(row["price"]),
                                deal_ticket=deal_ticket,
                                position_id=position_id,
                                now=max(time.time(), plan.updated_at),
                            )
                            plan_changed = plan_changed or plan is not before

                        # A position can publish before its opening deal. It is
                        # sufficient to cancel sibling orders, but it must not
                        # be added as a synthetic fill: when the same deal later
                        # appears, incremental accounting would double-count it.
                        position_rows = positions_by_comment.get(comment, [])
                        position_fill_visible = (
                            position_fill_visible or bool(position_rows)
                        )

                if plan_changed:
                    self.pending_limits[symbol] = plan
                    self._save_pending_limit_state()
                    book_dirty = True

                plan_comments = {
                    plan.broker_comment_for_leg(leg.index)
                    for leg in plan.legs
                }
                active_trade = self.active_trades.get(symbol)
                plan_idea_id = str(
                    plan.signal_payload.get("idea_id") or plan.plan_id
                )
                active_idea_id = str(
                    getattr(active_trade, "idea_id", "") or ""
                )
                incompatible_active = bool(
                    active_trade is not None
                    and active_idea_id != plan_idea_id
                )
                incompatible_broker_position = any(
                    str(row.get("symbol") or "").upper()
                    == symbol.upper()
                    and str(row.get("comment") or "")
                    not in plan_comments
                    for row in positions
                )
                incompatible_exposure = bool(
                    incompatible_active
                    or incompatible_broker_position
                )
                conflict_plan = self._is_pending_conflict_plan(plan)
                if incompatible_exposure and not conflict_plan:
                    plan = self._mark_pending_conflict(plan)
                    self.pending_limits[symbol] = plan
                    self._save_pending_limit_state()
                    book_dirty = True
                    conflict_plan = True

                trade = None
                if plan.has_fills and not conflict_plan:
                    try:
                        trade = self._materialize_pending_trade(
                            plan,
                            fill_time_msc=fill_time_msc,
                        )
                    except Exception as exc:
                        self._pending_state_safe = False
                        print(
                            f"[Persistent LIMIT] {symbol} fill promotion "
                            f"failed closed: {exc}"
                        )
                        return
                elif plan.has_fills and conflict_plan:
                    self._record_pending_fill_terminal(
                        plan,
                        fill_time_msc=fill_time_msc,
                    )
                    print(
                        f"[Persistent LIMIT] {symbol} fill conflicts with "
                        "another active idea; durable cleanup retained"
                    )

                quote = self._capture_quote(symbol)
                if conflict_plan:
                    cancel_reason = str(
                        plan.reason or _PENDING_CONFLICT_REASON
                    )
                elif (
                    startup
                    and plan.state == PendingLimitState.PLACING
                    and not all(
                        leg.broker_order_ticket is not None
                        for leg in plan.legs
                    )
                ):
                    cancel_reason = (
                        "restart interrupted basket placement"
                    )
                elif position_fill_visible:
                    cancel_reason = (
                        "broker position visible; awaiting deal history"
                    )
                else:
                    cancel_reason = self._pending_cancel_reason(
                        plan,
                        now=now,
                        quote=quote,
                    )
                if cancel_reason and not plan.is_terminal:
                    has_broker_evidence = bool(
                        plan.has_fills
                        or any(
                            leg.broker_order_ticket is not None
                            or leg.submission_attempted
                            for leg in plan.legs
                        )
                    )
                    if has_broker_evidence:
                        if plan.state != PendingLimitState.CANCELLING:
                            plan = plan.transition(
                                PendingLimitState.CANCELLING,
                                now=max(time.time(), plan.updated_at),
                                reason=cancel_reason,
                            )
                    elif position_fill_visible:
                        # A live position proves broker acceptance, but the
                        # opening deal is not yet available for exact,
                        # idempotent volume accounting. Keep ownership durable.
                        pass
                    else:
                        target = (
                            PendingLimitState.EXPIRED
                            if plan.is_due(now)
                            else PendingLimitState.CANCELLED
                        )
                        plan = plan.transition(
                            target,
                            now=max(time.time(), plan.updated_at),
                            reason=cancel_reason,
                        )
                    self.pending_limits[symbol] = plan
                    self._save_pending_limit_state()
                    book_dirty = True

                should_cancel = bool(
                    cancel_reason
                    or plan.state
                    in {
                        PendingLimitState.CANCELLING,
                        PendingLimitState.CANCELLED,
                        PendingLimitState.EXPIRED,
                        PendingLimitState.FILLED,
                    }
                )
                if should_cancel:
                    for leg in plan.legs:
                        if leg.remaining_volume <= 1e-9:
                            continue
                        ticket = int(
                            leg.broker_order_ticket or 0
                        )
                        row = live_by_ticket.get(ticket)
                        if row is None:
                            comment = plan.broker_comment_for_leg(
                                leg.index
                            )
                            rows = live_by_comment.get(comment, [])
                            row = rows[0] if rows else None
                            if row is not None:
                                ticket = int(row.get("ticket") or 0)
                        if row is None or ticket <= 0:
                            continue
                        try:
                            if executor.cancel_pending_order(
                                ticket,
                                expected_comment_prefix=plan.broker_comment,
                            ):
                                live_by_ticket.pop(ticket, None)
                                live_by_comment.pop(
                                    str(row.get("comment") or ""),
                                    None,
                                )
                        except Exception as exc:
                            print(
                                f"[Persistent LIMIT] {symbol} cancel "
                                f"{ticket} deferred: {exc}"
                            )

                all_resolved = True
                for leg in plan.legs:
                    if leg.remaining_volume <= 1e-9:
                        continue
                    ticket = int(leg.broker_order_ticket or 0)
                    if ticket <= 0:
                        if leg.submission_attempted:
                            if (
                                plan.state == PendingLimitState.FAILED
                                or str(plan.reason or "").startswith(
                                    _PENDING_DEFINITIVE_REJECTION_PREFIX
                                )
                            ):
                                continue
                            if now < float(plan.expires_at) + 30.0:
                                all_resolved = False
                            # After server-side expiry plus publication grace,
                            # healthy exact-comment live/order/deal queries with
                            # no evidence prove the ambiguous send did not leave
                            # working risk.
                            continue
                        if plan.state in {
                            PendingLimitState.CANCELLING,
                            PendingLimitState.CANCELLED,
                            PendingLimitState.EXPIRED,
                            PendingLimitState.FILLED,
                            PendingLimitState.FAILED,
                        }:
                            continue
                        all_resolved = False
                        continue
                    if ticket in live_by_ticket:
                        all_resolved = False
                        continue
                    history = history_by_ticket.get(ticket)
                    if history is None:
                        try:
                            history = executor.get_pending_order_history(
                                ticket
                            )
                        except Exception as exc:
                            print(
                                f"[Persistent LIMIT] {symbol} history "
                                f"{ticket} deferred: {exc}"
                            )
                            all_resolved = False
                            continue
                    if history is None:
                        all_resolved = False
                        continue
                    if (
                        bool(history.get("had_execution"))
                        and leg.filled_volume <= 1e-9
                    ):
                        all_resolved = False
                        continue
                    done_msc = int(
                        history.get("time_done_msc") or 0
                    )
                    if (
                        leg.filled_volume <= 1e-9
                        and done_msc > 0
                        and int(time.time() * 1000) - done_msc
                        < 30_000
                    ):
                        # Terminal order history can precede the opening deal
                        # during a cancel/fill race. Preserve ownership through
                        # a bounded publication grace window.
                        all_resolved = False
                        continue
                    if bool(history.get("fill_expected")):
                        # Order history may publish before its opening deal.
                        # Once at least one opening deal is already durable,
                        # a terminal FILLED history row also resolves a broker
                        # short-fill remainder. With no deal yet, keep waiting.
                        if not (
                            bool(history.get("terminal"))
                            and leg.filled_volume > 0
                        ):
                            all_resolved = False
                            continue
                    if not bool(history.get("terminal")):
                        all_resolved = False

                if not all_resolved:
                    continue

                watcher = getattr(self, "shadow_tick_watcher", None)
                if conflict_plan:
                    if PERSISTENT_LIMIT_CONFLICT_AUTOCLOSE_ENABLED:
                        try:
                            capabilities = (
                                executor.pending_limit_capabilities(
                                    self.universe.get(
                                        plan.symbol,
                                        plan.symbol,
                                    )
                                )
                            )
                        except Exception as exc:
                            self._pending_state_safe = False
                            print(
                                f"[Persistent LIMIT] {symbol} emergency "
                                "close capability probe failed; durable "
                                f"plan retained: {exc}"
                            )
                            continue
                        if not bool(
                            isinstance(capabilities, dict)
                            and capabilities.get("ready") is True
                            and capabilities.get("account_hedging") is True
                        ):
                            self._pending_state_safe = False
                            print(
                                f"[Persistent LIMIT] {symbol} emergency "
                                "close disabled by broker/account "
                                "capabilities; durable plan retained: "
                                f"{capabilities!r}"
                            )
                            continue
                        close_targets, ownership_errors = (
                            self._pending_plan_close_targets(
                                plan,
                                positions,
                                executor_magic=getattr(
                                    getattr(executor, "settings", None),
                                    "magic",
                                    None,
                                ),
                            )
                        )
                        if ownership_errors:
                            self._pending_state_safe = False
                            print(
                                f"[Persistent LIMIT] {symbol} emergency "
                                "close ownership ambiguous; durable plan "
                                "retained: "
                                + "; ".join(ownership_errors)
                            )
                            continue

                        if close_targets:
                            retry_deferred = bool(
                                plan.reason
                                == _PENDING_CONFLICT_CLOSE_REQUESTED_REASON
                                and time.time() - float(plan.updated_at)
                                < _PENDING_CONFLICT_ABSENCE_GRACE_SEC
                            )
                            if retry_deferred:
                                print(
                                    f"[Persistent LIMIT] {symbol} emergency "
                                    "close awaiting broker publication; "
                                    "durable plan retained"
                                )
                                continue

                            close_failures: List[str] = []
                            for (
                                leg_index,
                                position_id,
                                expected_comment,
                            ) in close_targets:
                                try:
                                    closed = executor.close_trade(
                                        plan.symbol,
                                        position_id=position_id,
                                        volume=None,
                                        expected_comment=expected_comment,
                                    )
                                except Exception as exc:
                                    close_failures.append(
                                        f"leg {leg_index} position "
                                        f"{position_id}: {exc}"
                                    )
                                    continue
                                if not closed:
                                    close_failures.append(
                                        f"leg {leg_index} position "
                                        f"{position_id}: broker close "
                                        "not confirmed"
                                    )

                            outcome = (
                                _PENDING_CONFLICT_CLOSE_FAILED_REASON
                                if close_failures
                                else _PENDING_CONFLICT_CLOSE_REQUESTED_REASON
                            )
                            updated_plan = (
                                self._with_pending_conflict_reason(
                                    plan,
                                    outcome,
                                    now=time.time(),
                                )
                            )
                            if updated_plan is not plan:
                                plan = updated_plan
                                self.pending_limits[symbol] = plan
                                self._save_pending_limit_state()
                                book_dirty = True
                            if close_failures:
                                print(
                                    f"[Persistent LIMIT] {symbol} emergency "
                                    "close deferred; durable plan retained: "
                                    + "; ".join(close_failures)
                                )
                            else:
                                print(
                                    f"[Persistent LIMIT] {symbol} emergency "
                                    "close requested for exact owned "
                                    "positions; awaiting absence: "
                                    f"{[row[1] for row in close_targets]}"
                                )
                            continue

                        if plan.reason in {
                            _PENDING_CONFLICT_CLOSE_REQUESTED_REASON,
                            _PENDING_CONFLICT_CLOSE_FAILED_REASON,
                        }:
                            resolved = self._with_pending_conflict_reason(
                                plan,
                                _PENDING_CONFLICT_CLOSE_RESOLVED_REASON,
                                now=time.time(),
                            )
                            if resolved is not plan:
                                plan = resolved
                                self.pending_limits[symbol] = plan
                                self._save_pending_limit_state()
                                book_dirty = True
                            print(
                                f"[Persistent LIMIT] {symbol} emergency "
                                "close exposure absent; durable plan retained "
                                "for explicit/manual recovery"
                            )
                            continue

                        if (
                            plan.reason
                            == _PENDING_CONFLICT_CLOSE_RESOLVED_REASON
                        ):
                            print(
                                f"[Persistent LIMIT] {symbol} resolved "
                                "emergency-close audit retained for "
                                "explicit/manual recovery"
                            )
                            continue

                        if (
                            time.time() - float(plan.updated_at)
                            < _PENDING_CONFLICT_ABSENCE_GRACE_SEC
                        ):
                            # A conflict can be detected from an active idea
                            # before the new position publishes. Never claim a
                            # resolved emergency close that was not requested.
                            continue
                        print(
                            f"[Persistent LIMIT] {symbol} conflicting fill "
                            "exposure absent without an emergency close "
                            "request; durable plan retained"
                        )
                        continue

                    owned_position_ids, ownership_errors = (
                        self._pending_plan_owned_position_ids(
                            plan,
                            positions,
                        )
                    )
                    if ownership_errors:
                        self._pending_state_safe = False
                        print(
                            f"[Persistent LIMIT] {symbol} conflict "
                            "ownership ambiguous; durable plan retained: "
                            + "; ".join(ownership_errors)
                        )
                        continue
                    if owned_position_ids:
                        print(
                            f"[Persistent LIMIT] {symbol} conflict "
                            "positions still open; durable plan retained: "
                            f"{list(owned_position_ids)}"
                        )
                        continue
                    if (
                        time.time() - float(plan.updated_at)
                        < _PENDING_CONFLICT_ABSENCE_GRACE_SEC
                    ):
                        # Deal history can precede publication of the position.
                        # Require one bounded absence window before retirement.
                        continue
                    if plan.has_fills:
                        print(
                            f"[Persistent LIMIT] {symbol} conflicting "
                            "fill exposure absent; durable plan retained "
                            "for explicit recovery"
                        )
                        continue

                if plan.has_fills:
                    if trade is None:
                        continue
                    if plan.state != PendingLimitState.FILLED:
                        plan = plan.transition(
                            PendingLimitState.FILLED,
                            now=max(time.time(), plan.updated_at),
                            reason="all pending remainders resolved",
                        )
                        self.pending_limits[symbol] = plan
                        self._save_pending_limit_state()
                    trade = self._materialize_pending_trade(
                        plan,
                        fill_time_msc=fill_time_msc,
                        queue_entry_event=True,
                    )
                    if trade is None:
                        continue
                    if plan.entry_announcement_pending:
                        # Durable outbox: retain terminal ownership until
                        # Telegram/journal delivery explicitly acknowledges it.
                        continue
                    removed.append(symbol)
                    print(
                        f"[Persistent LIMIT] {symbol} fill promoted; "
                        "pending plan retired"
                    )
                else:
                    if not plan.is_terminal:
                        target = (
                            PendingLimitState.EXPIRED
                            if plan.is_due(now)
                            else PendingLimitState.CANCELLED
                        )
                        plan = plan.transition(
                            target,
                            now=max(time.time(), plan.updated_at),
                            reason=(
                                cancel_reason
                                or "broker pending order ended without fill"
                            ),
                        )
                        self.pending_limits[symbol] = plan
                        self._save_pending_limit_state()
                    definitive_rejection = bool(
                        plan.state == PendingLimitState.FAILED
                        and str(plan.reason or "").startswith(
                            _PENDING_DEFINITIVE_REJECTION_PREFIX
                        )
                    )
                    if watcher is not None and not definitive_rejection:
                        if plan.state == PendingLimitState.EXPIRED:
                            terminal_msc = int(
                                float(plan.expires_at) * 1000
                            )
                        elif (
                            quote is not None
                            and cancel_reason is not None
                            and "path crossed" in cancel_reason
                        ):
                            # ``symbol_info_tick.time_msc`` can be broker-server
                            # local while history queries are UTC.  The cancel
                            # decision happened now, so its causal horizon is
                            # the UTC management observation, not the raw quote
                            # timestamp.
                            terminal_msc = int(now * 1000)
                        else:
                            terminal_msc = int(time.time() * 1000)
                        watcher.record_terminal(
                            plan.plan_id,
                            (
                                "EXPIRED"
                                if plan.state
                                == PendingLimitState.EXPIRED
                                else "CANCELLED"
                            ),
                            at_utc=datetime.fromtimestamp(
                                terminal_msc / 1000.0,
                                tz=timezone.utc,
                            ),
                            tick_msc=terminal_msc,
                            reason=plan.reason,
                            inclusive=False,
                        )
                    removed.append(symbol)
                    print(
                        f"[Persistent LIMIT] {symbol} retired "
                        f"without fill ({plan.state.value})"
                    )

            if removed:
                for symbol in removed:
                    self.pending_limits.pop(symbol, None)
                self._save_pending_limit_state()
                book_dirty = True
            if book_dirty:
                self._pending_reconcile_error_ts = 0.0

    def ack_pending_limit_entry(self, plan_id: str) -> bool:
        """Durably acknowledge delivery of one broker-filled ENTER outbox."""
        lock = getattr(self, "_management_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._management_lock = lock
        with lock:
            for symbol, plan in self.pending_limits.items():
                if plan.plan_id != str(plan_id):
                    continue
                if not plan.has_fills:
                    return False
                updated = plan.mark_entry_announced(
                    now=max(time.time(), plan.updated_at),
                )
                self.pending_limits[symbol] = updated
                self._save_pending_limit_state()
                return True
        return False

    def bind_pending_entry_message(
        self,
        symbol: str,
        plan_id: str,
        *,
        chat_id: int,
        message_id: int,
    ) -> bool:
        """Persist Telegram identity before the durable outbox is acked."""
        lock = getattr(self, "_management_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._management_lock = lock
        with lock:
            plan = self.pending_limits.get(symbol)
            trade = self.active_trades.get(symbol)
            if (
                plan is None
                or plan.plan_id != str(plan_id)
                or trade is None
            ):
                return False
            trade.telegram_chat_id = int(chat_id)
            trade.telegram_message_id = int(message_id)
            save_active_trades(
                self.active_trades,
                AI_DATA_DIR / "active_trades.json",
            )
            return True

    def _hydrate_active_trades_from_mt5(self) -> None:
        try:
            positions = self.mt5_executor.list_positions() if self.mt5_executor else []
        except Exception as exc:
            print(f"[Core] MT5 hydration skipped: {exc}")
            return

        if not positions:
            return

        # Group positions by symbol so we can properly rebuild split_position_ids
        from collections import defaultdict
        by_symbol: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        pending_comments = {
            plan.broker_comment_for_leg(leg.index)
            for plan in getattr(self, "pending_limits", {}).values()
            for leg in plan.legs
        }
        for pos in positions:
            symbol = pos.get("symbol")
            if str(pos.get("comment") or "") in pending_comments:
                # The richer pending plan owns this position until opening deal
                # history is reconciled. Generic hydration would create a
                # second, incompatible idea identity.
                continue
            if symbol and symbol in self.universe:
                by_symbol[symbol].append(pos)

        hydrated = 0
        relinked = 0
        for symbol, sym_positions in by_symbol.items():
            first = sym_positions[0]
            all_tickets = [int(p["ticket"]) for p in sym_positions if p.get("ticket")]
            total_volume = round(sum(float(p.get("volume", 0.0) or 0.0) for p in sym_positions), 2)
            trade = self.active_trades.get(symbol)
            was_split = bool(
                trade
                and (
                    getattr(trade, "split_legs", {})
                    or getattr(trade, "split_position_ids", [])
                )
            )
            comment_marks_split = any(
                re.search(r"(?:^|\s)TP\s*\d+(?:\s|$)", str(p.get("comment") or ""), re.IGNORECASE)
                for p in sym_positions
            )
            # A restarted split setup can have only its final leg left. Do not
            # downgrade that one remaining leg to monitor mode.
            is_split = was_split or len(all_tickets) > 1 or comment_marks_split

            if trade:
                # A position-adding idea holds several entries on one symbol, so
                # the broker rows are first split by the tickets each entry
                # already owns. Unclaimed rows fall back to the primary entry —
                # the pre-pyramid behaviour for an ordinary single-entry setup.
                for entry, entry_positions in self._partition_positions_by_entry(
                    trade, sym_positions
                ):
                    if not entry_positions:
                        # No visible legs for this entry — leave its persisted
                        # state alone and let the lifecycle poll confirm the
                        # closes against deal history.
                        continue
                    entry_tickets = [
                        int(p["ticket"]) for p in entry_positions if p.get("ticket")
                    ]
                    entry_volume = round(
                        sum(float(p.get("volume", 0.0) or 0.0) for p in entry_positions), 2
                    )
                    entry_is_split = bool(
                        self._is_split_trade(entry)
                        or len(entry_tickets) > 1
                        or any(
                            re.search(
                                r"(?:^|\s)TP\s*\d+(?:\s|$)",
                                str(p.get("comment") or ""),
                                re.IGNORECASE,
                            )
                            for p in entry_positions
                        )
                    )
                    updated = False
                    if entry_is_split:
                        if self._ensure_split_leg_mapping(entry, positions=entry_positions):
                            entry.mt5_position_id = entry_tickets[0]
                            entry.mt5_ticket = entry_tickets[0]
                            updated = True
                    else:
                        ticket = entry_tickets[0] if entry_tickets else None
                        if ticket and entry.mt5_position_id != ticket:
                            entry.mt5_position_id = ticket
                            entry.mt5_ticket = ticket
                            updated = True
                    if abs(float(entry.volume or 0.0) - entry_volume) > 1e-6:
                        entry.volume = entry_volume
                        updated = True
                    if updated:
                        relinked += 1
                        label = (
                            symbol
                            if int(getattr(entry, "entry_index", 1) or 1) <= 1
                            else f"{symbol} entry#{entry.entry_index}"
                        )
                        print(
                            f"[Core] relinked {label}: split_ids={entry_tickets}, vol={entry_volume}"
                            if entry_is_split else
                            f"[Core] relinked {label}: ticket={entry_tickets[0] if entry_tickets else None}, vol={entry_volume}"
                        )
                continue

            # Position(s) not in active_trades — hydrate from MT5
            tp_prices = sorted({float(p.get("tp", 0.0) or 0.0) for p in sym_positions} - {0.0})
            narrative = first.get("comment") or "Hydrated from MT5"
            trade = ActiveTrade(
                side=str(first.get("side", "LONG")).upper(),
                entry=float(first.get("entry_price", 0.0) or 0.0),
                stop=float(first.get("stop", 0.0) or 0.0),
                tp_prices=tp_prices,
                tf="15m",
                narrative=narrative,
                symbol=symbol,
            )
            trade.volume = total_volume
            trade.volume_remaining = total_volume
            trade.mt5_ticket = all_tickets[0] if all_tickets else None
            trade.mt5_position_id = all_tickets[0] if all_tickets else None
            if is_split:
                trade.split_position_ids = all_tickets
                self._ensure_split_leg_mapping(trade, positions=sym_positions)
            trade.ts_open = float(first.get("time", time.time()) or time.time())
            trade.last_price_ts = time.time()
            self.active_trades[symbol] = trade
            hydrated += 1
            print(
                f"[Core] hydrated {symbol} (split {len(all_tickets)} legs, vol={total_volume}, tps={tp_prices})"
                if is_split else
                f"[Core] hydrated {symbol} (vol={total_volume}, tp={tp_prices})"
            )

        if hydrated or relinked:
            save_active_trades(self.active_trades, AI_DATA_DIR / "active_trades.json")
            print(f"[Core] hydration done: {hydrated} new, {relinked} relinked")

    @staticmethod
    def _is_split_trade(trade: ActiveTrade) -> bool:
        """True for both current and legacy persisted split setups."""
        return bool(
            getattr(trade, "split_legs", {})
            or getattr(trade, "split_position_ids", [])
        )

    @staticmethod
    def _idea_entries(trade: ActiveTrade) -> List[ActiveTrade]:
        """All entries of one idea: primary first, then position-adding add-ons."""
        entries = getattr(trade, "entries", None)
        if callable(entries):
            return entries()
        return [trade]

    @staticmethod
    def _partition_positions_by_entry(
        trade: ActiveTrade,
        positions: List[Dict[str, Any]],
    ) -> List[Tuple[ActiveTrade, List[Dict[str, Any]]]]:
        """Assign broker rows to the idea entry that already owns their ticket.

        Persisted state wins over broker grouping: only tickets no entry claims
        are handed to the primary entry, which keeps single-entry setups (and
        genuinely unknown positions) behaving exactly as before.
        """
        entries = Core._idea_entries(trade)
        buckets: List[Tuple[ActiveTrade, List[Dict[str, Any]]]] = [
            (entry, []) for entry in entries
        ]
        owner_by_ticket: Dict[int, int] = {}
        for idx, entry in enumerate(entries):
            known = set(int(t) for t in (getattr(entry, "split_legs", {}) or {}))
            known.update(int(t) for t in (getattr(entry, "split_position_ids", []) or []))
            for attr in ("mt5_position_id", "mt5_ticket"):
                value = getattr(entry, attr, None)
                if value:
                    known.add(int(value))
            for ticket in known:
                owner_by_ticket.setdefault(ticket, idx)

        for pos in positions or []:
            try:
                ticket = int(pos.get("ticket") or 0)
            except (TypeError, ValueError):
                ticket = 0
            buckets[owner_by_ticket.get(ticket, 0)][1].append(pos)
        return buckets

    def _ensure_split_leg_mapping(
        self,
        trade: ActiveTrade,
        *,
        positions: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Normalize/backfill the durable ticket -> TP metadata mapping.

        ``positions`` is supplied during startup hydration and lets us recover a
        TP index from the broker TP/comment. Without broker rows (legacy state),
        the old ordered ``split_position_ids`` list plus ``tp_hit`` is used.
        Returns True when persisted state changed.
        """
        before_legs = {
            int(k): dict(v or {})
            for k, v in (getattr(trade, "split_legs", {}) or {}).items()
            if k is not None
        }
        before_ids = [int(x) for x in (getattr(trade, "split_position_ids", []) or [])]
        legs: Dict[int, Dict[str, Any]] = {}
        for raw_ticket, raw_meta in before_legs.items():
            try:
                ticket = int(raw_ticket)
            except (TypeError, ValueError):
                continue
            if ticket <= 0:
                continue
            meta = dict(raw_meta or {})
            try:
                meta["tp_index"] = int(meta.get("tp_index") or 0)
            except (TypeError, ValueError):
                meta["tp_index"] = 0
            for key in ("tp", "volume"):
                try:
                    meta[key] = float(meta.get(key) or 0.0)
                except (TypeError, ValueError):
                    meta[key] = 0.0
            meta["status"] = str(meta.get("status") or "open")
            legs[ticket] = meta

        position_by_ticket: Dict[int, Dict[str, Any]] = {}
        for pos in positions or []:
            try:
                ticket = int(pos.get("ticket") or 0)
            except (TypeError, ValueError):
                continue
            if ticket > 0:
                position_by_ticket[ticket] = pos

        tracked_ids: List[int] = []
        for raw_ticket in before_ids:
            if raw_ticket > 0 and raw_ticket not in tracked_ids:
                tracked_ids.append(raw_ticket)
        for ticket in position_by_ticket:
            if ticket not in tracked_ids:
                tracked_ids.append(ticket)

        tps = [float(x) for x in (getattr(trade, "tp_prices", []) or [])]
        volumes = [float(x) for x in (getattr(trade, "volume_per_tp", []) or [])]
        used_indices = {
            int(meta.get("tp_index") or 0)
            for meta in legs.values()
            if int(meta.get("tp_index") or 0) > 0
            and not meta.get("legacy_inferred")
        }

        def _resolve_index(ticket: int, pos: Optional[Dict[str, Any]]) -> int:
            comment = str((pos or {}).get("comment") or "")
            match = re.search(r"(?:^|\s)TP\s*(\d+)(?:\s|$)", comment, re.IGNORECASE)
            if match:
                idx = int(match.group(1))
                if idx > 0:
                    return idx
            broker_tp = float((pos or {}).get("tp") or 0.0)
            if broker_tp > 0 and tps:
                candidates = [
                    (abs(tp - broker_tp), idx)
                    for idx, tp in enumerate(tps, start=1)
                    if idx not in used_indices
                ]
                if candidates:
                    return min(candidates)[1]
            start = max(1, int(getattr(trade, "tp_hit", 0) or 0) + 1)
            for idx in range(start, max(start, len(tps)) + 2):
                if idx not in used_indices:
                    return idx
            return start

        for ticket in tracked_ids:
            pos = position_by_ticket.get(ticket)
            meta = legs.get(ticket)
            if meta is None:
                idx = _resolve_index(ticket, pos)
                used_indices.add(idx)
                meta = {
                    "tp_index": idx,
                    "tp": (
                        float((pos or {}).get("tp") or 0.0)
                        or (float(tps[idx - 1]) if idx <= len(tps) else 0.0)
                    ),
                    "volume": (
                        float((pos or {}).get("volume") or 0.0)
                        or (float(volumes[idx - 1]) if idx <= len(volumes) else 0.0)
                    ),
                    "status": "open",
                }
                legs[ticket] = meta
            elif pos is not None:
                # Broker says this ticket is currently open; refresh mutable
                # broker values without losing its original TP identity.
                if meta.pop("legacy_inferred", False):
                    idx = _resolve_index(ticket, pos)
                    meta["tp_index"] = idx
                    used_indices.add(idx)
                meta["status"] = "open"
                if float(pos.get("tp") or 0.0) > 0:
                    meta["tp"] = float(pos["tp"])
                if float(pos.get("volume") or 0.0) > 0:
                    meta["volume"] = float(pos["volume"])

        if positions is not None:
            # During hydration this list must mean *currently visible* legs. The
            # complete historical identity remains in split_legs.
            tracked_ids = sorted(
                position_by_ticket,
                key=lambda ticket: (
                    int(legs.get(ticket, {}).get("tp_index") or 10**6),
                    ticket,
                ),
            )

        trade.split_legs = legs
        trade.split_position_ids = tracked_ids
        return legs != before_legs or tracked_ids != before_ids

    def _poll_split_lifecycle(
        self,
        symbol: str,
        trade: ActiveTrade,
        *,
        last_price: Optional[float] = None,
        state_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reconcile one split setup against open positions and deal history.

        The function is idempotent: a TP event is emitted only on the transition
        from open/pending to closed. Missing deal history keeps the ticket in a
        pending state and is retried on the next management tick.
        """
        result: Dict[str, Any] = {
            "events": [],
            "changed": self._ensure_split_leg_mapping(trade),
            "final": False,
            "query_failed": False,
            "pending_history": False,
            "visible_open_count": 0,
        }
        if not self.mt5_executor or not getattr(trade, "split_legs", {}):
            return result

        open_ids = self.mt5_executor.get_open_position_ids(symbol)
        if open_ids is None:
            result["query_failed"] = True
            return result

        remaining: List[int] = []
        ordered_legs = sorted(
            trade.split_legs.items(),
            key=lambda item: (int(item[1].get("tp_index") or 10**6), int(item[0])),
        )
        for ticket, meta in ordered_legs:
            ticket = int(ticket)
            status = str(meta.get("status") or "open")
            if ticket in open_ids:
                remaining.append(ticket)
                result["visible_open_count"] += 1
                if status != "open":
                    if status == "closed" and meta.get("tp_event_emitted"):
                        trade.tp_hit = max(0, int(getattr(trade, "tp_hit", 0) or 0) - 1)
                    meta["status"] = "open"
                    meta["missing_count"] = 0
                    for key in list(meta):
                        if key.startswith("close_") or key in {"closed_at", "tp_event_emitted"}:
                            meta.pop(key, None)
                    result["changed"] = True
                continue
            if status == "closed":
                continue

            close_info = self.mt5_executor.get_position_close_info(ticket)
            if close_info is None:
                remaining.append(ticket)
                result["pending_history"] = True
                if status != "pending_history":
                    meta["status"] = "pending_history"
                    result["changed"] = True
                continue

            expected_volume = float(meta.get("volume") or 0.0)
            closed_volume = float(close_info.get("volume") or 0.0)
            volume_tolerance = max(1e-8, expected_volume * 1e-4)
            if expected_volume > 0 and closed_volume + volume_tolerance < expected_volume:
                remaining.append(ticket)
                result["pending_history"] = True
                meta["status"] = "pending_history"
                meta["observed_exit_volume"] = closed_volume
                result["changed"] = True
                continue

            # Require two consecutive trusted absence polls before consuming a
            # close. This prevents a transient empty positions_get() result from
            # turning an older partial-exit deal into a terminal leg event.
            missing_count = int(meta.get("missing_count") or 0) + 1
            meta["missing_count"] = missing_count
            required = max(2, int(getattr(self, "_broker_missing_confirm", 2) or 2))
            if missing_count < required:
                remaining.append(ticket)
                result["pending_history"] = True
                meta["status"] = "pending_close_confirmation"
                result["changed"] = True
                continue

            meta["status"] = "closed"
            meta["close_reason"] = str(close_info.get("reason") or "OTHER")
            meta["close_reason_code"] = int(close_info.get("reason_code") or 0)
            meta["close_deal_ticket"] = int(close_info.get("deal_ticket") or 0)
            meta["close_price"] = float(close_info.get("price") or 0.0)
            meta["close_volume"] = float(close_info.get("volume") or 0.0)
            meta["close_profit"] = float(close_info.get("profit") or 0.0)
            meta["close_commission"] = float(close_info.get("commission") or 0.0)
            meta["close_swap"] = float(close_info.get("swap") or 0.0)
            meta["close_fee"] = float(close_info.get("fee") or 0.0)
            meta["close_net"] = float(
                close_info.get("net")
                if close_info.get("net") is not None
                else meta["close_profit"]
                + meta["close_commission"]
                + meta["close_swap"]
                + meta["close_fee"]
            )
            meta["closed_at"] = int(close_info.get("time") or 0)
            result["changed"] = True

            reason = meta["close_reason"]
            tp_index = int(meta.get("tp_index") or 0)
            if reason == "TP":
                trade.tp_hit = max(
                    int(getattr(trade, "tp_hit", 0) or 0) + 1,
                    tp_index,
                )
                meta["tp_event_emitted"] = True
                tp_price = float(meta.get("tp") or 0.0)
                hit_price = float(meta.get("close_price") or tp_price or last_price or 0.0)
                result["events"].append({
                    "type": "TP",
                    "tp_index": tp_index,
                    "tp_price": tp_price or None,
                    "hit_price": hit_price,
                    "source": "broker_deal",
                    "position_id": ticket,
                    "close_deal_ticket": meta["close_deal_ticket"],
                })
                print(
                    f"[Core] {symbol} leg {ticket} closed by broker TP{tp_index} "
                    f"at {hit_price:.5f}"
                )
            else:
                print(f"[Core] {symbol} leg {ticket} closed by broker ({reason})")

        if trade.split_position_ids != remaining:
            trade.split_position_ids = remaining
            result["changed"] = True

        remaining_volume = sum(
            float(meta.get("volume") or 0.0)
            for meta in trade.split_legs.values()
            if str(meta.get("status") or "open") != "closed"
        )
        if remaining_volume > 0 and abs(float(trade.volume_remaining or 0.0) - remaining_volume) > 1e-8:
            trade.volume_remaining = round(remaining_volume, 8)
            result["changed"] = True
        elif not remaining and not result["pending_history"] and trade.volume_remaining != 0.0:
            trade.volume_remaining = 0.0
            result["changed"] = True

        # Each entry of a position-adding idea confirms its own disappearance,
        # so the absence counter is keyed per entry, not per symbol. The
        # confirmation is then latched: an idea closes only once every entry is
        # confirmed gone, and the per-entry confirmations rarely land on the
        # same tick.
        missing_key = state_key or symbol
        if remaining:
            self._broker_missing_counts.pop(missing_key, None)
            if getattr(trade, "broker_closed", False):
                trade.broker_closed = False
                result["changed"] = True
        elif getattr(trade, "broker_closed", False):
            result["final"] = True
        elif self._position_gone_confirmed(missing_key, query_failed=False):
            self._broker_missing_counts.pop(missing_key, None)
            trade.broker_closed = True
            result["changed"] = True
            result["final"] = True
        return result

    def _poll_idea_lifecycle(
        self,
        symbol: str,
        trade: ActiveTrade,
        *,
        last_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Reconcile every entry of one idea against the broker.

        The idea is final only once all of its entries are, so a closed add-on
        never terminates a still-open primary entry (or the other way round).
        Events carry the entry index they came from.
        """
        aggregate: Dict[str, Any] = {
            "events": [],
            "changed": False,
            "final": True,
            "query_failed": False,
            "pending_history": False,
            "visible_open_count": 0,
        }
        for entry in self._idea_entries(trade):
            index = int(getattr(entry, "entry_index", 1) or 1)
            state_key = symbol if index <= 1 else f"{symbol}#{index}"
            if self._is_split_trade(entry):
                result = self._poll_split_lifecycle(
                    symbol, entry, last_price=last_price, state_key=state_key
                )
            else:
                # An entry without broker-managed legs still has to reach a
                # terminal state, or the whole idea would never close.
                result = self._poll_single_position(symbol, entry, state_key=state_key)
            for event in result.get("events") or []:
                if index > 1:
                    event.setdefault("entry_index", index)
            aggregate["events"].extend(result.get("events") or [])
            aggregate["changed"] = aggregate["changed"] or bool(result.get("changed"))
            aggregate["query_failed"] = aggregate["query_failed"] or bool(result.get("query_failed"))
            aggregate["pending_history"] = aggregate["pending_history"] or bool(result.get("pending_history"))
            aggregate["visible_open_count"] += int(result.get("visible_open_count") or 0)
            aggregate["final"] = aggregate["final"] and bool(result.get("final"))
        return aggregate

    def _poll_single_position(
        self, symbol: str, entry: ActiveTrade, *, state_key: str
    ) -> Dict[str, Any]:
        """Lifecycle of one entry that is a single broker position, not split legs."""
        result: Dict[str, Any] = {
            "events": [],
            "changed": False,
            "final": False,
            "query_failed": False,
            "pending_history": False,
            "visible_open_count": 0,
        }
        if not self.mt5_executor:
            return result
        pos_id = getattr(entry, "mt5_position_id", None) or getattr(entry, "mt5_ticket", None)
        if not pos_id:
            return result
        if self.mt5_executor.get_position(symbol, pos_id) is not None:
            result["visible_open_count"] = 1
            self._broker_missing_counts.pop(state_key, None)
            if getattr(entry, "broker_closed", False):
                entry.broker_closed = False
                result["changed"] = True
            return result
        if getattr(entry, "broker_closed", False):
            result["final"] = True
            return result
        if self._position_gone_confirmed(state_key, query_failed=False):
            self._broker_missing_counts.pop(state_key, None)
            entry.broker_closed = True
            result["changed"] = True
            result["final"] = True
        return result

    def _move_idea_entries_to_breakeven(
        self, symbol: str, trade: ActiveTrade
    ) -> List[Dict[str, Any]]:
        """Run the TP1 break-even rule for each entry that has hit its own TP1."""
        events: List[Dict[str, Any]] = []
        for entry in self._idea_entries(trade):
            if getattr(entry, "moved_to_be", False):
                continue
            if int(getattr(entry, "tp_hit", 0) or 0) < 1:
                continue
            if self._is_split_trade(entry):
                # Nothing left to modify once every leg is gone.
                if not getattr(entry, "split_position_ids", []):
                    continue
            elif not (
                getattr(entry, "mt5_position_id", None) or getattr(entry, "mt5_ticket", None)
            ):
                continue
            if self._move_to_breakeven(symbol, entry):
                event: Dict[str, Any] = {"type": "BE", "price": float(entry.entry)}
                index = int(getattr(entry, "entry_index", 1) or 1)
                if index > 1:
                    event["entry_index"] = index
                events.append(event)
        return events

    def manage_active_trades(self) -> Dict[str, dict]:
        """Fast broker-only management pass (safe to run every 1–5 seconds).

        This deliberately does not fetch candles, generate signals, call the AI
        filter, or journal partial HOLDs. It only consumes authoritative broker
        leg-close events, moves remaining split legs to BE, and emits one terminal
        EXIT_BROKER when all legs are confirmed closed.
        """
        if not self.mt5_executor:
            return {}
        results: Dict[str, dict] = {}
        dirty = False
        lock = getattr(self, "_management_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._management_lock = lock
        with lock:
            self._manage_pending_limits()
            pending_events = getattr(
                self, "_pending_entry_events", None
            )
            if pending_events:
                results.update(pending_events)
                pending_events.clear()
            fresh_pending_fills = {
                symbol
                for symbol, signal in results.items()
                if signal.get("signal") == "ENTER"
            }
            for symbol, trade in list(self.active_trades.items()):
                if symbol in fresh_pending_fills:
                    # Deliver the durable ENTER first. Normal lifecycle polling
                    # resumes on the next fast tick.
                    continue
                scheduled_flat = self._is_friday_weekend_close() or self._is_daily_flat_close()
                if scheduled_flat:
                    last_price = self.mt5_executor.get_current_price(symbol, trade.side)
                    label = (
                        f"Weekend close (Friday {FRIDAY_CLOSE_HOUR}:00 UTC+3)"
                        if self._is_friday_weekend_close()
                        else f"Daily close ({DAILY_CLOSE_HOUR}:00 UTC+3)"
                    )
                    manage, closed = self._force_flat_trade(
                        symbol, trade, label=label, last_price=last_price
                    )
                    results[symbol] = manage
                    dirty = True
                    if closed:
                        self.active_trades.pop(symbol, None)
                    continue
                if not self._is_split_trade(trade):
                    continue
                last_price = self.mt5_executor.get_current_price(symbol, trade.side)
                lifecycle = self._poll_idea_lifecycle(symbol, trade, last_price=last_price)
                dirty = dirty or bool(lifecycle.get("changed"))
                events = list(lifecycle.get("events") or [])

                be_events = self._move_idea_entries_to_breakeven(symbol, trade)
                if be_events:
                    dirty = True
                    events.extend(be_events)

                if lifecycle.get("final"):
                    manage = {
                        "signal": "EXIT_BROKER",
                        "side": trade.side,
                        "exit_price": float(last_price or trade.entry),
                        "info": "All split legs closed",
                        "tf": trade.tf,
                        "narrative": trade.narrative,
                        "events": events + [
                            {"type": "BROKER_CLOSE", "info": "All split TP legs closed in MT5"}
                        ],
                        "telegram_chat_id": getattr(trade, "telegram_chat_id", None),
                        "telegram_message_id": getattr(trade, "telegram_message_id", None),
                    }
                    self._register_broker_close(symbol, trade, manage)
                    results[symbol] = manage
                    self.active_trades.pop(symbol, None)
                    dirty = True
                    self._log_signal(symbol, manage)
                elif events:
                    results[symbol] = {
                        "signal": "HOLD",
                        "side": trade.side,
                        "entry_price": float(trade.entry),
                        "stop_price": float(trade.stop),
                        "tp_prices": [float(x) for x in (trade.tp_prices or [])],
                        "tp_hit": int(trade.tp_hit),
                        "tf": trade.tf,
                        "narrative": trade.narrative,
                        "events": events,
                    }

            if dirty:
                save_active_trades(self.active_trades, AI_DATA_DIR / "active_trades.json")
        return results

    def _force_flat_trade(
        self,
        symbol: str,
        trade: ActiveTrade,
        *,
        label: str,
        last_price: Optional[float],
    ) -> tuple[Dict[str, Any], bool]:
        """Broker-only scheduled close. Returns (signal, fully_confirmed_send)."""
        manage: Dict[str, Any] = {
            "signal": "EXIT_TIME",
            "side": trade.side,
            "exit_price": float(last_price or trade.entry),
            "info": label,
            "tf": trade.tf,
            "narrative": trade.narrative,
            "telegram_chat_id": getattr(trade, "telegram_chat_id", None),
            "telegram_message_id": getattr(trade, "telegram_message_id", None),
            "execution": {},
        }
        failed = False
        closed_position_ids: List[int] = []
        any_split_ids = False
        complete_id_set = True
        # Every entry of a position-adding idea is flattened; the idea counts as
        # closed only once each of them is confirmed closed by the broker.
        for entry in self._idea_entries(trade):
            split_ids = list(getattr(entry, "split_position_ids", []) or [])
            entry_closed_ids: List[int] = []
            if split_ids:
                any_split_ids = True
                remaining: List[int] = []
                closed_count = 0
                for pid in split_ids:
                    try:
                        if self.mt5_executor.close_trade(symbol, position_id=pid, volume=None):
                            closed_count += 1
                            entry_closed_ids.append(int(pid))
                        else:
                            remaining.append(pid)
                    except Exception as exc:
                        remaining.append(pid)
                        manage.setdefault("execution_error", str(exc))
                entry.split_position_ids = remaining
                manage["execution"]["mt5_closed_split"] = (
                    int(manage["execution"].get("mt5_closed_split") or 0) + closed_count
                )
                failed = failed or bool(remaining)
                known_split_legs = getattr(entry, "split_legs", {}) or {}
                if not (known_split_legs and len(entry_closed_ids) == len(known_split_legs)):
                    complete_id_set = False
            else:
                pos_id = getattr(entry, "mt5_position_id", None) or getattr(entry, "mt5_ticket", None)
                if not pos_id:
                    failed = True
                    manage["execution_error"] = "No broker position id to close"
                else:
                    try:
                        volume = float(getattr(entry, "volume_remaining", 0.0) or entry.volume or 0.0)
                        closed = self.mt5_executor.close_trade(
                            symbol, position_id=pos_id, volume=volume or None
                        )
                        manage["execution"]["mt5_closed"] = bool(closed)
                        failed = failed or not bool(closed)
                        if closed:
                            entry_closed_ids.append(int(pos_id))
                    except Exception as exc:
                        failed = True
                        manage["execution_error"] = str(exc)
            closed_position_ids.extend(entry_closed_ids)

        if failed:
            manage["signal"] = "HOLD"
            manage["info"] = f"{label}: broker close not confirmed; retrying"
            trade.last_price_ts = time.time()
        elif not any_split_ids or complete_id_set:
            self._attach_position_close_metrics(manage, closed_position_ids)
        self._log_signal(symbol, manage)
        return manage, not failed


    def _position_gone_confirmed(self, state_key: str, query_failed: bool) -> bool:
        """True only after N consecutive ticks where the position is absent AND the
        MT5 connection is verifiably alive. Any failed/untrusted query resets nothing
        and confirms nothing — better to hold a closed trade one extra tick than to
        forget a live position."""
        if query_failed:
            return False
        if not (self.mt5_executor and self.mt5_executor.connection_alive()):
            return False
        count = self._broker_missing_counts.get(state_key, 0) + 1
        self._broker_missing_counts[state_key] = count
        return count >= self._broker_missing_confirm

    @staticmethod
    def _trade_rr(trade: ActiveTrade) -> Optional[float]:
        try:
            tps = list(trade.tp_prices or [])
            risk = abs(float(trade.entry) - float(trade.stop))
            if not tps or risk <= 0:
                return None
            return abs(float(tps[-1]) - float(trade.entry)) / risk
        except Exception:
            return None

    @staticmethod
    def _closed_bars_view(data: Dict[str, Any]) -> Dict[str, Any]:
        """Strategy input with the still-forming last candle removed per timeframe.

        MT5 copy_rates returns the current forming bar as the last row while the
        market is open; triggers computed on it can appear mid-bar and vanish by
        the close (repaint). Dropping the last row makes signals deterministic
        per closed bar. Price/SLTP checks keep using the live tick, not this view.
        """
        out: Dict[str, Any] = {}
        for tf, df in (data or {}).items():
            if df is not None and getattr(df, "empty", True) is False and len(df) > 1:
                out[tf] = df.iloc[:-1]
            else:
                out[tf] = df
        return out

    def _strategy_view(
        self,
        data: Dict[str, Any],
        *,
        symbol: str,
    ) -> Dict[str, Any]:
        """Closed OHLC plus matching causal cluster and DOM M15 events."""

        view = self._closed_bars_view(data) if SIGNAL_ON_CLOSED_BARS else data
        df_15m = view.get("15M") if isinstance(view, dict) else None
        if df_15m is None or getattr(df_15m, "empty", True):
            return view

        decision_time = datetime.now(timezone.utc)
        enriched = dict(view)
        cluster_dataset = self._refresh_fxpro_cluster_dataset()
        if cluster_dataset is not None:
            cluster_event = cluster_dataset.event_asof_latest(
                symbol,
                df_15m.index[-1],
                decision_time,
            )
            if cluster_event is not None:
                enriched["FXPRO_CLUSTER_REJECTION_15M"] = cluster_event

        recorder = getattr(self, "fxpro_dom_recorder", None)
        if recorder is not None:
            quote_event = recorder.event_asof(
                symbol,
                df_15m.index[-1],
                decision_time,
            )
            if quote_event is not None:
                enriched["FXPRO_QUOTE_PRESSURE_15M"] = quote_event
        return enriched

    def _move_entry_to_breakeven(
        self, symbol: str, entry: ActiveTrade, *, reason: str
    ) -> bool:
        """Move one entry's SL to its own fill price. Returns True on success.

        Split mode: updates every remaining leg of that entry. Monitor mode:
        updates the single position. move_stop() itself clamps the level to the
        broker's minimum stop distance, so this never produces [Invalid stops].
        Each entry of a position-adding idea has its own fill price, so the level
        is always taken from the entry being moved — never from the idea's first.
        """
        if not self.mt5_executor:
            return False
        if getattr(entry, "moved_to_be", False):
            return False
        be_price = float(entry.entry)
        if be_price <= 0:
            return False
        try:
            split_ids = list(getattr(entry, "split_position_ids", []) or [])
            if split_ids:
                updated = self.mt5_executor.move_stop_all(symbol, position_ids=split_ids, new_stop=be_price)
                # A partial success is not completion: leave moved_to_be=False
                # so the remaining tickets are retried on the next fast poll.
                ok = updated == len(split_ids)
            else:
                pos_id = getattr(entry, "mt5_position_id", None) or getattr(entry, "mt5_ticket", None)
                if not pos_id:
                    return False
                ok = self.mt5_executor.move_stop(symbol, position_id=pos_id, new_stop=be_price)
        except Exception as exc:
            print(f"[Core] {symbol} BE move failed: {exc}")
            return False
        if ok:
            entry.moved_to_be = True
            if entry.side == "LONG":
                entry.stop = max(float(entry.stop or 0.0), be_price)
            else:
                current = float(entry.stop or 0.0)
                entry.stop = min(current, be_price) if current > 0 else be_price
            print(f"[Core] {symbol}{_entry_label(entry)} SL moved to break-even {be_price:.5f} ({reason})")
        return ok

    def _move_to_breakeven(self, symbol: str, trade: ActiveTrade) -> bool:
        """TP1-driven break-even for one entry (config-gated)."""
        if not MOVE_BE_AFTER_TP1:
            return False
        return self._move_entry_to_breakeven(symbol, trade, reason="TP1 hit")

    def _attach_position_close_metrics(
        self,
        manage: Dict[str, Any],
        position_ids: List[int],
    ) -> bool:
        """Attach complete broker P&L when every requested history is visible."""
        if not self.mt5_executor or not hasattr(self.mt5_executor, "get_position_close_info"):
            return False
        ids = list(dict.fromkeys(int(pid) for pid in position_ids if pid))
        if not ids:
            return False
        infos: List[Dict[str, Any]] = []
        for pid in ids:
            try:
                info = self.mt5_executor.get_position_close_info(pid)
            except Exception:
                return False
            if not info:
                # MT5 can publish the exit deal a little after the position
                # disappears. Unknown is safer than a fabricated zero P&L.
                return False
            infos.append(info)

        total_net = sum(
            float(info.get("net"))
            if info.get("net") is not None
            else float(info.get("profit") or 0.0)
            + float(info.get("commission") or 0.0)
            + float(info.get("swap") or 0.0)
            + float(info.get("fee") or 0.0)
            for info in infos
        )
        total_volume = sum(float(info.get("volume") or 0.0) for info in infos)
        if total_volume > 0:
            manage["exit_price"] = sum(
                float(info.get("price") or 0.0) * float(info.get("volume") or 0.0)
                for info in infos
            ) / total_volume
        manage["realized_net"] = total_net
        manage["pnl_complete"] = True
        manage["outcome"] = "TP" if total_net > 0.005 else "SL" if total_net < -0.005 else "BE"
        return True

    def _register_broker_close(self, symbol: str, trade: ActiveTrade, manage: Dict[str, Any]) -> None:
        """Infer TP/SL outcome of a broker-side close from MT5 deal history and
        feed it to the AI stats store. Split mode ends every trade via EXIT_BROKER,
        so without this neither the journal nor the AI filter ever sees outcomes."""
        if not self.mt5_executor:
            return
        # A position-adding idea is journaled as one result: every entry's legs
        # contribute to the same realized P&L and outcome.
        entries = self._idea_entries(trade)
        split_legs: Dict[int, Dict[str, Any]] = {}
        for entry in entries:
            split_legs.update(getattr(entry, "split_legs", {}) or {})
        ids = [int(pid) for pid in split_legs]
        closed_meta = [
            meta for meta in split_legs.values()
            if str(meta.get("status") or "") == "closed"
        ]
        total_net = sum(
            float(meta.get("close_net"))
            if meta.get("close_net") is not None
            else float(meta.get("close_profit") or 0.0)
            + float(meta.get("close_commission") or 0.0)
            + float(meta.get("close_swap") or 0.0)
            + float(meta.get("close_fee") or 0.0)
            for meta in closed_meta
        )
        total_close_volume = sum(float(meta.get("close_volume") or 0.0) for meta in closed_meta)
        if total_close_volume > 0:
            manage["exit_price"] = sum(
                float(meta.get("close_price") or 0.0)
                * float(meta.get("close_volume") or 0.0)
                for meta in closed_meta
            ) / total_close_volume
        outcome: Optional[str] = None
        planned_leg_count = 0
        for entry in entries:
            entry_legs = sum(
                1 for volume in (getattr(entry, "volume_per_tp", []) or [])
                if float(volume or 0.0) > 0.0
            )
            planned_leg_count += entry_legs or len(entry.tp_prices or [])
        complete_mapping = bool(split_legs) and len(split_legs) >= planned_leg_count
        pnl_complete = complete_mapping and len(closed_meta) == len(split_legs)
        manage["pnl_complete"] = pnl_complete
        if pnl_complete:
            manage["realized_net"] = total_net
        if pnl_complete:
            if total_net > 1e-8:
                outcome = "TP"
            elif total_net < -1e-8:
                outcome = "SL"
            else:
                outcome = "BE"
        elif any(int(getattr(entry, "tp_hit", 0) or 0) > 0 for entry in entries):
            # Legacy state may only know the still-open final leg. Confirmed
            # earlier TPs must not be reclassified as a loss when that leg exits
            # at the break-even stop.
            outcome = "TP"
        if not ids:
            ids = [
                pid
                for pid in (
                    getattr(entry, "mt5_position_id", None) or getattr(entry, "mt5_ticket", None)
                    for entry in entries
                )
                if pid
            ]
        if not split_legs and self._attach_position_close_metrics(manage, ids):
            outcome = str(manage.get("outcome") or "") or None
        if outcome is None:
            for pid in ids:
                reason = self.mt5_executor.get_position_close_reason(pid)
                if reason == "SL":
                    outcome = "SL"
                    break
                if reason == "TP":
                    outcome = outcome or "TP"
        if outcome:
            manage["outcome"] = outcome
            if outcome == "SL":
                until = time.time() + self._post_sl_cooldown_sec
                self._entry_cooldowns[symbol] = max(self._entry_cooldowns.get(symbol, 0.0), until)
                print(
                    f"[Core] {symbol} stopped out — entry cooldown "
                    f"{self._post_sl_cooldown_sec / 60:.0f}m (until {datetime.fromtimestamp(until).strftime('%H:%M')})"
                )
            if outcome in {"TP", "SL"}:
                try:
                    self.ai_store.update_on_close(
                        symbol,
                        outcome,
                        rr_numeric=self._trade_rr(trade),
                        trigger_kind=(
                            primary_idea_trigger_kind(trade) or None
                        ),
                    )
                except Exception:
                    traceback.print_exc()

    @staticmethod
    def _parse_time_str(value: str) -> dt_time:
        hour, minute = [int(x) for x in value.split(":", 1)]
        return dt_time(hour=hour, minute=minute)

    @staticmethod
    def _time_in_window(now: dt_time, start: dt_time, end: dt_time) -> bool:
        if start <= end:
            return start <= now < end
        return now >= start or now < end

    def _session_allowance(self) -> tuple[bool, str]:
        if not self.allowed_session_windows:
            return True, "ALL"
        now = datetime.now(self.session_tz).time()
        for name, start, end in self.allowed_session_windows:
            if self._time_in_window(now, start, end):
                return True, name
        return False, "OFF"

    @staticmethod
    def _is_friday_weekend_close() -> bool:
        """True on Friday at or after FRIDAY_CLOSE_HOUR UTC+3 (Europe/Moscow, no DST)."""
        now = datetime.now(ZoneInfo("Europe/Moscow"))
        return now.weekday() == 4 and now.hour >= FRIDAY_CLOSE_HOUR

    @staticmethod
    def _is_daily_flat_close() -> bool:
        """True at or after DAILY_CLOSE_HOUR UTC+3 (Europe/Moscow, no DST) —
        all positions must be flat by this time every day."""
        if not DAILY_FLAT_ENABLED:
            return False
        now = datetime.now(ZoneInfo("Europe/Moscow"))
        return now.hour >= DAILY_CLOSE_HOUR

    @staticmethod
    def _is_daily_entry_cutoff() -> bool:
        """Block fresh risk shortly before the optional daily flat close."""
        if not DAILY_FLAT_ENABLED or DAILY_CLOSE_BUFFER_MIN <= 0:
            return False
        now = datetime.now(ZoneInfo("Europe/Moscow"))
        current_minute = now.hour * 60 + now.minute
        close_minute = DAILY_CLOSE_HOUR * 60
        cutoff_minute = max(0, close_minute - DAILY_CLOSE_BUFFER_MIN)
        return cutoff_minute <= current_minute < close_minute

    def _get_symbols(self) -> list[str]:
        return self.scanner.scan()

    def _build_tf_data(self, symbol_key: str) -> dict:
        def _get(tf: str):
            df = self.data_cache.request(symbol_key, tf, limit=self.N_BARS)
            if df is None or df.empty:
                return self.feed.get_klines(symbol_key, tf, limit=self.N_BARS)
            return df

        return {
            "D": _get("1d"),
            "4H": _get("4h"),
            "1H": _get("1h"),
            "15M": _get("15m"),
            "5M": _get("5m"),
            "1M": _get("1m"),
        }

    def _roll_daily_counters(self) -> None:
        today = datetime.now(timezone.utc).date()
        if self._counters_day == today:
            return
        self._counters_day = today
        self._entries_today = {}
        self._trigger_signatures = {}
        self._day_baseline_balance = None
        if self._daily_loss_stop:
            print("[Core] new UTC day — daily loss stop reset")
        self._daily_loss_stop = False

    def _check_daily_loss_stop(self) -> None:
        """Bot-wide brake: once equity is DAILY_MAX_LOSS_PCT below the day's
        starting balance, block new entries until the next UTC day. The baseline
        is captured on the first tick of the day (or after a restart)."""
        if self._daily_loss_stop or not self.mt5_executor:
            return
        try:
            import MetaTrader5 as _mt5
            account = _mt5.account_info()
        except Exception:
            return
        if account is None:
            return
        if self._day_baseline_balance is None:
            self._day_baseline_balance = float(account.balance)
            return
        limit = self._day_baseline_balance * (1.0 - float(DAILY_MAX_LOSS_PCT))
        if float(account.equity) <= limit:
            self._daily_loss_stop = True
            print(
                f"[Core] DAILY LOSS STOP: equity {account.equity:.2f} <= {limit:.2f} "
                f"({DAILY_MAX_LOSS_PCT:.0%} of day baseline {self._day_baseline_balance:.2f}) — "
                f"no new entries until next UTC day"
            )

    def _update_global_context(self) -> None:
        allowed, session_name = self._session_allowance()
        self.global_context["session"] = session_name
        self.global_context["session_allowed"] = allowed
        self.global_context["friday_close"] = self._is_friday_weekend_close()
        self.global_context["daily_close"] = self._is_daily_flat_close()
        self.global_context["daily_entry_cutoff"] = self._is_daily_entry_cutoff()
        self._roll_daily_counters()
        self._check_daily_loss_stop()
        self.global_context["daily_loss_stop"] = self._daily_loss_stop

    def _apply_session_filter(self, symbol: str, sig: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(sig, dict):
            return sig
        if sig.get("signal") != "ENTER":
            return sig
        if self.global_context.get("session_allowed", True):
            return sig
        new_sig = dict(sig)
        new_sig["signal"] = "WAIT_SESSION"
        reason = f"Blocked by session ({self.global_context.get('session')})"
        narrative = str(new_sig.get("narrative", ""))
        if reason not in narrative:
            new_sig["narrative"] = (narrative + " | " + reason).strip(" |")
        new_sig["session_blocked"] = self.global_context.get("session")
        return new_sig

    def _apply_global_filters(self, symbol: str, sig: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(sig, dict) and sig.get("signal") == "ENTER":
            if self.global_context.get("friday_close"):
                new_sig = dict(sig)
                new_sig["signal"] = "WAIT_SESSION"
                new_sig["info"] = f"Заблокировано: закрытие перед выходными (пятница {FRIDAY_CLOSE_HOUR}:00 UTC+3)"
                return new_sig
            if self.global_context.get("daily_close"):
                new_sig = dict(sig)
                new_sig["signal"] = "WAIT_SESSION"
                new_sig["info"] = f"Заблокировано: дневное закрытие ({DAILY_CLOSE_HOUR}:00 UTC+3)"
                return new_sig
            if self.global_context.get("daily_entry_cutoff"):
                new_sig = dict(sig)
                new_sig["signal"] = "WAIT_SESSION"
                new_sig["info"] = (
                    f"Заблокировано: за {DAILY_CLOSE_BUFFER_MIN} мин до дневного закрытия"
                )
                return new_sig
            if self.global_context.get("daily_loss_stop"):
                new_sig = dict(sig)
                new_sig["signal"] = "WAIT_RISK"
                new_sig["info"] = f"Заблокировано: дневной лимит убытка {DAILY_MAX_LOSS_PCT:.0%} достигнут"
                return new_sig
        return sig

    def _get_vol_context(self, symbol: str, data: Dict[str, Any]) -> Optional[VolContext]:
        """Cached 7-signal vol-surface context from the symbol's daily candles.
        Returns None when daily data is missing/short — callers must fail open."""
        cached = self._vol_contexts.get(symbol)
        now = time.time()
        if cached is not None and now - cached[0] < VOL_REGIME_REFRESH_MIN * 60.0:
            return cached[1]

        ctx: Optional[VolContext] = None
        try:
            df_d = data.get("D")
            df_15m = data.get("15M")
            spot = None
            if df_15m is not None and not df_15m.empty:
                spot = float(df_15m["close"].iloc[-1])
            if df_d is not None and not df_d.empty:
                ctx = build_vol_context(
                    symbol, df_d["close"].to_list(), spot=spot, df_15m=df_15m
                )
        except Exception as exc:
            print(f"[VolRegime] {symbol} context failed: {exc}")
        if ctx is not None:
            print(
                f"[VolRegime] {symbol} R(t)={ctx.r_t:.1f} [{ctx.regime}] "
                f"base={ctx.base_r_t:.1f} tape={ctx.tape_stress:.1f} "
                f"IV={ctx.atm_iv:.1%} RV={ctx.rv:.1%} EM1D={ctx.em_1d:.5g}"
            )
        self._vol_contexts[symbol] = (now, ctx)
        return ctx

    def _apply_vol_regime_filter(
        self, symbol: str, sig: Dict[str, Any], data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """PANIC-score entry brake + expected-move TP sanity check (IV Surface port)."""
        if not VOL_REGIME_FILTER_ENABLED or symbol.upper() not in VOL_REGIME_SYMBOLS:
            return sig
        if not isinstance(sig, dict) or sig.get("signal") != "ENTER":
            return sig
        ctx = self._get_vol_context(symbol, data)
        if ctx is None:
            return sig  # no daily data — the filter must not block trading

        # Attach context to the signal for journaling / AI stats even when passing
        sig = dict(sig)
        sig["vol_R"] = round(ctx.r_t, 1)
        sig["vol_regime"] = ctx.regime
        sig["vol_em_1d"] = round(ctx.em_1d, 6)

        tp_prices = sig.get("tp_prices") or (
            [sig["tp_price"]] if sig.get("tp_price") is not None else []
        )
        try:
            tp_prices = [float(x) for x in tp_prices if x is not None]
        except (TypeError, ValueError):
            tp_prices = []
        # Same definition as the offline ``vol_tp1_em_ratio`` so a live signal
        # and a fitted row describe the feature identically. Journalling only.
        ratio = tp1_em_ratio(sig.get("entry_price"), tp_prices, ctx.em_1d)
        if ratio is not None:
            sig["vol_tp1_em_ratio"] = ratio

        ok, reason = entry_gate(
            ctx,
            sig.get("entry_price"),
            tp_prices,
            max_r=VOL_REGIME_MAX_R,
            em_tp_ratio=EM_TP_MAX_RATIO,
        )
        if ok:
            return sig
        sig["signal"] = "SKIP_VOL_REGIME" if ctx.r_t >= VOL_REGIME_MAX_R else "SKIP_EM_TP"
        sig["info"] = reason
        print(f"[VolRegime] {symbol} entry blocked: {reason}")
        return sig

    def _apply_execution_quality_filter(
        self,
        symbol: str,
        sig: Dict[str, Any],
    ) -> Dict[str, Any]:
        if (
            not EXECUTION_QUALITY_FILTER_ENABLED
            or not isinstance(sig, dict)
            or sig.get("signal") != "ENTER"
        ):
            return sig
        new_sig = dict(sig)
        profile = getattr(self, "execution_quality_profile", None)
        executor = getattr(self, "mt5_executor", None)
        if profile is None or executor is None:
            new_sig["signal"] = "WAIT_EXECUTION_QUALITY_DATA"
            new_sig["info"] = (
                "Execution-quality filter is enabled but its profile or "
                "causal broker quote source is unavailable"
            )
            return new_sig
        source_symbol = self.universe.get(symbol, symbol)
        try:
            quote = executor.get_current_quote(source_symbol)
            if quote is None:
                raise ValueError("broker bid/ask quote unavailable")
            assessment = assess_execution_quality(
                profile,
                symbol=symbol,
                signal=new_sig,
                bid=quote.get("bid"),
                ask=quote.get("ask"),
                observed_at_utc=datetime.now(timezone.utc),
                news_calendar=getattr(self, "news_calendar", None),
            )
            assessment["quote_time_msc"] = quote.get("time_msc")
        except Exception as exc:
            new_sig["signal"] = "WAIT_EXECUTION_QUALITY_DATA"
            new_sig["info"] = (
                f"Execution-quality evidence unavailable: {exc}"
            )
            return new_sig
        new_sig["execution_quality"] = assessment
        if assessment.get("allowed"):
            return new_sig
        reason = str(assessment.get("reason") or "DATA_UNAVAILABLE")
        new_sig["signal"] = {
            "SPREAD": "SKIP_SPREAD",
            "ROLLOVER": "WAIT_ROLLOVER",
            "NEWS": "WAIT_NEWS",
            "NEWS_DATA_UNAVAILABLE": "WAIT_NEWS_DATA",
        }.get(reason, "WAIT_EXECUTION_QUALITY_DATA")
        new_sig["info"] = f"Execution-quality block: {reason}"
        print(f"[Execution Quality] {symbol} entry blocked: {reason}")
        return new_sig

    def _attach_shadow_score(
        self, symbol: str, sig: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Annotate an ENTER signal with the frozen shadow model's expected R.

        Diagnostic only. Only ``shadow_*`` keys are ever written, so this can
        not reach direction, entry, stop, targets, disposition, or risk even if
        the model file changes. Any failure is swallowed: a broken annotation
        must never stop a trade.
        """
        scorer = getattr(self, "shadow_scorer", None)
        if scorer is None:
            return sig
        if not isinstance(sig, dict) or sig.get("signal") != "ENTER":
            return sig
        try:
            annotation = scorer.score(sig, symbol=symbol)
        except Exception as exc:
            print(f"[Shadow] {symbol} scoring failed: {exc}")
            return sig
        safe = {
            key: value
            for key, value in (annotation or {}).items()
            if str(key).startswith("shadow_")
        }
        if not safe:
            return sig
        annotated = dict(sig)
        annotated.update(safe)
        return annotated

    def _try_position_add(
        self,
        symbol: str,
        trade: ActiveTrade,
        data: Dict[str, Any],
        last_price: float,
    ) -> Optional[Dict[str, Any]]:
        """Stage one more entry into the open idea when it is confirmed again.

        Returns the executed add-on signal, or None when nothing was opened —
        the caller keeps reporting the idea's HOLD either way. Every gate that
        applies to a first entry applies here too; on top of them the pyramid
        manager enforces the idea's aggregate risk cap and moves older entries
        to break-even before the new risk is added.
        """
        if not (POSITION_ADDING_ENABLED and self.mt5_executor):
            return None
        if PARTIAL_TP_MODE == "monitor":
            return None
        entries = self._idea_entries(trade)
        if len(entries) >= IDEA_MAX_ENTRIES:
            return None
        # Add-ons rely on broker-side split legs for their own SL/TP management.
        if not all(self._is_split_trade(entry) for entry in entries):
            return None
        if any(
            int(getattr(entry, "tp_hit", 0) or 0) >= len(entry.tp_prices or [])
            and (entry.tp_prices or [])
            for entry in entries
        ):
            return None

        strategy_data = self._strategy_view(data, symbol=symbol)
        raw_sig = self.strategy.generate_signal(strategy_data, symbol=symbol)
        if raw_sig.get("signal") != "ENTER":
            return None
        sig = self.ai.on_signal(symbol, raw_sig, data, self.active_trades)
        if raw_sig.get("tp_prices") and not sig.get("tp_prices"):
            sig["tp_prices"] = raw_sig["tp_prices"]
        sig = self._apply_global_filters(symbol, sig)
        sig = self._apply_session_filter(symbol, sig)
        sig = self._apply_vol_regime_filter(symbol, sig, strategy_data)
        sig = self._attach_shadow_score(symbol, sig)
        if sig.get("signal") != "ENTER" or not sig.get("side"):
            return None
        if (
            str(sig.get("entry_order_type") or "").upper()
            == "LIMIT_RETEST"
        ):
            # Add-ons currently have no durable pending-plan lifecycle.  An
            # exact structural retest must never degrade into a market add.
            print(
                f"[Pyramid] {symbol} add-on blocked: "
                "LIMIT_RETEST requires persistent pending execution"
            )
            return None
        # An add-on must open as broker-managed split legs like the entries it
        # joins; a single-TP signal would land in monitor mode inside an
        # otherwise split idea.
        if len([x for x in (sig.get("tp_prices") or []) if x is not None]) < 2:
            return None

        side = sig.get("side")
        trig_sig = self._trigger_signature(sig)
        if self._apply_entry_guards(symbol, sig, side=side, trig_sig=trig_sig):
            print(f"[Pyramid] {symbol} add-on blocked: {sig.get('signal')} {sig.get('info', '')}")
            return None

        try:
            decision = self.pyramid.evaluate(
                trade,
                sig,
                executor=self.mt5_executor,
                last_price=last_price,
                trigger_signature=trig_sig,
            )
        except PyramidRiskError as exc:
            print(f"[Pyramid] {symbol} add-on skipped: {exc}")
            return None
        if not decision.allowed:
            print(f"[Pyramid] {symbol} add-on rejected: {decision.reason}")
            return None

        # Free the required room BEFORE new risk enters the market. A failed
        # break-even means the idea would exceed its cap, so nothing is opened.
        be_moved: List[ActiveTrade] = []
        for entry in decision.entries_to_breakeven:
            if not self._move_entry_to_breakeven(
                symbol, entry, reason=f"освобождение риска под вход {decision.entry_index}"
            ):
                print(
                    f"[Pyramid] {symbol} add-on cancelled: не удалось перевести "
                    f"вход #{getattr(entry, 'entry_index', 1)} в безубыток"
                )
                return None
            entry.be_for_idea_risk = True
            be_moved.append(entry)

        try:
            addon = self._execute_entry_signal(symbol, sig)
        except RiskCapacityError as exc:
            print(f"[Pyramid] {symbol} add-on waiting for risk-compatible entry: {exc}")
            return None
        except Exception as exc:
            err_str = str(exc)
            is_stale = "Stale signal rejected" in err_str
            cooldown_sec = self._stale_cooldown_sec if is_stale else self._entry_cooldown_sec
            self._entry_cooldowns[symbol] = time.time() + cooldown_sec
            print(f"[Pyramid] {symbol} add-on execution failed: {exc}. Cooldown {cooldown_sec:.0f}s")
            return None

        addon.idea_id = trade.idea_id
        addon.entry_index = decision.entry_index
        trade.addons.append(addon)
        trade.idea_trigger_signatures.append(trig_sig)
        self._entries_today[symbol] = self._entries_today.get(symbol, 0) + 1
        self._trigger_signatures.setdefault(symbol, set()).add(trig_sig)

        sig.setdefault("setup_id", uuid.uuid4().hex)
        sig["idea_id"] = trade.idea_id
        sig["entry_index"] = decision.entry_index
        sig["idea_entries"] = len(trade.addons) + 1
        sig["idea_max_entries"] = IDEA_MAX_ENTRIES
        sig["idea_risk_pct"] = round(decision.projected_risk_pct, 5)
        sig["position_add"] = True
        if be_moved:
            sig["idea_be_entries"] = [
                int(getattr(entry, "entry_index", 1) or 1) for entry in be_moved
            ]
        print(
            f"[Pyramid] {symbol} entry {decision.entry_index}/{IDEA_MAX_ENTRIES} opened, "
            f"idea risk {decision.idea_risk_pct:.2%} → {decision.projected_risk_pct:.2%}"
            + (f", BE: {sig['idea_be_entries']}" if be_moved else "")
        )
        return sig

    def _apply_entry_guards(
        self, symbol: str, sig: Dict[str, Any], *, side: str, trig_sig: str
    ) -> bool:
        """Per-symbol entry brakes. Rewrites ``sig`` in place and returns True
        when the entry must not be opened.

        Shared by first entries and position-adding add-ons: an add-on is new
        risk in the market and passes exactly the same frequency, hedging and
        correlation checks as any other entry.
        """
        if not getattr(self, "_pending_state_safe", True):
            sig["signal"] = "WAIT_PENDING_STATE"
            sig["info"] = (
                "Persistent LIMIT ownership is ambiguous; "
                "new risk is fail-closed"
            )
            return True

        if self._is_exact_h1_rb_signal(sig):
            try:
                consumed = self._rb_event_is_consumed(
                    symbol,
                    sig,
                    trigger_signature=trig_sig,
                )
            except RBConsumedEventsValidationError as exc:
                sig["signal"] = "WAIT_RB_EVENT_STATE"
                sig["info"] = (
                    "Exact H1-RB one-shot state is unavailable; "
                    f"entry is fail-closed ({exc})"
                )
                return True
            if consumed:
                sig["signal"] = "SKIP_CONSUMED_RB_EVENT"
                sig["info"] = (
                    "Exact H1-RB first-touch event was already consumed "
                    "and cannot be re-armed"
                )
                return True

        # Skip if this symbol is in cooldown (failed execution or stop-out)
        cooldown_until = self._entry_cooldowns.get(symbol, 0.0)
        if time.time() < cooldown_until:
            sig["signal"] = "WAIT_COOLDOWN"
            sig["info"] = f"Entry cooldown ({cooldown_until - time.time():.0f}s left)"
            return True

        # Daily frequency brakes: setup cap per symbol + one-shot triggers
        if self._entries_today.get(symbol, 0) >= MAX_SETUPS_PER_SYMBOL_PER_DAY:
            sig["signal"] = "SKIP_DAILY_LIMIT"
            sig["info"] = f"Достигнут лимит {MAX_SETUPS_PER_SYMBOL_PER_DAY} сетапов/день"
            return True

        if trig_sig in self._trigger_signatures.get(symbol, set()):
            sig["signal"] = "SKIP_DUP_TRIGGER"
            sig["info"] = f"Зона/бар триггера уже отторгована сегодня ({trig_sig})"
            return True

        # Anti-hedging guard: block entry if opposite MT5 position is open
        if self.mt5_executor:
            import MetaTrader5 as _mt5
            _open_pos = _mt5.positions_get(symbol=symbol)
            if _open_pos is None:
                sig["signal"] = "WAIT_RISK"
                sig["info"] = (
                    "MT5 positions_get unavailable; anti-hedge "
                    "check is fail-closed"
                )
                return True
            _expected_type = _mt5.POSITION_TYPE_BUY if side == "LONG" else _mt5.POSITION_TYPE_SELL
            _opposite = [p for p in _open_pos if p.magic == self.mt5_executor.settings.magic and p.type != _expected_type]
            if _opposite:
                sig["signal"] = "SKIP_HEDGE"
                sig["info"] = f"Opposite MT5 position still open ({len(_opposite)} legs)"
                print(f"[Core] {symbol} anti-hedge block: {len(_opposite)} opposite leg(s) still open in MT5")
                return True

        # Correlation guard: same-direction trades on correlated symbols
        # (e.g. EURUSD + GBPUSD) double the risk on a single idea.
        _corr_partner = None
        for _group in CORRELATED_GROUPS:
            if symbol.upper() not in _group:
                continue
            for _other in _group:
                if _other == symbol.upper():
                    continue
                _other_trade = self.active_trades.get(_other)
                if _other_trade is not None and _other_trade.side == side:
                    _corr_partner = _other
                    break
                _other_pending = getattr(
                    self, "pending_limits", {}
                ).get(_other)
                if (
                    _other_pending is not None
                    and not _other_pending.is_terminal
                    and _other_pending.side == side
                ):
                    _corr_partner = _other
                    break
            if _corr_partner:
                break
        if _corr_partner:
            sig["signal"] = "SKIP_CORRELATED"
            sig["info"] = f"Коррелированный {_corr_partner} уже открыт в ту же сторону ({side})"
            print(f"[Core] {symbol} correlation block: {_corr_partner} already open {side}")
            return True

        return False

    @staticmethod
    def _trigger_signature(sig: Dict[str, Any]) -> str:
        """Stable identity of a setup: side + trigger kind + its zone (or stop
        level for zoneless triggers). The same zone/stop is traded at most once
        per day — a still-valid M15 trigger cannot re-enter after a stop-out."""
        trig_kind = str(sig.get("trigger_kind") or "").strip().lower()
        if not trig_kind:
            trig_kind = (
                str(sig.get("trigger_reason") or "")
                .split("|")[0]
                .strip()
                .split(" ")[0]
                .lower()
            )
        event_id = str(sig.get("trigger_event_id") or "").strip()
        zl, zh = sig.get("zone_low"), sig.get("zone_high")
        if event_id:
            anchor = f"e{event_id}"
        elif zl is not None and zh is not None:
            anchor = f"z{float(zl):.5f}-{float(zh):.5f}"
        else:
            anchor = f"s{float(sig.get('stop_price') or 0.0):.5f}"
        return f"{sig.get('side')}|{trig_kind}|{anchor}"

    def _log_signal(self, symbol: str, sig: Dict[str, Any]) -> None:
        if not isinstance(sig, dict):
            return
        signal_type = sig.get("signal")
        # Always log anything that isn't a silent HOLD
        important = signal_type not in {"HOLD", None}
        if not (self.log_tick or important):
            return
        fvg = sig.get("vc")
        fvg_part = f" | fvg={fvg}" if fvg else ""
        info = (
            f"[Ticker] {symbol} {signal_type}"
            f" | side={sig.get('side')}"
            f" | session={self.global_context.get('session')}"
            f" | trigger={sig.get('trigger_reason')}"
            f"{fvg_part}"
            f" | narrative={sig.get('narrative')}"
        )
        if signal_type == "EXECUTION_ERROR" and sig.get("execution_error"):
            info += f" | error={sig.get('execution_error')}"
        if signal_type == "WAIT_SESSION":
            info += f" | blocked_by={sig.get('session_blocked')}"
        if signal_type == "SKIP_BUDGET":
            info += f" | reason=time_budget_exceeded"
        if signal_type in ("SKIP_VOL_REGIME", "SKIP_EM_TP"):
            info += f" | reason={sig.get('info')} | vol_R={sig.get('vol_R')}"
        if signal_type == "NO_DATA":
            info += f" | reason=no_data_from_feed"
        if signal_type == "EXIT_BROKER":
            info += f" | reason=position_closed_externally_in_MT5"
        try:
            print(info)
        except UnicodeEncodeError:
            print(info.encode("utf-8", errors="replace").decode("ascii", errors="replace"))

    def _check_active_trade(self, symbol: str, last_price: float, trade: ActiveTrade) -> dict:
        side = trade.side
        tps = trade.tp_prices or []
        n_tps = len(tps)

        stop = float(trade.stop or 0.0)
        # stop == 0 means "no SL known" (e.g. a position hydrated from MT5 without SL).
        # A zero stop must never trigger an exit — for SHORT `price >= 0` is always true.
        if stop > 0:
            if side == "LONG":
                if last_price <= stop:
                    return {"signal": "EXIT_SL", "side": side, "exit_price": last_price, "info": "Стоп-лосс"}
            else:
                if last_price >= stop:
                    return {"signal": "EXIT_SL", "side": side, "exit_price": last_price, "info": "Стоп-лосс"}

        # In split mode the broker owns every leg's TP and tp_hit is advanced from
        # deal history when a leg disappears (see the split watcher in get_signals).
        # Price polling must NOT advance tp_hit here — the two sources would double
        # count the same leg (poll sees price beyond TP1, then the closed leg is
        # detected next tick).
        _is_split = self._is_split_trade(trade)

        # Final TP was already reached earlier but the close failed (trade kept for
        # retry) → re-emit EXIT_TP so the close is retried instead of holding forever.
        if n_tps > 0 and trade.tp_hit >= n_tps and not _is_split:
            return {
                "signal": "EXIT_TP",
                "side": side,
                "exit_price": float(last_price),
                "info": f"TP{n_tps} (final, retry)",
            }

        events: List[Dict[str, Any]] = []
        next_idx = trade.tp_hit + 1

        def _tp_hit_condition(idx: int) -> bool:
            tp = float(tps[idx - 1])
            return (last_price >= tp) if side == "LONG" else (last_price <= tp)

        while not _is_split and next_idx <= n_tps and _tp_hit_condition(next_idx):
            tp_price = float(tps[next_idx - 1])
            events.append({"type": "TP", "tp_index": next_idx, "tp_price": tp_price, "hit_price": float(last_price)})
            trade.tp_hit = next_idx
            next_idx += 1

        if events:
            if trade.tp_hit >= n_tps and n_tps > 0:
                return {
                    "signal": "EXIT_TP",
                    "side": side,
                    "exit_price": float(last_price),
                    "info": f"TP{n_tps} (final)",
                    "events": events,
                }

            return {
                "signal": "HOLD",
                "side": side,
                "entry_price": float(trade.entry),
                "stop_price": float(trade.stop),
                "tp_prices": [float(x) for x in tps],
                "tp_hit": int(trade.tp_hit),
                "events": events,
                "tf": trade.tf,
                "narrative": trade.narrative,
            }

        return {
            "signal": "HOLD",
            "side": side,
            "entry_price": float(trade.entry),
            "stop_price": float(trade.stop),
            "tp_prices": [float(x) for x in tps],
            "tp_hit": int(trade.tp_hit),
            "tf": trade.tf,
            "narrative": trade.narrative,
        }

    def _execute_entry_signal(self, symbol: str, sig: Dict[str, Any]) -> ActiveTrade:
        """Build an ActiveTrade from a validated ENTER signal and execute it in MT5.

        The caller has already applied every entry guard (side, cooldowns, daily
        limits, hedge/correlation checks). Mutates ``sig`` in place with the
        execution payload, the actual fill price and the surviving tp_prices.
        Raises RiskCapacityError while the entry geometry cannot fit the risk
        budget, or any other exception on execution failure — cooldown/retry
        policy and trade registration stay with the caller.
        """
        side = sig.get("side")
        tp_prices = sig.get("tp_prices") or [sig.get("tp_price")]
        tp_prices = [float(x) for x in tp_prices if x is not None]

        new_trade = ActiveTrade(
            side=side,
            entry=float(sig.get("entry_price", 0.0)),
            stop=float(sig.get("stop_price", 0.0)),
            tp_prices=tp_prices,
            tf=str(sig.get("tf", "")),
            narrative=str(sig.get("narrative", "")),
            symbol=symbol,
            ts_open=time.time(),
            last_price_ts=time.time(),
        )
        # Keep the strategy's intended entry before the broker fill
        # replaces sig["entry_price"], so execution drift is auditable.
        sig.setdefault("planned_entry_price", float(new_trade.entry))

        if self.mt5_executor:
            import MetaTrader5 as _mt5
            # Auto-select mode:
            #   split  → 2+ TPs and PARTIAL_TP_MODE != "monitor"
            #   monitor → 1 TP, or forced via PARTIAL_TP_MODE=monitor
            use_split = (
                len(tp_prices) > 1
                and PARTIAL_TP_MODE != "monitor"
            )
            if use_split:
                # MK-style: calculate total volume once, then open N legs
                tick = _mt5.symbol_info_tick(symbol)
                if tick is None:
                    raise RuntimeError(f"No tick for {symbol}")
                actual_entry = float(tick.ask if new_trade.side == "LONG" else tick.bid)
                total_vol = self.mt5_executor._calc_volume(
                    symbol,
                    actual_entry,
                    new_trade.stop,
                    side=new_trade.side,
                )
                # Sort TPs nearest-first: leg-0 (largest volume at 4+ TPs)
                # targets the nearest TP, not the furthest.
                tp_prices = (
                    sorted(tp_prices)
                    if new_trade.side == "LONG"
                    else sorted(tp_prices, reverse=True)
                )
                new_trade.tp_prices = tp_prices
                _info = _mt5.symbol_info(symbol)
                _step = float(getattr(_info, "volume_step", 0.0) or 0.0) if _info else 0.0
                vols = _compute_tp_volumes(total_vol, len(tp_prices), step=_step or 0.01)
                legs = self.mt5_executor.execute_split_entry(
                    symbol,
                    side=new_trade.side,
                    entry_price=new_trade.entry,
                    stop_price=new_trade.stop,
                    tp_prices=tp_prices,
                    volumes_per_tp=vols,
                    comment=new_trade.narrative[:20] if new_trade.narrative else None,
                    entry_min=sig.get("entry_min"),
                    entry_max=sig.get("entry_max"),
                )
                if not legs:
                    raise RuntimeError(
                        "Split entry opened no legs (all volumes below broker minimum)"
                    )
                new_trade.volume = round(sum(l["volume"] for l in legs), 2)
                new_trade.volume_remaining = new_trade.volume
                new_trade.volume_per_tp = [0.0] * len(tp_prices)
                for leg in legs:
                    idx = int(leg.get("tp_index") or 0) - 1
                    if 0 <= idx < len(new_trade.volume_per_tp):
                        new_trade.volume_per_tp[idx] = float(leg.get("volume") or 0.0)
                new_trade.split_legs = {}
                for leg in legs:
                    # position_id is the ticket used by positions_get;
                    # order ticket is retained as metadata for audits.
                    leg_ticket = leg.get("position_id")
                    if not leg_ticket:
                        raise RuntimeError(
                            "Split entry returned a leg without an exact position_id"
                        )
                    new_trade.split_legs[int(leg_ticket)] = {
                        "tp_index": int(leg.get("tp_index") or 0),
                        "tp": float(leg.get("tp") or 0.0),
                        "volume": float(leg.get("volume") or 0.0),
                        "order_ticket": int(leg.get("ticket") or 0),
                        "status": "open",
                    }
                new_trade.split_position_ids = [
                    ticket
                    for ticket, _ in sorted(
                        new_trade.split_legs.items(),
                        key=lambda item: (
                            int(item[1].get("tp_index") or 10**6),
                            item[0],
                        ),
                    )
                ]
                # anchor mt5_position_id to first leg for backward compat
                if new_trade.split_position_ids:
                    new_trade.mt5_position_id = new_trade.split_position_ids[0]
                new_trade.mt5_ticket = legs[0]["ticket"] if legs else None
                new_trade.execution_comment = legs[0].get("comment") if legs else None
                sig["execution"] = {"legs": legs, "mode": "split"}
                # Update entry to actual volume-weighted fill price
                _fill_vols = [l["volume"] for l in legs]
                _fill_prices = [l["price"] for l in legs]
                _total_vol = sum(_fill_vols)
                if _total_vol > 0:
                    _avg_fill = sum(p * v for p, v in zip(_fill_prices, _fill_vols)) / _total_vol
                    new_trade.entry = round(_avg_fill, 6)
                    sig["entry_price"] = new_trade.entry
                print(
                    f"[Core] {symbol} SPLIT entry: {len(legs)} legs, "
                    f"vols={vols}, total={new_trade.volume:.2f}, "
                    f"position_ids={new_trade.split_position_ids}"
                )
            else:
                execution_payload = self.mt5_executor.execute_entry(
                    symbol,
                    side=new_trade.side,
                    entry_price=new_trade.entry,
                    stop_price=new_trade.stop,
                    # Pass LAST TP as hard broker TP so intermediate TPs
                    # are handled by the bot via partial market closes.
                    tp_price=tp_prices[-1] if tp_prices else None,
                    comment=new_trade.narrative[:28] if new_trade.narrative else None,
                    entry_min=sig.get("entry_min"),
                    entry_max=sig.get("entry_max"),
                )
                new_trade.volume = execution_payload.get("volume", 0.0)
                new_trade.volume_remaining = new_trade.volume
                _info = _mt5.symbol_info(symbol)
                _step = float(getattr(_info, "volume_step", 0.0) or 0.0) if _info else 0.0
                new_trade.volume_per_tp = _compute_tp_volumes(
                    new_trade.volume, len(tp_prices), step=_step or 0.01
                )
                print(
                    f"[Core] {symbol} MONITOR entry: volume_per_tp={new_trade.volume_per_tp} "
                    f"(total={new_trade.volume:.2f}, n_tps={len(tp_prices)})"
                )
                new_trade.mt5_ticket = execution_payload.get("ticket")
                new_trade.mt5_position_id = execution_payload.get("position_id")
                new_trade.execution_comment = execution_payload.get("comment")
                sig["execution"] = execution_payload
                # Update entry to actual broker fill price
                _fill = execution_payload.get("price")
                if _fill:
                    new_trade.entry = round(float(_fill), 6)
                    sig["entry_price"] = new_trade.entry

            # Strip TPs that the fill price has already passed.
            # Slippage can push the fill beyond near TPs, causing the bot
            # to immediately mark them as hit even though MT5 never triggered.
            _e = new_trade.entry
            if new_trade.side == "LONG":
                new_trade.tp_prices = [t for t in new_trade.tp_prices if t > _e]
            else:
                new_trade.tp_prices = [t for t in new_trade.tp_prices if t < _e]
            sig["tp_prices"] = [round(t, 6) for t in new_trade.tp_prices]
            if new_trade.tp_prices:
                sig["tp_price"] = round(new_trade.tp_prices[-1], 6)
            print(
                f"[Core] {symbol} after fill={new_trade.entry:.5f}: "
                f"active tp_prices={[round(t,5) for t in new_trade.tp_prices]}"
            )
            self._entry_cooldowns.pop(symbol, None)
        return new_trade

    def get_signals(self) -> Dict[str, dict]:
        results: Dict[str, dict] = {}
        shadow_candidate_jobs: List[Dict[str, Any]] = []
        symbols = self._get_symbols()
        t_start = time.time()
        prof = self.profiler

        with prof.section("context_update"):
            self._update_global_context()

        # MT5 health check: the executor connects once at startup and the link can
        # silently die (terminal restart, network). Try to re-initialize, throttled.
        if self.mt5_executor and not self.mt5_executor.connection_alive():
            now = time.time()
            if now - self._last_reconnect_ts >= 60.0:
                self._last_reconnect_ts = now
                ok = self.mt5_executor.reconnect()
                print(f"[MT5] connection lost — reconnect {'ok' if ok else 'failed'}")
                if ok:
                    self._refresh_pending_limit_capabilities()
        elif self.mt5_executor is None and MT5_EXECUTION_ENABLED:
            # Executor never came up (e.g. terminal was still cold-starting when the
            # bot launched) — keep retrying until the terminal is reachable.
            now = time.time()
            if now - self._executor_retry_ts >= 60.0:
                self._executor_retry_ts = now
                if self._try_create_executor():
                    self._refresh_pending_limit_capabilities()
                    self._manage_pending_limits(startup=True)
                    self._hydrate_active_trades_from_mt5()

        if self.mt5_executor:
            self._manage_pending_limits()

        dirty = False

        for symbol in symbols:
            self._record_shadow_scan_raw(symbol)
            pending_plan = self.pending_limits.get(symbol)
            if (
                pending_plan is not None
                and symbol not in self.active_trades
            ):
                results[symbol] = {
                    "signal": "WAIT_PENDING_LIMIT",
                    "side": pending_plan.side,
                    "entry_price": pending_plan.limit_price,
                    "pending_limit": {
                        "plan_id": pending_plan.plan_id,
                        "state": pending_plan.state.value,
                        "created_at": pending_plan.created_at,
                        "expires_at": pending_plan.expires_at,
                    },
                    "info": (
                        "Durable exact-retest LIMIT lifecycle owns symbol; "
                        "fresh entry blocked"
                    ),
                }
                self._log_signal(symbol, results[symbol])
                continue
            if time.time() - t_start > self.TIME_BUDGET_SEC:
                results[symbol] = {"signal": "SKIP_BUDGET", "info": "Time budget exceeded, continue next tick"}
                continue

            try:
                with prof.section("data_fetch"):
                    data = self._build_tf_data(symbol)
                df_1H = data.get("1H")

                if df_1H is None or df_1H.empty:
                    print(f"[Ticker] {symbol} NO_DATA | 1H frame is empty — check data source / MT5 bridge")
                    results[symbol] = {"signal": "NO_DATA"}
                    continue

                df_1M = data.get("1M")
                df_5M = data.get("5M")
                df_15M = data.get("15M")

                # MT5 live tick is primary — bridge candles are fallback only.
                # Pass the open trade's side so SHORT positions are monitored on
                # ASK (the price MT5 uses for their SL/TP) and LONG on BID.
                _trade_side = getattr(self.active_trades.get(symbol), "side", None)
                _mt5_live = self.mt5_executor.get_current_price(symbol, _trade_side) if self.mt5_executor else None
                if _mt5_live is not None:
                    last_price = _mt5_live
                elif df_1M is not None and not df_1M.empty:
                    last_price = float(df_1M["close"].iloc[-1])
                elif df_5M is not None and not df_5M.empty:
                    last_price = float(df_5M["close"].iloc[-1])
                elif df_15M is not None and not df_15M.empty:
                    last_price = float(df_15M["close"].iloc[-1])
                else:
                    last_price = float(df_1H["close"].iloc[-1])

                if symbol in self.active_trades:
                    trade = self.active_trades[symbol]
                    # TP events recovered from broker-closed split legs this tick
                    # (deal-reason based — the authoritative tp_hit source in split mode).
                    leg_events: List[Dict[str, Any]] = []

                    if self.mt5_executor and (
                        self._is_split_trade(trade)
                        or getattr(trade, "mt5_position_id", None)
                        or getattr(trade, "mt5_ticket", None)
                    ):
                        if self._is_split_trade(trade):
                            # Shared with the 3-second broker-management job. The
                            # durable map makes this call idempotent if the fast job
                            # already consumed a close event.
                            with self._management_lock:
                                lifecycle = self._poll_idea_lifecycle(
                                    symbol, trade, last_price=last_price
                                )
                            leg_events.extend(lifecycle.get("events") or [])
                            dirty = dirty or bool(lifecycle.get("changed"))
                            if lifecycle.get("final"):
                                manage = {
                                    "signal": "EXIT_BROKER",
                                    "side": trade.side,
                                    "exit_price": float(last_price),
                                    "info": "All split legs closed",
                                    "tf": trade.tf,
                                    "narrative": trade.narrative,
                                    "events": leg_events + [
                                        {"type": "BROKER_CLOSE", "info": "All split TP legs closed in MT5"}
                                    ],
                                    "telegram_chat_id": getattr(trade, "telegram_chat_id", None),
                                    "telegram_message_id": getattr(trade, "telegram_message_id", None),
                                }
                                self._register_broker_close(symbol, trade, manage)
                                results[symbol] = manage
                                self.active_trades.pop(symbol, None)
                                dirty = True
                                self._log_signal(symbol, manage)
                                continue
                            if lifecycle.get("query_failed"):
                                results[symbol] = {
                                    "signal": "HOLD",
                                    "info": "MT5 query failed — keeping split trade",
                                }
                                continue
                            if (
                                lifecycle.get("pending_history")
                                and not lifecycle.get("visible_open_count")
                                and not leg_events
                            ):
                                results[symbol] = {
                                    "signal": "HOLD",
                                    "info": "Split leg disappeared — awaiting broker deal history",
                                }
                                continue
                        else:
                            pos_id = trade.mt5_position_id or trade.mt5_ticket
                            position = self.mt5_executor.get_position(symbol, pos_id)
                            if position is not None:
                                self._broker_missing_counts.pop(symbol, None)
                            elif self._position_gone_confirmed(symbol, query_failed=False):
                                self._broker_missing_counts.pop(symbol, None)
                                manage = {
                                    "signal": "EXIT_BROKER",
                                    "side": trade.side,
                                    "exit_price": float(last_price),
                                    "info": "Position closed externally",
                                    "tf": trade.tf,
                                    "narrative": trade.narrative,
                                    "events": [
                                        {"type": "BROKER_CLOSE", "info": "Position disappeared from MT5"}
                                    ],
                                    "telegram_chat_id": getattr(trade, "telegram_chat_id", None),
                                    "telegram_message_id": getattr(trade, "telegram_message_id", None),
                                }
                                self._register_broker_close(symbol, trade, manage)
                                results[symbol] = manage
                                self.active_trades.pop(symbol, None)
                                dirty = True
                                self._log_signal(symbol, manage)
                                continue
                            else:
                                print(f"[Core] {symbol} position not visible — awaiting confirmation before EXIT_BROKER")
                                results[symbol] = {"signal": "HOLD", "info": "position not visible — awaiting confirmation"}
                                continue

                    with prof.section("trade_manage"):
                        raw_manage = self._check_active_trade(symbol, last_price, trade)

                    # Surface TP legs closed by the broker this tick (detected via
                    # deal reasons above) — drives Telegram notifications and the
                    # break-even move below.
                    if leg_events:
                        raw_manage = dict(raw_manage)
                        raw_manage.setdefault("events", []).extend(leg_events)

                    # In split mode the broker owns each leg's SL and TP.
                    # EXIT_SL, EXIT_TP, and EXIT_TIME are all suppressed — the bot waits
                    # for EXIT_BROKER (all split_position_ids disappear from MT5) instead
                    # of sending redundant close orders that would fill at market price
                    # rather than the exact TP/SL levels set on each leg.
                    _is_split_active = self._is_split_trade(trade)
                    if _is_split_active and raw_manage.get("signal") in ("EXIT_SL", "EXIT_TP"):
                        raw_manage = dict(raw_manage)
                        raw_manage["signal"] = "HOLD"
                        raw_manage["info"] = "split_mode: SL/TP managed by broker"

                    with prof.section("ai_filter"):
                        manage = self.ai.on_signal(symbol, raw_manage, data, self.active_trades)

                    risk_action = self.risk_rules.check_trade(trade, last_price=last_price)
                    # In split mode the broker manages each leg's TP/SL automatically.
                    # EXIT_TIME would force-close still-open legs that haven't reached TP yet,
                    # turning a winning trade into a loss. Let broker handle the exit instead.
                    if risk_action and not _is_split_active:
                        manage = dict(manage)
                        manage.update(risk_action)

                    if raw_manage.get("events"):
                        dirty = True

                    if raw_manage.get("events") and not manage.get("events"):
                        manage["events"] = raw_manage["events"]
                    if raw_manage.get("signal") in ("EXIT_SL", "EXIT_TP"):
                        manage["signal"] = raw_manage["signal"]
                        manage["exit_price"] = raw_manage.get("exit_price", manage.get("exit_price"))

                    # Block all MT5 closes during startup grace period.
                    _in_grace = (time.time() - self._startup_time) < self._startup_grace_sec
                    if _in_grace and self.mt5_executor:
                        all_events = list(manage.get("events") or [])
                        broker_events = [
                            e for e in all_events
                            if e.get("source") in {"leg_close", "broker_deal"}
                        ]
                        dropped_events = [e for e in all_events if e not in broker_events]
                        if broker_events:
                            # These are already-completed broker facts, not close
                            # intents; retain their one-shot Telegram event in grace.
                            manage["events"] = broker_events
                        else:
                            manage.pop("events", None)
                        # _check_active_trade already advanced tp_hit for these events;
                        # roll it back, otherwise the partial closes (and Telegram
                        # notifications) for those TPs are swallowed forever.
                        # Leg-close events are excluded: the broker already closed
                        # those legs — that tp_hit advance is a fact, not an intent.
                        n_tp_events = sum(
                            1 for e in dropped_events
                            if e.get("type") == "TP"
                            and e.get("source") not in {"leg_close", "broker_deal"}
                        )
                        if n_tp_events:
                            trade.tp_hit = max(0, int(trade.tp_hit) - n_tp_events)
                            dirty = True
                        if manage.get("signal") in ("EXIT_SL", "EXIT_TP", "EXIT_TIME"):
                            print(f"[Core] {symbol} startup grace ({self._startup_grace_sec:.0f}s) — skipping close")
                            manage["signal"] = "HOLD"

                    # Process TP partial closes
                    for ev in (manage.get("events") or []):
                        ev_type = ev.get("type")

                        if ev_type == "TP":
                            # In split mode each leg has its own broker TP — the broker
                            # closes it automatically at the exact price.  Sending an
                            # additional close_trade() here would fill at the current
                            # market price, which is always worse than the broker TP.
                            if self._is_split_trade(trade):
                                continue
                            tp_idx = int(ev.get("tp_index", 0))
                            vol_per_tp = getattr(trade, "volume_per_tp", [])
                            has_executor = self.mt5_executor and getattr(trade, "mt5_position_id", None)
                            if has_executor and tp_idx > 0 and tp_idx <= len(vol_per_tp):
                                partial_vol = vol_per_tp[tp_idx - 1]
                                remaining = getattr(trade, "volume_remaining", 0.0) or trade.volume
                                close_vol = round(min(partial_vol, remaining), 2)
                                if close_vol > 0:
                                    try:
                                        self.mt5_executor.close_trade(
                                            symbol,
                                            position_id=trade.mt5_position_id,
                                            volume=close_vol,
                                        )
                                        trade.volume_remaining = round(max(0.0, remaining - close_vol), 2)
                                        ev["partial_close_vol"] = close_vol
                                        ev["volume_remaining"] = trade.volume_remaining
                                        dirty = True
                                        print(
                                            f"[Core] {symbol} TP{tp_idx} partial close "
                                            f"{close_vol:.2f} lots → remaining: {trade.volume_remaining:.2f}"
                                        )
                                    except Exception as exc:
                                        manage.setdefault("execution_error", str(exc))

                    # Move SL to break-even once TP1 is reached. tp_hit is persisted, so
                    # a failed modify retries every tick until it succeeds. Works for both
                    # modes: split (all remaining legs) and monitor (single position).
                    if manage.get("signal") not in ("EXIT_SL", "EXIT_TP", "EXIT_TIME", "EXIT_BROKER"):
                        be_events = self._move_idea_entries_to_breakeven(symbol, trade)
                        if be_events:
                            dirty = True
                            manage.setdefault("events", []).extend(be_events)

                    # Daily/Friday flat close: hard rule — force exit all positions at
                    # DAILY_CLOSE_HOUR (every day) / FRIDAY_CLOSE_HOUR UTC+3. Placed
                    # after TP partial-close events and after grace period so it
                    # always fires.
                    if (
                        self.global_context.get("friday_close") or self.global_context.get("daily_close")
                    ) and manage.get("signal") not in ("EXIT_BROKER",):
                        _close_label = (
                            f"Закрытие перед выходными (пятница {FRIDAY_CLOSE_HOUR}:00 UTC+3)"
                            if self.global_context.get("friday_close")
                            else f"Дневное закрытие ({DAILY_CLOSE_HOUR}:00 UTC+3)"
                        )
                        manage = dict(manage)
                        manage["signal"] = "EXIT_TIME"
                        manage["info"] = _close_label
                        manage["exit_price"] = float(last_price)
                        print(f"[Core] {symbol} {_close_label} — принудительное закрытие позиции")

                    if manage.get("signal") in ("EXIT_SL", "EXIT_TP", "EXIT_TIME"):
                        manage.setdefault("telegram_chat_id", getattr(trade, "telegram_chat_id", None))
                        manage.setdefault("telegram_message_id", getattr(trade, "telegram_message_id", None))
                        close_failed = False
                        closed_position_ids: List[int] = []
                        if self.mt5_executor:
                            exec_block = manage.setdefault("execution", {})
                            if not isinstance(exec_block, dict):
                                exec_block = {}
                                manage["execution"] = exec_block
                            conn_ok = self.mt5_executor.connection_alive()
                            # Closing the idea means closing every entry it holds.
                            for _entry in self._idea_entries(trade):
                                split_ids = getattr(_entry, "split_position_ids", [])
                                if split_ids:
                                    # Close any remaining split legs (already-hit TPs are gone)
                                    closed_count = 0
                                    remaining_legs: List[int] = []
                                    for pid in split_ids:
                                        try:
                                            ok = self.mt5_executor.close_trade(symbol, position_id=pid, volume=None)
                                            if ok:
                                                closed_count += 1
                                            else:
                                                # Any unconfirmed close is retried. A transient
                                                # ticket lookup failure is not proof that the
                                                # broker position is gone, even on a live link.
                                                remaining_legs.append(pid)
                                        except Exception as exc:
                                            manage.setdefault("execution_error", str(exc))
                                            remaining_legs.append(pid)
                                    exec_block["mt5_closed_split"] = (
                                        int(exec_block.get("mt5_closed_split") or 0) + closed_count
                                    )
                                    if remaining_legs:
                                        close_failed = True
                                        _entry.split_position_ids = remaining_legs
                                elif getattr(_entry, "mt5_position_id", None):
                                    try:
                                        close_vol = getattr(_entry, "volume_remaining", 0.0) or _entry.volume
                                        closed = self.mt5_executor.close_trade(
                                            symbol,
                                            position_id=_entry.mt5_position_id,
                                            volume=close_vol,
                                        )
                                        exec_block["mt5_closed"] = closed
                                        if not closed:
                                            close_failed = True
                                        else:
                                            closed_position_ids.append(int(_entry.mt5_position_id))
                                    except Exception as exc:
                                        manage.setdefault("execution_error", str(exc))
                                        close_failed = True
                        if close_failed:
                            # Keep the trade tracked and retry next tick — deleting it
                            # here would leave a live position unmanaged in the market.
                            print(
                                f"[Core] {symbol} close failed ({manage.get('execution_error', 'position not confirmed')})"
                                f" — keeping trade, retrying next tick"
                            )
                            manage = dict(manage)
                            manage["signal"] = "HOLD"
                            manage["info"] = "MT5 close failed — retrying next tick"
                            trade.last_price_ts = time.time()
                            results[symbol] = manage
                            dirty = True
                            self._log_signal(symbol, manage)
                            continue
                        if closed_position_ids:
                            self._attach_position_close_metrics(manage, closed_position_ids)
                        results[symbol] = manage
                        self.active_trades.pop(symbol, None)
                        dirty = True
                        self._log_signal(symbol, manage)
                        continue

                    trade.last_price_ts = time.time()

                    # Position Adding: a still-working idea may be confirmed
                    # again and earn one more entry under the shared risk cap.
                    if (
                        manage.get("signal") == "HOLD"
                        and not _in_grace
                        and symbol not in self.pending_limits
                    ):
                        addon_sig = self._try_position_add(symbol, trade, data, last_price)
                        if addon_sig is not None:
                            dirty = True
                            manage = dict(manage)
                            manage.setdefault("events", []).append({
                                "type": "POSITION_ADD",
                                "entry_index": addon_sig.get("entry_index"),
                                "idea_entries": addon_sig.get("idea_entries"),
                                "idea_max_entries": addon_sig.get("idea_max_entries"),
                                "idea_risk_pct": addon_sig.get("idea_risk_pct"),
                                "be_entries": addon_sig.get("idea_be_entries") or [],
                                "entry_price": addon_sig.get("entry_price"),
                                "stop_price": addon_sig.get("stop_price"),
                                "tp_prices": addon_sig.get("tp_prices") or [],
                                "volume": addon_sig.get("execution", {}).get("volume"),
                            })
                            manage["position_add"] = addon_sig

                    results[symbol] = manage
                    self._log_signal(symbol, manage)
                    continue

                strategy_data = self._strategy_view(data, symbol=symbol)
                with prof.section("strategy"):
                    raw_sig = self.strategy.generate_signal(strategy_data, symbol=symbol)
                candidate_ledger = getattr(
                    self, "shadow_candidate_ledger", None
                )
                if (
                    candidate_ledger is not None
                    and candidate_ledger.enabled
                ):
                    shadow_candidate_jobs.append({
                        "deployment_id": (
                            os.getenv("DEPLOYMENT_ID", "").strip()
                            or os.getenv(
                                "STRATEGY_VERSION", "unversioned"
                            ).strip()
                            or "unversioned"
                        ),
                        "symbol": symbol,
                        "observed_at_utc": datetime.now(timezone.utc),
                        "decision_bar_close": (
                            self._closed_m15_decision_time(data)
                        ),
                        # The dict is immutable to the diagnostic worker.
                        "production_signal": dict(raw_sig),
                        # DataFrames are closed/as-of views and are only read.
                        "strategy_data": dict(strategy_data),
                        # Frozen portfolio state at this causal decision. The
                        # worker can score concentration but cannot change it.
                        "active_exposures": {
                            active_symbol: str(active_trade.side)
                            for active_symbol, active_trade
                            in self.active_trades.items()
                            if active_trade is not None
                        },
                    })
                if DEBUG_RAW_SIGNALS and raw_sig.get("signal") != "NO_TRIGGER":
                    print(f"[RAW_SIG] {symbol} {raw_sig}")
                with prof.section("ai_filter"):
                    sig = self.ai.on_signal(symbol, raw_sig, data, self.active_trades)

                if raw_sig.get("tp_prices") and not sig.get("tp_prices"):
                    sig["tp_prices"] = raw_sig["tp_prices"]
                # Do NOT force ENTER when AI explicitly rejected — AI_REJECT dict has no
                # side/entry_price/stop_price and bypassing the filter causes SKIP_NO_SIDE every tick.

                sig = self._apply_global_filters(symbol, sig)
                sig = self._apply_session_filter(symbol, sig)
                sig = self._apply_vol_regime_filter(symbol, sig, strategy_data)
                sig = self._apply_execution_quality_filter(symbol, sig)
                sig = self._attach_shadow_score(symbol, sig)

                if sig.get("signal") == "ENTER":
                    side = sig.get("side")
                    if not side:
                        sig["signal"] = "SKIP_NO_SIDE"
                        results[symbol] = sig
                        self._log_signal(symbol, sig)
                        continue

                    trig_sig = self._trigger_signature(sig)
                    if self._apply_entry_guards(symbol, sig, side=side, trig_sig=trig_sig):
                        results[symbol] = sig
                        self._log_signal(symbol, sig)
                        continue

                    (
                        shadow_plan_id,
                        entry_quote,
                        pending_expires_at,
                        pending_observed_at,
                    ) = self._register_shadow_candidate(
                        symbol,
                        sig,
                        trigger_signature=trig_sig,
                        data=data,
                    )
                    if shadow_plan_id:
                        sig.setdefault("setup_id", shadow_plan_id)

                    requires_exact_limit = (
                        str(sig.get("entry_order_type") or "").upper()
                        == "LIMIT_RETEST"
                    )
                    persistent_limit_routed = (
                        PERSISTENT_LIMIT_ENABLED
                        and symbol.upper() in PERSISTENT_LIMIT_SYMBOLS
                        and self.mt5_executor
                    )
                    if requires_exact_limit and not persistent_limit_routed:
                        sig["signal"] = "WAIT_LIMIT_UNSUPPORTED"
                        sig["info"] = (
                            "LIMIT_RETEST cannot fall through to a market "
                            "entry; persistent native LIMIT is unavailable "
                            "for this symbol"
                        )
                        results[symbol] = sig
                        self._log_signal(symbol, sig)
                        continue

                    if persistent_limit_routed:
                        if entry_quote is None:
                            sig["signal"] = "WAIT_LIMIT_QUOTE"
                            sig["info"] = (
                                "No causal broker quote; exact LIMIT "
                                "placement is fail-closed"
                            )
                            results[symbol] = sig
                            self._log_signal(symbol, sig)
                            continue

                        side_key = str(side).upper()
                        executable_price = float(
                            entry_quote.ask
                            if side_key == "LONG"
                            else entry_quote.bid
                        )
                        planned_entry = float(
                            sig.get("entry_price") or 0.0
                        )
                        entry_min = float(
                            sig.get("entry_min")
                            if sig.get("entry_min") is not None
                            else planned_entry
                        )
                        entry_max = float(
                            sig.get("entry_max")
                            if sig.get("entry_max") is not None
                            else planned_entry
                        )
                        lower, upper = sorted((entry_min, entry_max))
                        strict_touch = (
                            lower <= executable_price <= upper
                        )
                        awaiting_retest = (
                            (
                                side_key == "LONG"
                                and executable_price > upper
                            )
                            or (
                                side_key == "SHORT"
                                and executable_price < lower
                            )
                        )

                        if awaiting_retest:
                            capabilities = (
                                self._get_pending_limit_capabilities(
                                    symbol
                                )
                            )
                            if capabilities.get("ready") is not True:
                                sig["signal"] = "WAIT_LIMIT_UNSUPPORTED"
                                sig["info"] = (
                                    "Broker/account does not prove LIMIT + "
                                    "specified expiry + hedging capability"
                                )
                                sig["pending_limit_capabilities"] = {
                                    key: capabilities.get(key)
                                    for key in (
                                        "limit_allowed",
                                        "specified_expiration",
                                        "account_hedging",
                                        "ready",
                                    )
                                }
                                results[symbol] = sig
                                self._log_signal(symbol, sig)
                                continue
                            try:
                                with self._management_lock:
                                    plan = self._arm_pending_limit(
                                        symbol,
                                        sig,
                                        trigger_signature=trig_sig,
                                        expires_at=pending_expires_at,
                                        shadow_plan_id=shadow_plan_id,
                                        observed_at=pending_observed_at,
                                    )
                            except RBConsumedEventsValidationError as exc:
                                sig["signal"] = "WAIT_RB_EVENT_STATE"
                                sig["info"] = (
                                    "Exact H1-RB one-shot reservation failed; "
                                    f"no broker order was sent ({exc})"
                                )
                                results[symbol] = sig
                                self._log_signal(symbol, sig)
                                continue
                            except RiskCapacityError as exc:
                                sig["signal"] = "WAIT_RISK_ENTRY"
                                sig["info"] = str(exc)
                                adjustment = exc.to_payload()
                                if adjustment:
                                    sig["risk_adjustment"] = adjustment
                                results[symbol] = sig
                                self._log_signal(symbol, sig)
                                continue
                            except PendingOrderRejected as exc:
                                retained = self.pending_limits.get(symbol)
                                cancelling = bool(
                                    retained is not None
                                    and retained.state
                                    == PendingLimitState.CANCELLING
                                )
                                sig["signal"] = (
                                    "WAIT_PENDING_RECONCILE"
                                    if cancelling
                                    else "WAIT_LIMIT_REJECTED"
                                )
                                sig["execution_error"] = str(exc)
                                sig["info"] = (
                                    "Broker synchronously rejected the "
                                    "pending order; no hidden order is "
                                    "possible"
                                )
                                if retained is not None:
                                    sig["pending_limit"] = {
                                        "plan_id": retained.plan_id,
                                        "state": retained.state.value,
                                        "created_at": retained.created_at,
                                        "expires_at": retained.expires_at,
                                    }
                                results[symbol] = sig
                                self._log_signal(symbol, sig)
                                print(
                                    f"[Core] Persistent LIMIT rejected for "
                                    f"{symbol}: {exc}"
                                )
                                continue
                            except Exception as exc:
                                retained = self.pending_limits.get(symbol)
                                if retained is not None:
                                    sig["signal"] = (
                                        "WAIT_PENDING_RECONCILE"
                                    )
                                    sig["pending_limit"] = {
                                        "plan_id": retained.plan_id,
                                        "state": retained.state.value,
                                        "created_at": retained.created_at,
                                        "expires_at": retained.expires_at,
                                    }
                                    sig["info"] = (
                                        "Placement outcome is ambiguous; "
                                        "durable intent retained"
                                    )
                                else:
                                    sig["signal"] = "EXECUTION_ERROR"
                                    sig["execution_error"] = str(exc)
                                    self._entry_cooldowns[symbol] = (
                                        time.time()
                                        + self._entry_cooldown_sec
                                    )
                                results[symbol] = sig
                                self._log_signal(symbol, sig)
                                print(
                                    f"[Core] Persistent LIMIT failed for "
                                    f"{symbol}: {exc}"
                                )
                                continue

                            sig["signal"] = "PENDING_LIMIT"
                            sig["pending_limit"] = {
                                "plan_id": plan.plan_id,
                                "state": plan.state.value,
                                "limit_price": plan.limit_price,
                                "entry_min": plan.entry_min,
                                "entry_max": plan.entry_max,
                                "created_at": plan.created_at,
                                "expires_at": plan.expires_at,
                                "broker_order_tickets": list(
                                    plan.working_order_tickets
                                ),
                            }
                            sig["info"] = (
                                "Native exact-retest LIMIT armed"
                            )
                            results[symbol] = sig
                            self._log_signal(symbol, sig)
                            continue

                        if strict_touch and requires_exact_limit:
                            try:
                                self._consume_exact_h1_rb_event(
                                    symbol,
                                    sig,
                                    trigger_signature=trig_sig,
                                    reason="first-touch-observed-before-arm",
                                    consumed_at_utc=pending_observed_at,
                                )
                            except Exception as exc:
                                sig["signal"] = "WAIT_RB_EVENT_STATE"
                                sig["info"] = (
                                    "Exact H1-RB touch could not be "
                                    f"persisted; entry is fail-closed ({exc})"
                                )
                            else:
                                sig["signal"] = "WAIT_LIMIT_FIRST_TOUCH"
                                sig["info"] = (
                                    "First exact retest was already present "
                                    "at decision time; no marketable LIMIT or "
                                    "market fallback is allowed"
                                )
                            results[symbol] = sig
                            self._log_signal(symbol, sig)
                            continue

                        if not strict_touch:
                            if requires_exact_limit:
                                try:
                                    self._consume_exact_h1_rb_event(
                                        symbol,
                                        sig,
                                        trigger_signature=trig_sig,
                                        reason=(
                                            "entry-crossed-before-native-limit"
                                        ),
                                        consumed_at_utc=pending_observed_at,
                                    )
                                except Exception as exc:
                                    sig["signal"] = "WAIT_RB_EVENT_STATE"
                                    sig["info"] = (
                                        "Exact H1-RB invalidation could not "
                                        f"be persisted; fail-closed ({exc})"
                                    )
                                    results[symbol] = sig
                                    self._log_signal(symbol, sig)
                                    continue
                            sig["signal"] = "WAIT_LIMIT_INVALIDATED"
                            sig["info"] = (
                                "Price already crossed beyond the "
                                "two-sided entry zone; gap fill rejected"
                            )
                            watcher = getattr(
                                self, "shadow_tick_watcher", None
                            )
                            if (
                                watcher is not None
                                and shadow_plan_id is not None
                            ):
                                watcher.record_terminal(
                                    shadow_plan_id,
                                    "INVALIDATED",
                                    at_utc=pending_observed_at,
                                    tick_msc=int(
                                        pending_observed_at.timestamp()
                                        * 1000
                                    ),
                                    reason=sig["info"],
                                    inclusive=True,
                                )
                            results[symbol] = sig
                            self._log_signal(symbol, sig)
                            continue

                    if requires_exact_limit:
                        # Structural invariant: LIMIT_RETEST is never converted
                        # to a market order, even if future routing branches are
                        # changed without handling every quote disposition.
                        sig["signal"] = "WAIT_LIMIT_NO_MARKET_FALLBACK"
                        sig["info"] = (
                            "Exact-retest signal reached no safe native-LIMIT "
                            "disposition; market entry is forbidden"
                        )
                        results[symbol] = sig
                        self._log_signal(symbol, sig)
                        continue

                    try:
                        new_trade = self._execute_entry_signal(symbol, sig)
                    except RiskCapacityError as exc:
                        # Keep the technical SL intact. The signal is retried
                        # on the next tick and becomes executable when price
                        # reaches the risk-compatible entry geometry.
                        sig["signal"] = "WAIT_RISK_ENTRY"
                        sig["info"] = str(exc)
                        risk_adjustment = exc.to_payload()
                        if risk_adjustment:
                            sig["risk_adjustment"] = risk_adjustment
                        results[symbol] = sig
                        self._log_signal(symbol, sig)
                        print(
                            f"[Core] {symbol} waiting for risk-compatible "
                            f"entry: {exc}"
                        )
                        continue
                    except Exception as exc:
                        err_str = str(exc)
                        is_stale = "Stale signal rejected" in err_str
                        cooldown_sec = self._stale_cooldown_sec if is_stale else self._entry_cooldown_sec
                        self._entry_cooldowns[symbol] = time.time() + cooldown_sec
                        sig["signal"] = "EXECUTION_ERROR"
                        sig["execution_error"] = err_str
                        results[symbol] = sig
                        self._log_signal(symbol, sig)
                        print(f"[Core] Execution failed for {symbol}: {exc}. Cooldown {cooldown_sec:.0f}s")
                        continue

                    # Stable identity makes journal retries idempotent without
                    # letting an unrelated stale open row hide this new setup.
                    sig.setdefault("setup_id", uuid.uuid4().hex)
                    # Idea identity — shared by every position-adding entry that
                    # later joins this setup.
                    new_trade.idea_id = uuid.uuid4().hex
                    new_trade.entry_index = 1
                    new_trade.idea_trigger_signatures = [trig_sig]
                    sig["idea_id"] = new_trade.idea_id
                    sig["entry_index"] = 1
                    self.active_trades[symbol] = new_trade
                    dirty = True
                    self._entries_today[symbol] = self._entries_today.get(symbol, 0) + 1
                    self._trigger_signatures.setdefault(symbol, set()).add(trig_sig)
                    watcher = getattr(
                        self, "shadow_tick_watcher", None
                    )
                    if (
                        watcher is not None
                        and shadow_plan_id is not None
                    ):
                        watcher.record_terminal(
                            shadow_plan_id,
                            "FILLED",
                            at_utc=datetime.now(timezone.utc),
                            reason=(
                                "minute-cycle market entry executed"
                            ),
                            inclusive=True,
                        )

                results[symbol] = sig
                self._log_signal(symbol, sig)

            except Exception:
                traceback.print_exc()
                results[symbol] = {"signal": "ERROR", "info": "Exception in get_signals() (see logs)"}

        if dirty:
            with prof.section("save_state"):
                save_active_trades(self.active_trades, AI_DATA_DIR / "active_trades.json")

        # Submit only after every live order decision and durable trade-state
        # write in this cycle. The worker's result is never read by Core.
        self._submit_shadow_candidate_jobs(
            shadow_candidate_jobs,
            results,
        )

        prof.dump(prefix="[Profiler:core]")
        return results


def _should_start_bridge() -> bool:
    # Bridge writes cache files used as DataFeed fallback — start whenever
    # MT5 credentials are available, regardless of execution mode.
    return bool(MT5_LOGIN and MT5_PASSWORD and MT5_SERVER)


def _start_bridge_thread(manage_connection: bool) -> Tuple[Optional[threading.Event], Optional[threading.Thread]]:
    try:
        mappings = parse_symbol_spec(MT5_BRIDGE_SYMBOLS)
        timeframes = parse_timeframes(MT5_BRIDGE_TIMEFRAMES)
    except Exception as exc:
        print(f"[MT5 Bridge] config error: {exc}")
        return None, None

    stop_event = threading.Event()

    def _runner():
        bridge = MT5NativeBridge(
            mappings=mappings,
            timeframes=timeframes,
            lookback_days=MT5_BRIDGE_LOOKBACK_DAYS,
            poll_interval=MT5_BRIDGE_INTERVAL,
            cache_dir=MT5_CACHE_DIR,
        )
        try:
            bridge.run(
                once=False,
                stop_event=stop_event,
                manage_connection=manage_connection,
                login=MT5_LOGIN,
                password=MT5_PASSWORD,
                server=MT5_SERVER,
            )
        except Exception as exc:
            print(f"[MT5 Bridge] stopped: {exc}")

    thread = threading.Thread(target=_runner, name="MT5Bridge", daemon=True)
    thread.start()
    print(
        f"[MT5 Bridge] started (symbols={MT5_BRIDGE_SYMBOLS}, tfs={MT5_BRIDGE_TIMEFRAMES}, interval={MT5_BRIDGE_INTERVAL}s)"
    )
    return stop_event, thread


def _acquire_pid_lock(path: Path) -> bool:
    """Return True if this is the only running instance, False if another is alive."""
    if path.exists():
        try:
            old_pid = int(path.read_text().strip())
            import psutil
            if psutil.pid_exists(old_pid):
                proc = psutil.Process(old_pid)
                if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                    print(f"[PID Lock] Another instance is already running (PID {old_pid}). Exiting.")
                    return False
        except Exception:
            pass  # stale lock — overwrite it
    path.write_text(str(os.getpid()))
    return True


def _release_pid_lock(path: Path) -> None:
    try:
        if path.exists() and path.read_text().strip() == str(os.getpid()):
            path.unlink()
    except Exception:
        pass


if __name__ == "__main__":
    _LOCK_PATH = AI_DATA_DIR / "bot.pid"
    if not _acquire_pid_lock(_LOCK_PATH):
        raise SystemExit(1)

    core: Optional[Core] = None
    bridge_stop: Optional[threading.Event] = None
    bridge_thread: Optional[threading.Thread] = None
    try:
        core = Core()
        if _should_start_bridge():
            manage_conn = not MT5_EXECUTION_ENABLED
            bridge_stop, bridge_thread = _start_bridge_thread(manage_conn)
        bot = TelegramBot(TELEGRAM_TOKEN, core)
        bot.run()
    except KeyboardInterrupt:
        pass
    finally:
        if core is not None:
            core.close()
        if bridge_stop:
            bridge_stop.set()
        if bridge_thread:
            bridge_thread.join(timeout=5)
        _release_pid_lock(_LOCK_PATH)
