from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load .env from the project root so the bot works the same under the Windows
# GUI launcher, a bare `python main.py`, and systemd on a VPS. Existing process
# environment variables take precedence over .env values.
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except Exception:
    pass
AI_DATA_DIR = BASE_DIR / "ai_data"
AI_DATA_DIR.mkdir(parents=True, exist_ok=True)

# MT5 bridge cache (backup for DataFeed when direct MT5 call fails)
MT5_CACHE_DIR = AI_DATA_DIR / "mt5_cache"
MT5_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# MT5 native execution settings
MT5_EXECUTION_ENABLED = os.getenv("MT5_EXECUTION", "0").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> int | None:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


MT5_LOGIN = _env_int("MT5_LOGIN")
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "FxPro-MT5 Demo")
MT5_MAGIC = _env_int("MT5_MAGIC") or 20260318
# A smaller configured value is allowed; values above 1% are hard-clamped.
# The executor enforces the same ceiling again at the final broker request.
MT5_RISK_PER_TRADE = min(max(_env_float("MT5_RISK_PCT", 0.01), 0.0), 0.01)
# Optional explicit strategy starting capital. Zero means: capture account
# balance once and persist it in ai_data/risk_capital.json.
MT5_INITIAL_CAPITAL = _env_float("MT5_INITIAL_CAPITAL", 0.0)
MT5_SLIPPAGE = _env_int("MT5_SLIPPAGE") or 20

# Exact-retest entry lifecycle. Both switches default off so a code deploy does
# not silently alter live execution before a broker canary is approved.
PERSISTENT_LIMIT_ENABLED = (
    os.getenv("PERSISTENT_LIMIT_ENABLED", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
# Emergency-only cleanup for a fill that conflicts with an already active
# idea.  It remains independently opt-in even when persistent LIMITs are on.
PERSISTENT_LIMIT_CONFLICT_AUTOCLOSE_ENABLED = (
    os.getenv(
        "PERSISTENT_LIMIT_CONFLICT_AUTOCLOSE_ENABLED",
        "0",
    ).strip().lower()
    in {"1", "true", "yes", "on"}
)
PERSISTENT_LIMIT_SYMBOLS = frozenset(
    symbol.strip().upper()
    for symbol in os.getenv(
        "PERSISTENT_LIMIT_SYMBOLS",
        "EURUSD",
    ).split(",")
    if symbol.strip()
)
PERSISTENT_LIMIT_TTL_MIN = min(
    240,
    max(1, _env_int("PERSISTENT_LIMIT_TTL_MIN") or 15),
)
PERSISTENT_LIMIT_STATE_PATH = AI_DATA_DIR / "pending_limits.json"

# Strictly diagnostic tick observer. It recovers the terminal tick stream in
# batches and writes to its own SQLite database; it cannot submit orders.
SHADOW_TICK_WATCHER_ENABLED = (
    os.getenv("SHADOW_TICK_WATCHER_ENABLED", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
SHADOW_TICK_WATCHER_POLL_SEC = min(
    30.0,
    max(0.25, _env_float("SHADOW_TICK_WATCHER_POLL_SEC", 5.0)),
)
SHADOW_TICK_WATCHER_MAX_BACKFILL_SEC = min(
    86_400,
    max(60, _env_int("SHADOW_TICK_WATCHER_MAX_BACKFILL_SEC") or 3_600),
)
SHADOW_TICK_WATCHER_DB_PATH = AI_DATA_DIR / "shadow_tick_touches.db"

if MT5_EXECUTION_ENABLED and (MT5_LOGIN is None or not MT5_PASSWORD or not MT5_SERVER):
    MT5_EXECUTION_ENABLED = False

MT5_BRIDGE_SYMBOLS = os.getenv("MT5_BRIDGE_SYMBOLS", "EURUSD,GBPUSD,USDCAD,GOLD")
MT5_BRIDGE_TIMEFRAMES = os.getenv("MT5_BRIDGE_TIMEFRAMES", "1m,5m,15m,1h,4h,1d")
MT5_BRIDGE_LOOKBACK_DAYS = _env_int("MT5_BRIDGE_LOOKBACK_DAYS") or 15
MT5_BRIDGE_INTERVAL = _env_int("MT5_BRIDGE_INTERVAL") or 60

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "CHANGE_ME_TELEGRAM_TOKEN")
TELEGRAM_CHANNEL_ID = int(os.getenv("TELEGRAM_CHANNEL_ID", "-1003871620174"))


def _parse_id_set(raw: str) -> frozenset[int]:
    out: set[int] = set()
    for part in (raw or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            continue
    return frozenset(out)


# User/chat ids allowed to run bot commands (/status, /open, /report, /universe).
# Comma-separated. The channel itself is always allowed. When empty, commands
# from anywhere except the channel are IGNORED — bot commands expose open
# positions and must not be public.
TELEGRAM_ADMIN_IDS = _parse_id_set(os.getenv("TELEGRAM_ADMIN_IDS", ""))

# ================== UNIVERSE ==================
# Maps bot_symbol → MT5 terminal symbol name.
# These must match the exact symbol names used in the MT5 terminal
# (as shown in Market Watch / trade logs).
UNIVERSE = {
    "GOLD":   "GOLD",    # XAU/USD — FxPro uses "GOLD"
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDCAD": "USDCAD",
}

# Additional symbols for HTF context only (not traded)
CONTEXT_SYMBOLS: dict[str, str] = {}

SYMBOL_DECIMALS = {
    "GOLD":   2,
    "EURUSD": 5,
    "GBPUSD": 5,
    "USDCAD": 5,
}

# ================== ORDERBLOCK / FXPRO LIQUIDITY SETTINGS ==================
ORDERBLOCK_ENTRY_ENABLED = os.getenv("ORDERBLOCK_ENTRY", "1").strip().lower() in {"1", "true", "yes", "on"}
# RB M15 is retired. Keep the legacy symbol hard-false so stale environments
# cannot put it back into live entry arbitration.
REJECTION_BLOCK_ENTRY_ENABLED = False
# RB H1 is the primary live rejection-block trigger. Keep the environment
# switch as an emergency rollback, but enable it by default for new releases.
REJECTION_BLOCK_H1_ENTRY_ENABLED = os.getenv(
    "REJECTION_BLOCK_H1_ENTRY_ENABLED", "1"
).strip().lower() in {"1", "true", "yes", "on"}
FXPRO_DOM_CAPTURE_ENABLED = os.getenv(
    "FXPRO_DOM_CAPTURE_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
FXPRO_DOM_SYMBOLS = [
    item.strip().upper()
    for item in os.getenv(
        "FXPRO_DOM_SYMBOLS", "EURUSD,GBPUSD,USDCAD"
    ).split(",")
    if item.strip()
]
FXPRO_DOM_DATA_DIR = Path(
    os.getenv("FXPRO_DOM_DATA_DIR", "").strip()
    or str(AI_DATA_DIR / "fxpro_dom")
)
FXPRO_DOM_POLL_MS = _env_int("FXPRO_DOM_POLL_MS") or 500
FXPRO_DOM_MAX_LEVELS = _env_int("FXPRO_DOM_MAX_LEVELS") or 20
FXPRO_DOM_HEARTBEAT_SEC = _env_float("FXPRO_DOM_HEARTBEAT_SEC", 1.0)

# FxPro/Quantower Cluster Rejection is an opt-in broker-specific proxy.
# Historical diagnostic conversions are research assumptions and stay refused
# by live detection unless an isolated research runner overrides the guard.
FXPRO_CLUSTER_REJECTION_ENTRY_ENABLED = os.getenv(
    "FXPRO_CLUSTER_REJECTION_ENTRY_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
_FXPRO_CLUSTER_LIVE_SIDECAR_DIR = os.getenv(
    "FXPRO_CLUSTER_LIVE_SIDECAR_DIR", ""
).strip()
FXPRO_CLUSTER_LIVE_SIDECAR_DIR = (
    Path(_FXPRO_CLUSTER_LIVE_SIDECAR_DIR)
    if _FXPRO_CLUSTER_LIVE_SIDECAR_DIR
    else None
)
FXPRO_CLUSTER_ALLOW_RESEARCH_ASSUMPTION = os.getenv(
    "FXPRO_CLUSTER_ALLOW_RESEARCH_ASSUMPTION", "0"
).strip().lower() in {"1", "true", "yes", "on"}
FXPRO_CLUSTER_EDGE_FRACTION = _env_float(
    "FXPRO_CLUSTER_EDGE_FRACTION", 0.20
)
FXPRO_CLUSTER_MIN_CLASSIFIED_VOLUME = _env_float(
    "FXPRO_CLUSTER_MIN_CLASSIFIED_VOLUME", 20.0
)
FXPRO_CLUSTER_MIN_CLASSIFICATION_RATIO = _env_float(
    "FXPRO_CLUSTER_MIN_CLASSIFICATION_RATIO", 0.80
)
FXPRO_CLUSTER_MIN_EDGE_VOLUME_SHARE = _env_float(
    "FXPRO_CLUSTER_MIN_EDGE_VOLUME_SHARE", 0.12
)
FXPRO_CLUSTER_MIN_EDGE_IMBALANCE_RATIO = _env_float(
    "FXPRO_CLUSTER_MIN_EDGE_IMBALANCE_RATIO", 1.50
)
FXPRO_CLUSTER_MIN_REJECTION_FRACTION = _env_float(
    "FXPRO_CLUSTER_MIN_REJECTION_FRACTION", 0.60
)
FXPRO_CLUSTER_MIN_WICK_FRACTION = _env_float(
    "FXPRO_CLUSTER_MIN_WICK_FRACTION", 0.20
)
FXPRO_CLUSTER_MIN_SCORE = _env_float(
    "FXPRO_CLUSTER_MIN_SCORE", 0.60
)
FXPRO_CLUSTER_CANDLE_TOLERANCE_TICKS = (
    _env_int("FXPRO_CLUSTER_CANDLE_TOLERANCE_TICKS")
)
if FXPRO_CLUSTER_CANDLE_TOLERANCE_TICKS is None:
    FXPRO_CLUSTER_CANDLE_TOLERANCE_TICKS = 2

# In-process forward capture of the FxPro cluster proxy from MT5 bid ticks.
# Capture is deliberately independent of
# FXPRO_CLUSTER_REJECTION_ENTRY_ENABLED: it may run for weeks of shadow
# collection while entries stay disabled. Point
# FXPRO_CLUSTER_LIVE_SIDECAR_DIR at FXPRO_TICK_CLUSTER_SIDECAR_DIR only after
# the captured population has been reviewed.
FXPRO_TICK_CLUSTER_CAPTURE_ENABLED = os.getenv(
    "FXPRO_TICK_CLUSTER_CAPTURE_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
FXPRO_TICK_CLUSTER_SYMBOLS = [
    item.strip().upper()
    for item in os.getenv(
        "FXPRO_TICK_CLUSTER_SYMBOLS", "EURUSD,GBPUSD,USDCAD"
    ).split(",")
    if item.strip()
]
FXPRO_TICK_CLUSTER_DATA_DIR = Path(
    os.getenv("FXPRO_TICK_CLUSTER_DATA_DIR", "").strip()
    or str(AI_DATA_DIR / "fxpro_tick_cluster")
)
_FXPRO_TICK_CLUSTER_SIDECAR_DIR = os.getenv(
    "FXPRO_TICK_CLUSTER_SIDECAR_DIR", ""
).strip()
FXPRO_TICK_CLUSTER_SIDECAR_DIR = (
    Path(_FXPRO_TICK_CLUSTER_SIDECAR_DIR)
    if _FXPRO_TICK_CLUSTER_SIDECAR_DIR
    else FXPRO_TICK_CLUSTER_DATA_DIR / "sidecar"
)
FXPRO_TICK_CLUSTER_RETENTION_DAYS = (
    _env_int("FXPRO_TICK_CLUSTER_RETENTION_DAYS") or 2
)
FXPRO_TICK_CLUSTER_SETTLE_SEC = _env_float(
    "FXPRO_TICK_CLUSTER_SETTLE_SEC", 2.0
)
FXPRO_TICK_CLUSTER_POLL_SEC = _env_float(
    "FXPRO_TICK_CLUSTER_POLL_SEC", 20.0
)
FXPRO_TICK_CLUSTER_MAX_CATCHUP_BARS = (
    _env_int("FXPRO_TICK_CLUSTER_MAX_CATCHUP_BARS") or 4
)
FXPRO_TICK_CLUSTER_ARCHIVE_RAW = os.getenv(
    "FXPRO_TICK_CLUSTER_ARCHIVE_RAW", "1"
).strip().lower() in {"1", "true", "yes", "on"}

# FxPro Quote Pressure Rejection is opt-in. DOM capture may run while
# entries stay disabled until the broker-specific history passes causal WFO.
FXPRO_QUOTE_PRESSURE_REJECTION_ENTRY_ENABLED = os.getenv(
    "FXPRO_QUOTE_PRESSURE_REJECTION_ENTRY_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
FXPRO_LIQUIDITY_MIN_ABS_QUOTE_PRESSURE = _env_float(
    "FXPRO_LIQUIDITY_MIN_ABS_QUOTE_PRESSURE", 0.15
)
FXPRO_LIQUIDITY_MIN_REPLENISHMENT_RATIO = _env_float(
    "FXPRO_LIQUIDITY_MIN_REPLENISHMENT_RATIO", 0.50
)
FXPRO_LIQUIDITY_MIN_REPLENISHMENT_SHARE = _env_float(
    "FXPRO_LIQUIDITY_MIN_REPLENISHMENT_SHARE", 0.55
)
FXPRO_LIQUIDITY_MIN_REJECTION_FRACTION = _env_float(
    "FXPRO_LIQUIDITY_MIN_REJECTION_FRACTION", 0.60
)
FXPRO_LIQUIDITY_MAX_PRICE_RESPONSE_EFFICIENCY = _env_float(
    "FXPRO_LIQUIDITY_MAX_PRICE_RESPONSE_EFFICIENCY", 0.35
)
FXPRO_LIQUIDITY_MIN_COVERAGE_RATIO = _env_float(
    "FXPRO_LIQUIDITY_MIN_COVERAGE_RATIO", 0.98
)
FXPRO_LIQUIDITY_MAX_GAP_SECONDS = _env_float(
    "FXPRO_LIQUIDITY_MAX_GAP_SECONDS", 5.0
)
FXPRO_LIQUIDITY_MIN_CHANGED_SNAPSHOTS = (
    _env_int("FXPRO_LIQUIDITY_MIN_CHANGED_SNAPSHOTS") or 20
)
FXPRO_LIQUIDITY_MIN_BOOK_LEVELS = (
    _env_int("FXPRO_LIQUIDITY_MIN_BOOK_LEVELS") or 2
)
FXPRO_LIQUIDITY_MAX_MEAN_SPREAD_BPS = _env_float(
    "FXPRO_LIQUIDITY_MAX_MEAN_SPREAD_BPS", 5.0
)
ORDERBLOCK_TOUCH_ATR_K = _env_float("ORDERBLOCK_TOUCH_ATR_K", 0.15)
ORDERBLOCK_TOUCH_MIN_ABS = _env_float("ORDERBLOCK_TOUCH_MIN_ABS", 0.0005)
ORDERBLOCK_MAX_AGE_BARS = _env_int("ORDERBLOCK_MAX_AGE_BARS") or 80

# ================== HTF SCORING ==================
HTF_SCORE_MARGIN = int(os.getenv("HTF_SCORE_MARGIN", "2"))

# ================== SHADOW SCORE (DIAGNOSTIC ONLY) ==================
# Path to a frozen ``shadow_models.json`` produced by an offline WFO run. When
# set, the newest fitted fold is replayed against live ENTER signals and its
# predicted expected R is journalled beside the trade.
#
# The shadow score must remain non-executing: it never rejects an entry, never
# changes direction, entry, stop, targets, disposition, setup count, or risk.
# Leave unset to disable the annotation entirely.
_SHADOW_SCORE_MODEL_PATH = os.getenv("SHADOW_SCORE_MODEL_PATH", "").strip()
SHADOW_SCORE_MODEL_PATH = (
    Path(_SHADOW_SCORE_MODEL_PATH) if _SHADOW_SCORE_MODEL_PATH else None
)

# ================== ENTRY FREQUENCY / RISK BRAKES ==================
# Cooldown (minutes) per symbol after a position is closed by stop-loss.
# Blocks the 2-3 minute revenge re-entries seen on 2026-07-10.
POST_SL_COOLDOWN_MIN = _env_int("POST_SL_COOLDOWN_MIN") or 60
# Hard cap of executed setups per symbol per day (counter resets at UTC midnight).
MAX_SETUPS_PER_SYMBOL_PER_DAY = _env_int("MAX_SETUPS_PER_SYMBOL_PER_DAY") or 3
# Bot-wide daily loss limit as a fraction of the day's starting balance.
# When equity drops below balance*(1-limit), new entries stop until next day.
DAILY_MAX_LOSS_PCT = _env_float("DAILY_MAX_LOSS_PCT", 0.03)

# ================== VOL REGIME FILTER (IV Surface port) ==================
# 7-signal volatility-surface score R(t) 0-100 computed from the symbol's own
# daily candles (port of the GOLD IV Surface indicator). Entries are blocked
# while R(t) >= VOL_REGIME_MAX_R (PANIC regime). Independently, EM_TP_MAX_RATIO
# blocks setups whose TP1 is further than N x the IV-implied 1-day expected
# move (unreachable before the daily flat close). Set ratio/threshold to 0 to
# disable that half of the filter.
VOL_REGIME_FILTER_ENABLED = os.getenv("VOL_REGIME_FILTER", "1").strip().lower() in {
    "1", "true", "yes", "on"
}
VOL_REGIME_SYMBOLS = [
    s.strip().upper()
    for s in os.getenv("VOL_REGIME_SYMBOLS", "GOLD,EURUSD,GBPUSD,USDCAD").split(",")
    if s.strip()
]
VOL_REGIME_MAX_R = _env_float("VOL_REGIME_MAX_R", 60.0)
EM_TP_MAX_RATIO = _env_float("EM_TP_MAX_RATIO", 1.0)
# Recompute the vol context at most this often per symbol (RV moves slowly).
VOL_REGIME_REFRESH_MIN = _env_int("VOL_REGIME_REFRESH_MIN") or 15

# ================== POSITION ADDING (PYRAMIDING) ==================
# Stage extra entries into a working idea instead of one all-in setup.  All
# split legs and all simultaneously risking entries share one fixed-capital
# 1% budget; older entries must be moved to break-even before an add-on can
# reuse the released risk.
# Opt-in: an upgraded bot must not start pyramiding on a VPS by itself.
POSITION_ADDING_ENABLED = os.getenv("POSITION_ADDING_ENABLED", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
# Hard ceiling on entries per idea (methodology allows 2-3).
IDEA_MAX_ENTRIES = max(1, min(5, _env_int("IDEA_MAX_ENTRIES") or 3))
# Aggregate risk of every simultaneously risking entry in one idea. Values
# above 1% are unconditionally clamped to the same hard setup-level ceiling.
IDEA_MAX_RISK_PCT = min(
    max(_env_float("IDEA_MAX_RISK_PCT", 0.01), 0.0),
    0.01,
)
# An add-on needs the idea to have proven itself: price must be at least this
# many R (of the previous entry) in profit before another entry is allowed.
ADDON_MIN_PROGRESS_R = max(0.0, _env_float("ADDON_MIN_PROGRESS_R", 0.5))
# Kolachi pyramiding rule: past this share of the way from entry 1 to the final
# TP the remaining reward no longer justifies a new entry's risk.
ADDON_MAX_PROGRESS_PCT = min(max(_env_float("ADDON_MAX_PROGRESS_PCT", 0.5), 0.0), 1.0)

# ================== EXECUTION SIZING GUARDS ==================
# Cap on total volume (lots) per setup, applied on top of the broker maximum.
MT5_MAX_VOLUME = _env_float("MT5_MAX_VOLUME", 10.0)
# Round-turn commission per 1.0 lot in the account currency — included in sizing so a
# tight stop cannot balloon volume past the planned risk.
MT5_COMMISSION_PER_LOT = _env_float("MT5_COMMISSION_PER_LOT", 7.0)

# ================== FRIDAY WEEKEND CLOSE ==================
# On Friday at/after this hour (Europe/Moscow, UTC+3 no DST) the bot blocks
# new entries and force-closes all open positions before the weekend.
FRIDAY_CLOSE_HOUR = _env_int("FRIDAY_CLOSE_HOUR") or 21

# ================== DAILY FLAT CLOSE ==================
# Every day at/after this hour (Europe/Moscow, UTC+3 no DST) the bot blocks
# new entries and force-closes all open positions — no positions held past
# this time. This experiment is opt-in; it must not silently change an existing
# VPS schedule just because the code was upgraded.
DAILY_FLAT_ENABLED = os.getenv("DAILY_FLAT_ENABLED", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
_daily_close_hour = _env_int("DAILY_CLOSE_HOUR")
DAILY_CLOSE_HOUR = _daily_close_hour if _daily_close_hour is not None and 0 <= _daily_close_hour <= 23 else 21
DAILY_CLOSE_BUFFER_MIN = max(0, min(180, _env_int("DAILY_CLOSE_BUFFER_MIN") or 30))

# ================== CORRELATION GUARD ==================
# Groups of correlated symbols: while one symbol of a group has an open trade,
# a same-direction entry on another symbol of that group is blocked (the two
# would effectively double the risk on a single idea).
# Format: "EURUSD+GBPUSD,AUDUSD+NZDUSD"
CORRELATED_GROUPS = [
    [s.strip().upper() for s in grp.split("+") if s.strip()]
    for grp in os.getenv("CORRELATED_GROUPS", "").split(",")
    if grp.strip()
]

# ================== SESSIONS / FILTERS ==================
SESSION_WINDOWS = {
    "ASIA":   ("00:00", "08:00"),
    "LONDON": ("06:30", "15:30"),
    "NY":     ("12:00", "21:00"),
}
SESSION_TIMEZONE = os.getenv("SESSION_TIMEZONE", "UTC")
ALLOWED_SESSIONS = [s.strip().upper() for s in os.getenv("ALLOWED_SESSIONS", "LONDON,NY").split(",") if s.strip()]

# ================== PERSISTENCE / REPORTS ==================
POST_STARTUP_REPORT = os.getenv("POST_STARTUP_REPORT", "0").strip() in ("1", "true", "True", "yes", "YES")
REPORT_DEFAULT_LIMIT = int(os.getenv("REPORT_DEFAULT_LIMIT", "15"))
LOG_TICK = os.getenv("LOG_TICK", "1").strip() in ("1", "true", "True", "yes", "YES")

DEBUG_RAW_SIGNALS = os.getenv("DEBUG_RAW_SIGNALS", "0").strip().lower() in {"1", "true", "yes", "on"}

# ================== PARTIAL TP MODE ==================
# "split"   → MK-style: N sub-positions, each with its own broker TP. Broker closes each leg.
# "monitor" → legacy: one position, bot monitors and closes partially via market orders.
PARTIAL_TP_MODE = os.getenv("PARTIAL_TP_MODE", "split").strip().lower()

# Move SL to break-even (entry price) once TP1 is hit:
#   split mode   → when the first leg is closed by the broker
#   monitor mode → after the first partial close
MOVE_BE_AFTER_TP1 = os.getenv("MOVE_BE_AFTER_TP1", "1").strip().lower() in {"1", "true", "yes", "on"}

# Run the strategy on CLOSED candles only (drop the still-forming last bar).
# Prevents repaint: a trigger that appears mid-bar can vanish by bar close.
SIGNAL_ON_CLOSED_BARS = os.getenv("SIGNAL_ON_CLOSED_BARS", "1").strip().lower() in {"1", "true", "yes", "on"}
