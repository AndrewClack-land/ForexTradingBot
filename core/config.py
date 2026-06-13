from __future__ import annotations

# This file re-exports everything from the root config so that code inside the
# core/ package can do `from config import ...` and get the same symbols whether
# run from the root or from within core/.
#
# TradingView has been removed.  All market data comes from MetaTrader 5.

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent   # project root
AI_DATA_DIR = BASE_DIR / "ai_data"
AI_DATA_DIR.mkdir(parents=True, exist_ok=True)

MT5_CACHE_DIR = AI_DATA_DIR / "mt5_cache"
MT5_CACHE_DIR.mkdir(parents=True, exist_ok=True)

MT5_EXECUTION_ENABLED = os.getenv("MT5_EXECUTION", "0").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str):
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
MT5_RISK_PER_TRADE = _env_float("MT5_RISK_PCT", 0.01)
MT5_SLIPPAGE = _env_int("MT5_SLIPPAGE") or 20

if MT5_EXECUTION_ENABLED and (MT5_LOGIN is None or not MT5_PASSWORD or not MT5_SERVER):
    MT5_EXECUTION_ENABLED = False

MT5_BRIDGE_SYMBOLS = os.getenv("MT5_BRIDGE_SYMBOLS", "EURUSD,GBPUSD,USDCAD,GOLD")
MT5_BRIDGE_TIMEFRAMES = os.getenv("MT5_BRIDGE_TIMEFRAMES", "1m,5m,15m,1h,4h,1d")
MT5_BRIDGE_LOOKBACK_DAYS = _env_int("MT5_BRIDGE_LOOKBACK_DAYS") or 15
MT5_BRIDGE_INTERVAL = _env_int("MT5_BRIDGE_INTERVAL") or 60

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "CHANGE_ME_TELEGRAM_TOKEN")
TELEGRAM_CHANNEL_ID = int(os.getenv("TELEGRAM_CHANNEL_ID", "-1003871620174"))

UNIVERSE = {
    "GOLD":   "GOLD",
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDCAD": "USDCAD",
}

CONTEXT_SYMBOLS: dict = {}

SYMBOL_DECIMALS = {
    "GOLD":   2,
    "EURUSD": 5,
    "GBPUSD": 5,
    "USDCAD": 5,
}

SESSION_WINDOWS = {
    "ASIA":   ("00:00", "08:00"),
    "LONDON": ("06:30", "15:30"),
    "NY":     ("12:00", "21:00"),
}
SESSION_TIMEZONE = os.getenv("SESSION_TIMEZONE", "UTC")
ALLOWED_SESSIONS = [s.strip().upper() for s in os.getenv("ALLOWED_SESSIONS", "LONDON,NY").split(",") if s.strip()]

POST_STARTUP_REPORT = os.getenv("POST_STARTUP_REPORT", "0").strip() in ("1", "true", "True", "yes", "YES")
REPORT_DEFAULT_LIMIT = int(os.getenv("REPORT_DEFAULT_LIMIT", "15"))
LOG_TICK = os.getenv("LOG_TICK", "1").strip() in ("1", "true", "True", "yes", "YES")
DEBUG_RAW_SIGNALS = os.getenv("DEBUG_RAW_SIGNALS", "0").strip().lower() in {"1", "true", "yes", "on"}

ORDERBLOCK_ENTRY_ENABLED = os.getenv("ORDERBLOCK_ENTRY", "1").strip().lower() in {"1", "true", "yes", "on"}
ORDERBLOCK_TOUCH_ATR_K = _env_float("ORDERBLOCK_TOUCH_ATR_K", 0.15)
ORDERBLOCK_TOUCH_MIN_ABS = _env_float("ORDERBLOCK_TOUCH_MIN_ABS", 0.0005)
ORDERBLOCK_MAX_AGE_BARS = _env_int("ORDERBLOCK_MAX_AGE_BARS") or 80

HTF_SCORE_MARGIN = int(os.getenv("HTF_SCORE_MARGIN", "2"))

PARTIAL_TP_MODE = os.getenv("PARTIAL_TP_MODE", "split").strip().lower()
