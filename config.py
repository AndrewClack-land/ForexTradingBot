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

# ================== ORDERBLOCK SETTINGS ==================
ORDERBLOCK_ENTRY_ENABLED = os.getenv("ORDERBLOCK_ENTRY", "1").strip().lower() in {"1", "true", "yes", "on"}
ORDERBLOCK_TOUCH_ATR_K = _env_float("ORDERBLOCK_TOUCH_ATR_K", 0.15)
ORDERBLOCK_TOUCH_MIN_ABS = _env_float("ORDERBLOCK_TOUCH_MIN_ABS", 0.0005)
ORDERBLOCK_MAX_AGE_BARS = _env_int("ORDERBLOCK_MAX_AGE_BARS") or 80

# ================== HTF SCORING ==================
HTF_SCORE_MARGIN = int(os.getenv("HTF_SCORE_MARGIN", "2"))

# ================== FRIDAY WEEKEND CLOSE ==================
# On Friday at/after this hour (Europe/Moscow, UTC+3 no DST) the bot blocks
# new entries and force-closes all open positions before the weekend.
FRIDAY_CLOSE_HOUR = _env_int("FRIDAY_CLOSE_HOUR") or 21

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
