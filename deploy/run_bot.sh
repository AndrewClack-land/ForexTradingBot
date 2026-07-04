#!/bin/bash
# Forex bot launcher for a headless Linux VPS.
# MetaTrader5 is Windows-only software: the terminal and the Windows Python
# that runs the bot both live inside a Wine prefix, with Xvfb providing the
# virtual display the MT5 terminal needs.
set -e

export WINEPREFIX="${WINEPREFIX:-$HOME/.mt5}"
export WINEDEBUG=-all
export DISPLAY=:99
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1

BOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MT5_EXE="$WINEPREFIX/drive_c/Program Files/MetaTrader 5/terminal64.exe"
PY_EXE="$WINEPREFIX/drive_c/Python311/python.exe"

cd "$BOT_DIR"

# Virtual display for the MT5 terminal GUI
if ! pgrep -f "Xvfb :99" > /dev/null; then
    Xvfb :99 -screen 0 1280x800x24 &
    sleep 2
fi

# The MetaTrader5 Python package can start the terminal itself, but a
# pre-started terminal makes initialize() faster and more reliable.
if ! pgrep -f "terminal64.exe" > /dev/null; then
    wine "$MT5_EXE" > /dev/null 2>&1 &
    sleep 25
fi

exec wine "$PY_EXE" main.py
