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
PY_EXE="$WINEPREFIX/drive_c/Python311/python.exe"

cd "$BOT_DIR"

# Virtual display for the MT5 terminal GUI
if ! pgrep -f "Xvfb :99" > /dev/null; then
    Xvfb :99 -screen 0 1280x800x24 &
    sleep 2
fi

# IMPORTANT: do NOT pre-start terminal64.exe here. mt5.initialize() must launch
# the terminal itself — it passes hidden command-line flags that enable the
# Python API IPC. A manually started terminal has no API enabled, and its
# single-instance guard prevents the package from starting its own, so
# initialize() times out. Kill any stray instance instead.
pkill -x terminal64.exe 2>/dev/null && sleep 3

# Stale PID-lock from the previous run: Wine assigns small Windows PIDs that
# collide with system processes after restart, so the psutil check in
# _acquire_pid_lock misfires. systemd already guarantees a single instance.
rm -f "$BOT_DIR/ai_data/bot.pid"

exec wine "$PY_EXE" main.py
