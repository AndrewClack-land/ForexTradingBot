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

# Virtual display :99 is provided by xvfb99.service (see deploy/xvfb99.service).
# Keeping Xvfb outside this unit means a bot restart never tears down the
# display under the MT5 terminal.

# IMPORTANT: do NOT pre-start terminal64.exe here. mt5.initialize() must launch
# the terminal itself — it passes hidden command-line flags that enable the
# Python API IPC. A manually started terminal has no API enabled, and its
# single-instance guard prevents the package from starting its own, so
# initialize() times out. Kill any stray instance instead.
# pkill -x does NOT match Wine processes (their comm is not the exe name), so
# match the full command line; -U keeps Docker containers' terminals alive.
pkill -9 -U "$(id -u)" -f 'terminal64\.exe' 2>/dev/null && sleep 4

# The terminal must already know the broker's server (Config/servers.dat,
# copied from a working Windows install). Without it initialize() dies with
# 'IPC timeout' — see gmag11/MetaTrader5-Docker#15.
SERVERS_DAT="$WINEPREFIX/drive_c/Program Files/MetaTrader 5/Config/servers.dat"
if [ ! -f "$SERVERS_DAT" ]; then
    echo "FATAL: $SERVERS_DAT is missing — initialize() would hang with IPC timeout." >&2
    exit 1
fi

# Stale PID-lock from the previous run: Wine assigns small Windows PIDs that
# collide with system processes after restart, so the psutil check in
# _acquire_pid_lock misfires. systemd already guarantees a single instance.
rm -f "$BOT_DIR/ai_data/bot.pid"

exec wine "$PY_EXE" main.py
