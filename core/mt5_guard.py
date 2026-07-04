# core/mt5_guard.py
"""Serialize all MetaTrader5 API calls across threads.

Three threads talk to the terminal concurrently: the Telegram tick loop
(executor + DataFeed direct fetch), the DataCacheLoop refresher, and the
MT5NativeBridge poller. The MetaTrader5 package is not documented as
thread-safe — interleaved calls can garble results and last_error().

install() wraps every public callable of the MetaTrader5 module with a shared
re-entrant lock. It patches the module object itself, so every importer
(``import MetaTrader5 as mt5`` anywhere in the codebase) is covered without
touching call sites. Idempotent — safe to call more than once.

Limitation: ``last_error()`` is only meaningful for the failing call if no
other thread ran an MT5 call in between; the lock makes each single call
atomic but not call+last_error pairs. That matches how the codebase uses it
(diagnostic logging only).
"""
from __future__ import annotations

import functools
import threading

import MetaTrader5 as mt5

MT5_LOCK = threading.RLock()

_installed = False


def _wrap(fn):
    @functools.wraps(fn)
    def _locked(*args, **kwargs):
        with MT5_LOCK:
            return fn(*args, **kwargs)

    _locked.__mt5_guarded__ = True
    return _locked


def install() -> None:
    global _installed
    if _installed:
        return
    for name in dir(mt5):
        if name.startswith("_"):
            continue
        attr = getattr(mt5, name)
        if not callable(attr) or isinstance(attr, type):
            continue
        if getattr(attr, "__mt5_guarded__", False):
            continue
        setattr(mt5, name, _wrap(attr))
    _installed = True
