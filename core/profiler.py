from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import DefaultDict, Dict


class TickProfiler:
    """Lightweight per-tick profiler.

    Enabled via BOT_PROFILING=1 (or pass enabled=True explicitly). When enabled,
    use ``with profiler.section("name")`` or ``profiler.add("name", dt)`` to
    accumulate timings. Call ``dump()`` to print aggregated stats and reset.
    """

    def __init__(self, enabled: bool | None = None) -> None:
        if enabled is None:
            env_flag = os.getenv("BOT_PROFILING", "0")
            enabled = bool(int(env_flag)) if env_flag.isdigit() else False
        self.enabled = enabled
        self._samples: DefaultDict[str, float] = defaultdict(float)
        self._counts: DefaultDict[str, int] = defaultdict(int)

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self._samples[name] += time.perf_counter() - start
            self._counts[name] += 1

    def add(self, name: str, duration: float) -> None:
        if not self.enabled:
            return
        self._samples[name] += duration
        self._counts[name] += 1

    def dump(self, prefix: str = "[Profiler]") -> None:
        if not self.enabled or not self._samples:
            return
        parts = []
        for name in sorted(self._samples.keys()):
            total = self._samples[name]
            count = self._counts[name]
            avg = total / max(count, 1)
            parts.append(
                f"{name}={total * 1000:.1f}ms(avg {avg * 1000:.1f}ms, n={count})"
            )
        print(f"{prefix} {' | '.join(parts)}")
        self._samples.clear()
        self._counts.clear()

    def merge(self, other_stats: Dict[str, float]) -> None:
        """Optional: merge external timings (name -> seconds)."""
        if not self.enabled:
            return
        for name, value in other_stats.items():
            self.add(name, value)
