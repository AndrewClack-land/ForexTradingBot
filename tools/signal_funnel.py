"""Read-only entry-funnel diagnostic over the live bot's signal log.

The bot prints one ``[Ticker] SYMBOL SIGNAL | ...`` line per polled tick per
symbol (``Core._log_signal``). Those lines already encode the full entry
funnel — bias, trigger, AI filter, context filters, per-symbol guards,
execution — but as one line per *tick*, not per decision. Counting them raw
overstates every stage by the polling rate.

This tool collapses ticks into decision slots (default: the M15 slot the bot
actually decides on) and reports, per slot, the deepest funnel stage reached.
That yields an honest "where do decisions die" histogram plus conversion rates
between stages.

Strictly diagnostic: it reads logs, never the strategy, broker, or journal
state, and changes nothing.

Usage::

    # on the VPS, current live history
    journalctl -u forexbot.service --no-pager -o short-iso \\
        | python3 -m tools.signal_funnel --since 2026-07-20

    # locally, against a captured file
    python -m tools.signal_funnel --input ai_data/bot.log --by day

Timestamps are required for slot collapsing and day grouping; run
``journalctl`` with ``-o short-iso`` (not ``-o cat``). Without them the tool
still reports raw tick counts and says so.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Funnel model
# ---------------------------------------------------------------------------

# Ordered entry-funnel stages. A decision that reaches stage N necessarily
# passed every earlier stage, so the deepest stage seen inside one slot is the
# stage the decision actually died at. Keep this table in sync with the signal
# vocabulary in ``main.py`` / ``core/m1/ai_live.py``.
STAGES: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("feed", "нет данных фида", ("NO_DATA", "ERROR")),
    ("bias", "bias NEUTRAL (score margin)", ("NO_TREND",)),
    ("trigger", "нет триггера входа", ("NO_TRIGGER",)),
    ("ai", "AI-фильтр p(TP)", ("AI_REJECT",)),
    (
        "context",
        "сессия / волатильность / бюджет",
        (
            "WAIT_SESSION",
            "WAIT_RISK",
            "SKIP_VOL_REGIME",
            "SKIP_EM_TP",
            "SKIP_BUDGET",
        ),
    ),
    (
        "guards",
        "частотные и риск-тормоза",
        (
            "SKIP_NO_SIDE",
            "WAIT_COOLDOWN",
            "SKIP_DAILY_LIMIT",
            "SKIP_DUP_TRIGGER",
            "SKIP_HEDGE",
            "SKIP_CORRELATED",
        ),
    ),
    ("execution", "исполнение", ("WAIT_RISK_ENTRY", "EXECUTION_ERROR")),
    ("filled", "ВХОД ОТКРЫТ", ("ENTER",)),
)

STAGE_ORDER: Dict[str, int] = {name: i for i, (name, _, _) in enumerate(STAGES)}
STAGE_LABEL: Dict[str, str] = {name: label for name, label, _ in STAGES}

SIGNAL_STAGE: Dict[str, str] = {
    signal: name for name, _, signals in STAGES for signal in signals
}

# Position-lifecycle signals. These are not entry decisions: they say the
# symbol's capacity was already consumed by an open idea, which is itself a
# frequency constraint but a different one.
LIFECYCLE_SIGNALS = frozenset(
    {
        "HOLD",
        "EXIT_TP",
        "EXIT_SL",
        "EXIT_TIME",
        "EXIT_BROKER",
        "POSITION_ADD",
    }
)

_TICKER_RE = re.compile(
    r"""
    ^
    (?:                                     # optional journald/log prefix
        (?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}
            (?:\.\d+)?
            (?:Z|[+-]\d{2}:?\d{2})?
        )
        \s+ .*?                             # host, unit[pid]:
    )?
    \[Ticker\]\s+
    (?P<symbol>\S+)\s+
    (?P<signal>\S+)
    (?P<rest>.*)
    $
    """,
    re.VERBOSE,
)

_EXEC_FAIL_RE = re.compile(
    r"""
    ^
    (?:
        (?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}
            (?:\.\d+)?
            (?:Z|[+-]\d{2}:?\d{2})?
        )
        \s+ .*?
    )?
    \[Core\]\s+Execution\s+failed\s+for\s+(?P<symbol>\S+?):\s*(?P<reason>.+)
    $
    """,
    re.VERBOSE,
)

# Numbers/prices differ on every occurrence; collapse them so reason families
# aggregate instead of producing one bucket per price.
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _parse_ts(raw: str) -> Optional[datetime]:
    text = raw.strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # journalctl short-iso emits +0000; fromisoformat wants +00:00
    if re.search(r"[+-]\d{4}$", text):
        text = text[:-5] + text[-5:-2] + ":" + text[-2:]
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _slot_start(ts: datetime, slot_minutes: int) -> datetime:
    minute = (ts.minute // slot_minutes) * slot_minutes
    return ts.replace(minute=minute, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TickerLine:
    ts: Optional[datetime]
    symbol: str
    signal: str
    fields: Dict[str, str]


def parse_ticker_line(line: str) -> Optional[TickerLine]:
    """Parse one ``[Ticker]`` log line, or return None for anything else."""
    match = _TICKER_RE.match(line.strip())
    if match is None:
        return None
    raw_ts = match.group("ts")
    ts = _parse_ts(raw_ts) if raw_ts else None
    fields: Dict[str, str] = {}
    for chunk in match.group("rest").split(" | "):
        key, sep, value = chunk.partition("=")
        if sep:
            fields[key.strip()] = value.strip()
    return TickerLine(
        ts=ts,
        symbol=match.group("symbol"),
        signal=match.group("signal"),
        fields=fields,
    )


def parse_exec_failure(line: str) -> Optional[Tuple[Optional[datetime], str, str]]:
    """Parse a ``[Core] Execution failed for X: reason`` line into a family."""
    match = _EXEC_FAIL_RE.match(line.strip())
    if match is None:
        return None
    raw_ts = match.group("ts")
    ts = _parse_ts(raw_ts) if raw_ts else None
    reason = _NUM_RE.sub("N", match.group("reason")).strip()
    return ts, match.group("symbol"), reason


def read_lines(path: Optional[Path]) -> Iterator[str]:
    """Yield log lines from a file or stdin, tolerating the encodings the bot's
    own launchers produce (PowerShell redirects ``bot.log`` as UTF-16)."""
    if path is None:
        yield from sys.stdin
        return
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-16", "cp1251"):
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
        if "[Ticker]" in text or encoding == "cp1251":
            yield from text.splitlines()
            return
    yield from raw.decode("utf-8", errors="replace").splitlines()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class Slot:
    """One decision slot for one symbol: the deepest stage its ticks reached."""

    stage_index: int = -1
    signal: str = ""
    lifecycle: bool = False

    def observe(self, signal: str) -> None:
        if signal in LIFECYCLE_SIGNALS:
            self.lifecycle = True
            return
        index = STAGE_ORDER.get(SIGNAL_STAGE.get(signal, ""), None)
        if index is None:
            return
        if index > self.stage_index:
            self.stage_index = index
            self.signal = signal


@dataclass
class FunnelResult:
    slot_minutes: int
    lines_seen: int = 0
    ticker_lines: int = 0
    undated_lines: int = 0
    raw_signals: Counter = field(default_factory=Counter)
    unknown_signals: Counter = field(default_factory=Counter)
    slots: Dict[Tuple[str, datetime], Slot] = field(default_factory=dict)
    exec_reasons: Counter = field(default_factory=Counter)
    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None

    # -- derived views -----------------------------------------------------

    def stage_deaths(self, symbol: Optional[str] = None) -> Counter:
        counts: Counter = Counter()
        for (sym, _), slot in self.slots.items():
            if symbol is not None and sym != symbol:
                continue
            if slot.stage_index < 0:
                continue
            counts[STAGES[slot.stage_index][0]] += 1
        return counts

    def signal_deaths(self, stage: str) -> Counter:
        counts: Counter = Counter()
        for slot in self.slots.values():
            if slot.stage_index < 0:
                continue
            if STAGES[slot.stage_index][0] == stage:
                counts[slot.signal] += 1
        return counts

    def symbols(self) -> List[str]:
        return sorted({sym for sym, _ in self.slots})

    def occupied_slots(self) -> int:
        return sum(1 for slot in self.slots.values() if slot.lifecycle)


def build_funnel(
    lines: Iterable[str],
    *,
    slot_minutes: int = 15,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    symbols: Optional[Sequence[str]] = None,
) -> FunnelResult:
    result = FunnelResult(slot_minutes=slot_minutes)
    wanted = {s.upper() for s in symbols} if symbols else None

    for line in lines:
        result.lines_seen += 1

        failure = parse_exec_failure(line)
        if failure is not None:
            ts, symbol, reason = failure
            if _in_window(ts, since, until) and (
                wanted is None or symbol.upper() in wanted
            ):
                result.exec_reasons[reason] += 1
            continue

        parsed = parse_ticker_line(line)
        if parsed is None:
            continue
        if wanted is not None and parsed.symbol.upper() not in wanted:
            continue
        if not _in_window(parsed.ts, since, until):
            continue

        result.ticker_lines += 1
        result.raw_signals[parsed.signal] += 1
        if (
            parsed.signal not in SIGNAL_STAGE
            and parsed.signal not in LIFECYCLE_SIGNALS
        ):
            result.unknown_signals[parsed.signal] += 1

        if parsed.ts is None:
            result.undated_lines += 1
            continue

        if result.first_ts is None or parsed.ts < result.first_ts:
            result.first_ts = parsed.ts
        if result.last_ts is None or parsed.ts > result.last_ts:
            result.last_ts = parsed.ts

        key = (parsed.symbol, _slot_start(parsed.ts, slot_minutes))
        result.slots.setdefault(key, Slot()).observe(parsed.signal)

    return result


def _in_window(
    ts: Optional[datetime],
    since: Optional[datetime],
    until: Optional[datetime],
) -> bool:
    if ts is None:
        # An undated line cannot be excluded by a window it may well fall in;
        # keep it for raw counts and let the report flag the ambiguity.
        return since is None and until is None
    if since is not None and ts < since:
        return False
    if until is not None and ts > until:
        return False
    return True


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _pct(part: int, whole: int) -> str:
    if whole <= 0:
        return "   —  "
    return f"{100.0 * part / whole:5.1f}%"


def render_report(result: FunnelResult, *, by: str = "none") -> str:
    out: List[str] = []
    add = out.append

    add("=" * 78)
    add("ВОРОНКА ВХОДОВ")
    add("=" * 78)
    if result.first_ts and result.last_ts:
        add(f"окно            : {result.first_ts:%Y-%m-%d %H:%M} → {result.last_ts:%Y-%m-%d %H:%M} UTC")
    add(f"строк прочитано : {result.lines_seen}  (из них [Ticker]: {result.ticker_lines})")
    add(f"слотов решений  : {len(result.slots)}  (по {result.slot_minutes} мин на символ)")
    if result.undated_lines:
        add(
            f"ВНИМАНИЕ        : {result.undated_lines} строк без времени — "
            "слоты по ним не считались (нужен journalctl -o short-iso)"
        )
    if result.unknown_signals:
        codes = ", ".join(f"{k}×{v}" for k, v in result.unknown_signals.most_common(8))
        add(
            "ВНИМАНИЕ        : неизвестные коды сигналов — окно охватывает "
            f"другую версию кода: {codes}"
        )
    add("")

    deaths = result.stage_deaths()
    total = sum(deaths.values())
    add(f"{'стадия':<12} {'умерло':>8} {'доля':>7} {'дошло далее':>12} {'конверсия':>10}   что это")
    add("-" * 78)
    remaining = total
    for name, label, _ in STAGES:
        died = deaths.get(name, 0)
        survived = remaining - died
        if name == "filled":
            add(f"{name:<12} {died:>8} {_pct(died, total):>7} {'':>12} {'':>10}   {label}")
            break
        add(
            f"{name:<12} {died:>8} {_pct(died, total):>7} {survived:>12} "
            f"{_pct(survived, remaining):>10}   {label}"
        )
        remaining = survived
    add("-" * 78)
    add(f"{'ИТОГО':<12} {total:>8}")
    add("")

    for stage_name in ("context", "guards", "execution"):
        detail = result.signal_deaths(stage_name)
        if detail:
            codes = ", ".join(f"{k}={v}" for k, v in detail.most_common())
            add(f"  {stage_name:<10}: {codes}")
    if result.occupied_slots():
        add(
            f"  слотов с открытой позицией (символ занят): "
            f"{result.occupied_slots()}"
        )
    add("")

    if result.exec_reasons:
        add("ПРИЧИНЫ ОТКАЗА НА ИСПОЛНЕНИИ (тиковые повторы, не уникальные сетапы)")
        add("-" * 78)
        for reason, count in result.exec_reasons.most_common(10):
            add(f"  {count:>6}  {reason[:66]}")
        add("")

    if by == "symbol":
        add("ПО СИМВОЛАМ")
        add("-" * 78)
        header = f"{'символ':<10}" + "".join(f"{n[:9]:>11}" for n, _, _ in STAGES)
        add(header)
        for symbol in result.symbols():
            per = result.stage_deaths(symbol)
            row = f"{symbol:<10}" + "".join(
                f"{per.get(n, 0):>11}" for n, _, _ in STAGES
            )
            add(row)
        add("")
    elif by == "day":
        add("ПО ДНЯМ (слоты, дошедшие до стадии)")
        add("-" * 78)
        per_day: Dict[str, Counter] = defaultdict(Counter)
        for (_, slot_ts), slot in result.slots.items():
            if slot.stage_index < 0:
                continue
            per_day[f"{slot_ts:%Y-%m-%d}"][STAGES[slot.stage_index][0]] += 1
        add(f"{'дата':<12}" + "".join(f"{n[:9]:>11}" for n, _, _ in STAGES))
        for day in sorted(per_day):
            counts = per_day[day]
            add(f"{day:<12}" + "".join(f"{counts.get(n, 0):>11}" for n, _, _ in STAGES))
        add("")

    return "\n".join(out)


def result_to_dict(result: FunnelResult) -> Dict[str, object]:
    per_symbol = {
        symbol: dict(result.stage_deaths(symbol)) for symbol in result.symbols()
    }
    return {
        "window_start": result.first_ts.isoformat() if result.first_ts else None,
        "window_end": result.last_ts.isoformat() if result.last_ts else None,
        "slot_minutes": result.slot_minutes,
        "lines_seen": result.lines_seen,
        "ticker_lines": result.ticker_lines,
        "undated_lines": result.undated_lines,
        "decision_slots": len(result.slots),
        "occupied_slots": result.occupied_slots(),
        "stage_deaths": dict(result.stage_deaths()),
        "stage_deaths_by_symbol": per_symbol,
        "raw_signal_counts": dict(result.raw_signals),
        "unknown_signals": dict(result.unknown_signals),
        "execution_failure_reasons": dict(result.exec_reasons),
        "stage_labels": {name: STAGE_LABEL[name] for name, _, _ in STAGES},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_boundary(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    parsed = _parse_ts(raw if "T" in raw or " " in raw else raw + "T00:00:00")
    if parsed is None:
        raise SystemExit(f"не разобрал дату: {raw!r}")
    return parsed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.signal_funnel",
        description="Воронка входов из лога живого бота (только чтение).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="файл лога; по умолчанию читает stdin",
    )
    parser.add_argument("--since", default=None, help="начало окна, YYYY-MM-DD[THH:MM]")
    parser.add_argument("--until", default=None, help="конец окна, YYYY-MM-DD[THH:MM]")
    parser.add_argument(
        "--symbol",
        action="append",
        default=None,
        help="ограничить символом (можно повторять)",
    )
    parser.add_argument(
        "--slot-minutes",
        type=int,
        default=15,
        help="длительность слота решения (по умолчанию 15 = M15)",
    )
    parser.add_argument(
        "--by",
        choices=("none", "symbol", "day"),
        default="symbol",
        help="дополнительная разбивка",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="дополнительно записать машинный отчёт в JSON",
    )
    args = parser.parse_args(argv)

    if args.slot_minutes < 1 or args.slot_minutes > 60 or 60 % args.slot_minutes:
        raise SystemExit("--slot-minutes должен быть делителем 60 в диапазоне 1..60")

    result = build_funnel(
        read_lines(args.input),
        slot_minutes=args.slot_minutes,
        since=_parse_boundary(args.since),
        until=_parse_boundary(args.until),
        symbols=args.symbol,
    )

    if result.ticker_lines == 0:
        print("Не найдено ни одной строки [Ticker] — проверьте источник лога.")
        return 1

    print(render_report(result, by=args.by))

    if args.json is not None:
        args.json.write_text(
            json.dumps(result_to_dict(result), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"JSON: {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
