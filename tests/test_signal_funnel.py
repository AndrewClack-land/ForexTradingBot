"""Tests for the read-only entry-funnel diagnostic."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from tools.signal_funnel import (
    LIFECYCLE_SIGNALS,
    SIGNAL_STAGE,
    STAGES,
    build_funnel,
    main,
    parse_exec_failure,
    parse_ticker_line,
    render_report,
    result_to_dict,
)

JOURNAL_PREFIX = "2026-08-05T06:40:01+00:00 vps-22c99371 run_bot.sh[940963]: "


def _line(ts: str, symbol: str, signal: str, rest: str = "") -> str:
    return (
        f"{ts} vps-22c99371 run_bot.sh[940963]: [Ticker] {symbol} {signal}"
        f" | side=None | session=LONDON{rest}"
    )


def test_parses_journald_prefixed_line():
    parsed = parse_ticker_line(
        JOURNAL_PREFIX
        + "[Ticker] USDCAD NO_TRIGGER | side=None | session=LONDON | trigger=None"
    )
    assert parsed is not None
    assert parsed.symbol == "USDCAD"
    assert parsed.signal == "NO_TRIGGER"
    assert parsed.fields["session"] == "LONDON"
    assert parsed.ts == datetime(2026, 8, 5, 6, 40, 1, tzinfo=timezone.utc)


def test_parses_bare_line_without_timestamp():
    parsed = parse_ticker_line("[Ticker] EURUSD ENTER | side=LONG | session=NY")
    assert parsed is not None
    assert parsed.ts is None
    assert parsed.signal == "ENTER"
    assert parsed.fields["side"] == "LONG"


def test_parses_short_iso_without_colon_in_offset():
    parsed = parse_ticker_line(
        "2026-08-05T06:40:01+0000 host unit[1]: [Ticker] GOLD NO_TREND | side=None"
    )
    assert parsed is not None
    assert parsed.ts == datetime(2026, 8, 5, 6, 40, 1, tzinfo=timezone.utc)


def test_ignores_unrelated_lines():
    assert parse_ticker_line("[Profiler:core] strategy=46.4ms") is None
    assert parse_ticker_line("") is None


def test_deepest_stage_wins_inside_one_slot():
    """A slot whose ticks reached execution died at execution, not at bias."""
    lines = [
        _line("2026-08-05T06:31:00+00:00", "EURUSD", "NO_TREND"),
        _line("2026-08-05T06:32:00+00:00", "EURUSD", "NO_TRIGGER"),
        _line("2026-08-05T06:33:00+00:00", "EURUSD", "EXECUTION_ERROR"),
        _line("2026-08-05T06:34:00+00:00", "EURUSD", "NO_TREND"),
    ]
    result = build_funnel(lines)
    assert len(result.slots) == 1
    assert result.stage_deaths() == {"execution": 1}
    assert result.signal_deaths("execution") == {"EXECUTION_ERROR": 1}


def test_slots_split_by_symbol_and_by_m15_boundary():
    lines = [
        _line("2026-08-05T06:31:00+00:00", "EURUSD", "NO_TREND"),
        _line("2026-08-05T06:46:00+00:00", "EURUSD", "NO_TRIGGER"),
        _line("2026-08-05T06:31:00+00:00", "GBPUSD", "ENTER"),
    ]
    result = build_funnel(lines)
    assert len(result.slots) == 3
    assert result.stage_deaths() == {"bias": 1, "trigger": 1, "filled": 1}
    assert result.stage_deaths(symbol="GBPUSD") == {"filled": 1}


def test_slot_minutes_is_configurable():
    lines = [
        _line("2026-08-05T06:01:00+00:00", "EURUSD", "NO_TREND"),
        _line("2026-08-05T06:31:00+00:00", "EURUSD", "NO_TREND"),
    ]
    assert len(build_funnel(lines, slot_minutes=15).slots) == 2
    assert len(build_funnel(lines, slot_minutes=60).slots) == 1


def test_window_filter_excludes_out_of_range_lines():
    lines = [
        _line("2026-08-01T06:31:00+00:00", "EURUSD", "NO_TREND"),
        _line("2026-08-05T06:31:00+00:00", "EURUSD", "ENTER"),
    ]
    result = build_funnel(
        lines, since=datetime(2026, 8, 3, tzinfo=timezone.utc)
    )
    assert result.stage_deaths() == {"filled": 1}


def test_symbol_filter():
    lines = [
        _line("2026-08-05T06:31:00+00:00", "EURUSD", "ENTER"),
        _line("2026-08-05T06:31:00+00:00", "GOLD", "NO_TREND"),
    ]
    result = build_funnel(lines, symbols=["eurusd"])
    assert result.symbols() == ["EURUSD"]


def test_lifecycle_signals_do_not_enter_the_entry_funnel():
    """An open position occupies capacity but is not a rejected entry."""
    lines = [
        _line("2026-08-05T06:31:00+00:00", "EURUSD", "HOLD"),
        _line("2026-08-05T06:32:00+00:00", "EURUSD", "EXIT_TP"),
    ]
    result = build_funnel(lines)
    assert result.stage_deaths() == {}
    assert result.occupied_slots() == 1


def test_unknown_signal_codes_are_flagged_not_silently_binned():
    """A window spanning an older code generation must say so."""
    lines = [
        _line("2026-07-10T06:31:00+00:00", "EURUSD", "WAIT_M15_EMA"),
        _line("2026-07-10T06:32:00+00:00", "EURUSD", "NO_TREND"),
    ]
    result = build_funnel(lines)
    assert result.unknown_signals == {"WAIT_M15_EMA": 1}
    assert result.stage_deaths() == {"bias": 1}
    assert "WAIT_M15_EMA" in render_report(result)


def test_undated_lines_are_counted_but_never_bucketed():
    result = build_funnel(["[Ticker] EURUSD NO_TREND | side=None"])
    assert result.ticker_lines == 1
    assert result.undated_lines == 1
    assert result.slots == {}
    assert "без времени" in render_report(result)


def test_execution_failure_reasons_collapse_prices():
    line = (
        "2026-08-05T06:33:00+00:00 host unit[1]: [Core] Execution failed for "
        "EURUSD: Stale signal rejected for EURUSD: execution price 1.16352 "
        "outside entry range [1.16100, 1.16200]"
    )
    parsed = parse_exec_failure(line)
    assert parsed is not None
    _, symbol, reason = parsed
    assert symbol == "EURUSD"
    assert "1.16352" not in reason
    assert reason.count("N") >= 3

    other = line.replace("1.16352", "1.17999")
    result = build_funnel([line, other])
    assert list(result.exec_reasons.values()) == [2]


def test_stage_table_is_internally_consistent():
    """Every stage signal maps back to its stage, with no duplicate codes."""
    seen: set[str] = set()
    for name, _, signals in STAGES:
        for signal in signals:
            assert signal not in seen, signal
            seen.add(signal)
            assert SIGNAL_STAGE[signal] == name
    assert not seen & LIFECYCLE_SIGNALS


def test_report_and_json_survive_an_empty_funnel():
    result = build_funnel([])
    assert result.stage_deaths() == {}
    text = render_report(result, by="day")
    assert "ВОРОНКА ВХОДОВ" in text
    payload = result_to_dict(result)
    assert payload["decision_slots"] == 0
    assert json.loads(json.dumps(payload, ensure_ascii=False))["stage_deaths"] == {}


def test_json_payload_reports_stage_and_symbol_breakdown(tmp_path):
    lines = [
        _line("2026-08-05T06:31:00+00:00", "EURUSD", "NO_TREND"),
        _line("2026-08-05T06:46:00+00:00", "GBPUSD", "ENTER"),
    ]
    result = build_funnel(lines)
    payload = result_to_dict(result)
    assert payload["stage_deaths"] == {"bias": 1, "filled": 1}
    assert payload["stage_deaths_by_symbol"]["GBPUSD"] == {"filled": 1}
    assert payload["window_start"] == "2026-08-05T06:31:00+00:00"

    out = tmp_path / "funnel.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    out.write_text(text, encoding="utf-8")
    assert json.loads(out.read_text(encoding="utf-8"))["decision_slots"] == 2


def test_cli_reads_a_file_and_writes_json(tmp_path, capsys):
    log = tmp_path / "bot.log"
    log.write_text(
        "\n".join(
            [
                _line("2026-08-05T06:31:00+00:00", "EURUSD", "NO_TREND"),
                _line("2026-08-05T06:46:00+00:00", "EURUSD", "NO_TRIGGER"),
                _line("2026-08-05T07:01:00+00:00", "EURUSD", "ENTER"),
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "funnel.json"
    assert main(["--input", str(log), "--json", str(out), "--by", "day"]) == 0
    printed = capsys.readouterr().out
    assert "ВОРОНКА ВХОДОВ" in printed
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["stage_deaths"] == {"bias": 1, "trigger": 1, "filled": 1}


def test_cli_reads_utf16_log_written_by_the_powershell_launcher(tmp_path):
    log = tmp_path / "bot-utf16.log"
    log.write_text(
        _line("2026-08-05T06:31:00+00:00", "EURUSD", "NO_TREND"), encoding="utf-16"
    )
    assert main(["--input", str(log)]) == 0


def test_cli_fails_when_no_ticker_lines_present(tmp_path, capsys):
    log = tmp_path / "empty.log"
    log.write_text("[Profiler:core] strategy=1ms\n", encoding="utf-8")
    assert main(["--input", str(log)]) == 1
    assert "Не найдено" in capsys.readouterr().out


def test_cli_rejects_a_slot_length_that_does_not_tile_an_hour(tmp_path):
    log = tmp_path / "bot.log"
    log.write_text(_line("2026-08-05T06:31:00+00:00", "EURUSD", "ENTER"), "utf-8")
    with pytest.raises(SystemExit):
        main(["--input", str(log), "--slot-minutes", "7"])
