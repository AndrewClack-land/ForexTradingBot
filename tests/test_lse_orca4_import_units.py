"""Deploy-artifact contracts for the two incremental ORCA-4 LSE imports.

The sealed ``fx-2020-2026-v1`` snapshot already holds EURUSD/GBPUSD/USDCAD
through 2026-07-23, so the ORCA-4 panel is completed by two much smaller
imports instead of a full re-download: a GOLD-only history over exactly the
same range, and a short four-symbol tail that carries the panel to the present.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEALED_FX_END = "2026-07-24"


def _text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_gold_import_is_single_symbol_and_aligned_with_the_sealed_fx_snapshot():
    env = _text("deploy/lse-gold-import.env.example")

    assert 'LSE_SYMBOLS="GOLD=XAU/USD"' in env
    assert "LSE_START=2020-01-01" in env
    # The exclusive end matches the sealed FX snapshot exactly, so the two
    # histories cover the same span and the inner join loses nothing at the
    # edge.
    assert f"LSE_END={SEALED_FX_END}" in env
    assert "LSE_TRANSPORT=export" in env
    assert "LSE_DATASET=commodity" in env
    assert 'LSE_TIMEFRAMES="1d"' in env


def test_tail_import_covers_the_whole_orca4_universe_in_contract_order():
    env = _text("deploy/lse-tail-import.env.example")

    assert 'LSE_SYMBOLS="EURUSD GBPUSD USDCAD GOLD=XAU/USD"' in env
    # A mixed fx/commodity run must resolve each symbol from the catalog.
    assert "LSE_DATASET=\n" in env
    assert "LSE_TRANSPORT=rest" in env
    assert 'LSE_TIMEFRAMES="1d"' in env
    assert "LSE_OUTPUT_DIR=/var/lib/forexbot-backtest/incoming/" in env


def test_tail_starts_inside_the_sealed_history_so_the_overlap_is_provable():
    env = _text("deploy/lse-tail-import.env.example")
    start = next(
        line.split("=", 1)[1].strip()
        for line in env.splitlines()
        if line.startswith("LSE_START=")
    )
    end = next(
        line.split("=", 1)[1].strip()
        for line in env.splitlines()
        if line.startswith("LSE_END=")
    )
    assert start < SEALED_FX_END < end


def test_import_quality_gates_match_so_d1_buckets_aggregate_identically():
    gold = _text("deploy/lse-gold-import.env.example")
    tail = _text("deploy/lse-tail-import.env.example")
    fx = _text("deploy/lse-import.env.example")

    for gate in (
        "LSE_BAR_TIMEZONE=Europe/Athens",
        "LSE_MIN_BUCKET_COVERAGE=0.95",
        "LSE_MAX_EDGE_GAP_DAYS=4",
        "LSE_MAX_INTERNAL_GAP_DAYS=4",
    ):
        assert gate in fx
        assert gate in gold
        assert gate in tail


def test_import_units_are_isolated_and_keep_the_existing_hardening():
    for name, env_file in (
        ("deploy/forexbot-lse-gold-import.service", "lse-gold-import.env"),
        ("deploy/forexbot-lse-tail-import.service", "lse-tail-import.env"),
    ):
        unit = _text(name)
        assert "EnvironmentFile=/etc/forexbot/lse.env" in unit
        assert f"EnvironmentFile=/etc/forexbot/{env_file}" in unit
        assert "EnvironmentFile=/etc/forexbot/lse-import.env" not in unit
        assert "User=forexbot-backtest" in unit
        assert "Type=oneshot" in unit
        assert "Restart=" not in unit
        assert "ReadWritePaths=/var/lib/forexbot-backtest" in unit
        assert "ReadOnlyPaths=/srv/forexbot-backtest/snapshots" in unit
        assert "NoNewPrivileges=true" in unit
        assert "ProtectSystem=strict" in unit
        # A research import must never touch the live MT5 bot.
        assert "forexbot.service" not in unit


def test_the_three_imports_publish_to_distinct_targets():
    outputs = set()
    for name in (
        "deploy/lse-import.env.example",
        "deploy/lse-gold-import.env.example",
        "deploy/lse-tail-import.env.example",
    ):
        outputs.add(
            next(
                line.split("=", 1)[1].strip()
                for line in _text(name).splitlines()
                if line.startswith("LSE_OUTPUT_DIR=")
            )
        )
    assert len(outputs) == 3
