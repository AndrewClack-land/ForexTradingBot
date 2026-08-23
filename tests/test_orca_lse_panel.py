"""Contract tests for the sealed LSE -> ORCA-4 panel materializer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import pytest

from monitoring.orca_lse_panel import (
    ORCA_LSE_MANIFEST_FILENAME,
    ORCA_LSE_PANEL_FILENAME,
    ORCA_LSE_UNIVERSE,
    OrcaLsePanelConfig,
    OrcaLsePanelError,
    build_orca_lse_panel,
    materialize_orca_lse_panel,
)


PROVIDER = "london_strategic_edge"
PROVIDER_SYMBOLS = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDCAD": "USD/CAD",
    "GOLD": "XAU/USD",
}
BASE_PRICE = {
    "EURUSD": 1.10,
    "GBPUSD": 1.27,
    "USDCAD": 1.36,
    "GOLD": 2400.0,
}
# The sealed FX snapshot closes its D1 bars at the Europe/Athens day boundary,
# which is 21:00 UTC during northern summer.
FIRST_CLOSE = datetime(2024, 1, 2, 21, 0, tzinfo=timezone.utc)


def _closes(symbol: str, count: int, *, offset: int = 0) -> pd.Series:
    """Deterministic, strictly positive closes keyed by absolute bar index."""

    index = pd.DatetimeIndex(
        [FIRST_CLOSE + timedelta(days=offset + step) for step in range(count)],
        name="bar_close_time",
    )
    base = BASE_PRICE[symbol]
    values = [
        base * (1.0 + 0.0007 * ((offset + step) % 23) - 0.0003 * ((offset + step) % 7))
        for step in range(count)
    ]
    return pd.Series(values, index=index, name=symbol, dtype=float)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_d1(root: Path, series: pd.Series) -> Path:
    symbol = str(series.name)
    frame = pd.DataFrame(
        {
            "symbol": symbol,
            "tf": "1d",
            "timestamp": series.index - pd.Timedelta(days=1),
            "bar_close_time": series.index,
            "open": series.to_numpy(dtype=float),
            "high": series.to_numpy(dtype=float) * 1.001,
            "low": series.to_numpy(dtype=float) * 0.999,
            "close": series.to_numpy(dtype=float),
            "volume": 0.0,
        }
    )
    path = root / f"{symbol}_1d.parquet"
    frame.to_parquet(path, index=False)
    return path


def seal_snapshot(root: Path, series: Sequence[pd.Series]) -> Path:
    """Write a snapshot shaped exactly like a real ``backtest lse-import``."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "environment.freeze.txt").write_text(
        "pandas==2.3.2\npyarrow==22.0.0\n",
        encoding="utf-8",
    )

    source_files = []
    for item in series:
        path = _write_d1(root, item)
        source_files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "rows": int(len(item)),
                "symbol": str(item.name),
                "timeframe": "1d",
                "start": item.index[0].isoformat(),
                "end_close": item.index[-1].isoformat(),
            }
        )

    source_manifest = {
        "schema_version": 1,
        "provider": PROVIDER,
        "bar_timezone": "Europe/Athens",
        "source_timeframe": "1m",
        "timestamp_semantics": "bar_open_utc",
        "transport": "rest",
        "requested_start": FIRST_CLOSE.isoformat(),
        "requested_end_exclusive": (
            series[0].index[-1] + pd.Timedelta(days=1)
        ).isoformat(),
        "symbols": [
            {
                "bot_symbol": str(item.name),
                "provider_symbol": PROVIDER_SYMBOLS[str(item.name)],
                "dataset": "commodity" if item.name == "GOLD" else "fx",
            }
            for item in series
        ],
        "files": source_files,
    }
    source_path = root / "source_manifest.json"
    source_path.write_text(
        json.dumps(source_manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    files = [
        {
            "path": entry["path"],
            "size": entry["bytes"],
            "sha256": entry["sha256"],
        }
        for entry in source_files
    ]
    for name in ("environment.freeze.txt", "source_manifest.json"):
        target = root / name
        files.append(
            {
                "kind": "metadata",
                "path": name,
                "size": target.stat().st_size,
                "sha256": _sha256(target),
            }
        )
    manifest = {"schema_version": 1, "series": len(source_files), "files": files}
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    (root / ORCA_LSE_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return root


def _history_roots(tmp_path: Path, *, rows: int = 40) -> tuple[Path, Path]:
    fx = seal_snapshot(
        tmp_path / "fx",
        [_closes(symbol, rows) for symbol in ORCA_LSE_UNIVERSE[:3]],
    )
    gold = seal_snapshot(tmp_path / "gold", [_closes("GOLD", rows)])
    return fx, gold


def _tail_root(
    tmp_path: Path,
    *,
    offset: int,
    rows: int,
    name: str = "tail",
    mutate: Mapping[str, Mapping[int, float]] | None = None,
) -> Path:
    series = []
    for symbol in ORCA_LSE_UNIVERSE:
        item = _closes(symbol, rows, offset=offset)
        for position, value in (mutate or {}).get(symbol, {}).items():
            item.iloc[position] = value
        series.append(item)
    return seal_snapshot(tmp_path / name, series)


def _as_of(rows: int = 400) -> datetime:
    return FIRST_CLOSE + timedelta(days=rows)


def test_history_only_panel_is_an_exact_no_fill_inner_join(tmp_path: Path) -> None:
    fx, gold = _history_roots(tmp_path, rows=40)

    panel, lineage = build_orca_lse_panel(
        fx,
        gold,
        as_of_utc=_as_of(),
        minimum_aligned_rows=30,
    )

    assert list(panel.columns) == list(ORCA_LSE_UNIVERSE)
    assert len(panel) == 40
    assert lineage["tail"] is None
    assert lineage["join_contract"]["fill"] == "none"
    assert lineage["history_rows"] == 40
    assert len(lineage["sources"]) == 2
    assert {item["role"] for item in lineage["sources"]} == {"fx", "gold"}


def test_tail_extends_history_and_records_the_verified_overlap(
    tmp_path: Path,
) -> None:
    fx, gold = _history_roots(tmp_path, rows=40)
    tail = _tail_root(tmp_path, offset=30, rows=25)

    panel, lineage = build_orca_lse_panel(
        fx,
        gold,
        tail_snapshot=tail,
        as_of_utc=_as_of(),
        minimum_aligned_rows=30,
    )

    # History covers bars 0..39 and the tail covers 30..54: 10 shared bars
    # must verify as identical before the 15 newer bars are appended.
    assert lineage["tail"]["overlap_rows"] == 10
    assert lineage["tail"]["rows_added_after_history"] == 15
    assert len(panel) == 55
    assert panel.index[-1] == FIRST_CLOSE + timedelta(days=54)
    assert len(lineage["sources"]) == 3
    assert lineage["sources"][-1]["role"] == "tail"

    # The spliced rows are the tail's own values, never a reconciliation.
    expected_tail = _closes("GOLD", 25, offset=30)
    assert panel["GOLD"].iloc[-15:].tolist() == expected_tail.iloc[-15:].tolist()


def test_tail_disagreement_on_an_overlapping_bar_fails_closed(
    tmp_path: Path,
) -> None:
    fx, gold = _history_roots(tmp_path, rows=40)
    # Tail position 2 is absolute bar 32, which the history also carries.
    tail = _tail_root(
        tmp_path,
        offset=30,
        rows=25,
        mutate={"GBPUSD": {2: 1.999}},
    )

    with pytest.raises(OrcaLsePanelError, match="disagrees with the sealed history"):
        build_orca_lse_panel(
            fx,
            gold,
            tail_snapshot=tail,
            as_of_utc=_as_of(),
            minimum_aligned_rows=30,
        )


def test_tail_without_enough_overlap_fails_closed(tmp_path: Path) -> None:
    fx, gold = _history_roots(tmp_path, rows=40)
    # Starts after the history ends: nothing proves the two imports agree.
    tail = _tail_root(tmp_path, offset=41, rows=20)

    with pytest.raises(OrcaLsePanelError, match="at least 5 exact D1 bars"):
        build_orca_lse_panel(
            fx,
            gold,
            tail_snapshot=tail,
            as_of_utc=_as_of(),
            minimum_aligned_rows=30,
        )


def test_tail_that_adds_no_new_bar_fails_closed(tmp_path: Path) -> None:
    fx, gold = _history_roots(tmp_path, rows=40)
    tail = _tail_root(tmp_path, offset=20, rows=20)

    with pytest.raises(OrcaLsePanelError, match="adds no D1 bar after"):
        build_orca_lse_panel(
            fx,
            gold,
            tail_snapshot=tail,
            as_of_utc=_as_of(),
            minimum_aligned_rows=30,
        )


def test_tail_universe_order_is_part_of_the_contract(tmp_path: Path) -> None:
    fx, gold = _history_roots(tmp_path, rows=40)
    reordered = seal_snapshot(
        tmp_path / "tail-reordered",
        [
            _closes(symbol, 25, offset=30)
            for symbol in ("GOLD", "EURUSD", "GBPUSD", "USDCAD")
        ],
    )

    with pytest.raises(OrcaLsePanelError, match="symbol order/universe mismatch"):
        build_orca_lse_panel(
            fx,
            gold,
            tail_snapshot=reordered,
            as_of_utc=_as_of(),
            minimum_aligned_rows=30,
        )


def test_a_tampered_source_file_fails_closed(tmp_path: Path) -> None:
    fx, gold = _history_roots(tmp_path, rows=40)
    target = gold / "GOLD_1d.parquet"
    target.write_bytes(target.read_bytes() + b"\x00")

    with pytest.raises(OrcaLsePanelError, match="size mismatch"):
        build_orca_lse_panel(fx, gold, as_of_utc=_as_of(), minimum_aligned_rows=30)


def test_publication_is_atomic_self_hashed_and_never_overwrites(
    tmp_path: Path,
) -> None:
    fx, gold = _history_roots(tmp_path, rows=40)
    tail = _tail_root(tmp_path, offset=30, rows=25)
    output = tmp_path / "published" / "orca4-v1"

    result = materialize_orca_lse_panel(
        OrcaLsePanelConfig(
            fx_snapshot=fx,
            gold_snapshot=gold,
            output_dir=output,
            tail_snapshot=tail,
            minimum_aligned_rows=30,
        ),
        generated_at_utc=_as_of(),
    )

    assert result["status"] == "PUBLISHED"
    assert result["rows"] == 55
    assert result["tail_rows_added"] == 15
    assert result["columns"] == list(ORCA_LSE_UNIVERSE)

    payload = json.loads(
        (output / ORCA_LSE_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    stored = payload.pop("manifest_sha256")
    assert stored == _canonical_hash(payload)
    assert payload["output"]["sha256"] == _sha256(output / ORCA_LSE_PANEL_FILENAME)

    reloaded = pd.read_parquet(output / ORCA_LSE_PANEL_FILENAME)
    assert list(reloaded.columns) == list(ORCA_LSE_UNIVERSE)
    assert len(reloaded) == 55

    with pytest.raises(OrcaLsePanelError, match="already exists"):
        materialize_orca_lse_panel(
            OrcaLsePanelConfig(
                fx_snapshot=fx,
                gold_snapshot=gold,
                output_dir=output,
                tail_snapshot=tail,
                minimum_aligned_rows=30,
            ),
            generated_at_utc=_as_of(),
        )


def test_failed_publication_leaves_no_staging_directory(tmp_path: Path) -> None:
    fx, gold = _history_roots(tmp_path, rows=40)
    output = tmp_path / "published" / "orca4-too-short"

    with pytest.raises(OrcaLsePanelError, match="needs at least"):
        materialize_orca_lse_panel(
            OrcaLsePanelConfig(
                fx_snapshot=fx,
                gold_snapshot=gold,
                output_dir=output,
                minimum_aligned_rows=5000,
            ),
            generated_at_utc=_as_of(),
        )

    assert not output.exists()
    assert list(output.parent.iterdir()) == []
