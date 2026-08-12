from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import backtest.__main__ as backtest_main
import backtest.counterfactual as counterfactual_module
from backtest.strategy_runner import (
    NarrativeBacktestConfig,
    _configure_strategy,
)


def _config(**overrides):
    values = {
        "symbols": ["EURUSD"],
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-02-01T00:00:00Z",
        "initial_capital": 10_000,
    }
    values.update(overrides)
    return NarrativeBacktestConfig.build(**values)


def _optimize_args(*extra):
    return backtest_main._build_parser().parse_args([
        "optimize-v2",
        "--data",
        "snapshot",
        "--symbols",
        "EURUSD",
        "--initial-capital",
        "10000",
        "--output",
        "report",
        "--profile",
        "signal-quality",
        "--train",
        "730D",
        "--test",
        "180D",
        "--json",
        *extra,
    ])


def test_h1_rb_oos_config_is_default_off_and_separate_from_m15():
    default = _config()
    assert default.rejection_block_entry_enabled is False
    assert default.rejection_block_h1_oos_enabled is False
    assert default.to_dict()["strategy_settings"][
        "rejection_block_h1_oos_enabled"
    ] is False

    enabled = _config(rejection_block_h1_oos_enabled=True)
    strategy = _configure_strategy(SimpleNamespace(), enabled)
    assert strategy.rejection_block_h1_entry_enabled is True
    assert strategy.rejection_block_entry_enabled is False
    assert enabled.to_dict()["strategy_settings"][
        "rejection_block_h1_oos_enabled"
    ] is True


def test_optimize_v2_cli_flag_defaults_off_and_does_not_enable_m15():
    default = _optimize_args()
    enabled = _optimize_args("--enable-rejection-block-h1-oos")

    assert default.enable_rejection_block_h1_oos is False
    assert enabled.enable_rejection_block_h1_oos is True
    assert enabled.enable_rejection_block_entry is False


def test_optimize_v2_handler_passes_h1_oos_flag_to_config(
    monkeypatch,
    tmp_path,
):
    captured = {}

    class Result:
        summary = {}

        @staticmethod
        def write(output):
            return Path(output)

    monkeypatch.setattr(
        backtest_main.HistoricalDataset,
        "load",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        backtest_main,
        "infer_common_strategy_range",
        lambda _dataset, _symbols: (
            pd.Timestamp("2020-01-01T00:00:00Z"),
            pd.Timestamp("2026-01-01T00:00:00Z"),
        ),
    )

    def fake_run(_dataset, config, **_kwargs):
        captured["config"] = config
        return Result()

    monkeypatch.setattr(
        counterfactual_module,
        "run_counterfactual_backtest",
        fake_run,
    )
    args = _optimize_args(
        "--output",
        str(tmp_path / "report"),
        "--enable-rejection-block-h1-oos",
    )

    assert backtest_main._counterfactual_run(args) == 0
    config = captured["config"]
    assert config.rejection_block_h1_oos_enabled is True
    assert config.rejection_block_entry_enabled is False
