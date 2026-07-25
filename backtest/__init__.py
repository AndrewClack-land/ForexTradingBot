"""Pure, offline backtesting primitives.

This package deliberately does not import ``main``, ``MetaTrader5`` or the
production trade/AI stores.  It is safe to use on immutable candle snapshots.
"""

from .data import (
    DataValidationError,
    HistoricalDataset,
    LIVE_CLOSED_BAR_LIMIT,
)
from .metrics import aggregate_setup_metrics
from .lse_ingest import (
    LSEImportResult,
    LSEIngestError,
    LSESymbolSpec,
    import_lse_snapshot,
)
from .simulator import SetupOutcome, simulate_split_outcome
from .strategy_runner import (
    NarrativeBacktestConfig,
    StrategyBacktestError,
    StrategyBacktestResult,
    infer_common_strategy_range,
    run_narrative_backtest,
    verify_release_manifest,
)
from .walkforward import WalkForwardFold, split_walk_forward

__all__ = [
    "DataValidationError",
    "HistoricalDataset",
    "LSEImportResult",
    "LSEIngestError",
    "LSESymbolSpec",
    "LIVE_CLOSED_BAR_LIMIT",
    "NarrativeBacktestConfig",
    "SetupOutcome",
    "StrategyBacktestError",
    "StrategyBacktestResult",
    "WalkForwardFold",
    "aggregate_setup_metrics",
    "infer_common_strategy_range",
    "import_lse_snapshot",
    "run_narrative_backtest",
    "simulate_split_outcome",
    "split_walk_forward",
    "verify_release_manifest",
]
