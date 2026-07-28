"""Command-line entry point for offline snapshot tooling."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .data import DataValidationError, HistoricalDataset
from .lse_ingest import (
    DEFAULT_BAR_TIMEZONE,
    DEFAULT_TARGET_TIMEFRAMES,
    LSEIngestError,
    import_lse_snapshot,
)
from .strategy_runner import (
    NarrativeBacktestConfig,
    StrategyBacktestError,
    infer_common_strategy_range,
    run_narrative_backtest,
    verify_release_manifest,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backtest",
        description="Pure offline backtest snapshot utilities",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser(
        "audit",
        help="validate candle files and report walk-forward readiness",
    )
    audit.add_argument("--data", required=True, help="JSON/Parquet snapshot directory")
    audit.add_argument("--symbols", nargs="*", help="optional symbol subset")
    audit.add_argument("--min-d1-bars", type=int, default=365)
    audit.add_argument("--min-months", type=int, default=12)
    audit.add_argument("--json", action="store_true", help="machine-readable output")
    audit.add_argument(
        "--manifest-out",
        help="optionally write the deterministic content manifest to this path",
    )
    verify = subparsers.add_parser(
        "verify",
        help="verify a stored snapshot manifest without WFO readiness semantics",
    )
    verify.add_argument("--data", required=True, help="sealed snapshot directory")
    verify.add_argument(
        "--json",
        action="store_true",
        help="machine-readable verification result",
    )

    strategy_run = subparsers.add_parser(
        "run",
        help="run the narrative strategy causally on an immutable snapshot",
    )
    strategy_run.add_argument(
        "--data",
        required=True,
        help="sealed snapshot directory",
    )
    strategy_run.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="bot symbols, e.g. EURUSD GBPUSD USDCAD",
    )
    strategy_run.add_argument(
        "--start",
        help="inclusive UTC decision start; defaults to common 15m coverage",
    )
    strategy_run.add_argument(
        "--end",
        help="exclusive UTC decision end; defaults to common 15m coverage",
    )
    strategy_run.add_argument(
        "--initial-capital",
        type=float,
        required=True,
        help="fixed starting capital used as the 1%% risk basis",
    )
    strategy_run.add_argument(
        "--risk-fraction",
        type=float,
        default=0.01,
        help="fixed fraction of initial capital per setup; maximum 0.01",
    )
    strategy_run.add_argument(
        "--output",
        required=True,
        help="new report directory; must not already exist",
    )
    strategy_run.add_argument(
        "--profile",
        choices=("signal-quality", "production-deterministic"),
        default="production-deterministic",
        help="raw signal engine or deterministic live-policy gates",
    )
    strategy_run.add_argument(
        "--intrabar-policy",
        choices=("stop-first", "tp-first", "both"),
        default="stop-first",
        help="M1 OHLC ordering assumption (default: conservative stop-first)",
    )
    strategy_run.add_argument(
        "--entry-ttl",
        default="15min",
        help="maximum time to find a later M1 open inside the entry range",
    )
    strategy_run.add_argument(
        "--max-holding",
        default="30D",
        help="maximum outcome horizon; remaining exposure gets a causal TIME exit",
    )
    strategy_run.add_argument(
        "--train",
        help="walk-forward train duration, e.g. 730D; requires --test",
    )
    strategy_run.add_argument(
        "--test",
        help="walk-forward OOS test duration, e.g. 180D; requires --train",
    )
    strategy_run.add_argument(
        "--step",
        help="walk-forward step; defaults to the test duration",
    )
    strategy_run.add_argument(
        "--history-limit",
        type=int,
        default=299,
        help="maximum closed bars per strategy timeframe (live parity: 299)",
    )
    strategy_run.add_argument(
        "--min-context-bars",
        type=int,
        default=299,
        help="strict 4H/1H/15M warmup requirement",
    )
    strategy_run.add_argument(
        "--min-daily-bars",
        type=int,
        default=22,
        help="minimum closed D1 bars for volatility context",
    )
    strategy_run.add_argument(
        "--sessions",
        nargs="*",
        default=["LONDON", "NY"],
        help="allowed sessions or ALL (production default: LONDON NY)",
    )
    strategy_run.add_argument(
        "--session-timezone",
        default="UTC",
    )
    strategy_run.add_argument(
        "--disable-vol-filter",
        action="store_true",
        help="disable the deterministic volatility/expected-move gate",
    )
    strategy_run.add_argument(
        "--vol-max-r",
        type=float,
        default=60.0,
    )
    strategy_run.add_argument(
        "--em-tp-ratio",
        type=float,
        default=1.0,
    )
    strategy_run.add_argument(
        "--max-setups-per-symbol-day",
        type=int,
        default=3,
    )
    strategy_run.add_argument(
        "--post-loss-cooldown",
        default="60min",
    )
    strategy_run.add_argument(
        "--enable-rejection-block-entry",
        action="store_true",
        help="enable the quarantined 15m rejection-block entry",
    )
    strategy_run.add_argument(
        "--disable-orderblock-entry",
        action="store_true",
    )
    strategy_run.add_argument(
        "--orderblock-max-age-bars",
        type=int,
        default=80,
    )
    strategy_run.add_argument(
        "--htf-score-margin",
        type=int,
        default=2,
    )
    strategy_run.add_argument(
        "--release-commit-file",
        help=(
            "file containing the exact strategy Git commit "
            "(required for production-deterministic)"
        ),
    )
    strategy_run.add_argument(
        "--release-manifest-file",
        help=(
            "root-owned SHA-256 manifest for deployed strategy files "
            "(required for production-deterministic)"
        ),
    )
    strategy_run.add_argument(
        "--environment-lock-file",
        help=(
            "root-owned full pip freeze for the backtest venv "
            "(required for production-deterministic)"
        ),
    )
    strategy_run.add_argument(
        "--json",
        action="store_true",
        help="print the final summary as JSON; progress goes to stderr",
    )

    optimize_v2 = subparsers.add_parser(
        "optimize-v2",
        help=(
            "regenerate all M15 LONG/SHORT trigger opportunities, fit "
            "train-only factor weights, and replay frozen OOS folds"
        ),
    )
    optimize_v2.add_argument("--data", required=True)
    optimize_v2.add_argument("--symbols", nargs="+", required=True)
    optimize_v2.add_argument("--start")
    optimize_v2.add_argument("--end")
    optimize_v2.add_argument(
        "--initial-capital",
        type=float,
        required=True,
    )
    optimize_v2.add_argument(
        "--risk-fraction",
        type=float,
        default=0.01,
    )
    optimize_v2.add_argument("--output", required=True)
    optimize_v2.add_argument(
        "--profile",
        choices=("signal-quality", "production-deterministic"),
        default="production-deterministic",
    )
    optimize_v2.add_argument(
        "--intrabar-policy",
        choices=("stop-first", "both"),
        default="stop-first",
    )
    optimize_v2.add_argument("--entry-ttl", default="15min")
    optimize_v2.add_argument("--max-holding", default="30D")
    optimize_v2.add_argument("--train", required=True)
    optimize_v2.add_argument("--test", required=True)
    optimize_v2.add_argument("--step")
    optimize_v2.add_argument("--history-limit", type=int, default=299)
    optimize_v2.add_argument(
        "--min-context-bars",
        type=int,
        default=299,
    )
    optimize_v2.add_argument("--min-daily-bars", type=int, default=22)
    optimize_v2.add_argument(
        "--sessions",
        nargs="*",
        default=["LONDON", "NY"],
    )
    optimize_v2.add_argument("--session-timezone", default="UTC")
    optimize_v2.add_argument(
        "--disable-vol-filter",
        action="store_true",
    )
    optimize_v2.add_argument("--vol-max-r", type=float, default=60.0)
    optimize_v2.add_argument("--em-tp-ratio", type=float, default=1.0)
    optimize_v2.add_argument(
        "--max-setups-per-symbol-day",
        type=int,
        default=3,
    )
    optimize_v2.add_argument("--post-loss-cooldown", default="60min")
    optimize_v2.add_argument(
        "--enable-rejection-block-entry",
        action="store_true",
    )
    optimize_v2.add_argument(
        "--disable-orderblock-entry",
        action="store_true",
    )
    optimize_v2.add_argument(
        "--orderblock-max-age-bars",
        type=int,
        default=80,
    )
    optimize_v2.add_argument(
        "--htf-score-margin",
        type=int,
        default=2,
    )
    optimize_v2.add_argument(
        "--cluster-proxy-data",
        help=(
            "optional sealed FxProClusterEventDataset; when omitted Cluster "
            "Rejection remains DATA_UNAVAILABLE; converted diagnostics retain "
            "their explicit research-only availability assumption"
        ),
    )
    optimize_v2.add_argument(
        "--cluster-candle-tolerance-ticks",
        action="append",
        metavar="SYMBOL=TICKS",
        help=(
            "research-only per-symbol override of the cluster candle-identity "
            "tolerance, e.g. EURUSD=4; the sealed cluster carries FxPro "
            "candles while the snapshot carries LSE candles, so the strict "
            "production tolerance vetoes bars for venue reasons unrelated to "
            "the pattern. Any run using this is a non-parity sensitivity "
            "experiment and is marked candle_identity_relaxed in the report"
        ),
    )
    optimize_v2.add_argument(
        "--quote-pressure-data",
        help=(
            "optional sealed FxProQuotePressureEventDataset; when omitted "
            "Quote Pressure Rejection remains DATA_UNAVAILABLE and no "
            "OHLCV/tick-volume proxy is used"
        ),
    )
    optimize_v2.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
    )
    optimize_v2.add_argument(
        "--min-train-opportunities",
        type=int,
        default=400,
        help=(
            "minimum unique decision_event_id x side train clusters "
            "required to fit a fold (default: 400)"
        ),
    )
    optimize_v2.add_argument("--release-commit-file")
    optimize_v2.add_argument("--release-manifest-file")
    optimize_v2.add_argument("--environment-lock-file")
    optimize_v2.add_argument("--json", action="store_true")

    lse_import = subparsers.add_parser(
        "lse-import",
        help="download LSE M1 candles into a new immutable Parquet snapshot",
    )
    lse_import.add_argument(
        "--output",
        required=True,
        help="new snapshot directory; must not already exist",
    )
    lse_import.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="bot symbols or BOT=LSE mappings, e.g. EURUSD GOLD=XAU/USD",
    )
    lse_import.add_argument(
        "--start",
        required=True,
        help="inclusive UTC start (export requires YYYY-MM-DD)",
    )
    lse_import.add_argument(
        "--end",
        required=True,
        help="exclusive UTC end (export requires YYYY-MM-DD)",
    )
    lse_import.add_argument(
        "--transport",
        choices=("export", "rest"),
        default="rest",
        help=(
            "paced UTC date-window REST with tail truncation probes "
            "(default), or chunked bulk Parquet export"
        ),
    )
    lse_import.add_argument(
        "--dataset",
        default=None,
        help="optional LSE dataset override; default resolves each symbol from catalog",
    )
    lse_import.add_argument(
        "--timeframes",
        nargs="+",
        default=list(DEFAULT_TARGET_TIMEFRAMES),
        help="canonical outputs built from M1 (default: 1m 5m 15m 1h 4h 1d)",
    )
    lse_import.add_argument(
        "--bar-timezone",
        default=DEFAULT_BAR_TIMEZONE,
        help="broker wall-clock timezone used for H4/D1 boundaries",
    )
    lse_import.add_argument(
        "--max-duplicate-ratio",
        type=float,
        default=0.001,
        help="fail when duplicate M1 timestamps exceed this fraction",
    )
    lse_import.add_argument(
        "--page-limit",
        type=int,
        default=5000,
        help=(
            "REST row limit; must exceed 1440 and reaching it fails closed "
            "(default: 5000)"
        ),
    )
    lse_import.add_argument(
        "--rest-min-interval",
        type=float,
        default=0.35,
        help=(
            "minimum seconds between every REST request, including tail "
            "probes and retries (default stays below 200/min)"
        ),
    )
    lse_import.add_argument(
        "--retry-after-seconds",
        type=float,
        default=60.0,
        help="minimum delay after HTTP 429",
    )
    lse_import.add_argument(
        "--min-bucket-coverage",
        type=float,
        default=0.95,
        help="drop derived candles below this fraction of expected M1 rows",
    )
    lse_import.add_argument(
        "--max-edge-gap-days",
        type=float,
        default=4.0,
        help="maximum allowed missing coverage at either requested range edge",
    )
    lse_import.add_argument(
        "--max-internal-gap-days",
        type=float,
        default=4.0,
        help="fail when an internal M1 history gap exceeds this many days",
    )
    lse_import.add_argument(
        "--export-chunk-days",
        type=int,
        default=600,
        help="date-only export chunk size; start/end must be YYYY-MM-DD",
    )
    lse_import.add_argument(
        "--max-export-jobs",
        type=int,
        default=5,
        help="fail before export when all symbol/chunk jobs exceed this budget",
    )
    lse_import.add_argument("--timeout", type=float, default=60.0)
    lse_import.add_argument(
        "--require-wfo-ready",
        action="store_true",
        help="return exit code 3 when the completed snapshot lacks WFO history",
    )
    lse_import.add_argument(
        "--release-commit-file",
        help="optional file containing the exact deployed 40/64-hex Git commit",
    )
    lse_import.add_argument(
        "--environment-lock-file",
        help="optional full pip freeze file; must accompany --release-commit-file",
    )
    lse_import.add_argument(
        "--json",
        action="store_true",
        help="machine-readable result without secrets",
    )
    return parser


def _audit(args: argparse.Namespace) -> int:
    dataset = HistoricalDataset.load(Path(args.data))
    if args.manifest_out:
        dataset.write_manifest(args.manifest_out)
    readiness = dataset.audit_readiness(
        min_d1_bars=args.min_d1_bars,
        min_months=args.min_months,
        symbols=args.symbols,
    )
    payload = {
        "data": str(dataset.root),
        "manifest_sha256": dataset.manifest_sha256,
        "coverage": [item.to_dict() for item in dataset.coverage()],
        **readiness,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(f"Snapshot: {dataset.root}")
        print(f"Manifest SHA-256: {dataset.manifest_sha256}")
        print("Coverage:")
        for item in dataset.coverage():
            print(
                f"  {item.symbol:10s} {item.timeframe:3s} "
                f"rows={item.rows:7d} duplicates={item.duplicate_rows:5d} "
                f"{item.start.isoformat()} -> {item.end_close.isoformat()}"
            )
        if readiness["wfo_ready"]:
            print("WFO READY")
        else:
            print("WFO BLOCKED")
            for reason in readiness["reasons"]:
                print(f"  - {reason}")

    # A blocked audit must fail closed so automation cannot accidentally start
    # walk-forward analysis on an inadequate snapshot.
    return 0 if readiness["wfo_ready"] else 3


def _verify(args: argparse.Namespace) -> int:
    root = Path(args.data).expanduser().resolve()
    if not (root / "manifest.json").is_file():
        raise DataValidationError(f"{root}: stored manifest.json is missing")
    dataset = HistoricalDataset.load(root)
    payload = {
        "verified": True,
        "data": str(dataset.root),
        "manifest_sha256": dataset.manifest_sha256,
        "files": len(dataset.manifest["files"]),
        "series": len(dataset.coverage()),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(f"SNAPSHOT VERIFIED: {dataset.root}")
        print(f"Manifest SHA-256: {dataset.manifest_sha256}")
    return 0


def _read_release_commit(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    commit_path = Path(path).expanduser().resolve()
    try:
        value = commit_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise StrategyBacktestError(
            f"Cannot read release commit file: {commit_path}"
        ) from exc
    if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", value):
        raise StrategyBacktestError(
            f"{commit_path}: release commit must be a 40/64-hex value"
        )
    return value.lower()


def _hash_environment_lock(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    lock_path = Path(path).expanduser().resolve()
    if not lock_path.is_file():
        raise StrategyBacktestError(
            f"Environment lock file does not exist: {lock_path}"
        )
    digest = hashlib.sha256()
    try:
        with lock_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise StrategyBacktestError(
            f"Cannot read environment lock file: {lock_path}"
        ) from exc
    return digest.hexdigest()


def _strategy_run(args: argparse.Namespace) -> int:
    release_commit = _read_release_commit(args.release_commit_file)
    attestation_fields = (
        release_commit is not None,
        bool(args.release_manifest_file),
        bool(args.environment_lock_file),
    )
    if any(attestation_fields) and not all(attestation_fields):
        raise StrategyBacktestError(
            "release commit, release manifest, and environment lock "
            "must be provided together"
        )
    if args.profile == "production-deterministic" and (
        release_commit is None
        or not args.release_manifest_file
        or not args.environment_lock_file
    ):
        raise StrategyBacktestError(
            "production-deterministic profile requires "
            "--release-commit-file, --release-manifest-file, and "
            "--environment-lock-file"
        )
    release_manifest_sha256 = None
    if args.release_manifest_file:
        if release_commit is None:
            raise StrategyBacktestError(
                "--release-manifest-file requires --release-commit-file"
            )
        release_manifest_sha256 = verify_release_manifest(
            args.release_manifest_file,
            expected_commit=release_commit,
        )
    environment_lock_sha256 = _hash_environment_lock(
        args.environment_lock_file
    )
    dataset = HistoricalDataset.load(Path(args.data))
    symbols = tuple(str(symbol).upper() for symbol in args.symbols)
    common_start, common_end = infer_common_strategy_range(dataset, symbols)
    config = NarrativeBacktestConfig.build(
        symbols=symbols,
        start=args.start or common_start,
        end=args.end or common_end,
        initial_capital=args.initial_capital,
        risk_fraction=args.risk_fraction,
        history_limit=args.history_limit,
        min_context_bars=args.min_context_bars,
        min_daily_bars=args.min_daily_bars,
        entry_ttl=args.entry_ttl,
        max_holding=args.max_holding,
        intrabar_policy=args.intrabar_policy,
        profile=args.profile,
        train=args.train,
        test=args.test,
        step=args.step,
        session_timezone=args.session_timezone,
        allowed_sessions=args.sessions,
        vol_filter_enabled=not args.disable_vol_filter,
        vol_max_r=args.vol_max_r,
        em_tp_ratio=args.em_tp_ratio,
        max_setups_per_symbol_day=args.max_setups_per_symbol_day,
        post_loss_cooldown=args.post_loss_cooldown,
        rejection_block_entry_enabled=args.enable_rejection_block_entry,
        orderblock_entry_enabled=not args.disable_orderblock_entry,
        orderblock_max_age_bars=args.orderblock_max_age_bars,
        htf_score_margin=args.htf_score_margin,
        release_commit=release_commit,
        release_manifest_sha256=release_manifest_sha256,
        environment_lock_sha256=environment_lock_sha256,
    )
    progress_stream = sys.stderr if args.json else sys.stdout

    def print_progress(event: Mapping[str, Any]) -> None:
        fields = [
            f"phase={event['phase']}",
            f"state={event['state']}",
        ]
        for name in (
            "symbol",
            "policy",
            "completed",
            "decisions",
            "candidates",
            "setups",
            "percent",
        ):
            if name in event:
                fields.append(f"{name}={event[name]}")
        print(
            "Backtest progress: " + " ".join(fields),
            file=progress_stream,
            flush=True,
        )

    result = run_narrative_backtest(
        dataset,
        config,
        progress=print_progress,
    )
    report = result.write(args.output)
    payload = {
        **dict(result.summary),
        "report": str(report),
    }
    if args.json:
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
        )
        return 0

    print(f"Backtest report: {report}")
    print(f"Snapshot manifest: {dataset.manifest_sha256}")
    for policy, policy_payload in result.summary["policies"].items():
        metrics = policy_payload["metrics"]
        equity = policy_payload["fixed_risk_equity"]
        profit_factor = metrics["profit_factor"]
        pf_text = "n/a" if profit_factor is None else f"{profit_factor:.3f}"
        print(
            f"  {policy}: setups={metrics['setups_closed']}/"
            f"{metrics['setups_total']} "
            f"net={metrics['net_r']:.3f}R "
            f"expectancy={metrics['expectancy_r']:.3f}R "
            f"win={metrics['win_rate']:.1%} PF={pf_text} "
            f"maxDD={metrics['max_drawdown_r']:.3f}R "
            f"capital={equity['ending_capital']:.2f}"
        )
    print(
        "Gross diagnostic only: historical spread, commission, swap, "
        "slippage and tick order are unavailable."
    )
    return 0


def _counterfactual_run(args: argparse.Namespace) -> int:
    """Run the all-M15 replacement-weight optimizer in an isolated release."""

    from .counterfactual import run_counterfactual_backtest
    from .fxpro_cluster_data import FxProClusterEventDataset
    from .fxpro_quote_pressure_data import FxProQuotePressureEventDataset

    release_commit = _read_release_commit(args.release_commit_file)
    attestation_fields = (
        release_commit is not None,
        bool(args.release_manifest_file),
        bool(args.environment_lock_file),
    )
    if any(attestation_fields) and not all(attestation_fields):
        raise StrategyBacktestError(
            "release commit, release manifest, and environment lock "
            "must be provided together"
        )
    if args.profile == "production-deterministic" and (
        release_commit is None
        or not args.release_manifest_file
        or not args.environment_lock_file
    ):
        raise StrategyBacktestError(
            "production-deterministic profile requires "
            "--release-commit-file, --release-manifest-file, and "
            "--environment-lock-file"
        )
    release_manifest_sha256 = None
    if args.release_manifest_file:
        if release_commit is None:
            raise StrategyBacktestError(
                "--release-manifest-file requires --release-commit-file"
            )
        release_manifest_sha256 = verify_release_manifest(
            args.release_manifest_file,
            expected_commit=release_commit,
        )
    environment_lock_sha256 = _hash_environment_lock(
        args.environment_lock_file
    )
    dataset = HistoricalDataset.load(Path(args.data))
    symbols = tuple(str(symbol).upper() for symbol in args.symbols)
    common_start, common_end = infer_common_strategy_range(
        dataset,
        symbols,
    )
    config = NarrativeBacktestConfig.build(
        symbols=symbols,
        start=args.start or common_start,
        end=args.end or common_end,
        initial_capital=args.initial_capital,
        risk_fraction=args.risk_fraction,
        history_limit=args.history_limit,
        min_context_bars=args.min_context_bars,
        min_daily_bars=args.min_daily_bars,
        entry_ttl=args.entry_ttl,
        max_holding=args.max_holding,
        intrabar_policy=args.intrabar_policy,
        profile=args.profile,
        train=args.train,
        test=args.test,
        step=args.step,
        session_timezone=args.session_timezone,
        allowed_sessions=args.sessions,
        vol_filter_enabled=not args.disable_vol_filter,
        vol_max_r=args.vol_max_r,
        em_tp_ratio=args.em_tp_ratio,
        max_setups_per_symbol_day=args.max_setups_per_symbol_day,
        post_loss_cooldown=args.post_loss_cooldown,
        rejection_block_entry_enabled=(
            args.enable_rejection_block_entry
        ),
        orderblock_entry_enabled=not args.disable_orderblock_entry,
        orderblock_max_age_bars=args.orderblock_max_age_bars,
        htf_score_margin=args.htf_score_margin,
        release_commit=release_commit,
        release_manifest_sha256=release_manifest_sha256,
        environment_lock_sha256=environment_lock_sha256,
    )
    cluster_proxy = (
        FxProClusterEventDataset.load(Path(args.cluster_proxy_data))
        if args.cluster_proxy_data
        else None
    )
    quote_pressure = (
        FxProQuotePressureEventDataset.load(Path(args.quote_pressure_data))
        if args.quote_pressure_data
        else None
    )
    candle_tolerance: dict[str, int] = {}
    for item in args.cluster_candle_tolerance_ticks or ():
        symbol, separator, ticks = str(item).partition("=")
        if not separator or not symbol.strip() or not ticks.strip():
            raise SystemExit(
                "--cluster-candle-tolerance-ticks expects SYMBOL=TICKS, "
                f"got: {item}"
            )
        try:
            parsed = int(ticks)
        except ValueError:
            raise SystemExit(
                f"--cluster-candle-tolerance-ticks needs an integer: {item}"
            ) from None
        key = symbol.strip().upper()
        if key in candle_tolerance:
            raise SystemExit(
                f"--cluster-candle-tolerance-ticks repeats symbol: {key}"
            )
        candle_tolerance[key] = parsed
    progress_stream = sys.stderr if args.json else sys.stdout

    def print_progress(event: Mapping[str, Any]) -> None:
        fields = [
            f"phase={event['phase']}",
            f"state={event['state']}",
        ]
        for name in (
            "symbol",
            "policy",
            "completed",
            "decisions",
            "candidates",
            "setups",
            "percent",
        ):
            if name in event:
                fields.append(f"{name}={event[name]}")
        print(
            "Optimizer v2 progress: " + " ".join(fields),
            file=progress_stream,
            flush=True,
        )

    result = run_counterfactual_backtest(
        dataset,
        config,
        cluster_proxy=cluster_proxy,
        quote_pressure=quote_pressure,
        cluster_candle_tolerance_ticks=candle_tolerance,
        ridge_alpha=args.ridge_alpha,
        min_train_opportunities=args.min_train_opportunities,
        progress=print_progress,
    )
    report = result.write(args.output)
    payload = {
        **dict(result.summary),
        "report": str(report),
    }
    if args.json:
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
        )
        return 0

    metrics = result.summary["oos_primary_metrics"]
    profit_factor = metrics.get("profit_factor")
    pf_text = (
        "n/a" if profit_factor is None else f"{profit_factor:.3f}"
    )
    print(f"Optimizer v2 report: {report}")
    print(
        "  stop-first OOS: "
        f"setups={metrics['setups_closed']}/{metrics['setups_total']} "
        f"net={metrics['net_r']:.3f}R "
        f"expectancy={metrics['expectancy_r']:.3f}R "
        f"PF={pf_text} maxDD={metrics['max_drawdown_r']:.3f}R"
    )
    print(
        "  FxPro Cluster Rejection sidecar: "
        + (
            "sealed research proxy loaded"
            if cluster_proxy is not None
            else "DATA_UNAVAILABLE (no OHLCV/MT5-volume substitute)"
        )
    )
    print(
        "  FxPro Quote Pressure Rejection sidecar: "
        + (
            "sealed and loaded"
            if quote_pressure is not None
            else "DATA_UNAVAILABLE (no OHLCV/tick-volume proxy)"
        )
    )
    return 0


def _lse_import(args: argparse.Namespace) -> int:
    progress_stream = sys.stderr if args.json else sys.stdout

    def print_progress(event: Mapping[str, Any]) -> None:
        fields = [
            f"phase={event['phase']}",
            f"state={event['state']}",
        ]
        if "symbol" in event:
            fields.append(f"symbol={event['symbol']}")
        if "provider_symbol" in event:
            fields.append(f"provider={event['provider_symbol']}")
        if "total_windows" in event:
            fields.append(
                f"window={event.get('window', 0)}/{event['total_windows']}"
            )
        if "request_count" in event:
            fields.append(f"requests={event['request_count']}")
        if "total_timeframes" in event:
            fields.append(
                f"step={event.get('step', 0)}/{event['total_timeframes']}"
            )
        if "timeframe" in event:
            fields.append(f"timeframe={event['timeframe']}")
        if "rows" in event:
            fields.append(f"rows={event['rows']}")
        if "symbols" in event:
            fields.append(f"symbols={event['symbols']}")
        if "files" in event:
            fields.append(f"files={event['files']}")
        if "output" in event:
            fields.append(f"output={event['output']}")
        print("LSE progress: " + " ".join(fields), file=progress_stream, flush=True)

    result = import_lse_snapshot(
        output=args.output,
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        timeframes=args.timeframes,
        transport=args.transport,
        dataset=args.dataset or None,
        bar_timezone=args.bar_timezone,
        max_duplicate_ratio=args.max_duplicate_ratio,
        page_limit=args.page_limit,
        rest_min_interval=args.rest_min_interval,
        retry_after_seconds=args.retry_after_seconds,
        min_bucket_coverage=args.min_bucket_coverage,
        max_edge_gap_days=args.max_edge_gap_days,
        max_internal_gap_days=args.max_internal_gap_days,
        export_chunk_days=args.export_chunk_days,
        max_export_jobs=args.max_export_jobs,
        release_commit_file=args.release_commit_file,
        environment_lock_file=args.environment_lock_file,
        timeout=args.timeout,
        progress=print_progress,
    )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(f"LSE snapshot: {result.output}")
        print(f"Manifest SHA-256: {result.manifest_sha256}")
        for item in result.symbols:
            rows = ", ".join(
                f"{timeframe}={count}"
                for timeframe, count in item["derived_rows"].items()
            )
            print(
                f"  {item['provider_symbol']} -> {item['bot_symbol']}: "
                f"{rows}"
            )
        print("WFO READY" if result.readiness["wfo_ready"] else "WFO BLOCKED")
    if args.require_wfo_ready and not result.readiness["wfo_ready"]:
        return 3
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            return _audit(args)
        if args.command == "verify":
            return _verify(args)
        if args.command == "run":
            return _strategy_run(args)
        if args.command == "optimize-v2":
            return _counterfactual_run(args)
        if args.command == "lse-import":
            return _lse_import(args)
    except (
        DataValidationError,
        LSEIngestError,
        StrategyBacktestError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"BACKTEST ERROR [{args.command}]: {exc}", file=sys.stderr)
        return 2
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
