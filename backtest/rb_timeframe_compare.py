"""Fast causal RB M15-vs-H1-vs-H4 comparison on a sealed broker snapshot."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
from .counterfactual import _Opportunity, _label_opportunity, _materialize_entry
from .data import HistoricalDataset
from .strategy_runner import NarrativeBacktestConfig, _default_strategy_factory, _prepare_symbol, infer_common_strategy_range
from .walkforward import split_walk_forward

SCHEMA = "rb-timeframe-comparison/v2"
TFS = {
    "rejection_block_15m": ("15m", "15M", "trigger_15m_rejection_block"),
    "rejection_block_1h": ("1h", "1H", "trigger_h1_rejection_block"),
    "rejection_block_4h": ("4h", "4H", "trigger_h4_rejection_block"),
}

def _candidate_positions(frame: pd.DataFrame, strategy: Any) -> list[tuple[int, str]]:
    o, h = frame["open"].astype(float), frame["high"].astype(float)
    lo, c = frame["low"].astype(float), frame["close"].astype(float)
    o1, h1, l1, c1 = o.shift(1), h.shift(1), lo.shift(1), c.shift(1)
    o2, c2 = o.shift(2), c.shift(2)
    top0, bot0 = pd.concat((o, c), axis=1).max(axis=1), pd.concat((o, c), axis=1).min(axis=1)
    top1, bot1 = pd.concat((o1, c1), axis=1).max(axis=1), pd.concat((o1, c1), axis=1).min(axis=1)
    top2, bot2 = pd.concat((o2, c2), axis=1).max(axis=1), pd.concat((o2, c2), axis=1).min(axis=1)
    body1, up1, dn1 = (o1 - c1).abs(), h1 - top1, bot1 - l1
    up0, dn0 = h - top0, bot0 - lo
    pw, bw = int(strategy.rb_pivot_lookback_left) + 1, int(strategy.rb_box_length)
    ph = h1.eq(h.rolling(pw).max().shift(1)) & h1.gt(h) & h1.eq(h.rolling(bw, min_periods=1).max().shift(1))
    pl = l1.eq(lo.rolling(pw).min().shift(1)) & l1.lt(lo) & l1.eq(lo.rolling(bw, min_periods=1).min().shift(1))
    rule = str(strategy.rb_body_rule or "").upper()
    if rule == "HARD_BOTH":
        bear_body, bull_body = top2.le(top1) & top0.le(top1), bot2.ge(bot1) & bot0.ge(bot1)
    elif rule == "HARD_RIGHT":
        bear_body, bull_body = top0.le(top1), bot0.ge(bot1)
    elif rule == "HARD_LEFT":
        bear_body, bull_body = top2.le(top1), bot2.ge(bot1)
    elif rule == "CLASSIC":
        bear_body, bull_body = top0.le((h1 + top1) / 2), bot0.ge((l1 + bot1) / 2)
    else:
        return []
    minimum = float(strategy.rb_min_wick_intrusion_pct)
    bear = ph & up1.gt(0) & up0.gt(0) & ((h - top1) / up1 * 100).ge(minimum) & h.lt(h1) & bear_body
    bull = pl & dn1.gt(0) & dn0.gt(0) & ((bot1 - lo) / dn1 * 100).ge(minimum) & lo.gt(l1) & bull_body
    if bool(strategy.rb_use_wick_to_body_filter):
        ratio = float(strategy.rb_wick_to_body_ratio)
        bear, bull = bear & up1.ge(body1 * ratio), bull & dn1.ge(body1 * ratio)
    if bool(strategy.rb_require_confirm_bearish_body):
        bear &= c.lt(o)
    if bool(strategy.rb_require_confirm_bullish_body):
        bull &= c.gt(o)
    events = [(int(i), "SHORT") for i in np.flatnonzero(bear.to_numpy(bool))]
    events += [(int(i), "LONG") for i in np.flatnonzero(bull.to_numpy(bool))]
    return sorted(events)

def _bootstrap(values: list[float], name: str) -> list[float | None]:
    if not values:
        return [None, None]
    data = np.asarray(values)
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:16], 16)
    rng, means = np.random.default_rng(seed), np.empty(2000)
    for i in range(2000):
        means[i] = rng.choice(data, len(data), replace=True).mean()
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]

def _metrics(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["label_status"]] += 1
    mature = [r for r in rows if r["label_status"] in {"FILLED", "NO_FILL"}]
    filled = [r for r in mature if r["label_status"] == "FILLED"]
    opp = [float(r["opportunity_r"]) for r in mature]
    fr = [float(r["opportunity_r"]) for r in filled]
    gp, gl = sum(x for x in fr if x > 0), -sum(x for x in fr if x < 0)
    eq = peak = dd = 0.0
    for value in opp:
        eq += value
        peak, dd = max(peak, eq), max(dd, peak - eq)
    return {
        "technical_events": len(rows), "labeled_opportunities": len(mature),
        "filled": len(filled), "no_fill": counts["NO_FILL"],
        "censored": counts["CENSORED"], "invalid": counts["INVALID"],
        "fill_rate": len(filled) / len(mature) if mature else None,
        "wins": sum(x > 0 for x in fr), "losses": sum(x < 0 for x in fr),
        "breakeven": sum(x == 0 for x in fr),
        "filled_win_rate": sum(x > 0 for x in fr) / len(fr) if fr else None,
        "net_opportunity_r": float(sum(opp)),
        "mean_opportunity_r": float(np.mean(opp)) if opp else None,
        "mean_filled_r": float(np.mean(fr)) if fr else None,
        "profit_factor": float(gp / gl) if gl > 0 else None,
        "max_drawdown_r": dd, "opportunity_expectancy_bootstrap_95": _bootstrap(opp, name),
    }

def _group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for group, aligned in (("all", False), ("bias_aligned", True)):
        chosen = [r for r in rows if not aligned or r["bias_aligned"]]
        output[group] = {}
        for trigger in TFS:
            tf_rows = [r for r in chosen if r["trigger_kind"] == trigger]
            output[group][trigger] = {
                "overall": _metrics(tf_rows, f"{group}:{trigger}"),
                "folds": {
                    str(i): _metrics([r for r in tf_rows if r["fold_index"] == i], f"{group}:{trigger}:{i}")
                    for i in sorted({r["fold_index"] for r in tf_rows})
                },
                "symbols": {
                    s: _metrics([r for r in tf_rows if r["symbol"] == s], f"{group}:{trigger}:{s}")
                    for s in sorted({r["symbol"] for r in tf_rows})
                },
            }
    return output

def _gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    aligned = metrics["bias_aligned"]
    h1 = aligned["rejection_block_1h"]["overall"]
    m15 = aligned["rejection_block_15m"]["overall"]
    eligible = [r for r in aligned["rejection_block_1h"]["folds"].values() if r["labeled_opportunities"] >= 10]
    positive = sum(r["mean_opportunity_r"] is not None and r["mean_opportunity_r"] > 0 for r in eligible)
    low = h1["opportunity_expectancy_bootstrap_95"][0]
    checks = {
        "at_least_100_labeled": h1["labeled_opportunities"] >= 100,
        "at_least_30_fills": h1["filled"] >= 30,
        "positive_opportunity_expectancy": h1["mean_opportunity_r"] is not None and h1["mean_opportunity_r"] > 0,
        "positive_filled_expectancy": h1["mean_filled_r"] is not None and h1["mean_filled_r"] > 0,
        "profit_factor_above_one": h1["profit_factor"] is not None and h1["profit_factor"] > 1,
        "better_than_m15": h1["mean_opportunity_r"] is not None and m15["mean_opportunity_r"] is not None and h1["mean_opportunity_r"] > m15["mean_opportunity_r"],
        "bootstrap_lower_bound_nonnegative": low is not None and low >= 0,
        "positive_in_60pct_eligible_folds": bool(eligible) and positive / len(eligible) >= .60,
    }
    enabled = all(checks.values())
    return {"enable_h1": enabled, "checks": checks, "eligible_folds": len(eligible),
            "positive_folds": positive, "decision": "ENABLE_RB_H1" if enabled else "KEEP_RB_H1_DISABLED",
            "rb_m15_decision": "HARD_DISABLED_REGARDLESS_OF_COMPARISON"}

def _gate_h4(metrics: Mapping[str, Any]) -> dict[str, Any]:
    aligned = metrics["bias_aligned"]
    h4 = aligned["rejection_block_4h"]["overall"]
    h1 = aligned["rejection_block_1h"]["overall"]
    m15 = aligned["rejection_block_15m"]["overall"]
    eligible = [
        row
        for row in aligned["rejection_block_4h"]["folds"].values()
        if row["labeled_opportunities"] >= 10
    ]
    positive = sum(
        row["mean_opportunity_r"] is not None
        and row["mean_opportunity_r"] > 0
        for row in eligible
    )
    low = h4["opportunity_expectancy_bootstrap_95"][0]
    checks = {
        "at_least_100_labeled": h4["labeled_opportunities"] >= 100,
        "at_least_30_fills": h4["filled"] >= 30,
        "positive_opportunity_expectancy": h4["mean_opportunity_r"] is not None and h4["mean_opportunity_r"] > 0,
        "positive_filled_expectancy": h4["mean_filled_r"] is not None and h4["mean_filled_r"] > 0,
        "profit_factor_above_one": h4["profit_factor"] is not None and h4["profit_factor"] > 1,
        "better_than_m15": h4["mean_opportunity_r"] is not None and m15["mean_opportunity_r"] is not None and h4["mean_opportunity_r"] > m15["mean_opportunity_r"],
        "better_than_h1": h4["mean_opportunity_r"] is not None and h1["mean_opportunity_r"] is not None and h4["mean_opportunity_r"] > h1["mean_opportunity_r"],
        "bootstrap_lower_bound_nonnegative": low is not None and low >= 0,
        "positive_in_60pct_eligible_folds": bool(eligible) and positive / len(eligible) >= .60,
    }
    enabled = all(checks.values())
    return {"enable_h4": enabled, "checks": checks, "eligible_folds": len(eligible),
            "positive_folds": positive, "decision": "ENABLE_RB_H4" if enabled else "KEEP_RB_H4_DISABLED",
            "rb_m15_decision": "HARD_DISABLED_REGARDLESS_OF_COMPARISON"}
def compare(snapshot: str, output: str, symbols: Sequence[str], latency: int = 60) -> dict[str, Any]:
    dataset = HistoricalDataset.load(snapshot)
    symbols = tuple(s.upper() for s in symbols)
    start, end = infer_common_strategy_range(dataset, symbols)
    train, test, step = "730D", "180D", "180D"
    config = NarrativeBacktestConfig.build(symbols=symbols, start=start, end=end, initial_capital=100000,
        profile="signal-quality", train=train, test=test, step=step, entry_ttl="15min", max_holding="30D",
        history_limit=300, min_context_bars=300)
    folds = split_walk_forward(start=start, end=end, train=train, test=test, step=step)
    strategy, rows = _default_strategy_factory(), []
    for symbol in symbols:
        prepared = _prepare_symbol(dataset, symbol)
        m1 = prepared["1m"].frame.drop(columns=["bar_close_time"], errors="ignore")
        for trigger, (tf, strategy_key, method_name) in TFS.items():
            candidates = _candidate_positions(prepared[tf].frame, strategy)
            print(f"{symbol} {trigger}: vector candidates={len(candidates)}", flush=True)
            detector = getattr(strategy, method_name)
            for position, side in candidates:
                closed = pd.Timestamp(prepared[tf].close_times[position])
                fold = next((f for f in folds if f.test_start <= closed < f.test_end), None)
                if fold is None:
                    continue
                data = {"4H": prepared["4h"].asof(closed, limit=300),
                        "1H": prepared["1h"].asof(closed, limit=300),
                        "15M": prepared["15m"].asof(closed, limit=300)}
                entry = detector(data[strategy_key], side)
                if entry is None:
                    raise RuntimeError(f"vector/exact mismatch: {symbol} {trigger} {closed} {side}")
                bias, narrative = strategy.calc_narrative(
                    data["4H"],
                    data["1H"],
                    data["15M"],
                    symbol,
                )
                signal = _materialize_entry(strategy=strategy, entry=entry, trigger_kind=trigger, data=data,
                    symbol=symbol, narrative=narrative, factor_vector={}, fvg_side="", fvg_text="")
                decision = closed + pd.Timedelta(int(latency), unit="s")
                digest = hashlib.sha256(f"{symbol}|{trigger}|{closed.isoformat()}|{side}".encode()).hexdigest()[:24]
                opp = _Opportunity(opportunity_id=f"rb-{digest}", decision_event_id=f"rb-event-{digest}",
                    symbol=symbol, bar_close_time=closed, decision_time=decision, trigger_kind=trigger,
                    trigger_tags=(trigger,), gate="ALLOW", gate_reason="RB timeframe comparison", signal=signal)
                label = _label_opportunity(opportunity=opp, m1=m1, config=config, available_end=fold.test_end)
                rows.append({
                    "fold_index": fold.index, "symbol": symbol, "trigger_kind": trigger,
                    "setup_tf": signal["setup_tf"], "bar_close_time": closed.isoformat(),
                    "decision_time": decision.isoformat(), "side": side, "production_bias": str(bias),
                    "bias_aligned": str(bias).upper() == side, "entry_min": signal["entry_min"],
                    "entry_max": signal["entry_max"], "planned_entry": signal["entry_price"],
                    "stop": signal["stop_price"], "tp_prices": json.dumps(signal["tp_prices"]),
                    "label_status": label["label_status"], "fill_time": label["fill_time"],
                    "fill_price": label["fill_price"], "exit_time": label["exit_time"],
                    "opportunity_r": label["opportunity_r"], "label_reason": label["label_reason"],
                    "trigger_event_id": signal.get("trigger_event_id"),
                })
    rows.sort(key=lambda r: (r["bar_close_time"], r["symbol"], r["trigger_kind"]))
    metrics, output_path = _group(rows), Path(output)
    h1_gate = _gate(metrics)
    h4_gate = _gate_h4(metrics)
    gates = {"rejection_block_1h": h1_gate, "rejection_block_4h": h4_gate}
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"schema": SCHEMA, "snapshot": str(Path(snapshot).resolve()),
        "snapshot_manifest_sha256": dataset.manifest_sha256, "symbols": list(symbols),
        "range": {"start": start.isoformat(), "end": end.isoformat()}, "latency_seconds": latency,
        "entry_ttl": config.entry_ttl.isoformat(), "max_holding": config.max_holding.isoformat(),
        "intrabar_policy": "stop-first", "profile": config.profile,
        "walk_forward": {"train": train, "test": test, "step": step, "folds": [f.to_dict() for f in folds]},
        "detector_parameters": {name: getattr(strategy, name) for name in (
            "rb_pivot_lookback_left", "rb_box_length", "rb_min_wick_intrusion_pct", "rb_body_rule",
            "rb_use_wick_to_body_filter", "rb_wick_to_body_ratio",
            "rb_require_confirm_bearish_body", "rb_require_confirm_bullish_body")},
        "metrics": metrics, "deployment_gates": gates, "deployment_gate": h4_gate, "events_csv": "events.csv"}
    (output_path / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(gates, indent=2), flush=True)
    return report

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD", "USDCAD"])
    parser.add_argument("--latency-seconds", type=int, default=60)
    args = parser.parse_args()
    compare(args.snapshot, args.output, args.symbols, args.latency_seconds)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
