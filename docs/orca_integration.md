# ORCA regime and Random Forest integration

This repository implements ORCA as an external, diagnostic market-regime
context. It is not a replacement for RB, Pivot, OB, Cluster, Quote Pressure,
technical stop geometry, or broker execution.

## Scientific contract

The source paper forecasts two ten-trading-day SPY events:

- rally: endpoint return above +3%;
- crash: minimum path return below -7%.

Balanced Crisis Detection AUC is the geometric mean
`sqrt(AUC_rally * AUC_crash)`. It is a model-selection metric, not WinRate,
probability calibration, expected R, or a trade-entry gate. If either OOS task
contains only one class, BCD-AUC is undefined.

The paper is internally inconsistent about its feature inventory (the prose
says 127 spectral plus 79 traditional features, while Table I reports 179 plus
27), calibration method, and rolling-versus-expanding folds. The code therefore
uses explicit versioned feature registries and never claims an exact paper
replication.

## Causal boundaries

- Only fully closed, aligned D1 prices may enter ORCA features.
- Rally/crash labels become known after the full ten-day horizon.
- Every WFO fold has an explicit purge/gap of at least that horizon.
- Probability calibration, imputation, vocabulary, and model selection are fit
  on past data only.
- The MT5 process never imports scikit-learn or calls a market-data vendor.
- The monitor publishes an immutable JSON snapshot with market-as-of and
  known-at timestamps. Future, stale, corrupt, or mismatched artifacts are
  unavailable rather than zero-risk.
- Actual fill, exit, and exact-retest duration are forbidden decision-time RF
  features. A historical OOF fill-time prediction may be used later.

## Runtime topology

```text
immutable 24-asset D1 panel
        -> 60D / 120D / EWM correlations
        -> spectral + signed network snapshot
        -> two walk-forward Random Forest classifiers
        -> atomic ORCA prediction/snapshot JSON
        -> monitoring dashboard
        -> optional shadow candidate RF context
```

Model refresh and model promotion are deliberately separate. A scheduled job
may train a challenger, but only an explicit promotion step after causal WFO
can replace the champion artifact.

The production data job uses the paper's exact ETF universe and the official
EODHD end-of-day endpoint. It consumes adjusted_close, maps every exchange
date to the conservative next-day 12:00 UTC availability timestamp, excludes
future rows, forward-fills at most five observations, requires all 24 assets
and at least 372 aligned rows, and atomically preserves the previous panel on
any partial download or validation failure.

## Candidate Random Forest

The candidate model estimates utility for each technically valid candidate; it
does not invent a side or price level. Its three heads estimate `P(fill)`,
`P(TP | fill)`, and `E(gross R | fill)`. Direction is a consequence of the
highest conservative candidate utility, not a direct LONG/SHORT classifier.

The first integration remains shadow-only. Production continues to use the
existing deterministic direction score and waterfall until a paired,
single-TP-compatible WFO proves higher net expectancy without unacceptable
drawdown or symbol/side concentration.

main.py loads the portable candidate profile only when the shadow ledger,
cost profile and exact strategy/factor/target contracts all match. Scoring is
performed in the existing post-decision worker on deep copies; only rf_*
fields reach the diagnostic ledger and rf_executing is always false.

## Monitor

Run the Linux-side service with a normalized wide Parquet/CSV price panel:

```bash
python -m monitoring.orca_monitor \
  --prices /srv/forexbot-backtest/orca/prices.parquet \
  --snapshot /var/lib/forexbot-orca/latest.json
```

The browser UI is served on `127.0.0.1:8765` by default. It shows every
correlation estimator, signed edges, deterministic three-dimensional spectral
coordinates, eigenvector-centrality node sizes, eigenvalues, absorption ratio,
effective rank, prediction lineage, calibration state, and freshness.

The paper publishes only the zero-exposure and 1.5x exposure zones. Intermediate
0.3/0.7/1.0/1.2 rules are not specified completely, so this implementation
returns no intermediate exposure instead of inventing it.

## Native VPS deployment

ORCA never runs inside the Wine/MT5 Python process. The three native services
use /opt/forexbot-orca/venv/bin/python, installed from
requirements-orca-runtime.txt:

- orca-eodhd.timer publishes the causal panel at 00:15 and 12:15 UTC;
- a successful ingest starts orca-prediction.service;
- orca-prediction.timer is a 00:30/12:30 fallback;
- orca-monitor.service refreshes the local dashboard every five minutes.

Real environment files live under /etc/forexbot-orca, outside the immutable
application release. Do not enable the timers until an EODHD token, a complete
panel, a manually promoted WFO artifact under the root-owned
/opt/forexbot-orca/models directory, the artifact SHA-256 and the full
feature-registry SHA-256 are present. The ingestor cannot write that model
directory. Missing or mismatched inputs are unavailable, never an implicit
low-risk regime.
