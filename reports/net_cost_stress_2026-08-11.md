# FX OOS cost stress — 2026-08-11

## Scope

This is a post-hoc sensitivity calculation, not a point-in-time historical
cost backtest. It reprices two completed gross OOS populations:

- `fx-wfo-baseline-df17610`: 1,676 setups;
- `fx-optimize-v2-df17610`: 4,570 setups.

The source candle snapshot is
`fx-2020-2026-v1` (manifest SHA-256
`e5500291e255a3f3acc7e638d2c0a5d6cfc4acb2453c750b365c94d3bd361d62`).

For each setup:

```text
cost_R = (spread_price + commission_price + 2 * slippage_per_side) /
         abs(entry_price - stop_price)
```

Commission is USD 7.00 round-turn per lot. For EURUSD and GBPUSD,
`commission_price = 7 / 100000`; for USDCAD the quote-to-account conversion
uses `1 / entry_price`.

Forward FxPro spread evidence covers 2026-07-28 through 2026-08-11,
06:30–21:00 UTC, with 594 eligible M15 bars and about 1.064 million DOM
snapshots per symbol:

| Symbol | Weighted mean spread | P95 M15 mean spread |
|---|---:|---:|
| EURUSD | 0.0000211875 | 0.00003000 |
| GBPUSD | 0.0000627680 | 0.00009000 |
| USDCAD | 0.0000354953 | 0.00009000 |

## Aggregate results

### Production-policy baseline (1,676)

| Scenario | Net R | Exp. R/setup | PF | Max DD R | WR |
|---|---:|---:|---:|---:|---:|
| Gross OHLCV | +9.754 | +0.00582 | 1.011 | 26.1 | 44.99% |
| Commission only | -106.428 | -0.06350 | 0.891 | 112.6 | 44.87% |
| Commission + 0.5x mean spread | -134.812 | -0.08044 | 0.864 | 140.0 | 44.81% |
| Commission + mean spread | -163.195 | -0.09737 | 0.838 | 167.7 | 44.81% |
| Commission + P95 spread | -206.639 | -0.12329 | 0.799 | 210.3 | 44.75% |

With commission + mean spread, every one of the nine OOS folds is negative.
The gross result had six positive folds.

### Optimize-v2 population (4,570)

| Scenario | Net R | Exp. R/setup | PF | Max DD R | WR |
|---|---:|---:|---:|---:|---:|
| Gross OHLCV | +29.896 | +0.00654 | 1.012 | 76.7 | 45.91% |
| Commission only | -262.347 | -0.05741 | 0.899 | 264.0 | 45.89% |
| Commission + 0.5x mean spread | -334.165 | -0.07312 | 0.873 | 335.8 | 45.84% |
| Commission + mean spread | -405.982 | -0.08884 | 0.848 | 407.5 | 45.80% |
| Commission + P95 spread | -514.944 | -0.11268 | 0.811 | 516.4 | 45.78% |

Adding an assumed one-pip slippage per side lowers the optimize-v2 population
to -1,263.5R. This is a deliberately severe stress band, not measured
slippage.

## Baseline symbol × trigger result at mean spread + commission

| Symbol / trigger | N | Gross R | Cost-adjusted R | Exp. R | PF |
|---|---:|---:|---:|---:|---:|
| EURUSD / OB1H | 87 | +6.79 | +1.85 | +0.0213 | 1.040 |
| EURUSD / Pivot reclaim | 459 | +3.94 | -41.95 | -0.0914 | 0.848 |
| GBPUSD / OB1H | 67 | -9.36 | -13.35 | -0.1992 | 0.685 |
| GBPUSD / Pivot reclaim | 481 | +8.51 | -46.07 | -0.0958 | 0.846 |
| USDCAD / OB1H | 90 | -1.38 | -7.25 | -0.0805 | 0.839 |
| USDCAD / Pivot reclaim | 492 | +1.26 | -56.43 | -0.1147 | 0.810 |

EURUSD OB1H is the only positive aggregate bucket under this sensitivity, but
it is positive in only 4/9 folds. Fold 8 is -5.06R after costs. It is not a
robust integration result.

## Interpretation and limitations

- These 2026 forward costs were not available in the earlier OOS folds.
  Therefore they must never choose or rank historical candidates. They are an
  explicitly static post-hoc stress scenario.
- Pure broker slippage is not measured because request-time bid/ask was not
  persisted. Swap is also excluded because both-side, symbol-specific rates
  and rollover rules are incomplete.
- The LSE OHLCV price convention is not an executable FxPro bid/ask stream.
  Commission-only is included as a lower-bound check; it is already negative.
- No claim of historical net P&L is made. The conclusion is narrower: the
  observed gross edge is much smaller than plausible execution friction, so a
  frequency increase cannot be accepted on WinRate alone.

## Acceptance implication

Future policy arms should be accepted only on frozen OOS support after:

1. rank selection is independent of future/static cost measurements;
2. realized results are repriced under versioned cost scenarios;
3. net expectancy and PF remain positive by symbol, trigger, and fold;
4. exact-retest, multi-candidate, or M1-trigger additions do not merely add
   low-margin gross setups.
