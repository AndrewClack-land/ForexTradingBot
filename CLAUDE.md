# Claude operating instructions

This file is the persistent engineering contract for Claude when working in
this repository. Read it before changing strategy, risk, backtest, deployment,
or historical-data code. `README.md` contains the longer operational
procedures; this file defines the non-negotiable rules and research semantics.

## Scope and sources of truth

- Production narrative logic: `core/strategy_narrative.py`
- Live orchestration and execution guards: `main.py`
- Live MT5 sizing: `executors/mt5_executor.py`
- Optional idea-level position adding: `core/position_adding.py`
- Pure factor-score contract: `core/narrative_scoring.py`
- Volatility context/gates: `core/vol_regime.py`
- Causal strategy runner: `backtest/strategy_runner.py`
- Walk-forward splitting and execution simulation:
  `backtest/walkforward.py`, `backtest/simulator.py`
- Conditional factor attribution: `backtest/attribution.py`
- Leakage-aware shadow scoring: `backtest/optimizer.py`
- Historical snapshot ingest/audit: `backtest/lse_ingest.py`,
  `backtest/data.py`
- Release manifest builder: `deploy/build_backtest_release_manifest.py`
- Tests are part of the contract, especially:
  `tests/test_narrative_scoring.py`,
  `tests/test_backtest_optimizer.py`,
  `tests/test_backtest_strategy_runner.py`, and
  `tests/test_htf_context_fixes.py`.

Do not reconstruct factor rules by parsing the human-readable narrative text.
Use the structured `factor_vector`.

## Current production factor contract

There are exactly five directional votes:

| Key | Meaning | Weight |
| --- | --- | ---: |
| `h1_premium_discount` | H1 Premium/Discount | +2 |
| `false_breakout_4h` | false breakout of a 4H fractal | +2 |
| `true_breakout_15m` | true breakout of a 15M fractal | +1 |
| `order_block_1h` | latest active Order Block 1H | +1 |
| `rejection_block_1h` | latest valid and unbroken Rejection Block 1H | +1 |

Important distinctions:

- Daily Premium/Discount is not used.
- Daily fractal breakouts are not used by the score.
- The former five-day dealing-range overlay is removed.
- Volatility is decision-time context and an execution gate, not a `+1`
  directional vote.
- FVG is not a vote. An H1 FVG against a direction increases that direction's
  required score margin by `+1`.
- The production base score margin is exactly `2` unless a separately approved
  strategy change says otherwise:
  `margin_long = 2 + int(fvg_side == "SHORT")` and
  `margin_short = 2 + int(fvg_side == "LONG")`.
- Bias is LONG when `score_long >= score_short + margin_long`, SHORT when the
  symmetric condition holds, and NEUTRAL otherwise.
- H1 Premium/Discount uses the latest closed M15 close inside the latest
  completed H1 candle range.
- `rejection_block_1h` is a score factor. The separate 15M rejection-block
  entry trigger is required to remain quarantined with
  `REJECTION_BLOCK_ENTRY_ENABLED=0` in the production environment and is
  disabled by default in the backtest. The permissive fallback currently
  present in `config.py`/`core/strategy_narrative.py` is not authorization to
  enable it. Do not confuse the trigger with the active RB1H score factor.

`build_factor_vector()` is the canonical arithmetic. A refactor must preserve
the resulting bias, scores, margins, factor rows, and live narrative text.
The structured vector must contain exactly five unique factors and must be
strictly JSON serializable.

`pivotal_without_factor` is a leave-one-factor-out diagnostic: it means that
removing one frozen vote changes the final bias while preserving the same FVG
margin. It is not a causal estimate and not a Shapley value.

## Risk contract

- Aggregate stop-loss risk for one setup/entry must never exceed `1%` of its
  verified fixed starting-capital basis.
- The basis is starting capital, not current equity or a compounded balance.
- Offline research requires an explicit `--initial-capital`. Live execution
  uses either explicit `MT5_INITIAL_CAPITAL` or, when it is `0`, the verified
  one-time per-account snapshot persisted in `ai_data/risk_capital.json`.
  Never silently recapture, refresh, or transfer that snapshot between
  accounts.
- The historical run from code commit `4aff64a` explicitly attests starting
  capital `78,652.24`, making one full-risk setup `786.5224`. These numbers are
  not a permanent default. Every future run/deployment must receive and verify
  its intended starting capital; never infer or silently refresh it.
- Split TP legs share this one aggregate risk budget. Never allocate `1%` to
  each leg.
- A model, factor weight, or execution path must not bypass the per-entry cap.
- Position Adding is a separate, optional live feature and is OFF by default.
  When explicitly enabled, distinct entries in one idea can retain up to the
  existing `IDEA_MAX_RISK_PCT=2%` aggregate idea risk while earlier entries
  may be moved to break-even. The historical strategy runner excludes
  pyramiding. Do not enable, redesign, or claim to backtest Position Adding
  without separate explicit user approval.
- The current backtest uses fixed, non-compounding risk. Variable risk sizing
  requires explicit user approval and must still remain at or below the same
  fixed per-entry `1%` cap.

## Causal backtest invariants

Never weaken these rules:

1. Evaluate the strategy only at a completed M15 decision candle.
2. Supply only D/4H/1H/15M candles closed by that decision time, with the
   configured live-parity history limit.
3. Do not call `Core._closed_bars_view` inside the offline runner; the snapshot
   view has already removed forming candles.
4. A signal may fill only from a later M1 open within its entry range and TTL.
5. Use `stop-first` as the primary conservative label. `tp-first` is
   sensitivity analysis; never combine both policies as independent trades.
6. There are exactly three targets at 1R/2R/3R with weights 50/30/20. TP1
   moves remaining exposure to break-even. `rr_min` is an AI quality floor,
   not a fourth target.
7. Open exposure receives a causal exit at Friday close, maximum holding, or
   the OOS fold boundary.
8. Rolling model features and labels must be known before the test fold.
   Require both decision-time cutoff and matured `exit_time`, with a purge of
   `entry_ttl + max_holding`.
9. Never reuse current live AI state, broker state, or later outcomes in a
   historical fold.
10. Historical LSE candles are a sealed research source, never FxPro Bid/Ask
    execution quotes and never a live trading feed.

The current backtest is a gross OHLCV diagnostic. It has no historical
bid/ask spread, commission, swap, slippage, or tick ordering. Its
production-deterministic profile includes session, Friday, volatility/expected
move, daily setup cap, duplicate-trigger, and post-loss cooldown gates. It
excludes live AI state, broker lot/margin constraints, the bot-wide daily
equity brake, anti-hedge/correlation logic, and Position Adding. Never claim
broker/live parity or authorize a live rollout from gross results alone.

The profile name `production-deterministic` does not by itself guarantee
parity: the CLI still permits overrides such as `--disable-vol-filter`,
`--sessions ALL`, `--enable-rejection-block-entry`,
`--disable-orderblock-entry`, and a different `--htf-score-margin`. Any run
using such an override is a non-parity sensitivity experiment and must be
labelled and reported as such. The canonical baseline keeps volatility/EM
enabled, the configured production sessions, RB15 entry disabled, OB1H entry
enabled, and score margin `2`.

Production-deterministic reports must attest:

- the immutable snapshot manifest;
- the exact Git release commit;
- the root-owned release file manifest;
- the full Python environment lock.

The historical job must run under the unprivileged `forexbot-backtest` account,
without network access and without LSE, MT5, or Telegram credentials. Never
print, copy into Git, or pass through the research job the contents of
`/etc/forexbot/lse.env`.

## Attribution semantics

The report intentionally contains two different selected populations:

- Candidate attribution begins at raw `ENTER`: the baseline score produced a
  directional bias and a production trigger was detected.
- Outcome attribution is narrower: the candidate survived deterministic
  execution gates and actually filled.

Therefore:

- `candidate_factors.csv` describes frozen factor vectors for raw ENTER
  candidates.
- `setup_factor_attribution.csv` joins factor rows to filled setup outcomes.
- `factor_summary.csv` aggregates conditional associations by factor,
  relation, symbol, side, fold, year, quarter, trigger, volatility regime,
  FVG side, and intrabar policy.
- These summaries do not estimate what would have happened to all M15 market
  decisions under different weights.
- Correlated or substitutable factors can both appear non-pivotal.
- Sparse groups must retain their sample-quality warning and shrinkage; do not
  promote small-sample expectancy as a production conclusion.

Keep CSV schemas stable for empty, partially fitted, and fully fitted runs.
The report writer must fail on undeclared columns, and every report file must
be covered by `manifest.json`.

## Shadow score: diagnostic only

`backtest/optimizer.py` fits a regularized rolling Ridge model with target
`net_r_given_production_fill`. It uses only earlier, matured, primary
`stop-first` OOS setups and applies the purge described above.

The shadow score must remain non-executing:

- no hard rejection;
- no change to candidate/setup IDs;
- no change to direction, entry, fill, SL, TP, disposition, or setup count;
- no change to fixed risk.

Its coefficients are conditional associations inside the population admitted
by the existing `+2/+1` score, trigger, gates, and fill process. Never copy
those coefficients directly into production factor weights.

The current shadow report emits:

- rank IC/Spearman correlation;
- top-minus-bottom predicted quintile expectancy;
- stability by fold, symbol, side, and trigger;
- predicted mean versus actual expectancy;
- MAE/RMSE and fitted sample count.

Formal calibration curves, cross-intrabar-policy shadow robustness, and
transaction-cost stress are required future/manual evaluations. The current
optimizer labels only `stop-first`; it does not fit or score a separate
`tp-first` model, and the LSE snapshot lacks broker costs.

## Allowed optimization path

Everything in this section is an offline challenger/research path only. It
does not authorize a live strategy, config, executor, or deployment change.
Any live activation requires separate explicit user approval after frozen OOS
evidence and broker-cost stress. The user prefers adaptation to market
conditions over simply rejecting more entries; preserve that intent in
research.

Recommended sequence:

1. Use detailed attribution to identify stable and unstable contexts.
2. Keep the shadow score as a quality/ranking signal.
3. Test contextual trigger selection when several production triggers are
   available.
4. Test smooth volatility adaptation of entry range, TTL, stop geometry, and
   targets while preserving aggregate fixed-capital risk.
5. Use hierarchical/shrunk symbol, side, session, and volatility adjustments
   instead of independent small-sample models.
6. Only then build a true counterfactual weight optimizer.

A valid replacement-weight optimizer requires a new v2 research population:

- freeze factor vectors at every completed M15 decision, not only baseline
  ENTER rows;
- generate technical LONG and SHORT opportunities independently of the
  baseline bias;
- fit only inside each training interval;
- constrain weights to be finite and non-negative;
- regularize toward the current `[2, 2, 1, 1, 1]` contract;
- freeze the fitted model before replaying the next OOS interval;
- compare against the untouched baseline with the same execution simulator.

The existing `--train` interval does not fit replacement factor weights. It
defines rolling history/OOS boundaries for the fixed baseline. The current
shadow model trains only on earlier matured baseline OOS fills.

Do not optimize only total Net R. Use a multi-objective decision including
expectancy, profit factor, maximum drawdown, fold/market/direction stability,
sample size, and transaction-cost stress. A nonlinear model may be a shadow
challenger only after an interpretable regularized baseline.

Current attribution evidence and results cover the imported FX symbols. Do
not generalize them to GOLD/XAUUSD without a separately sealed, audited
dataset and independent OOS evaluation.

Operational safety gates remain separate from score optimization. Do not
silently remove the volatility, session, Friday, frequency, duplicate-trigger,
or post-loss protections merely to increase backtest trade count.

## Engineering workflow

Before work:

1. Inspect `git status`; preserve unrelated or untracked user files.
2. Read the exact source and tests listed above.
3. Inspect active VPS jobs before starting, replacing, or restarting anything.
4. Never expose secrets in commands, logs, reports, commits, or chat output.

After code changes:

```bash
python -m ruff check core/narrative_scoring.py \
  backtest/attribution.py backtest/optimizer.py \
  backtest/strategy_runner.py
python -m pytest -q -p no:cacheprovider --basetemp <fresh-temp-dir>
git diff --check
```

Also verify:

- old/new score equivalence for the production factor combinations;
- strict JSON serialization;
- future-outcome mutation cannot alter earlier models/scores;
- input row permutation cannot alter model IDs or report hashes;
- shadow scoring does not mutate executions or risk;
- empty and non-empty CSV headers are identical.

## Git and VPS release discipline

- Commit the complete tested change; do not deploy an uncommitted working
  tree.
- Push the requested commit to the configured Git remote.
- Build the VPS research release only with `git archive` from the exact commit.
- Run every multi-step release recipe fail-closed with
  `set -euo pipefail`; do not continue after archive, manifest, dependency,
  freeze, smoke, or verification failure.
- Never extract over an existing release. Use a new immutable directory under
  `/opt/forexbot-backtest/releases/<full-commit>`.
- Build and hash the release manifest and environment lock before atomically
  switching `/opt/forexbot-backtest/current`.
- Before switching `current`, run a short production-deterministic smoke
  against the exact new candidate release path. The README helper normally
  resolves `current`; for this pre-switch smoke set `release_root` explicitly
  to `/opt/forexbot-backtest/releases/<full-commit>` in its sandbox
  properties. Verify the attestation, factor coverage, manifest, and expected
  report files, then switch `current` atomically.
- Full runs must be disconnect-safe, nonblocking `systemd-run` jobs.
- Do not restart or modify `forexbot.service` for a research-only release.
  Confirm that the live bot remains active.
- Report the exact commit, release path, unit name, output path, test result,
  and whether the full calculation is still running or complete.

The reference snapshot is normally:

```text
/srv/forexbot-backtest/snapshots/fx-2020-2026-v1
```

The full attribution run launched from commit `4aff64a` uses:

```text
unit:   forexbot-backtest-wfo-factor-4aff64a.service
output: /var/lib/forexbot-backtest/runs/fx-wfo-factor-2020-2026-4aff64a-both
```

Always inspect the unit before assuming it is still active or before starting
a replacement. The output directory is atomically published and must not
already exist or be reused.

Useful read-only status commands:

```bash
sudo -u forexbot-backtest env \
  PYTHONPATH=/opt/forexbot-backtest/current/app \
  /opt/forexbot-backtest/current/venv/bin/python -m backtest verify \
  --data /srv/forexbot-backtest/snapshots/fx-2020-2026-v1
sudo systemctl show forexbot-backtest-wfo-factor-4aff64a.service \
  -p ActiveState -p SubState -p Result -p ExecMainStatus
sudo journalctl \
  -u forexbot-backtest-wfo-factor-4aff64a.service -n 20 --no-pager
sudo journalctl -fu forexbot-backtest-wfo-factor-4aff64a.service
systemctl is-active forexbot.service
```
