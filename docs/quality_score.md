# Hierarchical quality score

The quality score is diagnostic. It runs only in the background shadow-candidate
worker and must never select, veto, resize, or execute a live order.

## Causal phases

- `DECISION` uses information available when the candidate is observed: symbol,
  side, trigger, H1 stop distance in ATR, session/weekday, FVG state and age,
  causal spread, and intended execution policy.
- `TOUCH` may be evaluated only after a strict tick touch is known. Realized
  time-to-exact-retest is post-decision information and is forbidden in a
  `DECISION` profile.

The decision score reports TP1 probability, expected net R, uncertainty,
effective sample size, and the hierarchy level used. Sparse
`trigger x symbol x side` cells shrink toward their parent instead of using a
raw win rate.

## Shadow arbitration

Simultaneous candidates are ranked by a conservative lower bound of net R,
then by the TP lower bound, expected net R, and stable opportunity identity.
The research policy can apply a profile-frozen penalty when the planned stop is
outside the inclusive `0.75-1.50 ATR` band and a causal portfolio-correlation
penalty. These fields are stored separately from the legacy shadow rank and the
production waterfall remains authoritative.

## Training rules

- One row is one independent idea, never a split TP leg or a repeated scan.
- Unselected candidates are unlabeled, not losses.
- `TIME`, incomplete broker history, and unresolved outcomes are censored.
- A net-R label already containing broker costs must not have a cost profile
  subtracted again.
- Profiles are immutable, point-in-time artifacts; future-dated profiles are
  rejected.

The append-only quality event ledger links candidate decisions, scores,
executions, broker outcomes, and explicit correction events without rewriting
prior evidence.

## Enabling the diagnostics

Set `SHADOW_QUALITY_PROFILE_PATH` to a frozen
`hierarchical-quality-profile-v1` JSON while `SHADOW_CANDIDATE_LEDGER_ENABLED=1`.
`main.Core` loads the profile fail-open at startup and passes the scorer only
into `_write_shadow_candidate_batch`; results are written by
`ShadowCandidateLedger.record_quality_ranking` into the v2 `quality_*` columns.
A missing, invalid, or future-dated profile disables scoring while candidate
capture continues, and a scoring failure never suppresses the scan row.
`tests/test_shadow_quality_worker.py` holds the non-interference proof:
production signals, outcomes, and strategy data stay byte-identical with the
scorer present, broken, or absent.

Profiles are fitted offline with
`core.hierarchical_quality_score.fit_quality_profile` from rows produced by
`tools/export_quality_training_rows.py` (one row per independent idea,
`NET_R_AFTER_BROKER_COSTS` basis).
