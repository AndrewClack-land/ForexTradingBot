"""Pure, serializable scoring contract for the narrative factor ensemble.

The production strategy detects market structures.  This module only turns
those already-known-at-decision-time votes into scores and a bias.  Keeping
the arithmetic pure makes the live narrative and offline attribution use the
same weights, margins, and pivotal-factor definition.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional


# v3 adds the two 1H fractal-breakout rows, so the vector carries eight rows.
# A named contract may set a row's effective weight to zero while retaining
# its observation for diagnostics. Frozen v1/v2 weight models are
# five-dimensional and must be refitted rather than reinterpreted against this
# layout.
FACTOR_VECTOR_SCHEMA = "narrative-factor-vector/v3"


@dataclass(frozen=True)
class FactorDefinition:
    key: str
    label: str
    configured_weight: int


# ``configured_weight`` is the canonical legacy/reference weight attached to
# the stable row. ``effective_weight`` is selected by the named contract and is
# the only value used for scoring. The 1H FVG regime carries a zero reference
# weight because under the legacy contract it only raises the opposing side's
# margin. Keeping zero-weight diagnostic rows in the schema lets all contracts
# share one factor-vector layout.
FACTOR_DEFINITIONS = (
    FactorDefinition(
        "h1_premium_discount",
        "H1 Premium/Discount",
        2,
    ),
    FactorDefinition(
        "false_breakout_4h",
        "4H false fractal breakout",
        2,
    ),
    FactorDefinition(
        "true_breakout_15m",
        "15M true fractal breakout",
        1,
    ),
    FactorDefinition(
        "false_breakout_1h",
        "1H false fractal breakout",
        1,
    ),
    FactorDefinition(
        "true_breakout_1h",
        "1H true fractal breakout",
        1,
    ),
    FactorDefinition(
        "order_block_1h",
        "Order Block 1H",
        1,
    ),
    FactorDefinition(
        "rejection_block_1h",
        "Valid Rejection Block 1H",
        1,
    ),
    FactorDefinition(
        "fvg_regime_1h",
        "1H FVG regime (IMFVG)",
        0,
    ),
)

FACTOR_WEIGHTS = {
    definition.key: definition.configured_weight
    for definition in FACTOR_DEFINITIONS
}

LEGACY_FACTOR_WEIGHTS = dict(FACTOR_WEIGHTS)

# Challenger: H1 Premium/Discount 2->1, Order Block 1H 1->2, and the FVG regime
# promoted from a margin penalty to a +1 directional vote. H1 P/D and the FVG
# regime both effectively always vote, so at 1 each they cancel when they
# disagree instead of H1 P/D's 2 dominating.
CHALLENGER_FACTOR_WEIGHTS = {
    "h1_premium_discount": 1,
    "false_breakout_4h": 2,
    "true_breakout_15m": 1,
    # The 1H breakout rows mirror the frozen legacy reference so this arm keeps
    # isolating the FVG change alone. It stays one point heavier than legacy
    # (10 vs 9) for the same reason it always was: here the FVG row votes.
    "false_breakout_1h": 1,
    "true_breakout_1h": 1,
    "order_block_1h": 2,
    "rejection_block_1h": 1,
    "fvg_regime_1h": 1,
}

# Production successor: H1 Premium/Discount remains observable in the frozen
# factor vector, but cannot contribute to either directional score. All other
# weights and the legacy FVG margin rule remain unchanged so this contract
# isolates one policy change.
NO_H1PD_FACTOR_WEIGHTS = dict(LEGACY_FACTOR_WEIGHTS)
NO_H1PD_FACTOR_WEIGHTS["h1_premium_discount"] = 0

FACTOR_CONTRACT_LEGACY = "v1-fvg-margin"
FACTOR_CONTRACT_FVG_VOTE = "v2-fvg-vote"
FACTOR_CONTRACT_NO_H1PD = "v3-no-h1pd-fvg-margin"

# ``fvg_margin_enabled`` and the FVG weight are deliberately mutually
# exclusive. As a margin rule the regime can only brake the opposing entry; as
# a vote it also relaxes its own direction by one point. Enabling both would
# shift the threshold by two points for a factor that never abstains.
FACTOR_CONTRACTS = {
    FACTOR_CONTRACT_LEGACY: {
        "weights": dict(LEGACY_FACTOR_WEIGHTS),
        "fvg_margin_enabled": True,
    },
    FACTOR_CONTRACT_FVG_VOTE: {
        "weights": dict(CHALLENGER_FACTOR_WEIGHTS),
        "fvg_margin_enabled": False,
    },
    FACTOR_CONTRACT_NO_H1PD: {
        "weights": dict(NO_H1PD_FACTOR_WEIGHTS),
        "fvg_margin_enabled": True,
    },
}


def resolve_factor_contract(name: Optional[str]) -> dict[str, Any]:
    """Return the frozen weight/margin pair for a named factor contract."""

    # ``None`` retains the legacy API default for historical/offline callers.
    # An explicitly empty environment/config value is a deployment error and
    # must not silently reactivate the legacy H1 Premium/Discount vote.
    if name is None:
        key = FACTOR_CONTRACT_LEGACY
    else:
        key = str(name).strip()
        if not key:
            raise ValueError("factor contract must not be empty")
    if key not in FACTOR_CONTRACTS:
        raise ValueError(
            f"unknown factor contract {key!r}; "
            f"expected one of {sorted(FACTOR_CONTRACTS)}"
        )
    contract = FACTOR_CONTRACTS[key]
    return {
        "name": key,
        "weights": dict(contract["weights"]),
        "fvg_margin_enabled": bool(contract["fvg_margin_enabled"]),
    }


def _factor_contract_identity(
    *,
    effective_weights: Mapping[str, float],
    fvg_margin_enabled: bool,
    explicit: Optional[str],
) -> str:
    """Bind a vector to one named contract or an explicit research identity."""

    normalized_weights = {
        key: float(value) for key, value in effective_weights.items()
    }
    matches = [
        name
        for name, contract in FACTOR_CONTRACTS.items()
        if bool(contract["fvg_margin_enabled"]) == bool(fvg_margin_enabled)
        and {
            key: float(value)
            for key, value in contract["weights"].items()
        }
        == normalized_weights
    ]

    if explicit is not None:
        identity = str(explicit).strip()
        if not identity:
            raise ValueError("factor_contract identity must not be empty")
        if identity in FACTOR_CONTRACTS and identity not in matches:
            raise ValueError(
                "factor_contract identity does not match the supplied "
                "weights/FVG-margin mode"
            )
        return identity
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            "factor_contract identity is required for non-canonical weights"
        )
    raise ValueError(
        "factor_contract identity is required because multiple canonical "
        "contracts match"
    )


_SIDES = {"LONG", "SHORT", "NEUTRAL"}


def _side(value: Any) -> str:
    normalized = str(value or "NEUTRAL").upper()
    return normalized if normalized in _SIDES else "NEUTRAL"


def bias_from_scores(
    *,
    score_long: float,
    score_short: float,
    margin_long: float,
    margin_short: float,
) -> str:
    """Return the exact production bias for two directional scores."""

    if score_long >= score_short + margin_long:
        return "LONG"
    if score_short >= score_long + margin_short:
        return "SHORT"
    return "NEUTRAL"


def build_factor_vector(
    votes: Mapping[str, Mapping[str, Any]],
    *,
    base_margin: int,
    fvg_side: str = "NEUTRAL",
    weights: Optional[Mapping[str, float]] = None,
    fvg_margin_enabled: bool = True,
    factor_contract: Optional[str] = None,
) -> dict[str, Any]:
    """Build a deterministic factor vector from frozen directional votes.

    ``votes`` may contain ``side`` and an optional JSON-safe ``evidence``
    mapping for every configured factor. Missing factors are retained as
    explicit ABSENT rows so absence never gets confused with a zero-valued
    observation.

    ``fvg_side`` is recorded on every vector for attribution. It only moves the
    margins while ``fvg_margin_enabled`` is set; under the FVG-vote contract the
    regime enters through the ``fvg_regime_1h`` vote instead and the margins
    stay symmetric.

    Canonical weights plus FVG-margin mode resolve to their immutable named
    contract automatically. Non-canonical research weights must supply an
    explicit non-empty ``factor_contract`` identity.
    """

    effective_weights = {
        definition.key: float(
            (weights or {}).get(
                definition.key,
                definition.configured_weight,
            )
        )
        for definition in FACTOR_DEFINITIONS
    }
    invalid_weights = {
        key: value
        for key, value in effective_weights.items()
        if not math.isfinite(value) or value < 0
    }
    if invalid_weights:
        raise ValueError(
            f"factor weights must be finite and non-negative: {invalid_weights}"
        )
    contract_identity = _factor_contract_identity(
        effective_weights=effective_weights,
        fvg_margin_enabled=fvg_margin_enabled,
        explicit=factor_contract,
    )
    normalized_fvg = _side(fvg_side)
    normalized_margin = max(1, int(base_margin))
    margin_penalty = bool(fvg_margin_enabled)
    margin_long = normalized_margin + (
        1 if margin_penalty and normalized_fvg == "SHORT" else 0
    )
    margin_short = normalized_margin + (
        1 if margin_penalty and normalized_fvg == "LONG" else 0
    )

    factor_rows: list[dict[str, Any]] = []
    score_long = 0.0
    score_short = 0.0
    for definition in FACTOR_DEFINITIONS:
        raw = votes.get(definition.key) or {}
        vote_side = _side(raw.get("side"))
        present = bool(raw.get("present", vote_side != "NEUTRAL"))
        if not present:
            vote_side = "NEUTRAL"
        weight = effective_weights[definition.key]
        long_contribution = weight if vote_side == "LONG" else 0.0
        short_contribution = weight if vote_side == "SHORT" else 0.0
        score_long += long_contribution
        score_short += short_contribution
        evidence = raw.get("evidence")
        factor_rows.append(
            {
                "key": definition.key,
                "label": definition.label,
                "present": present,
                "vote_side": vote_side,
                "configured_weight": definition.configured_weight,
                "effective_weight": weight,
                "long_contribution": long_contribution,
                "short_contribution": short_contribution,
                "evidence": dict(evidence)
                if isinstance(evidence, Mapping)
                else {},
            }
        )

    selected_bias = bias_from_scores(
        score_long=score_long,
        score_short=score_short,
        margin_long=margin_long,
        margin_short=margin_short,
    )
    for row in factor_rows:
        without_long = score_long - float(row["long_contribution"])
        without_short = score_short - float(row["short_contribution"])
        without_bias = bias_from_scores(
            score_long=without_long,
            score_short=without_short,
            margin_long=margin_long,
            margin_short=margin_short,
        )
        if selected_bias == "NEUTRAL":
            selected_contribution = 0.0
        elif row["vote_side"] == selected_bias:
            selected_contribution = float(row["effective_weight"])
        elif row["vote_side"] in {"LONG", "SHORT"}:
            selected_contribution = -float(row["effective_weight"])
        else:
            selected_contribution = 0.0
        row["selected_side_contribution"] = selected_contribution
        row["bias_without_factor"] = without_bias
        row["pivotal_without_factor"] = without_bias != selected_bias

    return {
        "schema": FACTOR_VECTOR_SCHEMA,
        "factor_contract": contract_identity,
        "bias": selected_bias,
        "score_long": score_long,
        "score_short": score_short,
        "score_delta_long_minus_short": score_long - score_short,
        "base_margin": normalized_margin,
        "margin_long": margin_long,
        "margin_short": margin_short,
        "fvg_side": normalized_fvg,
        "fvg_margin_enabled": margin_penalty,
        "factors": factor_rows,
    }


def rescore_factor_vector(
    factor_vector: Mapping[str, Any],
    *,
    weights: Mapping[str, float],
    factor_contract: Optional[str] = None,
) -> dict[str, Any]:
    """Re-score frozen votes without re-reading candles or future outcomes.

    Registered weights auto-resolve their contract. Custom research weights
    require an explicit ``factor_contract`` identity.
    """

    votes = {
        str(row.get("key")): {
            "present": bool(row.get("present")),
            "side": row.get("vote_side"),
            "evidence": row.get("evidence") or {},
        }
        for row in factor_vector.get("factors", ())
        if isinstance(row, Mapping) and row.get("key")
    }
    return build_factor_vector(
        votes,
        base_margin=int(factor_vector.get("base_margin") or 1),
        fvg_side=str(factor_vector.get("fvg_side") or "NEUTRAL"),
        weights=weights,
        fvg_margin_enabled=bool(
            factor_vector.get("fvg_margin_enabled", True)
        ),
        factor_contract=factor_contract,
    )
