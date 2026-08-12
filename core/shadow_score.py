"""Non-executing live shadow score.

`backtest/optimizer.py` fits a regularized rolling Ridge model whose target is
``net_r_given_production_fill``.  This module replays one *frozen* model from a
completed WFO run against a live signal so the score can be journalled beside
the trade.

The contract is deliberately one-directional: this module reads a signal and
returns annotation fields.  It never rejects, reorders, resizes, or otherwise
touches direction, entry, stop, targets, disposition, or risk.  A missing,
unreadable, or unfitted model is not an error — the annotation is simply
absent and trading continues unchanged.

Feature construction is imported from the optimizer rather than reimplemented,
so a live score is computed by the exact code path that produced the fit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from backtest.optimizer import _TRIGGERS, SHADOW_SCORE_SCHEMA, _feature_vector


# Mirror of ``backtest.strategy_runner._trigger_kind``.  It is duplicated
# instead of imported because that module pulls the whole offline report stack
# into the live process; ``tests/test_shadow_score_live.py`` pins the two
# implementations to the same answers so the copy cannot silently drift.
_STRUCTURED_TRIGGERS = frozenset(
    {
        "rejection_block_15m",
        "absorption_15m",
        "fxpro_cluster_rejection_15m",
        "fxpro_quote_pressure_rejection_15m",
        "h1_pivot_reclaim_15m",
        "order_block_1h",
        "turtle_soup_15m",
    }
)

_REASON_PREFIXES = (
    ("rejectionblock 15m", "rejection_block_15m"),
    ("absorption 15m", "absorption_15m"),
    ("fxpro cluster rejection 15m", "fxpro_cluster_rejection_15m"),
    ("fxpro quote pressure rejection 15m", "fxpro_quote_pressure_rejection_15m"),
    ("turtlesoup 15m", "turtle_soup_15m"),
    ("h1 pivothigh reclaim", "h1_pivot_reclaim_15m"),
    ("h1 pivotlow reclaim", "h1_pivot_reclaim_15m"),
    ("orderblock touch", "order_block_1h"),
)


def normalize_trigger_kind(signal: Mapping[str, Any]) -> str:
    """Canonical trigger family for a signal, matching the offline runner."""

    structured = str(signal.get("trigger_kind") or "").strip().lower()
    if structured in _STRUCTURED_TRIGGERS:
        return structured
    reason = str(signal.get("trigger_reason") or "").strip().lower()
    for prefix, kind in _REASON_PREFIXES:
        if reason.startswith(prefix):
            return kind
    return "unknown"


def tp1_em_ratio(
    entry_price: Any,
    tp_prices: Sequence[Any],
    em_1d: Any,
) -> Optional[float]:
    """Nearest-target distance in expected-move units.

    Mirrors the offline ``vol_tp1_em_ratio`` so the live feature and the fitted
    feature mean the same thing.  Returns ``None`` when the inputs cannot
    produce a finite ratio; the model already carries an explicit
    missing-indicator for that case.
    """

    try:
        entry = float(entry_price)
        expected_move = float(em_1d)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(entry) or not np.isfinite(expected_move) or expected_move <= 0:
        return None
    distances = []
    for target in tp_prices or ():
        try:
            value = float(target)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            distances.append(abs(value - entry))
    if not distances:
        return None
    return round(float(min(distances) / expected_move), 6)


class LiveShadowScorer:
    """One frozen WFO shadow model, replayed against live signals."""

    def __init__(
        self,
        *,
        model_id: str,
        fold_index: int,
        feature_names: Sequence[str],
        intercept: float,
        coefficients: np.ndarray,
        vol_mean: float,
        vol_scale: float,
        ratio_mean: float,
        ratio_scale: float,
        source_path: str,
    ) -> None:
        self.model_id = model_id
        self.fold_index = fold_index
        self.feature_names = tuple(feature_names)
        self.intercept = float(intercept)
        self.coefficients = coefficients
        self.vol_mean = float(vol_mean)
        self.vol_scale = float(vol_scale)
        self.ratio_mean = float(ratio_mean)
        self.ratio_scale = float(ratio_scale)
        self.source_path = source_path
        # ``_feature_names`` emits one ``trigger:`` column per family except the
        # first, which stays the dropped reference level. Reconstruct the
        # vocabulary this model was actually fitted with from its own columns
        # plus that baseline, rather than trusting the current ``_TRIGGERS``:
        # the two already differ, because models fitted before Absorption was
        # archived carry a different column set.
        trigger_columns = {
            name.split(":", 1)[1]
            for name in self.feature_names
            if name.startswith("trigger:")
        }
        self.trigger_vocab = (
            frozenset(trigger_columns | {_TRIGGERS[0]})
            if trigger_columns
            else frozenset(_TRIGGERS)
        )

    @classmethod
    def load(cls, path: Optional[Path]) -> Optional["LiveShadowScorer"]:
        """Load the newest fitted fold from a ``shadow_models.json``.

        Returns ``None`` for every failure mode — absent path, unreadable
        file, wrong schema, or a run whose folds were all ``NOT_FIT``.
        """

        if path is None:
            return None
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

        models = payload.get("models") if isinstance(payload, Mapping) else None
        if not isinstance(models, list):
            return None

        fitted = [
            model
            for model in models
            if isinstance(model, Mapping)
            and model.get("status") == "FIT"
            and model.get("model_id")
            and isinstance(model.get("coefficients"), Mapping)
            and isinstance(model.get("feature_names"), list)
        ]
        if not fitted:
            return None

        newest = max(
            fitted,
            key=lambda model: (
                str(model.get("test_end") or ""),
                int(model.get("fold_index") or 0),
            ),
        )
        if str(newest.get("schema") or "") != SHADOW_SCORE_SCHEMA:
            return None

        names = [str(name) for name in newest["feature_names"]]
        raw_coefficients = newest["coefficients"]
        try:
            coefficients = np.asarray(
                [float(raw_coefficients[name]) for name in names],
                dtype=float,
            )
            intercept = float(newest["intercept"])
            vol_scale = float(newest["vol_scale"])
            ratio_scale = float(newest["ratio_scale"])
        except (KeyError, TypeError, ValueError):
            return None
        if not np.all(np.isfinite(coefficients)) or not np.isfinite(intercept):
            return None
        if vol_scale <= 0 or ratio_scale <= 0:
            return None

        return cls(
            model_id=str(newest["model_id"]),
            fold_index=int(newest.get("fold_index") or 0),
            feature_names=names,
            intercept=intercept,
            coefficients=coefficients,
            vol_mean=float(newest["vol_mean"]),
            vol_scale=vol_scale,
            ratio_mean=float(newest["ratio_mean"]),
            ratio_scale=ratio_scale,
            source_path=str(path),
        )

    def score(self, signal: Mapping[str, Any], *, symbol: str) -> dict[str, Any]:
        """Diagnostic annotation for one live ENTER signal.

        The returned mapping is additive journalling only.  Callers must merge
        it into the signal without letting any value influence execution.
        """

        trigger_kind = normalize_trigger_kind(signal)
        ratio = signal.get("vol_tp1_em_ratio")
        if ratio is None:
            ratio = tp1_em_ratio(
                signal.get("entry_price"),
                signal.get("tp_prices") or (),
                signal.get("vol_em_1d"),
            )
        row = {
            "side": signal.get("side"),
            "symbol": symbol,
            "trigger_kind": trigger_kind,
            "factor_vector": signal.get("factor_vector"),
            # The offline row spells these lowercase; the live signal carries
            # ``vol_R``.  Missing values stay missing so the model's own
            # missing-indicator features fire instead of a fabricated zero.
            "vol_r": signal.get("vol_R"),
            "vol_tp1_em_ratio": ratio,
            "fvg_age_bars": signal.get("fvg_age_bars"),
        }
        features = _feature_vector(
            row,
            names=self.feature_names,
            vol_mean=self.vol_mean,
            vol_scale=self.vol_scale,
            ratio_mean=self.ratio_mean,
            ratio_scale=self.ratio_scale,
        )
        predicted = float(self.intercept + features @ self.coefficients)
        if not np.isfinite(predicted):
            return {}

        # Every fitted vocabulary so far predates the FxPro proxies, so a live
        # cluster or quote-pressure entry has no representation at all and is
        # scored as the baseline trigger family. Flag that rather than let the
        # number read as if the family had been learned.
        return {
            "shadow_score": round(predicted, 4),
            "shadow_model_id": self.model_id,
            "shadow_fold_index": self.fold_index,
            "shadow_trigger_kind": trigger_kind,
            "shadow_trigger_in_vocab": bool(trigger_kind in self.trigger_vocab),
            "shadow_executing": False,
        }


__all__ = [
    "LiveShadowScorer",
    "normalize_trigger_kind",
    "tp1_em_ratio",
]
