"""Durable JSON storage for the causal H1 Rejection Block tracker state.

This module deliberately validates only the storage envelope.  The tracker owns
the semantic schema of the nested state and validates it when importing it.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional


RB_TRACKER_STATE_SCHEMA_VERSION = 1


class RBTrackerStateValidationError(ValueError):
    """Raised when an existing RB tracker state file is not trustworthy."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RBTrackerStateValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> None:
    raise RBTrackerStateValidationError(
        f"non-finite JSON number {value!r} is not allowed"
    )


def load(path: Path) -> Optional[dict[str, Any]]:
    """Load tracker state, returning ``None`` only when the file is missing.

    Any existing but unreadable or malformed file raises instead of being
    treated as an empty state.  That fail-closed distinction prevents a corrupt
    snapshot from silently re-enabling already observed RB events.
    """

    source = Path(path)
    if not source.exists():
        return None

    try:
        payload = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RBTrackerStateValidationError(
            f"cannot read RB tracker state {source}: {exc}"
        ) from exc

    try:
        raw = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except RBTrackerStateValidationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RBTrackerStateValidationError(
            f"cannot decode RB tracker state {source}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise RBTrackerStateValidationError(
            "RB tracker state store root must be a JSON object"
        )
    if set(raw) != {"schema_version", "state"}:
        raise RBTrackerStateValidationError("RB tracker state envelope is malformed")
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != RB_TRACKER_STATE_SCHEMA_VERSION
    ):
        raise RBTrackerStateValidationError(
            "unsupported RB tracker state schema_version "
            f"{raw['schema_version']!r}"
        )

    state = raw["state"]
    if not isinstance(state, dict):
        raise RBTrackerStateValidationError("RB tracker state must be a JSON object")
    return state


def save(state: Mapping[str, Any], path: Path) -> None:
    """Atomically save a tracker state mapping without pruning its contents."""

    if not isinstance(state, Mapping):
        raise RBTrackerStateValidationError("RB tracker state must be a mapping")

    envelope = {
        "schema_version": RB_TRACKER_STATE_SCHEMA_VERSION,
        "state": dict(state),
    }
    try:
        encoded = json.dumps(
            envelope,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
    except (TypeError, ValueError) as exc:
        raise RBTrackerStateValidationError(
            f"RB tracker state is not strict JSON: {exc}"
        ) from exc

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
