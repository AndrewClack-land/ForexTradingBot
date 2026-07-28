#!/usr/bin/env python3
"""Build the file-hash manifest for a fresh isolated backtest release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


SCHEMA = "forexbot-backtest-release/v1"
RELEASE_PATHS = (
    "backtest",
    "core/__init__.py",
    "core/absorption.py",
    "core/htf_context.py",
    "core/fxpro_quote_pressure.py",
    "core/narrative_scoring.py",
    "core/pivot_trigger.py",
    "core/strategy_narrative.py",
    "core/vol_regime.py",
    "deploy/build_backtest_release_manifest.py",
    "requirements-backtest.txt",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for relative_name in RELEASE_PATHS:
        candidate = root / relative_name
        if candidate.is_symlink():
            raise ValueError(f"release path cannot be a symlink: {relative_name}")
        if candidate.is_dir():
            for child in candidate.rglob("*"):
                if child.is_symlink():
                    raise ValueError(
                        "release path cannot be a symlink: "
                        f"{child.relative_to(root).as_posix()}"
                    )
                if child.is_file():
                    files.append(child)
        elif candidate.is_file():
            files.append(candidate)
        else:
            raise ValueError(f"required release path is missing: {relative_name}")
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def build_manifest(root: Path, commit: str) -> dict[str, object]:
    root = root.expanduser().resolve()
    normalized_commit = commit.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", normalized_commit):
        raise ValueError("commit must be a 40/64-character hexadecimal value")
    files = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in _release_files(root)
    }
    return {
        "schema": SCHEMA,
        "release_commit": normalized_commit,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    payload = build_manifest(Path(args.root), args.commit)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
