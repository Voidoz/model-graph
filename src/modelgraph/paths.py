"""Filesystem locations used by the package.

The schemas and examples live at the repository root during development but are included in
the wheel under ``modelgraph/schemas`` for installed use.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def project_root() -> Path:
    """Best-effort repository root, overridable with MODELGRAPH_ROOT."""
    override = os.environ.get("MODELGRAPH_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    cwd = Path.cwd()
    if (cwd / "schemas" / "graph.schema.json").exists():
        return cwd
    return PACKAGE_DIR.parents[1]


def schema_dir() -> Path:
    override = os.environ.get("MODELGRAPH_SCHEMA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    packaged = PACKAGE_DIR / "schemas"
    if packaged.exists():
        return packaged
    return project_root() / "schemas"


def examples_dir() -> Path:
    return project_root() / "examples"


def runs_dir() -> Path:
    override = os.environ.get("MG_RUNS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.cwd() / "runs"


def cache_dir() -> Path:
    """Default local response-cache directory (``cache/responses``).

    Override with ``MG_CACHE_DIR`` or the CLI ``--cache-dir`` flag. The cache is local
    and per-user; there is no shared or remote cache.
    """
    override = os.environ.get("MG_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.cwd() / "cache" / "responses"
