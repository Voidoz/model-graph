"""Shared offline fixtures for observability tests (mock provider only)."""

from __future__ import annotations

from pathlib import Path

import pytest
from support import run_budget_failure, run_mock


@pytest.fixture
def mock_run_dir(tmp_path: Path) -> Path:
    return run_mock(tmp_path)


@pytest.fixture
def mock_runs_dir(tmp_path: Path) -> Path:
    run_mock(tmp_path)
    run_mock(tmp_path)
    run_budget_failure(tmp_path)
    return tmp_path
