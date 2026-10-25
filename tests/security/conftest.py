"""Pytest fixtures for the offline security suite."""

from __future__ import annotations

import pytest
from security_helpers import CompiledGraph, compile_example


@pytest.fixture
def hello_graph() -> CompiledGraph:
    return compile_example()
