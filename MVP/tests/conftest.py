from __future__ import annotations

from pathlib import Path

import pytest

_MVP_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def settings():
    from newstart_ai_mvp.config import load_settings

    return load_settings()


@pytest.fixture(scope="session")
def project_root() -> Path:
    """The frozen-artifact root (newstart_ai_benchmark/, or NEWSTART_BENCHMARK_ROOT if set) --
    not simply MVP's parent directory, since MVP/ and newstart_ai_benchmark/ are siblings."""
    from newstart_ai_mvp.config import BENCHMARK_ROOT

    return BENCHMARK_ROOT


@pytest.fixture(scope="session")
def mvp_root() -> Path:
    return _MVP_ROOT
