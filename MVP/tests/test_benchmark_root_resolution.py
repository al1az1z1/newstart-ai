"""Proves MVP/'s path resolution correctly treats newstart_ai_benchmark/ as a sibling
directory (not a parent of MVP/), and that NEWSTART_BENCHMARK_ROOT overrides the default.

config.py computes MVP_ROOT/REPOSITORY_ROOT/BENCHMARK_ROOT/PROJECT_ROOT once at import time,
so the override test reloads the module with the env var set, then reloads it again
afterward to restore the default for every other test in the suite.
"""

from __future__ import annotations

import importlib

from newstart_ai_mvp import config as config_module


def test_mvp_root_is_the_mvp_directory_itself():
    assert config_module.MVP_ROOT.name == "MVP"
    assert (config_module.MVP_ROOT / "newstart_ai_mvp").is_dir()


def test_benchmark_root_defaults_to_the_sibling_directory_not_a_parent():
    assert config_module.BENCHMARK_ROOT == config_module.REPOSITORY_ROOT / "newstart_ai_benchmark"
    assert config_module.BENCHMARK_ROOT != config_module.MVP_ROOT.parent.parent
    assert config_module.BENCHMARK_ROOT.is_dir()
    assert (config_module.BENCHMARK_ROOT / "src" / "newstart_ai").is_dir()


def test_project_root_used_by_settings_equals_benchmark_root(settings):
    assert settings.project_root == config_module.BENCHMARK_ROOT


def test_newstart_benchmark_root_env_var_overrides_the_default(monkeypatch, tmp_path):
    override_dir = tmp_path / "alternate_benchmark_checkout"
    override_dir.mkdir()
    monkeypatch.setenv("NEWSTART_BENCHMARK_ROOT", str(override_dir))
    try:
        reloaded = importlib.reload(config_module)
        assert reloaded.BENCHMARK_ROOT == override_dir.resolve()
        assert reloaded.PROJECT_ROOT == override_dir.resolve()
    finally:
        # Undo the env var *before* reloading, since monkeypatch only reverts it after this
        # test function returns -- reloading while it's still set would just reapply the override.
        monkeypatch.delenv("NEWSTART_BENCHMARK_ROOT", raising=False)
        importlib.reload(config_module)  # restore the default for every later test

    assert config_module.BENCHMARK_ROOT == config_module.REPOSITORY_ROOT / "newstart_ai_benchmark"
