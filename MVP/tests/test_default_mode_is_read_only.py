"""Proves the default (no-flag) mode of every stage module writes nothing and modifies
nothing under artifacts/family_aware/ or data/family_aware_*/ -- snapshots mtimes and the
full file list before and after, asserts both are unchanged."""

from __future__ import annotations

from pathlib import Path

from newstart_ai_mvp import (
    build_rag_index,
    evaluate_bert,
    evaluate_llm,
    evaluate_rag,
    prepare_data,
    stage1_validate_and_audit,
    stage2_build_split,
    stage3_build_chunks,
    stage4_build_masked,
    stage5_build_conditions,
    train_bert,
)

STAGE_MODULES = [
    stage1_validate_and_audit,
    stage2_build_split,
    stage3_build_chunks,
    stage4_build_masked,
    stage5_build_conditions,
    prepare_data,
    train_bert,
    evaluate_bert,
    build_rag_index,
    evaluate_llm,
    evaluate_rag,
]

WATCHED_DIRS = ("artifacts/family_aware", "data/family_aware_splits", "data/family_aware_chunks",
                 "data/family_aware_masked", "data/family_aware_conditions")


def _snapshot(project_root: Path) -> dict[str, float]:
    snap: dict[str, float] = {}
    for rel in WATCHED_DIRS:
        base = project_root / rel
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                snap[str(path)] = path.stat().st_mtime
    return snap


def test_default_mode_never_writes_or_modifies_any_frozen_file(project_root, settings):
    before = _snapshot(project_root)
    for module in STAGE_MODULES:
        module.run_safe(settings)
    after = _snapshot(project_root)
    assert before.keys() == after.keys(), "A file was added or removed by default (safe) mode"
    changed = {k for k in before if before[k] != after[k]}
    assert not changed, f"Default mode modified frozen files: {changed}"


def test_no_new_run_directories_created_by_default_mode(mvp_root, settings):
    runs_dir = mvp_root / "runs"
    before = {p.name for p in runs_dir.iterdir()} if runs_dir.exists() else set()
    for module in STAGE_MODULES:
        module.run_safe(settings)
    after = {p.name for p in runs_dir.iterdir()} if runs_dir.exists() else set()
    assert before == after, "Default mode created a new MVP/runs/ directory"
