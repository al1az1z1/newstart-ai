"""Pure path-resolution proof that redirect_frozen_outputs() never lets a frozen-output
write path resolve under the real project root, and never redirects a non-frozen path."""

from __future__ import annotations

from newstart_ai_mvp.run_scope import FROZEN_OUTPUT_PREFIXES, new_run_id, redirect_frozen_outputs, run_root


def test_every_known_frozen_prefix_redirects_under_the_run_root(settings):
    run_id = "unit_test_run_id"
    dest = run_root(run_id)
    with redirect_frozen_outputs(run_id):
        for prefix in FROZEN_OUTPUT_PREFIXES:
            resolved = settings.resolve_path(f"{prefix}/some_file.json")
            assert str(resolved).startswith(str(dest)), f"{prefix} did not redirect under the run root"


def test_non_frozen_paths_are_never_redirected(settings):
    run_id = "unit_test_run_id"
    dest = run_root(run_id)
    non_frozen_paths = [
        "data/processed/final_dataset.csv",
        "configs/family_aware.yaml",
        "prompts/classification/v1.yaml",
    ]
    outside_context = [settings.resolve_path(p) for p in non_frozen_paths]
    with redirect_frozen_outputs(run_id):
        inside_context = [settings.resolve_path(p) for p in non_frozen_paths]
    assert outside_context == inside_context
    for resolved in inside_context:
        assert not str(resolved).startswith(str(dest))


def test_resolve_path_is_restored_after_the_context_exits(settings):
    run_id = "unit_test_run_id"
    before = settings.resolve_path("data/family_aware_splits/train.csv")
    with redirect_frozen_outputs(run_id):
        pass
    after = settings.resolve_path("data/family_aware_splits/train.csv")
    assert before == after
    assert "runs" not in str(after).split("\\") and "runs" not in str(after).split("/")


def test_new_run_id_is_sortable_and_unique_enough():
    ids = {new_run_id() for _ in range(3)}
    assert all(len(i) == 16 for i in ids)  # YYYYMMDDTHHMMSSZ
