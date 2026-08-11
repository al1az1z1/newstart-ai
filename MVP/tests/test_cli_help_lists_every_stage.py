"""--help smoke test for every newstart_ai_mvp entry point -- import-only, never triggers
real work. Proves every stage module has a working, self-describing CLI."""

from __future__ import annotations

import pytest

from newstart_ai_mvp import (
    build_rag_index,
    compare_models,
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

ALL_MODULES = [
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
    compare_models,
]


@pytest.mark.parametrize("module", ALL_MODULES, ids=lambda m: m.__name__)
def test_help_flag_exits_cleanly_without_side_effects(module):
    with pytest.raises(SystemExit) as exc_info:
        module.main(["--help"])
    assert exc_info.value.code == 0
