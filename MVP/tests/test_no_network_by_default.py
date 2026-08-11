"""Proves the default (no-flag) mode of every stage module never calls the Gemini
classification API or the Gemini embedding API. GeminiProvider.classify /
FamilyAwareGeminiEmbeddingProvider.embed_texts are monkeypatched to raise -- if any default
mode reaches them, this test fails."""

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

STAGE_MODULES_WITH_SAFE_MODE = [
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


@pytest.fixture(autouse=True)
def forbid_network_calls(monkeypatch):
    def _raise(*args, **kwargs):
        raise AssertionError("Network call attempted during default (safe) mode.")

    monkeypatch.setattr("newstart_ai_mvp.llm_pipeline.GeminiProvider.classify", _raise, raising=False)
    monkeypatch.setattr("newstart_ai_mvp.llm_pipeline.GeminiProvider.classify_with_context", _raise, raising=False)
    monkeypatch.setattr(
        "newstart_ai_mvp.rag_pipeline.FamilyAwareGeminiEmbeddingProvider.embed_texts", _raise, raising=False
    )


@pytest.mark.parametrize("module", STAGE_MODULES_WITH_SAFE_MODE, ids=lambda m: m.__name__)
def test_default_mode_never_calls_gemini_or_embeddings(module, settings, capsys):
    module.run_safe(settings)
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err


def test_compare_models_never_calls_gemini(settings, capsys):
    tables = compare_models.load_comparison_tables(settings)
    assert len(tables["per_condition"]) > 0
