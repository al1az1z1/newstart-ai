"""Tests for Version 6 Checkpoint 10: the frozen head-to-head test evaluation.

Uses mocked Gemini classification/embedding responses throughout -- no API quota spent on
routine tests.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from newstart_ai.config import load_settings
from newstart_ai.models.llm.family_aware_evaluation import (
    compute_cache_key,
    format_context_no_labels,
    run_llm_rag_case,
    run_plain_llm_case,
    truncate_for_llm,
)
from newstart_ai.models.llm.family_aware_integrity import build_evaluation_integrity_proof
from newstart_ai.models.llm.family_aware_metrics import (
    build_method_condition_metrics,
    build_primary_paired_comparison,
    build_statistical_uncertainty,
)
from newstart_ai.schemas.checkpoint10 import CaseResult
from newstart_ai.schemas.classification import ClassificationResult, TokenUsage

LABEL_ORDER = ["USCIS", "DMV", "SSA", "IRS"]


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture()
def isolated_settings(settings, tmp_path):
    settings.family_aware.evaluation.cache_dir = str(tmp_path / "llm_eval_cache")
    return settings


class FakeLLMProvider:
    def __init__(self, model_name="gemini-3.6-flash", fixed_label="USCIS", fail_mode=None):
        self.model_name = model_name
        self.fixed_label = fixed_label
        self.fail_mode = fail_mode
        self.calls = []

    def classify(self, text, document_id, prompt, method="llm"):
        self.calls.append({"text": text, "document_id": document_id, "context": None})
        if self.fail_mode == "invalid":
            raise ValueError("Gemini returned label OUT_OF_SCHEMA")
        if self.fail_mode == "transient":
            raise RuntimeError("upstream rate limit exceeded, please retry")
        return ClassificationResult(
            method=method, document_id=document_id, predicted_label=self.fixed_label,
            latency_ms=12.3, token_usage=TokenUsage(prompt_tokens=100, completion_tokens=5, total_tokens=105),
            estimated_cost=0.0001, metadata={},
        )

    def classify_with_context(self, text, context, document_id, prompt, method="llm_rag"):
        self.calls.append({"text": text, "document_id": document_id, "context": context})
        if self.fail_mode == "invalid":
            raise ValueError("Gemini returned label OUT_OF_SCHEMA")
        if self.fail_mode == "transient":
            raise RuntimeError("upstream rate limit exceeded, please retry")
        return ClassificationResult(
            method=method, document_id=document_id, predicted_label=self.fixed_label,
            latency_ms=15.0, token_usage=TokenUsage(prompt_tokens=300, completion_tokens=5, total_tokens=305),
            estimated_cost=0.0003, metadata={},
        )


class FakePrompt:
    version = "test_v1"
    allowed_labels = LABEL_ORDER
    system_prompt = "system"
    user_template = "{text}"
    response_schema = {"type": "object"}


class FakeEmbeddingProvider:
    def embed_texts(self, texts, task_type):
        vectors = [[0.1] * 8 for _ in texts]
        usage = {"total_texts_requested": len(texts), "cache_hits": 0, "cache_misses": len(texts), "api_requests_made": 1, "total_tokens_billed": 10, "retries": 0, "failures": 0, "wall_clock_seconds": 0.01}
        return vectors, usage


def test_plain_llm_case_success_and_cached_on_rerun(isolated_settings):
    provider = FakeLLMProvider(fixed_label="USCIS")
    case1 = run_plain_llm_case("1", "FAM:1", "complete_unmasked", "USCIS", "some doc text", "fp1", provider, FakePrompt(), isolated_settings)
    assert case1.status == "success"
    assert case1.predicted_label == "USCIS"
    assert len(provider.calls) == 1

    case2 = run_plain_llm_case("1", "FAM:1", "complete_unmasked", "USCIS", "some doc text", "fp1", provider, FakePrompt(), isolated_settings)
    assert case2 == case1
    assert len(provider.calls) == 1  # no new call -- resumed from cache


def test_plain_llm_invalid_output_is_not_retried(isolated_settings):
    provider = FakeLLMProvider(fail_mode="invalid")
    case = run_plain_llm_case("2", "FAM:2", "complete_unmasked", "DMV", "text", "fp2", provider, FakePrompt(), isolated_settings)
    assert case.status == "invalid"
    assert case.predicted_label is None
    assert len(provider.calls) == 1  # exactly one attempt, no retry for invalid output


def test_plain_llm_transient_error_retries_then_fails(isolated_settings):
    isolated_settings.family_aware.evaluation.max_attempts = 3
    isolated_settings.family_aware.evaluation.retry_backoff_seconds = 0.001
    provider = FakeLLMProvider(fail_mode="transient")
    case = run_plain_llm_case("3", "FAM:3", "complete_unmasked", "SSA", "text", "fp3", provider, FakePrompt(), isolated_settings)
    assert case.status == "failed"
    assert case.attempt_count == 3
    assert len(provider.calls) == 3


def test_failed_case_is_scored_as_an_error_never_excluded():
    cases = [
        CaseResult(method="llm", document_id="1", effective_family_id="F1", condition="c", true_label="USCIS", input_fingerprint="fp", predicted_label="USCIS", status="success", attempt_count=1, latency_ms=1.0, cache_key="k1"),
        CaseResult(method="llm", document_id="2", effective_family_id="F2", condition="c", true_label="DMV", input_fingerprint="fp", predicted_label=None, status="failed", attempt_count=3, latency_ms=0.0, cache_key="k2"),
    ]
    metrics = build_method_condition_metrics(cases, LABEL_ORDER)
    assert metrics.document_count == 2
    assert metrics.error_count == 1  # the failed case counts as an error
    assert metrics.coverage_rate == 50.0
    assert metrics.failed_count == 1


def test_no_agency_label_or_metadata_in_rag_context():
    retrieved = [{"text": "Some retrieved chunk text about forms."}, {"text": "Another excerpt."}]
    context = format_context_no_labels(retrieved)
    assert "Agency" not in context
    assert "USCIS" not in context and "DMV" not in context and "SSA" not in context and "IRS" not in context
    assert "Some retrieved chunk text about forms." in context


def test_llm_rag_case_never_passes_metadata_to_classify_with_context(isolated_settings):
    provider = FakeLLMProvider(fixed_label="IRS")
    embedding_provider = FakeEmbeddingProvider()

    fake_collection = MagicMock()
    fake_collection.query.return_value = {
        "ids": [["c1", "c2"]],
        "distances": [[0.1, 0.2]],
        "metadatas": [[
            {"document_id": "d1", "effective_family_id": "FAM:D1", "effective_agency": "IRS", "text_hash": "h1"},
            {"document_id": "d2", "effective_family_id": "FAM:D2", "effective_agency": "IRS", "text_hash": "h2"},
        ]],
    }
    chunk_text_by_id = {"c1": "Chunk one text.", "c2": "Chunk two text."}

    case = run_llm_rag_case(
        "10", "FAM:10", "complete_unmasked", "IRS", "query document text", "fp10", masked=False,
        unmasked_collection=fake_collection, masked_collection=MagicMock(),
        chunk_text_by_id=chunk_text_by_id, embedding_provider=embedding_provider,
        llm_provider=provider, prompt=FakePrompt(), settings=isolated_settings,
    )
    assert case.status == "success"
    sent_context = provider.calls[0]["context"]
    assert "IRS" not in sent_context
    assert "FAM:D1" not in sent_context and "d1" not in sent_context.lower().split()
    assert "Chunk one text." in sent_context
    assert len(case.retrieved_chunks) == 2
    assert case.retrieved_chunks[0].effective_agency == "IRS"  # stored separately for analysis, just not shown to Gemini


def test_masked_condition_uses_masked_collection_only(isolated_settings):
    provider = FakeLLMProvider(fixed_label="SSA")
    embedding_provider = FakeEmbeddingProvider()
    unmasked_collection = MagicMock()
    masked_collection = MagicMock()
    masked_collection.query.return_value = {"ids": [["m1"]], "distances": [[0.1]], "metadatas": [[{"document_id": "d1", "effective_family_id": "FAM:D1", "effective_agency": "SSA", "text_hash": "hm1"}]]}

    run_llm_rag_case(
        "11", "FAM:11", "complete_masked", "SSA", "[AGENCY_NAME] query text", "fp11", masked=True,
        unmasked_collection=unmasked_collection, masked_collection=masked_collection,
        chunk_text_by_id={"m1": "masked chunk text"}, embedding_provider=embedding_provider,
        llm_provider=provider, prompt=FakePrompt(), settings=isolated_settings,
    )
    masked_collection.query.assert_called_once()
    unmasked_collection.query.assert_not_called()


def test_unmasked_condition_uses_unmasked_collection_only(isolated_settings):
    provider = FakeLLMProvider(fixed_label="DMV")
    embedding_provider = FakeEmbeddingProvider()
    unmasked_collection = MagicMock()
    unmasked_collection.query.return_value = {"ids": [["u1"]], "distances": [[0.1]], "metadatas": [[{"document_id": "d2", "effective_family_id": "FAM:D2", "effective_agency": "DMV", "text_hash": "hu1"}]]}
    masked_collection = MagicMock()

    run_llm_rag_case(
        "12", "FAM:12", "complete_unmasked", "DMV", "query text", "fp12", masked=False,
        unmasked_collection=unmasked_collection, masked_collection=masked_collection,
        chunk_text_by_id={"u1": "unmasked chunk text"}, embedding_provider=embedding_provider,
        llm_provider=provider, prompt=FakePrompt(), settings=isolated_settings,
    )
    unmasked_collection.query.assert_called_once()
    masked_collection.query.assert_not_called()


def test_cache_key_depends_on_method_model_prompt_document_condition_and_retrieval_context():
    k1 = compute_cache_key("llm", "gemini-3.6-flash", "v1", "1", "complete_unmasked", "fpA", None)
    k2 = compute_cache_key("llm_rag", "gemini-3.6-flash", "v1", "1", "complete_unmasked", "fpA", "retrieval-fp-X")
    k3 = compute_cache_key("llm_rag", "gemini-3.6-flash", "v1", "1", "complete_unmasked", "fpA", "retrieval-fp-Y")
    assert len({k1, k2, k3}) == 3  # all distinct given different method/retrieval context


def test_cache_key_distinguishes_conditions_sharing_identical_content_fingerprint():
    """Regression test for the real bug found in Checkpoint 10: single-chunk test documents
    can have byte-identical registered text (and therefore an identical content fingerprint)
    across multiple conditions (e.g. beginning_only == middle_only == end_only). The cache
    key must still treat them as distinct cases."""
    k_beginning = compute_cache_key("llm", "gemini-3.6-flash", "v1", "42", "beginning_only_unmasked", "same-fingerprint", None)
    k_middle = compute_cache_key("llm", "gemini-3.6-flash", "v1", "42", "middle_only_unmasked", "same-fingerprint", None)
    k_end = compute_cache_key("llm", "gemini-3.6-flash", "v1", "42", "end_only_unmasked", "same-fingerprint", None)
    assert len({k_beginning, k_middle, k_end}) == 3


def test_truncate_for_llm_reuses_frozen_6000_char_policy():
    long_text = "a" * 10000
    truncated, was_truncated = truncate_for_llm(long_text)
    assert was_truncated is True
    assert len(truncated) == 6000

    short_text = "short document"
    truncated2, was_truncated2 = truncate_for_llm(short_text)
    assert was_truncated2 is False
    assert truncated2 == short_text


def test_primary_paired_comparison_buckets_are_mutually_correct():
    bert_preds = {"1": "USCIS", "2": "DMV", "3": "SSA", "4": "USCIS"}
    llm_preds = {"1": "USCIS", "2": "USCIS", "3": "SSA", "4": "IRS"}  # doc2 wrong, doc4 right
    rag_preds = {"1": "USCIS", "2": "DMV", "3": "IRS", "4": "IRS"}  # doc3 wrong, doc2 corrected, doc4 right
    true_labels = {"1": "USCIS", "2": "DMV", "3": "SSA", "4": "IRS"}

    comparison = build_primary_paired_comparison(bert_preds, llm_preds, rag_preds, true_labels)
    assert "1" in comparison.all_three_correct
    assert "2" in comparison.rag_corrects_plain_llm  # llm wrong, rag right
    assert "3" in comparison.rag_breaks_plain_llm  # llm right, rag wrong
    assert "4" in comparison.bert_only_errors  # bert wrong (USCIS != IRS), llm and rag both right


def test_deterministic_bootstrap_produces_identical_results_across_runs():
    cases_a = [
        CaseResult(method="llm", document_id=str(i), effective_family_id=f"F{i}", condition="c", true_label=LABEL_ORDER[i % 4], input_fingerprint=f"fp{i}", predicted_label=LABEL_ORDER[i % 4] if i % 5 != 0 else LABEL_ORDER[(i + 1) % 4], status="success", attempt_count=1, latency_ms=1.0, cache_key=f"k{i}")
        for i in range(30)
    ]
    cases_b = [
        CaseResult(method="llm_rag", document_id=str(i), effective_family_id=f"F{i}", condition="c", true_label=LABEL_ORDER[i % 4], input_fingerprint=f"fp{i}", predicted_label=LABEL_ORDER[i % 4], status="success", attempt_count=1, latency_ms=1.0, cache_key=f"k{i}b")
        for i in range(30)
    ]
    result_1 = build_statistical_uncertainty({"llm": cases_a, "llm_rag": cases_b}, LABEL_ORDER, n_bootstrap=200, seed=7)
    result_2 = build_statistical_uncertainty({"llm": cases_a, "llm_rag": cases_b}, LABEL_ORDER, n_bootstrap=200, seed=7)
    assert result_1.bootstrap_results == result_2.bootstrap_results
    assert result_1.paired_bootstrap_results == result_2.paired_bootstrap_results


def test_evaluation_integrity_proof_detects_missing_case():
    proof = build_evaluation_integrity_proof(
        evaluated_document_ids={"1", "2"}, expected_test_document_ids={"1", "2", "3"},
        cases_by_method_document_condition={("llm", "1", "c"): 1, ("llm", "2", "c"): 1},
        expected_method_condition_document_counts={"llm": 990},
        condition_fingerprints_by_method={"llm": {("1", "c"): "fpA"}, "llm_rag": {("1", "c"): "fpA"}},
        no_test_label_in_prompts=True, masked_queries_used_masked_index=True, unmasked_queries_used_unmasked_index=True,
        approved_configuration_used=True, no_training_or_policy_selection_ran=True, cache_had_no_duplicates=True,
        historical_artifacts_unchanged=True, checkpoint_4_9_artifacts_unchanged=True,
        both_rag_indexes_train_only=True, no_train_validation_text_outside_rag=True,
    )
    assert proof.exact_99_test_documents_evaluated is False


def test_evaluation_integrity_proof_detects_fingerprint_mismatch_across_methods():
    proof = build_evaluation_integrity_proof(
        evaluated_document_ids={"1"}, expected_test_document_ids={"1"},
        cases_by_method_document_condition={("llm", "1", "c"): 1, ("llm_rag", "1", "c"): 1},
        expected_method_condition_document_counts={"llm": 1, "llm_rag": 1},
        condition_fingerprints_by_method={"llm": {("1", "c"): "fpA"}, "llm_rag": {("1", "c"): "fpDIFFERENT"}},
        no_test_label_in_prompts=True, masked_queries_used_masked_index=True, unmasked_queries_used_unmasked_index=True,
        approved_configuration_used=True, no_training_or_policy_selection_ran=True, cache_had_no_duplicates=True,
        historical_artifacts_unchanged=True, checkpoint_4_9_artifacts_unchanged=True,
        both_rag_indexes_train_only=True, no_train_validation_text_outside_rag=True,
    )
    assert proof.condition_fingerprints_match_across_methods is False


def test_evaluation_integrity_proof_detects_duplicate_case_record():
    proof = build_evaluation_integrity_proof(
        evaluated_document_ids={"1"}, expected_test_document_ids={"1"},
        cases_by_method_document_condition={("llm", "1", "c"): 2},  # duplicate!
        expected_method_condition_document_counts={"llm": 1},
        condition_fingerprints_by_method={"llm": {("1", "c"): "fpA"}},
        no_test_label_in_prompts=True, masked_queries_used_masked_index=True, unmasked_queries_used_unmasked_index=True,
        approved_configuration_used=True, no_training_or_policy_selection_ran=True, cache_had_no_duplicates=True,
        historical_artifacts_unchanged=True, checkpoint_4_9_artifacts_unchanged=True,
        both_rag_indexes_train_only=True, no_train_validation_text_outside_rag=True,
    )
    assert proof.one_record_per_document_condition_per_method is False


def test_retrieval_diagnostics_module_is_not_imported_by_metrics_module():
    """Structural proof that Checkpoint 9's retrieval hit-rate/MRR machinery is not being
    silently reused as a classification metric here."""
    import inspect

    import newstart_ai.models.llm.family_aware_metrics as metrics_module

    source = inspect.getsource(metrics_module)
    assert "evaluate_condition_retrieval" not in source
    assert "mean_reciprocal_rank" not in source
