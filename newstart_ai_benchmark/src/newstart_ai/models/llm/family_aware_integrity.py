"""Leakage/integrity proof for Version 6 Checkpoint 10's head-to-head evaluation."""

from __future__ import annotations

from datetime import datetime, timezone

from newstart_ai.schemas.checkpoint10 import EvaluationIntegrityProof


def build_evaluation_integrity_proof(
    evaluated_document_ids: set[str],
    expected_test_document_ids: set[str],
    cases_by_method_document_condition: dict[tuple[str, str, str], int],
    expected_method_condition_document_counts: dict[str, int],
    condition_fingerprints_by_method: dict[str, dict[tuple[str, str], str]],
    no_test_label_in_prompts: bool,
    masked_queries_used_masked_index: bool,
    unmasked_queries_used_unmasked_index: bool,
    approved_configuration_used: bool,
    no_training_or_policy_selection_ran: bool,
    cache_had_no_duplicates: bool,
    historical_artifacts_unchanged: bool,
    checkpoint_4_9_artifacts_unchanged: bool,
    both_rag_indexes_train_only: bool,
    no_train_validation_text_outside_rag: bool,
) -> EvaluationIntegrityProof:
    exact_99 = evaluated_document_ids == expected_test_document_ids

    # `cases_by_method_document_condition` maps (method, document_id, condition) -> count of
    # cache records found for that exact case. Exactly one record per case, for every
    # expected case, is both requirements 2 ("exactly 990 cases per method") and 8/12 ("one
    # record per (document_id, condition) per method") at once.
    one_record_per_case = all(count == 1 for count in cases_by_method_document_condition.values())
    exactly_990_per_method = {
        method: sum(1 for (m, _doc, _cond), count in cases_by_method_document_condition.items() if m == method and count == 1)
        for method in expected_method_condition_document_counts
    }
    exactly_990_per_method_ok = all(
        exactly_990_per_method.get(method) == expected for method, expected in expected_method_condition_document_counts.items()
    )

    methods = list(condition_fingerprints_by_method.keys())
    fingerprints_match = True
    if len(methods) > 1:
        reference = condition_fingerprints_by_method[methods[0]]
        for method in methods[1:]:
            if condition_fingerprints_by_method[method] != reference:
                fingerprints_match = False
                break

    return EvaluationIntegrityProof(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        exact_99_test_documents_evaluated=exact_99,
        exactly_990_cases_per_method=exactly_990_per_method_ok,
        one_record_per_document_condition_per_method=one_record_per_case,
        condition_fingerprints_match_across_methods=fingerprints_match,
        no_train_validation_text_used_outside_rag_indexes=no_train_validation_text_outside_rag,
        both_rag_indexes_train_only=both_rag_indexes_train_only,
        no_test_label_or_agency_metadata_in_prompts=no_test_label_in_prompts,
        masked_queries_used_masked_index_only=masked_queries_used_masked_index,
        unmasked_queries_used_unmasked_index_only=unmasked_queries_used_unmasked_index,
        approved_model_prompt_parser_checkpoint_retrieval_used=approved_configuration_used,
        no_training_or_policy_selection_function_ran=no_training_or_policy_selection_ran,
        cached_resumed_results_no_duplicates=cache_had_no_duplicates,
        historical_artifacts_unchanged=historical_artifacts_unchanged,
        checkpoint_4_9_artifacts_unchanged=checkpoint_4_9_artifacts_unchanged,
        notes=[
            "condition_fingerprints_match_across_methods compares the EXACT SHA-256 "
            "fingerprint recorded by each method for every (document_id, condition) pair -- "
            "not just that a fingerprint field is present.",
        ],
    )
