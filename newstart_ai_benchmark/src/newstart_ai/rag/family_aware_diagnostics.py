"""Validation retrieval diagnostics and leakage/integrity proofs for the family-aware RAG
layer (Version 6, Checkpoint 9).

Diagnostics here measure RETRIEVAL behavior only (agency hit rate, MRR, similarity
distributions, diversification effect) -- never classifier accuracy. No generative Gemini
call is made anywhere in this module.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from newstart_ai.rag.family_aware_index import retrieve_diversified
from newstart_ai.schemas.checkpoint9 import (
    ConditionRetrievalDiagnostic,
    DiversificationEffectReport,
    PerAgencyRetrievalDiagnostic,
    RagIntegrityProof,
    RetrievalBeforeAfterSample,
    RetrievalResultRecord,
    ValidationRetrievalDiagnosticsManifest,
)

_MASKED_CONDITIONS = {
    "complete_masked", "beginning_only_masked", "middle_only_masked", "end_only_masked", "beginning_middle_end_masked",
}


def _duplication_ratio(results: list[dict], key: str) -> float:
    if not results:
        return 0.0
    unique = len({r[key] for r in results})
    return 1 - (unique / len(results))


def evaluate_condition_retrieval(
    condition_name: str,
    query_rows: pd.DataFrame,  # condition registry rows for this condition, validation split
    true_label_by_doc: dict[str, str],
    unmasked_collection,
    masked_collection,
    embedding_provider,
    settings,
) -> tuple[ConditionRetrievalDiagnostic, list[RetrievalResultRecord], dict]:
    rag_cfg = settings.family_aware.rag
    masked = condition_name in _MASKED_CONDITIONS
    collection = masked_collection if masked else unmasked_collection

    query_doc_ids = query_rows["document_id"].astype(str).tolist()
    query_texts = query_rows["text"].astype(str).tolist()
    query_vectors, usage = embedding_provider.embed_texts(query_texts, rag_cfg.query_task_type)

    top_k = rag_cfg.retrieval.top_k
    label_order = list(settings.base.labels)

    all_records: list[RetrievalResultRecord] = []
    hits = 0
    reciprocal_ranks = []
    all_similarities = []
    fewer_than_k_count = 0
    dup_before_list = []
    dup_after_list = []
    fam_dup_before_list = []
    fam_dup_after_list = []
    per_agency_hits: dict[str, int] = {label: 0 for label in label_order}
    per_agency_rr: dict[str, list[float]] = {label: [] for label in label_order}
    per_agency_counts: dict[str, int] = {label: 0 for label in label_order}

    for doc_id, vector in zip(query_doc_ids, query_vectors):
        true_label = true_label_by_doc[doc_id]
        per_agency_counts[true_label] += 1

        before, after = retrieve_diversified(collection, vector, settings)

        if len(after) < top_k:
            fewer_than_k_count += 1

        dup_before_list.append(_duplication_ratio(before, "parent_document_id"))
        dup_after_list.append(_duplication_ratio(after, "parent_document_id"))
        fam_dup_before_list.append(_duplication_ratio(before, "effective_family_id"))
        fam_dup_after_list.append(_duplication_ratio(after, "effective_family_id"))

        rr = 0.0
        hit = False
        for rank, result in enumerate(after, start=1):
            all_similarities.append(result["similarity"])
            all_records.append(
                RetrievalResultRecord(
                    query_document_id=doc_id, condition=condition_name, masked=masked, rank=rank,
                    chunk_id=result["chunk_id"], parent_document_id=result["parent_document_id"],
                    effective_family_id=result["effective_family_id"], effective_agency=result["effective_agency"],
                    similarity=result["similarity"], text_hash=result["text_hash"],
                )
            )
            if not hit and result["effective_agency"] == true_label:
                hit = True
                rr = 1.0 / rank
        if hit:
            hits += 1
            per_agency_hits[true_label] += 1
        reciprocal_ranks.append(rr)
        per_agency_rr[true_label].append(rr)

    n = len(query_doc_ids)
    per_agency = [
        PerAgencyRetrievalDiagnostic(
            agency=label,
            query_document_count=per_agency_counts[label],
            top_k_agency_hit_rate=round(100 * per_agency_hits[label] / per_agency_counts[label], 2) if per_agency_counts[label] else 0.0,
            mean_reciprocal_rank=round(float(np.mean(per_agency_rr[label])), 4) if per_agency_rr[label] else 0.0,
            note="IRS validation support is very small -- treat as high-variance, not precise." if label == "IRS" else "",
        )
        for label in label_order
    ]

    diagnostic = ConditionRetrievalDiagnostic(
        condition=condition_name,
        masked=masked,
        index_used="masked" if masked else "unmasked",
        query_document_count=n,
        top_k_agency_hit_rate=round(100 * hits / n, 2) if n else 0.0,
        mean_reciprocal_rank=round(float(np.mean(reciprocal_ranks)), 4) if reciprocal_ranks else 0.0,
        mean_similarity=round(float(np.mean(all_similarities)), 4) if all_similarities else 0.0,
        similarity_std=round(float(np.std(all_similarities)), 4) if all_similarities else 0.0,
        percent_queries_with_fewer_than_k_results=round(100 * fewer_than_k_count / n, 2) if n else 0.0,
        mean_parent_duplication_before=round(float(np.mean(dup_before_list)), 4) if dup_before_list else 0.0,
        mean_parent_duplication_after=round(float(np.mean(dup_after_list)), 4) if dup_after_list else 0.0,
        mean_family_duplication_before=round(float(np.mean(fam_dup_before_list)), 4) if fam_dup_before_list else 0.0,
        mean_family_duplication_after=round(float(np.mean(fam_dup_after_list)), 4) if fam_dup_after_list else 0.0,
        per_agency=per_agency,
    )
    return diagnostic, all_records, usage


def build_diversification_effect_report(
    query_doc_ids: list[str],
    condition_name: str,
    unmasked_collection,
    query_vectors: list[list[float]],
    settings,
    dominant_document_id: str = "739",
) -> DiversificationEffectReport:
    samples = []
    dom_before_counts = []
    dom_after_counts = []
    for doc_id, vector in zip(query_doc_ids, query_vectors):
        before, after = retrieve_diversified(unmasked_collection, vector, settings)
        top_k = settings.family_aware.rag.retrieval.top_k

        def _top_share(results, key):
            if not results:
                return 0.0
            counts: dict[str, int] = {}
            for r in results:
                counts[r[key]] = counts.get(r[key], 0) + 1
            return max(counts.values()) / len(results)

        samples.append(
            RetrievalBeforeAfterSample(
                query_document_id=doc_id, condition=condition_name,
                top_parent_document_share_before=round(_top_share(before, "parent_document_id"), 4),
                top_parent_document_share_after=round(_top_share(after, "parent_document_id"), 4),
                top_family_share_before=round(_top_share(before, "effective_family_id"), 4),
                top_family_share_after=round(_top_share(after, "effective_family_id"), 4),
                result_count_before=len(before), result_count_after=len(after),
            )
        )
        dom_before_counts.append(sum(1 for r in before if r["parent_document_id"] == dominant_document_id))
        dom_after_counts.append(sum(1 for r in after if r["parent_document_id"] == dominant_document_id))

    return DiversificationEffectReport(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        sample_size=len(samples),
        mean_top_parent_share_before=round(float(np.mean([s.top_parent_document_share_before for s in samples])), 4) if samples else 0.0,
        mean_top_parent_share_after=round(float(np.mean([s.top_parent_document_share_after for s in samples])), 4) if samples else 0.0,
        mean_top_family_share_before=round(float(np.mean([s.top_family_share_before for s in samples])), 4) if samples else 0.0,
        mean_top_family_share_after=round(float(np.mean([s.top_family_share_after for s in samples])), 4) if samples else 0.0,
        document_739_max_positions_before=max(dom_before_counts) if dom_before_counts else 0,
        document_739_max_positions_after=max(dom_after_counts) if dom_after_counts else 0,
        samples=samples,
        notes=[
            f"document_{dominant_document_id} has 519 training chunks (the largest in the "
            "corpus) -- this report directly measures whether it dominates retrieved "
            "positions before vs. after diversification.",
        ],
    )


def build_rag_integrity_proof(
    unmasked_indexed_doc_ids: set[str],
    masked_indexed_doc_ids: set[str],
    train_doc_ids: set[str],
    validation_doc_ids: set[str],
    test_doc_ids: set[str],
    unmasked_indexed_family_ids: set[str],
    masked_indexed_family_ids: set[str],
    train_family_ids: set[str],
    validation_family_ids: set[str],
    test_family_ids: set[str],
    excluded_doc_ids: set[str],
    masked_index_text_sample_matches_masking_policy: bool,
    no_unmasked_identifier_found_in_masked_payloads: bool,
    query_fingerprint_check_passed: bool,
    metadata_consistency_check_passed: bool,
    rebuild_corpus_fingerprint_matches: bool,
    rebuild_retrieval_ordering_matches: bool,
    historical_rag_hashes_match: bool,
) -> RagIntegrityProof:
    all_indexed_doc_ids = unmasked_indexed_doc_ids | masked_indexed_doc_ids
    all_indexed_family_ids = unmasked_indexed_family_ids | masked_indexed_family_ids

    return RagIntegrityProof(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        every_indexed_vector_from_train_split=(all_indexed_doc_ids <= train_doc_ids),
        no_validation_document_indexed=len(all_indexed_doc_ids & validation_doc_ids) == 0,
        no_test_document_indexed=len(all_indexed_doc_ids & test_doc_ids) == 0,
        no_validation_family_indexed=len(all_indexed_family_ids & validation_family_ids) == 0,
        no_test_family_indexed=len(all_indexed_family_ids & test_family_ids) == 0,
        no_excluded_document_indexed=len(all_indexed_doc_ids & excluded_doc_ids) == 0,
        masked_index_text_matches_frozen_masking_policy=masked_index_text_sample_matches_masking_policy,
        no_unmasked_identifier_recovered_via_masked_index=no_unmasked_identifier_found_in_masked_payloads,
        query_fingerprints_match_condition_registry=query_fingerprint_check_passed,
        chunk_document_family_metadata_internally_consistent=metadata_consistency_check_passed,
        rebuild_from_cache_reproduces_identical_corpus_fingerprint=rebuild_corpus_fingerprint_matches,
        rebuild_reproduces_identical_retrieval_ordering=rebuild_retrieval_ordering_matches,
        determinism_caveat=(
            "Embedding vectors themselves are supplied by the Gemini API and cached "
            "immediately after first retrieval, so a rebuild FROM CACHE (no new API calls) "
            "is exactly reproducible; a rebuild that re-calls the API for previously-uncached "
            "text is subject to whatever numerical determinism the provider itself guarantees, "
            "which this repository does not control."
        ),
        historical_rag_store_unchanged=historical_rag_hashes_match,
        notes=[
            "every_indexed_vector_from_train_split is a strict subset check (indexed doc IDs "
            "<= frozen train doc IDs), not just a non-overlap check against validation/test.",
        ],
    )
