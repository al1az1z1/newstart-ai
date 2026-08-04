"""Document-level error analysis for the family-aware chunked BERT (Version 6, Checkpoint 7).

All analysis operates on the document-level prediction produced by the frozen/reconfirmed
aggregation method on the `complete_unmasked` condition -- never on individual chunks.
"""

from __future__ import annotations

import pandas as pd

from newstart_ai.schemas.checkpoint7 import ErrorConcentrationReport, MisclassifiedDocument


def build_confusion_matrix(y_true: list[str], y_pred: list[str], label_order: list[str]) -> dict[str, dict[str, int]]:
    matrix = {true_label: {pred_label: 0 for pred_label in label_order} for true_label in label_order}
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


def find_misclassified_documents(
    document_ids: list[str],
    y_true: list[str],
    y_pred: list[str],
    family_by_doc: dict[str, str],
    total_chunks_by_doc: dict[str, int],
) -> list[MisclassifiedDocument]:
    misclassified = []
    for doc_id, true_label, pred_label in zip(document_ids, y_true, y_pred):
        if true_label != pred_label:
            misclassified.append(
                MisclassifiedDocument(
                    document_id=doc_id,
                    effective_family_id=family_by_doc[doc_id],
                    total_chunks=total_chunks_by_doc[doc_id],
                    true_label=true_label,
                    predicted_label=pred_label,
                )
            )
    return misclassified


def _bucket_for_chunk_count(n: int) -> str:
    if n == 1:
        return "1_chunk"
    if n <= 5:
        return "2_to_5_chunks"
    if n <= 20:
        return "6_to_20_chunks"
    return "21_plus_chunks"


def build_error_concentration_report(
    document_ids: list[str],
    y_true: list[str],
    y_pred: list[str],
    total_chunks_by_doc: dict[str, int],
    condition_results: dict[str, tuple[list[str], list[str]]],
) -> ErrorConcentrationReport:
    """`condition_results` maps condition_name -> (y_true, y_pred) for that condition, used
    to compare error rates for masked-vs-unmasked and complete-vs-partial without needing a
    second confusion matrix per condition."""
    buckets: dict[str, dict[str, int]] = {}
    for doc_id, true_label, pred_label in zip(document_ids, y_true, y_pred):
        bucket = _bucket_for_chunk_count(total_chunks_by_doc[doc_id])
        buckets.setdefault(bucket, {"correct": 0, "incorrect": 0})
        key = "correct" if true_label == pred_label else "incorrect"
        buckets[bucket][key] += 1

    def _error_rate(pair):
        yt, yp = pair
        if not yt:
            return 0.0
        errors = sum(1 for t, p in zip(yt, yp) if t != p)
        return round(100 * errors / len(yt), 2)

    masked_vs_unmasked = {
        name: _error_rate(pair)
        for name, pair in condition_results.items()
        if name in ("complete_unmasked", "complete_masked")
    }
    complete_vs_partial = {
        name: _error_rate(pair)
        for name, pair in condition_results.items()
        if name
        in (
            "complete_unmasked",
            "beginning_only_unmasked",
            "middle_only_unmasked",
            "end_only_unmasked",
            "beginning_middle_end_unmasked",
        )
    }

    return ErrorConcentrationReport(
        by_chunk_count_bucket=buckets,
        by_masked_vs_unmasked_error_rate=masked_vs_unmasked,
        by_complete_vs_partial_error_rate=complete_vs_partial,
        unseen_family_note=(
            "Not a meaningful axis here: the family-aware split (Checkpoint 4) guarantees "
            "zero effective_family_id overlap between train and validation by construction, "
            "so 100% of validation documents belong to families unseen during training -- "
            "this is a constant, not a variable, for this dataset/split design."
        ),
    )
