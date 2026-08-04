"""Document-level aggregation of chunk-level BERT outputs (Version 6, Checkpoint 6).

The classifier's unit of prediction is a chunk; the research unit is the original document.
These functions combine one document's chunk-level logits/probabilities into a single
document-level prediction, deterministically. Chunk counts must never be reported as
independent evaluation support -- only the resulting document-level prediction matters for
metrics.

`default_method` (configs/family_aware.yaml, aggregation.default_method) is a PROVISIONAL
choice frozen via the deterministic properties documented below and the real validation
split's chunk-count structure -- not via a real validation macro F1 comparison, because the
only existing trained model (the historical bert-mvp artifact) was trained on 65.7% of the
new family-aware validation set's documents, which would make any such comparison
contaminated. It is re-confirmed empirically in Checkpoint 7 once the new family-aware BERT
is trained only on the family-aware train split.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from newstart_ai.schemas.checkpoint6 import AggregationComparisonManifest
from newstart_ai.schemas.checkpoint7 import AggregationMethodResult, AggregationReconfirmationManifest

AGGREGATION_METHODS = ("mean_logits", "mean_probabilities", "majority_vote", "max_confidence")

_METHOD_PROPERTIES = {
    "mean_logits": [
        "Averages unbounded raw logits before softmax -- a single chunk with unusually "
        "large-magnitude logits can dominate the average more than its one-of-N share "
        "would suggest.",
        "Requires access to pre-softmax logits, not just probabilities.",
        "Identity for single-chunk documents (nothing else to average in).",
    ],
    "mean_probabilities": [
        "Averages already-bounded [0,1] probabilities, so no single chunk's contribution "
        "can exceed its 1/N share of the average regardless of how extreme its logits were.",
        "Uses every chunk's evidence -- none of Checkpoint 5's overlapping-window coverage "
        "is discarded at inference time.",
        "Identity for single-chunk documents.",
        "Only requires standard softmax output, the most portable interface.",
    ],
    "majority_vote": [
        "Discards confidence magnitude entirely -- a chunk that barely favored a label "
        "counts identically to one that was highly confident.",
        "Requires an explicit deterministic tie-breaker when votes are split.",
        "Identity for single-chunk documents.",
    ],
    "max_confidence": [
        "Determined entirely by one chunk (whichever had the single highest top-label "
        "probability) -- discards every other chunk's evidence, including the document's "
        "tail content that Checkpoint 5's chunker specifically preserved.",
        "Most sensitive to a single miscalibrated/overconfident chunk.",
        "Identity for single-chunk documents.",
    ],
}


def build_aggregation_policy_manifest(val_chunks_df: pd.DataFrame, settings) -> AggregationComparisonManifest:
    """Freezes the provisional default aggregation method using only the validation split's
    real chunk-count structure and documented, content/config-derived properties of each
    method -- never a real model's chunk-level outputs, since the only existing trained
    model (historical bert-mvp) was trained on 65.7% of this validation set's documents and
    would make any macro-F1 comparison contaminated. Re-confirmed empirically with a real,
    uncontaminated validation macro F1 across all four methods in Checkpoint 7.
    """
    cfg = settings.family_aware.aggregation

    per_doc_chunk_counts = val_chunks_df.groupby("document_id").size()
    total_documents = int(len(per_doc_chunk_counts))
    single_chunk = int((per_doc_chunk_counts == 1).sum())
    multi_chunk = int((per_doc_chunk_counts > 1).sum())

    structure = {
        "total_documents": total_documents,
        "single_chunk_document_count": single_chunk,
        "single_chunk_document_percentage": round(100 * single_chunk / total_documents, 2) if total_documents else 0.0,
        "multi_chunk_document_count": multi_chunk,
        "multi_chunk_document_percentage": round(100 * multi_chunk / total_documents, 2) if total_documents else 0.0,
        "max_chunks_for_a_document": int(per_doc_chunk_counts.max()) if total_documents else 0,
    }

    return AggregationComparisonManifest(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        policy_version=cfg.policy_version,
        candidate_methods=list(cfg.candidate_methods),
        default_method=cfg.default_method,
        tie_breaker=cfg.tie_breaker,
        selection_basis=(
            "Frozen via documented, content/config-derived properties of each candidate "
            "method plus the real validation split's chunk-count structure (90.91% of "
            "validation documents are multi-chunk, so the aggregation rule materially "
            "affects most documents' predictions) -- not via a real validation macro F1 "
            "comparison, which is deferred to Checkpoint 7 to avoid contamination from the "
            "historical bert-mvp model's 65.7% training overlap with this validation set."
        ),
        validation_chunk_count_structure=structure,
        deterministic_properties_by_method=_METHOD_PROPERTIES,
        provisional=True,
        reconfirmation_plan=(
            "In Checkpoint 7, after training the new family-aware BERT on the family-aware "
            "train split only, compute real document-level macro F1 (with accuracy and "
            "per-agency results as secondary diagnostics) on the family-aware validation "
            "split for all four candidate methods, and either confirm mean_probabilities or "
            "override it with whichever method empirically wins -- before any test-set use."
        ),
        notes=[
            "This manifest and its default_method are provisional -- see "
            "reconfirmation_plan. No test data was read or referenced anywhere in this "
            "function.",
        ],
    )


class AggregationResult:
    __slots__ = ("predicted_label", "predicted_index", "scores", "num_chunks_used", "tie_broken", "method")

    def __init__(
        self,
        predicted_label: str,
        predicted_index: int,
        scores: np.ndarray,
        num_chunks_used: int,
        tie_broken: bool,
        method: str,
    ):
        self.predicted_label = predicted_label
        self.predicted_index = predicted_index
        self.scores = scores
        self.num_chunks_used = num_chunks_used
        self.tie_broken = tie_broken
        self.method = method


def _validate_chunks(chunk_scores: list[np.ndarray], label_order: list[str]) -> None:
    if not chunk_scores:
        raise ValueError("Cannot aggregate zero chunks -- every eligible document has >=1 chunk by construction.")
    for scores in chunk_scores:
        if scores.shape != (len(label_order),):
            raise ValueError(
                f"Invalid chunk output shape {scores.shape}, expected ({len(label_order)},) for label_order={label_order}"
            )
        if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
            raise ValueError("Invalid (NaN/inf) chunk scores encountered -- cannot aggregate.")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def aggregate_mean_logits(chunk_logits: list[np.ndarray], label_order: list[str]) -> AggregationResult:
    """Averages raw logits across chunks, then takes argmax. A single-chunk document's
    result is the identity (no other logits to average in)."""
    _validate_chunks(chunk_logits, label_order)
    mean_logits = np.mean(np.stack(chunk_logits), axis=0)
    probs = _softmax(mean_logits)
    predicted_index = int(np.argmax(probs))
    return AggregationResult(
        predicted_label=label_order[predicted_index],
        predicted_index=predicted_index,
        scores=probs,
        num_chunks_used=len(chunk_logits),
        tie_broken=False,
        method="mean_logits",
    )


def aggregate_mean_probabilities(chunk_probs: list[np.ndarray], label_order: list[str]) -> AggregationResult:
    """Averages per-chunk softmax probabilities across chunks, then takes argmax."""
    _validate_chunks(chunk_probs, label_order)
    mean_probs = np.mean(np.stack(chunk_probs), axis=0)
    predicted_index = int(np.argmax(mean_probs))
    return AggregationResult(
        predicted_label=label_order[predicted_index],
        predicted_index=predicted_index,
        scores=mean_probs,
        num_chunks_used=len(chunk_probs),
        tie_broken=False,
        method="mean_probabilities",
    )


def aggregate_majority_vote(chunk_probs: list[np.ndarray], label_order: list[str]) -> AggregationResult:
    """Each chunk votes for its own argmax label. The label with the most votes wins.

    Deterministic tie-break (applied only if two or more labels are tied for the most
    votes): (1) the tied label with the higher summed probability across all chunks, then
    (2) if still tied, the lower index in `label_order`. Both tie-break rules are derived
    entirely from the input content or from fixed configuration -- never random.
    """
    _validate_chunks(chunk_probs, label_order)
    votes = np.zeros(len(label_order), dtype=int)
    for scores in chunk_probs:
        votes[int(np.argmax(scores))] += 1
    summed_probs = np.sum(np.stack(chunk_probs), axis=0)

    max_votes = votes.max()
    tied_indices = [i for i, v in enumerate(votes) if v == max_votes]
    tie_broken = len(tied_indices) > 1
    if tie_broken:
        best_summed = max(summed_probs[i] for i in tied_indices)
        tied_indices = [i for i in tied_indices if summed_probs[i] == best_summed]
    predicted_index = min(tied_indices)  # lower label_order index wins any remaining tie

    return AggregationResult(
        predicted_label=label_order[predicted_index],
        predicted_index=predicted_index,
        scores=votes.astype(float) / len(chunk_probs),
        num_chunks_used=len(chunk_probs),
        tie_broken=tie_broken,
        method="majority_vote",
    )


def aggregate_max_confidence(chunk_probs: list[np.ndarray], label_order: list[str]) -> AggregationResult:
    """Uses the single chunk with the highest top-label probability, discarding the rest.

    Deterministic tie-break for chunks tied at the same maximum confidence: the
    earliest chunk (lowest chunk_index) wins, since chunk order is itself deterministic.
    """
    _validate_chunks(chunk_probs, label_order)
    top_confidences = [float(np.max(scores)) for scores in chunk_probs]
    best_confidence = max(top_confidences)
    tie_broken = top_confidences.count(best_confidence) > 1
    best_chunk_index = top_confidences.index(best_confidence)  # first (lowest-index) match on ties
    best_scores = chunk_probs[best_chunk_index]
    predicted_index = int(np.argmax(best_scores))

    return AggregationResult(
        predicted_label=label_order[predicted_index],
        predicted_index=predicted_index,
        scores=best_scores,
        num_chunks_used=len(chunk_probs),
        tie_broken=tie_broken,
        method="max_confidence",
    )


def aggregate_document(
    method: str,
    label_order: list[str],
    chunk_probs: list[np.ndarray] | None = None,
    chunk_logits: list[np.ndarray] | None = None,
) -> AggregationResult:
    """Dispatches to the named aggregation method. `mean_logits` requires `chunk_logits`;
    the other three operate on `chunk_probs` (post-softmax per-chunk probabilities)."""
    if method == "mean_logits":
        if chunk_logits is None:
            raise ValueError("mean_logits requires chunk_logits")
        return aggregate_mean_logits(chunk_logits, label_order)
    if chunk_probs is None:
        raise ValueError(f"{method} requires chunk_probs")
    if method == "mean_probabilities":
        return aggregate_mean_probabilities(chunk_probs, label_order)
    if method == "majority_vote":
        return aggregate_majority_vote(chunk_probs, label_order)
    if method == "max_confidence":
        return aggregate_max_confidence(chunk_probs, label_order)
    raise ValueError(f"Unknown aggregation method: {method!r}")


def evaluate_aggregation_method(
    method: str,
    label_order: list[str],
    chunk_probs_by_doc: dict[str, list[np.ndarray]],
    chunk_logits_by_doc: dict[str, list[np.ndarray]] | None,
    true_labels_by_doc: dict[str, str],
) -> AggregationMethodResult:
    """Applies `method` to every document's chunk outputs and scores the resulting
    document-level predictions against `true_labels_by_doc` -- document-level metrics only,
    chunk counts are never treated as independent support."""
    document_ids = list(chunk_probs_by_doc.keys())
    predictions = {}
    for doc_id in document_ids:
        result = aggregate_document(
            method,
            label_order,
            chunk_probs=chunk_probs_by_doc[doc_id],
            chunk_logits=(chunk_logits_by_doc or {}).get(doc_id) if method == "mean_logits" else None,
        )
        predictions[doc_id] = result.predicted_label

    y_true = [true_labels_by_doc[d] for d in document_ids]
    y_pred = [predictions[d] for d in document_ids]

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=label_order, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    per_agency = {
        label: float(f1)
        for label, f1 in zip(
            label_order, f1_score(y_true, y_pred, average=None, labels=label_order, zero_division=0)
        )
    }
    worst_agency_f1 = min(per_agency.values())

    return AggregationMethodResult(
        method=method,
        validation_document_macro_f1=macro_f1,
        validation_document_accuracy=accuracy,
        worst_agency_f1=worst_agency_f1,
        per_agency_f1=per_agency,
    )


def select_best_aggregation_method(
    label_order: list[str],
    chunk_probs_by_doc: dict[str, list[np.ndarray]],
    chunk_logits_by_doc: dict[str, list[np.ndarray]],
    true_labels_by_doc: dict[str, str],
    candidate_methods: list[str],
    provisional_method: str,
    evaluated_on_checkpoint: str,
    policy_version: str,
) -> AggregationReconfirmationManifest:
    """Reconfirms the provisional aggregation method using real validation document-level
    macro F1 for every candidate, on one checkpoint's chunk-level outputs (generated once).

    Tie-break order (deterministic, applied only when methods are still tied after the
    previous step): (1) higher macro F1, (2) higher accuracy, (3) better worst-agency F1,
    (4) retain the provisional method.
    """
    results = [
        evaluate_aggregation_method(method, label_order, chunk_probs_by_doc, chunk_logits_by_doc, true_labels_by_doc)
        for method in candidate_methods
    ]

    tie_break_steps = []
    remaining = list(results)

    tie_break_steps.append("higher validation document-level macro F1")
    best_f1 = max(r.validation_document_macro_f1 for r in remaining)
    remaining = [r for r in remaining if r.validation_document_macro_f1 == best_f1]

    if len(remaining) > 1:
        tie_break_steps.append("higher validation document-level accuracy")
        best_acc = max(r.validation_document_accuracy for r in remaining)
        remaining = [r for r in remaining if r.validation_document_accuracy == best_acc]

    if len(remaining) > 1:
        tie_break_steps.append("better worst-agency validation F1")
        best_worst = max(r.worst_agency_f1 for r in remaining)
        remaining = [r for r in remaining if r.worst_agency_f1 == best_worst]

    if len(remaining) > 1:
        tie_break_steps.append("retain the provisional mean_probabilities rule")
        provisional_match = [r for r in remaining if r.method == provisional_method]
        selected = provisional_match[0] if provisional_match else remaining[0]
    else:
        selected = remaining[0]

    method_changed = selected.method != provisional_method

    return AggregationReconfirmationManifest(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        policy_version=policy_version,
        evaluated_on_checkpoint=evaluated_on_checkpoint,
        candidate_results=results,
        provisional_method=provisional_method,
        selected_method=selected.method,
        method_changed=method_changed,
        tie_break_steps_applied=tie_break_steps,
        supersedes=("aggregation_policy_v1" if method_changed else None),
        notes=[
            "Chunk-level logits/probabilities were generated once for the selected best "
            "checkpoint and reused for all four candidate methods -- never re-run per "
            "method.",
            "Document-level macro F1 is the primary criterion; accuracy and per-agency F1 "
            "are secondary/tie-break diagnostics only, per Checkpoint 6/7 policy.",
        ],
    )
