"""Prevents documents with many chunks from dominating BERT training loss (Version 6,
Checkpoint 6).

Every eligible training document should influence optimization roughly in proportion to
being *one document*, not in proportion to however many overlapping token windows Checkpoint
5's chunker happened to produce for it. Document 739 (IRS, ~639k characters) alone produced
519 of the 4,300 training chunks (12.07%) -- with equal per-chunk weight, one document would
out-influence hundreds of ordinary single-chunk documents combined.

This is a deliberately separate concept from agency class weighting (configs/bert.yaml
imbalance section): document balancing corrects unequal chunk MULTIPLICITY per document;
agency class weighting corrects unequal DOCUMENT counts per label. Class weights for the new
family-aware model must be computed from eligible TRAINING DOCUMENT counts (one row per
document), never from training chunk counts -- this module never touches class weights.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from newstart_ai.schemas.checkpoint6 import DocumentBalancingManifest, LargeDocumentContribution


def compute_inverse_chunk_count_weights(train_chunks_df: pd.DataFrame) -> pd.Series:
    """Returns one per-chunk training weight, indexed like `train_chunks_df`, equal to
    1 / (number of chunks that document produced).

    This is the frozen policy (configs/family_aware.yaml, document_balancing.method).
    Chosen over the alternatives below because it is the only one of the four that:
      - uses every chunk (no information from any window is discarded, unlike a hard cap),
      - requires no dataloader-level custom sampling logic (a plain per-example loss weight
        works with a standard DataLoader), and
      - guarantees each document's total training contribution sums to exactly 1.0
        regardless of its chunk count, which is the literal statement of "no document should
        out-influence another because of chunk multiplicity."

    Considered and rejected as the frozen default (kept here as documented alternatives, not
    implemented as competing code paths, since only one policy may be frozen):
      - A deterministic max-chunks-per-document cap: simpler, but silently discards
        information from any excess window of a long document (e.g. would drop over 500 of
        document 739's windows), contradicting Checkpoint 5's "retain the tail, never
        silently discard" chunking requirement in spirit.
      - A document-balanced sampler (uniformly sample one document, then one of its chunks
        per training step): mathematically equivalent in expectation to inverse-chunk-count
        weighting, but adds stateful, harder-to-reproduce sampling logic for no additional
        benefit over the simpler per-example weight.
    """
    chunk_counts = train_chunks_df.groupby("document_id")["chunk_id"].transform("size")
    return 1.0 / chunk_counts


def build_document_balancing_report(train_chunks_df: pd.DataFrame, top_n: int = 10) -> dict:
    """Quantifies the effect of `compute_inverse_chunk_count_weights` -- in particular, the
    reduction in effective training-loss share for the largest documents by chunk count."""
    weights = compute_inverse_chunk_count_weights(train_chunks_df)
    total_documents = train_chunks_df["document_id"].nunique()
    total_chunks = len(train_chunks_df)

    per_doc = train_chunks_df.groupby("document_id").agg(
        agency=("agency", "first"),
        total_chunks=("chunk_id", "size"),
    )
    per_doc["raw_chunk_share_percent"] = round(100 * per_doc["total_chunks"] / total_chunks, 4)
    # Under inverse-chunk-count weighting, every document contributes total weight exactly
    # 1.0, so its share of the total weighted training mass is uniformly 1/total_documents.
    per_doc["weighted_contribution_share_percent"] = round(100 / total_documents, 4)

    largest = per_doc.sort_values("total_chunks", ascending=False).head(top_n).reset_index()

    weight_sums = train_chunks_df.assign(_weight=weights).groupby("document_id")["_weight"].sum()
    weight_sum_equals_document_count = bool(
        (weight_sums.round(9) == 1.0).all()
    )

    return {
        "total_training_documents": int(total_documents),
        "total_training_chunks": int(total_chunks),
        "weight_sum_equals_document_count": weight_sum_equals_document_count,
        "largest_documents_effect": [
            {
                "document_id": str(r.document_id),
                "agency": r.agency,
                "total_chunks": int(r.total_chunks),
                "raw_chunk_share_percent": float(r.raw_chunk_share_percent),
                "weighted_contribution_share_percent": float(r.weighted_contribution_share_percent),
            }
            for r in largest.itertuples(index=False)
        ],
    }


def build_document_balancing_manifest(train_chunks_df: pd.DataFrame, settings) -> DocumentBalancingManifest:
    cfg = settings.family_aware.document_balancing
    report = build_document_balancing_report(train_chunks_df)

    return DocumentBalancingManifest(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        policy_version=cfg.policy_version,
        method=cfg.method,
        total_training_documents=report["total_training_documents"],
        total_training_chunks=report["total_training_chunks"],
        weight_sum_equals_document_count=report["weight_sum_equals_document_count"],
        largest_documents_effect=[
            LargeDocumentContribution(**row) for row in report["largest_documents_effect"]
        ],
        separate_from_agency_class_weighting=True,
        notes=[
            "Document balancing corrects unequal chunk multiplicity per document (this "
            "manifest). Agency class weighting corrects unequal document counts per label "
            "and must be computed from eligible TRAINING DOCUMENT counts, never training "
            "chunk counts -- deferred to Checkpoint 7's training step, not implemented here.",
        ],
    )
