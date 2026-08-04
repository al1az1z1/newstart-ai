"""Agency class weights for the family-aware chunked BERT (Version 6, Checkpoint 7).

Computed strictly from eligible TRAINING DOCUMENT counts (one row per document in the
frozen family-aware train split) -- never from chunk counts, and never from validation or
test label frequencies. This is a deliberately separate concept from
`newstart_ai.models.bert.document_balancing`: class weighting corrects unequal document
counts per label; document balancing corrects unequal chunk multiplicity per document.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from newstart_ai.models.bert.imbalance import compute_class_weights, imbalance_ratio
from newstart_ai.schemas.checkpoint7 import AgencyClassWeightManifest


def compute_training_document_counts(train_split_df: pd.DataFrame, label_order: list[str]) -> dict[str, int]:
    """One count per label, from one row per document (`train_split_df` must already be
    document-level, e.g. the frozen family-aware train.csv -- never a chunk DataFrame)."""
    counts = train_split_df["effective_agency"].value_counts().to_dict()
    return {label: int(counts.get(label, 0)) for label in label_order}


def build_agency_class_weight_manifest(train_split_df: pd.DataFrame, label_order: list[str], settings) -> AgencyClassWeightManifest:
    threshold = settings.family_aware.training.imbalance.weighted_loss_threshold
    counts = compute_training_document_counts(train_split_df, label_order)

    raw = compute_class_weights(counts, label_order, threshold)
    weighting_applied = raw is not None
    if raw is None:
        raw_dict = {label: 1.0 for label in label_order}
    else:
        raw_dict = {label: float(w) for label, w in zip(label_order, raw)}

    mean_weight = sum(raw_dict.values()) / len(raw_dict)
    normalized_dict = {label: w / mean_weight for label, w in raw_dict.items()}

    return AgencyClassWeightManifest(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        label_order=label_order,
        training_document_counts=counts,
        imbalance_ratio=float(imbalance_ratio(counts)),
        weighted_loss_threshold=threshold,
        weighting_applied=weighting_applied,
        formula="weight[label] = total_training_documents / (num_labels * training_document_count[label])",
        raw_weights=raw_dict,
        normalized_weights=normalized_dict,
        computed_from="eligible training-document counts (family_aware_splits/train.csv, one row per document)",
        notes=[
            "Never computed from chunk counts, validation label frequencies, or test label "
            "frequencies -- only from the frozen family-aware train split's effective_agency "
            "column, one row per document.",
            "Separate from document_balancing.compute_inverse_chunk_count_weights, which "
            "corrects unequal chunk multiplicity per document, not unequal document counts "
            "per label.",
        ],
    )
