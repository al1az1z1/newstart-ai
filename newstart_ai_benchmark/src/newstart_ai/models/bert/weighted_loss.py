"""Combined per-chunk loss weighting for the family-aware chunked BERT (Version 6,
Checkpoint 7).

    combined_weight(chunk) = agency_class_weight[effective_agency] * (1 / total_chunks_for_parent_document)

The two factors are deliberately independent: agency class weighting corrects unequal
TRAINING DOCUMENT counts per label; the inverse-chunk-count factor corrects unequal chunk
MULTIPLICITY per document. Per-example cross-entropy is computed first (unreduced), then
multiplied by `combined_weight`, then normalized by the sum of weights in the batch (a
weighted mean) -- so a batch's overall loss scale never depends on how many low-weight chunks
of a long document happen to land in it.
"""

from __future__ import annotations

import torch


def compute_combined_weights(effective_agencies: list[str], chunk_counts: list[int], class_weight_by_label: dict[str, float]) -> torch.Tensor:
    weights = [class_weight_by_label[agency] * (1.0 / count) for agency, count in zip(effective_agencies, chunk_counts)]
    return torch.tensor(weights, dtype=torch.float32)


def weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, combined_weights: torch.Tensor) -> torch.Tensor:
    """Per-example cross-entropy (reduction='none'), multiplied by `combined_weights`, then
    normalized by the sum of weights in this batch -- a weighted mean, not a weighted sum, so
    the loss magnitude stays comparable across batches regardless of composition."""
    per_example_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    weighted = per_example_loss * combined_weights
    return weighted.sum() / combined_weights.sum()
