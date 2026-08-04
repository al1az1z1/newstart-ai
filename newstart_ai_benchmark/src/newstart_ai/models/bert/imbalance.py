"""Class-weight computation for imbalanced training data.

Weights are always computed from the training split only (docs/BLUEPRINT.md Section 6) --
never from validation or test -- and only applied when the training-set imbalance ratio
meets or exceeds the configured threshold. No oversampling is used.
"""

from __future__ import annotations

import numpy as np


def imbalance_ratio(label_counts: dict[str, int]) -> float:
    """Majority-class count divided by minority-class count."""
    counts = list(label_counts.values())
    return max(counts) / min(counts)


def compute_class_weights(
    label_counts: dict[str, int], label_order: list[str], threshold: float
) -> np.ndarray | None:
    """Returns inverse-frequency class weights (indexed to match label_order), or None if the
    training set isn't imbalanced enough (per `threshold`) to warrant weighting."""
    if imbalance_ratio(label_counts) < threshold:
        return None

    total = sum(label_counts.values())
    num_classes = len(label_order)
    weights = np.array(
        [total / (num_classes * label_counts[label]) for label in label_order],
        dtype=np.float32,
    )
    return weights
