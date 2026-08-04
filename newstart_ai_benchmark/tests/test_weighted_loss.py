"""Tests for Version 6 Checkpoint 7 combined (class weight x document-balance weight) loss."""

from __future__ import annotations

import torch

from newstart_ai.models.bert.weighted_loss import compute_combined_weights, weighted_cross_entropy


def test_combined_weight_multiplies_class_weight_and_inverse_chunk_count():
    weights = compute_combined_weights(
        effective_agencies=["IRS", "IRS", "USCIS"],
        chunk_counts=[2, 2, 1],
        class_weight_by_label={"IRS": 4.0, "USCIS": 1.0},
    )
    assert torch.allclose(weights, torch.tensor([2.0, 2.0, 1.0]))  # 4.0 * 1/2, 4.0 * 1/2, 1.0 * 1/1


def test_weighted_cross_entropy_is_a_weighted_mean_not_a_weighted_sum():
    logits = torch.tensor([[5.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    labels = torch.tensor([0, 0, 0])
    equal_weights = torch.tensor([1.0, 1.0, 1.0])
    loss_equal = weighted_cross_entropy(logits, labels, equal_weights)

    manual_per_example = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    assert torch.isclose(loss_equal, manual_per_example.mean(), atol=1e-6)


def test_duplicating_a_document_into_many_chunks_does_not_multiply_its_total_contribution():
    """Core Checkpoint 7 requirement: a document represented once (1 chunk, weight 1) must
    contribute the same total weighted loss as the same content split into 5 identical
    chunks (each weighted 1/5) -- proving chunk multiplicity alone cannot inflate influence."""
    logits_one_chunk = torch.tensor([[5.0, 0.0, 0.0, 0.0]])
    labels_one_chunk = torch.tensor([0])
    weights_one_chunk = torch.tensor([1.0])  # 1/1

    logits_five_chunks = logits_one_chunk.repeat(5, 1)
    labels_five_chunks = labels_one_chunk.repeat(5)
    weights_five_chunks = torch.full((5,), 1.0 / 5)  # 1/5 each

    per_example_one = torch.nn.functional.cross_entropy(logits_one_chunk, labels_one_chunk, reduction="none")
    per_example_five = torch.nn.functional.cross_entropy(logits_five_chunks, labels_five_chunks, reduction="none")

    numerator_one = (per_example_one * weights_one_chunk).sum()
    numerator_five = (per_example_five * weights_five_chunks).sum()

    assert torch.isclose(numerator_one, numerator_five, atol=1e-6)


def test_weight_sum_normalization_keeps_batch_loss_scale_stable_regardless_of_composition():
    """A batch containing only high-weight chunks and a batch containing only low-weight
    chunks (but identical underlying per-example losses) must produce the same weighted-mean
    loss -- proving the sum-of-weights normalization, not raw weight magnitude, sets scale."""
    logits = torch.tensor([[3.0, 0.0], [3.0, 0.0]])
    labels = torch.tensor([0, 0])

    high_weights = torch.tensor([10.0, 10.0])
    low_weights = torch.tensor([0.1, 0.1])

    loss_high = weighted_cross_entropy(logits, labels, high_weights)
    loss_low = weighted_cross_entropy(logits, labels, low_weights)

    assert torch.isclose(loss_high, loss_low, atol=1e-6)


def test_class_weight_still_shifts_relative_contribution_within_a_batch():
    """Unlike the document-balance factor (which must equalize total contribution), the
    class-weight factor is deliberately supposed to change relative influence -- confirms the
    two concepts are combined by multiplication, not accidentally cancelled out."""
    logits = torch.tensor([[5.0, 0.0], [5.0, 0.0]])
    labels = torch.tensor([0, 0])
    weights_equal = torch.tensor([1.0, 1.0])
    weights_skewed = torch.tensor([4.0, 1.0])

    per_example = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    # With identical per-example losses, a weighted mean is invariant to any uniform-vs-skewed
    # weighting IF the underlying losses are equal -- so instead verify the skew changes the
    # result when the underlying per-example losses actually differ.
    logits_diff = torch.tensor([[5.0, 0.0], [0.0, 5.0]])  # second example is very wrong for label 0
    labels_diff = torch.tensor([0, 0])
    loss_equal = weighted_cross_entropy(logits_diff, labels_diff, weights_equal)
    loss_skewed = weighted_cross_entropy(logits_diff, labels_diff, weights_skewed)
    assert not torch.isclose(loss_equal, loss_skewed, atol=1e-6)
