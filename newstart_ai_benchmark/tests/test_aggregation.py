"""Tests for Version 6 Checkpoint 6 document-level aggregation of chunk outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from newstart_ai.config import load_settings
from newstart_ai.models.bert.aggregation import (
    aggregate_document,
    aggregate_majority_vote,
    aggregate_max_confidence,
    aggregate_mean_logits,
    aggregate_mean_probabilities,
    build_aggregation_policy_manifest,
)

LABEL_ORDER = ["USCIS", "DMV", "SSA", "IRS"]


def _probs(*values: float) -> np.ndarray:
    arr = np.array(values, dtype=float)
    return arr / arr.sum()


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def test_mean_probabilities_single_chunk_is_identity():
    chunk = [_probs(0.7, 0.1, 0.1, 0.1)]
    result = aggregate_mean_probabilities(chunk, LABEL_ORDER)
    assert result.predicted_label == "USCIS"
    assert result.num_chunks_used == 1
    assert not result.tie_broken


def test_mean_probabilities_multi_chunk_averages():
    chunks = [_probs(0.9, 0.05, 0.03, 0.02), _probs(0.1, 0.8, 0.05, 0.05)]
    result = aggregate_mean_probabilities(chunks, LABEL_ORDER)
    expected_mean = np.mean(np.stack(chunks), axis=0)
    assert np.allclose(result.scores, expected_mean)
    assert result.predicted_index == int(np.argmax(expected_mean))


def test_mean_logits_uses_softmax_and_matches_manual_computation():
    logits = [np.array([2.0, 0.0, 0.0, 0.0]), np.array([0.0, 3.0, 0.0, 0.0])]
    result = aggregate_mean_logits(logits, LABEL_ORDER)
    mean_logits = np.mean(np.stack(logits), axis=0)
    shifted = mean_logits - mean_logits.max()
    expected = np.exp(shifted) / np.exp(shifted).sum()
    assert np.allclose(result.scores, expected)
    assert result.predicted_label == "DMV"  # mean logits favor index 1


def test_majority_vote_clear_winner():
    chunks = [_probs(0.9, 0.05, 0.03, 0.02), _probs(0.8, 0.1, 0.05, 0.05), _probs(0.1, 0.8, 0.05, 0.05)]
    result = aggregate_majority_vote(chunks, LABEL_ORDER)
    assert result.predicted_label == "USCIS"
    assert not result.tie_broken


def test_majority_vote_tie_breaks_by_summed_probability():
    # Two chunks vote USCIS (index 0), two vote DMV (index 1) -- tied 2-2.
    # USCIS's votes are much more confident, so summed probability should favor it.
    chunks = [
        _probs(0.99, 0.01, 0.0, 0.0),
        _probs(0.9, 0.05, 0.03, 0.02),
        _probs(0.05, 0.55, 0.2, 0.2),
        _probs(0.1, 0.51, 0.19, 0.2),
    ]
    result = aggregate_majority_vote(chunks, LABEL_ORDER)
    assert result.tie_broken
    assert result.predicted_label == "USCIS"


def test_majority_vote_tie_breaks_by_label_order_as_final_fallback():
    # Perfectly symmetric tie: one vote each for USCIS and DMV, identical summed
    # probability -- must fall back to the lower label_order index (USCIS, index 0).
    chunks = [_probs(0.7, 0.1, 0.1, 0.1), _probs(0.1, 0.7, 0.1, 0.1)]
    result = aggregate_majority_vote(chunks, LABEL_ORDER)
    assert result.tie_broken
    assert result.predicted_label == "USCIS"


def test_max_confidence_picks_the_single_most_confident_chunk():
    chunks = [_probs(0.5, 0.2, 0.2, 0.1), _probs(0.05, 0.9, 0.03, 0.02)]
    result = aggregate_max_confidence(chunks, LABEL_ORDER)
    assert result.predicted_label == "DMV"
    assert not result.tie_broken


def test_max_confidence_tie_breaks_to_earliest_chunk():
    chunks = [_probs(0.8, 0.1, 0.05, 0.05), _probs(0.1, 0.8, 0.05, 0.05)]
    result = aggregate_max_confidence(chunks, LABEL_ORDER)
    assert result.tie_broken
    assert result.predicted_label == "USCIS"  # first (chunk_index 0) wins the tie


def test_missing_chunks_raises_a_clear_error():
    with pytest.raises(ValueError):
        aggregate_mean_probabilities([], LABEL_ORDER)


def test_invalid_chunk_output_shape_raises():
    with pytest.raises(ValueError):
        aggregate_mean_probabilities([np.array([0.5, 0.5])], LABEL_ORDER)


def test_invalid_nan_chunk_output_raises():
    with pytest.raises(ValueError):
        aggregate_mean_probabilities([np.array([np.nan, 0.5, 0.3, 0.2])], LABEL_ORDER)


def test_dispatcher_routes_to_correct_method():
    chunks = [_probs(0.9, 0.05, 0.03, 0.02)]
    for method in ("mean_probabilities", "majority_vote", "max_confidence"):
        result = aggregate_document(method, LABEL_ORDER, chunk_probs=chunks)
        assert result.method == method
    result = aggregate_document("mean_logits", LABEL_ORDER, chunk_logits=[np.array([2.0, 0.0, 0.0, 0.0])])
    assert result.method == "mean_logits"


def test_dispatcher_unknown_method_raises():
    with pytest.raises(ValueError):
        aggregate_document("unknown_method", LABEL_ORDER, chunk_probs=[_probs(0.5, 0.2, 0.2, 0.1)])


def test_aggregation_policy_manifest_uses_only_validation_structure(settings):
    val_chunks = pd.DataFrame(
        {
            "document_id": ["d1", "d1", "d2", "d3", "d3", "d3"],
        }
    )
    manifest = build_aggregation_policy_manifest(val_chunks, settings)
    assert manifest.default_method == settings.family_aware.aggregation.default_method
    assert manifest.provisional is True
    assert manifest.validation_chunk_count_structure["total_documents"] == 3
    assert manifest.validation_chunk_count_structure["single_chunk_document_count"] == 1
    assert manifest.validation_chunk_count_structure["multi_chunk_document_count"] == 2
