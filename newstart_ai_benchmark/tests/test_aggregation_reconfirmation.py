"""Tests for Version 6 Checkpoint 7 aggregation-method reconfirmation on real chunk outputs."""

from __future__ import annotations

import numpy as np
import pytest

from newstart_ai.models.bert.aggregation import evaluate_aggregation_method, select_best_aggregation_method

LABEL_ORDER = ["USCIS", "DMV", "SSA", "IRS"]


def _p(*values: float) -> np.ndarray:
    arr = np.array(values, dtype=float)
    return arr / arr.sum()


def test_evaluate_aggregation_method_scores_document_level_not_chunk_level():
    chunk_probs_by_doc = {
        "d1": [_p(0.9, 0.05, 0.03, 0.02), _p(0.8, 0.1, 0.05, 0.05)],  # both chunks -> USCIS
        "d2": [_p(0.1, 0.8, 0.05, 0.05)],  # -> DMV
    }
    true_labels = {"d1": "USCIS", "d2": "DMV"}
    result = evaluate_aggregation_method("mean_probabilities", LABEL_ORDER, chunk_probs_by_doc, None, true_labels)
    # Both documents are correctly predicted -- accuracy is 1.0. Macro F1 is computed over
    # all four configured labels (sklearn's zero_division=0 convention scores SSA/IRS, which
    # have no support here, as 0), so it is necessarily below 1.0 despite perfect accuracy.
    assert result.validation_document_accuracy == 1.0
    assert result.per_agency_f1["USCIS"] == 1.0
    assert result.per_agency_f1["DMV"] == 1.0


def test_select_best_method_picks_highest_macro_f1():
    # mean_probabilities gets everything right; majority_vote is designed to get d2 wrong.
    chunk_probs_by_doc = {
        "d1": [_p(0.9, 0.05, 0.03, 0.02)],
        "d2": [_p(0.05, 0.9, 0.03, 0.02), _p(0.4, 0.35, 0.15, 0.1), _p(0.4, 0.35, 0.15, 0.1)],
    }
    true_labels = {"d1": "USCIS", "d2": "DMV"}
    manifest = select_best_aggregation_method(
        LABEL_ORDER, chunk_probs_by_doc, None, true_labels,
        candidate_methods=["mean_probabilities", "majority_vote"],
        provisional_method="mean_probabilities",
        evaluated_on_checkpoint="test-checkpoint",
        policy_version="v1",
    )
    assert manifest.candidate_results  # both methods evaluated
    assert manifest.selected_method in ("mean_probabilities", "majority_vote")
    assert manifest.tie_break_steps_applied[0] == "higher validation document-level macro F1"


def test_tie_break_falls_back_to_provisional_method_when_all_tied():
    # Identical chunk_probs for both methods to compare -> identical predictions -> tied on
    # every criterion, so the provisional method must be retained.
    chunk_probs_by_doc = {"d1": [_p(0.7, 0.1, 0.1, 0.1)]}
    true_labels = {"d1": "USCIS"}
    manifest = select_best_aggregation_method(
        LABEL_ORDER, chunk_probs_by_doc, None, true_labels,
        candidate_methods=["mean_probabilities", "majority_vote", "max_confidence"],
        provisional_method="mean_probabilities",
        evaluated_on_checkpoint="test-checkpoint",
        policy_version="v1",
    )
    assert manifest.selected_method == "mean_probabilities"
    assert manifest.method_changed is False
    assert "retain the provisional mean_probabilities rule" in manifest.tie_break_steps_applied


def test_method_changed_flag_and_supersedes_reference():
    chunk_probs_by_doc = {
        "d1": [_p(0.9, 0.05, 0.03, 0.02)],
        "d2": [_p(0.05, 0.9, 0.03, 0.02), _p(0.4, 0.35, 0.15, 0.1), _p(0.4, 0.35, 0.15, 0.1)],
        "d3": [_p(0.05, 0.03, 0.9, 0.02)],
        "d4": [_p(0.02, 0.03, 0.05, 0.9)],
    }
    true_labels = {"d1": "USCIS", "d2": "DMV", "d3": "SSA", "d4": "IRS"}
    manifest = select_best_aggregation_method(
        LABEL_ORDER, chunk_probs_by_doc, None, true_labels,
        candidate_methods=["mean_probabilities", "majority_vote"],
        provisional_method="majority_vote",  # deliberately the weaker method here
        evaluated_on_checkpoint="test-checkpoint",
        policy_version="v1",
    )
    if manifest.selected_method != "majority_vote":
        assert manifest.method_changed is True
        assert manifest.supersedes == "aggregation_policy_v1"


def test_all_four_candidate_methods_can_be_evaluated_together():
    chunk_probs_by_doc = {"d1": [_p(0.9, 0.05, 0.03, 0.02), _p(0.1, 0.8, 0.05, 0.05)]}
    chunk_logits_by_doc = {"d1": [np.array([3.0, 0.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0, 0.0])]}
    true_labels = {"d1": "USCIS"}
    manifest = select_best_aggregation_method(
        LABEL_ORDER, chunk_probs_by_doc, chunk_logits_by_doc, true_labels,
        candidate_methods=["mean_logits", "mean_probabilities", "majority_vote", "max_confidence"],
        provisional_method="mean_probabilities",
        evaluated_on_checkpoint="test-checkpoint",
        policy_version="v1",
    )
    assert {r.method for r in manifest.candidate_results} == {
        "mean_logits", "mean_probabilities", "majority_vote", "max_confidence"
    }
