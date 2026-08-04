"""Tests for Version 6 Checkpoint 7 agency class weights (training-document counts only)."""

from __future__ import annotations

import pandas as pd
import pytest

from newstart_ai.config import load_settings
from newstart_ai.models.bert.agency_class_weights import (
    build_agency_class_weight_manifest,
    compute_training_document_counts,
)

LABEL_ORDER = ["USCIS", "DMV", "SSA", "IRS"]


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def _doc_df(counts: dict[str, int]) -> pd.DataFrame:
    rows = []
    doc_id = 1
    for label, n in counts.items():
        for _ in range(n):
            rows.append({"document_id": str(doc_id), "effective_agency": label})
            doc_id += 1
    return pd.DataFrame(rows)


def test_counts_are_one_row_per_document_not_chunks():
    df = _doc_df({"USCIS": 3, "DMV": 2, "SSA": 1, "IRS": 1})
    counts = compute_training_document_counts(df, LABEL_ORDER)
    assert counts == {"USCIS": 3, "DMV": 2, "SSA": 1, "IRS": 1}


def test_manifest_never_reads_validation_or_test_columns(settings):
    """The function signature only accepts a single document-level DataFrame -- there is no
    way to pass validation/test label frequencies into it, which is itself the isolation
    guarantee (verified further by the dedicated isolation-proof tests)."""
    df = _doc_df({"USCIS": 183, "DMV": 157, "SSA": 104, "IRS": 17})
    manifest = build_agency_class_weight_manifest(df, LABEL_ORDER, settings)
    assert manifest.training_document_counts == {"USCIS": 183, "DMV": 157, "SSA": 104, "IRS": 17}
    assert manifest.computed_from.startswith("eligible training-document counts")


def test_weight_formula_matches_balanced_inverse_frequency(settings):
    df = _doc_df({"USCIS": 100, "DMV": 100, "SSA": 100, "IRS": 25})
    manifest = build_agency_class_weight_manifest(df, LABEL_ORDER, settings)
    total = 325
    expected_irs = total / (4 * 25)
    assert manifest.weighting_applied is True
    assert abs(manifest.raw_weights["IRS"] - expected_irs) < 1e-6


def test_no_weighting_applied_when_balanced(settings):
    df = _doc_df({"USCIS": 100, "DMV": 100, "SSA": 100, "IRS": 100})
    manifest = build_agency_class_weight_manifest(df, LABEL_ORDER, settings)
    assert manifest.weighting_applied is False
    assert all(w == 1.0 for w in manifest.raw_weights.values())


def test_normalized_weights_have_mean_one(settings):
    df = _doc_df({"USCIS": 183, "DMV": 157, "SSA": 104, "IRS": 17})
    manifest = build_agency_class_weight_manifest(df, LABEL_ORDER, settings)
    mean_normalized = sum(manifest.normalized_weights.values()) / len(manifest.normalized_weights)
    assert abs(mean_normalized - 1.0) < 1e-9


def test_label_order_is_preserved_in_manifest(settings):
    df = _doc_df({"USCIS": 10, "DMV": 10, "SSA": 10, "IRS": 10})
    manifest = build_agency_class_weight_manifest(df, LABEL_ORDER, settings)
    assert manifest.label_order == LABEL_ORDER
