"""Tests for Version 6 Checkpoint 6 document-balancing (long-document training-loss control)."""

from __future__ import annotations

import pandas as pd
import pytest

from newstart_ai.config import load_settings
from newstart_ai.models.bert.document_balancing import (
    build_document_balancing_manifest,
    build_document_balancing_report,
    compute_inverse_chunk_count_weights,
)


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def _make_chunks(doc_chunk_counts: dict[str, int], agency: str = "USCIS") -> pd.DataFrame:
    rows = []
    for doc_id, n in doc_chunk_counts.items():
        for i in range(n):
            rows.append({"document_id": doc_id, "chunk_id": f"{doc_id}_{i}", "agency": agency})
    return pd.DataFrame(rows)


def test_single_chunk_document_gets_weight_one():
    df = _make_chunks({"d1": 1})
    weights = compute_inverse_chunk_count_weights(df)
    assert weights.tolist() == [1.0]


def test_multi_chunk_document_weight_is_inverse_of_count():
    df = _make_chunks({"d1": 5})
    weights = compute_inverse_chunk_count_weights(df)
    assert all(abs(w - 0.2) < 1e-12 for w in weights)


def test_every_documents_total_weight_sums_to_one():
    df = _make_chunks({"d1": 1, "d2": 519, "d3": 7})
    weights = compute_inverse_chunk_count_weights(df)
    totals = df.assign(w=weights).groupby("document_id")["w"].sum()
    assert all(abs(t - 1.0) < 1e-9 for t in totals)


def test_long_document_contribution_report_shows_large_reduction():
    doc_counts = {"dominant": 519}
    doc_counts.update({f"ordinary_{i}": 1 for i in range(460)})
    df = _make_chunks(doc_counts)

    report = build_document_balancing_report(df, top_n=1)
    assert report["weight_sum_equals_document_count"] is True
    dominant = report["largest_documents_effect"][0]
    assert dominant["document_id"] == "dominant"
    assert dominant["total_chunks"] == 519
    # Raw chunk share should be large...
    assert dominant["raw_chunk_share_percent"] > 50.0
    # ...but the weighted contribution share must equal every other document's fair 1/N share.
    total_documents = df["document_id"].nunique()
    assert abs(dominant["weighted_contribution_share_percent"] - 100 / total_documents) < 1e-3
    assert dominant["weighted_contribution_share_percent"] < dominant["raw_chunk_share_percent"]


def test_document_balancing_manifest_matches_configured_policy(settings):
    df = _make_chunks({"d1": 1, "d2": 3}, agency="DMV")
    manifest = build_document_balancing_manifest(df, settings)
    assert manifest.method == settings.family_aware.document_balancing.method
    assert manifest.total_training_documents == 2
    assert manifest.total_training_chunks == 4
    assert manifest.weight_sum_equals_document_count is True
    assert manifest.separate_from_agency_class_weighting is True
