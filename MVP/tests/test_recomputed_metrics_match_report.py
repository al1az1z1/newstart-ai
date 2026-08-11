"""Recomputes metrics directly from raw, frozen predictions (never trusts a saved number
on faith) and asserts they agree with the value already in the submitted manifests. This is
the mechanical proof that the offline analysis actually recalculates results rather than
hard-coding them."""

from __future__ import annotations

import pytest

from newstart_ai_mvp import artifact_report as ar


def test_bert_primary_condition_recomputed_matches_report(settings):
    report = ar.describe_bert_test_results(settings)
    assert report["agrees_with_report"]
    assert report["recomputed_macro_f1"] == pytest.approx(report["reported_macro_f1"], abs=1e-9)
    assert report["recomputed_accuracy"] == pytest.approx(report["reported_accuracy"], abs=1e-9)


def test_llm_primary_condition_recomputed_matches_report(settings):
    report = ar.describe_llm_predictions(settings)
    assert report["agrees_with_report"]
    assert report["total_cases"] == 990


def test_rag_primary_condition_recomputed_matches_report(settings):
    report = ar.describe_rag_predictions(settings)
    assert report["agrees_with_report"]
    assert report["total_cases"] == 990


def test_split_composition_matches_documented_counts(settings):
    report = ar.describe_split(settings)
    assert report["counts"]["train"]["documents"] == 461
    assert report["counts"]["validation"]["documents"] == 99
    assert report["counts"]["test"]["documents"] == 99


def test_chunk_counts_match_documented_totals(settings):
    report = ar.describe_chunks(settings)
    assert report["counts"]["train"]["chunks"] == 4300
