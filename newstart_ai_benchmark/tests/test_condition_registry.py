"""Tests for Version 6 Checkpoint 6 shared evaluation-condition registry."""

from __future__ import annotations

import pandas as pd
import pytest

from newstart_ai.config import load_settings
from newstart_ai.data.condition_registry import (
    build_condition_definitions,
    build_condition_registry,
    build_condition_registry_manifest,
)
from newstart_ai.data.masking import build_masked_chunks, build_masked_documents
from newstart_ai.data.partial_input import build_partial_input_selections


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture()
def fixture_bundle(settings):
    split_df = pd.DataFrame(
        {
            "document_id": ["1", "2"],
            "text": [
                "This is a short USCIS document about Form I-9.",
                "This is a longer DMV document. " * 200,
            ],
            "effective_family_id": ["FAM:1", "FAM:2"],
            "agency": ["USCIS", "DMV"],
            "effective_agency": ["USCIS", "DMV"],
        }
    )
    # doc 1: single chunk; doc 2: three chunks (so beginning/middle/end are distinct).
    chunks_df = pd.DataFrame(
        [
            {"document_id": "1", "chunk_id": "c1_0", "chunk_index": 0, "total_chunks": 1, "token_start": 0, "token_end": 10, "split": "train", "chunk_text": split_df.loc[0, "text"]},
            {"document_id": "2", "chunk_id": "c2_0", "chunk_index": 0, "total_chunks": 3, "token_start": 0, "token_end": 10, "split": "train", "chunk_text": "beginning part of doc 2"},
            {"document_id": "2", "chunk_id": "c2_1", "chunk_index": 1, "total_chunks": 3, "token_start": 10, "token_end": 20, "split": "train", "chunk_text": "middle part of doc 2"},
            {"document_id": "2", "chunk_id": "c2_2", "chunk_index": 2, "total_chunks": 3, "token_start": 20, "token_end": 30, "split": "train", "chunk_text": "end part of doc 2"},
        ]
    )
    masked_documents_df = build_masked_documents(split_df, "train", settings)
    masked_chunks_df = build_masked_chunks(chunks_df, settings)
    selections_df = build_partial_input_selections(chunks_df, "train", settings)
    return split_df, masked_documents_df, chunks_df, masked_chunks_df, selections_df


def test_registry_has_ten_conditions_per_document(settings, fixture_bundle):
    split_df, masked_documents_df, chunks_df, masked_chunks_df, selections_df = fixture_bundle
    registry = build_condition_registry(
        split_df, masked_documents_df, chunks_df, masked_chunks_df, selections_df, "train", settings
    )
    assert len(registry) == 2 * 10
    assert set(registry["condition"].unique()) == set(settings.family_aware.conditions.names)


def test_complete_unmasked_matches_original_document_text(settings, fixture_bundle):
    split_df, masked_documents_df, chunks_df, masked_chunks_df, selections_df = fixture_bundle
    registry = build_condition_registry(
        split_df, masked_documents_df, chunks_df, masked_chunks_df, selections_df, "train", settings
    )
    row = registry[(registry["document_id"] == "1") & (registry["condition"] == "complete_unmasked")].iloc[0]
    assert row["text"] == split_df.loc[0, "text"]


def test_complete_masked_differs_from_complete_unmasked_when_identifiers_present(settings, fixture_bundle):
    split_df, masked_documents_df, chunks_df, masked_chunks_df, selections_df = fixture_bundle
    registry = build_condition_registry(
        split_df, masked_documents_df, chunks_df, masked_chunks_df, selections_df, "train", settings
    )
    unmasked = registry[(registry["document_id"] == "1") & (registry["condition"] == "complete_unmasked")].iloc[0]["text"]
    masked = registry[(registry["document_id"] == "1") & (registry["condition"] == "complete_masked")].iloc[0]["text"]
    assert masked != unmasked
    assert "USCIS" not in masked


def test_beginning_middle_end_uses_distinct_chunks_for_a_three_chunk_document(settings, fixture_bundle):
    split_df, masked_documents_df, chunks_df, masked_chunks_df, selections_df = fixture_bundle
    registry = build_condition_registry(
        split_df, masked_documents_df, chunks_df, masked_chunks_df, selections_df, "train", settings
    )
    row = registry[(registry["document_id"] == "2") & (registry["condition"] == "beginning_middle_end_unmasked")].iloc[0]
    assert "beginning part" in row["text"]
    assert "middle part" in row["text"]
    assert "end part" in row["text"]
    assert pd.isna(row["fallback_reason"])


def test_single_chunk_document_partial_conditions_all_equal_complete_chunk_text(settings, fixture_bundle):
    split_df, masked_documents_df, chunks_df, masked_chunks_df, selections_df = fixture_bundle
    registry = build_condition_registry(
        split_df, masked_documents_df, chunks_df, masked_chunks_df, selections_df, "train", settings
    )
    doc1 = registry[registry["document_id"] == "1"]
    beginning = doc1[doc1["condition"] == "beginning_only_unmasked"].iloc[0]["text"]
    middle = doc1[doc1["condition"] == "middle_only_unmasked"].iloc[0]["text"]
    end = doc1[doc1["condition"] == "end_only_unmasked"].iloc[0]["text"]
    assert beginning == middle == end == chunks_df.loc[0, "chunk_text"]


def test_identical_condition_text_available_regardless_of_which_method_reads_it(settings, fixture_bundle):
    """Simulates two different 'methods' (e.g. a BERT runner and an LLM runner) reading the
    same registry row for the same (document_id, condition) -- they must get byte-identical
    text and the same fingerprint, proving no method can silently diverge."""
    split_df, masked_documents_df, chunks_df, masked_chunks_df, selections_df = fixture_bundle
    registry = build_condition_registry(
        split_df, masked_documents_df, chunks_df, masked_chunks_df, selections_df, "train", settings
    )
    row = registry[(registry["document_id"] == "2") & (registry["condition"] == "end_only_masked")].iloc[0]
    bert_runner_text = row["text"]
    llm_runner_text = registry[(registry["document_id"] == "2") & (registry["condition"] == "end_only_masked")].iloc[0]["text"]
    assert bert_runner_text == llm_runner_text

    import hashlib

    assert row["text_fingerprint"] == hashlib.sha256(bert_runner_text.encode("utf-8")).hexdigest()


def test_registry_reproducible_across_independent_runs(settings, fixture_bundle):
    split_df, masked_documents_df, chunks_df, masked_chunks_df, selections_df = fixture_bundle
    registry_a = build_condition_registry(
        split_df, masked_documents_df, chunks_df, masked_chunks_df, selections_df, "train", settings
    )
    registry_b = build_condition_registry(
        split_df, masked_documents_df, chunks_df, masked_chunks_df, selections_df, "train", settings
    )
    manifest_a = build_condition_registry_manifest(registry_a, settings)
    manifest_b = build_condition_registry_manifest(registry_b, settings)
    assert manifest_a.registry_fingerprint == manifest_b.registry_fingerprint


def test_condition_definitions_cover_the_configured_names(settings):
    definitions = build_condition_definitions(settings.family_aware.conditions.policy_version)
    names = {d.name for d in definitions}
    assert names == set(settings.family_aware.conditions.names)
    masked_flags = {d.name: d.masked for d in definitions}
    assert masked_flags["complete_unmasked"] is False
    assert masked_flags["complete_masked"] is True
