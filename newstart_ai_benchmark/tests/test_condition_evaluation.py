"""Tests for Version 6 Checkpoint 7 condition evaluation (chunk assembly + fingerprint use)."""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from transformers import AutoModelForSequenceClassification

from newstart_ai.config import load_settings
from newstart_ai.data import get_bert_tokenizer
from newstart_ai.data.partial_input import build_partial_input_selections
from newstart_ai.models.bert.condition_evaluation import _condition_chunk_texts_for_document, evaluate_all_conditions

LABEL_ORDER = ["USCIS", "DMV", "SSA", "IRS"]


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def tokenizer(settings):
    return get_bert_tokenizer(settings)


@pytest.fixture()
def tiny_bundle(settings):
    unmasked = pd.DataFrame(
        [
            {"document_id": "1", "chunk_id": "1_0", "chunk_index": 0, "total_chunks": 3, "chunk_text": "beginning", "effective_agency": "USCIS"},
            {"document_id": "1", "chunk_id": "1_1", "chunk_index": 1, "total_chunks": 3, "chunk_text": "middle", "effective_agency": "USCIS"},
            {"document_id": "1", "chunk_id": "1_2", "chunk_index": 2, "total_chunks": 3, "chunk_text": "end", "effective_agency": "USCIS"},
            {"document_id": "2", "chunk_id": "2_0", "chunk_index": 0, "total_chunks": 1, "chunk_text": "only chunk", "effective_agency": "DMV"},
        ]
    )
    masked = pd.DataFrame(
        {
            "chunk_id": ["1_0", "1_1", "1_2", "2_0"],
            "document_id": ["1", "1", "1", "2"],
            "chunk_index": [0, 1, 2, 0],
            "masked_chunk_text": ["[MASKED] beginning", "[MASKED] middle", "[MASKED] end", "[MASKED] only chunk"],
        }
    )
    selections = build_partial_input_selections(unmasked, "validation", settings)
    return unmasked, masked, selections


def test_complete_condition_uses_all_chunks_in_order(tiny_bundle):
    unmasked, masked, selections = tiny_bundle
    unmasked_by_doc = {doc_id: g for doc_id, g in unmasked.groupby("document_id")}
    masked_by_doc = {doc_id: g for doc_id, g in masked.groupby("document_id")}
    selections_lookup = {(row.document_id, row.condition): row for row in selections.itertuples(index=False)}

    texts = _condition_chunk_texts_for_document("1", "complete_unmasked", unmasked_by_doc, masked_by_doc, selections_lookup)
    assert texts == ["beginning", "middle", "end"]


def test_complete_masked_condition_uses_masked_text(tiny_bundle):
    unmasked, masked, selections = tiny_bundle
    unmasked_by_doc = {doc_id: g for doc_id, g in unmasked.groupby("document_id")}
    masked_by_doc = {doc_id: g for doc_id, g in masked.groupby("document_id")}
    selections_lookup = {(row.document_id, row.condition): row for row in selections.itertuples(index=False)}

    texts = _condition_chunk_texts_for_document("1", "complete_masked", unmasked_by_doc, masked_by_doc, selections_lookup)
    assert texts == ["[MASKED] beginning", "[MASKED] middle", "[MASKED] end"]


def test_beginning_only_condition_uses_single_selected_chunk(tiny_bundle):
    unmasked, masked, selections = tiny_bundle
    unmasked_by_doc = {doc_id: g for doc_id, g in unmasked.groupby("document_id")}
    masked_by_doc = {doc_id: g for doc_id, g in masked.groupby("document_id")}
    selections_lookup = {(row.document_id, row.condition): row for row in selections.itertuples(index=False)}

    texts = _condition_chunk_texts_for_document("1", "beginning_only_unmasked", unmasked_by_doc, masked_by_doc, selections_lookup)
    assert texts == ["beginning"]


def test_single_chunk_document_partial_conditions_all_resolve_to_the_one_chunk(tiny_bundle):
    unmasked, masked, selections = tiny_bundle
    unmasked_by_doc = {doc_id: g for doc_id, g in unmasked.groupby("document_id")}
    masked_by_doc = {doc_id: g for doc_id, g in masked.groupby("document_id")}
    selections_lookup = {(row.document_id, row.condition): row for row in selections.itertuples(index=False)}

    for condition in ("beginning_only_unmasked", "middle_only_unmasked", "end_only_unmasked"):
        texts = _condition_chunk_texts_for_document("2", condition, unmasked_by_doc, masked_by_doc, selections_lookup)
        assert texts == ["only chunk"]


def test_evaluate_all_conditions_returns_ten_results_and_uses_passed_fingerprint(settings, tokenizer, tiny_bundle):
    unmasked, masked, selections = tiny_bundle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(settings.bert.base_model, num_labels=len(LABEL_ORDER))
    model.to(device)

    true_labels = {"1": "USCIS", "2": "DMV"}
    manifest, raw_predictions = evaluate_all_conditions(
        model=model,
        tokenizer=tokenizer,
        val_chunks_df=unmasked,
        val_masked_chunks_df=masked,
        val_selections_df=selections,
        true_labels_by_doc=true_labels,
        label_order=LABEL_ORDER,
        aggregation_method="mean_probabilities",
        max_seq_length=settings.family_aware.chunking.max_seq_length,
        num_special_tokens=settings.family_aware.chunking.num_special_tokens,
        condition_registry_fingerprint="fingerprint-under-test-abc123",
        device=device,
    )
    assert len(manifest.results) == 10
    assert manifest.condition_registry_fingerprint == "fingerprint-under-test-abc123"
    assert set(raw_predictions.keys()) == {r.condition for r in manifest.results}
    for result in manifest.results:
        assert result.document_count == 2
