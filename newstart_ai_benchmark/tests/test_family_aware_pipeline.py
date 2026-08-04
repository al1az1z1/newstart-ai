"""End-to-end-ish tests for Version 6 Checkpoint 7 training/evaluation plumbing, using tiny
real data slices against the real (cached, no network call) bert-base-uncased tokenizer and
a freshly-initialized model -- fast, but exercises the actual torch code paths rather than
pure mocks.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest
import torch
from transformers import AutoModelForSequenceClassification

from newstart_ai.config import load_settings
from newstart_ai.data import get_bert_tokenizer
from newstart_ai.models.bert import (
    generate_chunk_level_outputs,
    load_family_aware_artifact,
    new_family_aware_artifact_id,
    save_family_aware_artifact,
    set_determinism,
    train_family_aware_bert,
)
from newstart_ai.schemas.checkpoint7 import FamilyAwareModelMetadata

LABEL_ORDER = ["USCIS", "DMV", "SSA", "IRS"]


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def tokenizer(settings):
    return get_bert_tokenizer(settings)


@pytest.fixture()
def tiny_chunks_and_texts():
    documents = pd.DataFrame(
        {
            "document_id": ["1", "2", "3", "4"],
            "text": [
                "This is a USCIS immigration form about employment authorization.",
                "This is a DMV vehicle registration renewal notice.",
                "This is a Social Security Administration benefits statement.",
                "This is an Internal Revenue Service tax withholding form.",
            ],
            "effective_agency": ["USCIS", "DMV", "SSA", "IRS"],
        }
    )
    chunks = pd.DataFrame(
        [
            {"document_id": d, "chunk_id": f"c{d}", "chunk_index": 0, "total_chunks": 1, "token_start": 0, "token_end": 12, "effective_agency": a}
            for d, a in zip(documents["document_id"], documents["effective_agency"])
        ]
    )
    return documents, chunks


def test_train_family_aware_bert_signature_has_no_test_split_parameter():
    """Structural isolation proof: the training function cannot reference a test split
    because no such parameter exists in its signature at all."""
    params = set(inspect.signature(train_family_aware_bert).parameters.keys())
    assert not any("test" in p.lower() for p in params)


def test_tiny_training_run_completes_and_selects_best_epoch_from_validation_only(settings, tokenizer, tiny_chunks_and_texts):
    documents, chunks = tiny_chunks_and_texts
    settings.family_aware.training.max_epochs = 1
    set_determinism(settings.family_aware.training.random_seed)
    model = AutoModelForSequenceClassification.from_pretrained(settings.bert.base_model, num_labels=len(LABEL_ORDER))

    texts = documents.set_index("document_id")["text"].to_dict()
    result = train_family_aware_bert(
        model=model,
        tokenizer=tokenizer,
        train_chunks_df=chunks,
        train_document_texts=texts,
        val_chunks_df=chunks,
        val_document_texts=texts,
        label_order=LABEL_ORDER,
        class_weight_by_label={label: 1.0 for label in LABEL_ORDER},
        settings=settings,
    )
    assert result["best_epoch"] == 1
    assert len(result["history"]) == 1
    assert result["history"][0].validation_document_macro_f1 >= 0.0
    assert "non_deterministic_op_warnings" in result


def test_save_load_artifact_produces_identical_predictions_on_a_fixed_sample(settings, tokenizer, tiny_chunks_and_texts, tmp_path):
    documents, chunks = tiny_chunks_and_texts
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(settings.bert.base_model, num_labels=len(LABEL_ORDER))
    model.to(device)
    texts = documents.set_index("document_id")["text"].to_dict()

    probs_before, _ = generate_chunk_level_outputs(model, tokenizer, chunks, texts, LABEL_ORDER, settings.family_aware.chunking.max_seq_length, device)

    original_output_dir = settings.family_aware.training.output_dir
    settings.family_aware.training.output_dir = str(tmp_path)
    try:
        artifact_id = new_family_aware_artifact_id()
        metadata = FamilyAwareModelMetadata(
            artifact_id=artifact_id,
            base_model=settings.bert.base_model,
            tokenizer_revision=settings.family_aware.chunking.tokenizer_revision,
            label_order=LABEL_ORDER,
            source_train_chunk_fingerprint="test",
            source_validation_chunk_fingerprint="test",
            source_train_split_fingerprint="test",
            source_validation_split_fingerprint="test",
            chunking_policy_version="v1",
            document_balancing_policy_version="v1",
            random_seed=42,
            torch_version=torch.__version__,
            transformers_version="test",
            cuda_available=torch.cuda.is_available(),
            best_epoch=1,
            stopping_reason="test",
            checkpoint_selection_metric="validation_macro_f1",
            checkpoint_selection_aggregation_method="mean_probabilities",
            best_validation_document_macro_f1=0.0,
            training_time_seconds=0.0,
            status="ready",
            created_at="2026-01-01T00:00:00Z",
        )
        save_family_aware_artifact(model, tokenizer, metadata, settings)
        loaded_model, loaded_tokenizer, loaded_metadata = load_family_aware_artifact(settings, artifact_id)
        loaded_model.to(device)

        probs_after, _ = generate_chunk_level_outputs(
            loaded_model, loaded_tokenizer, chunks, texts, LABEL_ORDER, settings.family_aware.chunking.max_seq_length, device
        )
        for doc_id in probs_before:
            for p_before, p_after in zip(probs_before[doc_id], probs_after[doc_id]):
                assert all(abs(a - b) < 1e-5 for a, b in zip(p_before, p_after))
        assert loaded_metadata.status == "ready"
    finally:
        settings.family_aware.training.output_dir = original_output_dir


def test_label_order_and_index_mapping_is_stable_across_repeated_settings_loads():
    settings_a = load_settings()
    settings_b = load_settings()
    assert list(settings_a.base.labels) == list(settings_b.base.labels)
    label_to_index_a = {label: i for i, label in enumerate(settings_a.base.labels)}
    label_to_index_b = {label: i for i, label in enumerate(settings_b.base.labels)}
    assert label_to_index_a == label_to_index_b
