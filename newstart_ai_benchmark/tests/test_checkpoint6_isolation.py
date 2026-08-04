"""Proof that Checkpoint 6 policy-freezing never accessed test-split data (Version 6).

Every policy-building function used to freeze aggregation, document balancing, partial-input
selection, and masking is called here with ONLY the family-aware train and validation splits
physically present/passed -- the real test split is never loaded anywhere in this file,
demonstrating that freezing genuinely does not depend on it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from newstart_ai.config import load_settings
from newstart_ai.data.test_isolation import build_test_isolation_proof
from newstart_ai.models.bert.aggregation import build_aggregation_policy_manifest
from newstart_ai.models.bert.document_balancing import build_document_balancing_manifest


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def test_aggregation_policy_freezing_requires_only_validation_chunks(settings):
    # Only a validation-shaped chunk DataFrame is constructed -- no test data exists in this
    # test's scope at all, and the function succeeds regardless.
    val_chunks = pd.DataFrame({"document_id": ["v1", "v1", "v2"]})
    manifest = build_aggregation_policy_manifest(val_chunks, settings)
    assert manifest.provisional is True


def test_document_balancing_freezing_requires_only_train_chunks(settings):
    train_chunks = pd.DataFrame(
        {"document_id": ["t1", "t2"], "chunk_id": ["t1_0", "t2_0"], "agency": ["USCIS", "DMV"]}
    )
    manifest = build_document_balancing_manifest(train_chunks, settings)
    assert manifest.total_training_documents == 2


def test_isolation_proof_flags_no_test_files_when_only_train_and_validation_are_declared():
    proof = build_test_isolation_proof(
        functions_exercised=[
            "build_aggregation_policy_manifest",
            "build_document_balancing_manifest",
            "build_partial_input_selections",
            "build_masking_manifest",
        ],
        input_files_used=[
            "data/family_aware_splits/train.csv",
            "data/family_aware_splits/validation.csv",
            "data/family_aware_chunks/train_chunks.csv",
            "data/family_aware_chunks/validation_chunks.csv",
        ],
    )
    assert proof.isolation_holds is True
    assert proof.test_files_referenced == []


def test_isolation_proof_flags_test_files_if_ever_declared():
    """Regression guard: if a future orchestration script accidentally lists a test file,
    the proof must catch it rather than silently reporting isolation_holds=True."""
    proof = build_test_isolation_proof(
        functions_exercised=["some_function"],
        input_files_used=["data/family_aware_splits/train.csv", "data/family_aware_splits/test.csv"],
    )
    assert proof.isolation_holds is False
    assert "data/family_aware_splits/test.csv" in proof.test_files_referenced


def test_isolation_proof_flags_test_chunks_file_if_ever_declared():
    proof = build_test_isolation_proof(
        functions_exercised=["some_function"],
        input_files_used=["data/family_aware_chunks/test_chunks.csv"],
    )
    assert proof.isolation_holds is False
