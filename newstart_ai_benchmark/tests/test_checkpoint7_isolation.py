"""Proof that Checkpoint 7 training/evaluation never accessed test-split data, and that the
historical bert-mvp artifact remains byte-for-byte unchanged (Version 6).
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

from newstart_ai.config import load_settings
from newstart_ai.data.test_isolation import build_test_isolation_proof
from newstart_ai.models.bert.condition_evaluation import evaluate_all_conditions
from newstart_ai.models.bert.family_aware_training import train_family_aware_bert

# Hashes captured directly from disk at the start of Checkpoint 7 (before any Checkpoint 7
# code ran) -- if these ever change, something modified the historical MVP artifact.
_HISTORICAL_ARTIFACT_DIR = "artifacts/models/3628681550d7433b94407f684946bb2f"
_EXPECTED_HISTORICAL_HASHES = {
    "checkpoint/config.json": "4f38d5af30890cca293834aaa93406c8a5f38207cbc4cfc53934996ed72e3e74",
    "checkpoint/model.safetensors": "c271ccb55d06304f6843b2eadff8b9d4d4d62c44b0ea4084be17188575ccc735",
    "checkpoint/tokenizer.json": "127b303a9d131abf935977caed19ab33f0def61426aa88f4423a85505587dfdc",
    "checkpoint/tokenizer_config.json": "00b8750b3928b958db3b6a44c7f4d4df2c656601c6f35f884357babd3c6d5334",
    "metadata.json": "50ecf8fadfbeef4443f5dd9d00d8675ca276ed85af7f1ba79b2f70bb3e654b3e",
}


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def test_historical_bert_mvp_artifact_files_unchanged():
    base = load_settings().resolve_path(_HISTORICAL_ARTIFACT_DIR)
    for relative_path, expected_hash in _EXPECTED_HISTORICAL_HASHES.items():
        path = base / relative_path
        assert path.exists(), f"Historical artifact file missing: {path}"
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"Historical artifact file changed: {path}"


def test_training_function_signature_excludes_any_test_split_parameter():
    params = list(inspect.signature(train_family_aware_bert).parameters.keys())
    assert not any("test" in p.lower() for p in params)


def test_condition_evaluation_signature_excludes_any_test_split_parameter():
    params = list(inspect.signature(evaluate_all_conditions).parameters.keys())
    assert not any("test" in p.lower() for p in params)


def test_checkpoint7_isolation_proof_flags_no_test_files_for_declared_training_inputs():
    proof = build_test_isolation_proof(
        functions_exercised=[
            "build_agency_class_weight_manifest",
            "train_family_aware_bert",
            "select_best_aggregation_method",
            "evaluate_all_conditions",
        ],
        input_files_used=[
            "data/family_aware_splits/train.csv",
            "data/family_aware_splits/validation.csv",
            "data/family_aware_chunks/train_chunks.csv",
            "data/family_aware_chunks/validation_chunks.csv",
            "data/family_aware_masked/validation_masked_chunks.csv",
            "data/family_aware_conditions/partial_input_selections.csv",
            "data/family_aware_conditions/condition_registry_train_validation.csv",
        ],
    )
    assert proof.isolation_holds is True
    assert proof.test_files_referenced == []


def test_checkpoint7_isolation_proof_catches_a_test_chunk_file_if_ever_declared():
    proof = build_test_isolation_proof(
        functions_exercised=["train_family_aware_bert"],
        input_files_used=["data/family_aware_chunks/train_chunks.csv", "data/family_aware_chunks/test_chunks.csv"],
    )
    assert proof.isolation_holds is False
    assert "data/family_aware_chunks/test_chunks.csv" in proof.test_files_referenced
