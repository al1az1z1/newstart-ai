"""Regression tests for Version 6 Checkpoint 8: the one-time family-aware BERT test
evaluation.

Most tests use small synthetic data (fast, no GPU/model needed); a few read the real,
already-generated Checkpoint 8 artifacts on disk as a completeness/integrity regression
guard, matching this repo's established pattern (e.g. tests/test_chunking.py's real-data
completeness test).
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from newstart_ai.config import load_settings
from newstart_ai.models.bert.family_aware_reproducibility import fingerprint_predictions
from newstart_ai.models.bert.test_evaluation import build_test_integrity_proof

LABEL_ORDER = ["USCIS", "DMV", "SSA", "IRS"]
APPROVED_CHECKPOINT_ARTIFACT_ID = "3e4c71a6758249aabe262fa12e39cce4"


@pytest.fixture(scope="module")
def settings():
    return load_settings()


# --- Integrity-proof unit tests (synthetic, fast) -----------------------------------------


def test_integrity_proof_passes_for_a_clean_synthetic_scenario():
    proof = build_test_integrity_proof(
        test_document_ids=["t1", "t2", "t3"],
        expected_test_document_ids={"t1", "t2", "t3"},
        train_document_ids={"a1", "a2"},
        validation_document_ids={"b1"},
        train_family_ids={"FAM:A"},
        validation_family_ids={"FAM:B"},
        test_family_ids={"FAM:T1", "FAM:T2"},
        excluded_document_ids={"excluded1"},
        checkpoint_artifact_id=APPROVED_CHECKPOINT_ARTIFACT_ID,
        approved_checkpoint_artifact_id=APPROVED_CHECKPOINT_ARTIFACT_ID,
        aggregation_method_used="mean_probabilities",
        frozen_aggregation_method="mean_probabilities",
        condition_policy_versions_used={"masking": "v1"},
        frozen_condition_policy_versions={"masking": "v1"},
    )
    assert proof.exact_document_set_matches_frozen_split is True
    assert proof.every_document_appears_exactly_once is True
    assert proof.no_train_document_overlap is True
    assert proof.no_validation_document_overlap is True
    assert proof.no_train_family_overlap is True
    assert proof.no_validation_family_overlap is True
    assert proof.no_excluded_document_evaluated is True
    assert proof.checkpoint_used_matches_approved is True
    assert proof.aggregation_matches_frozen_policy is True
    assert proof.condition_policy_versions_match_frozen is True


def test_integrity_proof_catches_a_train_document_leaking_into_test():
    proof = build_test_integrity_proof(
        test_document_ids=["t1", "a1"],  # a1 leaked from train
        expected_test_document_ids={"t1", "a1"},
        train_document_ids={"a1", "a2"},
        validation_document_ids=set(),
        train_family_ids=set(), validation_family_ids=set(), test_family_ids=set(),
        excluded_document_ids=set(),
        checkpoint_artifact_id=APPROVED_CHECKPOINT_ARTIFACT_ID,
        approved_checkpoint_artifact_id=APPROVED_CHECKPOINT_ARTIFACT_ID,
        aggregation_method_used="mean_probabilities", frozen_aggregation_method="mean_probabilities",
        condition_policy_versions_used={}, frozen_condition_policy_versions={},
    )
    assert proof.no_train_document_overlap is False


def test_integrity_proof_catches_a_mismatched_checkpoint():
    proof = build_test_integrity_proof(
        test_document_ids=["t1"], expected_test_document_ids={"t1"},
        train_document_ids=set(), validation_document_ids=set(),
        train_family_ids=set(), validation_family_ids=set(), test_family_ids=set(),
        excluded_document_ids=set(),
        checkpoint_artifact_id="some-other-checkpoint",
        approved_checkpoint_artifact_id=APPROVED_CHECKPOINT_ARTIFACT_ID,
        aggregation_method_used="mean_probabilities", frozen_aggregation_method="mean_probabilities",
        condition_policy_versions_used={}, frozen_condition_policy_versions={},
    )
    assert proof.checkpoint_used_matches_approved is False


def test_integrity_proof_catches_a_changed_aggregation_method():
    proof = build_test_integrity_proof(
        test_document_ids=["t1"], expected_test_document_ids={"t1"},
        train_document_ids=set(), validation_document_ids=set(),
        train_family_ids=set(), validation_family_ids=set(), test_family_ids=set(),
        excluded_document_ids=set(),
        checkpoint_artifact_id=APPROVED_CHECKPOINT_ARTIFACT_ID,
        approved_checkpoint_artifact_id=APPROVED_CHECKPOINT_ARTIFACT_ID,
        aggregation_method_used="majority_vote", frozen_aggregation_method="mean_probabilities",
        condition_policy_versions_used={}, frozen_condition_policy_versions={},
    )
    assert proof.aggregation_matches_frozen_policy is False


def test_integrity_proof_catches_a_missing_document_from_the_frozen_set():
    proof = build_test_integrity_proof(
        test_document_ids=["t1"], expected_test_document_ids={"t1", "t2"},  # t2 missing
        train_document_ids=set(), validation_document_ids=set(),
        train_family_ids=set(), validation_family_ids=set(), test_family_ids=set(),
        excluded_document_ids=set(),
        checkpoint_artifact_id=APPROVED_CHECKPOINT_ARTIFACT_ID,
        approved_checkpoint_artifact_id=APPROVED_CHECKPOINT_ARTIFACT_ID,
        aggregation_method_used="mean_probabilities", frozen_aggregation_method="mean_probabilities",
        condition_policy_versions_used={}, frozen_condition_policy_versions={},
    )
    assert proof.exact_document_set_matches_frozen_split is False


def test_integrity_proof_catches_a_duplicate_document_id():
    proof = build_test_integrity_proof(
        test_document_ids=["t1", "t1", "t2"],  # duplicate
        expected_test_document_ids={"t1", "t2"},
        train_document_ids=set(), validation_document_ids=set(),
        train_family_ids=set(), validation_family_ids=set(), test_family_ids=set(),
        excluded_document_ids=set(),
        checkpoint_artifact_id=APPROVED_CHECKPOINT_ARTIFACT_ID,
        approved_checkpoint_artifact_id=APPROVED_CHECKPOINT_ARTIFACT_ID,
        aggregation_method_used="mean_probabilities", frozen_aggregation_method="mean_probabilities",
        condition_policy_versions_used={}, frozen_condition_policy_versions={},
    )
    assert proof.every_document_appears_exactly_once is False


# --- Structural "no training/policy-selection function called" proofs ---------------------


def test_test_evaluation_module_never_imports_training_or_policy_selection_functions():
    import newstart_ai.models.bert.test_evaluation as test_eval_module

    source = inspect.getsource(test_eval_module)
    for banned in ("train_family_aware_bert", "select_best_aggregation_method", "build_agency_class_weight_manifest"):
        assert banned not in source, f"test_evaluation.py must never call {banned}"


# --- Real, already-generated Checkpoint 8 artifact regression guards ----------------------


@pytest.fixture(scope="module")
def real_artifacts_available(settings):
    manifests_dir = settings.resolve_path("artifacts/family_aware/manifests")
    path = manifests_dir / "checkpoint8_primary_test_result_v1.json"
    if not path.exists():
        pytest.skip("Checkpoint 8 artifacts not yet generated in this environment")
    return manifests_dir


def test_real_test_predictions_cover_exactly_the_99_frozen_test_documents(settings, real_artifacts_available):
    split_dir = settings.resolve_path(settings.family_aware.split.output_dir)
    expected_ids = set(pd.read_csv(split_dir / "test.csv")["document_id"].astype(str))
    assert len(expected_ids) == 99

    preds = pd.read_csv(settings.resolve_path("artifacts/family_aware/reports/checkpoint8_test_predictions.csv"))
    complete_unmasked = preds[preds["condition"] == "complete_unmasked"]
    actual_ids = set(complete_unmasked["document_id"].astype(str))
    assert actual_ids == expected_ids


def test_real_document_level_predictions_are_unique_per_condition(settings, real_artifacts_available):
    preds = pd.read_csv(settings.resolve_path("artifacts/family_aware/reports/checkpoint8_test_predictions.csv"))
    for condition, group in preds.groupby("condition"):
        assert group["document_id"].is_unique, f"Duplicate document_id predictions in condition {condition}"


def test_real_confusion_matrix_totals_equal_99(real_artifacts_available):
    primary = json.loads((real_artifacts_available / "checkpoint8_primary_test_result_v1.json").read_text(encoding="utf-8"))
    total = sum(sum(row.values()) for row in primary["confusion_matrix"].values())
    assert total == 99


def test_real_primary_result_used_the_approved_checkpoint_and_frozen_aggregation(real_artifacts_available):
    integrity = json.loads((real_artifacts_available / "checkpoint8_test_integrity_proof_v1.json").read_text(encoding="utf-8"))
    assert integrity["checkpoint_artifact_id"] == APPROVED_CHECKPOINT_ARTIFACT_ID
    assert integrity["checkpoint_used_matches_approved"] is True
    assert integrity["aggregation_method_used"] == "mean_probabilities"
    assert integrity["aggregation_matches_frozen_policy"] is True
    assert integrity["no_retraining_or_policy_revision_triggered"] is True


def test_real_no_train_or_validation_overlap_with_test(real_artifacts_available):
    integrity = json.loads((real_artifacts_available / "checkpoint8_test_integrity_proof_v1.json").read_text(encoding="utf-8"))
    assert integrity["no_train_document_overlap"] is True
    assert integrity["no_validation_document_overlap"] is True
    assert integrity["no_train_family_overlap"] is True
    assert integrity["no_validation_family_overlap"] is True
    assert integrity["no_excluded_document_evaluated"] is True


def test_real_condition_fingerprints_match_frozen_policy_versions(real_artifacts_available):
    integrity = json.loads((real_artifacts_available / "checkpoint8_test_integrity_proof_v1.json").read_text(encoding="utf-8"))
    assert integrity["condition_policy_versions_match_frozen"] is True


def test_real_pre_test_freeze_record_fingerprints_match_established_checkpoint_4_5_values(settings, real_artifacts_available):
    freeze = json.loads((real_artifacts_available / "checkpoint8_pre_test_freeze_v1.json").read_text(encoding="utf-8"))
    chunk_manifest = json.loads((real_artifacts_available / "chunk_manifest_v1.json").read_text(encoding="utf-8"))
    split_dir = settings.resolve_path(settings.family_aware.split.output_dir)
    split_manifest = json.loads((split_dir / "family_split_manifest_v1.json").read_text(encoding="utf-8"))

    assert freeze["test_chunk_fingerprint"] == chunk_manifest["chunk_fingerprints"]["test"]
    assert freeze["test_split_fingerprint"] == split_manifest["split_fingerprints"]["test"]
    assert freeze["aggregation_method"] == "mean_probabilities"
    assert freeze["frozen"] is True


def test_real_ten_conditions_all_reported_with_diff_from_complete_unmasked(real_artifacts_available):
    sweep = json.loads((real_artifacts_available / "checkpoint8_test_condition_sweep_v1.json").read_text(encoding="utf-8"))
    conditions = {r["condition"] for r in sweep["results"]}
    assert len(conditions) == 10
    complete_unmasked = next(r for r in sweep["results"] if r["condition"] == "complete_unmasked")
    assert complete_unmasked["difference_from_complete_unmasked_macro_f1"] == 0.0
    for r in sweep["results"]:
        assert r["document_count"] == 99


def test_real_historical_artifacts_and_frozen_checkpoints_unchanged(settings, real_artifacts_available):
    """Golden hashes captured at the start of Checkpoint 7 -- re-asserted here for the
    historical artifact, plus confirmation the Checkpoint 7 best-checkpoint files still
    match what Checkpoint 8's integrity proof recorded."""
    base = settings.resolve_path("artifacts/models/3628681550d7433b94407f684946bb2f/checkpoint")
    expected = {
        "config.json": "4f38d5af30890cca293834aaa93406c8a5f38207cbc4cfc53934996ed72e3e74",
        "model.safetensors": "c271ccb55d06304f6843b2eadff8b9d4d4d62c44b0ea4084be17188575ccc735",
        "tokenizer.json": "127b303a9d131abf935977caed19ab33f0def61426aa88f4423a85505587dfdc",
        "tokenizer_config.json": "00b8750b3928b958db3b6a44c7f4d4df2c656601c6f35f884357babd3c6d5334",
    }
    for name, expected_hash in expected.items():
        path = base / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash

    repro = json.loads((real_artifacts_available / "checkpoint8_test_reproducibility_v1.json").read_text(encoding="utf-8"))
    fa_model_dir = settings.resolve_path(settings.family_aware.training.output_dir) / APPROVED_CHECKPOINT_ARTIFACT_ID / "checkpoint"
    for name, expected_hash in repro["checkpoint_file_hashes"].items():
        actual = hashlib.sha256((fa_model_dir / name).read_bytes()).hexdigest()
        assert actual == expected_hash, f"Family-aware checkpoint file {name} changed since Checkpoint 8 ran"


# --- Metric-from-document-level-predictions + stable fingerprint tests (synthetic) --------


def test_prediction_fingerprint_is_stable_across_repeated_calls():
    doc_ids = ["1", "2", "3"]
    predictions = ["USCIS", "DMV", "SSA"]
    fp1 = fingerprint_predictions(doc_ids, predictions)
    fp2 = fingerprint_predictions(list(reversed(doc_ids)), list(reversed(predictions)))
    assert fp1 == fp2  # order-independent, sorted internally


def test_prediction_fingerprint_changes_if_a_single_prediction_changes():
    fp1 = fingerprint_predictions(["1", "2"], ["USCIS", "DMV"])
    fp2 = fingerprint_predictions(["1", "2"], ["USCIS", "SSA"])
    assert fp1 != fp2
