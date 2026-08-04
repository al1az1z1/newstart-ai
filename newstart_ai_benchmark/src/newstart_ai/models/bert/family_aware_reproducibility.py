"""Assembles the full Checkpoint 7 reproducibility manifest (Version 6).

Ties together every fingerprint/identity artifact required to reproduce the family-aware
chunked BERT training run: source data fingerprints, configuration fingerprint, resolved
tokenizer identity, seed and package versions, label mapping, class weights, the best
checkpoint's file hashes, a fingerprint of its validation predictions, and which aggregation
policy version is now authoritative.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from newstart_ai.schemas.checkpoint7 import Checkpoint7ReproducibilityManifest


def fingerprint_configuration(settings) -> str:
    payload = settings.family_aware.model_dump_json()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_predictions(document_ids: list[str], predictions: list[str]) -> str:
    ordered = sorted(zip(document_ids, predictions), key=lambda pair: pair[0])
    payload = "\n".join(f"{doc_id}|{pred}" for doc_id, pred in ordered)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def collect_package_versions() -> dict[str, str]:
    import sklearn
    import transformers

    return {
        "torch": __import__("torch").__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }


def build_checkpoint7_reproducibility_manifest(
    settings,
    source_train_split_fingerprint: str,
    source_validation_split_fingerprint: str,
    source_train_chunk_fingerprint: str,
    source_validation_chunk_fingerprint: str,
    tokenizer_resolved_commit_hash: str | None,
    label_order: list[str],
    class_weights: dict[str, float],
    best_checkpoint_artifact_id: str,
    best_checkpoint_file_hashes: dict[str, str],
    validation_prediction_fingerprint: str,
    final_aggregation_policy_version: str,
) -> Checkpoint7ReproducibilityManifest:
    import torch

    cfg = settings.family_aware
    return Checkpoint7ReproducibilityManifest(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        source_train_split_fingerprint=source_train_split_fingerprint,
        source_validation_split_fingerprint=source_validation_split_fingerprint,
        source_train_chunk_fingerprint=source_train_chunk_fingerprint,
        source_validation_chunk_fingerprint=source_validation_chunk_fingerprint,
        configuration_fingerprint=fingerprint_configuration(settings),
        tokenizer_name=settings.bert.base_model,
        tokenizer_revision=cfg.chunking.tokenizer_revision,
        tokenizer_resolved_commit_hash=tokenizer_resolved_commit_hash,
        random_seed=cfg.training.random_seed,
        torch_version=torch.__version__,
        transformers_version=collect_package_versions()["transformers"],
        python_packages=collect_package_versions(),
        label_order=label_order,
        class_weights=class_weights,
        document_balancing_policy_version=cfg.document_balancing.policy_version,
        best_checkpoint_artifact_id=best_checkpoint_artifact_id,
        best_checkpoint_file_hashes=best_checkpoint_file_hashes,
        validation_prediction_fingerprint=validation_prediction_fingerprint,
        final_aggregation_policy_version=final_aggregation_policy_version,
        notes=[
            "Re-running training with the same seed, configuration, and package versions "
            "against these same source fingerprints is expected to reproduce this run -- "
            "any documented non-deterministic CUDA operations are recorded on the model "
            "artifact's metadata.json, not here.",
        ],
    )
